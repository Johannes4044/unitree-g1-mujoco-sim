"""Pytest bootstrap and the fixtures that are expensive enough to share.

Why this file sits at the repo root
-----------------------------------
* **`src/` layout.** The tests import `g1` the way any other consumer does. When the
  package is installed (`uv sync`, `pip install -e .`) that import already works. When it
  is not - a bare clone, or a virtualenv where the editable install's `.pth` file never
  took effect - we put `src/` on `sys.path` below, so `pytest` works from a fresh checkout
  with no install step and the suite never silently tests a *different*, older `g1`.
* **Model paths.** Everything is resolved relative to this file, never to the current
  working directory, so `pytest` behaves the same from the repo root and from `tests/`.
  `$SIM_MODEL` is deliberately ignored here: `default_model_path()` honours it, and a test
  suite that silently followed an environment variable would assert the committed
  baselines against somebody else's scene.

Running the suite
-----------------
    uv run pytest                 the fast set: contracts only, a few seconds
    uv run pytest -m slow         the slow set: rendering sweeps and long rollouts
    uv run pytest -m ""           everything

`-m "not slow"` is the default, from `addopts` in `pyproject.toml`. `pytest` on its own
therefore runs the fast set - a plain `pytest` is *not* the full suite. With pip instead of
uv: `pip install -e ".[dev]"` and then the same commands without the `uv run`.

State leaking between tests
---------------------------
A `SimSource` is mutable: `sim_randomise`, `sim_teleport`, `set_mode` and a few hundred
physics steps all persist. A simulator reused after one of those is a genuine source of
flaky cross-talk, so there is no session-scoped simulator here. Tests either build their
own through the `make_sim` factory (which closes everything it handed out) or share a
module-scoped one *and* reset it explicitly before every test.
"""
from __future__ import annotations

import importlib.util
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent
SRC = REPO_ROOT / "src"
if importlib.util.find_spec("g1") is None:          # not installed: test the checkout
    sys.path.insert(0, str(SRC))

MODELS = REPO_ROOT / "models"
DEFAULT_MODEL = MODELS / "g1_pick_place.xml"
BUILT_SCENARIOS = MODELS / "scenarios"
SCENARIO_SRC = REPO_ROOT / "scenarios"
HOLOSOMA_ONNX = SRC / "g1" / "wbc" / "models" / "holosoma" / "fastsac_g1_29dof.onnx"

# The five worlds the README and docs/cheatsheet.md promise. Hard-coded, not globbed: a
# scenario that stops being built should fail the suite, not quietly leave it with four.
SCENARIOS = ("warehouse_aisle", "rubble_yard", "pick_place_table", "cluttered_bench", "house_rooms")


def built_scenario(name: str) -> Path:
    return BUILT_SCENARIOS / f"{name}.xml"


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def default_model() -> str:
    assert DEFAULT_MODEL.exists(), f"missing {DEFAULT_MODEL}"
    return str(DEFAULT_MODEL)


@pytest.fixture(scope="session")
def scenario_models() -> dict[str, str]:
    return {n: str(built_scenario(n)) for n in SCENARIOS}


@pytest.fixture
def make_sim():
    """Factory for simulators that are closed again when the test ends.

        sim = make_sim()                        # default scene, kinematic, clock owned by us
        sim = make_sim(controller="holosoma")
        sim = make_sim(xml=scenario_models["house_rooms"])

    `realtime=False` is the default on purpose: a background physics thread makes every
    measurement below a race against the wall clock.
    """
    from g1.robot.sim_source import SimSource

    made = []

    def _make(xml: str | None = None, **kw):
        kw.setdefault("realtime", False)
        src = SimSource(xml=xml or str(DEFAULT_MODEL), **kw)
        made.append(src)
        return src

    yield _make
    for src in made:
        src.close()


@pytest.fixture(scope="session")
def episode_module():
    """`workspace/06_record_an_episode.py` loaded as a module.

    The 31-slot layout has no home of its own in this repo - `g1.layout` does not ship
    here - so example 06 *is* the definition, and the tests read it from there rather than
    restating it. Loaded by path because the filename starts with a digit.
    """
    path = REPO_ROOT / "workspace" / "06_record_an_episode.py"
    spec = importlib.util.spec_from_file_location("g1_episode_example", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="session")
def holosoma_onnx() -> Path:
    return HOLOSOMA_ONNX


@pytest.fixture(scope="session")
def xml_parse():
    """Parse an XML file, for the asset-integrity tests."""
    def _parse(path) -> ET.Element:
        return ET.parse(os.fspath(path)).getroot()
    return _parse
