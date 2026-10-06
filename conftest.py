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
# Was the fallback needed? Reported in the pytest header, because it is the difference
# between "the package is installed and the tests exercise it" and "the package is not
# importable and the tests are exercising the checkout instead". A green run means something
# different in the two cases: in the second one, `python workspace/01_hello_sim.py` and
# `scripts/check_scenarios.py` - which have no conftest - will still fail.
USING_PATH_FALLBACK = importlib.util.find_spec("g1") is None
if USING_PATH_FALLBACK:
    sys.path.insert(0, str(SRC))

MODELS = REPO_ROOT / "models"
DEFAULT_MODEL = MODELS / "g1_pick_place.xml"
BUILT_SCENARIOS = MODELS / "scenarios"
SCENARIO_SRC = REPO_ROOT / "scenarios"
HOLOSOMA_ONNX = SRC / "g1" / "wbc" / "models" / "holosoma" / "fastsac_g1_29dof.onnx"

# The five worlds the README and docs/cheatsheet.md promise. Hard-coded, not globbed: a
# scenario that stops being built should fail the suite, not quietly leave it with four.
SCENARIOS = ("warehouse_aisle", "rubble_yard", "pick_place_table", "cluttered_bench", "house_rooms")

# --- can this machine render at all? ----------------------------------------------------
# Probed once per session, because some machines cannot: GitHub's macos-14 runners have no
# display and MuJoCo's CGL backend raises `CGLError: invalid pixel format` on every
# `Renderer`, and a headless Linux box without the Mesa/EGL packages fails differently.
#
# The probe is deliberately narrow. `mujoco.Renderer.__init__` is where the GL context is
# created (`gl_context.GLContext(...)` then `MjrContext`), so a failure *there* is the only
# thing treated as "this machine cannot render", and it is the only thing that skips. Once a
# context exists, everything afterwards - a render that raises, or one that returns the wrong
# shape or dtype - is a real defect and must reach the tests that assert it. Otherwise the
# probe would quietly swallow exactly the regressions the suite exists to catch.
FORCE_NO_GL = "G1_TEST_FORCE_NO_GL"
# The smallest possible scene: one geom, one camera, 16x16. Nothing about the G1.
_PROBE_XML = """<mujoco><worldbody>
  <geom name="g" type="sphere" size="1"/>
  <camera name="c" pos="0 -4 0" xyaxes="1 0 0 0 0 1"/>
</worldbody></mujoco>"""
_probe_result: tuple[bool, str | None] | None = None


def _probe_rendering() -> tuple[bool, str | None]:
    """(can_render, reason_it_cannot). Only a failure to create the GL context says False."""
    forced = os.environ.get(FORCE_NO_GL, "").strip()
    if forced and forced != "0":
        return False, (f"${FORCE_NO_GL} is set: the skip path is being exercised deliberately, "
                       "no GL context was attempted")

    import mujoco

    model = mujoco.MjModel.from_xml_string(_PROBE_XML)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    try:
        renderer = mujoco.Renderer(model, 16, 16)
    except Exception as exc:  # noqa: BLE001 - see below
        # Broad on purpose. CGLError on a display-less mac, OSError/RuntimeError/ImportError
        # for a missing EGL or OSMesa library on Linux, and MuJoCo validates $MUJOCO_GL at
        # import time, so the set of things that can come out of here is not enumerable. The
        # alternative to catching broadly is the whole session erroring out, which tells
        # whoever reads the log strictly less.
        return False, (f"no GL context could be created ({type(exc).__module__}."
                       f"{type(exc).__name__}: {str(exc).strip() or 'no detail'})")
    try:
        renderer.update_scene(data, camera="c")
        renderer.render()
    except Exception:  # noqa: BLE001 - a drawing failure is not a capability question
        # A context exists but drawing through it failed. Not a capability question any more:
        # let the real tests run and report it.
        return True, None
    finally:
        renderer.close()
    return True, None


def rendering_probe() -> tuple[bool, str | None]:
    global _probe_result
    if _probe_result is None:
        _probe_result = _probe_rendering()
    return _probe_result


@pytest.fixture(scope="session")
def rendering_available() -> bool:
    return rendering_probe()[0]


@pytest.fixture
def requires_rendering():
    """Skip the test when this machine has no usable GL context.

    Use it on anything that calls `rgb()`, `depth()` or `lidar()`. The reason text is written
    for whoever is reading a CI log and has to decide whether the robot is broken: it is not.
    """
    ok, reason = rendering_probe()
    if not ok:
        pytest.skip(f"rendering unavailable - {reason}. Physics, the RobotSource contract, "
                    "the 31-slot layout, the safety refusals and seeding all still ran; only "
                    "camera, depth and LiDAR coverage is missing from this run.")


def built_scenario(name: str) -> Path:
    return BUILT_SCENARIOS / f"{name}.xml"


def pytest_report_header() -> list[str]:
    """Say which `g1` is under test, and say so loudly when it is not the installed one.

    The path alone cannot tell you which it is, and that is not a defect to be fixed: an
    *editable* install puts `src/` on `sys.path` through its `.pth` file, so a perfectly
    healthy `uv sync` also reports `<repo>/src/g1/__init__.py` - byte-identical to what the
    fallback below reports. `USING_PATH_FALLBACK` is the only signal that distinguishes them.
    So do not "harden" this by asserting the path contains `site-packages`: that holds for a
    non-editable install only, and would fail every normal development checkout.
    """
    import g1

    lines = [f"g1: {Path(g1.__file__).resolve()}"]
    if USING_PATH_FALLBACK:
        lines.append(
            "g1: NOT IMPORTABLE as an installed package - testing src/ via the conftest "
            "sys.path fallback. The suite is valid, but it is not evidence that `import g1` "
            "works outside it: run `python -c 'import g1'` before trusting a green run. "
            "(Seen on macOS when every site-packages .pth file carries the UF_HIDDEN flag, "
            "which CPython >= 3.11 skips, leaving an editable install inert.)")
    ok, reason = rendering_probe()
    lines.append(f"rendering: {'available' if ok else f'UNAVAILABLE - {reason}'}")
    if not ok:
        lines.append("rendering: camera, depth and LiDAR tests will be SKIPPED; physics, the "
                     "RobotSource contract, the layout and the safety refusals are unaffected")
    return lines


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
