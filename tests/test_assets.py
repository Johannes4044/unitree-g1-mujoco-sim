"""Asset integrity: the files that ship, and the hashes that identify them.

This repo is meant to cross an air gap as a tree of files and still be the same simulator on
the other side. That only holds if the assets are the assets: the Holosoma policy weights are
the bytes the provenance note in `g1/wbc/holosoma.py` names, the scene and all five scenarios
compile, and the committed built scenarios have not drifted from their DSL sources.
"""
from __future__ import annotations

import hashlib

import mujoco
import pytest

from conftest import BUILT_SCENARIOS, DEFAULT_MODEL, SCENARIO_SRC, SCENARIOS, built_scenario
from g1.wbc.holosoma import MODEL as HOLOSOMA_MODEL
from g1.wbc.holosoma import MODEL_SHA256


# --- the ONNX policy ---------------------------------------------------------------------
def test_the_holosoma_policy_is_the_documented_file():
    """`g1/wbc/holosoma.py` records the provenance - amazon-far/holosoma at bccd4d7, path
    `.../loco/g1_29dof/fastsac_g1_29dof.onnx` - and a sha256. Apache-2.0 redistribution rests
    on that identification, and so does any claim about what the policy was trained to do.
    Exact: a hash is not a measurement.
    """
    assert HOLOSOMA_MODEL.exists(), f"missing {HOLOSOMA_MODEL}"
    digest = hashlib.sha256(HOLOSOMA_MODEL.read_bytes()).hexdigest()
    # Written out, not just compared to the constant: editing `MODEL_SHA256` to match a
    # swapped-in file would otherwise keep this test green.
    expected = "8346fd90778439395922a8c7256f24125ae84b8dea949128bac9e23c02bc7717"
    assert digest == expected
    assert MODEL_SHA256 == expected


def test_the_upstream_licence_ships_next_to_the_weights():
    """Apache-2.0 section 4: a redistributor carries the upstream LICENSE and NOTICE."""
    for name in ("LICENSE", "NOTICE"):
        path = HOLOSOMA_MODEL.parent / name
        assert path.exists() and path.stat().st_size > 0, f"missing {path}"


def test_the_policy_metadata_is_the_g1_29_dof_order():
    """The controller refuses to load a model whose `dof_names` are not `BODY_JOINTS`. Assert
    the shipped file actually satisfies that, so the check is proven rather than merely
    present - a silently reordered action vector drives the wrong motors."""
    import json

    from g1.robot.robot_source import BODY_JOINTS
    from g1.wbc.holosoma import HolosomaController

    ctrl = HolosomaController()
    meta = ctrl._sess.get_modelmeta().custom_metadata_map
    assert json.loads(meta["dof_names"]) == list(BODY_JOINTS)
    assert len(ctrl.kp) == 29 and len(ctrl.kd) == 29
    assert ctrl.velocity_limits == pytest.approx((1.0, 1.0, 1.0))


def test_the_policy_input_and_output_widths():
    """100-d observation, 29-d action, as the module docstring states."""
    from g1.wbc.holosoma import HolosomaController

    ctrl = HolosomaController()
    (inp,) = ctrl._sess.get_inputs()
    (out,) = ctrl._sess.get_outputs()
    assert list(inp.shape)[-1] == 100
    assert list(out.shape)[-1] == 29


# --- the models --------------------------------------------------------------------------
def test_the_default_scene_is_present_and_compiles():
    assert DEFAULT_MODEL.exists(), f"missing {DEFAULT_MODEL}"
    model = mujoco.MjModel.from_xml_path(str(DEFAULT_MODEL))
    assert model.nu == 43
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "stand") >= 0
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis") >= 0


@pytest.mark.parametrize("name", SCENARIOS)
def test_each_built_scenario_compiles(name):
    """All five, through the same `_compile` the simulator uses - which repairs short
    keyframes. Compiling with plain `from_xml_path` would hide that bug class."""
    from g1.robot.sim_source import _compile

    path = built_scenario(name)
    assert path.exists(), f"missing {path}"
    model = _compile(str(path))
    assert model.nu == 43
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "stand") >= 0


@pytest.mark.parametrize("name", SCENARIOS)
def test_each_scenario_source_parses_and_names_itself(name, xml_parse):
    """`scenarios/*.xml` is the declarative DSL, not MJCF; `scripts/build_scenarios.py`
    compiles it into `models/scenarios/`. The `name` attribute is what the built file is
    called, so a mismatch produces a scenario nobody can find by name."""
    path = SCENARIO_SRC / f"{name}.xml"
    assert path.exists(), f"missing {path}"
    root = xml_parse(path)
    assert root.tag == "scenario"
    assert root.get("name") == name


def test_there_are_exactly_five_scenarios():
    """Five is a documented fact (README, cheatsheet), not an accident of the directory."""
    sources = sorted(p.stem for p in SCENARIO_SRC.glob("*.xml"))
    assert sources == sorted(SCENARIOS)
    # seeded variants are transient and gitignored, so the committed set is the five
    built = sorted(p.stem for p in BUILT_SCENARIOS.glob("*.xml") if "_s" not in p.stem)
    assert built == sorted(SCENARIOS)


def test_the_cameras_the_interface_promises_exist_in_the_model():
    """`CAMERAS`, `DEPTHS` and the four Mid-360 rig cameras are hardware names shared with
    the real robot. A renamed camera in the MJCF turns `rgb()` into a silent `None`."""
    from g1.robot.lidar import CAMS
    from g1.robot.robot_source import CAMERAS, DEPTHS

    model = mujoco.MjModel.from_xml_path(str(DEFAULT_MODEL))
    for cam in list(CAMERAS) + list(DEPTHS.values()) + list(CAMS):
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam) >= 0, cam
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "mid360_site") >= 0


def test_the_randomisation_surface_exists_in_the_default_scene():
    """`sim_randomise` has no fallback: without `pick_table_top` there is no pick surface and
    it raises rather than scattering objects over a guessed table."""
    from g1.robot.sim_source import PICK_TABLE_GEOM

    model = mujoco.MjModel.from_xml_path(str(DEFAULT_MODEL))
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, PICK_TABLE_GEOM) >= 0


def test_no_runtime_dependency_crept_in():
    """The five-package runtime set is a licence commitment ("Permissive only", and nothing
    GPL anywhere near it). The tests add pytest and ruff as dev dependencies and nothing else.
    """
    import tomllib

    from conftest import REPO_ROOT

    cfg = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    names = sorted(d.split(">")[0].split("=")[0].split("<")[0].strip()
                   for d in cfg["project"]["dependencies"])
    assert names == ["imageio", "mujoco", "numpy", "onnxruntime", "pillow"]
    dev = cfg["project"]["optional-dependencies"]["dev"]
    group = cfg["dependency-groups"]["dev"]
    assert sorted(dev) == sorted(group), "the dev extra and the dev group have drifted apart"
    assert sorted(d.split(">")[0].strip() for d in dev) == ["pytest", "ruff"]
