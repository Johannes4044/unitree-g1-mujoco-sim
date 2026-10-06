"""Numeric baselines, with the reason for every tolerance written next to it.

The rule used throughout: a tolerance has to be wide enough that a MuJoCo patch release -
which can change solver iteration order, contact ordering and the last bits of a float -
does not break the build, and tight enough that a real regression cannot hide inside it.
Where a quantity is exact by construction (a model dimension, a point count, a dtype) the
tolerance is zero and the test says why.

Every number here was measured on this checkout (mujoco 3.12.0, numpy 2.4.6, macOS arm64),
not copied from documentation.
"""
from __future__ import annotations

import numpy as np
import pytest

from conftest import SCENARIOS

# --- model dimensions --------------------------------------------------------------------
# Exact. `models/g1_pick_place.xml` is committed and `mujoco` is pinned to an exact version
# ("the model files are compiled against this version" - pyproject.toml), so these cannot
# move without someone editing the model or the pin. nu=43 is 29 body + 12 hand + 2 head
# actuators; nq=88 / nv=83 include the free joints of the four pickable objects.
G1_NQ, G1_NV, G1_NU = 88, 83, 43
# The keyframe pelvis height, which the welded kinematic base holds. `scripts/smoke_test.py`
# prints 0.790 m after 2000 steps.
STAND_BASE_Z = 0.790


def test_the_robot_model_has_the_expected_dimensions(make_sim):
    sim = make_sim()
    assert (sim.m.nq, sim.m.nv, sim.m.nu) == (G1_NQ, G1_NV, G1_NU)


def test_the_actuator_set_is_the_commandable_joints(make_sim):
    """43 actuators = 29 body + 12 Revo 2 hand proximals + 2 head module joints. If this
    count changes, the 31-slot layout no longer covers what the robot can be told to do."""
    from g1.robot.robot_source import BODY_JOINTS, HAND_JOINTS, HEAD_JOINTS

    sim = make_sim()
    expected = set(BODY_JOINTS) | set(HAND_JOINTS) | set(HEAD_JOINTS)
    assert len(expected) == G1_NU
    assert set(sim.act) == expected


def test_the_physics_timestep_is_two_milliseconds(make_sim):
    """500 Hz, which is what the per-step PD loop in `SimSource._control` assumes when it
    says "the PD loop at the physics rate (500 Hz), as on the real robot's motor drivers"."""
    assert make_sim().m.opt.timestep == pytest.approx(0.002, abs=1e-12)


def test_the_standing_pelvis_height(make_sim):
    """0.790 m. Tolerance 10 mm: the pelvis is welded to a mocap body, so the only thing that
    can move it is the weld's own constraint residual (measured: microns). 10 mm is three
    orders above that and still far below any change worth calling a regression - a different
    keyframe or a different robot would move it by centimetres."""
    sim = make_sim()
    sim.step(0.5)
    assert sim.state().base_pos[2] == pytest.approx(STAND_BASE_Z, abs=0.010)


@pytest.mark.parametrize("controller", ["pd_stand", "holosoma"])
def test_a_free_base_controller_holds_the_robot_up(make_sim, controller):
    """Not a fixed number: these controllers actually balance, so the height is an outcome.
    The contract is "it is standing", i.e. the band `scripts/check_scenarios.py` uses
    (0.5 .. 1.2 m) narrowed to the measured 0.64 .. 0.80 m for a G1 on flat ground."""
    sim = make_sim(controller=controller)
    sim.step(1.0)
    state = sim.state()
    assert 0.60 < state.base_pos[2] < 0.82
    assert state.extra["controller_fault"] is None
    # upright: the body-frame z component of world "up". cos(20 deg) = 0.94.
    _w, x, y, _z = state.base_quat
    assert 1 - 2 * (x * x + y * y) > 0.94


# --- LiDAR -------------------------------------------------------------------------------
# Exactly 15000 points: `LIDAR_POINTS` defaults to 15000 and every scene here returns more
# candidate points than that, so the sampler always fills the frame.
LIDAR_POINTS = 15000
# Measured per-scenario Euclidean range from the sensor, over three sweeps each.
#   name                nearest       farthest
RANGE_BASELINE = {
    "warehouse_aisle":  (0.194, "open"),
    "rubble_yard":      (0.166, "open"),
    "pick_place_table": (0.085, "open"),
    "cluttered_bench":  (0.086, "open"),
    "house_rooms":      (0.170, 7.19),     # enclosed: four walls and a ceiling at ~7 m
}
# `g1.robot.lidar` gates on MIN_RANGE 0.1 m / MAX_RANGE 40.0 m and *then* adds 1 cm of
# Gaussian noise, so a returned range may sit up to a few sigma outside the gate. 0.05 m
# covers 5 sigma; anything wider would stop being a check on the gate at all.
NOISE_SLACK = 0.05


def _ranges(sim):
    points, pos, _quat = sim.lidar()
    return points, np.linalg.norm(points - pos, axis=1)


def test_a_sweep_is_fifteen_thousand_points(make_sim):
    points, _ = _ranges(make_sim())
    assert len(points) == LIDAR_POINTS


def test_a_sweep_stays_inside_the_sensor_range_gate(make_sim):
    from g1.robot.lidar import MAX_RANGE, MIN_RANGE

    _, rng = _ranges(make_sim())
    assert rng.min() > MIN_RANGE - NOISE_SLACK
    assert rng.max() < MAX_RANGE + NOISE_SLACK


def test_a_sweep_stays_inside_the_sensor_elevation_band(make_sim):
    """-52 .. +7 degrees, after the upside-down mount on the crown.

    Only points beyond 1 m are checked, and that is not a fudge: `LidarSim.scan` applies the
    elevation gate first and adds 1 cm of Gaussian noise *after* it, so a point at the 8 cm
    blind-zone edge can be displaced by 7 degrees of elevation and a few hundred of the 15000
    points legitimately sit outside the band. Measured: the full cloud spans -61 .. +14
    degrees, the 89% of it beyond 1 m spans -52.7 .. -1.8. Beyond 1 m, 1 cm of noise is
    0.6 degrees, so 1.5 degrees of slack is the honest bound there.
    """
    from g1.robot.lidar import V_FOV_DEG

    sim = make_sim()
    points, pos, _ = sim.lidar()
    rel = points - pos
    far = np.linalg.norm(rel, axis=1) > 1.0
    assert far.sum() > 0.5 * len(points)
    elev = np.degrees(np.arctan2(rel[far, 2], np.linalg.norm(rel[far, :2], axis=1)))
    assert elev.min() > V_FOV_DEG[0] - 1.5
    assert elev.max() < V_FOV_DEG[1] + 1.5


@pytest.mark.slow
@pytest.mark.parametrize("name", SCENARIOS)
def test_each_scenario_lidar_range(make_sim, scenario_models, name):
    """Slow: five models, each settled for 2 s and scanned.

    Tolerances. The *nearest* return is a fixed piece of the robot's own crown or torso, so
    it repeats to within the 1 cm noise: 0.03 m (3 sigma) around the measured value.
    The *farthest* return is whichever of 15000 randomly sampled points happens to be
    furthest, so it is much noisier - measured 39.21 .. 39.98 m across three sweeps of the
    same scene. Pinning it to a value would be pinning the sampler's luck, so the contract is
    the thing that actually matters: an open scenario reaches the 40 m gate and an enclosed
    one cannot.
    """
    nearest, farthest = RANGE_BASELINE[name]
    sim = make_sim(xml=scenario_models[name])
    sim.step(2.0)
    points, rng = _ranges(sim)
    assert len(points) == LIDAR_POINTS
    assert rng.min() == pytest.approx(nearest, abs=0.03)
    if farthest == "open":
        assert rng.max() > 38.0, "an open scenario stopped reaching the 40 m range gate"
        assert rng.max() < 40.0 + NOISE_SLACK
    else:
        # house_rooms is enclosed. 15% either way: the far wall is a fixed distance, the only
        # scatter is which pixel of it got sampled plus the 1 cm noise.
        assert rng.max() == pytest.approx(farthest, rel=0.15)
        assert rng.max() < 10.0, "house_rooms stopped being an enclosed scene"


# --- scenario models ---------------------------------------------------------------------
def test_every_scenario_loads_and_settles(make_sim, scenario_models):
    """Compile + keyframe + a short settle, for all five. The full 2 s settle and the render
    sweep that `scripts/check_scenarios.py` does are the slow test below."""
    for name, path in scenario_models.items():
        sim = make_sim(xml=path)
        sim.step(0.2)
        state = sim.state()
        assert sim.m.nu == G1_NU, name
        assert state.base_pos[2] == pytest.approx(STAND_BASE_Z, abs=0.010), name
        assert np.isfinite(sim.d.qacc).all(), name
        assert np.abs(sim.d.qacc).max() < 1e5, f"{name} exploded"


@pytest.mark.slow
def test_every_scenario_survives_a_two_second_settle(make_sim, scenario_models):
    """What `scripts/check_scenarios.py` checks, as a test: nothing explodes, nothing flies
    off, and every free object stays in the world. Bounds copied from that script."""
    for name, path in scenario_models.items():
        sim = make_sim(xml=path)
        objects_before = set(sim.state().extra["objects"])
        sim.step(2.0)
        state = sim.state()
        assert 0.5 < state.base_pos[2] < 1.2, name
        assert np.abs(sim.d.qacc).max() < 1e5, name
        assert set(state.extra["objects"]) == objects_before, name
        for obj, p in state.extra["objects"].items():
            assert np.isfinite(p).all(), f"{name}/{obj}"
            assert abs(p[0]) < 50 and abs(p[1]) < 50 and -1 < p[2] < 5, f"{name}/{obj} flew to {p}"


@pytest.mark.slow
def test_every_scenario_resets_without_drift(make_sim, scenario_models):
    """"5 objects restored", 0.0 mm drift - for every scenario, not just the default scene."""
    for name, path in scenario_models.items():
        sim = make_sim(xml=path)
        q0 = sim.d.qpos.copy()
        sim.step(1.0)
        sim.sim_reset()
        drift_mm = 1000 * float(np.abs(sim.d.qpos - q0).max())
        assert drift_mm == 0.0, f"{name}: {drift_mm:.3f} mm of drift after sim_reset"
