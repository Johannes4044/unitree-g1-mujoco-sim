"""Determinism and seeding: the property RL and VLA work is built on.

A sim that is not reproducible for a seed makes every A/B comparison noise, and makes a
recorded episode impossible to replay. Four things are pinned here:

* same seed -> same object placement, and the full `sim_randomise` return value,
* different seed -> different placement,
* same seed -> same trajectory (placement *and* the physics that follows from it),
* `sim_reset` restores the keyframe, every object and the clock, bit for bit.

And one thing is pinned because it is *not* seeded: `lidar()` draws from NumPy's global
legacy RNG. See `test_lidar_draws_from_the_global_numpy_rng`.
"""
from __future__ import annotations

import numpy as np
import pytest


def _object_poses(sim) -> np.ndarray:
    """Every free object's qpos (position + quaternion), in a stable name order."""
    return np.array([sim.d.qpos[a:a + 7].copy() for _, (a, _) in sorted(sim._obj_adr.items())])


# --- randomisation -----------------------------------------------------------------------
def test_the_same_seed_places_the_objects_identically(make_sim):
    sim = make_sim()
    first = sim.sim_randomise(seed=7)
    sim.sim_reset()
    second = sim.sim_randomise(seed=7)
    assert first == second, "sim_randomise(seed=7) is not reproducible"


def test_a_different_seed_places_the_objects_differently(make_sim):
    sim = make_sim()
    first = sim.sim_randomise(seed=7)
    sim.sim_reset()
    other = sim.sim_randomise(seed=8)
    assert first["objects"] == other["objects"]
    assert first["placed"] != other["placed"]
    moved = max(abs(first["placed"][k][i] - other["placed"][k][i])
                for k in first["placed"] for i in (0, 1))
    # 1 cm: two independent uniform draws over a 0.7 x 1.2 m surface land this far apart with
    # overwhelming probability. Anything smaller would be indistinguishable from the same draw.
    assert moved > 0.01


def test_randomisation_is_reproducible_across_two_fresh_simulators(make_sim):
    """Reproducibility must not depend on the process having drawn the same randoms before -
    `sim_randomise` spawns its streams from the seed alone."""
    a = make_sim().sim_randomise(seed=11)
    np.random.default_rng(0).random(1000)                # perturb any shared RNG state
    b = make_sim().sim_randomise(seed=11)
    assert a == b


def test_each_randomisation_axis_has_its_own_stream(make_sim):
    """Turning colour jitter on must not move the objects: the axes are independent streams
    spawned from one seed, so a domain-randomisation experiment can add an axis without
    invalidating the layouts it already recorded."""
    sim = make_sim()
    poses_only = sim.sim_randomise(seed=4, pose=True)
    sim.sim_reset()
    with_colour = sim.sim_randomise(seed=4, pose=True, colour=True, mass=True, friction=True)
    assert poses_only["placed"] == with_colour["placed"]
    assert with_colour["colour"] and with_colour["mass"] and with_colour["friction"]
    assert not poses_only["colour"] and not poses_only["mass"] and not poses_only["friction"]


def test_a_drawn_seed_is_returned_so_an_episode_can_be_replayed(make_sim):
    sim = make_sim()
    drawn = sim.sim_randomise(seed=None)
    assert isinstance(drawn["seed"], int)
    sim.sim_reset()
    replay = sim.sim_randomise(seed=drawn["seed"])
    assert replay["placed"] == drawn["placed"]


# --- trajectories ------------------------------------------------------------------------
def test_the_same_seed_gives_the_same_trajectory(make_sim):
    """Placement being reproducible is not enough: the physics that follows has to be too."""
    def run(seed):
        sim = make_sim()
        sim.sim_reset()
        sim.sim_randomise(seed=seed)
        sim.step(1.0)
        return _object_poses(sim)

    a, b, other = run(5), run(5), run(6)
    assert np.array_equal(a, b), f"same seed diverged by {np.abs(a - b).max():.3g}"
    assert np.abs(a - other).max() > 0.01


def test_a_scripted_command_sequence_is_reproducible(make_sim):
    """Same seed, same commands, same joint trajectory - including the actuator clipping and
    the controller's own 50 Hz decimation."""
    def run():
        sim = make_sim(controller="holosoma")
        sim.sim_reset()
        sim.set_mode("walk")
        out = []
        for i in range(10):
            sim.set_base_velocity(0.3, 0.0, 0.1)
            sim.set_joint_targets({"left_elbow_joint": 0.5 + 0.02 * i})
            sim.step(0.05)
            out.append(sim.d.qpos.copy())
        return np.array(out)

    a, b = run(), run()
    assert np.array_equal(a, b), f"replay diverged by {np.abs(a - b).max():.3g}"


# --- reset -------------------------------------------------------------------------------
def test_sim_reset_restores_the_whole_state_exactly(make_sim):
    """`scripts/check_scenarios.py` reports "5 objects restored" and 0.0 mm drift. Exact, not
    approximate: the reset writes the keyframe and the recorded initial object poses straight
    into qpos, so any drift at all would mean something else is being restored."""
    sim = make_sim()
    q0, v0, t0 = sim.d.qpos.copy(), sim.d.qvel.copy(), float(sim.d.time)
    objects0 = dict(sim.state().extra["objects"])

    sim.sim_randomise(seed=3, colour=True, mass=True, friction=True)
    sim.step(1.0)
    sim.sim_teleport(-1.0, 0.4, 0.5)
    sim.step(0.5)
    assert sim.state().extra["objects"] != objects0     # the test actually disturbed something

    sim.sim_reset()
    assert np.array_equal(sim.d.qpos, q0), f"{1000 * np.abs(sim.d.qpos - q0).max():.3f} mm of drift"
    assert np.array_equal(sim.d.qvel, v0)
    assert float(sim.d.time) == t0
    assert sim.state().extra["objects"] == objects0


def test_sim_reset_restores_the_randomised_model_fields(make_sim):
    """Colour, mass and friction randomisation change the *model*, not just the state. They
    persist until a reset, and a reset has to put all three back or the next episode runs in a
    world that silently differs from the one its sha256 names."""
    sim = make_sim()
    rgba0 = sim.m.geom_rgba.copy()
    mass0 = sim.m.body_mass.copy()
    friction0 = sim.m.geom_friction.copy()
    sim.sim_randomise(seed=2, colour=True, mass=True, friction=True)
    assert not np.array_equal(sim.m.geom_rgba, rgba0)
    assert not np.array_equal(sim.m.body_mass, mass0)
    assert not np.array_equal(sim.m.geom_friction, friction0)
    sim.sim_reset()
    assert np.array_equal(sim.m.geom_rgba, rgba0)
    assert np.array_equal(sim.m.body_mass, mass0)
    assert np.array_equal(sim.m.geom_friction, friction0)


def test_the_model_sha256_identifies_the_world(make_sim, scenario_models):
    """`model_info["sha256"]` is the hash of the *compiled* model, and belongs in every
    episode's metadata. Two loads of one file must agree; two different worlds must not."""
    a = make_sim().model_info
    b = make_sim().model_info
    other = make_sim(xml=scenario_models["house_rooms"]).model_info
    assert a["sha256"] == b["sha256"]
    assert a["sha256"] != other["sha256"]
    assert len(a["sha256"]) == 64
    assert other["scenario"] == "house_rooms" and other["scenario_seed"] is None


def test_a_randomisation_does_not_change_the_reported_sha256(make_sim):
    """`model_info` is computed once at load, before anything mutates the model, so the hash
    names the world as built. The randomisation seed is what identifies the rest."""
    sim = make_sim()
    before = sim.model_info["sha256"]
    sim.sim_randomise(seed=9, colour=True, mass=True)
    assert sim.model_info["sha256"] == before


# --- the one thing that is not seeded ----------------------------------------------------
def test_lidar_draws_from_the_global_numpy_rng(make_sim, requires_rendering):
    """`LidarSim.scan` samples its 15000 points with `np.random.choice` and adds noise with
    `np.random.normal` - the *global* legacy RandomState, not a stream of its own.

    Consequences, both pinned here because both matter to anyone recording data:

    * two consecutive scans of an unchanged scene differ, so a LiDAR episode is not
      reproducible by replaying the sim alone;
    * it is reproducible if, and only if, you seed the global RNG - which also means any
      other code in the process drawing from `np.random` shifts the point cloud.

    This is a gap, not a feature. If `LidarSim` ever takes its own `Generator`, this test is
    what will tell you the contract changed.
    """
    sim = make_sim()
    np.random.seed(0)
    first = sim.lidar()[0]
    np.random.seed(0)
    again = sim.lidar()[0]
    assert np.array_equal(first, again), "seeding the global RNG did not reproduce the scan"
    unseeded = sim.lidar()[0]
    assert not np.array_equal(first, unseeded), "lidar() became independent of the global RNG"


@pytest.mark.slow
def test_a_long_rollout_is_reproducible(make_sim):
    """2000 steps (4 s) through the learned controller, twice. Slow because this is the run
    where an uninitialised buffer or a `dict` iteration order would finally show up."""
    def run():
        sim = make_sim(controller="holosoma")
        sim.sim_reset()
        sim.set_mode("walk")
        for _ in range(40):
            sim.set_base_velocity(0.4, 0.0, 0.0)
            sim.step(0.1)
        return sim.d.qpos.copy(), sim.d.qvel.copy()

    (qa, va), (qb, vb) = run(), run()
    assert np.array_equal(qa, qb), f"qpos diverged by {np.abs(qa - qb).max():.3g}"
    assert np.array_equal(va, vb)
