"""The episode files: `.npz` + `.json`, and that they round-trip exactly.

`workspace/06_record_an_episode.py` is the only writer of the dataset format in this repo and
the only definition of the 31-slot layout, so these tests record a short episode with its own
code and check the two things a downstream LeRobot conversion depends on:

* the arrays have the documented shapes and dtypes - `(N, 31) float32` for
  `observation.state` and `action`, `(N, 144, 256, 3) uint8` for each camera stream;
* what comes back out of the `.npz` is bit-for-bit what went in, and the `.json` identifies
  it - `state_names`, the slot table, the fps, the model sha256 and the randomisation seed.

The fast tests record three frames. The full 90-frame `main()` is behind the `slow` marker:
it is 180 renders, which is minutes on software GL and is the dominant cost of this suite.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from tests.test_layout import EXPECTED_STATE_NAMES

FRAMES = 3


def _record(ep, sim, frames=FRAMES):
    """A miniature of example 06's recording loop, using its own layout helpers."""
    states = np.zeros((frames, ep.STATE_DIM), dtype=np.float32)
    actions = np.zeros((frames, ep.STATE_DIM), dtype=np.float32)
    stamps = np.zeros(frames, dtype=np.float64)
    d455 = np.zeros((frames, ep.CAM_H, ep.CAM_W, 3), dtype=np.uint8)
    d435i = np.zeros((frames, ep.CAM_H, ep.CAM_W, 3), dtype=np.uint8)

    held = ep.observation(sim.state())
    held[ep.BASE_SLICE] = 0.0
    dt = 1.0 / ep.FPS
    for i in range(frames):
        state = sim.state()
        states[i] = ep.observation(state)
        d455[i] = sim.rgb("d455_rgb")
        d435i[i] = sim.rgb("d435i_rgb")
        stamps[i] = state.t
        for name, args in ep.script(i * dt):
            ep.send(sim, name, args, held)
        actions[i] = held
        sim.step(dt)
    return {"observation.state": states, "action": actions, "timestamp": stamps,
            "observation.images.d455_rgb": d455, "observation.images.d435i_rgb": d435i}


@pytest.fixture
def recorded(episode_module, make_sim):
    ep = episode_module
    sim = make_sim(cam_w=ep.CAM_W, cam_h=ep.CAM_H)
    sim.sim_reset()
    rand = sim.sim_randomise(seed=7)
    sim.step(0.5)
    return ep, sim, rand, _record(ep, sim)


# --- shapes and dtypes -------------------------------------------------------------------
def test_the_documented_shapes_and_dtypes(recorded):
    ep, _sim, _rand, arrays = recorded
    assert (ep.CAM_H, ep.CAM_W) == (144, 256)
    assert arrays["observation.state"].shape == (FRAMES, 31)
    assert arrays["observation.state"].dtype == np.float32
    assert arrays["action"].shape == (FRAMES, 31)
    assert arrays["action"].dtype == np.float32
    for key in ("observation.images.d455_rgb", "observation.images.d435i_rgb"):
        assert arrays[key].shape == (FRAMES, 144, 256, 3), key
        assert arrays[key].dtype == np.uint8, key
    assert arrays["timestamp"].dtype == np.float64


def test_the_arrays_are_finite_and_the_images_are_not_blank(recorded):
    _ep, _sim, _rand, arrays = recorded
    assert np.isfinite(arrays["observation.state"]).all()
    assert np.isfinite(arrays["action"]).all()
    for key in ("observation.images.d455_rgb", "observation.images.d435i_rgb"):
        assert len(np.unique(arrays[key])) > 8, f"{key} is an almost constant image"


def test_the_timestamps_advance_by_whole_physics_steps_not_by_one_over_fps(recorded):
    """`timestamp` is the sim clock, and the sim clock only moves in 2 ms steps.

    `SimSource.step(seconds)` rounds to whole physics steps, so example 06's `dt = 1/30 s`
    advances the clock by 17 steps = 0.034 s, not 0.03333 s. The recorded episode therefore
    runs at 29.41 Hz while `meta["fps"]` says 30: a 2% stretch, 60 ms of accumulated skew
    over the 3 s episode. Harmless for a single-camera reach, wrong the moment anyone aligns
    these timestamps against an external stream or resamples by `fps` instead of by
    `timestamp`. Pinned here so the discrepancy is a known quantity rather than a surprise.
    """
    ep, _sim, _rand, arrays = recorded
    stamps = arrays["timestamp"]
    timestep = 0.002
    nominal = 1.0 / ep.FPS
    assert (np.diff(stamps) > 0).all()
    assert np.diff(stamps) == pytest.approx(round(nominal / timestep) * timestep, abs=1e-9)
    assert np.diff(stamps) != pytest.approx(nominal, abs=1e-6)


def test_the_first_action_is_hold_still_with_zero_base_velocity(recorded):
    """Documented: "The first action of an episode is 'stay where you are, zero base
    velocity'". A non-zero first action teaches a policy to jerk on frame one."""
    _ep, _sim, _rand, arrays = recorded
    assert arrays["action"][0, 28:31].tolist() == [0.0, 0.0, 0.0]


def test_the_episode_never_commands_the_base(recorded):
    """The scripted trajectory is a reach, so slots 28-30 of every action stay at zero - and
    a policy trained on it has no reason to move the base."""
    _ep, _sim, _rand, arrays = recorded
    assert np.abs(arrays["action"][:, 28:31]).max() == 0.0


def test_the_action_is_a_held_setpoint_stream(recorded):
    """A joint slot keeps its last commanded value until something commands it again, so
    consecutive actions differ only where the script spoke."""
    ep, _sim, _rand, arrays = recorded
    actions = arrays["action"]
    changed = np.flatnonzero(np.abs(actions[-1] - actions[0]) > 0)
    commanded = {ep.JOINT_INDEX[n] for n in ("left_shoulder_pitch_joint", "left_shoulder_roll_joint",
                                             "left_elbow_joint", "head_pan_joint", "head_tilt_joint")}
    assert set(changed.tolist()) <= commanded, "an uncommanded action slot moved"


def test_commanded_and_measured_slots_are_not_the_same_numbers(recorded):
    """The observation is where the joint *is*, the action where it was *told to go*. They
    converge, but on a 3-frame episode of a 1.5 s ramp they must not be identical - if they
    are, something is copying one into the other."""
    _ep, _sim, _rand, arrays = recorded
    assert not np.array_equal(arrays["observation.state"][:, :28], arrays["action"][:, :28])


# --- the round trip ----------------------------------------------------------------------
def test_npz_round_trips_exactly(recorded, tmp_path):
    """`np.savez_compressed` is lossless; this is the assertion example 06 prints as
    "state round-trips exactly: True"."""
    _ep, _sim, _rand, arrays = recorded
    path = tmp_path / "episode_000.npz"
    np.savez_compressed(path, **arrays)
    back = np.load(path)
    assert sorted(back.keys()) == sorted(arrays)
    for key, value in arrays.items():
        assert back[key].dtype == value.dtype, key
        assert back[key].shape == value.shape, key
        assert np.array_equal(back[key], value), key


def test_the_json_sidecar_identifies_the_arrays(recorded, tmp_path):
    ep, sim, rand, arrays = recorded
    meta = {
        "fps": ep.FPS, "frames": FRAMES, "state_dim": ep.STATE_DIM,
        "state_names": ep.STATE_NAMES,
        "slots": {"arm": [0, 14], "hand": [14, 26], "head": [26, 28], "base": [28, 31]},
        "units": {"joints": "rad", "base_vx": "m/s", "base_vy": "m/s", "base_wz": "rad/s"},
        "cameras": {"d455_rgb": [ep.CAM_H, ep.CAM_W, 3], "d435i_rgb": [ep.CAM_H, ep.CAM_W, 3]},
        "task": "pick up the red box",
        "sim_model": sim.model_info,
        "controller": sim.controller_caps(),
        "randomisation": {k: v for k, v in rand.items() if k in ("seed", "objects", "placed")},
    }
    path = tmp_path / "episode_000.json"
    path.write_text(json.dumps(meta, indent=2, default=str))
    back = json.loads(path.read_text())

    assert back["state_dim"] == arrays["observation.state"].shape[1] == 31
    assert back["frames"] == len(arrays["observation.state"])
    assert back["state_names"] == EXPECTED_STATE_NAMES
    assert len(back["state_names"]) == back["state_dim"]
    assert back["cameras"]["d455_rgb"] == list(arrays["observation.images.d455_rgb"].shape[1:])
    # the two things that make a dataset reproducible rather than merely well-formed
    assert len(back["sim_model"]["sha256"]) == 64
    assert back["randomisation"]["seed"] == 7
    assert sorted(back["randomisation"]["placed"]) == sorted(back["randomisation"]["objects"])
    # and the honest label on the base columns: the kinematic stand-in cannot measure them
    assert back["controller"]["stand_in"] is True


def test_the_slot_table_in_the_sidecar_matches_the_vector(recorded, tmp_path):
    ep, _sim, _rand, arrays = recorded
    slots = {"arm": [0, 14], "hand": [14, 26], "head": [26, 28], "base": [28, 31]}
    total = sum(hi - lo for lo, hi in slots.values())
    assert total == arrays["observation.state"].shape[1]
    for lo, hi in slots.values():
        assert ep.STATE_NAMES[lo:hi] == EXPECTED_STATE_NAMES[lo:hi]


# --- the real thing ----------------------------------------------------------------------
@pytest.mark.slow
def test_example_06_writes_a_loadable_episode(episode_module, monkeypatch, tmp_path):
    """Run `main()` end to end - 90 frames, 180 renders - and read the files back the way a
    dataloader would. This is the test that would catch the recorder drifting away from the
    layout the fast tests pin."""
    ep = episode_module
    monkeypatch.setattr(ep, "OUT", tmp_path)
    # `scene_path()` reads sys.argv[1], which under pytest is pytest's own argument.
    monkeypatch.setattr("sys.argv", ["06_record_an_episode.py"])
    ep.main()

    npz = tmp_path / "episode_000.npz"
    js = tmp_path / "episode_000.json"
    assert npz.exists() and js.exists()

    data = np.load(npz)
    meta = json.loads(js.read_text())
    frames = int(ep.SECONDS * ep.FPS)
    assert meta["frames"] == frames
    assert data["observation.state"].shape == (frames, 31)
    assert data["observation.state"].dtype == np.float32
    assert data["action"].shape == (frames, 31)
    assert data["action"].dtype == np.float32
    assert data["observation.images.d455_rgb"].shape == (frames, 144, 256, 3)
    assert data["observation.images.d455_rgb"].dtype == np.uint8
    assert data["observation.images.d435i_rgb"].shape == (frames, 144, 256, 3)
    assert meta["state_names"] == EXPECTED_STATE_NAMES
    assert meta["randomisation"]["seed"] == 7
    assert np.isfinite(data["observation.state"]).all()
    # the scripted reach actually moved the left arm, so the episode is not 90 identical frames
    moved = np.abs(data["observation.state"][-1, :28] - data["observation.state"][0, :28])
    assert moved.max() > 0.1
