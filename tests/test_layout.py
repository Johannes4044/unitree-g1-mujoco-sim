"""The 31-slot observation/action layout.

Why this module exists
----------------------
`g1.layout` does not ship in this repo. The layout is *reconstructed* in
`workspace/06_record_an_episode.py` from `BODY_JOINTS`, `HAND_JOINTS` and `HEAD_JOINTS` by
substring matching ("shoulder", "elbow", "wrist") and concatenation. That makes the layout a
derived quantity with no checksum: renaming a joint, reordering `BODY_JOINTS`, or adding a
`left_wrist_*` joint silently shifts every column after it. Nothing raises. Every episode
recorded before the change stays byte-for-byte valid and means something different, and a
policy trained on the mixture learns the average of two layouts.

So the slot counts, the ordering and the asymmetry at slots 28-30 are pinned here by value,
not by re-deriving them the same way the code does.
"""
from __future__ import annotations

import numpy as np
import pytest

from g1.robot.robot_source import BODY_JOINTS, HAND_JOINTS, HEAD_JOINTS

# --- the layout, written out. Not derived - that is the point. -------------------------
EXPECTED_ARM = [
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]
EXPECTED_HAND = [
    "left_thumb_metacarpal_joint", "left_thumb_proximal_joint", "left_index_proximal_joint",
    "left_middle_proximal_joint", "left_ring_proximal_joint", "left_pinky_proximal_joint",
    "right_thumb_metacarpal_joint", "right_thumb_proximal_joint", "right_index_proximal_joint",
    "right_middle_proximal_joint", "right_ring_proximal_joint", "right_pinky_proximal_joint",
]
EXPECTED_HEAD = ["head_pan_joint", "head_tilt_joint"]
EXPECTED_BASE = ["base_vx", "base_vy", "base_wz"]
EXPECTED_STATE_NAMES = EXPECTED_ARM + EXPECTED_HAND + EXPECTED_HEAD + EXPECTED_BASE


# --- the joint lists the layout is built from -------------------------------------------
def test_body_joints_is_the_29_dof_motor_order():
    assert len(BODY_JOINTS) == 29
    assert len(set(BODY_JOINTS)) == 29
    assert BODY_JOINTS[:12] == [f"{s}_{j}_joint" for s in ("left", "right")
                                for j in ("hip_pitch", "hip_roll", "hip_yaw", "knee",
                                          "ankle_pitch", "ankle_roll")]
    assert BODY_JOINTS[12:15] == ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"]
    assert list(BODY_JOINTS[15:]) == EXPECTED_ARM


def test_hand_and_head_joint_lists():
    assert list(HAND_JOINTS) == EXPECTED_HAND
    assert list(HEAD_JOINTS) == EXPECTED_HEAD


def test_arm_slots_are_left_seven_then_right_seven():
    """Side ordering is the thing a mirrored-dataset bug looks like, so pin it directly."""
    arm = list(BODY_JOINTS[15:])
    assert all(n.startswith("left_") for n in arm[:7])
    assert all(n.startswith("right_") for n in arm[7:])
    assert [n.removeprefix("left_") for n in arm[:7]] == [n.removeprefix("right_") for n in arm[7:]]


def test_hand_slots_are_left_six_then_right_six():
    assert all(n.startswith("left_") for n in HAND_JOINTS[:6])
    assert all(n.startswith("right_") for n in HAND_JOINTS[6:])
    assert ([n.removeprefix("left_") for n in HAND_JOINTS[:6]]
            == [n.removeprefix("right_") for n in HAND_JOINTS[6:]])


# --- the layout as example 06 builds it --------------------------------------------------
def test_slot_counts(episode_module):
    ep = episode_module
    assert ep.STATE_DIM == 31
    assert len(ep.ARM_JOINTS) == 14
    assert len(HAND_JOINTS) == 12
    assert len(HEAD_JOINTS) == 2
    assert len(ep.BASE_NAMES) == 3
    assert len(ep.JOINT_NAMES) == 28


def test_slot_order_and_boundaries(episode_module):
    ep = episode_module
    assert ep.STATE_NAMES == EXPECTED_STATE_NAMES
    assert ep.STATE_NAMES[0:14] == EXPECTED_ARM
    assert ep.STATE_NAMES[14:26] == EXPECTED_HAND
    assert ep.STATE_NAMES[26:28] == EXPECTED_HEAD
    assert ep.STATE_NAMES[28:31] == EXPECTED_BASE
    assert ep.BASE_SLICE == slice(28, 31)
    assert ep.JOINT_INDEX["left_shoulder_pitch_joint"] == 0
    assert ep.JOINT_INDEX["right_wrist_yaw_joint"] == 13
    assert ep.JOINT_INDEX["left_thumb_metacarpal_joint"] == 14
    assert ep.JOINT_INDEX["right_pinky_proximal_joint"] == 25
    assert ep.JOINT_INDEX["head_pan_joint"] == 26
    assert ep.JOINT_INDEX["head_tilt_joint"] == 27


def test_the_substring_derivation_still_selects_exactly_the_arms(episode_module):
    """Example 06 picks the arm joints out of `BODY_JOINTS` by substring. A new joint whose
    name contains "wrist", "elbow" or "shoulder" - a wrist force sensor joint, a shoulder
    brace - would join the arm block and push every later slot along by one."""
    assert episode_module.ARM_JOINTS == list(BODY_JOINTS[15:])
    legs_and_waist = BODY_JOINTS[:15]
    assert not [n for n in legs_and_waist
                if any(k in n for k in ("shoulder", "elbow", "wrist"))]


def test_legs_and_waist_are_not_in_the_layout(episode_module):
    """A policy trained on these columns structurally cannot command a leg."""
    assert not set(BODY_JOINTS[:15]) & set(episode_module.STATE_NAMES)


def test_episode_metadata_slot_table_matches_the_layout(episode_module):
    """The `slots` dict written into episode_000.json is what a converter reads. It must not
    be able to disagree with the vector it describes."""
    ep = episode_module
    slots = {"arm": [0, 14], "hand": [14, 26], "head": [26, 28], "base": [28, 31]}
    assert ep.STATE_NAMES[slots["arm"][0]:slots["arm"][1]] == EXPECTED_ARM
    assert ep.STATE_NAMES[slots["hand"][0]:slots["hand"][1]] == EXPECTED_HAND
    assert ep.STATE_NAMES[slots["head"][0]:slots["head"][1]] == EXPECTED_HEAD
    assert ep.STATE_NAMES[slots["base"][0]:slots["base"][1]] == EXPECTED_BASE


# --- the asymmetry at slots 28-30 --------------------------------------------------------
def test_observation_is_float32_and_31_wide(episode_module, make_sim):
    sim = make_sim()
    vec = episode_module.observation(sim.state())
    assert vec.shape == (31,)
    assert vec.dtype == np.float32
    assert np.isfinite(vec).all()


def test_observation_slots_28_to_30_are_measured_not_commanded(episode_module, make_sim):
    """Slots 28-30 of an *observation* are the measured base velocity. Under the kinematic
    stand-in the pelvis is welded, so the source reports `base_lin_vel is None` - "not
    measured" - and the observation stays at zero *even while a velocity is commanded*.

    That is the asymmetry. Substituting the command here is the single mistake that teaches a
    policy its own output is a measurement, and nothing in the data would show it.
    """
    sim = make_sim()
    sim.set_mode("walk")
    sim.set_base_velocity(0.3, 0.0, 0.2)
    sim.step(0.1)
    state = sim.state()
    assert state.base_lin_vel is None and state.base_ang_vel is None
    obs = episode_module.observation(state)
    assert obs[28:31].tolist() == [0.0, 0.0, 0.0]

    held = np.zeros(31, dtype=np.float32)
    episode_module.apply_command("cmd_vel", {"vx": 0.3, "vy": 0.0, "wz": 0.2}, held)
    # float32 round-trip of the commanded float64, hence approx rather than ==
    assert held[28:31] == pytest.approx([0.3, 0.0, 0.2], abs=1e-7)   # the action holds it


def test_observation_slots_28_to_30_are_filled_by_a_free_base(episode_module, make_sim):
    """With a real controller the base velocity *is* measured, so the same slots are no longer
    zero - the observation changes meaning with the controller, which is why the episode
    metadata carries `controller_caps()`."""
    sim = make_sim(controller="holosoma")
    sim.set_mode("walk")
    for _ in range(10):                                  # refresh inside the 0.3 s dead-man
        sim.set_base_velocity(0.4, 0.0, 0.0)
        sim.step(0.1)
    state = sim.state()
    assert state.base_lin_vel is not None and state.base_ang_vel is not None
    obs = episode_module.observation(state)
    assert np.isfinite(obs).all()
    assert np.abs(obs[28:31]).max() > 1e-3, "a walking free base measured no velocity at all"


def test_action_joint_slots_hold_their_setpoint(episode_module):
    """An action column is a held setpoint, not a delta: a command for one joint must not
    disturb any other slot."""
    held = np.zeros(31, dtype=np.float32)
    episode_module.apply_command("set_joints", {"targets": {"left_elbow_joint": 1.0}}, held)
    assert held[episode_module.JOINT_INDEX["left_elbow_joint"]] == 1.0
    assert held.sum() == 1.0
    episode_module.apply_command("set_head", {"pan": 0.0, "tilt": 0.5}, held)
    assert held[27] == 0.5
    assert held[episode_module.JOINT_INDEX["left_elbow_joint"]] == 1.0   # still held


def test_an_unknown_command_raises(episode_module):
    with pytest.raises(ValueError):
        episode_module.apply_command("fly", {}, np.zeros(31, dtype=np.float32))
