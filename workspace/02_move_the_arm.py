"""02 - Move the arm: send joint targets, watch the state follow, save before/after frames.

Run it:

    uv run python workspace/02_move_the_arm.py

What to take away:

* `set_joint_targets({joint_name: radians})` is the whole command API for the upper body.
  It is a *position target*, not a torque: the arm, hand and head joints are position servos
  (on the real G1 the motor drivers do the same PD loop), so you set a target and then let
  physics run until the joint gets there.
* The targets are clipped to each actuator's `ctrlrange`, which is the joint limit out of
  Unitree's own model. Commanding 10 rad does not break anything; it just saturates.
* Legs and waist belong to the whole-body controller (example 05). As soon as a real
  controller is active, `set_joint_targets` raises `ValueError` if you name one of its joints
  - the same rule the real robot needs, which is why it lives in the source and not in a UI.
  Under the default `kinematic` stand-in the controller owns no joints at all, so the leg
  position servos are still writable; example 05 shows the refusal.
* Physics is physics. If the target pose puts the forearm through the table, the joint simply
  does not get there and the error never converges - worth checking before you blame the code.
"""
from __future__ import annotations

import sys
from pathlib import Path

import imageio.v3 as iio
import numpy as np

from g1.robot.robot_source import HAND_JOINTS
from g1.robot.sim_source import SimSource, default_model_path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "workspace" / "out"

# A reach-forward pose for the left arm, in radians. Joint names come straight from
# g1.robot.robot_source.BODY_JOINTS - the 29-DoF G1 motor order.
REACH = {
    "left_shoulder_pitch_joint": -1.2,    # swing the whole arm up and forward
    "left_shoulder_roll_joint": 0.9,      # and out to the side, clear of the table
    "left_elbow_joint": 1.0,
    "left_wrist_pitch_joint": -0.3,
}
# Closing the left hand: the Revo 2 has 6 commandable joints per hand; the distal links follow
# through model equalities, so you drive the proximal ones only.
CLOSE_LEFT_HAND = {j: 1.0 for j in HAND_JOINTS if j.startswith("left_")}


def scene_path() -> str:
    if len(sys.argv) > 1:
        return sys.argv[1]
    local = REPO / "models" / "g1_pick_place.xml"
    return str(local) if local.exists() else default_model_path()


def arm_now(sim: SimSource) -> dict[str, float]:
    s = sim.state()
    return {j: s.joints[j] for j in REACH}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    sim = SimSource(xml=scene_path(), realtime=False)
    try:
        sim.step(0.5)                       # let the keyframe settle before measuring anything
        before = arm_now(sim)
        iio.imwrite(OUT / "02_before.png", sim.rgb("third_person"))

        print("left arm before:")
        for j, v in before.items():
            print(f"  {j:<28} {v:+.4f} rad")

        # --- command ---------------------------------------------------------------------
        # One call sets all four targets. Nothing moves yet: the command only writes the
        # actuator setpoints. Physics moves the arm.
        sim.set_joint_targets(REACH)
        sim.set_joint_targets(CLOSE_LEFT_HAND)

        # Step in slices so we can watch it converge rather than only see the end state.
        print("\nconverging on the target (max |target - measured| over the four arm joints):")
        for _ in range(6):
            sim.step(0.25)
            now = arm_now(sim)
            err = max(abs(REACH[j] - now[j]) for j in REACH)
            print(f"  t={sim.state().t:5.2f}s   error {err:.4f} rad")

        after = arm_now(sim)
        iio.imwrite(OUT / "02_after.png", sim.rgb("third_person"))

        print("\nleft arm after:")
        for j, v in after.items():
            print(f"  {j:<28} {v:+.4f} rad   (target {REACH[j]:+.2f}, moved {v - before[j]:+.4f})")

        hand = sim.state().joints
        closed = np.mean([hand[j] for j in CLOSE_LEFT_HAND])
        print(f"\nleft hand mean finger angle: {closed:.3f} rad (target 1.0 - the fingers "
              "stall against each other and the thumb, which is what a real grasp does too)")

        # --- clipping --------------------------------------------------------------------
        sim.set_joint_targets({"left_elbow_joint": 99.0})
        sim.step(1.0)
        print(f"after commanding 99 rad of elbow: {sim.state().joints['left_elbow_joint']:+.4f} rad "
              "(clipped to the actuator's ctrlrange)")

        print(f"\nwrote {OUT / '02_before.png'} and {OUT / '02_after.png'}")
    finally:
        sim.close()


if __name__ == "__main__":
    main()
