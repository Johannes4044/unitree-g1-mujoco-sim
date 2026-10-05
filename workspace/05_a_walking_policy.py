"""05 - Driving the base: the same velocity command through two whole-body controllers.

Run it:

    uv run python workspace/05_a_walking_policy.py

The mental model
----------------
Nothing in this project commands leg joints directly. A policy, a navigator or your own code
emits a **body-frame base velocity** `(vx, vy, wz)` in m/s and rad/s, and a *whole-body
controller* (`g1.wbc`) turns that into leg and waist joint targets at 50 Hz, which the physics
loop tracks with a PD at 500 Hz - the same split the real G1's motor drivers have.

Two backends ship here, and the difference between them is the single most important thing to
understand about this simulator:

  kinematic   NOT A CONTROLLER. The pelvis is welded to a mocap body and the simulator simply
              integrates the commanded velocity into that body's pose. The legs never move and
              the robot cannot fall. It exists so you can exercise navigation, cameras, LiDAR
              and manipulation without a walking policy getting in the way. **Nothing about it
              transfers to hardware.** `controller_caps()["stand_in"]` is True and you should
              propagate that flag anywhere a human might mistake the two.

  holosoma    A real trained policy: Amazon FAR's FastSAC G1 29-DoF velocity-tracking network,
              run on CPU through onnxruntime (`src/g1/wbc/models/holosoma/`, Apache-2.0). The
              base is free, the robot balances, and it can fall over. This is the one whose
              behaviour is worth measuring.

A third backend, `pd_stand`, holds an analytic squat/lean pose with stiff PD. It does not walk;
it is there for posture work and as the simplest example of what a controller has to implement.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import imageio.v3 as iio
import numpy as np

from g1.wbc import CONTROLLERS, holosoma_available
from g1.robot.sim_source import SimSource, default_model_path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "workspace" / "out"

VX, WZ = 0.4, 0.0        # commanded body-frame velocity: 0.4 m/s straight ahead
SECONDS = 8.0
SETTLE = 2.0             # ignore the first seconds: both backends accelerate from standing
TICK = 0.1               # command refresh period; must be < SimSource.DEADMAN_S (0.3 s)
LANE_YAW = -math.pi / 2  # face -y, so the robot has clear floor ahead of it in every scene


def scene_path() -> str:
    if len(sys.argv) > 1:
        return sys.argv[1]
    local = REPO / "models" / "g1_pick_place.xml"
    return str(local) if local.exists() else default_model_path()


def drive(name: str) -> dict:
    """Walk one controller forward for SECONDS and report what happened."""
    # The controller is chosen at construction. With "kinematic" the scene's mocap weld stays
    # on; with a free-base controller SimSource switches the weld off at load time, so the same
    # XML serves both - you do not rebuild the scene to change controller.
    sim = SimSource(xml=scene_path(), realtime=False, controller=name)
    try:
        caps = sim.controller_caps()
        # Turn to face an empty lane first. Walking into the work table is a perfectly real
        # result but it measures the table, not the controller.
        sim.sim_teleport(0.0, 0.0, LANE_YAW)
        sim.step(1.5)                       # let the controller settle into its stance

        leg_joints = [j for j in sim.state().joints if "knee" in j or "ankle" in j]
        prev = np.array([sim.state().joints[j] for j in leg_joints])
        start = np.array(sim.state().base_pos)
        settled = None

        sim.set_mode("walk")                # non-zero velocity is refused in any other mode
        leg_travel = 0.0
        for i in range(int(SECONDS / TICK)):
            sim.set_base_velocity(VX, 0.0, WZ)   # refresh: the command expires after 0.3 s
            sim.step(TICK)
            q = np.array([sim.state().joints[j] for j in leg_joints])
            leg_travel += float(np.abs(q - prev).sum())
            prev = q
            if i == int(SETTLE / TICK) - 1:
                settled = np.array(sim.state().base_pos)
        sim.set_base_velocity(0.0, 0.0, 0.0)

        s = sim.state()
        end = np.array(s.base_pos)
        iio.imwrite(OUT / f"05_{name}.png", sim.rgb("third_person"))

        # Upright check: the body-frame z of the world "up" axis. 1.0 is perfectly upright.
        w, x, y, z = s.base_quat
        steady = float(np.linalg.norm((end - settled)[:2]) / (SECONDS - SETTLE))
        return {
            "name": name, "caps": caps, "travelled": float(np.linalg.norm((end - start)[:2])),
            "steady": steady, "pelvis_z": float(end[2]), "upright": float(1 - 2 * (x * x + y * y)),
            "leg_travel": leg_travel, "fault": s.extra["controller_fault"],
        }
    finally:
        sim.close()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"available controllers: {sorted(CONTROLLERS)}")
    print(f"onnxruntime present (holosoma needs it): {holosoma_available()}")
    print(f"\ncommanding vx={VX} m/s for {SECONDS} s through each backend\n")

    rows = []
    for name in ("kinematic", "holosoma"):
        if name == "holosoma" and not holosoma_available():
            print("skipping holosoma: onnxruntime is not installed")
            continue
        r = drive(name)
        rows.append(r)
        c = r["caps"]
        print(f"{name}")
        print(f"  caps          walk={c['walk']} posture={c['posture']} "
              f"free_base={c['free_base']} stand_in={c['stand_in']} rate={c['rate_hz']} Hz")
        print(f"  distance      {r['travelled']:.3f} m in {SECONDS:.0f} s")
        print(f"  steady speed  {r['steady']:.3f} m/s against a {VX} m/s command "
              f"({100 * r['steady'] / VX:.0f}% of commanded)")
        print(f"  pelvis height {r['pelvis_z']:.3f} m, uprightness {r['upright']:.3f}")
        print(f"  leg joint motion (knees + ankles, summed): {r['leg_travel']:.2f} rad")
        print(f"  controller fault: {r['fault']}")
        print()

    if len(rows) == 2:
        k, h = rows
        print("the difference, in one line each:")
        print(f"  kinematic: tracked the command exactly ({k['steady']:.3f} m/s) because there is "
              "nothing to track it with - the pelvis is moved by arithmetic. Its "
              f"{k['leg_travel']:.1f} rad of leg motion is the legs being *dragged* along the "
              "floor by the weld, not driving anything.")
        print(f"  holosoma : {h['leg_travel']:.1f} rad of leg motion - it stepped - and reached "
              f"{h['steady']:.3f} m/s, {100 * h['steady'] / VX:.0f}% of the command. Tracking "
              "error, foot slip and acceleration limits are all real, and it can fall over.")
        print("\nIf you are writing a locomotion policy, holosoma is your baseline and the thing")
        print("to beat. If you are writing a manipulation or navigation policy, kinematic keeps")
        print("the base out of your way - but anything you measure about the base under it is a")
        print("property of the stand-in, not of the robot. Check `caps['stand_in']` and say so")
        print("in anything you log, or you will compare numbers that do not mean the same thing.")

    # --- what a real controller refuses ---------------------------------------------------
    if holosoma_available():
        sim = SimSource(xml=scene_path(), realtime=False, controller="holosoma")
        try:
            try:
                sim.set_joint_targets({"left_knee_joint": 0.5})
            except ValueError as e:
                print(f"\nleg joints are not commandable while a controller owns them: {e}")
            try:
                sim.set_base_velocity(0.3, 0.0, 0.0)     # still in "stand"
            except RuntimeError as e:
                print(f"and a non-zero velocity outside walk mode: {e}")
            sim.set_base_velocity(0.0, 0.0, 0.0)         # zero is always accepted: it is the e-stop
            print("zero velocity is accepted in every mode - that is what makes it a safe stop.")
        finally:
            sim.close()

    print(f"\nwrote {OUT / '05_kinematic.png'} and {OUT / '05_holosoma.png'}")


if __name__ == "__main__":
    main()
