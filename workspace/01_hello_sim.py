"""01 - Hello sim: load the world, step physics, look at the state, save a picture.

Run it:

    uv run python workspace/01_hello_sim.py
    uv run python workspace/01_hello_sim.py models/scenarios/house_rooms.xml

What to take away:

* `SimSource` is a `RobotSource` - the one interface the rest of the project talks to.
  Everything below (`state`, `rgb`, `depth`, `lidar`) is defined on that interface, so a real
  robot adapter would answer the same calls and this script would not change.
* `realtime=False` means *you* own the clock: nothing advances until you call `step(seconds)`.
  That is what you want in a script, a test or a training loop. `realtime=True` (the default)
  spawns a background thread that steps in wall-clock time, which is what an interactive
  viewer or a live teleop session wants.
"""
from __future__ import annotations

import sys
from pathlib import Path

import imageio.v3 as iio

from g1.robot.sim_source import SimSource, default_model_path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "workspace" / "out"


def scene_path() -> str:
    """The model to load: an argument if given, else the robot scene in this checkout.

    `default_model_path()` is the package's own resolver - it honours $SIM_MODEL first, so
    `SIM_MODEL=... python 01_hello_sim.py` works for every example in this directory.
    """
    if len(sys.argv) > 1:
        return sys.argv[1]
    local = REPO / "models" / "g1_pick_place.xml"
    return str(local) if local.exists() else default_model_path()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = scene_path()
    print(f"loading {path}")

    # The constructor compiles the MJCF, allocates MjData, resets to the `stand` keyframe and
    # picks a base controller. With the default (mocap-welded) scene that is "kinematic".
    sim = SimSource(xml=path, realtime=False)
    try:
        # --- what did we actually load? -------------------------------------------------
        # `sim.m` is the compiled mujoco.MjModel and `sim.d` the live mujoco.MjData. Reading
        # them directly is fine; writing them behind SimSource's back is not (it owns the
        # physics thread and the render snapshot).
        print(f"model      : nq={sim.m.nq} dof={sim.m.nv} joints={sim.m.njnt} "
              f"actuators={sim.m.nu} geoms={sim.m.ngeom}")
        print(f"timestep   : {sim.m.opt.timestep * 1000:.1f} ms "
              f"({1 / sim.m.opt.timestep:.0f} Hz physics)")
        caps = sim.controller_caps()
        print(f"controller : {sim.controller}  walks={caps['walk']}  stand_in={caps['stand_in']}")
        # `model_info` is how a recorded episode says which world it came from: the sha256 is of
        # the *compiled* model, so a rebuilt-but-changed scene cannot pass as the old one.
        print(f"world      : {sim.model_info['scenario']}  sha256={sim.model_info['sha256'][:12]}")

        # --- the state frame -------------------------------------------------------------
        s0 = sim.state()
        print(f"\nstate(): {len(s0.joints)} named joints, sim t={s0.t:.3f}s, mode={s0.mode!r}")
        # A few slots of the 31-slot observation layout, by name (see workspace/README.md):
        for name in ("left_shoulder_pitch_joint", "right_elbow_joint",
                     "left_index_proximal_joint", "head_pan_joint"):
            print(f"  {name:<28} {s0.joints[name]:+.4f} rad")
        print(f"  pelvis position              {[round(v, 3) for v in s0.base_pos]} m")
        # Free bodies in the scene are reported in extra["objects"] - this is the prompt
        # vocabulary a language-conditioned policy gets to refer to.
        print(f"  manipulable objects          {sorted(s0.extra['objects'])}")

        # --- step physics ----------------------------------------------------------------
        sim.step(1.0)                      # one second of simulated time, as fast as it computes
        s1 = sim.state()
        print(f"\nafter step(1.0): sim t={s1.t:.3f}s, {s1.extra['ncon']} contacts")
        drift = max(abs(s1.joints[n] - s0.joints[n]) for n in s0.joints)
        print(f"largest joint move while just standing: {drift:.5f} rad "
              "(the position servos hold the keyframe pose)")

        # --- save something you can look at ----------------------------------------------
        # `third_person` is a sim-only camera (it does not exist on the robot, so
        # RobotSource.rgb returns None for it on a real source).
        frame = sim.rgb("third_person")
        png = OUT / "01_third_person.png"
        iio.imwrite(png, frame)
        print(f"\nwrote {png.relative_to(REPO)}  {frame.shape} {frame.dtype}")
    finally:
        # Releases this thread's GL contexts and stops the physics thread if there was one.
        sim.close()


if __name__ == "__main__":
    main()
