"""06 - Record an episode: a short trajectory in plain .npz + .json, in the 31-slot layout.

Run it:

    uv run python workspace/06_record_an_episode.py

Why plain files
---------------
This repo deliberately does **not** depend on lerobot. LeRobot is a fine dataset format and the
full project uses it, but it pulls in torch and an ffmpeg build whose licence we do not want in
a simulator-only public repo. So the episode written here is two files anyone can read:

    workspace/out/episode_000.npz    the arrays
    workspace/out/episode_000.json   what the arrays mean

Converting to LeRobot later is a pure data transform and needs none of this code: see the
comment at the bottom of this file for exactly where it slots in.

The 31-slot layout
------------------
`observation.state` and `action` are the same 31 columns, in the same order, every frame:

    index  count  what                                units   source
    0-13      14  arm joint positions                 rad     left 7 then right 7, BODY_JOINTS order
    14-25     12  hand joint positions                rad     left 6 then right 6, Revo 2 proximals
    26-27      2  head_pan_joint, head_tilt_joint     rad     the 2-DoF head module
    28-30      3  base_vx, base_vy [m/s], base_wz     m/s,    body frame
                                           [rad/s]   rad/s

The last three slots mean different things in the two vectors, and this is the single easiest
thing to get wrong:

    observation.state[28:31]  the base velocity as *measured*
    action[28:31]             the base velocity as *commanded*

The joint slots follow the same rule: the observation holds where the joint *is*, the action
holds where it was last *told to go*. An action column therefore holds its value between
commands - it is a held setpoint stream, not a delta stream. The first action of an episode is
"stay where you are, zero base velocity", i.e. the first observation with slots 28-31 zeroed.

Legs and waist are not in the layout at all. They belong to the whole-body controller
(example 05), so a policy trained on these columns cannot command them - by construction.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

from g1.robot.robot_source import BODY_JOINTS, HAND_JOINTS, HEAD_JOINTS, RobotSource, RobotState
from g1.robot.sim_source import SimSource, default_model_path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "workspace" / "out"

# --- the layout, built from the joint lists the robot interface already defines ------------
ARM_JOINTS = [j for j in BODY_JOINTS if any(k in j for k in ("shoulder", "elbow", "wrist"))]
JOINT_NAMES = ARM_JOINTS + HAND_JOINTS + HEAD_JOINTS          # 14 + 12 + 2 = 28
BASE_NAMES = ["base_vx", "base_vy", "base_wz"]
STATE_NAMES = JOINT_NAMES + BASE_NAMES                        # 31
STATE_DIM = len(STATE_NAMES)
JOINT_INDEX = {n: i for i, n in enumerate(JOINT_NAMES)}
BASE_SLICE = slice(len(JOINT_NAMES), STATE_DIM)
assert STATE_DIM == 31, STATE_DIM

FPS = 30.0            # control and recording rate
SECONDS = 3.0
CAM_W, CAM_H = 256, 144    # small on purpose: 90 frames x 2 cameras is already 20 MB raw


def scene_path() -> str:
    if len(sys.argv) > 1:
        return sys.argv[1]
    local = REPO / "models" / "g1_pick_place.xml"
    return str(local) if local.exists() else default_model_path()


def observation(state: RobotState) -> np.ndarray:
    """The 31-slot observation: joint positions, then the *measured* base velocity.

    A joint the source does not report is zero - which is also what it would be on a robot
    whose hand controller is offline, so read it as "no data", not as "the joint is at zero".
    """
    vec = np.zeros(STATE_DIM, dtype=np.float32)
    for i, name in enumerate(JOINT_NAMES):
        vec[i] = float(state.joints.get(name, 0.0))
    if state.base_lin_vel is not None and state.base_ang_vel is not None:
        # A free-base controller measures these. Under the kinematic stand-in they are None,
        # because the welded pelvis's velocity is constraint-driven, not the one we commanded;
        # leaving them at zero there is honest - do not substitute the command.
        vec[BASE_SLICE] = [state.base_lin_vel[0], state.base_lin_vel[1], state.base_ang_vel[2]]
    return vec


def apply_command(name: str, args: dict, held: np.ndarray) -> None:
    """Fold one command into the held action vector, in place. The inverse of reading it back."""
    if name == "set_joints":
        for joint, value in args["targets"].items():
            i = JOINT_INDEX.get(joint)
            if i is not None:
                held[i] = float(value)
    elif name == "set_head":
        for key, joint in (("pan", "head_pan_joint"), ("tilt", "head_tilt_joint")):
            if args.get(key) is not None:
                held[JOINT_INDEX[joint]] = float(args[key])
    elif name == "cmd_vel":
        held[BASE_SLICE] = [args.get("vx", 0.0), args.get("vy", 0.0), args.get("wz", 0.0)]
    else:
        raise ValueError(f"unknown command {name!r}")


def send(source: RobotSource, name: str, args: dict, held: np.ndarray) -> None:
    """Apply a command to the robot *and* to the held action vector, so the two cannot drift."""
    if name == "set_joints":
        source.set_joint_targets(args["targets"])
    elif name == "set_head":
        source.set_head(pan=args.get("pan"), tilt=args.get("tilt"))
    elif name == "cmd_vel":
        source.set_base_velocity(args["vx"], args["vy"], args["wz"])
    apply_command(name, args, held)


def script(t: float) -> list[tuple[str, dict]]:
    """A tiny scripted "reach and close the hand" trajectory. Replace with a teleop stream, a
    scripted skill or a policy's own output - the recording code below does not care."""
    out: list[tuple[str, dict]] = []
    if t < 0.05:
        out.append(("set_head", {"pan": 0.0, "tilt": 0.75}))      # look at the table
    a = min(1.0, t / 1.5)                                          # 0 -> 1 over 1.5 s
    out.append(("set_joints", {"targets": {
        "left_shoulder_pitch_joint": 0.2 + a * (-1.2),
        "left_shoulder_roll_joint": 0.2 + a * 0.7,
        "left_elbow_joint": 1.28 + a * (-0.3),
    }}))
    if t > 1.6:                                                    # then close the left hand
        b = min(1.0, (t - 1.6) / 1.0)
        out.append(("set_joints", {"targets": {
            j: b for j in HAND_JOINTS if j.startswith("left_")}}))
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    sim = SimSource(xml=scene_path(), realtime=False, cam_w=CAM_W, cam_h=CAM_H)
    try:
        sim.sim_reset()
        # Randomising the object layout per episode is one line, deterministic for a seed, and
        # is what stops a VLA memorising absolute positions. The returned dict (including the
        # seed) belongs in the episode metadata so the episode can be reproduced exactly.
        rand = sim.sim_randomise(seed=7)
        sim.step(0.5)

        n = int(SECONDS * FPS)
        dt = 1.0 / FPS
        states = np.zeros((n, STATE_DIM), dtype=np.float32)
        actions = np.zeros((n, STATE_DIM), dtype=np.float32)
        stamps = np.zeros(n, dtype=np.float64)
        d455 = np.zeros((n, CAM_H, CAM_W, 3), dtype=np.uint8)
        d435i = np.zeros((n, CAM_H, CAM_W, 3), dtype=np.uint8)

        # The first action is "hold the current pose, zero base velocity".
        held = observation(sim.state())
        held[BASE_SLICE] = 0.0

        t_start = time.perf_counter()
        for i in range(n):
            t = i * dt
            # Record the observation BEFORE acting on it: frame i is (what the robot saw,
            # what we then commanded). Getting this one step out is the classic way to train a
            # policy that looks great offline and lags by a frame on the robot.
            s = sim.state()
            states[i] = observation(s)
            d455[i] = sim.rgb("d455_rgb")
            d435i[i] = sim.rgb("d435i_rgb")
            stamps[i] = s.t

            for name, args in script(t):
                send(sim, name, args, held)
            actions[i] = held

            sim.step(dt)
        wall = time.perf_counter() - t_start

        # --- write it out ------------------------------------------------------------------
        npz = OUT / "episode_000.npz"
        np.savez_compressed(
            npz,
            **{"observation.state": states, "action": actions, "timestamp": stamps,
               "observation.images.d455_rgb": d455, "observation.images.d435i_rgb": d435i},
        )
        meta = {
            "fps": FPS, "frames": n, "state_dim": STATE_DIM,
            "state_names": STATE_NAMES,
            "slots": {"arm": [0, 14], "hand": [14, 26], "head": [26, 28], "base": [28, 31]},
            "units": {"joints": "rad", "base_vx": "m/s", "base_vy": "m/s", "base_wz": "rad/s"},
            "semantics": {"observation.state[28:31]": "measured base velocity, body frame",
                          "action[28:31]": "commanded base velocity, body frame",
                          "action[0:28]": "held joint setpoints, not deltas"},
            "cameras": {"d455_rgb": [CAM_H, CAM_W, 3], "d435i_rgb": [CAM_H, CAM_W, 3]},
            "task": "pick up the red box",
            "sim_model": sim.model_info,
            "controller": sim.controller_caps(),
            "randomisation": {k: v for k, v in rand.items() if k in ("seed", "objects", "placed")},
        }
        js = OUT / "episode_000.json"
        js.write_text(json.dumps(meta, indent=2, default=str))

        # --- prove it worked ---------------------------------------------------------------
        print(f"recorded {n} frames at {FPS:g} Hz ({SECONDS:g} s of sim) in {wall:.1f} s wall clock")
        print(f"  observation.state {states.shape} {states.dtype}")
        print(f"  action            {actions.shape} {actions.dtype}")
        print(f"  images            2 x {d455.shape} {d455.dtype}")
        print(f"  randomisation seed {rand['seed']}, placed {sorted(rand['placed'])}")
        print(f"\nwrote {npz.name} ({npz.stat().st_size / 1e6:.1f} MB) and {js.name}")

        moved = np.abs(states[-1, :28] - states[0, :28])
        print("\njoints that moved most over the episode:")
        for i in np.argsort(moved)[::-1][:5]:
            print(f"  {STATE_NAMES[i]:<28} {states[0, i]:+.3f} -> {states[-1, i]:+.3f} rad "
                  f"(commanded {actions[-1, i]:+.3f})")
        print(f"\nbase slots stayed at {actions[-1, 28:31].tolist()} - this episode never "
              "commanded the base, so a policy trained on it has no reason to.")

        # --- read it back, the way a dataloader would ---------------------------------------
        z = np.load(npz)
        back = z["observation.state"]
        print(f"\nreloaded: keys={list(z.keys())}, state round-trips exactly: "
              f"{np.array_equal(back, states)}")
    finally:
        sim.close()


# -------------------------------------------------------------------------------------------
# Where a LeRobot conversion would go
# -------------------------------------------------------------------------------------------
# Nothing above needs to change. A converter is a separate script, in a separate environment
# that is allowed to depend on lerobot (and therefore on torch and on an ffmpeg build), and it
# reads only the two files written here:
#
#     ds = LeRobotDataset.create(repo_id, fps=meta["fps"], features={
#         "observation.state": {"dtype": "float32", "shape": (31,), "names": meta["state_names"]},
#         "action":            {"dtype": "float32", "shape": (31,), "names": meta["state_names"]},
#         "observation.images.d455_rgb":  {"dtype": "video", "shape": (H, W, 3)},
#         "observation.images.d435i_rgb": {"dtype": "video", "shape": (H, W, 3)},
#     })
#     for i in range(meta["frames"]):
#         ds.add_frame({k: z[k][i] for k in z}, task=meta["task"])
#     ds.save_episode()
#
# Two things to carry across, because they are what makes a dataset trainable rather than
# merely well-formed: the `names` list (so the columns stay identified by name, not by luck)
# and the sim_model sha256 from the JSON (so you can tell which worlds an episode came from
# when a policy turns out to work in one scenario and not another).
# -------------------------------------------------------------------------------------------

if __name__ == "__main__":
    main()
