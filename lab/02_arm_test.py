# /// script
# requires-python = "==3.10.*"   # cyclonedds 0.10.x (what the G1 speaks) ships wheels up to 3.10 only
# dependencies = [
#   "unitree_sdk2py @ https://github.com/unitreerobotics/unitree_sdk2_python/archive/814556d15970dd2ecf1c9984e845ca02ab07e206.zip",
#   "cyclonedds==0.10.2",
#   "numpy",
# ]
# ///
"""02 - Move ONE arm joint a little, through Unitree's arm_sdk, and record how it tracked.

    uv run lab/02_arm_test.py en7                                   # left elbow, +0.3 rad
    uv run lab/02_arm_test.py en7 --joint right_shoulder_roll_joint --delta -0.2

Safe by construction, as far as code can be:
* rt/arm_sdk, not rt/lowcmd: Unitree's own controller keeps balancing the legs, we only borrow
  the upper body. No debug mode needed.
* Every arm and waist joint is held at the pose measured at start; only --joint moves, by at
  most 0.5 rad, never past the sim model's joint limit minus 0.1 rad.
* Control is faded in and out over 1 s (arm_sdk weight 0 -> 1 -> 0), motion is a cosine ramp.
* Ctrl-C, a state older than 0.1 s, or a tracking error over 0.5 rad -> hold where the arm is
  and fade control back to Unitree's controller.
The e-stop on the Unitree remote is still the real safety. Keep a hand on it.

Writes lab/out/arm_<time>.npz: commanded vs measured q of the joint, weight, phase - run the
same motion in the sim afterwards and compare.
"""
from __future__ import annotations

import argparse
import math
import signal
import time
import xml.etree.ElementTree as ET

import numpy as np
from common import BODY_JOINTS, OUT, REPO, LowState

HELD = list(range(12, 29))      # waist + both arms; Unitree's arm_sdk examples hold exactly these
ARM = BODY_JOINTS[15:29]
WEIGHT_SLOT = 29                # motor_cmd[29].q is the arm_sdk blend weight
KP, KD = 60.0, 1.5              # gains from Unitree's g1_arm7_sdk_dds_example.py
DT = 0.02
MAX_DELTA, LIMIT_MARGIN, MAX_ERR, MAX_AGE = 0.5, 0.1, 0.5, 0.1


def joint_limits() -> dict[str, tuple[float, float]]:
    root = ET.parse(REPO / "models" / "g1_pick_place.xml").getroot()
    return {j.get("name"): tuple(map(float, j.get("range").split()))
            for j in root.iter("joint") if j.get("range") and j.get("name")}


def smooth(s: float) -> float:
    return 0.5 - 0.5 * math.cos(math.pi * min(max(s, 0.0), 1.0))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("iface", help="network interface on the 192.168.123.x network, e.g. en7")
    ap.add_argument("--joint", default="left_elbow_joint", choices=ARM)
    ap.add_argument("--delta", type=float, default=0.3, help=f"rad, |delta| <= {MAX_DELTA}")
    ap.add_argument("--seconds", type=float, default=3.0, help="duration of the move out (and back)")
    args = ap.parse_args()
    if abs(args.delta) > MAX_DELTA:
        raise SystemExit(f"--delta {args.delta} is over {MAX_DELTA} rad; this is a first test")
    if args.seconds < 1.0:
        raise SystemExit("--seconds under 1 s is not a first test")

    from unitree_sdk2py.core.channel import ChannelPublisher
    from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_
    from unitree_sdk2py.utils.crc import CRC

    robot = LowState(args.iface)
    start = robot.q()
    j = BODY_JOINTS.index(args.joint)
    a, b = start[j], start[j] + args.delta
    lo, hi = joint_limits()[args.joint]
    if not lo + LIMIT_MARGIN <= b <= hi - LIMIT_MARGIN:
        raise SystemExit(f"{args.joint}: {a:+.3f} -> {b:+.3f} rad leaves [{lo:+.3f}, {hi:+.3f}] minus margin")

    print(f"{args.joint}: {a:+.3f} -> {b:+.3f} -> {a:+.3f} rad, {args.seconds:.1f} s each way. "
          f"Every other arm/waist joint holds where it is now.")
    if input("Robot standing or on the gantry, area clear, e-stop in hand? type yes: ").strip() != "yes":
        raise SystemExit("not moving")

    pub = ChannelPublisher("rt/arm_sdk", LowCmd_)
    pub.Init()
    cmd, crc = unitree_hg_msg_dds__LowCmd_(), CRC()

    def send(q: list[float], weight: float) -> None:
        cmd.motor_cmd[WEIGHT_SLOT].q = weight
        for i in HELD:
            m = cmd.motor_cmd[i]
            m.q, m.dq, m.tau, m.kp, m.kd = q[i], 0.0, 0.0, KP, KD
        cmd.crc = crc.Crc(cmd)
        pub.Write(cmd)

    phases = [  # name, seconds, s -> (joint target, weight)
        ("engage", 1.0, lambda s: (a, s)),
        ("out", args.seconds, lambda s: (a + (b - a) * smooth(s), 1.0)),
        ("hold", 1.0, lambda s: (b, 1.0)),
        ("back", args.seconds, lambda s: (b + (a - b) * smooth(s), 1.0)),
        ("release", 1.0, lambda s: (a, 1.0 - s)),
    ]
    rows, q, weight, reason = [], list(start), 0.0, None
    t0 = tick = time.time()
    try:
        for p, (name, dur, f) in enumerate(phases):
            ts = time.time()
            while (s := (time.time() - ts) / dur) < 1.0:
                q[j], weight = f(s)
                send(q, weight)
                msg, age = robot.latest()
                measured = msg.motor_state[j].q
                rows.append((time.time() - t0, p, q[j], measured, weight))
                if age > MAX_AGE:
                    reason = f"robot state is {age * 1000:.0f} ms old"
                elif name != "release" and abs(measured - q[j]) > MAX_ERR:
                    reason = f"tracking error {measured - q[j]:+.2f} rad"
                if reason:
                    raise RuntimeError(reason)
                tick += DT
                time.sleep(max(0.0, tick - time.time()))
            if name == "hold":
                print(f"at target: commanded {b:+.3f}, measured {robot.q()[j]:+.3f} rad")
    except (KeyboardInterrupt, RuntimeError) as e:
        signal.signal(signal.SIGINT, signal.SIG_IGN)   # a second Ctrl-C must not cut the fade short
        print(f"\nABORT ({reason or type(e).__name__}): holding the measured pose, fading arm_sdk out")
        hold = robot.q()
        w0 = weight
        for k in range(1, 51):
            send(hold, w0 * (1 - k / 50))
            time.sleep(DT)
    finally:
        send(q, 0.0)   # make sure the last word on rt/arm_sdk is weight 0
        OUT.mkdir(parents=True, exist_ok=True)
        path = OUT / f"arm_{time.strftime('%Y%m%d_%H%M%S')}.npz"
        r = np.array(rows) if rows else np.zeros((0, 5))
        np.savez(path, joint=args.joint, phases=[p[0] for p in phases], t=r[:, 0], phase=r[:, 1],
                 q_cmd=r[:, 2], q_meas=r[:, 3], weight=r[:, 4], kp=KP, kd=KD)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
