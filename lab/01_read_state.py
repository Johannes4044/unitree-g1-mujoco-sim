# /// script
# requires-python = "==3.10.*"   # cyclonedds 0.10.x (what the G1 speaks) ships wheels up to 3.10 only
# dependencies = [
#   "unitree_sdk2py @ https://github.com/unitreerobotics/unitree_sdk2_python/archive/814556d15970dd2ecf1c9984e845ca02ab07e206.zip",
#   "cyclonedds==0.10.2",
#   "numpy",
# ]
# ///
"""01 - Read the robot. Read-only: publishes nothing, cannot move anything.

    uv run lab/01_read_state.py en7               # en7 = the Ethernet adapter wired to the G1
    uv run lab/01_read_state.py en7 --seconds 30  # record longer

Prints every motor with the sim's joint name next to it, the IMU and the message rate, then
records rt/lowstate for --seconds into lab/out/state_<time>.npz (q, dq, tau_est, both motor
temperatures, IMU) - the real numbers the sim and RealSource get checked against.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
from common import BODY_JOINTS, OUT, LowState


def snapshot(msg) -> dict[str, np.ndarray]:
    m = [msg.motor_state[i] for i in range(len(BODY_JOINTS))]
    imu = msg.imu_state
    return {
        "q": np.array([x.q for x in m]), "dq": np.array([x.dq for x in m]),
        "tau_est": np.array([x.tau_est for x in m]), "temperature": np.array([list(x.temperature) for x in m]),
        "motor_mode": np.array([x.mode for x in m]), "quat_wxyz": np.array(imu.quaternion),
        "gyro": np.array(imu.gyroscope), "acc": np.array(imu.accelerometer), "rpy": np.array(imu.rpy),
        "mode_machine": np.array(msg.mode_machine), "tick": np.array(msg.tick),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("iface", help="network interface on the 192.168.123.x network, e.g. en7")
    ap.add_argument("--seconds", type=float, default=5.0)
    args = ap.parse_args()

    robot = LowState(args.iface)
    n0 = robot.count
    time.sleep(1.0)
    msg, _ = robot.latest()
    s = snapshot(msg)
    print(f"rt/lowstate: {robot.count - n0} msg/s (expect ~500), mode_machine={msg.mode_machine}")
    print(f"\n{'#':>2}  {'joint (sim name)':<28} {'q rad':>8} {'dq':>7} {'tau Nm':>7} {'temp':>9} mode")
    for i, name in enumerate(BODY_JOINTS):
        t = s["temperature"][i]
        print(f"{i:>2}  {name:<28} {s['q'][i]:+8.3f} {s['dq'][i]:+7.2f} {s['tau_est'][i]:+7.2f}"
              f" {t[0]:>4},{t[1]:<4} {s['motor_mode'][i]}")
    print(f"\nIMU rpy {np.round(s['rpy'], 3)}  gyro {np.round(s['gyro'], 3)}  acc {np.round(s['acc'], 2)}")

    print(f"\nrecording {args.seconds:.0f} s ...")
    rows, t0 = [], time.time()
    while time.time() - t0 < args.seconds:
        msg, _ = robot.latest()
        rows.append({"t": np.array(time.time() - t0), **snapshot(msg)})
        time.sleep(0.01)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"state_{time.strftime('%Y%m%d_%H%M%S')}.npz"
    np.savez(path, joints=np.array(BODY_JOINTS), **{k: np.stack([r[k] for r in rows]) for k in rows[0]})
    print(f"wrote {path} ({len(rows)} samples, {len(rows) / args.seconds:.0f} Hz)")


if __name__ == "__main__":
    main()
