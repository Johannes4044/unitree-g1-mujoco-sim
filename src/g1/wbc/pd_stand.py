"""A PD stand controller with a posture input: base height and torso pitch, no walking.

What it is: stiff joint PD around an analytic leg pose. For a commanded pelvis height the two
sagittal leg links (thigh, shank) are solved so the feet stay flat and directly under the hips;
torso pitch goes to the waist pitch joint, and the hips shift back a little so the centre of mass
stays over the feet when the torso leans forward. Nothing is learned and nothing balances
actively - it holds the robot up on a free base on flat ground because the pose is statically
stable, and it would not survive a real push. It cannot walk.

Why it exists: posture (squat to reach a low crate, lean over a table) is a command dimension a
VLA can use, and the learned walking controller shipped here (Holosoma) has no posture input.
This controller makes the posture interface real in the simulator - the bridge's `set_posture`,
the UI sliders and the recorded action all work end to end against it - and is the reference a
posture-capable learned controller (AMO, a retrained Holosoma variant) has to match.

Geometry is the G1's, measured from the MuJoCo model (identical to Unitree's URDF): hip pitch
axis 0.1027 m below the pelvis origin, thigh 0.3366 m and shank 0.300 m in the sagittal plane,
ankle pitch axis 0.0526 m above the sole.
"""
from __future__ import annotations

import math
from typing import ClassVar

import numpy as np

from g1.wbc.base import BodyState, ControlOutput, JointCommand, Ramp, WholeBodyController

HIP_Z = 0.1027          # hip pitch axis below the pelvis origin [m]
THIGH = 0.3366          # hip pitch -> knee, sagittal [m]
SHANK = 0.300           # knee -> ankle pitch, sagittal [m]
SOLE = 0.0526           # ankle pitch axis above the sole [m]
# Forward shift of the whole-body centre of mass per unit sin(torso pitch): torso subtree
# (16.5 kg of 34.9 kg) with its COM 0.197 m above the waist pitch axis. The hips move back by
# this much so a forward lean does not carry the COM off the front of the feet.
COM_SHIFT = 16.46 / 34.94 * 0.197
ANKLE_PITCH_MIN = -0.87  # joint limit [rad]; bounds how deep the robot can squat with flat feet
KNEE_MAX = 2.0

# Stiff hold gains: Holosoma's "stiff startup" gains for the G1 (hold-a-pose values, not the
# policy's own compliant gains). Order: hip pitch, roll, yaw, knee, ankle pitch, roll (x2), waist.
KP = np.array([350, 200, 200, 300, 300, 150, 350, 200, 200, 300, 300, 150, 200, 200, 200], dtype=float)
KD = np.array([5, 5, 5, 10, 5, 5, 5, 5, 5, 10, 5, 5, 5, 5, 5], dtype=float)

HEIGHT_RATE = 0.15      # posture targets move at most this fast [m/s]
PITCH_RATE = 0.5        # [rad/s]
RAMP_S = 1.0            # stand-up interpolation after reset / leaving damp [s]


def leg_angles(base_height: float, torso_pitch: float = 0.0) -> tuple[float, float, float]:
    """(hip_pitch, knee, ankle_pitch) for a pelvis at `base_height` above flat ground, pelvis level.

    Raises ValueError when the height is out of reach (legs too short or ankle past its limit)."""
    depth = base_height - HIP_Z - SOLE                   # hip axis above the ankle axis
    fwd = COM_SHIFT * math.sin(torso_pitch)              # ankle ahead of the hip = hips move back
    r2 = depth * depth + fwd * fwd
    c = (r2 - THIGH ** 2 - SHANK ** 2) / (2 * THIGH * SHANK)
    if c > 1.0 + 1e-9:
        raise ValueError(f"base height {base_height:.3f} m is above the straight-leg height")
    knee = math.acos(min(1.0, max(-1.0, c)))
    thigh = math.atan2(fwd, depth) + math.atan2(SHANK * math.sin(knee), THIGH + SHANK * math.cos(knee))
    hip, ankle = -thigh, thigh - knee                    # G1 sign convention: flexion is hip < 0
    if ankle < ANKLE_PITCH_MIN or knee > KNEE_MAX:
        raise ValueError(f"base height {base_height:.3f} m needs ankle {ankle:.2f} / knee {knee:.2f} rad")
    return hip, knee, ankle


def _reachable_min(pitch_extreme: float = 0.5) -> float:
    lo, hi = 0.35, 0.78
    for _ in range(40):                                   # bisect the lowest height valid at any pitch
        mid = (lo + hi) / 2
        try:
            leg_angles(mid, 0.0), leg_angles(mid, pitch_extreme), leg_angles(mid, -pitch_extreme)
            hi = mid
        except ValueError:
            lo = mid
    return math.ceil(hi * 100) / 100


class PDStandController(WholeBodyController):
    name = "pd_stand"
    rate_hz = 100.0
    supports_walk = False
    supports_posture = True
    modes = ("damp", "stand")
    posture_limits: ClassVar[dict[str, tuple[float, float]]] = {
        "base_height": (_reachable_min(), 0.78), "torso_pitch": (-0.5, 0.5)}
    DEFAULT_POSTURE: ClassVar[dict[str, float]] = {"base_height": 0.74, "torso_pitch": 0.0}

    def __init__(self) -> None:
        super().__init__()
        self._posture = dict(self.DEFAULT_POSTURE)
        self._current = dict(self.DEFAULT_POSTURE)     # rate-limited posture actually tracked
        self._ramp: Ramp | None = None
        self._t = None

    def targets(self, base_height: float, torso_pitch: float) -> np.ndarray:
        """Leg + waist joint targets (15, LEG_WAIST_JOINTS order) for a posture."""
        hip, knee, ankle = leg_angles(base_height, torso_pitch)
        leg = [hip, 0.0, 0.0, knee, ankle, 0.0]
        return np.array(leg + leg + [0.0, 0.0, torso_pitch])

    def start_pose(self) -> dict[str, float]:
        q = self.targets(**self.DEFAULT_POSTURE)
        return dict(zip(self.joints, (float(v) for v in q), strict=True))

    def reset(self, s: BodyState) -> None:
        self._current = dict(self._posture)
        self._ramp = Ramp(t0=s.t, q0=np.asarray(s.q[:15], dtype=float).copy(), duration=RAMP_S)
        self._t = s.t

    def _on_mode(self, old: str, new: str) -> None:
        if old == "damp" and new != "damp":
            self._ramp = None                            # re-ramp from wherever damping left it
            self._t = None

    def _step(self, s: BodyState) -> ControlOutput:
        if self._ramp is None:
            self.reset(s)
        dt = 0.0 if self._t is None else max(0.0, s.t - self._t)
        self._t = s.t
        for k, rate in (("base_height", HEIGHT_RATE), ("torso_pitch", PITCH_RATE)):
            d = self._posture[k] - self._current[k]
            self._current[k] += max(-rate * dt, min(rate * dt, d))
        goal = self.targets(self._current["base_height"], self._current["torso_pitch"])
        q = goal if self._ramp.done else self._ramp.target(s.t, goal)
        return ControlOutput(joints=JointCommand(joints=self.joints, q=q, kp=KP.copy(), kd=KD.copy()))
