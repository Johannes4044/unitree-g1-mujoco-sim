"""The whole-body controller interface: what stands between a velocity command and the leg motors.

A VLA or the navigator emits a base velocity (and arm / hand targets); a whole-body controller
turns that into leg and waist joint targets that keep the robot upright. This module defines the
contract every such controller implements, so the simulator (and later the real robot) can drive
any of them the same way:

    ctrl.set_mode("stand" | "walk" | "damp")
    ctrl.set_base_velocity(vx, vy, wz)            # body frame, only in "walk"
    ctrl.set_posture(base_height, torso_pitch)    # only if ctrl.supports_posture
    ctrl.reset(state)                             # after a teleport / sim reset
    out = ctrl.step(state)                        # once per control period (ctrl.rate_hz)

`step` returns a `ControlOutput`: a `JointCommand` (PD targets for the joints the controller owns)
that the caller turns into torques at the *physics* rate, `tau = kp (q* - q) + kd (qd* - qd) + tau`.
So the controller runs at its own rate (50 Hz for the learned policies) while the PD loop runs
inside every physics step, the way the G1's motor drivers do it on the real robot.

Ownership: a controller owns `ctrl.joints` (legs and waist). Those joints are never commandable
from the API - `SafetyGate` rejects them and `SimSource.set_joint_targets` refuses them - so the
only thing that drives them is the controller, or the damping fallback below.

Damping: mode "damp" is zero stiffness plus joint damping, what the G1's own damping mode does.
The robot sags to the floor in a controlled way instead of freezing stiffly in whatever pose it
had. It is implemented here, not by each controller, so it is the same everywhere and it is what
the caller falls back to when a controller faults (`damping_command`).
"""
from __future__ import annotations

import abc
import math
from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np

from g1.robot.robot_source import BODY_JOINTS

LEG_WAIST_JOINTS: tuple[str, ...] = tuple(BODY_JOINTS[:15])   # 12 leg + 3 waist joints, G1 motor order
ARM_JOINTS: tuple[str, ...] = tuple(BODY_JOINTS[15:])
MODES = ("damp", "stand", "walk")

# Joint damping in damp mode [N m s/rad]: enough that a collapse is a slow sag, not a drop, and
# far too little to hold a pose. Unitree's own damping mode is kd-only as well.
DAMP_KD = 4.0


@dataclass
class BodyState:
    """What a controller may observe: proprioception only, as the real robot provides it."""
    t: float                      # time [s] (sim time in the simulator)
    q: np.ndarray                 # (29,) joint positions in BODY_JOINTS order [rad]
    qd: np.ndarray                # (29,) joint velocities [rad/s]
    base_quat: np.ndarray         # (4,) pelvis orientation, world frame, wxyz
    base_ang_vel: np.ndarray      # (3,) pelvis angular velocity in the pelvis frame (IMU gyro) [rad/s]


@dataclass
class JointCommand:
    """PD targets for `joints`: tau = kp (q - q_meas) + kd (qd - qd_meas) + tau."""
    joints: tuple[str, ...]
    q: np.ndarray
    kp: np.ndarray
    kd: np.ndarray
    qd: np.ndarray | None = None
    tau: np.ndarray | None = None

    def torques(self, q_meas: np.ndarray, qd_meas: np.ndarray) -> np.ndarray:
        qd_ref = 0.0 if self.qd is None else self.qd
        out = self.kp * (self.q - q_meas) + self.kd * (qd_ref - qd_meas)
        return out if self.tau is None else out + self.tau

    def validate(self) -> None:
        """Raise ControllerFault unless every array has one finite entry per joint."""
        n = len(self.joints)
        for nm in ("q", "kp", "kd", "qd", "tau"):
            v = getattr(self, nm)
            if v is None:
                continue
            a = np.asarray(v, dtype=float)
            if a.shape != (n,):
                raise ControllerFault(f"{nm} has shape {a.shape}, expected ({n},)")
            if not np.isfinite(a).all():
                raise ControllerFault(f"{nm} is not finite")
        if (np.asarray(self.kp) < 0).any() or (np.asarray(self.kd) < 0).any():
            raise ControllerFault("negative gain")


@dataclass
class ControlOutput:
    joints: JointCommand | None = None
    # Kinematic stand-in only: body-frame (vx, vy, wz) the simulator integrates into the mocap base.
    base_twist: tuple[float, float, float] | None = None


class ControllerFault(RuntimeError):
    """A controller produced something that must not reach the motors."""


def damping_command(joints: tuple[str, ...], kd: float = DAMP_KD) -> JointCommand:
    n = len(joints)
    return JointCommand(joints=joints, q=np.zeros(n), kp=np.zeros(n), kd=np.full(n, float(kd)))


def projected_gravity(quat_wxyz) -> np.ndarray:
    """World gravity direction (0, 0, -1) expressed in the body frame."""
    w, x, y, z = (float(v) for v in quat_wxyz)
    # third row of R(q), negated: R^T @ (0, 0, -1)
    return -np.array([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)])


class WholeBodyController(abc.ABC):
    """Base class. Subclasses implement `reset` and `_step`; everything else is shared."""

    name = "abstract"
    rate_hz: float | None = 50.0         # None: step every physics step
    needs_free_base = True               # False only for the kinematic stand-in
    needs_state = True                   # False: reset/step ignore the BodyState (callers may pass None)
    stand_in = False                     # True: a simulation stand-in, not a controller that transfers
    supports_walk = False
    supports_posture = False
    modes: tuple[str, ...] = MODES
    initial_mode = "stand"
    joints: tuple[str, ...] = LEG_WAIST_JOINTS
    # commanded velocity is clipped to these [m/s, m/s, rad/s] (the SafetyGate clips tighter)
    velocity_limits = (1.0, 1.0, 1.0)
    # {"base_height": (lo, hi), "torso_pitch": (lo, hi)} for posture-capable controllers
    posture_limits: ClassVar[dict[str, tuple[float, float]]] = {}

    def __init__(self) -> None:
        self._mode = self.initial_mode
        self._vel = np.zeros(3)
        self._posture: dict[str, float] = {}

    # --- commands -----------------------------------------------------------------
    @property
    def mode(self) -> str:
        return self._mode

    def set_mode(self, mode: str) -> None:
        if mode not in self.modes:
            raise ValueError(f"the {self.name} controller has no mode {mode!r}; modes: {list(self.modes)}")
        old, self._mode = self._mode, mode
        self._vel[:] = 0.0              # a mode change never carries a stale command across
        self._on_mode(old, mode)

    def set_base_velocity(self, vx: float, vy: float, wz: float) -> None:
        """Body-frame velocity command.

        A zero velocity is the safe command and is accepted in every mode by every controller -
        before any mode is set, in damp and stand, after a fault, and by controllers that cannot
        walk at all. Where the base is not being driven, zero already is the true state, so it is
        a correct no-op. The bridge's dead-man, disarm and e-stop paths rely on this: a refused
        zero is reported as a failed stop and retried every tick. Only a *non-zero* velocity
        outside walk mode (or on a controller that cannot walk) is refused. Subclasses must not
        override this without keeping that property."""
        v = np.array([vx, vy, wz], dtype=float)
        if not np.isfinite(v).all():
            raise ValueError("velocity must be finite")
        if not v.any():
            self.zero_velocity()
            return
        if not self.supports_walk:
            raise NotImplementedError(f"the {self.name} controller cannot walk")
        if self._mode != "walk":
            raise RuntimeError("set_mode('walk') first")
        lim = np.asarray(self.velocity_limits, dtype=float)
        self._vel[:] = np.clip(v, -lim, lim)

    def zero_velocity(self) -> None:
        """Dead-man / e-stop path: zero the velocity command in any mode, never raises."""
        self._vel[:] = 0.0

    def set_posture(self, base_height: float, torso_pitch: float) -> dict[str, float]:
        """Set the posture target; returns the values actually applied (after clamping)."""
        if not self.supports_posture:
            raise NotImplementedError(f"the {self.name} controller has no posture input")
        applied = {}
        for k, v in (("base_height", base_height), ("torso_pitch", torso_pitch)):
            f = float(v)
            if not math.isfinite(f):
                raise ValueError(f"{k} must be finite")
            lo, hi = self.posture_limits[k]
            applied[k] = min(hi, max(lo, f))
        self._posture.update(applied)
        return dict(applied)

    @property
    def velocity(self) -> tuple[float, float, float]:
        return tuple(float(v) for v in self._vel)

    def caps(self) -> dict:
        """What the controller can do, for the UI and the bridge's command gating."""
        return {"name": self.name, "modes": list(self.modes), "walk": self.supports_walk,
                "posture": self.supports_posture,
                "posture_limits": {k: list(v) for k, v in self.posture_limits.items()},
                "rate_hz": self.rate_hz, "free_base": self.needs_free_base,
                # sim_teleport can move a free base; a non-free one only via the source's mocap
                # target, which only the source knows about (SimSource.controller_caps refines it)
                "teleport": self.needs_free_base,
                "stand_in": self.stand_in}

    # --- control --------------------------------------------------------------------
    def step(self, s: BodyState) -> ControlOutput:
        """One control period. Damp mode is handled here so it is identical for every controller."""
        if self._mode == "damp":
            return ControlOutput(joints=damping_command(self.joints) if self.joints else None)
        return self._step(s)

    @abc.abstractmethod
    def reset(self, s: BodyState) -> None:
        """Forget internal state (history, phase, ramps) and start from `s`."""

    @abc.abstractmethod
    def _step(self, s: BodyState) -> ControlOutput: ...

    def start_pose(self) -> dict[str, float]:
        """Joint pose a simulator should start the robot in (may include arms). Empty: keyframe."""
        return {}

    def _on_mode(self, old: str, new: str) -> None:
        """Optional hook: called after every mode change."""


@dataclass
class Ramp:
    """Linear interpolation of joint targets from where the robot is to where it should be."""
    t0: float
    q0: np.ndarray
    duration: float
    done: bool = field(default=False)

    def target(self, t: float, goal: np.ndarray) -> np.ndarray:
        a = 1.0 if self.duration <= 0 else min(1.0, max(0.0, (t - self.t0) / self.duration))
        self.done = a >= 1.0
        return self.q0 + (goal - self.q0) * a
