"""Holosoma's pretrained G1 locomotion policy (Amazon FAR), run through onnxruntime on CPU.

Provenance and licence
----------------------
`models/holosoma/fastsac_g1_29dof.onnx` is copied byte for byte from
https://github.com/amazon-far/holosoma at commit bccd4d7451640a2800ddc77e469d911a84f91994,
path src/holosoma_inference/holosoma_inference/models/loco/g1_29dof/fastsac_g1_29dof.onnx
(sha256 8346fd90778439395922a8c7256f24125ae84b8dea949128bac9e23c02bc7717; first added in the
repository's "Initial public release", 9c238cf80f, 2025-11-15). The weights are part of that
repository and ship under its single licence, Apache-2.0; the repository has no separate model
licence, model card or usage restriction. `LICENSE` and `NOTICE` next to the model are the
upstream files, which is what Apache-2.0 section 4 asks a redistributor to carry. The runtime is
onnxruntime (MIT). Nothing here needs network access: the model is committed, so it crosses the
air gap with the source tree.

What the policy is
------------------
A 29-DoF velocity-tracking policy trained with FastSAC in simulation: 50 Hz, 100-d observation,
29-d action. Joint targets are `default + 0.25 * action`, tracked by joint PD with the gains
stored in the model's metadata. Commands: (vx, vy, wz), each within +-1. With a zero command it
stands (both feet at the same gait phase); there is no base-height or torso-pitch input, so this
controller does not support posture.

Observation, reproduced from holosoma_inference (policies/base.py, policies/locomotion.py,
config_values/observation.py `loco_g1_29dof`): the terms are concatenated in *sorted name order*
- actions(29), base_ang_vel(3) x0.25, command_ang_vel(1), command_lin_vel(2), cos_phase(2),
dof_pos - default(29), dof_vel(29) x0.05, projected_gravity(3), sin_phase(2) - and the gait phase
per foot advances by 2 pi / (50 Hz x 1.0 s), is pinned to pi for both feet while the command is
~zero, and restarts at (0, pi) when motion resumes.

Arms
----
The policy outputs all 29 joints, but this controller only applies the leg and waist targets:
arms stay the operator's (or the VLA's) to command. Walking holds up with the arms held
externally at the policy's nominal arm pose (measured: 0.44 m/s at a 0.5 m/s command, the same
as with the policy driving them); with the arms hanging straight down it still walks, but slower
(0.27 m/s). `start_pose()` therefore includes the nominal arm pose, which the simulator applies
as the initial arm target after a reset.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np

from g1.robot.robot_source import BODY_JOINTS
from g1.wbc.base import (
    BodyState,
    ControlOutput,
    JointCommand,
    Ramp,
    WholeBodyController,
    projected_gravity,
)
from g1.wbc.pd_stand import KD as HOLD_KD
from g1.wbc.pd_stand import KP as HOLD_KP

MODEL = Path(__file__).parent / "models" / "holosoma" / "fastsac_g1_29dof.onnx"
MODEL_SHA256 = "8346fd90778439395922a8c7256f24125ae84b8dea949128bac9e23c02bc7717"

# holosoma_inference config_values/robot.py g1_29dof.default_dof_angles, BODY_JOINTS order
DEFAULT_ANGLES = np.array([
    -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,       # left leg
    -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,       # right leg
    0.0, 0.0, 0.0,                              # waist
    0.2, 0.2, 0.0, 0.6, 0.0, 0.0, 0.0,          # left arm
    0.2, -0.2, 0.0, 0.6, 0.0, 0.0, 0.0,         # right arm
])
ACTION_SCALE = 0.25
GAIT_PERIOD = 1.0
RATE_HZ = 50.0
ANG_VEL_SCALE = 0.25
DOF_VEL_SCALE = 0.05
STAND_EPS = 0.01        # |command| below this counts as "stand still" (holosoma: 0.01)
# Start-up: when the legs are not already in the default pose (re-engaging after damping), the
# targets are interpolated to it over RAMP_S with stiff hold gains before the policy takes over,
# as holosoma_inference's own get-ready phase does. Within RAMP_SKIP of the default pose the policy
# starts at once: holding the pose with the policy's own compliant gains lets the robot tip over
# (measured: 25 deg after 1 s), and the policy is what balances.
RAMP_S = 1.0
RAMP_SKIP = 0.1         # rad, max leg/waist joint error
N_LEG_WAIST = 15


def _session(path: Path):
    try:
        import onnxruntime as ort
    except ImportError as e:   # pragma: no cover - depends on the environment
        raise RuntimeError("the holosoma controller needs onnxruntime: "
                           "uv sync --inexact --extra sim (it is in the `sim` extra)") from e
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1       # a 220k-parameter MLP: threads only add latency jitter
    opts.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), sess_options=opts, providers=["CPUExecutionProvider"])


class HolosomaController(WholeBodyController):
    name = "holosoma"
    rate_hz = RATE_HZ
    supports_walk = True
    supports_posture = False

    def __init__(self, model_path: str | Path | None = None) -> None:
        super().__init__()
        self.model_path = Path(model_path) if model_path else MODEL
        self._sess = _session(self.model_path)
        meta = self._sess.get_modelmeta().custom_metadata_map
        names = json.loads(meta["dof_names"])
        if names != list(BODY_JOINTS):
            raise ValueError(f"{self.model_path.name}: joint order {names} is not the G1 29-DoF order")
        self.kp = np.array(json.loads(meta["kp"]), dtype=float)
        self.kd = np.array(json.loads(meta["kd"]), dtype=float)
        ranges = json.loads(meta.get("command_ranges", "{}"))
        self.velocity_limits = (
            float(max(abs(v) for v in ranges.get("lin_vel_x", [1.0]))),
            float(max(abs(v) for v in ranges.get("lin_vel_y", [1.0]))),
            float(max(abs(v) for v in ranges.get("ang_vel_yaw", [1.0]))),
        )
        self._in = self._sess.get_inputs()[0].name
        self._last_action = np.zeros(29, dtype=np.float32)
        self._phase = np.array([math.pi, math.pi])
        self._standing = True
        self._ramp: Ramp | None = None
        self.last_inference_s = 0.0

    def start_pose(self) -> dict[str, float]:
        return dict(zip(BODY_JOINTS, (float(v) for v in DEFAULT_ANGLES), strict=True))

    def reset(self, s: BodyState) -> None:
        self._last_action[:] = 0.0
        self._phase[:] = math.pi
        self._standing = True
        q0 = np.asarray(s.q[:N_LEG_WAIST], dtype=float).copy()
        self._ramp = Ramp(t0=s.t, q0=q0, duration=RAMP_S)
        self._ramp.done = float(np.max(np.abs(q0 - DEFAULT_ANGLES[:N_LEG_WAIST]))) < RAMP_SKIP

    def _on_mode(self, old: str, new: str) -> None:
        if old == "damp" and new != "damp":
            self._ramp = None            # stand up again from wherever damping left the legs

    # --- policy ---------------------------------------------------------------------
    def _advance_phase(self) -> None:
        vx, vy, wz = self._vel
        self._phase = np.fmod(self._phase + 2 * math.pi / (RATE_HZ * GAIT_PERIOD) + math.pi, 2 * math.pi) - math.pi
        if math.hypot(vx, vy) < STAND_EPS and abs(wz) < STAND_EPS:
            self._phase[:] = math.pi
            self._standing = True
        elif self._standing:
            self._phase = np.array([0.0, math.pi])
            self._standing = False

    def observation(self, s: BodyState) -> np.ndarray:
        """The (1, 100) float32 policy input, in holosoma's sorted-term order."""
        vx, vy, wz = self._vel
        return np.concatenate([
            self._last_action,                                  # actions
            np.asarray(s.base_ang_vel, dtype=float) * ANG_VEL_SCALE,  # base_ang_vel
            [wz],                                               # command_ang_vel
            [vx, vy],                                           # command_lin_vel
            np.cos(self._phase),                                # cos_phase
            np.asarray(s.q, dtype=float) - DEFAULT_ANGLES,      # dof_pos
            np.asarray(s.qd, dtype=float) * DOF_VEL_SCALE,      # dof_vel
            projected_gravity(s.base_quat),                     # projected_gravity
            np.sin(self._phase),                                # sin_phase
        ]).astype(np.float32)[None, :]

    def _step(self, s: BodyState) -> ControlOutput:
        if self._ramp is None:
            self.reset(s)
        goal = DEFAULT_ANGLES[:N_LEG_WAIST]
        if not self._ramp.done:
            q = self._ramp.target(s.t, goal)
            if not self._ramp.done:
                return ControlOutput(joints=JointCommand(joints=self.joints, q=q, kp=HOLD_KP.copy(), kd=HOLD_KD.copy()))
        self._advance_phase()
        t0 = time.perf_counter()
        action = self._sess.run(None, {self._in: self.observation(s)})[0][0]
        self.last_inference_s = time.perf_counter() - t0
        action = np.clip(np.asarray(action, dtype=np.float32), -100.0, 100.0)
        self._last_action = action
        target = DEFAULT_ANGLES + ACTION_SCALE * action.astype(float)
        return ControlOutput(joints=self._cmd(target[:N_LEG_WAIST]))

    def _cmd(self, q: np.ndarray) -> JointCommand:
        return JointCommand(joints=self.joints, q=np.asarray(q, dtype=float),
                            kp=self.kp[:N_LEG_WAIST].copy(), kd=self.kd[:N_LEG_WAIST].copy())
