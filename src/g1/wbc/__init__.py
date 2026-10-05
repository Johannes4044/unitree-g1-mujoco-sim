"""Whole-body controllers: the layer that keeps the G1 upright under velocity and posture commands.

    from g1.wbc import make_controller
    ctrl = make_controller("holosoma")      # or "pd_stand", "kinematic"

| name       | walks | posture | free base | what it is                                             |
|------------|-------|---------|-----------|--------------------------------------------------------|
| kinematic  | yes*  | no      | no        | sim stand-in: the pelvis is moved, the legs never are  |
| holosoma   | yes   | no      | yes       | Amazon FAR's pretrained G1 locomotion policy (ONNX)    |
| pd_stand   | no    | yes     | yes       | stiff PD around an analytic squat / lean pose          |

Plug-in point for another controller (AMO, a GR00T-WBC variant SAP legal accepts, a policy
trained in unitree_rl_mjlab): subclass `WholeBodyController`, implement `reset` and `_step`
(return a `JointCommand` for the leg and waist joints, PD targets plus gains), set the class
flags (`supports_walk`, `supports_posture`, `posture_limits`, `rate_hz`) and register it in
`CONTROLLERS` below. The simulator, the bridge's `set_posture` gate and the UI all key off those
flags, so nothing else changes.
"""
from __future__ import annotations

from g1.wbc.base import (
    ARM_JOINTS,
    DAMP_KD,
    LEG_WAIST_JOINTS,
    MODES,
    BodyState,
    ControllerFault,
    ControlOutput,
    JointCommand,
    WholeBodyController,
    damping_command,
)


def _kinematic():
    from g1.wbc.kinematic import KinematicController
    return KinematicController()


def _holosoma():
    from g1.wbc.holosoma import HolosomaController
    return HolosomaController()


def _pd_stand():
    from g1.wbc.pd_stand import PDStandController
    return PDStandController()


CONTROLLERS = {"kinematic": _kinematic, "holosoma": _holosoma, "pd_stand": _pd_stand}


def make_controller(name: str) -> WholeBodyController:
    try:
        factory = CONTROLLERS[name]
    except KeyError:
        raise ValueError(f"unknown controller {name!r}; available: {sorted(CONTROLLERS)}") from None
    return factory()


def holosoma_available() -> bool:
    """True when onnxruntime is importable (the model itself is committed)."""
    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        return False
    return True


__all__ = ["ARM_JOINTS", "CONTROLLERS", "DAMP_KD", "LEG_WAIST_JOINTS", "MODES", "BodyState", "ControlOutput",
           "ControllerFault", "JointCommand", "WholeBodyController", "damping_command", "holosoma_available",
           "make_controller"]
