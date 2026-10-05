"""The kinematic stand-in: no controller at all, the simulator moves the pelvis directly.

The pelvis is welded to a mocap body (scene built with FIXED_BASE=1) and the simulator integrates
the commanded body-frame velocity into that body's pose. The legs never move and the robot cannot
fall. It exists so navigation can be exercised without a walking controller, and the navigator's
regression tests are written against its exact kinematics - which is why it stays selectable.
The UI flags it as a "sim stand-in", because nothing about it transfers to the real robot.
"""
from __future__ import annotations

import math

from g1.wbc.base import BodyState, ControlOutput, WholeBodyController


class KinematicController(WholeBodyController):
    name = "kinematic"
    rate_hz = None                       # every physics step, like the integration it replaces
    needs_free_base = False
    needs_state = False                  # it only integrates the command: no snapshot per step
    stand_in = True
    supports_walk = True
    modes = ("sim", "damp", "stand", "walk")
    initial_mode = "sim"
    joints: tuple[str, ...] = ()         # the legs keep their position servos; nothing to drive
    # no clipping: the stand-in integrates exactly what it is given (the SafetyGate clips upstream)
    velocity_limits = (math.inf, math.inf, math.inf)

    def reset(self, s: BodyState) -> None:
        self._vel[:] = 0.0

    def _step(self, s: BodyState) -> ControlOutput:
        if self._mode != "walk":
            return ControlOutput()
        vx, vy, wz = (float(v) for v in self._vel)
        return ControlOutput(base_twist=(vx, vy, wz))
