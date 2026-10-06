"""Run the `RobotSource` conformance suite against `SimSource`.

Three times, once per whole-body controller, because the controller changes what the
interface is allowed to do: `kinematic` welds the pelvis and owns no joints, `pd_stand` and
`holosoma` take a free base and own the twelve leg and three waist joints. All three have to
answer the same contract.

`tests/conformance.py` says how to point the identical suite at another implementation -
that is the whole reason it is a mixin and not a pile of `SimSource` tests.
"""
from __future__ import annotations

import pytest

from tests.conformance import RobotSourceContract


class _SimContract(RobotSourceContract):
    """Shared plumbing: one simulator per controller, explicitly reset between tests.

    Class scope, not function scope, because compiling the model and hashing it costs ~0.15 s
    and the contract has thirty-odd tests. That is only safe with the reset below.
    """

    controller: str = "kinematic"

    @pytest.fixture(scope="class")
    @classmethod
    def source(cls, request):
        from g1.robot.sim_source import SimSource

        src = SimSource(xml=request.getfixturevalue("default_model"), realtime=False,
                        controller=cls.controller, cam_w=64, cam_h=48)
        request.cls.initial_mode = src.state().mode
        yield src
        src.close()

    def reset(self, source):
        # Order matters: set_mode clears the velocity command and re-engages after a fault,
        # sim_reset puts the keyframe, the objects and the clock back.
        source.set_mode(self.initial_mode)
        source.sim_reset()


class TestSimSourceKinematic(_SimContract):
    controller = "kinematic"


class TestSimSourcePDStand(_SimContract):
    controller = "pd_stand"


class TestSimSourceHolosoma(_SimContract):
    controller = "holosoma"
