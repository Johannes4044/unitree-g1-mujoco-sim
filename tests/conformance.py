"""The `RobotSource` conformance suite: one set of tests, every implementation.

`RobotSource` is the single seam between a policy and a robot. `SimSource` implements it on
MuJoCo today; `g1.robot.real_source.RealSource` is the stub that will implement it on DDS,
RealSense and a SLAM node tomorrow. A policy written against the interface only transfers if
both sides answer the *same* contract, so the contract is written once, here, and run against
each implementation.

Running it against a new implementation
---------------------------------------
Subclass `RobotSourceContract`, override the `source` fixture, and that is all:

    # tests/test_real_source_contract.py
    import pytest
    from g1.robot.real_source import RealSource
    from tests.conformance import RobotSourceContract

    class TestRealSource(RobotSourceContract):
        @pytest.fixture(scope="class")
        @classmethod
        def source(cls):
            src = RealSource(iface="eth0")
            yield src
            src.close()

        def reset(self, source):
            # optional: put the robot back in a known state between tests. The default is a
            # no-op, which is only safe for an implementation nothing can disturb.
            source.set_mode("damp")

Nothing else is declarative. The suite asks the implementation what it can do through
`controller_caps()` and probes the rest, so it does not need a per-implementation table of
capabilities that could drift away from the thing it describes. Where sim and robot
legitimately differ, the contract is written as "either A or B, and never something else" -
e.g. `rgb("third_person")` is an image in sim and `None` on the robot, so the contract is
"a uint8 HxWx3 array or None", which is what a caller actually has to handle.

What is deliberately *not* here: anything that needs `step()`, `sim_reset()` or a known world.
Those are simulator tools, not interface, and they live in the other test modules.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from g1.robot.robot_source import (
    BODY_JOINTS,
    CAMERAS,
    DEPTHS,
    HAND_JOINTS,
    HEAD_JOINTS,
    RobotSource,
    RobotState,
)
from g1.wbc import LEG_WAIST_JOINTS

# The 28 joints the 31-slot layout reads (see tests/test_layout.py for the layout itself).
LAYOUT_JOINTS = list(BODY_JOINTS[15:]) + list(HAND_JOINTS) + list(HEAD_JOINTS)
# Commandable by anyone, in every mode, on both sides of the seam: arms, hands, the head module.
COMMANDABLE = LAYOUT_JOINTS


class RobotSourceContract:
    """Mixin of the interface contract. See the module docstring for how to apply it."""

    # --- hooks -------------------------------------------------------------------
    @pytest.fixture(scope="class")
    @classmethod
    def source(cls):
        """Override me. `@classmethod` because pytest deprecates class-scoped fixtures
        defined as instance methods; a function-scoped override may drop it."""
        raise NotImplementedError("override the `source` fixture with your implementation")

    def reset(self, source: RobotSource) -> None:
        """Put the implementation back in a known state. Called before and after each test."""

    @pytest.fixture(autouse=True)
    def _isolate(self, source):
        """Every test starts and leaves the source in the same state as every other test.

        Without this a `set_mode("walk")` in one test is still in force in the next one, and
        the suite's pass/fail depends on collection order.
        """
        self.reset(source)
        yield
        self.reset(source)

    # --- identity ----------------------------------------------------------------
    def test_implements_the_interface(self, source):
        assert isinstance(source, RobotSource)
        assert isinstance(source.name, str) and source.name not in ("", "abstract")

    # --- state -------------------------------------------------------------------
    def test_state_is_a_robot_state(self, source):
        s = source.state()
        assert isinstance(s, RobotState)
        assert isinstance(s.t, float) and math.isfinite(s.t)
        assert isinstance(s.mode, str) and s.mode
        assert isinstance(s.controller, str) and s.controller
        assert isinstance(s.extra, dict)

    def test_state_joint_names_are_known_joints(self, source):
        """A joint the interface reports under a name nobody else knows is data nobody can use."""
        known = set(BODY_JOINTS) | set(HAND_JOINTS) | set(HEAD_JOINTS)
        unknown = sorted(set(source.state().joints) - known)
        assert not unknown, f"state() reports joints that are in no joint list: {unknown}"

    def test_state_reports_every_layout_joint(self, source):
        """The 28 joint slots of the episode layout must all be fillable from one state frame.

        A missing name is recorded as 0.0 by `workspace/06_record_an_episode.py` - a joint at
        the origin, indistinguishable from a real measurement. That is the failure this guards.
        """
        missing = [n for n in LAYOUT_JOINTS if n not in source.state().joints]
        assert not missing, f"state() does not report layout joints {missing}"

    def test_state_joint_values_are_finite_floats(self, source):
        s = source.state()
        for d, what in ((s.joints, "joints"), (s.joint_vel, "joint_vel"), (s.joint_torque, "joint_torque")):
            for n, v in d.items():
                assert isinstance(v, float), f"{what}[{n}] is {type(v).__name__}, not float"
                assert math.isfinite(v), f"{what}[{n}] is {v}"

    def test_state_pose_is_position_and_wxyz_quaternion(self, source):
        s = source.state()
        assert len(s.base_pos) == 3
        assert len(s.base_quat) == 4
        assert all(math.isfinite(v) for v in list(s.base_pos) + list(s.base_quat))
        # wxyz, unit norm. 1e-6 is round-off in a float64 quaternion, not a tolerance on physics.
        assert abs(float(np.linalg.norm(s.base_quat)) - 1.0) < 1e-6

    def test_base_velocity_is_none_or_three_finite_numbers(self, source):
        """`None` means "this source does not estimate it". Zeros would mean "it is standing
        still", which is a measurement and ends up in the training data as one - see the
        comment on `RobotState.base_lin_vel`."""
        s = source.state()
        for v, what in ((s.base_lin_vel, "base_lin_vel"), (s.base_ang_vel, "base_ang_vel")):
            if v is None:
                continue
            assert len(v) == 3, what
            assert all(math.isfinite(float(x)) for x in v), what
        # both or neither: a consumer reads them as one 6-vector
        assert (s.base_lin_vel is None) == (s.base_ang_vel is None)

    def test_joint_temp_is_absent_or_real(self, source):
        """Empty is the honest answer when there is no thermal data. An all-zero dict is not:
        it arms the over-temperature alarm against a fake measurement."""
        temp = source.state().joint_temp
        if not temp:
            return
        assert any(v != 0.0 for v in temp.values()), "joint_temp is filled with zeros"
        assert all(math.isfinite(v) and -50.0 < v < 300.0 for v in temp.values())

    # --- cameras -----------------------------------------------------------------
    def test_rgb_is_uint8_hwc_or_none(self, source):
        for cam in CAMERAS:
            img = source.rgb(cam)
            if img is None:
                continue                       # legitimate: third_person on a real robot
            assert img.dtype == np.uint8, f"{cam} rgb dtype {img.dtype}"
            assert img.ndim == 3 and img.shape[2] == 3, f"{cam} rgb shape {img.shape}"
            assert img.shape[0] > 0 and img.shape[1] > 0

    def test_depth_is_float32_metres_or_none(self, source):
        for cam in CAMERAS:
            d = source.depth(cam)
            if d is None:
                continue
            assert d.dtype == np.float32, f"{cam} depth dtype {d.dtype}"
            assert d.ndim == 2, f"{cam} depth shape {d.shape}"
            assert np.isfinite(d).all(), f"{cam} depth has non-finite values"
            assert (d > 0).all(), f"{cam} depth has non-positive values"

    def test_depth_is_asked_for_by_the_colour_camera_name(self, source):
        """The aligned-depth convention: you ask `depth("d455_rgb")`, not `depth("d455_depth")`."""
        for colour in DEPTHS:
            if source.rgb(colour) is None:
                continue
            assert source.depth(colour) is not None, f"depth({colour!r}) returned None"

    def test_rgb_and_depth_of_an_unknown_camera_are_none(self, source):
        assert source.rgb("no_such_camera") is None
        assert source.depth("no_such_camera") is None

    # --- lidar -------------------------------------------------------------------
    def test_lidar_is_a_cloud_with_a_sensor_pose_or_none(self, source):
        out = source.lidar()
        if out is None:
            return                             # a source with no LiDAR yet
        points, pos, quat = out
        assert points.dtype == np.float32 and points.ndim == 2 and points.shape[1] == 3
        assert len(points) > 0
        assert np.isfinite(points).all()
        assert np.asarray(pos).shape == (3,) and np.isfinite(pos).all()
        assert np.asarray(quat).shape == (4,)
        assert abs(float(np.linalg.norm(quat)) - 1.0) < 1e-5

    # --- modes -------------------------------------------------------------------
    def test_caps_describe_the_controller(self, source):
        caps = source.controller_caps()
        for key in ("name", "modes", "walk", "posture"):
            assert key in caps, f"controller_caps() has no {key!r}"
        assert isinstance(caps["modes"], list)
        assert isinstance(caps["walk"], bool) and isinstance(caps["posture"], bool)
        assert source.state().mode in caps["modes"] or not caps["modes"]

    def test_every_advertised_mode_can_be_entered(self, source):
        for mode in source.controller_caps()["modes"]:
            source.set_mode(mode)
            assert source.state().mode == mode

    def test_an_unadvertised_mode_raises(self, source):
        with pytest.raises((RuntimeError, ValueError)):
            source.set_mode("no_such_mode")

    # --- velocity: the e-stop contract -------------------------------------------
    def test_zero_velocity_is_accepted_in_every_mode(self, source):
        """The e-stop. `zero_velocity` is the dead-man, disarm and stop path, which retries
        every tick and reports a refusal as a failed stop, so it must never raise - in any
        mode, on a controller that cannot walk, after a fault."""
        for mode in source.controller_caps()["modes"]:
            source.set_mode(mode)
            source.set_base_velocity(0.0, 0.0, 0.0)      # must not raise

    def test_nonzero_velocity_is_refused_outside_walk(self, source):
        caps = source.controller_caps()
        for mode in caps["modes"]:
            if mode == "walk":
                continue
            source.set_mode(mode)
            with pytest.raises((RuntimeError, ValueError, NotImplementedError)):
                source.set_base_velocity(0.2, 0.0, 0.0)

    def test_nonzero_velocity_in_walk_matches_the_advertised_capability(self, source):
        caps = source.controller_caps()
        if "walk" not in caps["modes"]:
            assert not caps["walk"], "caps say the controller walks but offer no walk mode"
            return
        source.set_mode("walk")
        if caps["walk"]:
            source.set_base_velocity(0.2, 0.0, 0.0)
        else:
            with pytest.raises((RuntimeError, ValueError, NotImplementedError)):
                source.set_base_velocity(0.2, 0.0, 0.0)

    def test_a_stop_is_always_accepted_after_a_move(self, source):
        caps = source.controller_caps()
        if not (caps["walk"] and "walk" in caps["modes"]):
            pytest.skip("this controller does not walk")
        source.set_mode("walk")
        source.set_base_velocity(0.2, 0.0, 0.1)
        source.set_base_velocity(0.0, 0.0, 0.0)          # must not raise

    # --- posture -----------------------------------------------------------------
    def test_posture_is_gated_by_the_advertised_capability(self, source):
        caps = source.controller_caps()
        if caps["posture"]:
            limits = caps["posture_limits"]
            lo, hi = limits["base_height"]
            applied = source.set_posture((lo + hi) / 2, 0.0)
            assert applied is None or abs(applied["base_height"] - (lo + hi) / 2) < 1e-9
            # out of range is clamped to the advertised limit, not refused and not obeyed
            clamped = source.set_posture(hi + 10.0, 0.0)
            if clamped is not None:
                assert clamped["base_height"] == pytest.approx(hi)
        else:
            with pytest.raises((NotImplementedError, RuntimeError)):
                source.set_posture(0.7, 0.0)

    # --- joint commands ----------------------------------------------------------
    def test_arm_hand_and_head_joints_are_commandable(self, source):
        """Everything in the 31-slot layout must be commandable, in the source's own mode."""
        here = source.state().joints
        for name in COMMANDABLE:
            source.set_joint_targets({name: here[name]})   # commanding "stay here" must be legal

    def test_set_head_is_the_head_joints(self, source):
        source.set_head(pan=0.0, tilt=0.0)
        source.set_head(pan=0.0)
        source.set_head(tilt=0.0)

    def test_controller_owned_joints_are_refused_all_or_nothing(self, source):
        """Legs and waist belong to the whole-body controller. A policy that can nudge one leg
        joint while the controller drives the other eleven is the worst of both worlds, so the
        refusal has to cover the whole set - and a source that refuses none of them has to be
        admitting, through `controller_caps()`, that nothing is driving them.
        """
        here = source.state().joints
        refused = []
        for name in LEG_WAIST_JOINTS:
            if name not in here:
                continue
            try:
                source.set_joint_targets({name: here[name]})
            except (ValueError, NotImplementedError, RuntimeError):
                refused.append(name)
        present = [n for n in LEG_WAIST_JOINTS if n in here]
        assert refused == present or refused == [], (
            f"partial refusal: {len(refused)} of {len(present)} leg/waist joints refused")
        if not refused:
            caps = source.controller_caps()
            assert caps["stand_in"] or caps["name"] == "none", (
                f"{caps['name']} owns the legs but set_joint_targets accepted them")

    def test_commanding_a_controller_owned_joint_raises_rather_than_no_ops(self, source):
        """If the legs are owned, the refusal must be an exception. A silently dropped command
        is a policy that believes it moved a leg."""
        here = source.state().joints
        owned = [n for n in LEG_WAIST_JOINTS if n in here]
        if not owned or source.controller_caps()["stand_in"]:
            pytest.skip("no controller owns the legs in this configuration")
        with pytest.raises((ValueError, NotImplementedError, RuntimeError)):
            source.set_joint_targets({owned[0]: here[owned[0]]})

    def test_a_mixed_command_is_refused_whole(self, source):
        """An arm target bundled with a leg target must not be half-applied."""
        here = source.state().joints
        owned = [n for n in LEG_WAIST_JOINTS if n in here]
        if not owned or source.controller_caps()["stand_in"]:
            pytest.skip("no controller owns the legs in this configuration")
        arm = BODY_JOINTS[15]
        with pytest.raises((ValueError, NotImplementedError, RuntimeError)):
            source.set_joint_targets({arm: here[arm], owned[0]: here[owned[0]]})
