"""Safety refusals, and the stop that must always work.

These are not generic interface tests (those live in `tests/conformance.py`); they pin the
specific, measured behaviours of `SimSource` plus `g1.wbc` that a policy author can break
without noticing:

* a joint the whole-body controller owns is not commandable,
* a non-zero base velocity outside walk mode is refused,
* a zero base velocity is accepted in every mode, by every controller, always,
* the velocity command expires after `DEADMAN_S` without a refresh,
* a controller that faults drops to damping rather than holding its last torque.

The asymmetry between controllers is deliberate and documented, not a bug: see
`test_kinematic_stand_in_does_not_own_the_legs`.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from g1.wbc import ARM_JOINTS, LEG_WAIST_JOINTS, ControllerFault

FREE_BASE_CONTROLLERS = ("pd_stand", "holosoma")


# --- who owns the legs -------------------------------------------------------------------
@pytest.mark.parametrize("controller", FREE_BASE_CONTROLLERS)
def test_leg_and_waist_joints_are_refused_while_a_controller_owns_them(make_sim, controller):
    sim = make_sim(controller=controller)
    here = sim.state().joints
    for name in LEG_WAIST_JOINTS:
        with pytest.raises(ValueError, match="not commandable"):
            sim.set_joint_targets({name: here[name]})


@pytest.mark.parametrize("controller", FREE_BASE_CONTROLLERS)
def test_a_refused_command_moves_nothing(make_sim, controller):
    """The refusal happens before anything is written, so a mixed arm+leg command is not
    half-applied. A partially applied action is worse than a rejected one: the robot ends up
    in a pose the policy never asked for."""
    sim = make_sim(controller=controller)
    here = sim.state().joints
    before = sim.d.ctrl.copy()
    with pytest.raises(ValueError):
        sim.set_joint_targets({ARM_JOINTS[0]: here[ARM_JOINTS[0]] + 0.5,
                               LEG_WAIST_JOINTS[3]: here[LEG_WAIST_JOINTS[3]] + 0.5})
    assert np.array_equal(sim.d.ctrl, before)


def test_kinematic_stand_in_does_not_own_the_legs(make_sim):
    """Documented asymmetry, pinned so it cannot change silently.

    `workspace/README.md` says "`set_joint_targets` refuses them while a controller owns
    them", and the default `kinematic` stand-in owns nothing: `KinematicController.joints`
    is `()`, the leg position servos stay in the model, and a leg target is *accepted*. The
    legs cannot carry the robot anywhere (the pelvis is welded to a mocap body), but a leg
    command is not an error either.

    So "legs are never commandable" is false for the default configuration. Anyone relying
    on the refusal as a safety property must assert `controller_caps()["stand_in"] is False`
    first - which is exactly what the conformance suite checks.
    """
    sim = make_sim()                                     # kinematic
    caps = sim.controller_caps()
    assert caps["stand_in"] is True
    here = sim.state().joints
    sim.set_joint_targets({"left_knee_joint": here["left_knee_joint"]})     # accepted
    assert sim.controller_caps()["name"] == "kinematic"


# --- velocity gating ---------------------------------------------------------------------
ALL_CONTROLLERS = ("kinematic", "pd_stand", "holosoma")


@pytest.mark.parametrize("controller", ALL_CONTROLLERS)
def test_zero_velocity_is_accepted_in_every_mode(make_sim, controller):
    """`cheatsheet.md`: "Zero velocity is accepted in every mode by every controller, always.
    That is what makes it a usable e-stop." A refused stop is reported upstream as a failed
    stop and retried every tick, so this one must hold unconditionally."""
    sim = make_sim(controller=controller)
    for mode in sim.controller_caps()["modes"]:
        sim.set_mode(mode)
        sim.set_base_velocity(0.0, 0.0, 0.0)
        sim.set_base_velocity(-0.0, 0.0, 0.0)            # negative zero is still zero


@pytest.mark.parametrize("controller", ALL_CONTROLLERS)
def test_nonzero_velocity_outside_walk_is_refused(make_sim, controller):
    sim = make_sim(controller=controller)
    for mode in sim.controller_caps()["modes"]:
        if mode == "walk":
            continue
        sim.set_mode(mode)
        for cmd in ((0.2, 0.0, 0.0), (0.0, 0.2, 0.0), (0.0, 0.0, 0.2)):
            with pytest.raises(RuntimeError, match="walk"):
                sim.set_base_velocity(*cmd)


def test_a_controller_that_cannot_walk_has_no_walk_mode(make_sim):
    sim = make_sim(controller="pd_stand")
    caps = sim.controller_caps()
    assert caps["walk"] is False
    assert "walk" not in caps["modes"]
    with pytest.raises(RuntimeError, match="no mode 'walk'"):
        sim.set_mode("walk")


def test_the_stop_still_works_after_a_fault(make_sim):
    """A faulted controller is damping. Zero velocity must still be accepted there - it is the
    path the dead-man takes while the operator is deciding what to do."""
    sim = make_sim(controller="holosoma")
    sim._trip("injected by the test suite")              # the fault path, without a fake policy
    assert sim.state().mode == "damp"
    sim.set_base_velocity(0.0, 0.0, 0.0)
    with pytest.raises(RuntimeError, match="walk"):
        sim.set_base_velocity(0.2, 0.0, 0.0)


def test_posture_is_refused_on_a_faulted_controller(make_sim):
    sim = make_sim(controller="pd_stand")
    sim._trip("injected by the test suite")
    with pytest.raises(RuntimeError, match="faulted"):
        sim.set_posture(0.7, 0.0)
    sim.set_mode("stand")                                # re-engaging clears the fault
    assert sim.set_posture(0.7, 0.0)["base_height"] == pytest.approx(0.7)


def test_a_non_finite_velocity_is_refused(make_sim):
    sim = make_sim(controller="holosoma")
    sim.set_mode("walk")
    for bad in (math.nan, math.inf):
        with pytest.raises(ValueError, match="finite"):
            sim.set_base_velocity(bad, 0.0, 0.0)


# --- the dead-man ------------------------------------------------------------------------
def test_the_velocity_command_expires_without_a_refresh(make_sim):
    """`SimSource.DEADMAN_S` is 0.3 s of *sim* time, like the robot's own velocity interface.
    A policy that forgets to re-send inside its control loop gets a robot that keeps stopping,
    not one that keeps going."""
    sim = make_sim()
    assert sim.DEADMAN_S == pytest.approx(0.3)
    sim.set_mode("walk")
    sim.set_base_velocity(0.5, 0.0, 0.0)
    sim.step(0.2)                                        # inside the dead-man: still moving
    x_live = sim.state().base_pos[0]
    assert x_live > 0.05
    sim.step(0.4)                                        # past it: stopped
    x_expired = sim.state().base_pos[0]
    sim.step(0.4)
    x_later = sim.state().base_pos[0]
    # 0.1 mm, not exact zero: `base_pos` is the pelvis, which the weld constraint holds to the
    # mocap target with a solver residual of a few microns. An unexpired 0.5 m/s command would
    # have moved it 200 mm over this interval, so 0.1 mm is three orders of margin either way.
    assert x_later == pytest.approx(x_expired, abs=1e-4), "the expired command kept integrating"


def test_set_mode_clears_a_live_velocity_command(make_sim):
    sim = make_sim()
    sim.set_mode("walk")
    sim.set_base_velocity(0.5, 0.0, 0.0)
    sim.set_mode("walk")                                 # same mode, but still a mode change
    x0 = sim.state().base_pos[0]
    sim.step(0.2)
    assert sim.state().base_pos[0] == pytest.approx(x0, abs=1e-4)   # weld residual, see above


def test_sim_reset_clears_a_live_velocity_command(make_sim):
    sim = make_sim()
    sim.set_mode("walk")
    sim.set_base_velocity(0.5, 0.0, 0.0)
    sim.sim_reset()
    sim.step(0.2)
    assert sim.state().base_pos[:2] == pytest.approx([0.0, 0.0], abs=1e-4)  # weld residual


# --- the fault path ----------------------------------------------------------------------
def test_a_faulting_controller_drops_to_damping_in_the_same_step(make_sim, monkeypatch):
    """A controller that raises, or returns a non-finite torque, must not leave its last
    command on the motors. The fault is reported in `state().extra["controller_fault"]`."""
    sim = make_sim(controller="pd_stand")
    sim.set_mode("stand")

    def explode(_state):
        raise ControllerFault("injected")

    monkeypatch.setattr(sim._wbc, "_step", explode)
    sim.step(0.05)
    state = sim.state()
    assert state.mode == "damp"
    fault = state.extra["controller_fault"]
    assert fault is not None and "injected" in fault["error"]
    assert fault["controller"] == "pd_stand"
    # damping is kd-only: zero stiffness, so no torque is holding the old pose
    assert np.all(sim._cmd.kp == 0.0)
    assert np.all(sim._cmd.kd > 0.0)


def test_a_reset_clears_the_fault(make_sim):
    sim = make_sim(controller="pd_stand")
    sim._trip("injected by the test suite")
    sim.sim_reset()
    assert sim.state().extra["controller_fault"] is None
    assert sim.state().mode == "stand"


# --- teleport refusals -------------------------------------------------------------------
def test_teleport_refuses_a_target_inside_a_table(make_sim):
    sim = make_sim()
    with pytest.raises(RuntimeError, match="inside the footprint"):
        sim.sim_teleport(0.55, 0.0, 0.0)


def test_teleport_accepts_a_clear_target(make_sim):
    sim = make_sim()
    out = sim.sim_teleport(-1.0, 0.5, 0.3)
    assert out["x"] == pytest.approx(-1.0)
    assert out["y"] == pytest.approx(0.5)
    assert out["yaw"] == pytest.approx(0.3)
    assert sim.state().base_pos[:2] == pytest.approx([-1.0, 0.5], abs=1e-4)


def test_randomise_rejects_an_unknown_object(make_sim):
    sim = make_sim()
    with pytest.raises(RuntimeError, match="unknown object"):
        sim.sim_randomise(seed=1, objects=["no_such_object"])


def test_randomise_rejects_an_unknown_scope(make_sim):
    sim = make_sim()
    with pytest.raises(RuntimeError, match="unknown scope"):
        sim.sim_randomise(seed=1, scope="everywhere")


def test_an_unknown_controller_name_raises(make_sim):
    with pytest.raises(ValueError, match="unknown controller"):
        make_sim(controller="no_such_controller")
