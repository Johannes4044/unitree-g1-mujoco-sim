"""MuJoCo implementation of RobotSource (runs physics in a background thread).

Base controller (see g1.wbc): chosen by `SimSource(controller=...)`, else $G1_SIM_CONTROLLER,
else "kinematic" when the scene has the mocap base and a real controller when it does not.

* "kinematic" - the stand-in. The scene is built with FIXED_BASE=1, so the pelvis is welded to a
  mocap body 'base_mocap'; in mode "walk", set_base_velocity(vx, vy, wz) moves that mocap body
  kinematically (body-frame velocity integrated in yaw-only 2D). The legs never move and the
  robot cannot fall. The navigator's regression tests rely on these exact kinematics.
* a whole-body controller ("holosoma", "pd_stand") - a free floating base. The `fixed_base` weld
  is switched off at load time (FIXED_BASE=0 without rebuilding the scene), the leg and waist
  actuators become torque motors (still clipped to each joint's effort limit), and every physics
  step applies the controller's PD targets: the controller itself runs at its own rate (50 Hz
  for Holosoma), the PD loop at the physics rate (500 Hz), as on the real robot's motor drivers.
  A controller that raises or produces a non-finite command drops the robot to damping in the
  same step - it never keeps the last torque applied - and the fault is reported in
  `state().extra["controller_fault"]`, from which the bridge raises an error event.

Velocity commands expire after DEADMAN_S seconds without a refresh, like the real robot's
velocity interface, and are cleared on every set_mode() and reset(). The only simulation clock
is `self.d.time` (advanced by mj_step, restored by the keyframe on reset).

Threading contract
------------------
* `self.lock` guards the live physics state `self.d`. The physics loop (realtime=True)
  and step() hold it only for the duration of mj_step; state() and commands hold it
  briefly. Nothing renders while holding it.
* Rendering never touches `self.d`. rgb()/depth()/lidar() copy the live state into a
  dedicated snapshot `self._snap` (mj_copyData under `self.lock`) and render from the
  copy, so a multi-camera LiDAR scan cannot stall physics.
* `mujoco.Renderer` owns an EGL/OSMesa context bound to the thread that created it, so
  renderers (and the LidarSim, which owns one) are thread-local: every calling thread
  lazily creates its own and only ever uses that one. close() releases the calling
  thread's renderers; other threads' contexts are released when their thread exits.
* `self.gl_lock` serialises rendering across threads. It is kept because the snapshot
  `self._snap` has a single owner (it is only copied and read under gl_lock) and because
  parallel software-GL contexts gave no throughput gain in the container, not because
  the GL backend is non-re-entrant.

Measured render costs (llvmpipe software GL in the sim container, 424x240 unless noted),
so callers can pick sensible rates: rgb ~15 ms, third_person rgb ~43 ms, depth ~7 ms,
lidar (4 x 192x192 depth + unprojection) ~59 ms.
"""
from __future__ import annotations

import hashlib
import logging
import math
import os
import re
import threading
import time

import mujoco
import numpy as np

from g1.robot.lidar import LidarSim
from g1.robot.robot_source import (
    BODY_JOINTS,
    CAMERAS,
    DEPTHS,
    HAND_JOINTS,
    HEAD_JOINTS,
    RobotSource,
    RobotState,
)
from g1.wbc import (
    ARM_JOINTS,
    BodyState,
    ControllerFault,
    damping_command,
    holosoma_available,
    make_controller,
)

log = logging.getLogger("g1.sim")
CONTROLLER_ENV = "G1_SIM_CONTROLLER"

# Model path resolution lives in default_model_path() below and nowhere else: callers (the CLI,
# the tests, the sim scripts) pass whatever they have - including the empty string `g1 bridge`
# uses for "not given" - and let it resolve here.
# There is no Docker in this repository: the model is found relative to the checkout, so every
# script runs natively. src/g1/robot/sim_source.py -> src/g1/robot -> src/g1 -> src -> repo root.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
REPO_MODEL = os.path.join(REPO_ROOT, "models", "g1_pick_place.xml")
# ... and, for a non-editable install (where __file__ is in site-packages), relative to the cwd.
CWD_MODEL = os.path.join("models", "g1_pick_place.xml")

# The randomisation surface. There is no fallback: a scene without this geom simply has no pick
# surface, and sim_randomise says so instead of scattering objects over a guessed table.
PICK_TABLE_GEOM = "pick_table_top"
RANDOMISE_SPACING = 0.08   # min distance between object centres [m]
RANDOMISE_MARGIN = 0.05    # min distance from the table edge [m] (grows with the object's radius)
PLACE_CLEARANCE = 0.03     # min gap between a placed object and anything already in that space [m]
PLACE_ATTEMPTS = 50        # whole-layout retries before sim_randomise gives up
SURFACE_TOL = 0.02         # an object counts as "on the pick surface" this far below the top [m]
SURFACE_MARGIN = 0.25      # ... and this far outside the surface footprint (overhang) [m]
# Default amounts for the optional randomisation axes, all relative and deliberately small: the
# point is a policy that survives a slightly wrong world model, not a different world.
COLOUR_JITTER = 0.15       # +/- per RGB channel (alpha untouched)
MASS_JITTER = 0.10         # +/- fraction of the object's mass (inertia scales with it)
FRICTION_JITTER = 0.20     # +/- fraction of the object's friction coefficients


_SEEDED = re.compile(r"^(?P<base>.+)_s(?P<seed>\d+)$")


def scenario_of(path: str) -> dict:
    """Scenario name of a model file, following scripts/build_scenarios.py's output naming:
    `<name>.xml`, or `<name>_s<seed>.xml` for a seeded variant. Any other file (the bare robot
    model g1_pick_place.xml, a hand-written scene) is named by its file stem, with no seed."""
    stem = os.path.splitext(os.path.basename(path))[0]
    m = _SEEDED.match(stem)
    return {"scenario": stem, "scenario_base": m["base"] if m else stem,
            "scenario_seed": int(m["seed"]) if m else None}


def _model_sha256(m) -> str:
    """SHA-256 of the compiled model (MuJoCo's binary serialisation). Byte-identical across
    compiles of the same files, and different as soon as any geometry, mass, joint, contact or
    option differs - so two runs can be shown to have used the same world even if the XML was
    rebuilt in between, and a rebuilt-but-changed world cannot pass as the old one. ~65 ms."""
    buf = np.zeros(mujoco.mj_sizeModel(m), dtype=np.uint8)
    mujoco.mj_saveModel(m, None, buf)
    return hashlib.sha256(buf.tobytes()).hexdigest()


def _yaw_of(q_wxyz) -> float:
    w, x, y, z = q_wxyz
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def default_model_path() -> str:
    """The model to load when the caller did not name one: $SIM_MODEL, the model in this
    checkout, or `models/g1_pick_place.xml` under the current directory. Returns the checkout
    path when nothing exists, so the error message names a path rather than an empty string."""
    env = os.environ.get("SIM_MODEL")
    if env:
        return env
    for p in (REPO_MODEL, CWD_MODEL):
        if os.path.exists(p):
            return os.path.abspath(p)
    return REPO_MODEL


def _compile(path: str):
    """Compile the model, repairing keyframes that are shorter than nq.

    MuJoCo pads a keyframe whose `qpos` has fewer than nq values with *zeros* (identity for the
    quaternion part), not with `qpos0`. The robot model's `stand` keyframe covers exactly the
    joints that existed when it was written, so any body added afterwards by a plain
    `<include file="g1_pick_place.xml"/>` is teleported to the world origin the moment
    `mj_resetDataKeyframe` runs - which `__init__` and every `sim_reset` do. It is silent: the
    model compiles, the objects are simply somewhere else.

    Compiling through `MjSpec` fixes it, because the spec still knows how many values each
    keyframe actually declared: everything past that point is re-padded from `qpos0`, i.e. from
    where the XML put the body. Included scenes therefore work without a compose step; the
    generated scenarios in models/scenarios/ declare a full-length keyframe and are
    unaffected. Costs one compile, same as `mj_MjModel.from_xml_path` (measured: no slower).
    """
    spec = mujoco.MjSpec.from_file(path)
    declared = {k.name: len(k.qpos) for k in spec.keys}
    m = spec.compile()
    for i in range(m.nkey):
        n = declared.get(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_KEY, i) or "", m.nq)
        if n < m.nq:
            m.key_qpos[i, n:] = m.qpos0[n:]
    return m


class SimSource(RobotSource):
    name = "sim"
    DEADMAN_S = 0.3     # velocity command expires after this long without a refresh [sim seconds]

    def __init__(self, xml=None, cam_w=424, cam_h=240, realtime=True, controller=None):
        self.model_path = xml or default_model_path()
        self.m = _compile(self.model_path)
        # Which world this is, reported in every state frame as extra["sim_model"]. Computed once,
        # here, before anything below changes the model at runtime (the controller's torque motors,
        # the weld switched off - those are identified by controller_caps instead). sim_randomise
        # changes colours / masses / frictions later; its seed is what identifies those.
        self.model_info = {"path": os.path.abspath(self.model_path), **scenario_of(self.model_path),
                           "sha256": _model_sha256(self.m)}
        self.d = mujoco.MjData(self.m)
        self._key_stand = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_KEY, "stand")
        mujoco.mj_resetDataKeyframe(self.m, self.d, self._key_stand)
        mujoco.mj_forward(self.m, self.d)   # populate body poses (xpos/xquat) before the first state()
        self.lock = threading.Lock()        # physics state (self.d)
        self.gl_lock = threading.Lock()     # rendering + ownership of the render snapshot
        self._snap = mujoco.MjData(self.m)  # copy of self.d that rendering reads (see module docstring)
        self._tls = threading.local()       # per-thread renderers / LidarSim
        self.cam_w, self.cam_h = cam_w, cam_h
        self.act = {mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_ACTUATOR, i): i for i in range(self.m.nu)}
        self.jnt = {mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_JOINT, j): j for j in range(self.m.njnt)}
        # ids used by state(), resolved once
        self._pelvis = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        self._imu_site = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_SITE, "imu_in_pelvis")
        self._state_joints = [(n, self.jnt[n]) for n in BODY_JOINTS + HAND_JOINTS + HEAD_JOINTS if n in self.jnt]
        # scene objects: every free-jointed body except the pelvis; initial qpos slice from the keyframe
        self._objects = {}          # name -> body id
        self._obj_adr = {}          # name -> (qpos address, dof address)
        for j in range(self.m.njnt):
            if self.m.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE:
                continue
            b = int(self.m.jnt_bodyid[j])
            nm = mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_BODY, b)
            if nm == "pelvis":
                continue
            self._objects[nm] = b
            self._obj_adr[nm] = (int(self.m.jnt_qposadr[j]), int(self.m.jnt_dofadr[j]))
        self._obj_init = {nm: self.d.qpos[a:a + 7].copy() for nm, (a, _) in self._obj_adr.items()}
        # geoms and bodies of each object's subtree, plus the model values sim_randomise may
        # perturb, so sim_reset can put the appearance and the dynamics back exactly.
        self._obj_geoms = {nm: self._subtree_geoms(b) for nm, b in self._objects.items()}
        self._obj_bodies = {nm: self._subtree_bodies(b) for nm, b in self._objects.items()}
        # horizontal half-extent of each object about its own origin, measured upright at the
        # keyframe pose. Yaw does not change it, so it is what keeps a placed object on the table
        # instead of hanging half off the edge. Conservative (geom_rbound is a bounding sphere).
        self._obj_radius = {
            nm: max((float(np.linalg.norm(self.d.geom_xpos[g][:2] - self.d.xpos[b][:2])
                           + self.m.geom_rbound[g]) for g in self._obj_geoms[nm]), default=0.0)
            for nm, b in self._objects.items()}
        self._obj_height = {
            nm: max((float(self.d.geom_xpos[g][2] + self.m.geom_rbound[g] - self.d.xpos[b][2])
                     for g in self._obj_geoms[nm]), default=0.0)
            for nm, b in self._objects.items()}
        self._model0 = {
            "geom_rgba": self.m.geom_rgba.copy(), "geom_matid": self.m.geom_matid.copy(),
            "geom_friction": self.m.geom_friction.copy(), "body_mass": self.m.body_mass.copy(),
            "body_inertia": self.m.body_inertia.copy(),
        }
        self._model_dirty = False       # True once a randomisation has touched self._model0's fields
        self._table = self._find_pick_table()
        self._tables = self._table_footprints()
        # pelvis free joint (qpos/dof address) so teleport can move the pelvis together with the mocap
        self._pelvis_qadr = self._pelvis_dadr = -1
        for j in range(self.m.njnt):
            if self.m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE and int(self.m.jnt_bodyid[j]) == self._pelvis:
                self._pelvis_qadr, self._pelvis_dadr = int(self.m.jnt_qposadr[j]), int(self.m.jnt_dofadr[j])
        # kinematic base (mocap body welded to the pelvis; absent when built with FIXED_BASE=0)
        b = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "base_mocap")
        self._mocap_id = int(self.m.body_mocapid[b]) if b >= 0 else -1
        self._vel_t = -math.inf          # sim time (self.d.time) of the last velocity command
        self._init_controller(controller)
        self.t0 = time.time()
        self.realtime = realtime
        self.running = True
        self.thread = None
        if realtime:
            self.thread = threading.Thread(target=self._loop, daemon=True)
            self.thread.start()

    # --- controller setup -----------------------------------------------------
    def _choose_controller(self, explicit):
        name = explicit or os.environ.get(CONTROLLER_ENV, "").strip()
        if name:
            return name
        if self._mocap_id >= 0:
            return "kinematic"
        # a scene built with FIXED_BASE=0 has nothing to hold the robot up but a controller
        return "holosoma" if holosoma_available() else "pd_stand"

    def _init_controller(self, controller):
        self._wbc = make_controller(self._choose_controller(controller))
        self.free_base = bool(self._wbc.needs_free_base)
        self._weld = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_EQUALITY, "fixed_base")
        if self.free_base:
            if self._pelvis_qadr < 0:
                raise RuntimeError(f"the {self._wbc.name} controller needs a free-jointed pelvis")
            if self._weld >= 0:
                # FIXED_BASE=0 at runtime: the weld stays in the model, switched off (eq_active0 is
                # what mj_resetDataKeyframe restores, so a sim_reset keeps it off)
                self.m.eq_active0[self._weld] = 0
                self.d.eq_active[self._weld] = 0
        missing = [n for n in BODY_JOINTS if n not in self.jnt]
        if missing and self.free_base:
            raise RuntimeError(f"model lacks body joints {missing}")
        self._body_q = np.array([self.m.jnt_qposadr[self.jnt[n]] for n in BODY_JOINTS if n in self.jnt])
        self._body_v = np.array([self.m.jnt_dofadr[self.jnt[n]] for n in BODY_JOINTS if n in self.jnt])
        own = self._wbc.joints
        self._own = frozenset(own)
        self._own_q = np.array([self.m.jnt_qposadr[self.jnt[n]] for n in own], dtype=int)
        self._own_v = np.array([self.m.jnt_dofadr[self.jnt[n]] for n in own], dtype=int)
        self._own_act = np.array([self.act[n] for n in own], dtype=int)
        for i in self._own_act:
            # position servo -> torque motor: ctrl is the torque, which MuJoCo still clips to the
            # joint's actuatorfrcrange (the G1 motor's effort limit)
            self.m.actuator_gainprm[i, :] = 0.0
            self.m.actuator_gainprm[i, 0] = 1.0
            self.m.actuator_biasprm[i, :] = 0.0
            self.m.actuator_ctrllimited[i] = 0
        rate = self._wbc.rate_hz
        self._decim = 1 if rate is None else max(1, round(1.0 / (rate * self.m.opt.timestep)))
        self._tick = 0
        self._cmd = None                # JointCommand being tracked by the per-step PD
        self._twist = None              # kinematic stand-in: body-frame velocity to integrate
        self._faulted = False
        self._fault = None              # last fault {"seq", "t", "controller", "error"}; cleared by reset
        self._fault_seq = 0
        self._mode = self._wbc.initial_mode
        if self.free_base:
            self._apply_start_pose()
        self._wbc.reset(self._body_state())

    def _sole_z(self):
        """Lowest point of the robot's foot collision geoms [m, world z]; needs a current mj_forward."""
        z = math.inf
        for side in ("left", "right"):
            b = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, f"{side}_ankle_roll_link")
            for g in range(self.m.ngeom):
                if int(self.m.geom_bodyid[g]) != b or not (self.m.geom_contype[g] or self.m.geom_conaffinity[g]):
                    continue
                rot = self.d.geom_xmat[g].reshape(3, 3)
                c = self.d.geom_xpos[g] + rot @ self.m.geom_aabb[g, :3]
                z = min(z, float(c[2] - (np.abs(rot) @ self.m.geom_aabb[g, 3:])[2]))
        return z

    def _apply_start_pose(self):
        """Put the robot at rest in the controller's start pose with its soles where the keyframe
        had them (so an uneven scenario floor is respected). Arm joints in the pose become the
        arms' initial targets; the operator may command them from there. Caller holds the lock
        or is __init__."""
        mujoco.mj_forward(self.m, self.d)
        sole0 = self._sole_z()
        for n, v in self._wbc.start_pose().items():
            j = self.jnt.get(n)
            if j is None:
                continue
            self.d.qpos[self.m.jnt_qposadr[j]] = v
            i = self.act.get(n)
            if i is not None and n not in self._own:
                self.d.ctrl[i] = v
        self.d.qvel[self._body_v] = 0.0
        self.d.qvel[self._pelvis_dadr:self._pelvis_dadr + 6] = 0.0
        mujoco.mj_forward(self.m, self.d)
        self.d.qpos[self._pelvis_qadr + 2] += sole0 - self._sole_z()
        mujoco.mj_forward(self.m, self.d)
        hit = self._arm_contacts()
        arms = {n: v for n, v in self._wbc.start_pose().items() if n in ARM_JOINTS and n in self.jnt}
        if hit and arms:
            # Never start inside the scenery: at the default spawn point the nominal arm pose (elbows
            # forward) puts the thumbs into the pick table, which shoves the robot back as it starts.
            # Use the largest fraction of the nominal pose, measured from arms hanging straight down,
            # that is clear. Not the keyframe arms: their elbows are raised 1.28 rad, a pose Holosoma
            # was never trained with, and it answers with a deep crouch (pelvis 0.70 m instead of 0.77).
            for frac in (0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0):
                for n, v in arms.items():
                    self.d.qpos[self.m.jnt_qposadr[self.jnt[n]]] = frac * v
                    i = self.act.get(n)
                    if i is not None and n not in self._own:
                        self.d.ctrl[i] = frac * v
                mujoco.mj_forward(self.m, self.d)
                if not self._arm_contacts():
                    break
            log.warning("start pose: nominal arms would touch %s here; arms at %.0f%% of the nominal pose",
                        sorted(hit), frac * 100)

    def _arm_contacts(self):
        """Names of non-robot bodies an arm (shoulder outwards, hands included) is in contact with."""
        m, d = self.m, self.d
        if getattr(self, "_arm_bodies", None) is None:
            roots = {int(m.jnt_bodyid[self.jnt[n]]) for n in ("left_shoulder_pitch_joint", "right_shoulder_pitch_joint")
                     if n in self.jnt}
            arm = set()
            for b in range(m.nbody):
                a = b
                while a > 0 and a not in roots:
                    a = int(m.body_parentid[a])
                if a in roots:
                    arm.add(b)
            self._arm_bodies = arm
        robot_root = int(m.body_rootid[self._pelvis])
        hit = set()
        for k in range(d.ncon):
            b1, b2 = int(m.geom_bodyid[d.contact[k].geom1]), int(m.geom_bodyid[d.contact[k].geom2])
            for a, o in ((b1, b2), (b2, b1)):
                if a in self._arm_bodies and int(m.body_rootid[o]) != robot_root:
                    hit.add(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, o) or "world")
        return hit

    def _body_state(self):
        a, v = self._pelvis_qadr, self._pelvis_dadr
        if self.free_base:
            quat = self.d.qpos[a:a + 7][3:].copy()
            ang = self.d.qvel[v + 3:v + 6].copy()     # free joint: angular velocity in the body frame
        else:
            quat = self.d.xquat[self._pelvis].copy()
            ang = np.zeros(3)
        return BodyState(t=float(self.d.time), q=self.d.qpos[self._body_q].copy(),
                         qd=self.d.qvel[self._body_v].copy(), base_quat=quat, base_ang_vel=ang)

    # --- stepping -------------------------------------------------------------
    def _step_once(self, dt):
        """One physics step (advances self.d.time by dt); caller holds self.lock."""
        self._control(dt)
        mujoco.mj_step(self.m, self.d)

    def _control(self, dt):
        """Controller tick at its own rate, joint PD on every physics step; caller holds self.lock."""
        if self.d.time - self._vel_t > self.DEADMAN_S:
            self._wbc.zero_velocity()
        if self._tick % self._decim == 0:
            self._cmd, self._twist = self._controller_output()
        self._tick += 1
        if self._twist is not None:
            self._integrate_base(self._twist, dt)
        if self._cmd is not None and len(self._own_act):
            tau = self._cmd.torques(self.d.qpos[self._own_q], self.d.qvel[self._own_v])
            if not np.isfinite(tau).all():
                self._trip("non-finite torque")
                self._cmd = damping_command(self._wbc.joints)
                tau = self._cmd.torques(self.d.qpos[self._own_q], self.d.qvel[self._own_v])
            self.d.ctrl[self._own_act] = tau

    def _controller_output(self):
        joints = self._wbc.joints
        if self._faulted:
            return (damping_command(joints) if joints else None), None
        try:
            out = self._wbc.step(self._body_state() if self._wbc.needs_state else None)
            if out.joints is not None:
                if tuple(out.joints.joints) != tuple(joints):
                    raise ControllerFault(f"commanded {out.joints.joints}, owns {joints}")
                out.joints.validate()
            return out.joints, out.base_twist
        except Exception as e:  # noqa: BLE001 - any controller failure must end in damping
            self._trip(f"{type(e).__name__}: {e}")
            return (damping_command(joints) if joints else None), None

    def _trip(self, error):
        """Controller fault: damping from now on, until an explicit set_mode() or reset()."""
        self._faulted = True
        self._mode = "damp"
        self._fault_seq += 1
        self._fault = {"seq": self._fault_seq, "t": round(float(self.d.time), 3),
                       "controller": self._wbc.name, "error": str(error)[:300]}
        log.error("controller %s fault at t=%.3f: %s; dropping to damping", self._wbc.name, self.d.time, error)

    def _loop(self):
        dt = self.m.opt.timestep
        nxt = time.time()
        while self.running:
            with self.lock:
                self._step_once(dt)
            nxt += dt
            s = nxt - time.time()
            if s > 0:
                time.sleep(s)
            elif s < -0.5:
                nxt = time.time()

    def step(self, seconds: float) -> None:
        """Advance the simulation by `seconds` (rounded to whole physics steps)."""
        dt = self.m.opt.timestep
        n = round(seconds / dt)
        with self.lock:
            for _ in range(n):
                self._step_once(dt)

    # --- base controller commands ----------------------------------------------
    @property
    def controller(self) -> str:
        if self._wbc.name == "kinematic" and self._mocap_id < 0:
            return "none"                # nothing to move: the scene has no mocap base
        return self._wbc.name

    def controller_caps(self) -> dict:
        caps = self._wbc.caps()
        caps["name"] = self.controller
        if caps["name"] == "none":
            caps["walk"] = False
            caps["stand_in"] = False     # nothing moves the base: there is no stand-in to warn about
        # exactly what sim_teleport accepts: a mocap base or a free base
        caps["teleport"] = bool(self.free_base or self._mocap_id >= 0)
        return caps

    def set_mode(self, mode: str) -> None:
        with self.lock:
            if mode not in self._wbc.modes:
                raise RuntimeError(f"the {self._wbc.name} controller has no mode {mode!r}; "
                                   f"modes: {list(self._wbc.modes)}")
            if self._faulted and "damp" in self._wbc.modes:
                self._wbc.set_mode("damp")   # re-engaging after a fault starts from damping
            self._faulted = False            # an explicit mode change is the operator re-engaging
            self._wbc.set_mode(mode)         # clears the velocity command
            self._mode = mode
            self._vel_t = -math.inf

    def set_base_velocity(self, vx: float, vy: float, wz: float) -> None:
        """Zero is accepted in every mode, faulted included (see WholeBodyController.set_base_velocity);
        a non-zero velocity needs mode "walk" and a controller that walks."""
        with self.lock:
            if float(vx) == 0.0 and float(vy) == 0.0 and float(wz) == 0.0:
                self._wbc.zero_velocity()
                return
            if self._mode != "walk":
                raise RuntimeError("set_mode('walk') first")
            self._wbc.set_base_velocity(vx, vy, wz)
            self._vel_t = self.d.time

    def set_posture(self, base_height: float, torso_pitch: float) -> dict:
        with self.lock:
            if self._faulted:
                raise RuntimeError(f"the {self._wbc.name} controller faulted; set_mode to re-engage it")
            return self._wbc.set_posture(base_height, torso_pitch)

    def _integrate_base(self, twist, dt):
        """Move the mocap body by the body-frame twist; caller holds self.lock."""
        if self._mocap_id < 0 or self._mode != "walk":
            return
        vx, vy, wz = twist
        if vx == 0.0 and vy == 0.0 and wz == 0.0:
            return
        i = self._mocap_id
        pos = self.d.mocap_pos[i]
        yaw = _yaw_of(self.d.mocap_quat[i])
        c, s = math.cos(yaw), math.sin(yaw)
        pos[0] += (c * vx - s * vy) * dt
        pos[1] += (s * vx + c * vy) * dt
        yaw += wz * dt
        self.d.mocap_quat[i] = (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))   # wxyz, yaw-only

    # --- rendering: thread-local renderers, snapshot of the physics state --------
    def _snapshot(self):
        """Copy the live state into the render snapshot; caller holds self.gl_lock."""
        with self.lock:
            mujoco.mj_copyData(self._snap, self.m, self.d)
        return self._snap

    def _renderer(self):
        r = getattr(self._tls, "r", None)
        if r is None:
            r = mujoco.Renderer(self.m, self.cam_h, self.cam_w)
            r.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
            r.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = False
            self._tls.r = r
        return r

    def _lidar_sim(self):
        lid = getattr(self._tls, "lidar", None)
        if lid is None:
            rl = mujoco.Renderer(self.m, 192, 192)
            lid = LidarSim(self.m, rl, points_per_frame=int(os.environ.get("LIDAR_POINTS", "15000")))
            self._tls.lidar_renderer = rl   # keep alive alongside the LidarSim
            self._tls.lidar = lid
        return lid

    def state(self) -> RobotState:
        with self.lock:
            q = {}
            qd = {}
            for n, j in self._state_joints:
                q[n] = float(self.d.qpos[self.m.jnt_qposadr[j]])
                qd[n] = float(self.d.qvel[self.m.jnt_dofadr[j]])
            tau = {n: float(self.d.actuator_force[i]) for n, i in self.act.items()}
            base_pos = self.d.xpos[self._pelvis].tolist()
            base_quat = self.d.xquat[self._pelvis].tolist()
            lin = ang = None
            if self.free_base:
                # Free floating base: the free joint's velocity is a real measurement. qvel[0:3] is
                # world-frame linear, qvel[3:6] body-frame angular (what the pelvis IMU reports).
                v = self._pelvis_dadr
                rot = self.d.xmat[self._pelvis].reshape(3, 3)
                lin = (rot.T @ self.d.qvel[v:v + 3]).tolist()
                ang = self.d.qvel[v + 3:v + 6].tolist()
                gyro = list(ang)
            else:
                # Mocap weld: no base_lin_vel/base_ang_vel - the pelvis velocity is constraint-driven,
                # not the kinematic one we command. The bridge differences the pose instead.
                gyro = self.d.cvel[self._pelvis, 0:3].tolist() if self._imu_site >= 0 else [0, 0, 0]
            objs = {nm: self.d.xpos[b].round(3).tolist() for nm, b in self._objects.items()}
            # No joint_temp: MuJoCo has no thermal model, and a made-up value would arm a fake alarm.
            extra = {"objects": objs, "ncon": int(self.d.ncon), "controller_caps": self.controller_caps(),
                     "controller_fault": dict(self._fault) if self._fault else None,
                     "sim_model": self.model_info}
            return RobotState(t=self.d.time, joints=q, joint_vel=qd, joint_torque=tau, base_pos=base_pos,
                              base_quat=base_quat, base_lin_vel=lin, base_ang_vel=ang, imu_gyro=gyro,
                              mode=self._mode, controller=self.controller, extra=extra)

    def rgb(self, camera):
        if camera not in CAMERAS:
            return None
        with self.gl_lock:
            r = self._renderer()
            r.update_scene(self._snapshot(), camera=camera)
            return r.render().copy()

    def depth(self, camera):
        cam = DEPTHS.get(camera, camera)
        if mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_CAMERA, cam) < 0:
            return None
        with self.gl_lock:
            r = self._renderer()
            r.enable_depth_rendering()
            try:
                r.update_scene(self._snapshot(), camera=cam)
                return r.render().copy()
            finally:
                r.disable_depth_rendering()

    def lidar(self):
        with self.gl_lock:
            lid = self._lidar_sim()
            pw, _pl, pos, quat = lid.scan(self._snapshot())
        return pw, pos, quat

    def set_joint_targets(self, targets):
        owned = sorted(n for n in targets if n in self._own)
        if owned:
            # the gate already rejects legs and waist; this is the second wall, for direct callers
            raise ValueError(f"{owned} are driven by the {self._wbc.name} controller, not commandable")
        with self.lock:
            for n, v in targets.items():
                i = self.act.get(n)
                if i is not None:
                    lo, hi = self.m.actuator_ctrlrange[i]
                    self.d.ctrl[i] = float(np.clip(v, lo, hi))

    def set_head(self, pan=None, tilt=None):
        t = {}
        if pan is not None:
            t["head_pan_joint"] = pan
        if tilt is not None:
            t["head_tilt_joint"] = tilt
        self.set_joint_targets(t)

    # --- simulator tools (bridge commands sim_reset / sim_randomise / sim_teleport) ----
    def _set_const(self):
        """Recompute the model constants derived from mass and inertia, on scratch state.

        mj_setConst evaluates the model at qpos0 and leaves the mjData it is given in that
        configuration, so it must never be handed `self.d`: that would silently reset the clock,
        the robot and every object to the initial pose in the middle of a randomisation."""
        if getattr(self, "_scratch", None) is None:
            self._scratch = mujoco.MjData(self.m)   # allocated once; only mass changes need it
        mujoco.mj_setConst(self.m, self._scratch)

    def _subtree_bodies(self, root):
        """Body ids of the subtree rooted at `root` (an object is usually one body, a crate is not)."""
        out = [int(root)]
        for b in range(int(root) + 1, self.m.nbody):     # children always follow their parent
            if int(self.m.body_parentid[b]) in out:
                out.append(int(b))
        return out

    def _subtree_geoms(self, root):
        bodies = set(self._subtree_bodies(root))
        return [g for g in range(self.m.ngeom) if int(self.m.geom_bodyid[g]) in bodies]

    def _find_pick_table(self):
        """(cx, cy, top_z, half_x, half_y) of the pick surface, or None if the scene has no such geom.

        No hard-coded fallback: guessing a table at (0.55, 0, 0.75) would drop every object of a
        scenario that has no pick surface into thin air in front of the robot, and say nothing.
        Loading such a scene is fine (a navigation-only world is legitimate) - only sim_randomise
        needs the surface, and it raises there, where the caller can act on it.
        """
        g = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_GEOM, PICK_TABLE_GEOM)
        if g < 0:
            return None
        # in world axes: rubble_yard's bench is turned 90 degrees, so its 0.35 x 0.60 top is
        # 0.60 x 0.35 on the ground and taking geom_size straight put objects over thin air
        rot = np.abs(self.d.geom_xmat[g].reshape(3, 3))     # valid after mj_forward in __init__
        hx, hy, hz = rot @ self.m.geom_size[g]
        cx, cy, cz = self.d.geom_xpos[g]
        return float(cx), float(cy), float(cz + hz), float(hx), float(hy)

    def _pick_surface(self):
        if self._table is None:
            raise RuntimeError(
                f"this scene has no '{PICK_TABLE_GEOM}' geom, so there is no pick surface to "
                f"randomise onto (model {self.model_path})")
        return self._table

    def _above_surface(self, p):
        """Is position `p` inside the pick surface footprint (plus overhang) and above its top?"""
        cx, cy, top_z, hx, hy = self._pick_surface()
        return (abs(float(p[0]) - cx) <= hx + SURFACE_MARGIN
                and abs(float(p[1]) - cy) <= hy + SURFACE_MARGIN
                and float(p[2]) > top_z - SURFACE_TOL)

    def _on_surface(self, nm):
        """Does object `nm` belong to the pick surface: is it there now, or was it there initially?

        The second half matters because objects fall off. An episode that pushed a cube onto the
        floor must not permanently shrink the set the next randomisation draws from - `sim_reset`
        would put it back, and so does the initial-pose test, without needing a reset first."""
        a, _ = self._obj_adr[nm]
        return self._above_surface(self.d.qpos[a:a + 3]) or self._above_surface(self._obj_init[nm][:3])

    def _select(self, objects, scope):
        """Which objects a randomisation applies to. Returns (selected, skipped)."""
        if objects is not None:
            if isinstance(objects, str):
                objects = [objects]
            unknown = [n for n in objects if n not in self._objects]
            if unknown:
                raise RuntimeError(f"unknown object(s) {unknown}; scene has {sorted(self._objects)}")
            sel = [n for n in self._objects if n in set(objects)]
        elif scope == "all":
            sel = list(self._objects)
        elif scope == "surface":
            sel = [n for n in self._objects if self._on_surface(n)]
        else:
            raise RuntimeError(f"unknown scope {scope!r}; expected 'surface' or 'all'")
        return sel, [n for n in self._objects if n not in set(sel)]

    def _table_footprints(self):
        """[(name, cx, cy, half_x, half_y)] of every '*_table_top' box geom in world xy.

        Used by sim_teleport to refuse a target the robot's body would be standing inside. It only
        knows about table tops, so it is a floor for obviously-wrong targets, not a collision
        check: teleporting into a shelf, a wall or a crate is still allowed and still wrong.
        """
        out = []
        for g in range(self.m.ngeom):
            nm = mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_GEOM, g)
            if nm and nm.endswith("_table_top"):
                hx, hy, _ = np.abs(self.d.geom_xmat[g].reshape(3, 3)) @ self.m.geom_size[g]  # world axes
                cx, cy, _ = self.d.geom_xpos[g]
                out.append((nm, float(cx), float(cy), float(hx), float(hy)))
        return out

    def _place_object(self, nm, x, y, z, yaw):
        """Write pose (yaw-only orientation) and zero velocity of one object; caller holds self.lock."""
        a, v = self._obj_adr[nm]
        self.d.qpos[a:a + 3] = (x, y, z)
        self.d.qpos[a + 3:a + 7] = (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))
        self.d.qvel[v:v + 6] = 0.0

    def sim_reset(self):
        """Keyframe reset plus every object back to its initial pose, colour and dynamics."""
        with self.lock:
            if self._model_dirty:       # undo colour / mass / friction randomisation
                for k, v in self._model0.items():
                    getattr(self.m, k)[:] = v
                self._set_const()
                self._model_dirty = False
            mujoco.mj_resetDataKeyframe(self.m, self.d, self._key_stand)   # restores mocap pose and d.time
            for nm, q0 in self._obj_init.items():
                a, v = self._obj_adr[nm]
                self.d.qpos[a:a + 7] = q0
                self.d.qvel[v:v + 6] = 0.0
            self._wbc.zero_velocity()
            self._vel_t = -math.inf
            mujoco.mj_forward(self.m, self.d)
            self._faulted, self._fault = False, None
            self._tick, self._cmd, self._twist = 0, None, None
            if self.free_base:
                if self._weld >= 0:
                    self.d.eq_active[self._weld] = 0
                self._apply_start_pose()
                if self._mode == "damp":             # a reset robot is standing again
                    self._wbc.set_mode(self._wbc.initial_mode)
                    self._mode = self._wbc.initial_mode
            self._wbc.reset(self._body_state())

    reset = sim_reset

    def sim_randomise(self, seed=None, objects=None, scope="surface",
                      pose=True, colour=False, mass=False, friction=False):
        """Randomise the scene's manipulable objects. Deterministic for a given seed.

        What it touches - each axis independent, each off unless asked for (except pose):
          pose      re-place on the pick surface: uniform xy (8 cm apart, 5 cm from the edge),
                    uniform yaw, upright, at the object's own rest height above its support - so
                    a tipped object does not inherit a wrong height.
          colour    +/- COLOUR_JITTER per RGB channel around the object's own colour. Bounded on
                    purpose: 'red_box' stays recognisably red, so a language-conditioned policy
                    still has a referent. Pass a float for a different amount.
          mass      +/- MASS_JITTER relative, per object, inertia scaled with it.
          friction  +/- FRICTION_JITTER relative on all three friction coefficients, per geom.
        colour/mass/friction change the *model*, so they persist until sim_reset, which restores
        them. None of them add any per-step cost.

        Which objects (default: whatever belongs on the pick surface):
          objects=["red_box", ...]   exactly these, wherever they are
          scope="surface"            free bodies that belong on the pick surface: inside its
                                     footprint and above its top, now or in the scene's initial
                                     pose - so a crate on the warehouse floor stays on the floor
                                     and a cube that fell off the table still comes back
          scope="all"                every free body (the pre-scoping behaviour)
        Objects that are not selected are left alone, and pose randomisation keeps its spacing
        clear of the ones that are already on the surface.

        `seed=None` draws a fresh seed and returns it, so an interesting episode can be replayed.
        Returns {"seed", "objects", "skipped", "placed": {name: [x, y, yaw]}, "colour", "mass",
        "friction"} with the per-object values of whatever was enabled.
        """
        seed = int(np.random.SeedSequence().generate_state(1)[0]) if seed is None else int(seed)
        # one independent stream per axis, so enabling colour cannot move the objects
        r_pose, r_col, r_mass, r_fric = (np.random.default_rng(s)
                                         for s in np.random.SeedSequence(seed).spawn(4))
        amounts = {"colour": COLOUR_JITTER, "mass": MASS_JITTER, "friction": FRICTION_JITTER}
        a_col, a_mass, a_fric = (0.0 if v is False or v is None else
                                 (amounts[k] if v is True else abs(float(v)))
                                 for k, v in (("colour", colour), ("mass", mass), ("friction", friction)))
        with self.lock:
            sel, skipped = self._select(objects, scope)
            if not sel:
                raise RuntimeError(
                    "nothing to randomise: no free body is resting on the pick surface "
                    f"{PICK_TABLE_GEOM}. Pass objects=[...] or scope='all'.")
            placed = self._sample_poses(sel, r_pose) if pose else {}
            out_col = self._sample_colours(sel, r_col, a_col) if a_col > 0 else {}
            out_mass = self._sample_masses(sel, r_mass, a_mass) if a_mass > 0 else {}
            out_fric = self._sample_frictions(sel, r_fric, a_fric) if a_fric > 0 else {}
            try:
                for nm, (x, y, yaw) in placed.items():
                    self._place_object(nm, x, y, self._rest_z(nm), yaw)
                for nm, rgba in out_col.items():
                    for g in self._obj_geoms[nm]:
                        self.m.geom_rgba[g] = rgba
                        self.m.geom_matid[g] = -1    # an explicit rgba only wins without a material
                for nm, f in out_mass.items():
                    for b in self._obj_bodies[nm]:
                        self.m.body_mass[b] = self._model0["body_mass"][b] * f
                        self.m.body_inertia[b] = self._model0["body_inertia"][b] * f
                for nm, f in out_fric.items():
                    for g in self._obj_geoms[nm]:
                        self.m.geom_friction[g] = self._model0["geom_friction"][g] * f
                if out_mass:
                    self._set_const()       # body_subtreemass and dof_invweight0 are derived
                self._model_dirty = self._model_dirty or bool(out_col or out_mass or out_fric)
            finally:
                mujoco.mj_forward(self.m, self.d)
        return {"seed": seed, "objects": sel, "skipped": skipped,
                "placed": {nm: [round(x, 4), round(y, 4), round(yaw, 4)] for nm, (x, y, yaw) in placed.items()},
                "colour": {nm: [round(float(c), 3) for c in rgba] for nm, rgba in out_col.items()},
                "mass": {nm: round(f, 4) for nm, f in out_mass.items()},
                "friction": {nm: round(f, 4) for nm, f in out_fric.items()}}

    def _rest_z(self, nm):
        """World z for object `nm` when placed upright on the pick surface.

        Its initial height above whatever it was resting on then (the pick surface, or the floor
        for an object that started on the ground and was named explicitly)."""
        _, _, top_z, _, _ = self._pick_surface()
        z0 = float(self._obj_init[nm][2])
        rest = z0 - top_z if z0 > top_z - SURFACE_TOL else z0
        return top_z + max(rest, 0.0)

    def _surface_obstacles(self, sel):
        """World AABBs of everything a placed object could land inside: [(cx, cy, cz, hx, hy, hz)].

        Anything standing on or reaching over the pick surface counts - the tote on the bench, the
        robot's own hands hanging over the near edge, another object that is staying put. Anything
        whose top is at or below the surface (the tabletop itself, its legs, the floor) does not:
        that is what the object rests on. Boxes give an exact AABB; a rotated or curved geom gives
        a slightly conservative one, which only costs a little placement freedom.
        """
        cx, cy, top_z, hx, hy = self._pick_surface()
        skip = {g for nm in sel for g in self._obj_geoms[nm]}
        out = []
        for g in range(self.m.ngeom):
            if g in skip or self.m.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE:
                continue
            rot = np.abs(self.d.geom_xmat[g].reshape(3, 3))
            centre = self.d.geom_xpos[g] + rot @ self.m.geom_aabb[g, :3]
            half = rot @ self.m.geom_aabb[g, 3:]
            if centre[2] + half[2] <= top_z + SURFACE_TOL:      # a support, not an obstacle
                continue
            if (abs(centre[0] - cx) > hx + half[0] + SURFACE_MARGIN
                    or abs(centre[1] - cy) > hy + half[1] + SURFACE_MARGIN):
                continue                                        # nowhere near the surface
            out.append((float(centre[0]), float(centre[1]), float(centre[2]),
                        float(half[0]), float(half[1]), float(half[2])))
        return out

    @staticmethod
    def _clear_of(x, y, r, z_lo, z_hi, obstacle):
        """Is a disc of radius r at (x, y) spanning z_lo..z_hi clear of one AABB obstacle?"""
        ox, oy, oz, hx, hy, hz = obstacle
        if oz + hz <= z_lo or oz - hz >= z_hi:
            return True                                          # passes under or over it
        dx = max(0.0, abs(x - ox) - hx)
        dy = max(0.0, abs(y - oy) - hy)
        return math.hypot(dx, dy) >= r + PLACE_CLEARANCE

    def _sample_poses(self, sel, rng):
        """{name: (x, y, yaw)} on the pick surface; samples everything before anything moves.

        Both the edge margin and the spacing grow with the objects' own radius, otherwise a crate
        sampled 5 cm from the edge hangs half off it and tips onto the floor during the first
        settle - which is what happened in warehouse_aisle and rubble_yard."""
        cx, cy, _top_z, hx, hy = self._pick_surface()
        obstacles = self._surface_obstacles(sel)
        # objects staying put on the surface still occupy space: keep clear of them
        rest = [(float(p[0]), float(p[1]), self._obj_radius[n])
                for n, p in ((n, self.d.qpos[self._obj_adr[n][0]:self._obj_adr[n][0] + 3])
                             for n in self._objects if n not in set(sel))
                if self._above_surface(p)]
        for nm in sel:                          # a surface too small for one object never works
            margin = max(RANDOMISE_MARGIN, self._obj_radius[nm])
            if margin >= hx or margin >= hy:
                raise RuntimeError(f"pick table too small for {nm} (needs a {margin:.2f} m edge margin)")
        order = sorted(sel, key=lambda n: -self._obj_radius[n])   # biggest first: it has fewest options
        for _attempt in range(PLACE_ATTEMPTS):
            placed, taken = {}, list(rest)
            for nm in order:
                r = self._obj_radius[nm]
                z_lo, z_hi = self._rest_z(nm), self._rest_z(nm) + self._obj_height[nm]
                margin = max(RANDOMISE_MARGIN, r)
                lo = (cx - hx + margin, cy - hy + margin)
                hi = (cx + hx - margin, cy + hy - margin)
                for _ in range(200):
                    x = float(rng.uniform(lo[0], hi[0]))
                    y = float(rng.uniform(lo[1], hi[1]))
                    if not all(math.hypot(x - px, y - py) >= max(RANDOMISE_SPACING, r + pr)
                               for px, py, pr in taken):
                        continue
                    if all(self._clear_of(x, y, r, z_lo, z_hi, o) for o in obstacles):
                        break
                else:
                    break                       # crowded corner: restart the whole layout
                taken.append((x, y, r))
                placed[nm] = (x, y, float(rng.uniform(-math.pi, math.pi)))
            if len(placed) == len(sel):
                return {nm: placed[nm] for nm in sel}       # report in scene order, not size order
        raise RuntimeError(f"could not lay out {len(sel)} objects on {PICK_TABLE_GEOM} clear of each "
                           f"other, the robot and the scenery")

    def _sample_colours(self, sel, rng, amount=COLOUR_JITTER):
        out = {}
        for nm in sel:
            gs = self._obj_geoms[nm]
            if not gs:
                continue
            base = self._model0["geom_rgba"][gs[0]].copy()   # one colour per object, not per panel
            base[:3] = np.clip(base[:3] + rng.uniform(-amount, amount, 3), 0.05, 1.0)
            out[nm] = base
        return out

    def _sample_masses(self, sel, rng, amount=MASS_JITTER):
        return {nm: float(1.0 + rng.uniform(-amount, amount)) for nm in sel}

    def _sample_frictions(self, sel, rng, amount=FRICTION_JITTER):
        return {nm: float(1.0 + rng.uniform(-amount, amount)) for nm in sel}

    def sim_teleport(self, x, y, yaw):
        """Move the robot to (x, y, yaw) [m, rad]; height is kept. Targets inside a table
        footprint are rejected.

        Kinematic base: both the mocap target and the pelvis free joint are written (velocity
        zeroed) so the weld starts at zero error instead of dragging the pelvis through the scene.
        Free base: the pelvis free joint is written, the robot's velocities are zeroed and the
        controller is reset, so it starts balancing from rest at the new pose.
        """
        if self._mocap_id < 0 and not self.free_base:
            raise RuntimeError("teleport needs the kinematic base (scene built with FIXED_BASE=1) "
                               "or a whole-body controller on a free base")
        x, y, yaw = float(x), float(y), float(yaw)
        for nm, cx, cy, hx, hy in self._tables:
            if abs(x - cx) <= hx and abs(y - cy) <= hy:
                raise RuntimeError(f"target ({x:.2f}, {y:.2f}) is inside the footprint of {nm} "
                                   f"(x {cx - hx:.2f}..{cx + hx:.2f}, y {cy - hy:.2f}..{cy + hy:.2f})")
        quat = (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))
        with self.lock:
            i = self._mocap_id
            if i >= 0:
                self.d.mocap_pos[i, 0:2] = (x, y)
                self.d.mocap_quat[i] = quat
            if self._pelvis_qadr >= 0:
                a, v = self._pelvis_qadr, self._pelvis_dadr
                self.d.qpos[a:a + 2] = (x, y)
                self.d.qpos[a + 3:a + 7] = quat
                self.d.qvel[v:v + 6] = 0.0
            self._wbc.zero_velocity()
            self._vel_t = -math.inf
            if self.free_base:
                self.d.qvel[self._body_v] = 0.0
                mujoco.mj_forward(self.m, self.d)
                self._wbc.reset(self._body_state())
                self._tick, self._cmd = 0, None
            mujoco.mj_forward(self.m, self.d)
            z = float(self.d.qpos[self._pelvis_qadr + 2]) if self.free_base else float(self.d.mocap_pos[i, 2])
        return {"x": x, "y": y, "z": z, "yaw": yaw}

    def close(self):
        """Stop the physics loop and release the calling thread's renderers.

        Renderers created by other threads are thread-local and are released when those
        threads exit (or when they call close() themselves).
        """
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=1.0)
        with self.gl_lock:
            for attr in ("r", "lidar_renderer"):
                r = getattr(self._tls, attr, None)
                if r is not None:
                    r.close()
                    delattr(self._tls, attr)
            if getattr(self._tls, "lidar", None) is not None:
                del self._tls.lidar
