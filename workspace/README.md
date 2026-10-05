# workspace — the guided path

Six runnable examples, in order, and the mental model that makes them legible. Read the first
section before running anything; it is twenty lines and it saves an afternoon.

Everything assumes you have installed the package (`uv sync`, or `pip install -e .`). Run the
examples from the repo root:

```bash
uv run python workspace/01_hello_sim.py
```

They write to `workspace/out/`, which is gitignored.

---

## The mental model

### 1. `RobotSource` is the only interface

`g1.robot.robot_source.RobotSource` is an abstract class with about ten methods. `SimSource`
implements it on top of MuJoCo. `RealSource` is a skeleton that would implement it on top of
the robot's DDS topics, the RealSense SDK and a SLAM node. Every other piece of code —
yours included — should talk to the interface, not to MuJoCo.

```python
state()                      -> RobotState   joints, velocities, torques, pose, IMU, extras
rgb(camera)                  -> uint8 HxWx3 | None
depth(camera)                -> float32 HxW, metres | None
lidar()                      -> (points Nx3 float32 in the map frame, sensor_pos, sensor_quat)
set_joint_targets({name: rad})               arms, hands, head — position targets
set_head(pan, tilt)                          the two head-module joints
set_base_velocity(vx, vy, wz)                body frame, m/s and rad/s
set_mode(mode)                               "stand" | "walk" | "damp" (+ "sim" for kinematic)
controller_caps()            -> dict         what the active base controller can actually do
```

Camera names are hardware names and identical on both sides: `d435i_rgb` / `d435i_depth`
(the RealSense behind the visor), `d455_rgb` / `d455_depth` (on the pan/tilt head module),
and `third_person`, which exists only in sim and returns `None` on a real source. You ask for
depth by the *colour* camera's name; the source maps it, which is the aligned-depth convention.

The payoff is that a policy written against `RobotSource` runs against the simulator today and
against a robot the day `RealSource` is finished, with no change to the policy. That has not
been demonstrated here — see "what is unproven" at the bottom — but it is why the interface
exists and why you should not reach past it.

Units are SI everywhere: metres, kilograms, seconds, radians. Quaternions are wxyz. The world
frame is +x forward, +y left, +z up, with the origin on the floor where the robot spawns.

### 2. The 31-slot layout

Observations and actions are the same 31 columns, in the same order, always:

| index | count | what | units |
|---|---|---|---|
| 0–13 | 14 | arm joint positions — left 7 then right 7 (shoulder pitch/roll/yaw, elbow, wrist roll/pitch/yaw) | rad |
| 14–25 | 12 | hand joint positions — left 6 then right 6 (Revo 2 proximals; the distal links follow through model equalities) | rad |
| 26–27 | 2 | `head_pan_joint`, `head_tilt_joint` | rad |
| 28–30 | 3 | `base_vx`, `base_vy`, `base_wz` — body frame | m/s, m/s, rad/s |

Two things about it are easy to get wrong and expensive to get wrong:

- **Slots 28–30 mean different things in the two vectors.** In an observation they are the base
  velocity as *measured*; in an action they are the base velocity as *commanded*.
- **The action is a held setpoint stream, not a delta stream.** Each joint slot holds its last
  commanded target until something commands it again. The first action of an episode is
  "stay where you are, zero base velocity".

Legs and waist are *not* in the layout. They belong to the whole-body controller, which is why
`set_joint_targets` refuses them while a controller owns them. A policy trained on these
columns structurally cannot command a leg.

Example 06 builds the layout from `BODY_JOINTS`, `HAND_JOINTS` and `HEAD_JOINTS` in
`g1.robot.robot_source` — those lists are the single source of truth for joint names and order
(the 29-DoF G1 motor order, straight from Unitree's description).

### 3. A WBC turns a velocity into joint targets

Nothing commands leg joints directly. You emit a body-frame base velocity `(vx, vy, wz)`; a
*whole-body controller* from `g1.wbc` turns it into leg and waist joint targets at its own rate
(50 Hz for the learned policies), and the physics loop tracks those with a PD at 500 Hz. That
split is not an implementation detail: it is how the real G1's motor drivers work, so a
controller written against this interface is the thing you would ship.

| backend | walks | free base | what it is |
|---|---|---|---|
| `kinematic` | "yes" | no | **The default, and not a controller at all.** The pelvis is welded to a mocap body and the simulator integrates the commanded velocity into its pose. The legs never drive anything and the robot cannot fall. Tracking is perfect because nothing is tracking. `controller_caps()["stand_in"]` is `True`. |
| `holosoma` | yes | yes | A real pretrained policy: Amazon FAR's FastSAC G1 29-DoF velocity-tracking network, run on CPU through onnxruntime. It balances, it has tracking error, and it can fall over. |
| `pd_stand` | no | yes | Stiff PD around an analytic squat/lean pose. Does not walk; supports posture (pelvis height, torso pitch). The simplest example of what a controller has to implement. |

Pick one at construction — `SimSource(controller="holosoma")`, or `$G1_SIM_CONTROLLER` — and
the same scene file serves all three: `SimSource` switches the mocap weld off at load time for
a free-base controller. Use `kinematic` when the base is not what you are studying and you want
it out of the way. Use `holosoma` for any number you intend to believe.

To add your own backend: subclass `WholeBodyController`, implement `reset` and `_step`, set the
class flags (`supports_walk`, `supports_posture`, `rate_hz`, `joints`) and register it in
`CONTROLLERS` in `g1/wbc/__init__.py`. The simulator and the capability reporting key off those
flags, so nothing else changes.

---

## The examples

| | | proves it worked by |
|---|---|---|
| `01_hello_sim.py` | Load the scene, step physics, read the state, save a frame. | `out/01_third_person.png` |
| `02_move_the_arm.py` | Command arm and hand joints; watch the state converge; see clipping. | `out/02_before.png`, `out/02_after.png` |
| `03_read_cameras.py` | RGB + metric depth from both RealSenses; move the head and see only the D455 follow. | six PNGs, depth ranges in metres |
| `04_lidar_and_a_map.py` | A Mid-360 scan as a point cloud, then a top-down occupancy grid accumulated over several poses. | `out/04_occupancy.png`, `out/04_cloud.npy` |
| `05_a_walking_policy.py` | The same velocity command through `kinematic` and `holosoma`, measured side by side. | two PNGs and the speed/leg-motion numbers |
| `06_record_an_episode.py` | Record a scripted trajectory into `.npz` + `.json` in the 31-slot layout. | `out/episode_000.npz`, `out/episode_000.json` |

Each file is standalone and takes an optional model path:

```bash
uv run python workspace/04_lidar_and_a_map.py models/scenarios/warehouse_aisle.xml
```

Two habits worth copying out of them:

- `SimSource(..., realtime=False)` means *you* own the clock: nothing advances until you call
  `step(seconds)`. That is what a script, a test or a training loop wants. The default
  `realtime=True` runs physics in a background thread at wall-clock speed, which is what an
  interactive session wants.
- Velocity commands expire after `SimSource.DEADMAN_S` (0.3 s) without a refresh, exactly like
  the robot's own velocity interface. Re-send inside your control loop.

`docs/cheatsheet.md` is the one-page command and API reference.

---

## Adding a scenario

A scenario is a ~40-line declarative file in `scenarios/` that composes a world out of the
robot, a room or terrain, furniture and manipulable objects. `scripts/build_scenarios.py`
compiles it into a self-contained MuJoCo model in `models/scenarios/`, which `SimSource` loads
unchanged — same cameras, same LiDAR, same `stand` keyframe, same `sim_reset` /
`sim_randomise` / `sim_teleport` behaviour as the base scene.

```bash
uv run python scripts/build_scenarios.py     # build all five
uv run python scripts/check_scenarios.py     # load each one and prove it works
uv run python scripts/check_scenarios.py --png
```

If all you want is one extra object in the existing scene you do not need any of this: a plain
MJCF `<include>` of `models/g1_pick_place.xml` plus a `<body>` of your own keeps its pose
through load and reset, because `SimSource` re-pads short keyframes from `qpos0`.

The build step is for what an include cannot do: instantiating the same part many times without
name collisions, walls with doorways, generated terrain, seeded variants, and the placement
checks that stop an object silently starting in mid-air or inside the furniture.

To add one: copy any file in `scenarios/`, keep the `<doc>` honest about what the scenario is
*for*, and build it. One hard requirement — **every scenario needs a table-like `<part>` called
`pick_table`**, because the randomisation surface is the geom `pick_table_top`. The build
script fails loudly if it is missing.

The full element reference, the randomisation attributes and what the checks actually enforce
are in **`docs/scenario-dsl.md`**.

## Adding an object

Drop a standalone MuJoCo model into `models/objects/<type>.xml` and refer to it by file stem.
Four conventions the build relies on:

- **Exactly one body** in `<worldbody>`. A free joint in it makes the object manipulable
  (`<object>`); without one, or when instantiated as `<part>`, it is scenery.
- **Origin at the bottom face**, centred in x and y — so a scenario places it on a surface by
  writing the surface height (`pos="0.6 0.1 0.75"` sits exactly on a 0.75 m tabletop).
- **A material named `mat`** for the main colour, so `rgba=` / `palette=` on the scenario
  element can recolour each instance independently.
- **Table-like parts name their working surface geom `top`**, so attaching with the prefix
  `pick_table_` produces `pick_table_top`.

Give it real mass and friction — the compiler derives inertia from the geom layout, which is
why the crate is built from panels rather than one solid box: a hollow crate has to swing like
one. Inspect a part on its own with `python -m mujoco.viewer --mjcf=models/objects/crate.xml`.

Everything already in `models/objects/` is a self-authored MuJoCo primitive — boxes, cylinders,
spheres — with no mesh, no texture and nothing to licence-clear. Keep it that way if you want
the repo to stay as portable as it is. The library and the friction conventions are listed in
`docs/scenario-dsl.md`.

---

## Where you take it from here

### Fine-tuning a VLA on recorded episodes

Example 06 is the recording half, and it is deliberately the boring half: 31-slot state, 31-slot
action, timestamps, two camera streams, and a JSON file that says what the columns mean. Scale
that up with `sim_randomise(seed=...)` per episode (it returns the seed, so any episode can be
replayed exactly) across the five scenarios, and vary the build-time seeds too
(`build_scenarios.py --seed N`) so the worlds differ and not just the object placements.

The training itself does not belong in this repo. Convert the `.npz` files to whatever your
trainer eats — the LeRobot recipe is sketched at the bottom of `06_record_an_episode.py` — in a
separate environment that is allowed to depend on torch. Two things must survive the conversion
or the policy will score well offline and do nothing useful: the **column names**, so the layout
is identified by name and not by luck, and the **`sim_model` sha256** from the JSON, so you can
tell which worlds an episode came from when a policy works in one scenario and not another.

Keep the action slots you train on explicit. A manipulation policy should be trained on slots
`[0, 28]` and never see the base columns, so it cannot learn to walk by accident.

### Writing a walking policy

`holosoma` is the baseline and the thing to beat; example 05 measures it against a command so
you have a number. To train your own, the useful surface is `WholeBodyController`: what it may
observe is `BodyState` — proprioception only (joint positions and velocities, pelvis quaternion,
IMU gyro), because that is all the real robot gives a controller at 50 Hz — and what it must
return is a `JointCommand`, i.e. PD targets and gains for the joints it owns. Write your policy
to that contract and it drops into the simulator, and later onto the robot, unchanged.

This repo does not train anything: there is no RL loop, no vectorised environment and no torch.
`SimSource` with `realtime=False` is single-instance and renders through a GL context, so it is
a fine evaluation harness and a poor sampler. For training, lift the physics setup (the scene,
the actuator conversion in `_init_controller`, the PD loop) into a batched environment and bring
the resulting ONNX back here to evaluate it alongside `holosoma`.

### Evaluating against a real robot

The intended path is that nothing about the policy changes. You construct `RealSource` instead
of `SimSource`, the same `state()` / `rgb()` / `depth()` / `set_joint_targets()` calls go to DDS
topics and RealSense pipelines instead of MuJoCo, and the same 31 columns come back — same
names, same order, same units, same frames.

**This is unproven.** Be precise about what that means:

- `RealSource` raises `NotImplementedError` in its constructor. Its TODOs name the right topics
  (`rt/lowstate`, `rt/arm_sdk`, the BrainCo hand services, `/livox/lidar`, the SLAM node's
  `/Odometry` and `/cloud_registered`) but none of it is implemented here.
- Several fields are explicitly marked "confirm on the robot, do not guess", because getting
  them wrong is silent: which frame the measured base velocity is in, which motor temperature
  sensor to read. A wrong frame in slot 28–30 is a wrong training column that nothing flags.
- Nothing in this repository has been run on hardware. The sensor poses, joint limits and
  inertias come from Unitree's and BrainCo's published descriptions, and the walking policy
  comes from a third party. Treat the simulator as a well-sourced guess, not as the robot.
- The `kinematic` controller does not transfer at all. Any base-motion number measured under it
  is a property of the stand-in. Propagate `controller_caps()["stand_in"]` into anything you
  log so that number is never compared with a real one.

The honest version of the plan is: the interface is designed so that sim-to-real is a
construction-site change rather than a rewrite, and that claim has not yet been tested.
