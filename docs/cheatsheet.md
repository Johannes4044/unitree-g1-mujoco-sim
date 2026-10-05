# Cheat sheet

One page. For the reasoning behind any of it, see `workspace/README.md`.

## Commands

```bash
uv sync                                            # install (or: pip install -e .)

uv run python scripts/smoke_test.py                # does MuJoCo work on this machine? exit 0 = yes
uv run python scripts/render_scene.py              # every camera + a rollout clip -> out/
uv run python scripts/render_scene.py models/scenarios/house_rooms.xml

uv run python scripts/build_scene.py               # regenerate models/g1_pick_place.xml (needs sources)
uv run python scripts/build_scenarios.py           # scenarios/*.xml -> models/scenarios/*.xml
uv run python scripts/build_scenarios.py --seed 3  # + seeded variants <name>_s3.xml
uv run python scripts/check_scenarios.py [--png]   # load each built model and verify it

uv run python workspace/01_hello_sim.py [model.xml]   # ... 01 through 06
python -m mujoco.viewer --mjcf=models/g1_pick_place.xml   # interactive viewer
```

The five built scenarios are committed, so a fresh clone can run everything offline without a
build step. Only `--seed` variants are transient.

Environment variables: `SIM_MODEL` (default model path, wins over everything),
`G1_SIM_CONTROLLER` (`kinematic` | `holosoma` | `pd_stand`), `SIM_OUT` (where `scripts/` writes),
`LIDAR_POINTS` (points per Mid-360 frame, default 15000).

## Constructing a source

```python
from g1.robot.sim_source import SimSource, default_model_path

sim = SimSource(
    xml=None,            # None -> default_model_path(): $SIM_MODEL, else models/g1_pick_place.xml
    cam_w=424, cam_h=240,# render size; the MJCF declares 848x480 to match the real sensors
    realtime=True,       # True: background thread steps at wall-clock speed
                         # False: nothing advances until you call step(seconds)  <- scripts want this
    controller=None,     # None -> $G1_SIM_CONTROLLER, else "kinematic" on a mocap scene
)
sim.step(1.0)            # advance 1 s of sim time (realtime=False)
sim.close()              # stop the thread, release this thread's GL contexts
```

## RobotSource

| call | returns / does |
|---|---|
| `state()` | `RobotState` — see below |
| `rgb(cam)` | `uint8 (H, W, 3)` or `None`. `cam` ∈ `d435i_rgb`, `d455_rgb`, `third_person` |
| `depth(cam)` | `float32 (H, W)` **in metres**, or `None`. Ask by the colour camera's name |
| `lidar()` | `(points (N, 3) float32 in the map frame, sensor_pos (3,), sensor_quat (4,) wxyz)` |
| `set_joint_targets({name: rad})` | position targets for arms / hands / head; clipped to `ctrlrange`; raises `ValueError` for joints a controller owns |
| `set_head(pan=, tilt=)` | the same, for the two head-module joints |
| `set_base_velocity(vx, vy, wz)` | body frame, m/s and rad/s. Non-zero needs `set_mode("walk")`. **Expires after 0.3 s** — refresh it |
| `set_mode(mode)` | `"damp"`, `"stand"`, `"walk"` (+ `"sim"` for `kinematic`). Clears the velocity command |
| `set_posture(base_height, torso_pitch)` | only if `controller_caps()["posture"]` |
| `controller_caps()` | `{"name", "modes", "walk", "posture", "free_base", "teleport", "stand_in", "rate_hz", ...}` |

`RobotState` fields: `t`, `joints`, `joint_vel`, `joint_torque`, `joint_temp` (empty in sim —
MuJoCo has no thermal model and a fake value would arm a fake alarm), `base_pos`, `base_quat`,
`base_lin_vel` / `base_ang_vel` (`None` unless the base is free), `imu_gyro`, `imu_acc`,
`battery`, `mode`, `controller`, `extra`.

`extra` carries `objects` (free-body positions, i.e. the prompt vocabulary), `ncon`,
`controller_caps`, `controller_fault` and `sim_model` (path, scenario name, seed, and the sha256
of the *compiled* model — put this in your episode metadata).

## Simulator-only tools

```python
sim.sim_reset()                       # keyframe + every object's pose, colour and dynamics back
sim.sim_randomise(seed=7, pose=True, colour=False, mass=False, friction=False,
                  scope="surface")    # -> {"seed", "objects", "skipped", "placed", ...}
sim.sim_teleport(x, y, yaw)           # -> {"x", "y", "z", "yaw"}; refuses targets inside a table
sim.model_info                        # {"path", "scenario", "scenario_base", "scenario_seed", "sha256"}
```

## Joints

```python
from g1.robot.robot_source import BODY_JOINTS, HAND_JOINTS, HEAD_JOINTS
```

`BODY_JOINTS` is the 29-DoF G1 motor order: 12 leg, 3 waist, 14 arm. `LEG_WAIST_JOINTS` is
`BODY_JOINTS[:15]` and `ARM_JOINTS` is `BODY_JOINTS[15:]` (both in `g1.wbc`). `HAND_JOINTS` is
12 Revo 2 proximals, left 6 then right 6; the distal links follow through model equalities.
`HEAD_JOINTS` is `head_pan_joint`, `head_tilt_joint`.

The 31-slot observation/action layout is
`arm (0–13) + hand (14–25) + head (26–27) + base velocity (28–30)` — see `workspace/README.md`
for the semantics and `workspace/06_record_an_episode.py` for the code.

## Whole-body controllers

```python
from g1.wbc import CONTROLLERS, make_controller, holosoma_available
```

| | `kinematic` | `holosoma` | `pd_stand` |
|---|---|---|---|
| walks | yes (non-physically) | yes | no |
| posture | no | no | yes |
| free base | no (mocap weld) | yes | yes |
| rate | every physics step | 50 Hz | 50 Hz |
| modes | `sim`, `damp`, `stand`, `walk` | `damp`, `stand`, `walk` | `damp`, `stand`, `walk` |
| `stand_in` | **True** | False | False |

`damp` mode is zero stiffness plus joint damping (`DAMP_KD = 4.0`), the same as the G1's own
damping mode: the robot sags in a controlled way rather than freezing. It is implemented once in
`g1/wbc/base.py`, so it behaves identically for every controller, and it is the fallback when a
controller faults — a non-finite or malformed command drops to damping in the same physics step
and is reported in `state().extra["controller_fault"]`.

Zero velocity is accepted in every mode by every controller, always. That is what makes it a
usable e-stop.

## Render costs

Per call, software GL (llvmpipe) at 424×240; a GPU is 3–5× faster:

| | |
|---|---|
| `rgb(d455_rgb)` | ~15 ms |
| `rgb(third_person)` | ~43 ms |
| `depth(...)` | ~7 ms |
| `lidar()` (4 × 192×192 depth + unprojection) | ~59 ms |

Physics is ~60 µs/step on an idle machine at a 2 ms timestep, i.e. 30× realtime. Rendering, not
physics, is what limits a recording loop.
