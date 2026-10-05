# G1 sim

A MuJoCo simulator of a Unitree G1 — 29 body joints, two BrainCo Revo 2 hands, a 2-DoF camera
head, two RealSense cameras and a Livox Mid-360 — with the Python interface you build policies
against. It is a stripped-down extract of a larger project: scene, scenarios, robot sources and
whole-body controllers, and nothing else.

Pure Python, permissive licences, no Docker, no network access at run time.

## Install

Python 3.11+. With [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

or with pip:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

Five dependencies: `mujoco`, `numpy`, `pillow`, `imageio`, `onnxruntime` (15 packages
in total, all permissive). Pinned in `requirements.txt`; fully locked in
`requirements-lock.txt`.

**No Python yet, or no internet on the target machine?**
[`docs/INSTALL.md`](docs/INSTALL.md) covers installing Python 3.11 on Windows, macOS
and Linux, and an offline install from a USB stick.

## See something

```bash
uv run python scripts/render_scene.py
```

Thirty seconds later `out/` holds a render from every camera plus a short rollout clip. Then:

```bash
uv run python workspace/01_hello_sim.py
```

which loads the scene, steps physics, prints the state and writes a PNG — and is the first of
six commented examples that are the actual point of this repo.

## What the pieces are

| | |
|---|---|
| `models/g1_pick_place.xml` | The robot plus a work table and four objects. Built by `scripts/build_scene.py` from Unitree's and BrainCo's published descriptions; `models/ASSETS.md` has the provenance. |
| `scenarios/*.xml` | Five worlds in a small declarative DSL — `pick_place_table`, `cluttered_bench`, `house_rooms`, `warehouse_aisle`, `rubble_yard`. `scripts/build_scenarios.py` compiles them into `models/scenarios/`. |
| `g1.robot` | `RobotSource` — the one interface the simulator and a real robot both implement: `state()`, `rgb()`, `depth()`, `lidar()`, `set_joint_targets()`, `set_base_velocity()`. `SimSource` is the MuJoCo implementation; `lidar.py` is the Mid-360 model. |
| `g1.wbc` | Whole-body controllers: what turns a base velocity command into leg joint targets. `kinematic` (a non-physical stand-in, the default), `holosoma` (a real pretrained locomotion policy, ONNX on CPU), `pd_stand`. |
| `workspace/` | **Start here.** Six runnable examples and a guided tour of the above. |
| `scripts/` | Build the scene, build and check the scenarios, render, smoke-test. |

## What this is not

- **No dashboard, no web server, no teleop UI.** Nothing here listens on a port.
- **No training pipeline.** No torch, no lerobot, no dataset tooling. Example 06 records
  episodes into plain `.npz` + `.json`; converting those to a training format is a separate
  job in a separate environment.
- **No real-robot drivers.** `g1.robot.real_source.RealSource` is the seam where a real G1
  would attach — it documents the DDS topics, the RealSense pipelines and the SLAM
  subscriptions it would need, and it raises `NotImplementedError` in its constructor. It is
  not exercised here and nothing in this repo has been validated on hardware.
- **No teacher.** The simulator will happily let you measure a number that means nothing. The
  `kinematic` controller in particular is arithmetic, not locomotion; `controller_caps()`
  reports `stand_in: True` for it and you should believe it.

## Next

`workspace/README.md` — the mental model, then the six examples, then how to add a scenario and
an object, then where to take it.

Licence: Apache-2.0 (`LICENSE`, `NOTICE`). Third-party components and their licences are listed
in `THIRD-PARTY-NOTICES.md`; model asset provenance is in `models/ASSETS.md`.
