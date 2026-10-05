"""Headless MuJoCo smoke test.

1. Loads a trivial model, steps physics.
2. Loads the G1 scene, steps it, renders a PNG and a short animated WebP
   (Pillow/libwebp; no ffmpeg, see scripts/webp_clip.py).
Exit code 0 means MuJoCo works on this machine.

The scene is this checkout's vendored `models/g1_pick_place.xml`. Set MENAGERIE to a
mujoco_menagerie checkout to use its `unitree_g1/scene_with_hands.xml` instead, and
SIM_OUT to override the output directory (default: `out/` at the repo root).
"""
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # so `webp_clip` imports from any cwd

import mujoco
import numpy as np
from webp_clip import write_webp

OUT = os.environ.get("SIM_OUT", os.path.join(ROOT, "out"))
MENAGERIE = os.environ.get("MENAGERIE", "")
os.makedirs(OUT, exist_ok=True)

print(f"MuJoCo {mujoco.__version__}, MUJOCO_GL={os.environ.get('MUJOCO_GL')}")

# 1. Trivial model: a ball falling onto a plane.
xml = """
<mujoco>
  <worldbody>
    <light pos="0 0 3"/>
    <geom type="plane" size="1 1 .1"/>
    <body pos="0 0 1"><freejoint/><geom type="sphere" size=".1" rgba="1 0 0 1"/></body>
  </worldbody>
</mujoco>"""
m = mujoco.MjModel.from_xml_string(xml)
d = mujoco.MjData(m)
for _ in range(1000):
    mujoco.mj_step(m, d)
z = d.qpos[2]
print(f"[1] ball z after 1000 steps: {z:.3f} (expected ~0.1)")
assert 0.05 < z < 0.2, "physics sanity check failed"

# 2. G1 humanoid: this checkout's model, or Menagerie's scene when MENAGERIE points at one.
candidates = ([os.path.join(MENAGERIE, "unitree_g1", "scene_with_hands.xml"),
               os.path.join(MENAGERIE, "unitree_g1", "scene.xml")] if MENAGERIE else [])
candidates.append(os.path.join(ROOT, "models", "g1_pick_place.xml"))
scene = next((p for p in candidates if os.path.exists(p)), None)
if scene is None:
    sys.exit(f"[2] no G1 scene found; looked at {candidates}")
m = mujoco.MjModel.from_xml_path(scene)
d = mujoco.MjData(m)
print(f"[2] loaded {scene}: nq={m.nq} nv={m.nv} nu={m.nu} nbody={m.nbody} ngeom={m.ngeom}")

# Start from the 'stand' keyframe if present.
if m.nkey > 0:
    mujoco.mj_resetDataKeyframe(m, d, 0)
    print(f"    reset to keyframe '{mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_KEY, 0)}'")

# Hold the keyframe pose with the position actuators (if any).
if m.nu > 0:
    # map each actuator to its joint's qpos address
    for i in range(m.nu):
        j = m.actuator_trnid[i, 0]
        d.ctrl[i] = d.qpos[m.jnt_qposadr[j]]

t0 = time.time()
n_steps = 2000
for _ in range(n_steps):
    mujoco.mj_step(m, d)
dt = time.time() - t0
print(f"    {n_steps} steps in {dt:.2f}s -> {n_steps*m.opt.timestep/dt:.1f}x realtime, base height {d.qpos[2]:.3f} m")

# 3. Render.
try:
    renderer = mujoco.Renderer(m, height=480, width=640)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [0, 0, 0.8]
    cam.distance = 3.0
    cam.azimuth = 140
    cam.elevation = -15

    renderer.update_scene(d, camera=cam)
    img = renderer.render()
    import imageio
    png = os.path.join(OUT, "g1_smoke.png")
    imageio.imwrite(png, img)
    print(f"[3] rendered {png} {img.shape}")

    # short video: let it settle from the keyframe
    mujoco.mj_resetDataKeyframe(m, d, 0) if m.nkey else mujoco.mj_resetData(m, d)
    if m.nu > 0:
        for i in range(m.nu):
            j = m.actuator_trnid[i, 0]
            d.ctrl[i] = d.qpos[m.jnt_qposadr[j]]
    fps = 30
    frames = []
    for _ in range(3 * fps):
        for _ in range(int(round(1 / fps / m.opt.timestep))):
            mujoco.mj_step(m, d)
        cam.azimuth += 1
        renderer.update_scene(d, camera=cam)
        frames.append(renderer.render().copy())
    clip = write_webp(frames, os.path.join(OUT, "g1_smoke.webp"), fps=fps)
    print(f"[3] wrote {clip} ({len(frames)} frames)")
    renderer.close()
except Exception as e:  # noqa: BLE001
    print(f"[3] rendering FAILED with MUJOCO_GL={os.environ.get('MUJOCO_GL')}: {e!r}")
    sys.exit(2)

print("OK: MuJoCo simulation works on this machine.")
