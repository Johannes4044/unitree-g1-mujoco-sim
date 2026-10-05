"""Render the pick-and-place scene from every camera, incl. head RGB + depth.
Also runs a short physics rollout and writes a video from the head camera and
third-person camera side by side, to prove the sim + cameras work together.

The rollout video is an animated WebP (Pillow/libwebp, permissive) rather than an
mp4: see scripts/webp_clip.py for why.

    python scripts/render_scene.py                                  # models/g1_pick_place.xml
    python scripts/render_scene.py models/scenarios/house_rooms.xml

Paths are resolved relative to this checkout, so nothing here needs Docker. SIM_OUT
overrides the output directory (default: `out/` at the repo root).
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # so `webp_clip` imports from any cwd

import imageio
import mujoco
import numpy as np
from webp_clip import write_webp

XML = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "models", "g1_pick_place.xml")
OUT = os.environ.get("SIM_OUT", os.path.join(ROOT, "out"))
os.makedirs(OUT, exist_ok=True)

m = mujoco.MjModel.from_xml_path(XML)
d = mujoco.MjData(m)
kid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "stand")
mujoco.mj_resetDataKeyframe(m, d, kid)
mujoco.mj_forward(m, d)
print(f"{XML}: nq={m.nq} nu={m.nu} ncam={m.ncam}")

W, H = 848, 480
r = mujoco.Renderer(m, height=H, width=W)

def rgb(cam):
    r.update_scene(d, camera=cam)
    return r.render().copy()

def depth(cam):
    r.enable_depth_rendering()
    r.update_scene(d, camera=cam)
    z = r.render().copy()
    r.disable_depth_rendering()
    return z

# detail views: hands, head, back (free camera)
def detail(name, lookat, dist, az, el):
    cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = lookat; cam.distance = dist; cam.azimuth = az; cam.elevation = el
    r.update_scene(d, camera=cam)
    return r.render().copy()
bid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
rw, lw = d.xpos[bid("right_wrist_yaw_link")], d.xpos[bid("left_wrist_yaw_link")]
tiles = [detail("hand_r", rw + [0.05, 0, -0.08], 0.45, 160, -20), detail("hand_r2", rw + [0.05, 0, -0.08], 0.45, 60, -10),
         detail("hand_l", lw + [0.05, 0, -0.08], 0.45, -160, -20), detail("hand_l2", lw + [0.05, 0, -0.08], 0.45, -60, -10)]
imageio.imwrite(f"{OUT}/detail_hands.png", np.concatenate(tiles, axis=1))
tiles = [detail("head", [0.02, 0, 1.2], 0.7, 160, -5), detail("head_side", [0.0, 0, 1.15], 0.8, 90, 0),
         detail("back", [-0.05, 0, 1.0], 1.2, 0, -10), detail("back2", [-0.05, 0, 1.0], 1.2, 40, -20)]
imageio.imwrite(f"{OUT}/detail_head_back.png", np.concatenate(tiles, axis=1))
print("wrote detail_hands.png, detail_head_back.png")

# stills from each camera
for i in range(m.ncam):
    name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_CAMERA, i)
    img = rgb(name)
    imageio.imwrite(f"{OUT}/cam_{name}.png", img)
    print(f"wrote cam_{name}.png")

# depth from head camera, saved as 16-bit mm (like a RealSense) and as a preview png
z = depth("head_cam_depth")
z_mm = np.clip(z * 1000, 0, 65535).astype(np.uint16)
imageio.imwrite(f"{OUT}/cam_head_cam_depth_mm.png", z_mm)
zn = np.clip((z - 0.2) / (3.0 - 0.2), 0, 1)
imageio.imwrite(f"{OUT}/cam_head_cam_depth_preview.png", (255 * (1 - zn)).astype(np.uint8))
print(f"head depth: min {z.min():.2f} m, max(finite<10) {z[z<10].max():.2f} m")

# Rollout: hold stand pose, and swing the right arm forward/up toward the table
# by ramping a few actuator targets. Records head + third-person video.
act = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i): i for i in range(m.nu)}
targets = {  # actuator name: target angle [rad]; unknown names are skipped
    "right_shoulder_pitch_joint": -1.2,
    "right_shoulder_roll_joint": -0.2,
    "right_elbow_joint": 0.6,
    "right_wrist_pitch_joint": -0.3,
    # Dex3
    "right_hand_thumb_0_joint": -0.5,
    "right_hand_thumb_1_joint": 0.5,
    "right_hand_index_0_joint": -0.8,
    "right_hand_middle_0_joint": -0.8,
    # Revo 2
    "right_thumb_metacarpal_joint": 1.2,
    "right_thumb_proximal_joint": 0.8,
    "right_index_proximal_joint": 1.2,
    "right_middle_proximal_joint": 1.2,
    "right_ring_proximal_joint": 1.2,
    "right_pinky_proximal_joint": 1.2,
}
targets = {k: v for k, v in targets.items() if k in act}
ctrl0 = d.ctrl.copy()
fps, secs = 30, 4
steps_per_frame = int(round(1 / fps / m.opt.timestep))
frames = []
for f in range(fps * secs):
    a = min(1.0, f / (fps * 2.0))
    for n, t in targets.items():
        i = act[n]
        lo, hi = m.actuator_ctrlrange[i]
        d.ctrl[i] = (1 - a) * ctrl0[i] + a * float(np.clip(t, lo, hi))
    for _ in range(steps_per_frame):
        mujoco.mj_step(m, d)
    left = rgb("head_cam")
    right = rgb("third_person")
    frames.append(np.concatenate([left, right], axis=1))
clip = write_webp(frames, f"{OUT}/scene_rollout.webp", fps=fps)
print(f"wrote {clip} ({len(frames)} frames, {os.path.getsize(clip) / 1e6:.1f} MB)")
imageio.imwrite(f"{OUT}/scene_rollout_last.png", frames[-1])
pel = d.qpos[2]
print(f"rollout done: {len(frames)} frames, pelvis height {pel:.3f} m, right palm at "
      f"{d.xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'right_wrist_yaw_link')].round(3)}")
for nm in ("red_box", "blue_can", "green_box", "yellow_ball"):
    print(f"  {nm}: {d.xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, nm)].round(3)}")
r.close()
print("OK")
