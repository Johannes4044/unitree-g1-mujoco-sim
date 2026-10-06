"""Smallest possible proof that MuJoCo can open a GL context and render.

Reads MUJOCO_GL from the environment (MuJoCo validates it at *import* time, see
.github/actions/headless-gl) and renders one 64x64 frame of a trivial scene.
Exit 0 means the backend named in MUJOCO_GL works on this machine.

Used by the headless-gl composite action to pick a backend on Linux runners.
Nothing in the repo depends on this file; it exists only for CI.
"""
import os
import sys

import mujoco

XML = """
<mujoco>
  <worldbody>
    <light pos="0 0 3"/>
    <geom type="plane" size="1 1 .1"/>
    <body pos="0 0 .3"><geom type="sphere" size=".1" rgba="1 0 0 1"/></body>
  </worldbody>
</mujoco>"""

backend = os.environ.get("MUJOCO_GL", "<unset>")
model = mujoco.MjModel.from_xml_string(XML)
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)

renderer = mujoco.Renderer(model, 64, 64)
renderer.update_scene(data)
img = renderer.render()
renderer.close()

if img.shape != (64, 64, 3):
    sys.exit(f"MUJOCO_GL={backend}: unexpected frame shape {img.shape}")
if img.max() == img.min():
    sys.exit(f"MUJOCO_GL={backend}: frame is uniform ({img.max()}) - context is a no-op")

print(f"MUJOCO_GL={backend}: rendered {img.shape}, mean={img.mean():.1f}")
