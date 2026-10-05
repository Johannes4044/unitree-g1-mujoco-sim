"""03 - Read the cameras: RGB and depth from the head D455 and the visor D435i.

Run it:

    uv run python workspace/03_read_cameras.py

What to take away:

* `RobotSource.rgb(name)` returns uint8 HxWx3 and `RobotSource.depth(name)` returns
  float32 HxW *in metres*. Same names, same units, same dtypes on the real robot - the
  camera names are hardware names, not sim names:

      d435i_rgb / d435i_depth   the RealSense D435i behind the visor
      d455_rgb  / d455_depth    the RealSense D455 on the 2-DoF pan/tilt head module
      third_person              sim only; a real source returns None

  You ask for depth by the *colour* camera's name (`depth("d455_rgb")`); the source maps it
  to the matching depth camera, which is the aligned-depth convention RealSense gives you.
* Render resolution is a constructor argument (`cam_w`, `cam_h`), not a property of the
  model. The MJCF declares 848x480 to match the real sensor; rendering at 424x240 is four
  times cheaper and is what the recording pipeline actually uses.
* Depth has no "invalid" sentinel here: sky and far geometry come back as the renderer's far
  plane, a large positive number. Gate it yourself before feeding it to anything.
* Moving the head moves the D455 and only the D455 - that is the point of the pan/tilt module.

MuJoCo may print "ARB_clip_control unavailable while mjDEPTH_ZEROFAR requested" on some GL
backends. It is a precision warning about the depth buffer, not an error; the metres you get
back are still the right metres for anything at normal scene distances.
"""
from __future__ import annotations

import sys
from pathlib import Path

import imageio.v3 as iio
import numpy as np

from g1.robot.robot_source import CAMERAS, DEPTHS
from g1.robot.sim_source import SimSource, default_model_path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "workspace" / "out"


def scene_path() -> str:
    if len(sys.argv) > 1:
        return sys.argv[1]
    local = REPO / "models" / "g1_pick_place.xml"
    return str(local) if local.exists() else default_model_path()


def depth_to_png(d: np.ndarray, near: float = 0.2, far: float = 4.0) -> np.ndarray:
    """A look-at-able 8-bit image from metric depth: near = white, far = black."""
    z = np.clip(d, near, far)
    return (255 * (1 - (z - near) / (far - near))).astype(np.uint8)


def report(sim: SimSource, cam: str, tag: str) -> None:
    rgb = sim.rgb(cam)
    depth = sim.depth(cam)                  # DEPTHS maps "d455_rgb" -> "d455_depth"
    iio.imwrite(OUT / f"03_{tag}_rgb.png", rgb)
    iio.imwrite(OUT / f"03_{tag}_depth.png", depth_to_png(depth))

    # Everything inside 10 m is real geometry in these scenes; beyond that is the far plane.
    scene = depth[depth < 10.0]
    print(f"{cam}")
    print(f"  rgb   {rgb.shape} {rgb.dtype}   mean brightness {rgb.mean():.1f}")
    print(f"  depth {depth.shape} {depth.dtype}  "
          f"range {depth.min():.3f} .. {depth.max():.3f} m")
    print(f"  depth on scene geometry (<10 m): {scene.min():.3f} .. {scene.max():.3f} m, "
          f"{100 * scene.size / depth.size:.0f}% of pixels")
    h, w = depth.shape
    print(f"  centre pixel is {depth[h // 2, w // 2]:.3f} m away")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    # 424x240 is the rate-friendly size the recording pipeline uses; raise it if you are
    # generating training images and can afford the render cost.
    sim = SimSource(xml=scene_path(), realtime=False, cam_w=424, cam_h=240)
    try:
        sim.step(0.5)
        print(f"cameras this source offers: {CAMERAS}")
        print(f"depth aliases: {DEPTHS}\n")

        report(sim, "d455_rgb", "d455")
        print()
        report(sim, "d435i_rgb", "d435i")

        # --- point the head down at the table --------------------------------------------
        # `set_head(pan, tilt)` is `set_joint_targets` for the two head-module joints. Only the
        # D455 rides on it; the D435i is fixed behind the visor. (The `stand` keyframe already
        # starts the head tilted down at the work surface, so level it first to see the effect.)
        sim.set_head(pan=0.0, tilt=0.0)      # level
        sim.step(1.5)
        level = sim.depth("d455_rgb")
        iio.imwrite(OUT / "03_d455_level_rgb.png", sim.rgb("d455_rgb"))
        s_level = sim.state()

        sim.set_head(pan=0.0, tilt=0.8)      # tilt down, radians (positive = down)
        sim.step(1.5)
        down = sim.depth("d455_rgb")
        iio.imwrite(OUT / "03_d455_down_rgb.png", sim.rgb("d455_rgb"))
        s_down = sim.state()

        h, w = down.shape
        print("\nhead module (the D455 rides on it, the D435i does not):")
        print(f"  level: tilt={s_level.joints['head_tilt_joint']:+.3f} rad  "
              f"centre depth {level[h // 2, w // 2]:7.3f} m  nearest {level.min():.3f} m")
        print(f"  down : tilt={s_down.joints['head_tilt_joint']:+.3f} rad  "
              f"centre depth {down[h // 2, w // 2]:7.3f} m  nearest {down.min():.3f} m")
        print("  (level, the centre ray runs off across the floor plane; tilted down it lands "
              "on the table)")
        print(f"  d435i centre depth is the same either way: "
              f"{sim.depth('d435i_rgb')[h // 2, w // 2]:.3f} m")

        print(f"\nwrote 6 PNGs to {OUT}")
    finally:
        sim.close()


if __name__ == "__main__":
    main()
