"""Write a rollout as an animated WebP.

Why not mp4: an mp4 needs an ffmpeg binary, and every redistributable one we could
reach (Debian's `ffmpeg`, the `imageio-ffmpeg` wheel) is a GPL build. This
this repository ship permissive-only, so the videos are animated WebP written by
Pillow instead — libwebp is BSD-3-Clause and is already inside the Pillow wheels.
Every current browser plays an animated WebP in an `<img>`.

Frames are HxWx3 uint8 arrays (what `mujoco.Renderer.render()` returns).
"""
from __future__ import annotations

import os
from pathlib import Path

QUALITY = 60   # lossy WebP quality; a rollout is a diagnostic, not a master
METHOD = 4     # libwebp effort, 0 fast .. 6 small


def webp_supported() -> bool:
    """Can this Pillow write WebP? Its wheels bundle libwebp; a source build may lack it."""
    try:
        from PIL import features
    except ImportError:
        return False
    return bool(features.check("webp"))


def write_webp(frames, path, fps: float = 30.0) -> str:
    """Write `frames` to `path` as one looping animated WebP. Returns the path written.

    Atomic: a temporary file in the same directory, fsynced, then renamed, so a reader
    (or a half-finished run) never sees a truncated animation.
    """
    from PIL import Image

    if not frames:
        raise ValueError("no frames")
    if not webp_supported():
        raise RuntimeError(
            "this Pillow cannot write WebP (no libwebp). The wheels on PyPI bundle it; "
            "a Pillow built from source without libwebp does not. Reinstall from a wheel.")
    target = Path(path)
    images = [f if isinstance(f, Image.Image) else Image.fromarray(f) for f in frames]
    tmp = target.with_name(f".{target.name}.tmp")
    try:
        with open(tmp, "wb") as f:
            images[0].save(f, format="WEBP", save_all=True, append_images=images[1:],
                           duration=round(1000 / fps), loop=0, quality=QUALITY, method=METHOD)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, target)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
    return str(target)
