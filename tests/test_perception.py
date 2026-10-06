"""Cameras and LiDAR: shapes, dtypes, units, and that the head module moves the right camera.

The unit contract is the one worth the most here. `depth()` returns float32 **metres**. The
RealSense SDK hands you uint16 millimetres, so the day `RealSource` is finished this is the
conversion that will be forgotten, and a policy trained on metres fed millimetres sees every
obstacle a thousand times too far away and drives into it. The tests below would catch a
factor of 1000 instantly, which is the point.

Rendering is the expensive part of this repo (~15 ms per camera on software GL, ~60 ms for a
LiDAR sweep), so the sweeps over every camera and the head-tilt comparison are kept small
(64x48) and the multi-camera sweep is marked slow.
"""
from __future__ import annotations

import numpy as np
import pytest

from g1.robot.robot_source import CAMERAS, DEPTHS


@pytest.fixture
def sim(make_sim):
    """A settled simulator at a small render size. 0.5 s is enough for the keyframe pose to
    be a contact-resolved pose; it is not meant to be a converged one."""
    src = make_sim(cam_w=64, cam_h=48)
    src.step(0.5)
    return src


# --- RGB ---------------------------------------------------------------------------------
def test_rgb_shape_follows_the_constructor_not_the_model(sim, make_sim):
    """The MJCF declares 848x480 to match the real sensors; `cam_w`/`cam_h` is what you get.
    Someone who assumes the model's resolution writes a dataloader that silently crops."""
    assert sim.rgb("d455_rgb").shape == (48, 64, 3)
    other = make_sim(cam_w=256, cam_h=144)
    assert other.rgb("d455_rgb").shape == (144, 256, 3)


def test_rgb_is_uint8_and_not_blank(sim):
    for cam in CAMERAS:
        img = sim.rgb(cam)
        assert img.dtype == np.uint8, cam
        # A renderer with no GL context, or a camera pointing into a wall, gives a constant
        # image. Any real view of these scenes has more than a handful of distinct values.
        assert len(np.unique(img)) > 8, f"{cam} rendered an almost constant image"


def test_an_unknown_camera_is_none_not_an_exception(sim):
    assert sim.rgb("left_eye") is None
    assert sim.depth("left_eye") is None


# --- depth, in metres --------------------------------------------------------------------
def test_depth_is_float32_metres(sim):
    """The visor D435i looks at the work table from ~0.7 m. Bounds 0.05 .. 20 m: in
    millimetres the same pixels would read 50 .. 20000, so this is a 1000x guard, not a
    tolerance on geometry. The upper bound is deliberately loose - it only has to exclude
    millimetres, and the far plane is tested separately."""
    for colour, depth_cam in DEPTHS.items():
        d = sim.depth(colour)
        assert d is not None, colour
        assert d.dtype == np.float32, colour
        assert d.shape == (48, 64), colour
        assert np.isfinite(d).all(), colour
        assert d.min() > 0.05, f"{colour} ({depth_cam}) nearest {d.min()} m"
        assert d.min() < 20.0, f"{colour} ({depth_cam}) nearest {d.min()} m - millimetres?"


def test_depth_has_no_invalid_sentinel_only_a_far_plane(sim):
    """Documented in `workspace/03_read_cameras.py`: sky and far geometry come back as the
    renderer's far plane, a large positive number - there is no NaN and no zero to filter on.
    A consumer that waits for a sentinel waits forever."""
    d = sim.depth("third_person")
    assert np.isfinite(d).all()
    assert (d > 0).all()
    assert d.max() > 50.0, "the far plane stopped being a large positive number"


def test_depth_is_asked_for_by_the_colour_camera_name(sim):
    """The aligned-depth convention. Asking by the depth camera's own name also works, but
    the documented call is the colour name - both must resolve to the same image."""
    by_colour = sim.depth("d455_rgb")
    by_depth = sim.depth("d455_depth")
    assert np.array_equal(by_colour, by_depth)


# --- the head module ---------------------------------------------------------------------
def test_tilting_the_head_moves_the_d455_and_not_the_d435i(sim):
    """The measured behaviour from `workspace/03_read_cameras.py`, and the whole reason the
    2-DoF module exists: the D455 rides on it, the D435i is fixed behind the visor.

    Measured at 424x240: level, the D455 centre ray runs off across the floor (21.7 m);
    tilted 0.8 rad down it lands on the table (0.78 m). The D435i centre stays at 0.681 m to
    within 1e-4 m. The assertions are written as ratios and orders of magnitude rather than
    those exact values, because the centre pixel of a 64x48 render is a different ray from
    the centre pixel of a 424x240 one.
    """
    def centre(cam):
        d = sim.depth(cam)
        h, w = d.shape
        return float(d[h // 2, w // 2])

    sim.set_head(pan=0.0, tilt=0.0)
    sim.step(1.5)
    level_455, level_435 = centre("d455_rgb"), centre("d435i_rgb")

    sim.set_head(pan=0.0, tilt=0.8)
    sim.step(1.5)
    down_455, down_435 = centre("d455_rgb"), centre("d435i_rgb")

    assert sim.state().joints["head_tilt_joint"] > 0.6, "the head did not reach the target"
    # The tilted D455 is looking at a table edge ~1 m away; level it sees the far floor.
    assert down_455 < 2.0, f"tilted D455 centre {down_455:.3f} m is not near geometry"
    assert level_455 > 3 * down_455, "tilting the head did not change the D455 centre depth"
    # 1 mm: the chest camera is rigidly mounted, so the only thing that can move this is the
    # robot's own settling between the two reads.
    assert down_435 == pytest.approx(level_435, abs=1e-3), "the D435i followed the head"


def test_panning_the_head_moves_the_d455_view(sim):
    sim.set_head(pan=0.0, tilt=0.6)
    sim.step(1.5)
    straight = sim.rgb("d455_rgb").astype(float)
    sim.set_head(pan=0.9, tilt=0.6)
    sim.step(1.5)
    panned = sim.rgb("d455_rgb").astype(float)
    assert np.abs(straight - panned).mean() > 1.0, "panning the head changed nothing"


# --- LiDAR -------------------------------------------------------------------------------
def test_lidar_returns_map_frame_points_and_a_sensor_pose(sim):
    """"what a SLAM node outputs": the cloud is already in the map (world) frame, and the
    sensor pose comes with it. A consumer that transforms it again gets a doubled rotation."""
    points, pos, quat = sim.lidar()
    assert points.shape == (15000, 3)
    assert points.dtype == np.float32
    assert pos.shape == (3,) and quat.shape == (4,)
    # the Mid-360 sits on the crown, ~1.26 m up on a robot whose pelvis is at 0.79 m
    assert 1.0 < float(pos[2]) < 1.6
    assert abs(float(np.linalg.norm(quat)) - 1.0) < 1e-5
    # map frame, so the floor returns sit near z=0 rather than near the sensor
    assert float(points[:, 2].min()) < 0.2


def test_lidar_sees_the_floor_and_the_robots_own_surroundings(sim):
    points, pos, _ = sim.lidar()
    rel = points - pos
    assert float(np.abs(rel[:, 0]).max()) > 1.0        # something ahead or behind
    assert float(np.abs(rel[:, 1]).max()) > 1.0        # something to the sides


def test_the_occupancy_map_accumulates_hits(sim):
    """`OccupancyMap` is the stand-in for LiDAR SLAM used by example 04."""
    from g1.robot.lidar import OccupancyMap

    grid = OccupancyMap(size_m=12.0, res=0.05)
    assert grid.grid.shape == (240, 240)
    points, _, _ = sim.lidar()
    grid.add(points)
    assert grid.grid.sum() > 0
    image = grid.image()
    assert image.shape == (240, 240) and image.dtype == np.uint8
    assert image.min() < 255, "every cell came back empty"


@pytest.mark.slow
def test_every_camera_renders_in_every_scenario(make_sim, scenario_models):
    """Five scenarios x three colour cameras x two depth cameras. Slow because rendering is
    what costs: the render sweep `scripts/render_scene.py` does, as an assertion."""
    for name, path in scenario_models.items():
        sim = make_sim(xml=path, cam_w=64, cam_h=48)
        sim.step(0.5)
        for cam in CAMERAS:
            img = sim.rgb(cam)
            assert img is not None and img.shape == (48, 64, 3), f"{name}/{cam}"
            assert len(np.unique(img)) > 8, f"{name}/{cam} rendered an almost constant image"
        for colour in DEPTHS:
            d = sim.depth(colour)
            assert d is not None and d.dtype == np.float32, f"{name}/{colour}"
            assert np.isfinite(d).all() and (d > 0).all(), f"{name}/{colour}"
