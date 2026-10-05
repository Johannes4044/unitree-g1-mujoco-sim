"""04 - LiDAR: a Mid-360 scan, and a top-down occupancy image built from it.

Run it:

    uv run python scripts/build_scenarios.py          # once, to get the richer worlds
    uv run python workspace/04_lidar_and_a_map.py     # prefers house_rooms if it is built
    uv run python workspace/04_lidar_and_a_map.py models/scenarios/warehouse_aisle.xml

What to take away:

* `RobotSource.lidar()` returns `(points, sensor_pos, sensor_quat)`: an Nx3 float32 array of
  points **in the map frame**, plus the sensor's pose. That is deliberately what a SLAM node
  publishes, not what a raw driver publishes - in sim the pose is ground truth, on the robot
  it comes from FAST-LIO2 (or point_lio) and the points come registered. Swapping the source
  therefore does not change the shape of the data a navigation stack consumes.
* The sim does not ray-cast. It renders four 192x192 depth cameras on the crown and
  unprojects them, then gates the result to the real Livox Mid-360 envelope: 360 deg around,
  -52..+7 deg vertically after the upside-down crown mount, 0.1 .. 40 m range, ~15k points
  per frame with 1 cm of noise. See `g1/robot/lidar.py` - the constants are the datasheet's.
* `OccupancyMap` in the same module is a deliberately minimal 2D grid: drop the floor and the
  ceiling, count hits per cell. It is a stand-in for a real mapper, good enough to see whether
  your LiDAR data is sane, and the exact thing you would replace with a proper mapping stack.
"""
from __future__ import annotations

import sys
from pathlib import Path

import imageio.v3 as iio
import numpy as np

from g1.robot.lidar import MAX_RANGE, MIN_RANGE, V_FOV_DEG, OccupancyMap
from g1.robot.sim_source import SimSource, default_model_path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "workspace" / "out"


def scene_path() -> str:
    if len(sys.argv) > 1:
        return sys.argv[1]
    # A room with walls makes a far more interesting map than the bare table scene.
    for p in (REPO / "models" / "scenarios" / "house_rooms.xml",
              REPO / "models" / "g1_pick_place.xml"):
        if p.exists():
            return str(p)
    return default_model_path()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = scene_path()
    print(f"scene: {path}")
    sim = SimSource(xml=path, realtime=False)
    try:
        sim.step(0.5)

        # --- one scan ---------------------------------------------------------------------
        points, pos, quat = sim.lidar()
        rel = points - pos
        rng = np.linalg.norm(rel, axis=1)
        elev = np.degrees(np.arctan2(rel[:, 2], np.linalg.norm(rel[:, :2], axis=1)))

        print(f"\none Mid-360 frame: {len(points)} points, {points.dtype}, map frame")
        print(f"  sensor at      {np.round(pos, 3).tolist()} m, quat {np.round(quat, 3).tolist()}")
        print(f"  measured range {rng.min():.2f} .. {rng.max():.2f} m "
              f"(sensor envelope {MIN_RANGE} .. {MAX_RANGE} m)")
        print(f"  median range   {np.median(rng):.2f} m")
        # The envelope is enforced before the 1 cm of range noise is added, so the measured
        # extremes spill a little past it - most at short range, where 1 cm is a big angle.
        print(f"  elevation      {elev.min():+.1f} .. {elev.max():+.1f} deg "
              f"(gate {V_FOV_DEG[0]:+.0f} .. {V_FOV_DEG[1]:+.0f} deg, applied before noise)")
        print(f"  height band    z {points[:, 2].min():.2f} .. {points[:, 2].max():.2f} m")

        # The raw cloud is useful on its own: save it so you can load it elsewhere.
        np.save(OUT / "04_cloud.npy", points)

        # --- accumulate a map while turning on the spot -----------------------------------
        # A single scan from one pose leaves shadows behind every object. Spinning the base
        # does not help here (the sensor is on the robot and the shadows turn with it), so
        # drive forward a little between scans as well. With the kinematic stand-in this is
        # free: set_mode("walk") then set_base_velocity - see example 05 for what that means.
        grid = OccupancyMap(size_m=16.0, res=0.05, z_range=(0.15, 1.8))
        grid.add(points)

        poses = []
        if sim.controller_caps()["walk"]:
            sim.set_mode("walk")
            for vx, wz, seconds in [(0.0, 0.8, 2.0), (0.3, 0.0, 2.0), (0.0, -0.8, 2.0)]:
                # Velocity commands expire after SimSource.DEADMAN_S without a refresh - the
                # same dead-man the real velocity interface has - so re-send inside the loop.
                for _ in range(int(seconds / 0.1)):
                    sim.set_base_velocity(vx, 0.0, wz)
                    sim.step(0.1)
                sim.set_base_velocity(0.0, 0.0, 0.0)
                sim.step(0.2)
                pts, p, _ = sim.lidar()
                grid.add(pts)
                poses.append([round(float(v), 2) for v in p])
            start = [round(float(v), 2) for v in pos]
            print(f"\nsensor positions scanned from: {[start] + poses}")

        img = grid.image()      # uint8, white = free/unseen, dark = many hits
        iio.imwrite(OUT / "04_occupancy.png", img)
        hit = int((grid.grid > 0).sum())
        print(f"\noccupancy grid {grid.grid.shape} at {grid.res * 100:.0f} cm/cell: "
              f"{hit} cells hit ({100 * hit / grid.grid.size:.1f}% of the map)")
        print(f"  a cell is {grid.res} m; the grid spans "
              f"{grid.origin:.1f} .. {grid.origin + grid.n * grid.res:.1f} m in x and y")
        print(f"\nwrote {OUT / '04_occupancy.png'} and {OUT / '04_cloud.npy'}")
    finally:
        sim.close()


if __name__ == "__main__":
    main()
