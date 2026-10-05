"""Livox Mid-360 simulation: unproject a 4-camera depth rig into a point cloud.

Mid-360 facts used: 360 deg horizontal, -7..+52 deg vertical (mounted upside down on
the G1 crown -> 7 deg up .. 52 deg down), ~200k points/s, 40 m range at 10 % reflectivity.
A 10 Hz frame therefore has ~20k points; we sample that many from the rig.
"""
import math

import mujoco
import numpy as np

CAMS = ["mid360_cam0", "mid360_cam1", "mid360_cam2", "mid360_cam3"]
V_FOV_DEG = (-52.0, 7.0)   # after the upside-down mount, relative to the horizontal plane
# Range gate, Euclidean from the sensor - not depth along a camera axis. A corner pixel of a
# 90 deg camera is 1.7x further away than its depth value says, so gating on depth returned
# points out to 60 m from a 40 m sensor: a floor carpet half as wide again as the Mid-360 can
# see, which the occupancy grid then believed and cleared as free space.
MAX_RANGE = 40.0           # Mid-360 at 10 % reflectivity
MIN_RANGE = 0.1            # its blind zone; closer "returns" are the robot's own crown, not geometry


class LidarSim:
    def __init__(self, model, renderer, points_per_frame=20000, noise_std=0.01):
        self.m = model
        self.r = renderer            # square renderer, e.g. mujoco.Renderer(m, 256, 256)
        self.n = points_per_frame
        self.noise = noise_std
        h, w = renderer.height, renderer.width
        fovy = math.radians(model.cam_fovy[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, CAMS[0])])
        fy = (h / 2) / math.tan(fovy / 2)
        fx = fy
        ys, xs = np.mgrid[0:h, 0:w]
        # ray directions in camera frame (mujoco camera looks along -z, +y up, +x right)
        self.dirs = np.stack([(xs + 0.5 - w / 2) / fx, -(ys + 0.5 - h / 2) / fy, -np.ones_like(xs, dtype=float)], axis=-1)
        self.dirs = self.dirs.reshape(-1, 3)
        # |dir| converts a depth value into a range along that ray (1 on the axis, 1.7 in a corner)
        self.dir_norm = np.linalg.norm(self.dirs, axis=1)
        self.site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "mid360_site")

    def scan(self, data):
        """Return (points_world Nx3 float32, points_lidar Nx3, lidar_pos, lidar_quat)."""
        self.r.enable_depth_rendering()
        pts = []
        for name in CAMS:
            self.r.update_scene(data, camera=name)
            z = self.r.render().reshape(-1)                # distance along -z (depth buffer, metric)
            cid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_CAMERA, name)
            R = data.cam_xmat[cid].reshape(3, 3)
            p = data.cam_xpos[cid]
            valid = (z > 0.05) & (z * self.dir_norm < MAX_RANGE)   # range, not depth
            d = self.dirs[valid] * z[valid, None]           # camera-frame coordinates (x, y, -z scaled by depth)
            pts.append(d @ R.T + p)
        self.r.disable_depth_rendering()
        pw = np.concatenate(pts, axis=0)
        # keep the Mid-360 vertical band (relative to the horizontal plane through the sensor)
        lp = data.site_xpos[self.site]
        rel = pw - lp
        horiz = np.linalg.norm(rel[:, :2], axis=1)
        elev = np.degrees(np.arctan2(rel[:, 2], horiz))
        # exact range gate about the sensor itself (the four cameras sit a few mm off it)
        rng = np.linalg.norm(rel, axis=1)
        keep = ((elev >= V_FOV_DEG[0]) & (elev <= V_FOV_DEG[1])
                & (rng >= MIN_RANGE) & (rng <= MAX_RANGE))
        pw = pw[keep]
        if len(pw) > self.n:
            pw = pw[np.random.choice(len(pw), self.n, replace=False)]
        if self.noise > 0:
            pw = pw + np.random.normal(0, self.noise, pw.shape)
        Rl = data.site_xmat[self.site].reshape(3, 3)
        pl = (pw - lp) @ Rl                                 # world -> lidar frame
        q = np.zeros(4)
        mujoco.mju_mat2Quat(q, Rl.flatten())
        return pw.astype(np.float32), pl.astype(np.float32), lp.astype(np.float32), q.astype(np.float32)


class OccupancyMap:
    """Minimal 2D occupancy grid accumulated from world-frame points. Stand-in for LiDAR SLAM:
    in sim the pose is ground truth; on the real robot the pose comes from the SLAM node."""

    def __init__(self, size_m=12.0, res=0.05, z_range=(0.1, 1.8)):
        self.res = res
        self.n = int(size_m / res)
        self.origin = -size_m / 2
        self.grid = np.zeros((self.n, self.n), dtype=np.uint16)
        self.z_range = z_range

    def add(self, pw):
        sel = (pw[:, 2] > self.z_range[0]) & (pw[:, 2] < self.z_range[1])
        ij = ((pw[sel, :2] - self.origin) / self.res).astype(int)
        ok = (ij[:, 0] >= 0) & (ij[:, 0] < self.n) & (ij[:, 1] >= 0) & (ij[:, 1] < self.n)
        np.add.at(self.grid, (ij[ok, 1], ij[ok, 0]), 1)

    def image(self):
        g = np.clip(self.grid.astype(float) / 20.0, 0, 1)
        img = (255 * (1 - g)).astype(np.uint8)
        return np.flipud(img)   # +y up
