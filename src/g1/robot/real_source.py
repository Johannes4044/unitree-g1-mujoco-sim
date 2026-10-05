"""Real Unitree G1 implementation of RobotSource (skeleton, to be finished on the Jetson).

Data paths on the real robot (all open source):
  * Joint state / IMU / battery: unitree_sdk2py DDS topic `rt/lowstate` (LowState_), 500 Hz.
    Joint order in `motor_state[0..28]` is the same as BODY_JOINTS for the 29-DoF G1.
  * Arm commands: `rt/arm_sdk` (LowCmd_) with the arm_sdk enable flag (motor 29 weight).
  * Revo 2 hands: unitree `brainco_hand_service` (serial -> DDS) topics
    `rt/brainco/left/hand/cmd|state`, `rt/brainco/right/hand/cmd|state` (6 floats 0..1 each).
  * D435i / D455: `pyrealsense2` pipelines on the Jetson (serial numbers select the device),
    colour 848x480@30, depth aligned to colour, depth scale from the device.
  * Mid-360: livox_ros_driver2 -> ROS 2 `/livox/lidar` (PointCloud2) and `/livox/imu`;
    map-frame pose + registered cloud from the SLAM node (FAST-LIO2 or point_lio: `/Odometry`,
    `/cloud_registered`). Use `rclpy` to subscribe.
  * Head dual-DoF module: Unitree's servo topic (to be confirmed with Unitree; see email).

Fill in the TODOs, run `ROBOT_SOURCE=real python scripts/monitor_server.py` on the Jetson
or on a laptop on the 192.168.123.x network. Everything above the source (server, UI) is unchanged.
"""
from __future__ import annotations

import time

import numpy as np

from g1.robot.robot_source import RobotSource, RobotState


class RealSource(RobotSource):
    name = "real"

    def __init__(self, iface: str = "eth0"):
        self.iface = iface
        self._joints = {}
        self._joint_temp = {}   # deg C by joint name; stays empty until the TODO in _connect is done
        self._rgb = {}
        self._depth = {}
        self._cloud = None
        self._pose = (np.zeros(3, np.float32), np.array([1, 0, 0, 0], np.float32))
        self._connect()

    def _connect(self):
        # TODO(jetson): unitree_sdk2py.core.channel.ChannelFactoryInitialize(0, self.iface)
        # TODO(jetson): ChannelSubscriber("rt/lowstate", LowState_).Init(self._on_lowstate, 10)
        # TODO(jetson, milestone 5): OVER-TEMPERATURE ALARM INPUT. In _on_lowstate fill
        #   self._joint_temp[BODY_JOINTS[i]] from the per-motor temperature of motor_state[i]
        #   (field name, units and which winding/driver sensor to use: confirm on the robot, do not
        #   guess). Until then state() publishes an empty joint_temp and the UI shows
        #   "no temperature data" with the over-temp alarm disarmed.
        # TODO(jetson, milestone 5): MEASURED BASE VELOCITY. The G1 low state carries a base
        #   velocity, and the SLAM node's /Odometry carries twist.twist.{linear,angular}. Fill
        #   self._base_vel = ([vx, vy, vz], [wx, wy, wz]) in the BODY frame from one of them
        #   (which field, which frame and whether it is world or body: confirm on the robot, do
        #   not guess - a wrong frame here is a silently wrong training column) and return it
        #   from state() as base_lin_vel/base_ang_vel. Until then state() leaves both None and
        #   the bridge finite-differences the pose instead (see `Bridge._base_velocity`).
        # TODO(jetson): subscribers for rt/brainco/{left,right}/hand/state
        # TODO(jetson): pyrealsense2 pipelines for D435i and D455 (by serial number)
        # TODO(jetson): rclpy node subscribing /cloud_registered + /Odometry from the SLAM node
        raise NotImplementedError("RealSource is a skeleton; implement the TODOs on the robot network")

    def state(self) -> RobotState:
        return RobotState(t=time.time(), joints=dict(self._joints), joint_temp=dict(self._joint_temp),
                          base_pos=self._pose[0].tolist(),
                          base_quat=self._pose[1].tolist(), mode="real")

    def rgb(self, camera):
        return self._rgb.get(camera)

    def depth(self, camera):
        return self._depth.get(camera)

    def lidar(self):
        if self._cloud is None:
            return None
        return self._cloud, self._pose[0], self._pose[1]

    def set_joint_targets(self, targets):
        # TODO(jetson): publish rt/arm_sdk for arm joints, brainco cmd for hand joints.
        # Must be gated behind an explicit "armed" flag in the UI. Never move legs/waist here.
        raise NotImplementedError

    def set_head(self, pan=None, tilt=None):
        raise NotImplementedError
