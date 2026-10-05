"""Robot data-source interface shared by the simulator and the real G1.

The monitor server only talks to this interface, so switching between sim and
robot is a config change (ROBOT_SOURCE=sim|real), not a code change.

Frames: `pose` fields are world/map frame (x forward, y left, z up, metres, quaternion wxyz).
Cameras: names are hardware names, identical on both sides:
    d435i_rgb, d435i_depth   internal RealSense D435i behind the visor
    d455_rgb,  d455_depth    RealSense D455 on the 2-DoF head module
    third_person             sim only (None on the real robot)
Depth images are float32 metres, HxW. RGB is uint8 HxWx3.
LiDAR: Nx3 float32 points in the map frame plus the sensor pose (what a SLAM node outputs).
"""
from __future__ import annotations

import abc
from dataclasses import dataclass, field

import numpy as np

CAMERAS = ["d435i_rgb", "d455_rgb", "third_person"]
DEPTHS = {"d435i_rgb": "d435i_depth", "d455_rgb": "d455_depth"}

BODY_JOINTS = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint",
    "left_ankle_pitch_joint", "left_ankle_roll_joint", "right_hip_pitch_joint", "right_hip_roll_joint",
    "right_hip_yaw_joint", "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint",
    "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint",
    "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]
HAND_JOINTS = [f"{s}_{j}_joint" for s in ("left", "right") for j in
               ("thumb_metacarpal", "thumb_proximal", "index_proximal", "middle_proximal", "ring_proximal", "pinky_proximal")]
HEAD_JOINTS = ["head_pan_joint", "head_tilt_joint"]


@dataclass
class RobotState:
    t: float
    joints: dict[str, float]                       # position [rad] by joint name (body + hands + head module)
    joint_vel: dict[str, float] = field(default_factory=dict)
    joint_torque: dict[str, float] = field(default_factory=dict)
    # motor temperature [deg C] by joint name. Empty = this source has no temperature data
    # (the UI then says so and does not arm the over-temperature alarm); never fill with zeros.
    joint_temp: dict[str, float] = field(default_factory=dict)
    base_pos: list[float] = field(default_factory=lambda: [0, 0, 0])
    base_quat: list[float] = field(default_factory=lambda: [1, 0, 0, 0])
    # Base velocity as the source *measures* it: body frame [vx, vy, vz] m/s and [wx, wy, wz]
    # rad/s. None means this source does not estimate it - the bridge then finite-differences
    # consecutive poses instead (see `Bridge._base_velocity`). Never fill these with zeros to
    # mean "unknown": zero is a measurement, and it ends up in the training data as one.
    base_lin_vel: list[float] | None = None
    base_ang_vel: list[float] | None = None
    imu_gyro: list[float] = field(default_factory=lambda: [0, 0, 0])
    imu_acc: list[float] = field(default_factory=lambda: [0, 0, 0])
    battery: float | None = None
    mode: str = "sim"
    # what moves the base: "none", "kinematic" (the sim's mocap stand-in) or the name of a whole-body
    # controller from g1.wbc ("holosoma", "pd_stand"); extra["controller_caps"] says what it can do
    controller: str = "none"
    extra: dict = field(default_factory=dict)


class RobotSource(abc.ABC):
    """Implementations must be thread-safe: the bridge calls state/rgb/depth/lidar from
    several telemetry threads without a lock while commands arrive on another thread."""

    name = "abstract"

    @abc.abstractmethod
    def state(self) -> RobotState: ...

    @abc.abstractmethod
    def rgb(self, camera: str) -> np.ndarray | None: ...

    @abc.abstractmethod
    def depth(self, camera: str) -> np.ndarray | None: ...

    @abc.abstractmethod
    def lidar(self) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """(points_map Nx3 float32, sensor_pos 3, sensor_quat_wxyz 4)"""

    # --- commands (optional; real robot must gate these behind an explicit arm/enable) ---
    def set_joint_targets(self, targets: dict[str, float]) -> None:
        raise NotImplementedError

    def set_head(self, pan: float | None = None, tilt: float | None = None) -> None:
        raise NotImplementedError

    # --- locomotion (optional) ---------------------------------------------
    def set_base_velocity(self, vx: float, vy: float, wz: float) -> None:
        """Body-frame velocity command [m/s, m/s, rad/s]. Sim: kinematic base. Real: WBC."""
        raise NotImplementedError

    def set_mode(self, mode: str) -> None:
        raise NotImplementedError

    def set_posture(self, base_height: float, torso_pitch: float) -> dict | None:
        """Pelvis height [m] and torso pitch [rad, forward positive]; only for a posture-capable
        controller. May return the values actually applied after the controller's own clamping."""
        raise NotImplementedError

    def controller_caps(self) -> dict:
        """What the active base controller can do: {"name", "modes", "walk", "posture", ...}.
        The bridge accepts `set_posture` only when "posture" is true; the UI reads the same dict
        from `state.extra["controller_caps"]`."""
        return {"name": "none", "modes": [], "walk": False, "posture": False, "teleport": False, "stand_in": False}

    def close(self) -> None:
        pass
