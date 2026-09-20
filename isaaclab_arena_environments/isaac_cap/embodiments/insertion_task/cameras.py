# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Physical camera definitions for the industrial FR3 workcell."""

from __future__ import annotations

import math
import torch

import isaaclab.sim as sim_utils
from isaaclab.sensors import CameraCfg
from isaaclab.utils.configclass import configclass

from isaaclab_arena.utils.cameras import ArenaCameraCfg

CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720


def _ros_optical_quaternion(eye: tuple[float, float, float], target: tuple[float, float, float]):
    """Return a ROS optical camera orientation aimed from ``eye`` to ``target``."""

    from isaaclab.utils.math import create_rotation_matrix_from_view, quat_from_matrix

    eye_tensor = torch.tensor([eye], dtype=torch.float32)
    target_tensor = torch.tensor([target], dtype=torch.float32)
    opengl_to_ros = torch.diag(torch.tensor([1.0, -1.0, -1.0]))
    rotation = create_rotation_matrix_from_view(eye_tensor, target_tensor, "Z")[0] @ opengl_to_ros
    return tuple(quat_from_matrix(rotation).tolist())


_EXTERIOR_EYE = (1.25, 0.0, 1.75)
_EXTERIOR_TARGET = (0.1, 0.015, 0.82)
_EXTERNAL_EYE = (0.9, 0.75, 1.8)
_EXTERNAL_EYE_2 = (0.9, -0.75, 1.8)
_EXTERNAL_TARGET = (0.1, 0.015, 0.82)
_WRIST_EYE = (0.0, 0.12, -0.02)
_WRIST_TARGET = (0.0, 0.12, 0.25)
_DROID_PINHOLE = dict(
    focal_length=2.1,
    focus_distance=28.0,
    horizontal_aperture=5.376,
    vertical_aperture=3.024,
)
_WRIST_PRIM = (
    "{ENV_REGEX_NS}/Robot/Geometry/base/fr3_link0/fr3_link1/"
    "fr3_link2/fr3_link3/fr3_link4/fr3_link5/fr3_link6/"
    "fr3_link7/robotiq_attach/Geometry/robotiq_base/wrist_camera"
)


@configclass
class IndustrialFr3RobotiqCameraCfg(ArenaCameraCfg):
    """One wrist, two exterior, and one top camera for the FR3 workcell."""

    exterior_left_camera: CameraCfg = CameraCfg(
        prim_path="{ENV_REGEX_NS}/exterior_left_camera",
        height=CAMERA_HEIGHT,
        width=CAMERA_WIDTH,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(**_DROID_PINHOLE),
        offset=CameraCfg.OffsetCfg(
            pos=_EXTERNAL_EYE,
            rot=_ros_optical_quaternion(_EXTERNAL_EYE, _EXTERNAL_TARGET),
            convention="ros",
        ),
    )
    exterior_right_camera: CameraCfg = CameraCfg(
        prim_path="{ENV_REGEX_NS}/exterior_right_camera",
        height=CAMERA_HEIGHT,
        width=CAMERA_WIDTH,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(**_DROID_PINHOLE),
        offset=CameraCfg.OffsetCfg(
            pos=_EXTERNAL_EYE_2,
            rot=_ros_optical_quaternion(_EXTERNAL_EYE_2, _EXTERNAL_TARGET),
            convention="ros",
        ),
    )
    wrist_camera: CameraCfg = CameraCfg(
        prim_path=_WRIST_PRIM,
        # The prim rides the wrist; without this the reported pose stays at
        # its spawn value, so anything projecting depth through it lands the
        # result where the hand was at reset rather than where it is now.
        update_latest_camera_pose=True,
        height=CAMERA_HEIGHT,
        width=CAMERA_WIDTH,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=2.8,
            focus_distance=28.0,
            horizontal_aperture=5.376,
            vertical_aperture=3.024,
        ),
        offset=CameraCfg.OffsetCfg(
            pos=_WRIST_EYE,
            rot=_ros_optical_quaternion(_WRIST_EYE, _WRIST_TARGET),
            convention="ros",
        ),
    )
    top_camera: CameraCfg = CameraCfg(
        prim_path="{ENV_REGEX_NS}/top_camera",
        update_period=0.0,
        update_latest_camera_pose=True,
        height=CAMERA_HEIGHT,
        width=CAMERA_WIDTH,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=5.0,
            focus_distance=28.0,
            horizontal_aperture=5.376,
            vertical_aperture=3.024,
        ),
        offset=CameraCfg.OffsetCfg(
            pos=_EXTERIOR_EYE,
            rot=_ros_optical_quaternion(_EXTERIOR_EYE, _EXTERIOR_TARGET),
            convention="ros",
        ),
    )

    def use_overhead_profile(self, profile: str) -> None:
        """Select a task-calibrated overhead view; retain the other three cameras."""
        if profile != "tool_sorting":
            raise ValueError(f"Unknown FR3 overhead profile: {profile!r}")
        camera = self.top_camera
        camera.width, camera.height = 1280, 960
        camera.offset.pos = (0.3, 0.0, 2.5)
        camera.offset.rot = (math.sqrt(0.5), math.sqrt(0.5), 0.0, 0.0)
        camera.offset.convention = "ros"
        camera.spawn = sim_utils.PinholeCameraCfg(
            focal_length=2.1,
            focus_distance=28.0,
            horizontal_aperture=5.376,
            vertical_aperture=3.024,
            clipping_range=(0.01, 5.0),
        )


def _zed_pinhole(horizontal_fov: float, vertical_fov: float):
    """Return the calibrated ZED pinhole model used by gear insertion v2."""
    focal_length = 5.0
    return sim_utils.PinholeCameraCfg(
        distortion=sim_utils.OpenCvPinholeDistortionCfg(
            fx=CAMERA_WIDTH / (2 * math.tan(math.radians(horizontal_fov / 2))),
            fy=CAMERA_HEIGHT / (2 * math.tan(math.radians(vertical_fov / 2))),
            cx=CAMERA_WIDTH / 2,
            cy=CAMERA_HEIGHT / 2,
            image_size=(CAMERA_WIDTH, CAMERA_HEIGHT),
            apply_lens_distortion=False,
        ),
        focal_length=focal_length,
        focus_distance=28.0,
        horizontal_aperture=2 * focal_length * math.tan(math.radians(horizontal_fov / 2)),
        vertical_aperture=2 * focal_length * math.tan(math.radians(vertical_fov / 2)),
    )


@configclass
class IndustrialFr3RobotiqGearV2CameraCfg(ArenaCameraCfg):
    """Camera calibration retained by the gear-insertion-v2 embodiment."""

    _top_eye = (0.1, 0.015, 1.75)
    _left_eye = (1.3213, 0.0826, 1.5159)
    _left_target = (-0.1495, 0.0711, 1.0465)

    exterior_left_camera: CameraCfg = CameraCfg(
        prim_path="{ENV_REGEX_NS}/exterior_left_camera",
        height=CAMERA_HEIGHT,
        width=CAMERA_WIDTH,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=_zed_pinhole(110.0, 70.0),
        offset=CameraCfg.OffsetCfg(
            pos=_left_eye,
            rot=_ros_optical_quaternion(_left_eye, _left_target),
            convention="ros",
        ),
    )
    exterior_right_camera: CameraCfg = CameraCfg(
        prim_path="{ENV_REGEX_NS}/exterior_right_camera",
        height=CAMERA_HEIGHT,
        width=CAMERA_WIDTH,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=_zed_pinhole(110.0, 70.0),
        offset=CameraCfg.OffsetCfg(
            pos=_EXTERNAL_EYE_2,
            rot=_ros_optical_quaternion(_EXTERNAL_EYE_2, _EXTERNAL_TARGET),
            convention="ros",
        ),
    )
    wrist_camera: CameraCfg = CameraCfg(
        prim_path=_WRIST_PRIM,
        update_latest_camera_pose=True,
        height=CAMERA_HEIGHT,
        width=CAMERA_WIDTH,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=_zed_pinhole(102.0, 57.0),
        offset=CameraCfg.OffsetCfg(
            pos=_WRIST_EYE,
            rot=_ros_optical_quaternion(_WRIST_EYE, _WRIST_TARGET),
            convention="ros",
        ),
    )
    top_camera: CameraCfg = CameraCfg(
        prim_path="{ENV_REGEX_NS}/top_camera",
        update_period=0.0,
        update_latest_camera_pose=True,
        height=CAMERA_HEIGHT,
        width=CAMERA_WIDTH,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=_zed_pinhole(110.0, 70.0),
        offset=CameraCfg.OffsetCfg(
            pos=_top_eye,
            rot=(math.sqrt(0.5), -math.sqrt(0.5), 0.0, 0.0),
            convention="ros",
        ),
    )

    def use_overhead_profile(self, profile: str) -> None:
        """Select the calibrated overhead view used by gear insertion v2."""
        if profile != "gear":
            raise ValueError(f"Unknown gear-v2 overhead profile: {profile!r}")
        width, height = 1280, 960
        fov_y = 50.0
        camera = self.top_camera
        camera.width, camera.height = width, height
        camera.offset.pos = (0.0490017409436448, 0.01556502252117765, 1.33)
        camera.offset.rot = (1.0, 0.0, 0.0, 0.0)
        camera.spawn = sim_utils.PinholeCameraCfg(
            focal_length=3.024 / (2 * math.tan(math.radians(fov_y / 2))),
            horizontal_aperture=3.024 * width / height,
            vertical_aperture=3.024,
            clipping_range=(0.01, 4.0),
        )
