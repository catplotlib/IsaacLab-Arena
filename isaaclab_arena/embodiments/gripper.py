# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Behavioral gripper interfaces and robot-specific implementations."""

from __future__ import annotations

import torch
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from isaaclab_arena.environments.arena_world import ArenaWorld


@runtime_checkable
class Gripper(Protocol):
    """Gripper state exposed to embodiment-agnostic tasks."""

    def get_position_w(self, world: ArenaWorld) -> torch.Tensor:
        """Return the gripper position in world frame with shape ``(num_envs, 3)``."""
        ...

    def get_opening_width_m(self, world: ArenaWorld) -> torch.Tensor:
        """Return the gripper opening width in meters with shape ``(num_envs,)``."""
        ...


@runtime_checkable
class ParallelJawGripper(Gripper, Protocol):
    """Gripper whose grasp state is represented by a physical jaw gap."""

    def get_jaw_gap_m(self, world: ArenaWorld) -> torch.Tensor:
        """Return the physical jaw gap in meters with shape ``(num_envs,)``."""
        ...

    def get_opening_width_m(self, world: ArenaWorld) -> torch.Tensor:
        """Return the jaw gap as the gripper opening width."""
        return self.get_jaw_gap_m(world)


@dataclass(frozen=True, kw_only=True)
class PandaGripper(ParallelJawGripper):
    """Franka Panda parallel-jaw gripper."""

    asset_name: str = "robot"
    """Scene key of the Franka articulation."""

    left_finger_joint_name: str = "panda_finger_joint1"
    """Joint measuring the left finger's distance from the centerline."""

    right_finger_joint_name: str = "panda_finger_joint2"
    """Joint measuring the right finger's distance from the centerline."""

    frame_transformer_name: str = "ee_frame"
    """Scene key of the gripper frame transformer."""

    target_frame_name: str = "end_effector"
    """Frame-transformer target representing the gripper center."""

    def get_jaw_gap_m(self, world: ArenaWorld) -> torch.Tensor:
        """Return the sum of the two prismatic finger positions."""
        left = world.get_joint_position(self.asset_name, self.left_finger_joint_name)
        right = world.get_joint_position(self.asset_name, self.right_finger_joint_name)
        return left + right

    def get_position_w(self, world: ArenaWorld) -> torch.Tensor:
        """Return the configured Franka grasp-frame position."""
        return world.get_frame_position_w(self.frame_transformer_name, self.target_frame_name)


@dataclass(frozen=True, kw_only=True)
class RobotiqGripper(ParallelJawGripper):
    """Robotiq 2F-85 gripper backed by tracked pads or its driver joint."""

    asset_name: str = "robot"
    """Scene key of the robot articulation."""

    driver_joint_name: str | None = None
    """Robotiq driver joint, or None to measure the tracked finger pads."""

    body_name: str | None = None
    """Robotiq base body, or None to use the frame-transformer target."""

    body_point_offset_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Gripper point offset expressed in the configured body frame."""

    frame_transformer_name: str = "ee_frame"
    """Scene key of the Robotiq frame transformer."""

    target_frame_name: str = "end_effector"
    """Frame-transformer target representing the gripper center."""

    left_finger_frame_name: str = "tool_leftfinger"
    """Frame-transformer target on the left finger pad."""

    right_finger_frame_name: str = "tool_rightfinger"
    """Frame-transformer target on the right finger pad."""

    def get_jaw_gap_m(self, world: ArenaWorld) -> torch.Tensor:
        """Return the physical distance between the two finger pads."""
        if self.driver_joint_name is not None:
            driver_position = world.get_joint_position(self.asset_name, self.driver_joint_name)
            # Robotiq's published 2F-85 linkage relation maps its driver angle to
            # the total inner-finger opening: 85 mm at zero and 0 mm near 0.8 rad.
            jaw_gap_m = 0.1143 * torch.sin(0.715 - driver_position) + 0.01
            return torch.clamp(jaw_gap_m, min=0.0, max=0.085)
        left = world.get_frame_position_w(self.frame_transformer_name, self.left_finger_frame_name)
        right = world.get_frame_position_w(self.frame_transformer_name, self.right_finger_frame_name)
        return torch.linalg.vector_norm(left - right, dim=-1)

    def get_position_w(self, world: ArenaWorld) -> torch.Tensor:
        """Return the configured Robotiq grasp-frame position."""
        if self.body_name is not None:
            return world.get_body_point_position_w(self.asset_name, self.body_name, self.body_point_offset_xyz)
        return world.get_frame_position_w(self.frame_transformer_name, self.target_frame_name)


__all__ = ["Gripper", "PandaGripper", "ParallelJawGripper", "RobotiqGripper"]
