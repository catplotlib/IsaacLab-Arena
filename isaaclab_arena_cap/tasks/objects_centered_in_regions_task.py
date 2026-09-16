# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""A success task requiring released objects to rest centered and settled in regions."""

from __future__ import annotations

import math
import torch
from collections.abc import Sequence

import warp as wp
from isaaclab.managers import ManagerTermBase, TerminationTermCfg

from isaaclab_arena.assets.asset import Asset
from isaaclab_arena.assets.register import register_task
from isaaclab_arena.tasks.objects_in_regions_task import (
    Bounds,
    ObjectsInRegionsTask,
    TerminationsCfg,
    points_in_oriented_regions,
)


def _as_torch(value) -> torch.Tensor:
    """Return a torch view of a warp array or torch tensor."""
    if isinstance(value, torch.Tensor):
        return value
    if hasattr(value, "torch"):
        return value.torch
    return wp.to_torch(value)


class objects_centered_in_regions_and_settled(ManagerTermBase):
    """Require released objects to remain centered and still for a dwell window."""

    def __init__(self, cfg: TerminationTermCfg, env) -> None:
        super().__init__(cfg, env)
        self.object_names = tuple(cfg.params["object_names"])
        self.region_names = tuple(cfg.params["region_names"])
        self.objects = tuple(env.scene[name] for name in self.object_names)
        self.regions = tuple(env.scene[name] for name in self.region_names)
        self.robot_asset_name = cfg.params["robot_asset_name"]
        self.robot = env.scene[self.robot_asset_name]
        self.gripper_joint_name = cfg.params["gripper_joint_name"]
        try:
            self.gripper_joint_index = self.robot.data.joint_names.index(self.gripper_joint_name)
        except ValueError as error:
            raise ValueError(f"expected gripper joint {self.gripper_joint_name!r} was not found") from error
        self.consecutive_success_count = torch.zeros(env.num_envs, device=env.device, dtype=torch.int32)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Reset the dwell counter for the selected environments."""
        if env_ids is None:
            env_ids = slice(None)
        self.consecutive_success_count[env_ids] = 0

    def __call__(
        self,
        env,
        object_names: list[str],
        region_names: list[str],
        bounds_xyzxyz: list[Bounds],
        linear_velocity_threshold: float,
        angular_velocity_threshold: float,
        robot_asset_name: str,
        gripper_joint_name: str,
        gripper_open_position_threshold: float,
        consecutive_success_steps: int,
    ) -> torch.Tensor:
        """Return true after every object is released, centered, and settled."""
        assert tuple(object_names) == self.object_names
        assert tuple(region_names) == self.region_names
        assert robot_asset_name == self.robot_asset_name
        assert gripper_joint_name == self.gripper_joint_name

        pair_results: list[torch.Tensor] = []
        for object_instance, region_instance, bounds in zip(self.objects, self.regions, bounds_xyzxyz, strict=True):
            center_positions_w = _as_torch(object_instance.data.root_com_pos_w)
            if hasattr(region_instance, "data"):
                region_positions_w = _as_torch(region_instance.data.root_pos_w)
                region_quaternions_w = _as_torch(region_instance.data.root_quat_w)
            else:
                region_positions_w, region_quaternions_w = (
                    _as_torch(value) for value in region_instance.get_world_poses()
                )
            bounds_tensor = torch.as_tensor(
                bounds,
                dtype=center_positions_w.dtype,
                device=center_positions_w.device,
            )
            centered = points_in_oriented_regions(
                center_positions_w,
                region_positions_w,
                region_quaternions_w,
                bounds_tensor.unsqueeze(0).expand(env.num_envs, -1),
            )
            linear_speed = torch.linalg.vector_norm(_as_torch(object_instance.data.root_lin_vel_w), dim=-1)
            angular_speed = torch.linalg.vector_norm(_as_torch(object_instance.data.root_ang_vel_w), dim=-1)
            pair_results.append(
                centered & (linear_speed <= linear_velocity_threshold) & (angular_speed <= angular_velocity_threshold)
            )

        gripper_position = _as_torch(self.robot.data.joint_pos)[:, self.gripper_joint_index]
        gripper_open = torch.abs(gripper_position) <= gripper_open_position_threshold
        success_now = torch.stack(pair_results, dim=0).all(dim=0) & gripper_open
        self.consecutive_success_count = torch.where(
            success_now,
            self.consecutive_success_count + 1,
            torch.zeros_like(self.consecutive_success_count),
        )
        return self.consecutive_success_count >= consecutive_success_steps


@register_task
class ObjectsCenteredInRegionsTask(ObjectsInRegionsTask):
    """Require every object's center of mass to rest, settled, inside its paired region."""

    def __init__(
        self,
        object_list: list[Asset],
        region_list: list[Asset],
        bounds_xyzxyz: list[Bounds],
        linear_velocity_threshold: float = 0.01,
        angular_velocity_threshold: float = 0.05,
        robot_asset_name: str = "robot",
        gripper_joint_name: str = "left_driver_joint",
        gripper_open_position_threshold: float = 0.1,
        consecutive_success_steps: int = 10,
        episode_length_s: float = 480.0,
        task_description: str | None = None,
    ) -> None:
        for name, value in (
            ("linear_velocity_threshold", linear_velocity_threshold),
            ("angular_velocity_threshold", angular_velocity_threshold),
            ("gripper_open_position_threshold", gripper_open_position_threshold),
        ):
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be a positive finite number")
        if not robot_asset_name or not gripper_joint_name:
            raise ValueError("robot asset and gripper joint names must be non-empty")
        if (
            isinstance(consecutive_success_steps, bool)
            or not isinstance(consecutive_success_steps, int)
            or consecutive_success_steps <= 0
        ):
            raise ValueError("consecutive_success_steps must be a positive integer")
        self.linear_velocity_threshold = float(linear_velocity_threshold)
        self.angular_velocity_threshold = float(angular_velocity_threshold)
        self.robot_asset_name = robot_asset_name
        self.gripper_joint_name = gripper_joint_name
        self.gripper_open_position_threshold = float(gripper_open_position_threshold)
        self.consecutive_success_steps = consecutive_success_steps
        super().__init__(
            object_list=object_list,
            region_list=region_list,
            bounds_xyzxyz=bounds_xyzxyz,
            episode_length_s=episode_length_s,
            task_description=task_description
            or "Place every object's center of mass inside its paired region and let it settle.",
        )

    def make_termination_cfg(self) -> TerminationsCfg:
        return TerminationsCfg(
            success=TerminationTermCfg(
                func=objects_centered_in_regions_and_settled,
                params={
                    "object_names": [object_.name for object_ in self.objects],
                    "region_names": [region.name for region in self.regions],
                    "bounds_xyzxyz": self.bounds,
                    "linear_velocity_threshold": self.linear_velocity_threshold,
                    "angular_velocity_threshold": self.angular_velocity_threshold,
                    "robot_asset_name": self.robot_asset_name,
                    "gripper_joint_name": self.gripper_joint_name,
                    "gripper_open_position_threshold": self.gripper_open_position_threshold,
                    "consecutive_success_steps": self.consecutive_success_steps,
                },
            )
        )
