# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Released, settled syringe containment using Arena's managed predicates."""

from __future__ import annotations

import json
import math
import torch
from dataclasses import MISSING

import isaaclab.envs.mdp as mdp
from isaaclab.managers import SceneEntityCfg, TerminationTermCfg
from isaaclab.utils.configclass import configclass
from isaaclab.utils.math import quat_apply_inverse

from isaaclab_arena.assets.asset import Asset
from isaaclab_arena.metrics.success_rate import SuccessRateMetric
from isaaclab_arena.tasks.predicates.composite import CompositePredicate
from isaaclab_arena.tasks.predicates.spatial import velocity_below_threshold
from isaaclab_arena.tasks.task_base import TaskBase
from isaaclab_arena.tasks.terminations import SuccessMode


def center_of_mass_in_region(env, object_name: str, region_name: str, bounds: tuple[float, ...]) -> torch.Tensor:
    """Check the object's center of mass against inclusive receiver-local bounds."""
    # ArenaWorld's geometry centroid is not the mass center required by CAP scoring.
    center_W = env.scene[object_name].data.root_com_pos_w.torch
    T_W_R = env.arena_world.get_pose_w(region_name)
    center_R = quat_apply_inverse(T_W_R[:, 3:], center_W - T_W_R[:, :3])
    limits = torch.as_tensor(bounds, device=center_R.device, dtype=center_R.dtype)
    return ((center_R >= limits[:3]) & (center_R <= limits[3:])).all(dim=-1)


def gripper_is_open(env, robot_cfg: SceneEntityCfg, threshold: float) -> torch.Tensor:
    """Require the Robotiq driver to be within the open-position tolerance."""
    positions = env.scene[robot_cfg.name].data.joint_pos.torch[:, robot_cfg.joint_ids]
    return (positions.abs() <= threshold).all(dim=-1)


def cap_episode_finished(env, object_names: list[str], region_names: list[str]):
    """End a disconnected CAP episode after settling; never count it as success."""
    finished = getattr(env, "cap_episode_finished", False)
    if finished:
        # Evaluation diagnostics stay on the environment side of the bridge.
        success = env.termination_manager.get_term_cfg("success").func
        robot = env.scene["robot"]
        driver = list(robot.joint_names).index("left_driver_joint")
        state = {
            "predicate_results": success.results.cpu().tolist(),
            "consecutive_success_steps": success.consecutive_true_steps.cpu().tolist(),
            "gripper_position": robot.data.joint_pos.torch[:, driver].cpu().tolist(),
        }
        for name in dict.fromkeys([*object_names, *region_names]):
            data = env.scene[name].data
            state[name] = {
                "center_w": data.root_com_pos_w.torch.cpu().tolist(),
                "root_pose_w": data.root_pose_w.torch.cpu().tolist(),
                "linear_speed": torch.linalg.vector_norm(data.root_lin_vel_w.torch, dim=-1).cpu().tolist(),
                "angular_speed": torch.linalg.vector_norm(data.root_ang_vel_w.torch, dim=-1).cpu().tolist(),
            }
        print("[SyringeSort] Final state: " + json.dumps(state), flush=True)
    return torch.full((env.num_envs,), finished, device=env.device, dtype=torch.bool)


@configclass
class TerminationsCfg:
    """Episode timeout and managed syringe success."""

    time_out: TerminationTermCfg = TerminationTermCfg(func=mdp.time_out, time_out=True)
    success: TerminationTermCfg = MISSING
    cap_finished: TerminationTermCfg = MISSING


# TODO(alexmillane) [berley-specific-tasks]: Generalize this task to work with any object and region,
# and move it into core Arena code.
class SyringeSortTask(TaskBase):
    """Require released syringes to remain settled inside their disposal regions."""

    def __init__(
        self,
        object_list: list[Asset],
        region_list: list[Asset],
        bounds_xyzxyz: list[tuple[float, ...]],
        linear_velocity_threshold: float = 0.01,
        angular_velocity_threshold: float = 0.05,
        gripper_open_position_threshold: float = 0.1,
        consecutive_success_steps: int = 50,
        episode_length_s: float = 228.0,
        task_description: str | None = None,
    ):
        assert object_list and len(object_list) == len(region_list) == len(bounds_xyzxyz)
        assert isinstance(consecutive_success_steps, int) and not isinstance(consecutive_success_steps, bool)
        assert consecutive_success_steps > 0
        for value in (
            linear_velocity_threshold,
            angular_velocity_threshold,
            gripper_open_position_threshold,
            episode_length_s,
        ):
            assert math.isfinite(value) and value > 0
        for bounds in bounds_xyzxyz:
            assert len(bounds) == 6 and all(math.isfinite(value) for value in bounds)
            assert all(lo <= hi for lo, hi in zip(bounds[:3], bounds[3:], strict=True))
        super().__init__(episode_length_s=episode_length_s, task_description=task_description)
        self.objects = object_list
        self.regions = region_list
        self.consecutive_success_steps = consecutive_success_steps
        predicates = [
            TerminationTermCfg(
                func=gripper_is_open,
                params={
                    "robot_cfg": SceneEntityCfg("robot", joint_names=["left_driver_joint"]),
                    "threshold": gripper_open_position_threshold,
                },
            )
        ]
        for obj, region, bounds in zip(object_list, region_list, bounds_xyzxyz, strict=True):
            predicates.extend([
                TerminationTermCfg(
                    func=center_of_mass_in_region,
                    params={"object_name": obj.name, "region_name": region.name, "bounds": tuple(bounds)},
                ),
                TerminationTermCfg(
                    func=velocity_below_threshold,
                    params={
                        "subject_name": obj.name,
                        "linear_velocity_threshold": linear_velocity_threshold,
                        "angular_velocity_threshold": angular_velocity_threshold,
                    },
                ),
            ])
        self.termination_cfg = TerminationsCfg(
            cap_finished=TerminationTermCfg(
                func=cap_episode_finished,
                params={
                    "object_names": [obj.name for obj in object_list],
                    "region_names": [obj.name for obj in region_list],
                },
            ),
            success=TerminationTermCfg(
                func=CompositePredicate,
                params={
                    "predicates": predicates,
                    "mode": SuccessMode.ALL,
                    "consecutive_steps": consecutive_success_steps,
                },
            ),
        )

    def get_scene_cfg(self):
        return None

    def get_termination_cfg(self):
        return self.termination_cfg

    def get_events_cfg(self):
        return None

    def get_mimic_env_cfg(self, arm_mode):
        return None

    def get_metrics(self):
        return [SuccessRateMetric()]
