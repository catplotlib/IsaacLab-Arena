# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Require a settled object to fit inside and contact a target."""

from __future__ import annotations

import math
from functools import partial

from isaaclab.managers import SceneEntityCfg, TerminationTermCfg

from isaaclab_arena.assets.asset import Asset
from isaaclab_arena.assets.object_type import ObjectType
from isaaclab_arena.assets.register import register_task
from isaaclab_arena.metrics.success_rate import SuccessRateMetric
from isaaclab_arena.progress_tracking.progress_objective import ProgressObjective
from isaaclab_arena.tasks.predicates.object_in import ObjectSettledInTarget
from isaaclab_arena.tasks.predicates.spatial import object_in_target_aabb
from isaaclab_arena.tasks.task_base import TaskBase
from isaaclab_arena.tasks.task_termination_cfg import TaskTerminationCfg
from isaaclab_arena.utils.configclass import make_configclass
from isaaclab_arena.utils.usd.rigid_bodies import read_asset_rigid_body_paths


@register_task
class ObjectInTask(TaskBase):
    """Require one object to remain settled inside and in contact with its target."""

    def __init__(
        self,
        object: Asset,
        target: Asset,
        linear_velocity_threshold: float = 0.01,
        angular_velocity_threshold: float = 0.05,
        consecutive_success_steps: int = 50,
        episode_length_s: float = 228.0,
        task_description: str | None = None,
        contact_force_threshold: float = 0.01,
        minimum_contained_fraction: float = 1.0,
    ):
        """Configure one object and target with settled AABB containment.

        Args:
            object: Object to deposit.
            target: Container receiving the object.
            linear_velocity_threshold: Maximum settled linear speed in meters per second.
            angular_velocity_threshold: Maximum settled angular speed in radians per second.
            consecutive_success_steps: Consecutive steps the object must remain contained, touching the target, and settled.
            episode_length_s: Episode timeout in seconds.
            task_description: Optional task description.
            contact_force_threshold: Minimum destination contact force in newtons.
            minimum_contained_fraction: Required object AABB volume fraction inside the target, in (0, 1].
        """
        assert isinstance(consecutive_success_steps, int) and not isinstance(consecutive_success_steps, bool)
        assert consecutive_success_steps > 0
        for value in (
            linear_velocity_threshold,
            angular_velocity_threshold,
            episode_length_s,
            contact_force_threshold,
        ):
            assert math.isfinite(value) and value > 0
        super().__init__(episode_length_s=episode_length_s, task_description=task_description)
        assert math.isfinite(minimum_contained_fraction) and 0 < minimum_contained_fraction <= 1
        self.minimum_contained_fraction = minimum_contained_fraction
        self.object = object
        self.target = target
        self.contact_force_threshold = contact_force_threshold
        self.contact_sensor_cfg = SceneEntityCfg(f"contact_sensor_{object.name}_in_{target.name}")
        self.consecutive_success_steps = consecutive_success_steps
        self.linear_velocity_threshold = linear_velocity_threshold
        self.angular_velocity_threshold = angular_velocity_threshold

    def get_scene_cfg(self):
        sensor_cfg = self.object.get_contact_sensor_cfg()
        if self.target.object_type == ObjectType.BASE:
            # PhysX needs a separate filter expression for each nested target body.
            body_paths = read_asset_rigid_body_paths(
                self.target.usd_path, variants=self.target.spawn_cfg_addon.get("variants")
            )
            assert body_paths, "Contact targets must contain rigid bodies."
            sensor_cfg.filter_prim_paths_expr = []
            for path in body_paths:
                relative_path = path.removeprefix("/Asset")
                sensor_cfg.filter_prim_paths_expr.append(self.target.get_prim_path() + relative_path)
        else:
            sensor_cfg = self.object.get_contact_sensor_cfg(contact_against_object=self.target)
        scene_cfg_type = make_configclass("SceneCfg", [(self.contact_sensor_cfg.name, type(sensor_cfg), sensor_cfg)])
        return scene_cfg_type()

    def get_termination_cfg(self) -> TaskTerminationCfg:
        return TaskTerminationCfg(
            timeout_s=self.episode_length_s,
            success=[
                ProgressObjective(
                    name="object_in",
                    predicate_sequence=[
                        partial(
                            object_in_target_aabb,
                            object_name=self.object.name,
                            target_name=self.target.name,
                            minimum_contained_fraction=self.minimum_contained_fraction,
                        ),
                        TerminationTermCfg(
                            func=ObjectSettledInTarget,
                            params={
                                "object_name": self.object.name,
                                "target_name": self.target.name,
                                "contact_sensor_cfg": self.contact_sensor_cfg,
                                "minimum_contained_fraction": self.minimum_contained_fraction,
                                "contact_force_threshold": self.contact_force_threshold,
                                "linear_velocity_threshold": self.linear_velocity_threshold,
                                "angular_velocity_threshold": self.angular_velocity_threshold,
                                "consecutive_steps": self.consecutive_success_steps,
                            },
                        ),
                    ],
                ),
            ],
        )

    def get_events_cfg(self):
        return None

    def get_mimic_env_cfg(self, arm_mode):
        return None

    def get_metrics(self):
        return [SuccessRateMetric()]
