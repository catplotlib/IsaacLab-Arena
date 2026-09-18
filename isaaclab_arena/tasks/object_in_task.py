# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Require a settled object to fit entirely inside a target AABB."""

from __future__ import annotations

import math

from isaaclab.managers import TerminationTermCfg

from isaaclab_arena.assets.asset import Asset
from isaaclab_arena.assets.register import register_task
from isaaclab_arena.metrics.success_rate import SuccessRateMetric
from isaaclab_arena.progress_tracking.progress_objective import ProgressObjective
from isaaclab_arena.tasks.predicates.composite import CompositePredicate
from isaaclab_arena.tasks.predicates.spatial import object_in_target_aabb, velocity_below_threshold
from isaaclab_arena.tasks.task_base import TaskBase
from isaaclab_arena.tasks.task_termination_cfg import TaskTerminationCfg
from isaaclab_arena.tasks.terminations import SuccessMode


@register_task
class ObjectInTask(TaskBase):
    """Require one object to remain settled inside its target AABB."""

    def __init__(
        self,
        object: Asset,
        target: Asset,
        linear_velocity_threshold: float = 0.01,
        angular_velocity_threshold: float = 0.05,
        consecutive_success_steps: int = 50,
        episode_length_s: float = 228.0,
        task_description: str | None = None,
    ):
        """Configure one object and target with settled AABB containment.

        Args:
            object: Object to deposit.
            target: Container receiving the object.
            linear_velocity_threshold: Maximum settled linear speed in meters per second.
            angular_velocity_threshold: Maximum settled angular speed in radians per second.
            consecutive_success_steps: Consecutive steps the object must remain contained and settled.
            episode_length_s: Episode timeout in seconds.
            task_description: Optional task description.
        """
        assert isinstance(consecutive_success_steps, int) and not isinstance(consecutive_success_steps, bool)
        assert consecutive_success_steps > 0
        for value in (
            linear_velocity_threshold,
            angular_velocity_threshold,
            episode_length_s,
        ):
            assert math.isfinite(value) and value > 0
        super().__init__(episode_length_s=episode_length_s, task_description=task_description)
        self.object = object
        self.target = target
        self.consecutive_success_steps = consecutive_success_steps
        self.linear_velocity_threshold = linear_velocity_threshold
        self.angular_velocity_threshold = angular_velocity_threshold

    def get_scene_cfg(self):
        return None

    def get_termination_cfg(self) -> TaskTerminationCfg:
        predicates = [
            TerminationTermCfg(
                func=object_in_target_aabb,
                params={"object_name": self.object.name, "target_name": self.target.name},
            ),
            TerminationTermCfg(
                func=velocity_below_threshold,
                params={
                    "subject_name": self.object.name,
                    "linear_velocity_threshold": self.linear_velocity_threshold,
                    "angular_velocity_threshold": self.angular_velocity_threshold,
                },
            ),
        ]
        settled_in_target = TerminationTermCfg(
            func=CompositePredicate,
            params={
                "predicates": predicates,
                "mode": SuccessMode.ALL,
                "consecutive_steps": self.consecutive_success_steps,
            },
        )
        return TaskTerminationCfg(
            timeout_s=self.episode_length_s,
            success=[ProgressObjective(name="object_in", predicate_sequence=[settled_in_target])],
        )

    def get_events_cfg(self):
        return None

    def get_mimic_env_cfg(self, arm_mode):
        return None

    def get_metrics(self):
        return [SuccessRateMetric()]
