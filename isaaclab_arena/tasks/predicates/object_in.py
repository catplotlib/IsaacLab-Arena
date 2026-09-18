# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Settled containment as the final predicate of an object-in progress objective."""

import torch

from isaaclab.managers import SceneEntityCfg

from isaaclab_arena.tasks.predicates.consecutive import ConsecutivePredicate
from isaaclab_arena.tasks.predicates.spatial import (
    object_in_contact_with_target,
    object_in_target_aabb,
    velocity_below_threshold,
)


class ObjectSettledInTarget(ConsecutivePredicate):
    """Require containment, target contact, and low velocity throughout a settling window."""

    def __call__(
        self,
        env,
        object_name: str,
        target_name: str,
        contact_sensor_cfg: SceneEntityCfg,
        minimum_contained_fraction: float,
        contact_force_threshold: float,
        linear_velocity_threshold: float,
        angular_velocity_threshold: float,
        consecutive_steps: int,
        active_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Construction consumes this manager parameter.
        del consecutive_steps
        contained = object_in_target_aabb(env, object_name, target_name, minimum_contained_fraction)
        touching = object_in_contact_with_target(env, contact_sensor_cfg, contact_force_threshold)
        settled = velocity_below_threshold(env, object_name, linear_velocity_threshold, angular_velocity_threshold)
        return self._update_consecutive_and_get_completion_mask(contained & touching & settled, active_mask)
