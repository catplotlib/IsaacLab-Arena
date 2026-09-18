# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Stateless predicates for parallel-jaw grippers."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Any

from isaaclab_arena.embodiments.end_effector import EndEffector

if TYPE_CHECKING:
    from isaaclab_arena.environments.isaaclab_arena_manager_based_env import IsaacLabArenaManagerBasedRLEnv


def end_effector_released(
    env: IsaacLabArenaManagerBasedRLEnv,
    end_effector: EndEffector,
    **release_params: Any,
) -> torch.Tensor:
    """Check whether an end effector satisfies its release condition.

    The end-effector implementation interprets the release parameters, allowing
    parallel jaws, dexterous hands, and vacuum tools to use different semantics.

    Args:
        env: Environment supplying live scene state through ArenaWorld.
        end_effector: Embodiment-owned end-effector implementation.
        release_params: Parameters interpreted by the concrete end effector.

    Returns:
        Boolean tensor with one result per environment.
    """
    return end_effector.is_released(env.arena_world, **release_params)
