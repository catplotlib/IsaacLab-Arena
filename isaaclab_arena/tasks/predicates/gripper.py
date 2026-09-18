# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Stateless predicates for parallel-jaw grippers."""

from __future__ import annotations

import math
import torch
from typing import TYPE_CHECKING

from isaaclab_arena.embodiments.end_effector import ParallelJawGripper

if TYPE_CHECKING:
    from isaaclab_arena.environments.isaaclab_arena_manager_based_env import IsaacLabArenaManagerBasedRLEnv


def parallel_jaw_gripper_released(
    env: IsaacLabArenaManagerBasedRLEnv,
    gripper: ParallelJawGripper,
    grasp_width_m: float,
    release_clearance_m: float = 1.5e-3,
) -> torch.Tensor:
    """Check that the measured jaw gap clears the object's grasp width.

    This checks current clearance, not whether a grasp happened earlier. An open
    gripper can pass before grasping; an opening command alone cannot make it pass.

    Args:
        env: Environment supplying measured joint positions through ArenaWorld.
        gripper: Embodiment-owned parallel-jaw gripper implementation.
        grasp_width_m: Object width at the grasp, in meters.
        release_clearance_m: Required extra gap beyond the object width, in meters.
            The comparison is strict, so exactly this clearance does not pass.

    Returns:
        Boolean tensor with one result per environment.
    """
    assert math.isfinite(grasp_width_m) and grasp_width_m > 0.0, "Grasp width must be positive and finite."
    assert (
        math.isfinite(release_clearance_m) and release_clearance_m >= 0.0
    ), "Release clearance must be non-negative and finite."
    jaw_gap_m = gripper.get_jaw_gap_m(env.arena_world)
    return jaw_gap_m > grasp_width_m + release_clearance_m
