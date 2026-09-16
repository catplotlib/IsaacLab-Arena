# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Stateless predicates for gripper release and hand withdrawal."""

from __future__ import annotations

import torch
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from isaaclab.utils.math import quat_apply

if TYPE_CHECKING:
    from isaaclab_arena.environments.isaaclab_arena_manager_based_env import IsaacLabArenaManagerBasedRLEnv


def released(
    env: IsaacLabArenaManagerBasedRLEnv,
    work_hand: Mapping[str, Any],
    stall_threshold: float = 2.0e-4,
    grasp_width_tolerance: float = 1.5e-3,
) -> torch.Tensor:
    """Check that a parallel-jaw gripper is not stalled at the object's grasp width.

    The jaw model assumes symmetric fingers whose joint positions increase when
    opening. A grasp requires both a measured-minus-commanded joint displacement
    above the stall threshold and a jaw gap within the grasp-width tolerance.

    Args:
        env: Environment supplying robot state and processed gripper targets.
        work_hand: Mapping with ``robot_name``, ``gripper_joint_name``,
            ``gripper_action_name``, ``span_m`` (fully open gap), ``open_joint_m``
            (fully open joint position), and ``grasp_width_m`` (object width).
            The first processed action is the commanded finger position.
        stall_threshold: Strict minimum joint displacement indicating a stall, in meters.
        grasp_width_tolerance: Strict maximum gap error indicating a grasp, in meters.

    Returns:
        Boolean tensor with one result per environment.
    """
    robot = env.scene[work_hand["robot_name"]]
    joint_index = robot.data.joint_names.index(work_hand["gripper_joint_name"])
    measured = robot.data.joint_pos[:, joint_index]
    commanded = env.action_manager.get_term(work_hand["gripper_action_name"]).processed_actions[:, 0]
    gap = work_hand["span_m"] - 2.0 * (work_hand["open_joint_m"] - measured)
    gripped = ((measured - commanded) > stall_threshold) & (
        torch.abs(gap - work_hand["grasp_width_m"]) < grasp_width_tolerance
    )
    return ~gripped


def withdrawn(
    env: IsaacLabArenaManagerBasedRLEnv,
    subject_name: str,
    work_hand: Mapping[str, Any],
    tcp_distance_min: float,
) -> torch.Tensor:
    """Check that the hand's TCP is farther than the minimum distance from an object.

    Args:
        env: Environment supplying robot state and object poses.
        subject_name: Object asset whose origin defines the TCP distance.
        work_hand: Mapping with ``robot_name``, ``tcp_body_name``, and
            ``tcp_offset_xyz`` (TCP position in the body frame, in meters).
        tcp_distance_min: Strict minimum distance between the TCP and object, in meters.

    Returns:
        Boolean tensor with one result per environment.
    """
    robot = env.scene[work_hand["robot_name"]]
    body_index = robot.data.body_names.index(work_hand["tcp_body_name"])
    # W is the world frame; B is the TCP's parent body frame.
    t_W_B = robot.data.body_link_pos_w[:, body_index]
    q_W_B = robot.data.body_link_quat_w[:, body_index]
    tcp_position_B = t_W_B.new_tensor(work_hand["tcp_offset_xyz"]).expand_as(t_W_B)
    tcp_position_W = t_W_B + quat_apply(q_W_B, tcp_position_B)
    subject_position_W = env.arena_world.get_pose_w(subject_name)[:, :3]
    return torch.linalg.vector_norm(subject_position_W - tcp_position_W, dim=-1) > tcp_distance_min
