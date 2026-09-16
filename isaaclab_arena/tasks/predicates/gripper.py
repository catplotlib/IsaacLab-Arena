# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Stateless predicates for gripper release and hand withdrawal."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.utils.math import quat_apply

if TYPE_CHECKING:
    from isaaclab_arena.environments.isaaclab_arena_manager_based_env import IsaacLabArenaManagerBasedRLEnv


def parallel_jaw_gripper_released(
    env: IsaacLabArenaManagerBasedRLEnv,
    robot_name: str,
    gripper_joint_name: str,
    gripper_action_name: str,
    span_m: float,
    open_joint_m: float,
    grasp_width_m: float,
    stall_threshold_m: float = 2.0e-4,
    grasp_width_tolerance_m: float = 1.5e-3,
) -> torch.Tensor:
    """Check that a parallel-jaw gripper is not stalled at the object's grasp width.

    The jaw model assumes symmetric fingers whose joint positions increase when
    opening. A grasp requires both a measured-minus-commanded joint displacement
    above the stall threshold and a jaw gap within the grasp-width tolerance.

    Args:
        env: Environment supplying robot state and processed gripper targets.
        robot_name: Robot scene entity name.
        gripper_joint_name: Finger joint whose position measures the jaw opening.
        gripper_action_name: Gripper action term name.
            The first processed action is the commanded finger position.
        span_m: Fully open jaw gap, in meters.
        open_joint_m: Fully open finger joint position, in meters.
        grasp_width_m: Object width at the grasp, in meters.
        stall_threshold_m: Strict minimum joint displacement indicating a stall, in meters.
        grasp_width_tolerance_m: Strict maximum gap error indicating a grasp, in meters.

    Returns:
        Boolean tensor with one result per environment.
    """
    robot = env.scene[robot_name]
    joint_index = robot.data.joint_names.index(gripper_joint_name)
    measured = robot.data.joint_pos[:, joint_index]
    commanded = env.action_manager.get_term(gripper_action_name).processed_actions[:, 0]
    gap = span_m - 2.0 * (open_joint_m - measured)
    gripped = ((measured - commanded) > stall_threshold_m) & (torch.abs(gap - grasp_width_m) < grasp_width_tolerance_m)
    return ~gripped


def withdrawn(
    env: IsaacLabArenaManagerBasedRLEnv,
    subject_name: str,
    robot_name: str,
    tcp_body_name: str,
    tcp_offset_xyz_m: tuple[float, float, float],
    tcp_distance_min_m: float,
) -> torch.Tensor:
    """Check that the hand's TCP is farther than the minimum distance from an object.

    Args:
        env: Environment supplying robot state and object poses.
        subject_name: Object asset whose origin defines the TCP distance.
        robot_name: Robot scene entity name.
        tcp_body_name: Body to which the TCP is attached.
        tcp_offset_xyz_m: TCP position in the body frame, in meters.
        tcp_distance_min_m: Strict minimum distance between the TCP and object, in meters.

    Returns:
        Boolean tensor with one result per environment.
    """
    robot = env.scene[robot_name]
    body_index = robot.data.body_names.index(tcp_body_name)
    # W is the world frame; B is the TCP's parent body frame.
    t_W_B = robot.data.body_link_pos_w[:, body_index]
    q_W_B = robot.data.body_link_quat_w[:, body_index]
    tcp_position_B = t_W_B.new_tensor(tcp_offset_xyz_m).expand_as(t_W_B)
    tcp_position_W = t_W_B + quat_apply(q_W_B, tcp_position_B)
    subject_position_W = env.arena_world.get_pose_w(subject_name)[:, :3]
    return torch.linalg.vector_norm(subject_position_W - tcp_position_W, dim=-1) > tcp_distance_min_m
