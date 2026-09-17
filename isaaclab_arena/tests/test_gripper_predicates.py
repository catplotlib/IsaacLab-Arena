# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Check release and withdrawal independently of any task or robot asset."""

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app


def _test_parallel_jaw_gripper_released(_simulation_app) -> bool:
    import torch
    from types import SimpleNamespace

    from isaaclab_arena.tasks.predicates.gripper import parallel_jaw_gripper_released

    release_params = dict(
        robot_name="robot",
        gripper_joint_name="finger",
        gripper_action_name="gripper",
        span_m=0.125,
        open_joint_m=0.0625,
        grasp_width_m=0.03125,
    )
    for device in ("cpu", "cuda:0"):
        for dtype in (torch.float32, torch.float64):
            # Open, stalled on the object, at target, wrong width, and exact thresholds.
            measured = torch.tensor(
                [[0.0625], [0.015625], [0.015625], [0.03125], [0.015625], [0.017578125]],
                device=device,
                dtype=dtype,
            )
            commanded = torch.tensor(
                [[0.0], [0.0], [0.015625], [0.0], [0.013671875], [0.0]], device=device, dtype=dtype
            )
            robot = SimpleNamespace(data=SimpleNamespace(joint_names=["finger"], joint_pos=measured))
            action = SimpleNamespace(processed_actions=commanded)
            env = SimpleNamespace(
                scene={"robot": robot},
                action_manager=SimpleNamespace(get_term=lambda _name: action),
            )
            result = parallel_jaw_gripper_released(
                env, **release_params, stall_threshold_m=0.001953125, grasp_width_tolerance_m=0.00390625
            )
            assert result.tolist() == [True, False, True, True, True, True]
            assert result.device == measured.device and result.dtype == torch.bool
            assert parallel_jaw_gripper_released(
                env, **release_params, stall_threshold_m=0.02, grasp_width_tolerance_m=0.00390625
            ).all()
            assert not parallel_jaw_gripper_released(env, **release_params, grasp_width_tolerance_m=0.01)[-1]
            assert parallel_jaw_gripper_released(env, **release_params).tolist() == [
                True,
                False,
                True,
                True,
                False,
                True,
            ]
    return True


def test_parallel_jaw_gripper_released() -> None:
    assert run_function_with_persistent_simulation_app(_test_parallel_jaw_gripper_released)


def _test_tcp_distance_from_object_exceeds_threshold(_simulation_app) -> bool:
    import math
    import torch
    from types import SimpleNamespace

    from isaaclab.utils.math import quat_from_euler_xyz

    from isaaclab_arena.tasks.predicates.gripper import tcp_distance_from_object_exceeds_threshold

    for device in ("cpu", "cuda:0"):
        for dtype in (torch.float32, torch.float64):
            zeros = torch.zeros(3, device=device, dtype=dtype)
            q_W_B = quat_from_euler_xyz(zeros, zeros, zeros)
            body_positions = torch.zeros((3, 1, 3), device=device, dtype=dtype)
            robot = SimpleNamespace(
                data=SimpleNamespace(
                    body_names=["wrist"],
                    body_link_pos_w=body_positions,
                    body_link_quat_w=q_W_B[:, None, :],
                )
            )
            subject_positions = torch.tensor([[0.125, 0, 0], [0.25, 0, 0], [0.5, 0, 0]], device=device, dtype=dtype)
            poses = torch.cat((subject_positions, q_W_B), dim=-1)
            env = SimpleNamespace(scene={"robot": robot}, arena_world=SimpleNamespace(get_pose_w=lambda _: poses))
            result = tcp_distance_from_object_exceeds_threshold(
                env,
                subject_name="object",
                robot_name="robot",
                tcp_body_name="wrist",
                tcp_offset_xyz_m=(0.0, 0.0, 0.0),
                tcp_distance_min_m=0.25,
            )
            assert result.tolist() == [False, False, True]
            assert result.device == body_positions.device and result.dtype == torch.bool

            # Rotate the local X offset onto world Y and translate the parent body.
            robot.data.body_link_quat_w = quat_from_euler_xyz(zeros, zeros, zeros + math.pi / 2)[:, None, :]
            body_positions[:] = body_positions.new_tensor([1.0, 2.0, 3.0])
            poses[:, :3] = poses.new_tensor([[1.0, 2.125, 3.0], [1.0, 2.0, 3.0], [1.0, 2.625, 3.0]])
            result = tcp_distance_from_object_exceeds_threshold(
                env,
                subject_name="object",
                robot_name="robot",
                tcp_body_name="wrist",
                tcp_offset_xyz_m=(0.125, 0.0, 0.0),
                tcp_distance_min_m=0.0625,
            )
            assert result.tolist() == [False, True, True]
    return True


def test_tcp_distance_from_object_exceeds_threshold() -> None:
    assert run_function_with_persistent_simulation_app(_test_tcp_distance_from_object_exceeds_threshold)
