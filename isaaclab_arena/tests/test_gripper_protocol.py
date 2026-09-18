# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for embodiment-owned behavioral grippers."""

import torch
from types import SimpleNamespace

import pytest

from isaaclab_arena.embodiments.embodiment_base import EmbodimentBase
from isaaclab_arena.embodiments.gripper import PandaGripper, RobotiqGripper
from isaaclab_arena.environments.arena_world import ArenaWorld
from isaaclab_arena.tasks.predicates.gripper import gripper_released
from isaaclab_arena.tasks.predicates.spatial import gripper_distance_from_object_exceeds_threshold


def _make_world() -> ArenaWorld:
    joint_positions = torch.tensor([[0.01, 0.02, 0.0], [0.04, 0.04, 0.8]])
    articulation_data = SimpleNamespace(
        joint_names=["panda_finger_joint1", "panda_finger_joint2", "left_driver_joint"],
        joint_pos=SimpleNamespace(torch=joint_positions),
        body_names=["robotiq_base"],
        body_link_pose_w=SimpleNamespace(
            torch=torch.tensor([[[0.2, 0.1, 0.3, 0.0, 0.0, 0.0, 1.0]], [[0.4, 0.2, 0.5, 0.0, 0.0, 0.0, 1.0]]])
        ),
    )
    target_positions = torch.tensor([
        [[0.1, 0.0, 0.0], [0.1, 0.02, 0.0], [0.1, -0.02, 0.0]],
        [[0.3, 0.0, 0.0], [0.3, 0.05, 0.0], [0.3, -0.05, 0.0]],
    ])
    sensor_data = SimpleNamespace(
        target_frame_names=["end_effector", "tool_leftfinger", "tool_rightfinger"],
        target_pos_w=SimpleNamespace(torch=target_positions),
    )
    object_poses = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
    scene = SimpleNamespace(
        num_envs=2,
        articulations={"robot": SimpleNamespace(data=articulation_data)},
        rigid_objects={
            "object": SimpleNamespace(data=SimpleNamespace(root_pose_w=SimpleNamespace(torch=object_poses)))
        },
        sensors={"ee_frame": SimpleNamespace(data=sensor_data)},
        extras={},
    )
    return ArenaWorld(scene)


def test_panda_gripper_implements_parallel_jaw_interface() -> None:
    world = _make_world()
    gripper = PandaGripper()

    torch.testing.assert_close(gripper.get_jaw_gap_m(world), torch.tensor([0.03, 0.08]))
    torch.testing.assert_close(gripper.get_opening_width_m(world), torch.tensor([0.03, 0.08]))
    torch.testing.assert_close(gripper.get_position_w(world), torch.tensor([[0.1, 0.0, 0.0], [0.3, 0.0, 0.0]]))


def test_robotiq_gripper_measures_tracked_finger_pad_gap() -> None:
    world = _make_world()
    gripper = RobotiqGripper()

    torch.testing.assert_close(gripper.get_jaw_gap_m(world), torch.tensor([0.04, 0.10]))
    torch.testing.assert_close(gripper.get_position_w(world), torch.tensor([[0.1, 0.0, 0.0], [0.3, 0.0, 0.0]]))


def test_robotiq_gripper_encapsulates_driver_joint_and_body_point_details() -> None:
    world = _make_world()
    gripper = RobotiqGripper(
        driver_joint_name="left_driver_joint",
        body_name="robotiq_base",
        body_point_offset_xyz=(0.0, 0.0, 0.157),
    )

    jaw_gap_m = gripper.get_jaw_gap_m(world)
    assert jaw_gap_m[0] > 0.084
    assert jaw_gap_m[1] < 0.001
    torch.testing.assert_close(
        gripper.get_position_w(world),
        torch.tensor([[0.2, 0.1, 0.457], [0.4, 0.2, 0.657]]),
    )


def test_embodiment_requires_a_supported_gripper() -> None:
    embodiment = object.__new__(EmbodimentBase)
    embodiment.name = "test"
    embodiment.gripper = None
    with pytest.raises(AssertionError, match="has no supported gripper"):
        embodiment.get_gripper()

    embodiment.gripper = PandaGripper()
    assert embodiment.get_gripper() is embodiment.gripper


def test_gripper_predicates_are_implementation_agnostic() -> None:
    env = SimpleNamespace(arena_world=_make_world())

    clears = gripper_released(env, PandaGripper(), grasp_width_m=0.035, release_clearance_m=0.004)
    away = gripper_distance_from_object_exceeds_threshold(
        env, subject_name="object", gripper=RobotiqGripper(), distance_threshold_m=0.2
    )
    assert clears.tolist() == [False, True]
    assert away.tolist() == [False, True]


def test_release_predicate_supports_multi_finger_hands() -> None:
    class ThreeFingerHand:
        def get_position_w(self, world):
            return world.get_frame_position_w("ee_frame", "end_effector")

        def get_opening_width_m(self, _world):
            return torch.tensor([0.03, 0.08])

    env = SimpleNamespace(arena_world=_make_world())
    hand = ThreeFingerHand()

    released = gripper_released(env, hand, grasp_width_m=0.035, release_clearance_m=0.004)
    assert released.tolist() == [False, True]


def test_gripper_predicates_validate_distances() -> None:
    env = SimpleNamespace(arena_world=_make_world())
    with pytest.raises(AssertionError, match="Grasp width"):
        gripper_released(env, PandaGripper(), grasp_width_m=0.0, release_clearance_m=0.001)
    with pytest.raises(AssertionError, match="clearance"):
        gripper_released(env, PandaGripper(), grasp_width_m=0.01, release_clearance_m=-0.001)
    with pytest.raises(AssertionError, match="Distance"):
        gripper_distance_from_object_exceeds_threshold(
            env, subject_name="object", gripper=PandaGripper(), distance_threshold_m=-0.1
        )


def test_gear_task_binds_release_checks_to_the_embodiment_gripper() -> None:
    from isaaclab_arena_environments.isaac_cap.gear_insertion_v2.embodiment import IndustrialFr3Robotiq2f85Embodiment
    from isaaclab_arena_environments.isaac_cap.gear_insertion_v2.task.task import GearMeshTask

    task = GearMeshTask(
        board=SimpleNamespace(name="board"),
        gear=SimpleNamespace(name="gear"),
        grasp_width_m=0.035,
        release_clearance_m=0.004,
    )
    embodiment = IndustrialFr3Robotiq2f85Embodiment()

    task.bind_embodiment(embodiment)

    params = task.get_termination_cfg().success.params
    assert params["gripper"] is embodiment.gripper
    assert "robot_asset_cfg" not in params
    assert "tcp_body_name" not in params
    assert "tcp_offset_xyz" not in params
    assert params["release_condition"](SimpleNamespace(arena_world=_make_world())).tolist() == [True, False]
