# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Validate managed predicate composition and partial-reset lifecycle."""

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app


def _test_predicate_group_lifecycle(_simulation_app) -> bool:
    import torch
    from types import SimpleNamespace

    from isaaclab.managers import TerminationManager, TerminationTermCfg

    from isaaclab_arena.tasks.predicates.consecutive import ConsecutivePredicate
    from isaaclab_arena.tasks.predicates.object_settling import (
        ObjectInitialRestPoseRecorder,
        ObjectsSettledForConsecutiveSteps,
    )
    from isaaclab_arena.tasks.predicates.predicate_group import PredicateGroup
    from isaaclab_arena.tasks.predicates.spatial import (
        depth_in_range,
        tilt_axis_aligned,
        velocity_below_threshold,
        xy_in_proximity,
    )
    from isaaclab_arena.tasks.terminations import SuccessMode

    class _PlayingSimulation:
        def is_playing(self) -> bool:
            return True

    class _ArenaWorld:
        def __init__(self):
            self.poses = {
                "receiver": torch.tensor([
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                ]),
                "subject": torch.tensor([
                    [0.005, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                    [1.005, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                ]),
            }
            self.linear_velocity = torch.tensor([
                [0.0, 0.0, 0.0],
                [0.1, 0.0, 0.0],
            ])
            self.angular_velocity = torch.zeros((2, 3))

        def get_pose_w(self, name: str) -> torch.Tensor:
            return self.poses[name]

        def get_position_w(self, name: str) -> torch.Tensor:
            return self.poses[name][:, :3]

        def get_root_linear_velocity_w(self, _name: str) -> torch.Tensor:
            return self.linear_velocity

        def get_root_angular_velocity_w(self, _name: str) -> torch.Tensor:
            return self.angular_velocity

    env = SimpleNamespace(
        num_envs=2,
        device="cpu",
        extras={},
        scene={},
        sim=_PlayingSimulation(),
        arena_world=_ArenaWorld(),
        object_initial_rest_pose_recorder=ObjectInitialRestPoseRecorder(num_envs=2, device="cpu"),
    )
    settled_cfg = TerminationTermCfg(
        func=ObjectsSettledForConsecutiveSteps,
        params={
            "object_names": ["subject"],
            "lin_vel_threshold": 0.05,
            "ang_vel_threshold": 0.1,
            "consecutive_steps": 2,
        },
    )
    group_cfg = TerminationTermCfg(
        func=PredicateGroup,
        params={
            "predicates": [
                TerminationTermCfg(
                    func=xy_in_proximity,
                    params={
                        "subject_name": "subject",
                        "receiver_name": "receiver",
                        "target_offset_xyz": (0.0, 0.0, 0.0),
                        "tolerance_xy": 0.01,
                    },
                ),
                TerminationTermCfg(
                    func=depth_in_range,
                    params={
                        "subject_name": "subject",
                        "receiver_name": "receiver",
                        "target_offset_xyz": (0.0, 0.0, 0.0),
                        "depth_min": -0.01,
                        "depth_max": 0.01,
                    },
                ),
                TerminationTermCfg(
                    func=tilt_axis_aligned,
                    params={
                        "subject_name": "subject",
                        "receiver_name": "receiver",
                        "max_tilt_rad": 0.1,
                    },
                ),
                settled_cfg,
            ],
            "mode": SuccessMode.ALL,
        },
    )
    manager = TerminationManager({"success": group_cfg}, env)
    resolved_group = manager.get_term_cfg("success").func
    resolved_settled = resolved_group.predicates[-1].func
    assert isinstance(resolved_group, PredicateGroup)
    assert isinstance(resolved_settled, ConsecutivePredicate)

    assert manager.compute().tolist() == [False, False]
    env.arena_world.linear_velocity[:] = 0.0
    assert manager.compute().tolist() == [True, False]

    # Spatial gates remain current; they do not reset the motion counter.
    env.arena_world.poses["subject"][0, 0] = 0.02
    assert manager.compute().tolist() == [False, True]

    # TerminationManager forwards a partial reset through PredicateGroup.
    manager.reset(env_ids=[0])
    env.arena_world.poses["subject"][0, 0] = 0.005
    assert resolved_settled.consecutive_true_steps.tolist() == [0, 2]
    assert manager.compute().tolist() == [False, True]

    # The consecutive window applies to the combined result and resets when any child fails.
    env.first_gate = torch.tensor([True, True])
    env.second_gate = torch.tensor([True, True])

    def _first_gate(env):
        return env.first_gate

    def _second_gate(env):
        return env.second_gate

    combined_manager = TerminationManager(
        {
            "success": TerminationTermCfg(
                func=PredicateGroup,
                params={
                    "predicates": [
                        TerminationTermCfg(func=_first_gate),
                        TerminationTermCfg(func=_second_gate),
                    ],
                    "mode": SuccessMode.ALL,
                    "consecutive_steps": 2,
                },
            )
        },
        env,
    )
    assert combined_manager.compute().tolist() == [False, False]
    env.first_gate[0] = False
    assert combined_manager.compute().tolist() == [False, True]
    env.first_gate[0] = True
    assert combined_manager.compute().tolist() == [False, True]

    # ManagerBase deep-copies configs, so task-build configuration stays declarative.
    assert settled_cfg.func is ObjectsSettledForConsecutiveSteps

    from isaaclab_arena.assets.asset import Asset
    from isaaclab_arena_environments.isaac_cap.gear_insertion.task.task import GearInsertionTask

    try:
        GearInsertionTask(
            plate=Asset("plate"),
            gears=[Asset("gear")],
            target_offsets_xyz=[(0.0, 0.0, 0.0)],
            upright_axis_threshold_deg=181.0,
        )
    except ValueError as error:
        assert "must be in (0, 180]" in str(error)
    else:
        raise AssertionError("GearInsertionTask should reject orientation thresholds above 180 degrees.")

    task = GearInsertionTask(
        plate=Asset("plate"),
        gears=[Asset("gear_a"), Asset("gear_b")],
        target_offsets_xyz=[(0.1, 0.0, 0.2), (-0.1, 0.0, 0.2)],
        consecutive_success_steps=3,
    )
    success_cfg = task.get_termination_cfg().success
    assert success_cfg.func is PredicateGroup
    assert success_cfg.params["consecutive_steps"] == 3
    gear_predicates = success_cfg.params["predicates"]
    assert len(gear_predicates) == 2
    assert all(predicate.func is PredicateGroup for predicate in gear_predicates)
    for gear_name, predicate in zip(("gear_a", "gear_b"), gear_predicates, strict=True):
        velocity = predicate.params["predicates"][-1]
        assert velocity.func is velocity_below_threshold
        assert velocity.params == {
            "subject_name": gear_name,
            "linear_velocity_threshold": 0.05,
            "angular_velocity_threshold": 0.5,
        }
    return True


def test_predicate_group_lifecycle():
    assert run_function_with_persistent_simulation_app(_test_predicate_group_lifecycle)
