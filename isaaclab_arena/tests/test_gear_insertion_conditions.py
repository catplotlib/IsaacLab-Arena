# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Check gear insertion's instantaneous diagnostics and runner-owned duration."""

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app


def _test_gear_insertion_requirement_configuration(_simulation_app):
    import pytest

    from isaaclab_arena.assets.asset import Asset
    from isaaclab_arena.tasks.predicates.temporal import TrueForConsecutiveStepsCfg
    from isaaclab_arena_environments.isaac_cap.gear_insertion.task.predicates import (
        GearInsertionConditions,
        reset_gear_insertion_diagnostics,
    )
    from isaaclab_arena_environments.isaac_cap.gear_insertion.task.task import GearInsertionTask
    from isaaclab_arena_environments.isaac_cap.gear_insertion_v2.task.task import (
        GearInsertionTask as LegacyGearInsertionTask,
    )

    assert LegacyGearInsertionTask is GearInsertionTask
    with pytest.raises(ValueError, match=r"must be in \(0, 180\]"):
        GearInsertionTask(
            plate=Asset("plate"),
            gears=[Asset("gear")],
            target_offsets_xyz=[(0.0, 0.0, 0.0)],
            upright_axis_threshold_deg=181.0,
        )

    task = GearInsertionTask(
        plate=Asset("plate"),
        gears=[Asset("gear_a"), Asset("gear_b")],
        target_offsets_xyz=[(0.1, 0.0, 0.2), (-0.1, 0.0, 0.2)],
        consecutive_success_steps=3,
    )
    diagnostic_reset = task.get_events_cfg().reset_gear_insertion_diagnostics
    assert diagnostic_reset.func is reset_gear_insertion_diagnostics
    assert diagnostic_reset.mode == "reset"
    success_objectives = task.get_termination_cfg().success
    assert len(success_objectives) == 1
    assert success_objectives[0].name == "gear_insertion"
    requirement = success_objectives[0].predicate_sequence[0]
    assert isinstance(requirement, TrueForConsecutiveStepsCfg)
    assert requirement.required_steps == 3
    assert requirement.predicate.func is GearInsertionConditions
    assert requirement.predicate.params == {
        "plate_name": "plate",
        "gear_names": ("gear_a", "gear_b"),
        "target_offsets_xyz": ((0.1, 0.0, 0.2), (-0.1, 0.0, 0.2)),
        "xy_threshold": 0.015,
        "z_threshold": 0.01,
        "upright_axis_threshold_deg": 15.0,
        "linear_velocity_threshold": 0.05,
        "angular_velocity_threshold": 0.5,
        "support_z_threshold": 0.005,
    }
    return True


def _test_gear_insertion_named_diagnostics(_simulation_app):
    import torch
    from types import SimpleNamespace

    import pytest

    from isaaclab_arena_environments.isaac_cap.gear_insertion.task.metrics import _terminal_diagnostics

    gear_a_gates = torch.tensor([
        [True, False, True],
        [False, True, True],
        [True, True, False],
        [False, False, True],
        [True, False, False],
    ])
    gear_b_gates = torch.tensor([
        [False, True, True],
        [True, False, True],
        [False, True, False],
        [True, True, False],
        [False, False, True],
    ])
    gate_names = ("xy", "z", "upright", "support", "velocity")
    conditions = SimpleNamespace(
        per_gear_results={"gear_a": torch.tensor([True, False, True]), "gear_b": torch.tensor([False, True, True])},
        per_gear_gate_results={
            "gear_a": dict(zip(gate_names, gear_a_gates, strict=True)),
            "gear_b": dict(zip(gate_names, gear_b_gates, strict=True)),
        },
    )
    diagnostics = _terminal_diagnostics(conditions, torch.tensor([2, 0]), ("gear_a", "gear_b"))
    assert diagnostics == [
        {
            "gear_a": dict(success=True, xy=True, z=True, upright=False, support=True, velocity=False),
            "gear_b": dict(success=True, xy=True, z=True, upright=False, support=False, velocity=True),
        },
        {
            "gear_a": dict(success=True, xy=True, z=False, upright=True, support=False, velocity=True),
            "gear_b": dict(success=False, xy=False, z=True, upright=False, support=True, velocity=False),
        },
    ]
    with pytest.raises(AssertionError, match="requested gears"):
        _terminal_diagnostics(conditions, [0], ("gear_a",))
    conditions.per_gear_gate_results["gear_a"]["xy"] = torch.zeros(2, dtype=torch.bool)
    with pytest.raises(AssertionError, match="shape"):
        _terminal_diagnostics(conditions, [0], ("gear_a", "gear_b"))
    return True


def _test_gear_insertion_overlap_reporting_and_partial_reset(_simulation_app):
    import torch
    from types import SimpleNamespace
    from unittest.mock import patch

    from isaaclab.managers import TerminationTermCfg

    from isaaclab_arena.progress_tracking.progress_objective import ProgressObjective
    from isaaclab_arena.progress_tracking.progress_tracker import ProgressTracker
    from isaaclab_arena.tasks.predicates.temporal import TrueForConsecutiveStepsCfg
    from isaaclab_arena_environments.isaac_cap.gear_insertion.task.metrics import (
        GearInsertionFractionRecorder,
        GearInsertionFractionRecorderCfg,
        _terminal_diagnostics,
    )
    from isaaclab_arena_environments.isaac_cap.gear_insertion.task.predicates import (
        GearInsertionConditions,
        GearIsSupported,
        reset_gear_insertion_diagnostics,
    )

    class _ArenaWorld:
        def __init__(self):
            self.poses = {}
            self.linear_velocity = {}
            for name, x, z in (("plate", 0.0, 0.0), ("gear_a", 0.1, 0.03), ("gear_b", -0.1, 0.03)):
                self.poses[name] = torch.tensor([[x, 0.0, z, 0.0, 0.0, 0.0, 1.0]]).repeat(2, 1)
                self.linear_velocity[name] = torch.zeros(2, 3)
            self.pose_reads = 0

        def get_pose_w(self, name):
            self.pose_reads += 1
            return self.poses[name]

        def get_root_linear_velocity_w(self, name):
            return self.linear_velocity[name]

        def get_root_angular_velocity_w(self, name):
            return torch.zeros(2, 3)

    def _collision_corners(asset, device, **_kwargs):
        if asset.name == "plate":
            return torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.02]], device=device)
        return torch.tensor([[0.0, 0.0, -0.01], [0.0, 0.0, 0.01]], device=device)

    conditions_cfg = TerminationTermCfg(
        func=GearInsertionConditions,
        params={
            "plate_name": "plate",
            "gear_names": ("gear_a", "gear_b"),
            "target_offsets_xyz": ((0.1, 0.0, 0.03), (-0.1, 0.0, 0.03)),
            "xy_threshold": 0.015,
            "z_threshold": 0.01,
            "upright_axis_threshold_deg": 15.0,
            "linear_velocity_threshold": 0.05,
            "angular_velocity_threshold": 0.5,
            "support_z_threshold": 0.005,
        },
    )
    objective = ProgressObjective(
        name="gear_insertion",
        predicate_sequence=[TrueForConsecutiveStepsCfg(predicate=conditions_cfg, required_steps=3)],
    )
    env = SimpleNamespace(
        num_envs=2,
        device="cpu",
        arena_world=_ArenaWorld(),
        scene={name: SimpleNamespace(name=name) for name in ("plate", "gear_a", "gear_b")},
    )
    with patch.object(GearIsSupported, "_collision_corners", side_effect=_collision_corners):
        tracker = ProgressTracker([objective], num_envs=2, device="cpu", env=env)
    env.progress_tracker = tracker
    conditions = tracker.get_predicate("gear_insertion")
    recorder = GearInsertionFractionRecorder(GearInsertionFractionRecorderCfg(gear_names=("gear_a", "gear_b")), env)
    assert recorder.record_pre_reset([0, 1]) == (None, None)

    # Different gears being ready on different steps must not add up to success.
    env.arena_world.linear_velocity["gear_b"][0, 0] = 0.1
    tracker.step(env, step_index=torch.tensor([1, 1]))
    diagnostics = _terminal_diagnostics(conditions, [0], ("gear_a", "gear_b"))[0]
    assert diagnostics["gear_a"]["success"]
    assert not diagnostics["gear_b"]["velocity"]
    env.arena_world.linear_velocity["gear_b"][:] = 0.0
    env.arena_world.poses["gear_a"][0, 0] = 0.2
    tracker.step(env, step_index=torch.tensor([2, 2]))
    diagnostics = _terminal_diagnostics(conditions, [0], ("gear_a", "gear_b"))[0]
    assert not diagnostics["gear_a"]["xy"]
    assert diagnostics["gear_b"]["success"]
    assert tracker.is_complete().tolist() == [False, False]

    env.arena_world.poses["gear_a"][0, 0] = 0.1
    for step_index in (3, 4, 5):
        tracker.step(env, step_index=torch.tensor([step_index, step_index]))
        pose_reads = env.arena_world.pose_reads
        for _ in range(3):
            name, fractions = recorder.record_pre_reset([0, 1])
            assert name == "gear_insertion_fraction"
            assert fractions.tolist() == [1.0, 1.0]
        assert env.arena_world.pose_reads == pose_reads
        assert tracker.is_complete().tolist() == [step_index == 5, True]

    unchanged_diagnostics = _terminal_diagnostics(conditions, [1], ("gear_a", "gear_b"))[0]
    reset_gear_insertion_diagnostics(env, env_ids=[0])
    assert tracker.is_complete().tolist() == [True, True], "The task callback clears diagnostics, not progress."
    tracker.reset([0])
    diagnostics = _terminal_diagnostics(conditions, [0, 1], ("gear_a", "gear_b"))
    for gear_name in ("gear_a", "gear_b"):
        assert not any(diagnostics[0][gear_name].values())
    assert diagnostics[1] == unchanged_diagnostics
    assert tracker.is_complete().tolist() == [False, True]
    for step_index in (1, 2, 3):
        tracker.step(env, step_index=torch.tensor([step_index, step_index + 5]))
        assert tracker.is_complete().tolist() == [step_index == 3, True]

    for _ in range(2):
        reset_gear_insertion_diagnostics(env)
        tracker.reset([0, 1])
        diagnostics = _terminal_diagnostics(conditions, [0, 1], ("gear_a", "gear_b"))
        for environment_diagnostics in diagnostics:
            for gear_diagnostics in environment_diagnostics.values():
                assert not any(gear_diagnostics.values())
        assert tracker.is_complete().tolist() == [False, False]
        for step_index in (1, 2, 3):
            tracker.step(env, step_index=torch.tensor([step_index, step_index]))
            assert tracker.is_complete().tolist() == [step_index == 3, step_index == 3]
    return True


def test_gear_insertion_requirement_configuration():
    assert run_function_with_persistent_simulation_app(_test_gear_insertion_requirement_configuration)


def test_gear_insertion_named_diagnostics():
    assert run_function_with_persistent_simulation_app(_test_gear_insertion_named_diagnostics)


def test_gear_insertion_overlap_reporting_and_partial_reset():
    assert run_function_with_persistent_simulation_app(_test_gear_insertion_overlap_reporting_and_partial_reset)
