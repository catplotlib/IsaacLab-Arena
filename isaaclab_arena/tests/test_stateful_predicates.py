# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Validate consecutive predicate evaluation and managed-term reset lifecycle."""

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app


def _test_stateful_predicates(_simulation_app) -> bool:
    import torch
    from types import SimpleNamespace

    from isaaclab.managers import TerminationManager, TerminationTermCfg

    from isaaclab_arena.progress_tracking.progress_objective import ProgressObjective
    from isaaclab_arena.progress_tracking.progress_tracker import ProgressTracker
    from isaaclab_arena.tasks.predicates.composition import ConsecutivePredicate
    from isaaclab_arena.tasks.predicates.object_settling import (
        ObjectInitialRestPoseRecorder,
        ObjectsSettledForConsecutiveSteps,
    )

    class _PlayingSimulation:
        def is_playing(self) -> bool:
            return True

    class _ArenaWorld:
        def __init__(self):
            self.linear_velocity = torch.zeros((2, 3))
            self.angular_velocity = torch.zeros((2, 3))
            self.position = torch.tensor([[0.0, 0.0, 0.5], [1.0, 0.0, 0.5]])

        def get_root_linear_velocity_w(self, _name: str) -> torch.Tensor:
            return self.linear_velocity

        def get_root_angular_velocity_w(self, _name: str) -> torch.Tensor:
            return self.angular_velocity

        def get_position_w(self, _name: str) -> torch.Tensor:
            return self.position

    arena_world = _ArenaWorld()
    env = SimpleNamespace(
        num_envs=2,
        device="cpu",
        scene={},
        sim=_PlayingSimulation(),
        arena_world=arena_world,
        object_initial_rest_pose_recorder=ObjectInitialRestPoseRecorder(2, "cpu"),
    )
    settled_cfg = TerminationTermCfg(
        func=ObjectsSettledForConsecutiveSteps,
        params={
            "object_names": ["sphere"],
            "lin_vel_threshold": 0.1,
            "ang_vel_threshold": 0.1,
            "consecutive_steps": 2,
        },
    )
    manager = TerminationManager({"success": settled_cfg}, env)

    # Isaac Lab resolves the managed term inside a copied config. The task's source config stays reusable.
    resolved_settled = manager.get_term_cfg("success").func
    assert isinstance(resolved_settled, ObjectsSettledForConsecutiveSteps)
    assert isinstance(resolved_settled, ConsecutivePredicate)
    assert settled_cfg.func is ObjectsSettledForConsecutiveSteps

    # Zero initial velocity is not enough: success requires two consecutive evaluations.
    assert manager.compute().tolist() == [False, False]
    assert resolved_settled.consecutive_true_steps.tolist() == [1, 1]
    assert manager.compute().tolist() == [True, True]
    assert resolved_settled.consecutive_true_steps.tolist() == [2, 2]

    # Linear falling and angular rolling independently clear the stability streak per environment.
    arena_world.linear_velocity[0, 2] = -1.0
    arena_world.angular_velocity[1, 1] = 1.0
    assert manager.compute().tolist() == [False, False]
    assert resolved_settled.consecutive_true_steps.tolist() == [0, 0]

    # Managed-term lifecycle propagation resets only the selected environment and its recorded rest pose.
    arena_world.linear_velocity.zero_()
    arena_world.angular_velocity.zero_()
    manager.compute()
    manager.compute()
    _, recorded = env.object_initial_rest_pose_recorder.get("sphere")
    assert recorded.tolist() == [True, True]
    manager.reset(env_ids=[1])
    assert resolved_settled.consecutive_true_steps.tolist() == [2, 0]
    _, recorded = env.object_initial_rest_pose_recorder.get("sphere")
    assert recorded.tolist() == [True, False]
    manager.reset()
    assert resolved_settled.consecutive_true_steps.tolist() == [0, 0]
    _, recorded = env.object_initial_rest_pose_recorder.get("sphere")
    assert recorded.tolist() == [False, False]

    # Progress tracking resolves and owns managed predicate configs, including their reset lifecycle.
    progress_cfg = TerminationTermCfg(
        func=ObjectsSettledForConsecutiveSteps,
        params={"object_names": ["sphere"], "consecutive_steps": 1},
    )
    progress_tracker = ProgressTracker(
        progress_objectives=[ProgressObjective(name="settled", predicate_groups=progress_cfg)],
        num_envs=env.num_envs,
        device=env.device,
        env=env,
    )
    progress_predicate = progress_tracker.runners[0].predicate_chains["default_group"][0][0]
    resolved_progress_settled = progress_predicate.func
    assert isinstance(resolved_progress_settled, ObjectsSettledForConsecutiveSteps)

    progress_tracker.step(env, step_index=torch.tensor([1, 1]))
    assert resolved_progress_settled.consecutive_true_steps.tolist() == [1, 1]
    assert [state.all_complete for state in progress_tracker.get_state()] == [True, True]
    progress_tracker.reset([1])
    assert resolved_progress_settled.consecutive_true_steps.tolist() == [1, 0]
    return True


def test_stateful_predicates():
    assert run_function_with_persistent_simulation_app(_test_stateful_predicates)


def _test_off_table_sphere_does_not_settle_before_falling(_simulation_app) -> bool:
    import torch
    from types import SimpleNamespace

    from isaaclab_arena.cli.isaaclab_arena_cli import arena_env_builder_cfg_from_argparse, get_isaaclab_arena_cli_parser
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.tasks.predicates.object_settling import ObjectsSettledForConsecutiveSteps
    from isaaclab_arena.utils.physics_settle import step_physics
    from isaaclab_arena_examples.external_environments.object_settled import ExternalObjectsSettledEnvironment

    environment = ExternalObjectsSettledEnvironment().get_env(SimpleNamespace(consecutive_steps=5))
    args_cli = get_isaaclab_arena_cli_parser().parse_args([])
    args_cli.num_envs = 1
    env = ArenaEnvBuilder(environment, arena_env_builder_cfg_from_argparse(args_cli)).make_registered()
    env.reset()

    try:
        arena_env = env.unwrapped
        settled = arena_env.termination_manager.get_term_cfg("success").func
        assert isinstance(settled, ObjectsSettledForConsecutiveSteps)

        falling_speed = arena_env.arena_world.get_root_linear_velocity_w("falling_sphere").norm(dim=-1)
        assert falling_speed.item() == 0.0
        assert not settled(arena_env, **settled.cfg.params).item()
        assert settled.consecutive_true_steps.item() == 1

        # The unsupported sphere starts accelerating on the next physics frame, clearing the false streak.
        step_physics(env, 1)
        falling_speed = arena_env.arena_world.get_root_linear_velocity_w("falling_sphere").norm(dim=-1)
        assert falling_speed.item() > 1e-2
        assert not settled(arena_env, **settled.cfg.params).item()
        assert settled.consecutive_true_steps.item() == 0

        # It can succeed only after falling to the ground and completing a fresh stability window.
        actions = torch.zeros(env.action_space.shape, device=arena_env.device)
        for _ in range(60):
            _, _, terminated, truncated, info = env.step(actions)
            progress = info["progress_tracking"]
            progress_state = progress["states"][0]
            if terminated.item():
                assert not truncated.item()
                settled_state = progress_state.progress_objectives["objects_settled"]
                assert settled_state.is_complete
                assert progress_state.all_complete
                assert progress_state.overall_score == 1.0
                settled_events = progress["events"][0]
                assert len(settled_events) == 1
                assert settled_events[0].predicate_name == "objects_settled_success"
                assert settled_events[0].step >= 5
                break
            assert not progress_state.progress_objectives["objects_settled"].is_complete
            assert progress["events"][0] == []
        else:
            raise AssertionError("The spheres did not settle before the example task timed out.")
    finally:
        env.close()
    return True


def test_off_table_sphere_does_not_settle_before_falling():
    assert run_function_with_persistent_simulation_app(_test_off_table_sphere_does_not_settle_before_falling)
