# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Check syringe containment, consecutive settling, and episode resets."""

import pytest

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app


def _test_syringe_success_streak_and_reset(_simulation_app):
    import torch
    from types import SimpleNamespace

    from isaaclab_arena.tasks.predicates.temporal import TrueForConsecutiveStepsCfg
    from isaaclab_arena.tests.test_task_success_from_progress import _make_environment_and_manager
    from isaaclab_arena_environments.isaac_cap.syringe_sort.tasks.task import SyringeSortTask

    task = SyringeSortTask(
        object_list=[SimpleNamespace(name="syringe_a"), SimpleNamespace(name="syringe_b")],
        region_list=[SimpleNamespace(name="receiver_a"), SimpleNamespace(name="receiver_b")],
        bounds_xyzxyz=[(-0.1, -0.1, -0.1, 0.1, 0.1, 0.1)] * 2,
        linear_velocity_threshold=0.01,
        angular_velocity_threshold=0.05,
        consecutive_success_steps=3,
    )
    termination_cfg = task.get_termination_cfg()
    requirement = termination_cfg.success[0].predicate_sequence[0]
    assert isinstance(requirement, TrueForConsecutiveStepsCfg)
    assert requirement.required_steps == 3
    assert requirement.predicate.to_dict()["func"] == "isaaclab_arena.tasks.terminations:check_success"
    assert termination_cfg.timeout_s == 228.0

    env, manager, _ = _make_environment_and_manager([], success_objectives=termination_cfg.success)
    centers = {name: torch.zeros((2, 3)) for name in ("syringe_a", "syringe_b")}
    linear_velocities = {name: torch.zeros((2, 3)) for name in centers}
    angular_velocities = {name: torch.zeros((2, 3)) for name in centers}
    receiver_pose = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]).repeat(2, 1)
    env.scene = {
        name: SimpleNamespace(data=SimpleNamespace(root_com_pos_w=SimpleNamespace(torch=center)))
        for name, center in centers.items()
    }
    env.arena_world = SimpleNamespace(
        get_pose_w=lambda _name: receiver_pose,
        get_root_linear_velocity_w=lambda name: linear_velocities[name],
        get_root_angular_velocity_w=lambda name: angular_velocities[name],
    )

    env.episode_length_buf += 1
    manager.compute()
    manager.compute()
    assert manager.get_term("success").tolist() == [False, False]

    angular_velocities["syringe_b"][0, 0] = 0.06
    centers["syringe_a"][1, 0] = 0.11
    env.episode_length_buf += 1
    manager.compute()
    assert manager.get_term("success").tolist() == [False, False]

    angular_velocities["syringe_b"][0, 0] = 0.0
    centers["syringe_a"][1, 0] = 0.0
    for expected_success in ([False, False], [False, False], [True, True]):
        env.episode_length_buf += 1
        manager.compute()
        assert manager.get_term("success").tolist() == expected_success

    manager.reset([1])
    env.episode_length_buf[1] = 0
    for expected_success in ([True, False], [True, False], [True, True]):
        env.episode_length_buf += 1
        manager.compute()
        assert manager.get_term("success").tolist() == expected_success

    cap_finished = termination_cfg.failures["cap_finished"]
    assert not cap_finished.func(env, **cap_finished.params).any()
    env.cap_episode_finished = True
    assert cap_finished.func(env, **cap_finished.params).all()
    return True


def test_syringe_success_streak_and_reset():
    assert run_function_with_persistent_simulation_app(_test_syringe_success_streak_and_reset)


def _test_syringe_drop(_simulation_app):
    import torch

    from isaaclab.utils.math import quat_apply

    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.policy.zero_action_policy import ZeroActionPolicy, ZeroActionPolicyCfg
    from isaaclab_arena_environments.isaac_cap.registration import register_components
    from isaaclab_arena_environments.isaac_cap.syringe_sort.environments.environment import (
        SyringeSingleEnvironment,
        SyringeSortEnvironmentCfg,
    )

    register_components()
    arena_env = SyringeSingleEnvironment().build(SyringeSortEnvironmentCfg())
    env = ArenaEnvBuilder(arena_env, ArenaEnvBuilderCfg(solve_relations=False)).make_registered()
    try:
        obs, _ = env.reset()
        base = env.unwrapped
        syringe = base.scene["syringe_0"]
        # W is world, R is the receiver, and S is the syringe root frame.
        T_W_R = base.arena_world.get_pose_w("sharps_container")
        # Place S above the aperture: t_W_S = t_W_R + R_W_R * t_R_S.
        t_R_S = T_W_R.new_tensor([[0.0975, -0.1225, 0.30]])
        T_W_S = T_W_R.clone()
        T_W_S[:, :3] = T_W_R[:, :3] + quat_apply(T_W_R[:, 3:], t_R_S)
        # q_W_S rotates +90 degrees about world X, making the syringe's Y axis vertical.
        T_W_S[:, 3:] = T_W_S.new_tensor([[2**-0.5, 0, 0, 2**-0.5]])
        syringe.write_root_pose_to_sim(T_W_S)
        syringe.write_root_velocity_to_sim(torch.zeros((1, 6), device=base.device))
        policy = ZeroActionPolicy(ZeroActionPolicyCfg())
        with torch.inference_mode():
            for step in range(500):
                obs, _, terminated, truncated, _ = env.step(policy.get_action(env, obs))
                success = base.termination_manager.get_term("success")
                if step == 0:
                    assert not success.any(), "Syringe above the container must not count as contained"
                assert not truncated.any(), "Drop test timed out"
                if terminated.any():
                    assert success.all(), "Episode ended without syringe containment success"
                    assert base.episode_length_buf[0] == 0, "Success did not reset the environment"
                    return True
        assert False, "Syringe did not fall into the container and settle"
    finally:
        env.close()


@pytest.mark.with_newton
def test_syringe_drop():
    assert run_function_with_persistent_simulation_app(_test_syringe_drop)
