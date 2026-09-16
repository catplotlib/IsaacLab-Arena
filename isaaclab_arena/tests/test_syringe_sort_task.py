# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Syringe success geometry, release, settling, and partial-reset behavior."""

import pytest

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app


def _test_syringe_success(_simulation_app):
    import torch
    from types import SimpleNamespace

    from isaaclab_arena.tasks.predicates.composite import CompositePredicate
    from isaaclab_arena_environments.isaac_cap.syringe_sort.tasks.task import SyringeSortTask, center_of_mass_in_region

    # Receiver 1 is translated and rotated 90 degrees around Z.
    poses = torch.tensor([[0, 0, 0, 0, 0, 0, 1], [2, 3, 0, 0, 0, 2**-0.5, 2**-0.5]], dtype=torch.float32)
    centers = torch.tensor([[0.1, -0.12, 0.05], [2.12, 3.1, 0.05]])
    linear = torch.zeros(2, 3)
    angular = torch.zeros(2, 3)
    joints = torch.zeros(2, 1)
    env = SimpleNamespace(
        num_envs=2,
        device="cpu",
        scene={
            "syringe": SimpleNamespace(data=SimpleNamespace(root_com_pos_w=SimpleNamespace(torch=centers))),
            "robot": SimpleNamespace(data=SimpleNamespace(joint_pos=SimpleNamespace(torch=joints))),
        },
        arena_world=SimpleNamespace(
            get_pose_w=lambda name: poses,
            get_root_linear_velocity_w=lambda name: linear,
            get_root_angular_velocity_w=lambda name: angular,
        ),
    )
    bounds = (0.053, -0.2055, -0.016, 0.142, -0.0395, 0.134)
    assert center_of_mass_in_region(env, "syringe", "box", bounds).tolist() == [True, True]
    centers[0, 2] = 0.2  # Above the box is not inside it.
    assert center_of_mass_in_region(env, "syringe", "box", bounds).tolist() == [False, True]
    centers[0, 2] = 0.05
    task = SyringeSortTask(
        [SimpleNamespace(name="syringe")], [SimpleNamespace(name="box")], [bounds], consecutive_success_steps=3
    )
    cfg = task.get_termination_cfg().success
    cfg.params["predicates"][0].params["robot_cfg"].joint_ids = [0]
    predicate = CompositePredicate(cfg, env)

    def evaluate():
        return predicate(env, **cfg.params).tolist()

    assert evaluate() == [False, False]
    joints[0] = 0.2  # A closed gripper breaks environment 0's streak.
    assert evaluate() == [False, False]
    joints[0] = 0.0
    assert evaluate() == [False, True]
    angular[0, 2] = 0.1  # Angular motion also prevents success.
    assert evaluate() == [False, True]
    angular.zero_()
    linear[0, 0] = 0.02  # So does linear motion.
    assert evaluate() == [False, True]
    linear.zero_()
    assert evaluate() == [False, True]
    assert evaluate() == [False, True]
    assert evaluate() == [True, True]
    predicate.reset([0])
    assert evaluate() == [False, True]
    predicate.reset()
    assert evaluate() == [False, False]
    # Both goals must meet the same conditions; one disposed syringe is insufficient.
    second = centers.clone()
    env.scene["blank"] = SimpleNamespace(data=SimpleNamespace(root_com_pos_w=SimpleNamespace(torch=second)))
    both = SyringeSortTask(
        [SimpleNamespace(name="syringe"), SimpleNamespace(name="blank")],
        [SimpleNamespace(name="box")] * 2,
        [bounds] * 2,
        consecutive_success_steps=1,
    )
    cfg = both.get_termination_cfg().success
    cfg.params["predicates"][0].params["robot_cfg"].joint_ids = [0]
    predicate = CompositePredicate(cfg, env)
    second[0, 2] = 0.2
    assert evaluate() == [False, True]
    second[0, 2] = 0.05
    assert evaluate() == [True, True]
    # Clutter requires every syringe, including the fourth, to remain contained.
    names = ["clutter_0", "clutter_1", "clutter_2", "clutter_3"]
    clutter_centers = [centers.clone() for _ in names]
    for name, positions in zip(names, clutter_centers):
        env.scene[name] = SimpleNamespace(data=SimpleNamespace(root_com_pos_w=SimpleNamespace(torch=positions)))
    clutter = SyringeSortTask(
        [SimpleNamespace(name=name) for name in names],
        [SimpleNamespace(name="box")] * 4,
        [bounds] * 4,
        consecutive_success_steps=1,
    )
    cfg = clutter.get_termination_cfg().success
    cfg.params["predicates"][0].params["robot_cfg"].joint_ids = [0]
    predicate = CompositePredicate(cfg, env)
    for positions in clutter_centers:
        positions[0, 2] = 0.2
        assert evaluate() == [False, True]
        positions[0, 2] = 0.05
        assert evaluate() == [True, True]
    return True


def test_syringe_success():
    assert run_function_with_persistent_simulation_app(_test_syringe_success)


def _test_syringe_environment(_simulation_app, variant, num_syringes):
    import torch
    from dataclasses import replace
    from pathlib import Path

    from pxr import UsdUtils

    from isaaclab_arena.evaluation.arena_experiment_config_loader import load_arena_experiment_from_config_file
    from isaaclab_arena.evaluation.run_execution import build_arena_builder_from_run_cfg
    from isaaclab_arena.policy.zero_action_policy import ZeroActionPolicy, ZeroActionPolicyCfg
    from isaaclab_arena_environments.isaac_cap import register_components, syringe_sort

    register_components()
    experiment_path = (
        Path(syringe_sort.__file__).parent / "experiment_configs" / f"{variant}_zero_action_experiment.yaml"
    )
    experiment = load_arena_experiment_from_config_file(
        experiment_path,
        device="cuda:0",
        overrides=["shared.environment.episode_length_s=2"],
    )
    run = next(iter(experiment.runs.values()))
    run = replace(run, environment_builder=replace(run.environment_builder, placement_seed=43))
    builder = build_arena_builder_from_run_cfg(run)
    syringe_names = [f"syringe_{i}" for i in range(num_syringes)]
    assert {obj.name for obj in builder.arena_env.task.objects} == set(syringe_names)
    for name in (*syringe_names, "instrument_tray", "sharps_container"):
        _, _, unresolved = UsdUtils.ComputeAllDependencies(builder.arena_env.scene.assets[name].usd_path)
        assert not unresolved, f"Unresolved USD dependencies for {name}: {unresolved}"
    env = builder.make_registered()
    try:
        obs, _ = env.reset()
        base = env.unwrapped
        initial_pose = base.arena_world.get_pose_w("syringe_0").clone()
        assert base.step_dt == 0.02
        policy = ZeroActionPolicy(ZeroActionPolicyCfg())
        timeouts = 0
        with torch.inference_mode():
            for _ in range(200):
                obs, _, terminated, truncated, _ = env.step(policy.get_action(env, obs))
                assert not terminated.any(), "Zero actions must not report syringe task success"
                assert all(torch.isfinite(value).all() for value in obs["policy"].values())
                for name in syringe_names:
                    pose = base.arena_world.get_pose_w(name)
                    assert torch.isfinite(pose).all(), name
                    assert (pose[:, 2] > 0.77).all(), f"{name} fell through the tray/workcell or was not placed"
                    assert torch.isfinite(base.arena_world.get_root_linear_velocity_w(name)).all(), name
                    assert torch.isfinite(base.arena_world.get_root_angular_velocity_w(name)).all(), name
                if truncated.any():
                    assert (base.episode_length_buf[truncated] == 0).all(), "Episode did not reset on timeout"
                timeouts += int(truncated.sum())
        assert timeouts == 2
        # Randomized variants draw new layouts on reset; single restores its fixed pose.
        if variant == "single":
            assert torch.allclose(base.arena_world.get_pose_w("syringe_0"), initial_pose, atol=1e-4)
        return True
    finally:
        env.close()


@pytest.mark.parametrize("variant,num_syringes", [("single", 1), ("both", 2), ("cluttered", 4)])
def test_syringe_environment(variant, num_syringes):
    assert run_function_with_persistent_simulation_app(
        _test_syringe_environment, variant=variant, num_syringes=num_syringes
    )
