# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Coverage for the industrial syringe tool-sort environment."""

from pathlib import Path

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app

REPO_ROOT = Path(__file__).resolve().parents[2]
ENVIRONMENT_YAML = REPO_ROOT / "isaaclab_arena_cap" / "environments" / "industrial_syringe_sort_environment.yaml"
ENVIRONMENT_NAME = "vabar_tool_sort__syringe_easy_newton"
EXPECTED_ASSETS = {
    "industrial__fr3_workcell_table",
    "industrial__hdr_shadow_receiver",
    "industrial__tool_sort_bin",
    "vabar_tool_sort__syringe",
    "vabar_tool_sort__instrument_tray",
    "industrial_fr3_robotiq_2f85",
}


def _test_syringe_registration_and_factory(_simulation_app) -> bool:
    from isaaclab_arena.assets.registries import AssetRegistry, EnvironmentRegistry, TaskRegistry
    from isaaclab_arena_cap.environments.industrial_syringe_sort_environment import (
        IndustrialSyringeSortEnvironment,
        IndustrialSyringeSortEnvironmentCfg,
    )

    asset_registry = AssetRegistry()
    assert EXPECTED_ASSETS <= set(asset_registry.get_all_keys())
    assert TaskRegistry().is_registered("ObjectsCenteredInRegionsTask")
    assert EnvironmentRegistry().is_registered(ENVIRONMENT_NAME, ensure_loaded=False)

    arena_env = IndustrialSyringeSortEnvironment().build(IndustrialSyringeSortEnvironmentCfg())
    assert arena_env.name == ENVIRONMENT_NAME
    assert arena_env.embodiment.name == "industrial_fr3_robotiq_2f85"
    assert type(arena_env.task).__name__ == "ObjectsCenteredInRegionsTask"
    assert [asset.name for asset in arena_env.task.objects] == ["syringe_0"]
    assert [region.name for region in arena_env.task.regions] == ["sharps_container"]
    assert arena_env.task.episode_length_s == 228.0
    assert arena_env.task.consecutive_success_steps == 50
    return True


def test_syringe_registration_and_factory():
    assert run_function_with_persistent_simulation_app(_test_syringe_registration_and_factory)


def _test_syringe_python_and_yaml_are_equivalent(_simulation_app) -> bool:
    from isaaclab_arena.environment_spec.arena_env_graph_spec import ArenaEnvGraphSpec
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.tests.test_industrial_tool_sort_environment import (
        _arena_env_snapshot,
        _assert_nested_equal,
        _normalize,
    )
    from isaaclab_arena_cap.environments.industrial_syringe_sort_environment import (
        IndustrialSyringeSortEnvironment,
        IndustrialSyringeSortEnvironmentCfg,
    )

    python_env = IndustrialSyringeSortEnvironment().build(IndustrialSyringeSortEnvironmentCfg())
    yaml_env = ArenaEnvGraphSpec.from_yaml(ENVIRONMENT_YAML).to_arena_env()
    _assert_nested_equal(_arena_env_snapshot(yaml_env), _arena_env_snapshot(python_env))

    builder_cfg = ArenaEnvBuilderCfg(num_envs=1, solve_relations=False)
    python_cfg, _ = ArenaEnvBuilder(python_env, builder_cfg).compose_manager_cfg()
    yaml_cfg, _ = ArenaEnvBuilder(yaml_env, builder_cfg).compose_manager_cfg()
    _assert_nested_equal(_normalize(yaml_cfg.to_dict()), _normalize(python_cfg.to_dict()), path="env_cfg")
    return True


def test_syringe_python_and_yaml_are_equivalent():
    assert run_function_with_persistent_simulation_app(_test_syringe_python_and_yaml_are_equivalent)


def _test_syringe_newton_smoke(_simulation_app) -> bool:
    import torch

    from isaaclab_arena.assets.registries import EnvironmentRegistry
    from isaaclab_arena.cli.isaaclab_arena_cli import arena_env_builder_cfg_from_argparse, get_isaaclab_arena_cli_parser
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder

    registry = EnvironmentRegistry()
    factory_type = registry.get_component_by_name(ENVIRONMENT_NAME)
    cfg_type = registry.get_environment_cfg_type(factory_type)
    arena_env = factory_type().build(cfg_type())
    args_cli = get_isaaclab_arena_cli_parser().parse_args(["--num_envs", "1"])
    env = ArenaEnvBuilder(arena_env, arena_env_builder_cfg_from_argparse(args_cli)).make_registered()
    try:
        assert env.unwrapped.cfg.sim.dt == 1.0 / 50.0
        assert env.unwrapped.cfg.sim.physics.num_substeps == 10
        observation, _ = env.reset()
        assert all(torch.isfinite(value).all() for value in observation["policy"].values())
        action = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
        observation, _, _, _, _ = env.step(action)
        assert all(torch.isfinite(value).all() for value in observation["policy"].values())
    finally:
        env.close()
    return True


def test_syringe_newton_smoke():
    assert run_function_with_persistent_simulation_app(_test_syringe_newton_smoke)
