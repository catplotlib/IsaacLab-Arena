# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for :class:`DistractorDisappearVariation` wired through :class:`ArenaEnvBuilder`."""

import pytest

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app
from isaaclab_arena.variations.bernoulli_sampler import BernoulliSamplerCfg
from isaaclab_arena.variations.distractor_disappear_variation import DistractorDisappearVariationCfg

HEADLESS = True

TEST_ASSET_NAME = "sphere"
TEST_AWAY_POSITION_XYZ = (0.0, 0.0, -10.0)


def get_test_environment(*, enabled: bool, probability: float):
    """Build a minimal arena env with an optional enabled distractor-disappear variation."""
    from isaaclab_arena.assets.registries import AssetRegistry
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.variations.distractor_disappear_variation import DistractorDisappearVariation

    sphere = AssetRegistry().get_asset_by_name(TEST_ASSET_NAME)()
    assert sphere.name == TEST_ASSET_NAME

    variation = DistractorDisappearVariation(
        sphere,
        DistractorDisappearVariationCfg(
            away_position_xyz=TEST_AWAY_POSITION_XYZ,
            sampler_cfg=BernoulliSamplerCfg(probability=probability),
        ),
    )
    if enabled:
        variation.enable()
    assert variation.enabled is enabled
    sphere.add_variation(variation)

    return IsaacLabArenaEnvironment(
        name="test_distractor_disappear_variation",
        scene=Scene(assets=[sphere]),
    )


def _test_disappearing_distractor_is_teleported_away(simulation_app):
    from isaaclab_arena.cli.isaaclab_arena_cli import arena_env_builder_cfg_from_argparse, get_isaaclab_arena_cli_parser
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder

    # probability=1.0 makes the draw deterministic: the distractor always disappears.
    arena_env = get_test_environment(enabled=True, probability=1.0)
    sphere = arena_env.scene.assets[TEST_ASSET_NAME]
    args_cli = get_isaaclab_arena_cli_parser().parse_args(["--num_envs", "1"])
    ArenaEnvBuilder(arena_env, arena_env_builder_cfg_from_argparse(args_cli)).compose_manager_cfg()

    pos = tuple(sphere.object_cfg.init_state.pos)
    assert pos == pytest.approx(
        TEST_AWAY_POSITION_XYZ
    ), f"Disappearing distractor must be teleported to {TEST_AWAY_POSITION_XYZ}; got {pos}."
    return True


def _test_non_disappearing_distractor_keeps_its_pose(simulation_app):
    from isaaclab_arena.cli.isaaclab_arena_cli import arena_env_builder_cfg_from_argparse, get_isaaclab_arena_cli_parser
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder

    # probability=0.0 makes the draw deterministic: the distractor never disappears.
    arena_env = get_test_environment(enabled=True, probability=0.0)
    sphere = arena_env.scene.assets[TEST_ASSET_NAME]
    default_pos = tuple(sphere.object_cfg.init_state.pos)
    args_cli = get_isaaclab_arena_cli_parser().parse_args(["--num_envs", "1"])
    ArenaEnvBuilder(arena_env, arena_env_builder_cfg_from_argparse(args_cli)).compose_manager_cfg()

    assert tuple(sphere.object_cfg.init_state.pos) == default_pos, (
        f"Non-disappearing distractor must keep its original pose {default_pos}; "
        f"got {tuple(sphere.object_cfg.init_state.pos)}."
    )
    return True


def _test_disabled_variation_not_applied(simulation_app):
    from isaaclab_arena.cli.isaaclab_arena_cli import arena_env_builder_cfg_from_argparse, get_isaaclab_arena_cli_parser
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder

    # probability=1.0 would always teleport the distractor away if the variation were enabled.
    arena_env = get_test_environment(enabled=False, probability=1.0)
    sphere = arena_env.scene.assets[TEST_ASSET_NAME]
    default_pos = tuple(sphere.object_cfg.init_state.pos)
    args_cli = get_isaaclab_arena_cli_parser().parse_args(["--num_envs", "1"])
    ArenaEnvBuilder(arena_env, arena_env_builder_cfg_from_argparse(args_cli)).compose_manager_cfg()

    assert tuple(sphere.object_cfg.init_state.pos) == default_pos, (
        f"Disabled variation must not mutate '{TEST_ASSET_NAME}.object_cfg.init_state.pos'; "
        f"expected {default_pos}, got {tuple(sphere.object_cfg.init_state.pos)}."
    )
    return True


def test_disappearing_distractor_is_teleported_away():
    assert run_function_with_persistent_simulation_app(
        _test_disappearing_distractor_is_teleported_away,
        headless=HEADLESS,
    )


def test_non_disappearing_distractor_keeps_its_pose():
    assert run_function_with_persistent_simulation_app(
        _test_non_disappearing_distractor_keeps_its_pose,
        headless=HEADLESS,
    )


def test_disabled_variation_not_applied():
    assert run_function_with_persistent_simulation_app(
        _test_disabled_variation_not_applied,
        headless=HEADLESS,
    )
