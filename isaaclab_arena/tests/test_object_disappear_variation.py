# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app

HEADLESS = True

TEST_ASSET_NAME = "sphere"
TEST_EVENT_NAME = f"{TEST_ASSET_NAME}_disappear"
TEST_AWAY_POSITION_XYZ = (0.0, 0.0, -10.0)

# Relation-placement regression case: a rigid object placed on an anchored table.
TEST_TABLE_NAME = "table"
TEST_BOX_NAME = "cracker_box"

# Comfortably below anything the scene places, but well above the away pose even after a few
# frames of free fall, so the check distinguishes "held away" from "placed in the scene".
DISAPPEARED_Z_CEILING_M = -5.0


def get_env_local_position(env, asset_name):
    """Return the asset's env-0 position relative to its environment origin."""
    import warp as wp

    pose_w = wp.to_torch(env.unwrapped.scene[asset_name].data.root_pose_w)
    return (pose_w[:, :3] - env.unwrapped.scene.env_origins)[0]


def get_test_environment(*, enabled: bool, probability: float):
    """Build a minimal arena env with an optional enabled disappear variation on a sphere."""
    from isaaclab_arena.assets.registries import AssetRegistry
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.variations.bernoulli_sampler import BernoulliSamplerCfg
    from isaaclab_arena.variations.object_disappear_variation import ObjectDisappearVariationCfg

    sphere = AssetRegistry().get_asset_by_name(TEST_ASSET_NAME)()
    variation = sphere.get_variation("disappear")
    variation.apply_cfg(
        ObjectDisappearVariationCfg(
            enabled=enabled,
            away_position_xyz=TEST_AWAY_POSITION_XYZ,
            sampler_cfg=BernoulliSamplerCfg(probability=probability),
        )
    )
    assert variation.enabled is enabled

    return IsaacLabArenaEnvironment(
        name="test_object_disappear_variation",
        scene=Scene(assets=[sphere]),
    )


def get_relation_test_environment(*, probability: float):
    """Build an env whose rigid object is relation-placed on an anchored table.

    Relation solving registers a placement reset event that rewrites every non-anchor object's
    pose on reset, so this is the configuration where a spawn-pose-only implementation is undone.
    """
    from isaaclab_arena.assets.registries import AssetRegistry
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.relations.relations import IsAnchor, On
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.utils.pose import Pose
    from isaaclab_arena.variations.bernoulli_sampler import BernoulliSamplerCfg
    from isaaclab_arena.variations.object_disappear_variation import ObjectDisappearVariationCfg

    asset_registry = AssetRegistry()
    table = asset_registry.get_asset_by_name(TEST_TABLE_NAME)()
    box = asset_registry.get_asset_by_name(TEST_BOX_NAME)()
    table.set_initial_pose(Pose(position_xyz=(0.0, 0.0, 0.0)))
    table.add_relation(IsAnchor())
    box.add_relation(On(table))

    variation = box.get_variation("disappear")
    variation.apply_cfg(
        ObjectDisappearVariationCfg(
            enabled=True,
            away_position_xyz=TEST_AWAY_POSITION_XYZ,
            sampler_cfg=BernoulliSamplerCfg(probability=probability),
        )
    )

    return IsaacLabArenaEnvironment(
        name="test_object_disappear_variation_relations",
        scene=Scene(assets=[table, box]),
    )


def _test_object_disappear_variation_registration(simulation_app):
    from isaaclab_arena.assets.registries import AssetRegistry

    registry = AssetRegistry()
    # Rigid objects carry the variation; non-rigid assets have nothing to teleport.
    assert "disappear" in registry.get_asset_by_name(TEST_ASSET_NAME)().variations
    assert "disappear" not in registry.get_asset_by_name("light")().variations
    assert "disappear" not in registry.get_asset_by_name(TEST_TABLE_NAME)().variations
    return True


def _test_disabled_disappear_variation_not_in_events_cfg(simulation_app):
    from isaaclab_arena.cli.isaaclab_arena_cli import arena_env_builder_cfg_from_argparse, get_isaaclab_arena_cli_parser
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder

    arena_env = get_test_environment(enabled=False, probability=1.0)
    args_cli = get_isaaclab_arena_cli_parser().parse_args(["--num_envs", "1"])
    env_cfg, _ = ArenaEnvBuilder(arena_env, arena_env_builder_cfg_from_argparse(args_cli)).compose_manager_cfg()

    assert not hasattr(env_cfg.events, TEST_EVENT_NAME), (
        f"Disabled variation must not add '{TEST_EVENT_NAME}' to env_cfg.events; "
        f"got event fields: {sorted(vars(env_cfg.events))}."
    )
    return True


def _test_enabled_disappear_variation_in_events_cfg(simulation_app):
    from isaaclab_arena.cli.isaaclab_arena_cli import arena_env_builder_cfg_from_argparse, get_isaaclab_arena_cli_parser
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.variations.object_disappear_variation import hold_object_away

    arena_env = get_test_environment(enabled=True, probability=1.0)
    args_cli = get_isaaclab_arena_cli_parser().parse_args(["--num_envs", "1"])
    env_cfg, _ = ArenaEnvBuilder(arena_env, arena_env_builder_cfg_from_argparse(args_cli)).compose_manager_cfg()

    event_cfg = getattr(env_cfg.events, TEST_EVENT_NAME)
    assert event_cfg.func is hold_object_away
    assert event_cfg.mode == "reset"
    assert event_cfg.params["asset_cfg"].name == TEST_ASSET_NAME
    assert event_cfg.params["disappeared"] is True
    assert event_cfg.params["pose"].position_xyz == TEST_AWAY_POSITION_XYZ
    return True


def _test_disappeared_object_is_held_away(simulation_app):
    from isaaclab_arena.cli.isaaclab_arena_cli import arena_env_builder_cfg_from_argparse, get_isaaclab_arena_cli_parser
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder

    args_cli = get_isaaclab_arena_cli_parser().parse_args(["--num_envs", "1"])
    env = ArenaEnvBuilder(
        get_test_environment(enabled=True, probability=1.0),
        arena_env_builder_cfg_from_argparse(args_cli),
    ).make_registered()
    try:
        env.reset()
        position = get_env_local_position(env, TEST_ASSET_NAME)
        assert (
            position[2] < DISAPPEARED_Z_CEILING_M
        ), f"A disappeared object must be held below {DISAPPEARED_Z_CEILING_M} m; got position {position.tolist()}."
    finally:
        env.close()
    return True


def _test_present_object_keeps_its_pose(simulation_app):
    from isaaclab_arena.cli.isaaclab_arena_cli import arena_env_builder_cfg_from_argparse, get_isaaclab_arena_cli_parser
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder

    args_cli = get_isaaclab_arena_cli_parser().parse_args(["--num_envs", "1"])
    env = ArenaEnvBuilder(
        get_test_environment(enabled=True, probability=0.0),
        arena_env_builder_cfg_from_argparse(args_cli),
    ).make_registered()
    try:
        env.reset()
        position = get_env_local_position(env, TEST_ASSET_NAME)
        assert position[2] > DISAPPEARED_Z_CEILING_M, (
            "An enabled variation that drew 'stay' must leave the object in the scene; "
            f"got position {position.tolist()}."
        )
    finally:
        env.close()
    return True


def _test_disappeared_object_survives_relation_placement(simulation_app):
    """A relation-placed object stays away even though placement rewrites poses on reset."""
    from isaaclab_arena.cli.isaaclab_arena_cli import arena_env_builder_cfg_from_argparse, get_isaaclab_arena_cli_parser
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder

    args_cli = get_isaaclab_arena_cli_parser().parse_args(["--num_envs", "1"])
    env = ArenaEnvBuilder(
        get_relation_test_environment(probability=1.0),
        arena_env_builder_cfg_from_argparse(args_cli),
    ).make_registered()
    try:
        assert hasattr(
            env.unwrapped.cfg.events, "placement_reset"
        ), "Test setup is wrong: relation solving did not register a placement reset event."
        # Reset twice: the first reset is where a spawn-pose-only implementation gets overwritten.
        env.reset()
        env.reset()
        position = get_env_local_position(env, TEST_BOX_NAME)
        assert position[2] < DISAPPEARED_Z_CEILING_M, (
            "Relation placement must not put a disappeared object back into the scene; "
            f"got position {position.tolist()}."
        )
    finally:
        env.close()
    return True


def _test_hydra_override_enables_disappear(simulation_app):
    from isaaclab_arena.cli.isaaclab_arena_cli import arena_env_builder_cfg_from_argparse, get_isaaclab_arena_cli_parser
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder

    args_cli = get_isaaclab_arena_cli_parser().parse_args(["--num_envs", "1"])
    env = ArenaEnvBuilder(
        get_test_environment(enabled=False, probability=0.0),
        arena_env_builder_cfg_from_argparse(args_cli),
        hydra_overrides=[
            f"{TEST_ASSET_NAME}.disappear.enabled=true",
            f"{TEST_ASSET_NAME}.disappear.sampler_cfg.probability=1.0",
        ],
    ).make_registered()
    try:
        env.reset()
        position = get_env_local_position(env, TEST_ASSET_NAME)
        assert (
            position[2] < DISAPPEARED_Z_CEILING_M
        ), f"Hydra override must enable the variation; got position {position.tolist()}."

        record = env.unwrapped.variation_recorder[f"{TEST_ASSET_NAME}.disappear"]
        episode_idx = env.unwrapped.get_episode_index(0)
        assert record.sample_for_episode(0, episode_idx) is True
    finally:
        env.close()
    return True


def test_object_disappear_variation_registration():
    assert run_function_with_persistent_simulation_app(
        _test_object_disappear_variation_registration,
        headless=HEADLESS,
    )


def test_disabled_disappear_variation_not_in_events_cfg():
    assert run_function_with_persistent_simulation_app(
        _test_disabled_disappear_variation_not_in_events_cfg,
        headless=HEADLESS,
    )


def test_enabled_disappear_variation_in_events_cfg():
    assert run_function_with_persistent_simulation_app(
        _test_enabled_disappear_variation_in_events_cfg,
        headless=HEADLESS,
    )


def test_disappeared_object_is_held_away():
    assert run_function_with_persistent_simulation_app(
        _test_disappeared_object_is_held_away,
        headless=HEADLESS,
    )


def test_present_object_keeps_its_pose():
    assert run_function_with_persistent_simulation_app(
        _test_present_object_keeps_its_pose,
        headless=HEADLESS,
    )


def test_disappeared_object_survives_relation_placement():
    assert run_function_with_persistent_simulation_app(
        _test_disappeared_object_survives_relation_placement,
        headless=HEADLESS,
    )


def test_hydra_override_enables_disappear():
    assert run_function_with_persistent_simulation_app(
        _test_hydra_override_enables_disappear,
        headless=HEADLESS,
    )
