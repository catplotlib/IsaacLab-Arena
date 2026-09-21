# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Ordinary placements are recorded after physics, then replayed without solving."""

from pathlib import Path

import pytest

from isaaclab_arena.tests.utils.constants import TestConstants
from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app
from isaaclab_arena.tests.utils.subprocess import run_subprocess


def register_no_embodiment():
    from isaaclab_arena.assets.registries import AssetRegistry
    from isaaclab_arena.embodiments.no_embodiment import NoEmbodiment

    AssetRegistry().register(NoEmbodiment, key="recording_no_embodiment")


def _write_scene(path):
    import yaml

    for name, kinematic, size in (
        ("table", "true", (0.8, 0.8, 0.04)),
        ("cube", "false", (0.05, 0.1, 0.1)),
        ("floor", "true", (4.0, 4.0, 0.04)),
    ):
        (path.parent / f"{name}.usda").write_text(f"""#usda 1.0
(
    defaultPrim = "Body"
    metersPerUnit = 1
    upAxis = "Z"
)
def Xform "Body" (prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]) {{
    bool physics:kinematicEnabled = {kinematic}
    float physics:mass = 1
    def Cube "geometry" (prepend apiSchemas = ["PhysicsCollisionAPI"]) {{
        double size = 1
        double3 xformOp:scale = {size}
        uniform token[] xformOpOrder = ["xformOp:scale"]
    }}
}}
""")
    data = {
        "env_name": "placement_recording",
        "embodiment": {"id": "robot", "registry_name": "recording_no_embodiment"},
        "background": {
            "id": "table",
            "registry_name": "simready_usd_object",
            "params": {
                "usd_path": str(path.parent / "table.usda"),
                "instance_name": "table",
                "initial_pose": {"position_xyz": [0, 0, 0.5]},
            },
        },
        "objects": [{
            "id": "cube",
            "registry_name": "simready_usd_object",
            "params": {"usd_path": str(path.parent / "cube.usda"), "instance_name": "cube_body"},
        }],
        "relations": [
            {"kind": "is_anchor", "subject": "table"},
            {"kind": "on", "subject": "cube", "reference": "table", "params": {"clearance_m": 0.03}},
        ],
        "task": {
            "composition": "atomic",
            "description": "record settled placements",
            "subtasks": [{"kind": "NoTask", "params": {}}],
        },
    }
    data["objects"].append({
        "id": "floor",
        "registry_name": "simready_usd_object",
        "params": {
            "usd_path": str(path.parent / "floor.usda"),
            "instance_name": "floor",
            "initial_pose": {"position_xyz": [0, 0, -0.5]},
        },
    })
    path.write_text(yaml.safe_dump(data))


@pytest.mark.with_subprocess
@pytest.mark.parametrize("backend", ["physx", "newton"])
def test_recording_cli_saves_final_poses(tmp_path, backend):
    import json

    source, output = tmp_path / "scene.yaml", tmp_path / "placements.jsonl"
    _write_scene(source)
    run_subprocess(
        [
            TestConstants.python_path,
            str(Path(TestConstants.scripts_dir) / "record_placement_layouts.py"),
            f"env_spec={source}",
            f"output={output}",
            f"presets={backend}",
            "num_envs=2",
            "layouts_per_env=2",
            "register=[isaaclab_arena.tests.test_settled_placement:register_no_embodiment]",
            "--viz",
            "none",
        ],
        timeout_sec=180,
    )
    records = [json.loads(line)["variations"]["scene.relation_placement"] for line in output.read_text().splitlines()]
    assert len(records) == 4
    for record in records:
        assert record["source"] == "settled"
        # Table top is 0.52, cube half-height is 0.05. The solver's 3 cm clearance is gone.
        assert set(record["poses"]) == {"cube"}
        x, y, _ = record["poses"]["cube"]["position_xyz"]
        assert abs(x) < 0.4 and abs(y) < 0.4
        assert record["poses"]["cube"]["position_xyz"][2] == pytest.approx(0.57, abs=0.005)

    positions = {tuple(record["poses"]["cube"]["position_xyz"]) for record in records}
    assert len(positions) > 1


def _test_filter_and_replay(simulation_app, tmp_path):
    import torch
    import yaml
    from dataclasses import replace
    from unittest.mock import patch

    from isaaclab_arena.environment_spec.arena_env_graph_conversion_utils import (
        build_arena_env_with_assets_from_graph_spec,
        get_scene_key_to_node_id,
    )
    from isaaclab_arena.environment_spec.arena_env_graph_spec import ArenaEnvGraphSpec
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.relations.physics_settle_params import PlacementRecordingParams
    from isaaclab_arena.relations.placement_events import get_placement_pool
    from isaaclab_arena.relations.placement_layouts import PlacementLayouts
    from isaaclab_arena.relations.placement_validation import PlacementCheck
    from isaaclab_arena.relations.relation_solver import RelationSolver
    from isaaclab_arena.relations.settled_placement import collect_settled_pool_layouts

    register_no_embodiment()
    source = tmp_path / "scene.yaml"
    _write_scene(source)
    data = yaml.safe_load(source.read_text())
    data["relations"][1]["params"]["overlap"] = True
    source.write_text(yaml.safe_dump(data))
    spec = ArenaEnvGraphSpec.from_yaml(source)
    arena_env, assets = build_arena_env_with_assets_from_graph_spec(spec)
    node_by_key = get_scene_key_to_node_id(spec, assets)
    arena_env.placer_params = replace(
        arena_env.placer_params,
        min_unique_layouts_per_env=1,
        placement_seed=42,
        random_yaw_init=False,
        enabled_checks={PlacementCheck.NO_OVERLAP},
        required_checks={PlacementCheck.NO_OVERLAP},
    )
    env = ArenaEnvBuilder(arena_env, ArenaEnvBuilderCfg(num_envs=2)).make_registered()
    try:
        env.reset()
        base = env.unwrapped
        pool = get_placement_pool(env)
        queues = pool.layouts_per_env()
        cube = arena_env.scene.assets["cube_body"]
        initial = base.arena_world.get_pose_e("cube_body").clone()
        # Both pass On(overlap=True); only the overhanging cube falls to the floor.
        queues[0][0].positions[cube] = (0.0, 0.0, 0.60)
        queues[1][0].positions[cube] = (0.42, 0.0, 0.60)
        from isaaclab_arena.relations.bounding_box_helpers import build_per_env_bounding_boxes
        from isaaclab_arena.relations.placement_validators import OnRelationValidator

        boxes = build_per_env_bounding_boxes(pool.objects, 2).get_bounding_boxes_for_all_envs()
        assert OnRelationValidator(pool.placer_params).validate_batch(
            [queue[0].positions for queue in queues], [{}, {}], boxes, []
        ) == [True, True]
        saved_positions = [dict(queue[0].positions) for queue in queues]
        result = collect_settled_pool_layouts(
            env, pool, PlacementRecordingParams(max_translation_m=10, max_rotation_deg=180)
        )
        layouts = result.layouts
        assert result.attempted == 2
        assert result.accepted_indices == [(0, 0)]
        assert "on_relation" in result.rejections[1, 0]
        assert layouts.num_layouts == 1
        assert layouts.poses["cube_body"][0].position_xyz[2] == pytest.approx(0.57, abs=0.005)
        torch.testing.assert_close(base.arena_world.get_pose_e("cube_body"), initial)
        assert [queue[0].positions for queue in queues] == saved_positions
        with pytest.raises(AssertionError, match="Accepted 0 layouts"):
            collect_settled_pool_layouts(env, pool, PlacementRecordingParams(max_translation_m=0.001))
        torch.testing.assert_close(base.arena_world.get_pose_e("cube_body"), initial)
        output = tmp_path / "poses.jsonl"
        PlacementLayouts({node_by_key[key]: poses for key, poses in layouts.poses.items()}).write_episode_jsonl(output)
    finally:
        env.close()
    cache = PlacementLayouts.from_episode_jsonl(output)
    with patch.object(RelationSolver, "solve", side_effect=AssertionError("Replay must not solve")):
        arena_env = spec.to_arena_env(placement_layouts_path=output)
        env = ArenaEnvBuilder(arena_env, ArenaEnvBuilderCfg(num_envs=3)).make_registered()
        try:
            for _ in range(2):
                env.reset()
                actual = env.unwrapped.arena_world.get_pose_e("cube_body")
                expected = cache.poses["cube"][0].to_tensor(env.unwrapped.device).expand(3, 7)
                torch.testing.assert_close(actual, expected, atol=2e-5, rtol=0)
                torch.testing.assert_close(
                    env.unwrapped.scene["cube_body"].data.root_vel_w.torch, torch.zeros((3, 6), device=actual.device)
                )
        finally:
            env.close()
    return True


def test_recording_filters_and_replays(tmp_path):
    assert run_function_with_persistent_simulation_app(_test_filter_and_replay, tmp_path=tmp_path)


def _test_mixed_recording(simulation_app, tmp_path):
    import torch
    import yaml
    from dataclasses import replace
    from unittest.mock import patch

    from isaaclab_arena.environment_spec.arena_env_graph_conversion_utils import (
        build_arena_env_with_assets_from_graph_spec,
        get_scene_key_to_node_id,
    )
    from isaaclab_arena.environment_spec.arena_env_graph_spec import ArenaEnvGraphSpec
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.relations.physics_settle_params import PlacementRecordingParams
    from isaaclab_arena.relations.placement_events import get_placement_pool
    from isaaclab_arena.relations.placement_layouts import PlacementLayouts
    from isaaclab_arena.relations.relation_solver import RelationSolver
    from isaaclab_arena.relations.settled_placement import collect_settled_pool_layouts

    source = tmp_path / "mixed.yaml"
    _write_scene(source)
    data = yaml.safe_load(source.read_text())
    data["embodiment"] = {"id": "robot", "registry_name": "franka_ik"}
    data["relations"].append({"kind": "at_position", "subject": "robot", "params": {"x": -1.0, "y": 0.0, "z": 0.0}})
    rod_path = tmp_path / "rod.usda"
    rod_path.write_text((tmp_path / "cube.usda").read_text().replace("(0.05, 0.1, 0.1)", "(0.12, 0.025, 0.025)"))
    for index in range(5):
        name = f"clutter_{index}"
        data["objects"].append({
            "id": name,
            "registry_name": "simready_usd_object",
            "params": {"usd_path": str(rod_path), "instance_name": name},
        })
        data["relations"].append({
            "kind": "clutter_on",
            "subject": name,
            "reference": "table",
            "params": {"spread": 0.3, "clearance_m": 0.02, "gap_m": 0.02},
        })
    source.write_text(yaml.safe_dump(data))
    spec = ArenaEnvGraphSpec.from_yaml(source)
    arena, assets = build_arena_env_with_assets_from_graph_spec(spec)
    node_by_key = get_scene_key_to_node_id(spec, assets)
    arena.placer_params = replace(
        arena.placer_params, min_unique_layouts_per_env=1, placement_seed=42, allow_best_loss_fallbacks=False
    )
    env = ArenaEnvBuilder(arena, ArenaEnvBuilderCfg(num_envs=1)).make_registered()
    try:
        env.reset()
        scene = env.unwrapped.scene
        state = scene.get_state()
        robot = scene.articulations["robot"]
        targets = robot.data.joint_pos_target.torch.clone()
        result = collect_settled_pool_layouts(
            env, get_placement_pool(env), PlacementRecordingParams(max_translation_m=0.04, settle_time_s=8.0)
        )
        assert result.layouts.num_layouts == 1
        assert set(result.layouts.poses) == {"cube_body", "robot", *(f"clutter_{i}" for i in range(5))}
        fell = []
        for index in range(5):
            name = f"clutter_{index}"
            z = result.layouts.poses[name][0].position_xyz[2]
            assert z >= 0.53
            fell.append(float(state["rigid_object"][name]["root_pose"][0, 2]) - z)
        assert max(fell) > 0.04  # Expected clutter drops are exempt from the ordinary-object limit.
        restored = scene.get_state()
        for kind, assets in state.items():
            for name, values in assets.items():
                for field, value in values.items():
                    torch.testing.assert_close(restored[kind][name][field], value)
        torch.testing.assert_close(robot.data.joint_pos_target.torch, targets)
        output = tmp_path / "mixed.jsonl"
        PlacementLayouts({node_by_key[key]: poses for key, poses in result.layouts.poses.items()}).write_episode_jsonl(
            output
        )
        with pytest.raises(AssertionError, match="joint states are not recorded"):
            collect_settled_pool_layouts(
                env,
                get_placement_pool(env),
                PlacementRecordingParams(settle_time_s=8.0, max_link_translation_m=0, max_link_rotation_deg=0),
            )
        restored = scene.get_state()
        for kind, assets in state.items():
            for name, values in assets.items():
                for field, value in values.items():
                    torch.testing.assert_close(restored[kind][name][field], value)
        torch.testing.assert_close(robot.data.joint_pos_target.torch, targets)
    finally:
        env.close()
    with patch.object(RelationSolver, "solve", side_effect=AssertionError("Replay must not solve")):
        env = ArenaEnvBuilder(
            spec.to_arena_env(placement_layouts_path=output), ArenaEnvBuilderCfg(num_envs=2)
        ).make_registered()
        try:
            env.reset()
            for key, poses in result.layouts.poses.items():
                expected = poses[0].to_tensor(env.unwrapped.device).expand(2, 7)
                torch.testing.assert_close(env.unwrapped.arena_world.get_pose_e(key), expected, atol=2e-5, rtol=0)
            from isaaclab_arena.utils.physics_settle import step_physics

            step_physics(env, 200)
            for key, poses in result.layouts.poses.items():
                if key == "robot":
                    continue
                expected = poses[0].to_tensor(env.unwrapped.device)[:3]
                actual = env.unwrapped.arena_world.get_pose_e(key)[:, :3]
                assert (actual - expected).norm(dim=-1).max() < 0.02
        finally:
            env.close()
    return True


def test_mixed_recording_restores_scene_and_replays(tmp_path):
    assert run_function_with_persistent_simulation_app(_test_mixed_recording, tmp_path=tmp_path)


def test_recording_does_not_drop_initialized_ik_check():
    from unittest.mock import Mock

    from isaaclab_arena.relations.object_placer_params import ObjectPlacerParams
    from isaaclab_arena.relations.placement_validation import PlacementCheck
    from isaaclab_arena.relations.placement_validators import PlacementValidator
    from isaaclab_arena.relations.pooled_object_placer import PooledObjectPlacer
    from isaaclab_arena.relations.settled_placement import _final_pose_validators

    validator = Mock(spec=PlacementValidator)
    validator.check = PlacementCheck.IK_REACHABLE
    pool = Mock(spec=PooledObjectPlacer)
    pool.validators = [validator]
    # The build-time embodiment context is gone, but the initialized check still gates the pool.
    pool.placer_params = ObjectPlacerParams()
    assert validator in _final_pose_validators(pool)
    pool.placer_params.required_checks = {PlacementCheck.NO_OVERLAP}
    assert validator in _final_pose_validators(pool)

    import torch

    from isaaclab_arena.relations.settled_placement import _validate_final_pose
    from isaaclab_arena.tests.dummy_object import DummyObject
    from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox
    from isaaclab_arena.utils.pose import Pose

    bounds = AxisAlignedBoundingBox((-0.1, -0.1, -0.1), (0.1, 0.1, 0.1))
    asset = DummyObject("target", bounds)
    measured = Pose((0.2, 0.3, 0.4), (0.6, 0, 0, 0.8))
    env = Mock()
    env.arena_world.get_pose_e.return_value = measured.to_tensor("cpu").reshape(1, 7)
    pool.collision_objects = []
    validator.validate_measured_poses.return_value = False
    from isaaclab_arena.relations.physics_settle_params import PlacementRecordingParams

    assert (
        _validate_final_pose(env, 0, {asset: bounds}, [validator], pool, PlacementRecordingParams())
        == "final pose failed ik_reachable"
    )
    forwarded = validator.validate_measured_poses.call_args.args[0][asset]
    torch.testing.assert_close(forwarded.to_tensor("cpu"), measured.to_tensor("cpu"))
