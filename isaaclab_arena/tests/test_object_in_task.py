# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Exercise AABB containment independently of simulation dynamics."""

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app


def _test_aabb_containment(_simulation_app):
    import torch
    from types import SimpleNamespace

    from isaaclab_arena.tasks.object_in_task import ObjectInTask
    from isaaclab_arena.tasks.predicates.spatial import object_in_target_aabb
    from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox

    # Inside, protruding despite an inside center, coincident, and outside.
    target = torch.tensor([[[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]]]).repeat(4, 1, 1)
    objects = torch.tensor([
        [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        [[-0.5, -0.5, -0.5], [1.1, 0.5, 0.5]],
        [[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]],
        [[2.0, 2.0, 2.0], [3.0, 3.0, 3.0]],
    ])
    geometry = {"object": objects, "target": target}
    world = SimpleNamespace(
        get_vertices_w=lambda name: geometry[name],
        get_aabb_w=lambda name: AxisAlignedBoundingBox(geometry[name].amin(dim=1), geometry[name].amax(dim=1)),
    )
    env = SimpleNamespace(arena_world=world)
    assert object_in_target_aabb(env, "object", "target").tolist() == [True, False, True, False]
    # Shared translations preserve containment; moving the target changes it.
    geometry["object"] = objects + 10
    geometry["target"] = target + 10
    assert object_in_target_aabb(env, "object", "target").tolist() == [True, False, True, False]
    geometry["target"] = target
    assert not object_in_target_aabb(env, "object", "target").any()
    task = ObjectInTask(SimpleNamespace(name="object"), SimpleNamespace(name="target"))
    termination = task.get_termination_cfg()
    assert termination.success[0].name == "object_in"
    assert not termination.failures
    return True


def test_aabb_containment():
    assert run_function_with_persistent_simulation_app(_test_aabb_containment)


def _test_syringe_task_composition(_simulation_app):
    import yaml
    from pathlib import Path
    from types import SimpleNamespace

    from isaaclab_arena.environment_spec.arena_env_graph_task_conversion_utils import build_task_from_spec
    from isaaclab_arena.environment_spec.arena_env_graph_types import CompositeTaskSpec
    from isaaclab_arena.tasks.composite_task_base import CompositeTaskBase
    from isaaclab_arena.tasks.object_in_task import ObjectInTask

    folder = Path(__file__).parents[2] / "isaaclab_arena_environments/isaac_cap/syringe_sort/environments"
    assets = {f"syringe_{i}": SimpleNamespace(name=f"syringe_{i}") for i in range(4)}
    assets["sharps_container"] = SimpleNamespace(name="sharps_container")
    for filename, count, timeout in (
        ("syringe_single.yaml", 1, 228.0),
        ("syringe_both.yaml", 2, 456.0),
        ("syringe_cluttered.yaml", 4, 1368.0),
    ):
        spec = CompositeTaskSpec.model_validate(yaml.safe_load((folder / filename).read_text())["task"])
        task = build_task_from_spec(spec, assets)
        if count == 1:
            assert isinstance(task, ObjectInTask)
            subtasks = [task]
        else:
            assert isinstance(task, CompositeTaskBase)
            assert not task.subtasks_are_sequential
            subtasks = task.subtasks
        assert len(subtasks) == count
        for index, subtask in enumerate(subtasks):
            assert isinstance(subtask, ObjectInTask)
            assert subtask.object is assets[f"syringe_{index}"]
            assert subtask.target is assets["sharps_container"]
        termination = task.get_termination_cfg()
        assert termination.timeout_s == timeout
        assert len(termination.success) == count
        assert len({objective.name for objective in termination.success}) == count
    return True


def test_syringe_task_composition():
    assert run_function_with_persistent_simulation_app(_test_syringe_task_composition)


def _test_world_aabb(_simulation_app):
    import torch
    from types import SimpleNamespace

    from isaaclab_arena.environments.arena_world import ArenaWorld
    from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox

    scene = SimpleNamespace(num_envs=2, deformable_objects={})
    world = ArenaWorld(scene)
    bounds = AxisAlignedBoundingBox(min_point=(-1.0, -2.0, -3.0), max_point=(1.0, 2.0, 3.0))
    world._aabbs_in_local_frame_cache["object"] = bounds
    # Rotate the second asset 90 degrees about Z; environments have distinct world positions.
    poses = torch.tensor([[10.0, 20.0, 30.0, 0.0, 0.0, 0.0, 1.0], [40.0, 50.0, 60.0, 0.0, 0.0, 2**-0.5, 2**-0.5]])
    world.get_pose_w = lambda name: poses
    aabb = world.get_aabb_w("object")
    torch.testing.assert_close(aabb.min_point, torch.tensor([[9.0, 18.0, 27.0], [38.0, 49.0, 57.0]]))
    torch.testing.assert_close(aabb.max_point, torch.tensor([[11.0, 22.0, 33.0], [42.0, 51.0, 63.0]]))
    poses[:, :3] += 5
    moved = world.get_aabb_w("object")
    torch.testing.assert_close(moved.min_point, aabb.min_point + 5)
    torch.testing.assert_close(moved.max_point, aabb.max_point + 5)
    return True


def test_world_aabb():
    assert run_function_with_persistent_simulation_app(_test_world_aabb)
