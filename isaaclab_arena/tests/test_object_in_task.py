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
    assert object_in_target_aabb(env, "object", "target", minimum_contained_fraction=0.9).tolist() == [
        True,
        True,
        True,
        False,
    ]
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


def _test_target_contact(_simulation_app):
    import torch
    from types import SimpleNamespace

    from isaaclab.managers import SceneEntityCfg

    from isaaclab_arena.tasks.predicates.spatial import object_in_target_contact

    # No target contact, side contact, and opposing contacts on separate target bodies.
    forces = torch.tensor([
        [[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]],
        [[[0.1, 0.0, 0.0], [0.0, 0.0, 0.0]]],
        [[[0.0, 0.0, 0.1], [0.0, 0.0, -0.1]]],
    ])
    sensor = SimpleNamespace(data=SimpleNamespace(force_matrix_w=SimpleNamespace(torch=forces)))
    env = SimpleNamespace(scene={"contact": sensor})
    assert object_in_target_contact(env, SceneEntityCfg("contact"), 0.01).tolist() == [False, True, True]
    return True


def test_target_contact():
    assert run_function_with_persistent_simulation_app(_test_target_contact)
