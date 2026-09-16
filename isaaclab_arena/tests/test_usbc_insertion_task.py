# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Validate the ported USB-C task semantics and reset ranges."""

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app


def _test_usbc_insertion_task(_simulation_app) -> bool:
    import torch
    from types import SimpleNamespace

    from isaaclab.managers import TerminationManager

    from isaaclab_arena.assets.asset import Asset
    from isaaclab_arena.assets.registries import EnvironmentRegistry, TaskRegistry
    from isaaclab_arena.tasks.predicates.composite import CompositePredicate
    from isaaclab_arena.tasks.predicates.spatial import (
        depth_in_range,
        tilt_axis_aligned,
        velocity_below_threshold,
        xy_in_proximity,
    )
    from isaaclab_arena_environments.isaac_cap import register_components
    from isaaclab_arena_environments.isaac_cap.usbc_insertion.task import UsbcInsertionTask

    class _World:
        def __init__(self):
            self.poses = {
                "plug": torch.tensor([[0.012, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]),
                "receiver": torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]),
            }
            self.velocity = torch.zeros((1, 3))

        def get_pose_w(self, name):
            return self.poses[name]

        def get_root_linear_velocity_w(self, _name):
            return self.velocity

    env = SimpleNamespace(num_envs=1, device="cpu", arena_world=_World())
    mating = {
        "subject_name": "plug",
        "receiver_name": "receiver",
        "subject_offset_xyz": (0.0, 0.0, 0.0),
        "target_offset_xyz": (0.0, 0.0, 0.0),
        "receiver_axis": (1.0, 0.0, 0.0),
    }
    assert depth_in_range(env, **mating, depth_min=0.01, depth_max=0.02).item()
    assert xy_in_proximity(env, **mating, tolerance_xy=0.001).item()
    assert tilt_axis_aligned(
        env,
        subject_name="plug",
        receiver_name="receiver",
        subject_axis=(1.0, 0.0, 0.0),
        receiver_axis=(1.0, 0.0, 0.0),
        max_tilt_rad=0.01,
    ).item()
    env.arena_world.poses["plug"][0, 1] = 0.002
    assert not xy_in_proximity(env, **mating, tolerance_xy=0.001).item()

    easy_task = UsbcInsertionTask(
        Asset("plug"),
        Asset("receiver"),
        receiver_mouth_offset_xyz=(-0.0045, 0.0, 0.0),
        receiver_axis=(1.0, 0.0, 0.0),
        subject_tip_offset_xyz=(0.0, 0.0, 0.0133),
        depth_min=0.0104,
        lateral_max=0.0087931792,
        speed_max=0.05,
        tilt_max=0.0873,
        allow_antiparallel_axes=True,
        episode_length_s=150.0,
    )
    success_cfg = easy_task.get_termination_cfg().success
    assert success_cfg.func is CompositePredicate
    assert success_cfg.params["consecutive_steps"] == 1
    predicates = success_cfg.params["predicates"]
    assert [predicate.func for predicate in predicates] == [
        depth_in_range,
        xy_in_proximity,
        tilt_axis_aligned,
        velocity_below_threshold,
    ]
    assert predicates[0].params["depth_max"] is None

    medium_task = UsbcInsertionTask(
        Asset("plug"),
        Asset("receiver"),
        receiver_mouth_offset_xyz=(0.0, 0.0, 0.0065),
        receiver_axis=(0.0, 0.0, -1.0),
        subject_tip_offset_xyz=(0.0, 0.0, 0.00665),
        depth_min=0.0052,
        depth_max=0.0085,
        lateral_max=0.0043965896,
        speed_max=0.05,
    )
    predicates = medium_task.get_termination_cfg().success.params["predicates"]
    assert [predicate.func for predicate in predicates] == [
        depth_in_range,
        xy_in_proximity,
        velocity_below_threshold,
    ]
    assert predicates[0].params["depth_max"] == 0.0085

    manager_env = SimpleNamespace(
        num_envs=1,
        device="cpu",
        arena_world=_World(),
        sim=SimpleNamespace(is_playing=lambda: True),
    )
    manager = TerminationManager({"success": success_cfg}, manager_env)
    assert manager.get_term_cfg("success").func.consecutive_true_steps.tolist() == [0]
    manager.reset([0])
    assert manager.get_term_cfg("success").func.consecutive_true_steps.tolist() == [0]

    register_components()
    assert TaskRegistry().get_task_by_name("UsbcInsertionTask") is UsbcInsertionTask
    environment_registry = EnvironmentRegistry()
    for name in (
        "vabar_contact_rich_insertion__usbc_insertion_easy",
        "vabar_contact_rich_insertion__usbc_insertion_medium",
    ):
        assert environment_registry.is_registered(name)
    return True


def test_usbc_insertion_task() -> None:
    assert run_function_with_persistent_simulation_app(_test_usbc_insertion_task)


def _test_usbc_asset_registration(_simulation_app) -> bool:
    from isaaclab_arena.assets.object_type import ObjectType
    from isaaclab_arena.assets.registries import AssetRegistry
    from isaaclab_arena_environments.isaac_cap import register_components

    registry = AssetRegistry()
    plug_class = registry.get_asset_by_name("usbc_insertion_plug")

    from isaaclab_arena_environments.isaac_cap.usbc_insertion.assets import ASSET_ROOT, USBC_ASSET_CLASSES, UsbcPlug
    from isaaclab_arena_environments.isaac_cap.usbc_insertion.environment import (
        UsbcInsertionEasyEnvironment,
        UsbcInsertionEasyEnvironmentCfg,
        UsbcInsertionMediumEnvironment,
        UsbcInsertionMediumEnvironmentCfg,
    )

    assert plug_class is UsbcPlug
    assert len(USBC_ASSET_CLASSES) == 12
    register_components()
    register_components()
    for asset_class in USBC_ASSET_CLASSES:
        assert registry.get_asset_by_name(asset_class.name) is asset_class
        assert "usbc_insertion" in asset_class.tags
        if "light" not in asset_class.tags and "cable" not in asset_class.tags:
            assert asset_class.usd_path.startswith(f"{ASSET_ROOT}/")

    precision_class = registry.get_asset_by_name("usbc_insertion_precision_plug")
    precision_plug = precision_class(instance_name="precision_plug", prim_path="{ENV_REGEX_NS}/PrecisionPlug")
    assert precision_plug.name == "precision_plug"
    assert precision_plug.prim_path == "{ENV_REGEX_NS}/PrecisionPlug"
    assert precision_plug.scale == (0.5, 0.5, 0.5)
    assert precision_plug.object_cfg.spawn.mass_props.mass == 0.004
    assert precision_plug.object_cfg.spawn.physics_material.static_friction == 1.1
    assert precision_plug.object_cfg.spawn.physics_material.dynamic_friction == 1.1
    assert precision_plug.reset_pose
    assert not precision_plug.object_cfg.spawn.rigid_props.kinematic_enabled

    precision_plug.spawn_cfg_addon["mass_props"].mass = 9.0
    precision_plug.spawn_cfg_addon["collision_props"][0].contact_gap = 0.5
    another_plug = precision_class()
    assert another_plug.spawn_cfg_addon["mass_props"].mass == 0.004
    assert another_plug.spawn_cfg_addon["collision_props"][0].contact_gap == 1.0e-4
    full_size_plug = plug_class()
    assert full_size_plug.scale == (1.0, 1.0, 1.0)
    assert "mass_props" not in full_size_plug.spawn_cfg_addon
    assert full_size_plug.spawn_cfg_addon["collision_props"][0].contact_gap == 1.0e-4

    for name in ("port", "bench", "cradle_front", "cradle_rear"):
        fixture = registry.get_asset_by_name(f"usbc_insertion_{name}")()
        assert fixture.object_type == ObjectType.RIGID
        assert fixture.object_cfg.spawn.rigid_props.kinematic_enabled
        assert fixture.reset_pose

    easy = UsbcInsertionEasyEnvironment().build(UsbcInsertionEasyEnvironmentCfg())
    medium = UsbcInsertionMediumEnvironment().build(UsbcInsertionMediumEnvironmentCfg())
    asset_types = {type(asset) for env in (easy, medium) for asset in env.scene.assets.values()}
    assert asset_types == set(USBC_ASSET_CLASSES)
    assert type(easy.task.plug) is plug_class
    assert type(medium.task.plug) is precision_class
    assert easy.task.plug.name == "plug" and easy.task.receiver.name == "bulkhead"
    assert medium.task.plug.name == "usbc_plug" and medium.task.receiver.name == "usbc_port"
    for scene in (easy.scene, medium.scene):
        assert scene.assets["table"].object_type == ObjectType.BASE
        assert scene.assets["table"].prim_path == "{ENV_REGEX_NS}/Table"
        shadow_receiver = scene.assets["hdr_shadow_receiver"]
        assert not shadow_receiver.object_cfg.spawn.visible
        assert shadow_receiver.object_cfg.collision_group == -1
    return True


def test_usbc_asset_registration() -> None:
    assert run_function_with_persistent_simulation_app(_test_usbc_asset_registration)


def _test_usbc_environment_yaml(_simulation_app) -> bool:
    import math

    import pytest

    from isaaclab_arena.assets.background import Background
    from isaaclab_arena.environment_spec.arena_env_graph_spec import ArenaEnvGraphSpec
    from isaaclab_arena.utils.pose import PoseRange
    from isaaclab_arena_environments.isaac_cap.usbc_insertion.cables import UsbcConnectorCable, reset_connector_cable
    from isaaclab_arena_environments.isaac_cap.usbc_insertion.environment import (
        UsbcInsertionEasyEnvironment,
        UsbcInsertionEasyEnvironmentCfg,
        UsbcInsertionMediumEnvironment,
        UsbcInsertionMediumEnvironmentCfg,
    )
    from isaaclab_arena_environments.isaac_cap.usbc_insertion.physics import (
        configure_easy_usbc_physics,
        configure_medium_usbc_physics,
    )

    easy_factory = UsbcInsertionEasyEnvironment()
    easy = easy_factory.build(UsbcInsertionEasyEnvironmentCfg(use_tiled_cameras=True, use_instanceable_meshes=True))
    medium_factory = UsbcInsertionMediumEnvironment()
    medium = medium_factory.build(UsbcInsertionMediumEnvironmentCfg(use_tiled_cameras=True))
    assert easy.name == easy_factory.name
    assert medium.name == medium_factory.name
    assert easy.env_cfg_callback is configure_easy_usbc_physics
    assert medium.env_cfg_callback is configure_medium_usbc_physics
    for environment in (easy, medium):
        table = environment.scene.assets["table"]
        assert isinstance(table, Background)
        assert table.reset_nested_physics
        assert table.get_event_cfg()[1] is None
        receiver_event = environment.task.receiver.get_event_cfg()[1]
        assert receiver_event is not None and receiver_event.mode == "reset"
    easy_events = easy.scene.get_events_cfg()
    for name, links in (("plug_cable", 24), ("bulkhead_cable", 8)):
        cable = easy.scene.assets[name]
        assert isinstance(cable, UsbcConnectorCable)
        event = getattr(easy_events, name)
        assert event.func is reset_connector_cable
        assert event.params == {"prim_path": cable.prim_path, "links": links}
        assert event.mode == "reset"
        assert name not in medium.scene.assets
    assert easy.task.get_events_cfg() is None
    assert easy.embodiment.scene_config.left_robot.spawn.usd_path.endswith("i2rt_yam_instanceable.usda")
    assert easy.embodiment.scene_config.right_robot.spawn.usd_path.endswith("i2rt_yam_instanceable.usda")
    assert medium.embodiment.camera_config.use_tiled_camera

    easy_range = easy.task.plug.get_initial_pose()
    medium_range = medium.task.plug.get_initial_pose()
    assert isinstance(easy_range, PoseRange) and isinstance(medium_range, PoseRange)
    assert easy_range.position_xyz_min == pytest.approx((0.42, -0.09, 0.811))
    assert easy_range.position_xyz_max == pytest.approx((0.46, -0.05, 0.811))
    assert easy_range.rpy_min == easy_range.rpy_max == (-math.pi / 2, -math.pi / 2, 0.0)
    assert medium_range.position_xyz_min == pytest.approx((-0.07, -0.05, 0.7825))
    assert medium_range.position_xyz_max == pytest.approx((-0.03, -0.03, 0.7825))
    assert medium_range.rpy_min == (math.pi / 2, 0.0, -math.pi)
    assert medium_range.rpy_max == (math.pi / 2, 0.0, math.pi)
    assert easy.task.receiver.get_initial_pose().position_xyz == (0.44, 0.0148, 0.8234)
    assert medium.task.receiver.get_initial_pose().position_xyz == (-0.05, -0.1, 0.805)

    for factory, env in ((easy_factory, easy), (medium_factory, medium)):
        assert env.task.plug is env.scene.assets[env.task.plug.name]
        assert env.task.receiver is env.scene.assets[env.task.receiver.name]
        light = env.scene.assets["sky_light"]
        assert light.spawner_cfg.intensity == 1500.0
        assert tuple(light.spawner_cfg.color) == (0.75, 0.75, 0.75)
        assert light.spawner_cfg.texture_file
        spec = ArenaEnvGraphSpec.from_yaml(factory.scene_spec)
        spec.task.subtasks[0].params["depth_min"] = 0.0055
        receiver = next(asset for asset in spec.objects if asset.id == env.task.receiver.name)
        receiver.params["initial_pose"]["position_xyz"][0] += 0.01
        edited = spec.to_arena_env()
        assert edited.task.get_termination_cfg().success.params["predicates"][0].params["depth_min"] == 0.0055
        assert edited.task.receiver.get_initial_pose().position_xyz[0] == pytest.approx(
            env.task.receiver.get_initial_pose().position_xyz[0] + 0.01
        )
    return True


def test_usbc_environment_yaml() -> None:
    assert run_function_with_persistent_simulation_app(_test_usbc_environment_yaml)


def _test_usbc_cable_reset_isolation(_simulation_app) -> bool:
    import numpy as np
    import torch
    from types import SimpleNamespace
    from unittest.mock import patch

    import warp as wp
    from isaaclab_newton.physics import NewtonManager

    from isaaclab_arena_environments.isaac_cap.usbc_insertion.cables import reset_connector_cable

    labels = [
        f"/World/envs/env_{world}/{name}/bend{joint}"
        for world in range(2)
        for name in ("TestCable", "TestCableOther")
        for joint in range(2)
    ]
    model = SimpleNamespace(
        joint_label=labels,
        joint_q_start=wp.array(np.arange(9, dtype=np.int32), device="cpu"),
        joint_qd_start=wp.array(np.arange(9, dtype=np.int32), device="cpu"),
    )
    states = [
        SimpleNamespace(
            joint_q=wp.array(np.full(8, 0.25, dtype=np.float32), device="cpu"),
            joint_qd=wp.array(np.full(8, 0.1, dtype=np.float32), device="cpu"),
        )
        for _ in range(2)
    ]
    with (
        patch.object(NewtonManager, "get_model", return_value=model),
        patch.object(NewtonManager, "get_state_0", return_value=states[0]),
        patch.object(NewtonManager, "get_state_1", return_value=states[1]),
        patch.object(NewtonManager, "invalidate_fk") as invalidate,
    ):
        env = SimpleNamespace(num_envs=2)
        params = {"prim_path": "{ENV_REGEX_NS}/TestCable", "links": 2}
        reset_connector_cable(env, [], **params)
        invalidate.assert_not_called()
        reset_connector_cable(env, torch.tensor([0]), **params)
        invalidate.assert_called_once()
        for state in states:
            np.testing.assert_allclose(state.joint_q.numpy(), [0, 0, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25])
            np.testing.assert_allclose(state.joint_qd.numpy(), [0, 0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        reset_connector_cable(env, None, **params)
        for state in states:
            np.testing.assert_allclose(state.joint_q.numpy(), [0, 0, 0.25, 0.25, 0, 0, 0.25, 0.25])
            np.testing.assert_allclose(state.joint_qd.numpy(), [0, 0, 0.1, 0.1, 0, 0, 0.1, 0.1])
    return True


def test_usbc_cable_reset_isolation() -> None:
    assert run_function_with_persistent_simulation_app(_test_usbc_cable_reset_isolation)


def _check_usbc_cable_reset(base_env, arena_environment) -> None:
    import numpy as np
    import torch

    from isaaclab_newton.physics import NewtonManager

    model = NewtonManager.get_model()
    states = (NewtonManager.get_state_0(), NewtonManager.get_state_1())
    coordinate_starts = model.joint_q_start.numpy()
    velocity_starts = model.joint_qd_start.numpy()
    cable_indices = [index for index, label in enumerate(model.joint_label) if "UsbcConnectorCable" in label]
    snapshots = []
    for state in states:
        coordinates = state.joint_q.numpy()
        velocities = state.joint_qd.numpy()
        for index in cable_indices:
            coordinates[coordinate_starts[index] : coordinate_starts[index + 1]] = 0.25
            velocities[velocity_starts[index] : velocity_starts[index + 1]] = 0.1
        state.joint_q.assign(coordinates)
        state.joint_qd.assign(velocities)
        snapshots.append((coordinates.copy(), velocities.copy()))

    _, event = arena_environment.scene.assets["plug_cable"].get_event_cfg()
    assert "reset_usbc_cables" not in base_env.event_manager.active_terms["reset"]
    event.func(base_env, [], **event.params)
    for state, (coordinates, velocities) in zip(states, snapshots, strict=True):
        np.testing.assert_array_equal(state.joint_q.numpy(), coordinates)
        np.testing.assert_array_equal(state.joint_qd.numpy(), velocities)

    event.func(base_env, torch.tensor([0], device=base_env.device), **event.params)
    for state, (coordinates, velocities) in zip(states, snapshots, strict=True):
        for index in cable_indices:
            if str(model.joint_label[index]).startswith("/World/envs/env_0/UsbcConnectorCablePlug/bend"):
                coordinates[coordinate_starts[index] : coordinate_starts[index + 1]] = 0.0
                velocities[velocity_starts[index] : velocity_starts[index + 1]] = 0.0
        np.testing.assert_array_equal(state.joint_q.numpy(), coordinates)
        np.testing.assert_array_equal(state.joint_qd.numpy(), velocities)

    base_env._reset_idx(torch.tensor([0], device=base_env.device))
    for state in states:
        coordinates = state.joint_q.numpy()
        velocities = state.joint_qd.numpy()
        for index in cable_indices:
            reset = str(model.joint_label[index]).startswith("/World/envs/env_0/")
            np.testing.assert_allclose(
                coordinates[coordinate_starts[index] : coordinate_starts[index + 1]], 0.0 if reset else 0.25
            )
            np.testing.assert_allclose(
                velocities[velocity_starts[index] : velocity_starts[index + 1]], 0.0 if reset else 0.1
            )
    base_env._reset_idx(torch.arange(base_env.num_envs, device=base_env.device))


def _test_usbc_insertion_environment(_simulation_app, variant: str, num_envs: int = 1) -> bool:
    import torch

    from isaaclab_arena.cli.isaaclab_arena_cli import arena_env_builder_cfg_from_argparse, get_isaaclab_arena_cli_parser
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena_environments.isaac_cap.usbc_insertion.environment import (
        UsbcInsertionEasyEnvironment,
        UsbcInsertionEasyEnvironmentCfg,
        UsbcInsertionMediumEnvironment,
        UsbcInsertionMediumEnvironmentCfg,
    )

    if variant == "easy":
        arena_environment = UsbcInsertionEasyEnvironment().build(UsbcInsertionEasyEnvironmentCfg())
    else:
        assert variant == "medium"
        arena_environment = UsbcInsertionMediumEnvironment().build(UsbcInsertionMediumEnvironmentCfg())

    args_cli = get_isaaclab_arena_cli_parser().parse_args(["--num_envs", str(num_envs)])
    env_builder = ArenaEnvBuilder(arena_environment, arena_env_builder_cfg_from_argparse(args_cli))
    env = env_builder.make_registered()
    try:
        env.reset()
        base_env = env.unwrapped
        plug_pose = base_env.arena_world.get_pose_w(arena_environment.task.plug.name)
        receiver_pose = base_env.arena_world.get_pose_w(arena_environment.task.receiver.name)
        assert plug_pose.shape == receiver_pose.shape == (num_envs, 7)
        assert torch.isfinite(plug_pose).all()
        assert torch.isfinite(receiver_pose).all()
        if variant == "easy":
            from isaaclab_newton.physics import NewtonManager

            bounds = ((0.42, 0.46), (-0.09, -0.05), (0.811, 0.811))
            tilt_cfg = arena_environment.task.get_termination_cfg().success.params["predicates"][2]
            assert tilt_cfg.func(base_env, **tilt_cfg.params).all()
            cable_joints = [label for label in NewtonManager.get_model().joint_label if "UsbcConnectorCable" in label]
            assert len(cable_joints) == 32 * num_envs
        else:
            bounds = ((-0.07, -0.03), (-0.05, -0.03), (0.7825, 0.7825))
        plug_positions = plug_pose[:, :3] - base_env.scene.env_origins
        for coordinates, (lower, upper) in zip(plug_positions.T, bounds, strict=True):
            assert torch.all(coordinates >= lower - 1.0e-6)
            assert torch.all(coordinates <= upper + 1.0e-6)

        if variant == "easy":
            _check_usbc_cable_reset(base_env, arena_environment)

        receiver = base_env.scene[arena_environment.task.receiver.name]
        moved_pose = receiver_pose.clone()
        moved_pose[:, 0] += 0.05
        receiver.write_root_pose_to_sim(moved_pose)
        base_env._reset_idx(torch.arange(num_envs, device=base_env.device))
        restored_pose = base_env.arena_world.get_pose_w(arena_environment.task.receiver.name)
        assert torch.allclose(restored_pose[:, :3], receiver_pose[:, :3], atol=1.0e-6)

        action = torch.zeros(env.action_space.shape, device=base_env.device)
        env.step(action)
        success = base_env.termination_manager.get_term("success")
        assert success.shape == (num_envs,)
    finally:
        env.close()
    return True


def test_usbc_insertion_medium_environment() -> None:
    assert run_function_with_persistent_simulation_app(
        _test_usbc_insertion_environment,
        variant="medium",
    )


def test_usbc_insertion_easy_environment() -> None:
    assert run_function_with_persistent_simulation_app(
        _test_usbc_insertion_environment,
        variant="easy",
    )
