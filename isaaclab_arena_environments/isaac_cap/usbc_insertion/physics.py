# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Newton configurations for the two USB-C insertion variants."""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab_arena.environments.isaaclab_arena_manager_based_env_cfg import IsaacLabArenaManagerBasedRLEnvCfg

_EASY_GRIP_FORCE = 40.0
_EASY_GRIP_SPEED = 0.04
_EASY_GRIP_STIFFNESS = 4_000.0


def _configure_easy_grippers(env_cfg: IsaacLabArenaManagerBasedRLEnvCfg) -> None:
    """Apply the source task's force-limited, damped jaw tuning."""
    for robot_name in ("left_robot", "right_robot"):
        robot_cfg = getattr(env_cfg.scene, robot_name)
        gripper_cfg = robot_cfg.actuators["gripper"]
        gripper_cfg.effort_limit_sim = _EASY_GRIP_FORCE
        gripper_cfg.stiffness = max(float(gripper_cfg.stiffness), _EASY_GRIP_STIFFNESS)
        gripper_cfg.velocity_limit_sim = _EASY_GRIP_SPEED
        gripper_cfg.damping = _EASY_GRIP_FORCE / _EASY_GRIP_SPEED


def _assert_medium_connector_meshes(_event_payload=None) -> None:
    """Require exact connector meshes so convexification cannot fill the port bore."""
    from isaaclab_newton.physics import NewtonManager
    from newton import GeoType

    model = NewtonManager.get_model()
    assert model is not None, "USB-C mesh validation ran before Newton finalized its model."
    connector_shapes = [
        index
        for index, label in enumerate(model.shape_label)
        if any(token in str(label).casefold() for token in ("usbcplug", "usbcport"))
    ]
    assert len(connector_shapes) >= 2, "USB-C mesh validation could not find both connector collision shapes."
    shape_types = model.shape_type.numpy()
    offenders = [
        str(model.shape_label[index]) for index in connector_shapes if int(shape_types[index]) != int(GeoType.MESH)
    ]
    assert not offenders, f"USB-C connector collision shapes must remain exact meshes: {offenders}."


def _usbc_newton_cfg(*, precision_fit: bool):
    """Build an exact-mesh Newton configuration for connector mating."""
    from isaaclab_newton.physics import NewtonCollisionPipelineCfg

    from isaaclab_arena.environments.isaaclab_arena_manager_based_env_cfg import ArenaPhysicsCfg

    physics = deepcopy(ArenaPhysicsCfg().newton)
    physics.solver_cfg.solver = "newton"
    physics.solver_cfg.integrator = "implicitfast" if precision_fit else "euler"
    physics.solver_cfg.use_mujoco_contacts = False
    physics.solver_cfg.njmax = 4096
    physics.solver_cfg.nconmax = 2048
    physics.solver_cfg.iterations = 100
    physics.solver_cfg.ls_iterations = 50
    physics.solver_cfg.cone = "elliptic"
    physics.solver_cfg.impratio = 1.0
    physics.solver_cfg.update_data_interval = 1
    physics.num_substeps = 1 if precision_fit else 4
    physics.default_shape_cfg.margin = 0.0
    physics.default_shape_cfg.gap = 1.0e-4 if precision_fit else 2.0e-4
    physics.default_shape_cfg.ke = 62_500.0
    physics.default_shape_cfg.kd = 500.0
    physics.collision_cfg = NewtonCollisionPipelineCfg(
        reduce_contacts=True,
        rigid_contact_max=8192,
        max_triangle_pairs=1_000_000,
    )
    return physics


def configure_easy_usbc_physics(
    env_cfg: IsaacLabArenaManagerBasedRLEnvCfg,
) -> IsaacLabArenaManagerBasedRLEnvCfg:
    """Configure the held-bulkhead task at 60 Hz control and 240 Hz physics."""
    from .cables import NewtonEasyUsbcManager

    env_cfg.sim.dt = 1.0 / 240.0
    env_cfg.decimation = 4
    env_cfg.sim.render_interval = env_cfg.decimation
    env_cfg.sim.use_newton_actuators = True
    env_cfg.sim.physics = _usbc_newton_cfg(precision_fit=False)
    env_cfg.sim.physics.class_type = NewtonEasyUsbcManager
    env_cfg.scene.replicate_physics = False
    _configure_easy_grippers(env_cfg)
    return env_cfg


def configure_medium_usbc_physics(
    env_cfg: IsaacLabArenaManagerBasedRLEnvCfg,
) -> IsaacLabArenaManagerBasedRLEnvCfg:
    """Configure the precision-fit task at 60 Hz control and 960 Hz physics."""
    from isaaclab.physics import PhysicsEvent
    from isaaclab_newton.physics import NewtonManager

    env_cfg.sim.dt = 1.0 / 960.0
    env_cfg.decimation = 16
    env_cfg.sim.render_interval = env_cfg.decimation
    env_cfg.sim.use_newton_actuators = True
    env_cfg.sim.physics = _usbc_newton_cfg(precision_fit=True)
    env_cfg.scene.replicate_physics = False
    NewtonManager.register_callback(
        _assert_medium_connector_meshes,
        PhysicsEvent.PHYSICS_READY,
        name="arena_cap_usbc_connector_mesh_validation",
        wrap_weak_ref=False,
    )
    return env_cfg
