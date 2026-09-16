# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Syringe sorting with the shared CAP FR3 embodiment and task-owned physics."""

from dataclasses import dataclass
from pathlib import Path

from isaaclab.utils.configclass import configclass
from isaaclab_newton.physics import MJWarpSolverCfg

from isaaclab_arena.environments.arena_environment_factory import ArenaEnvironmentCfg, ArenaEnvironmentFactory


@configclass
class SyringeSolverCfg(MJWarpSolverCfg):
    enable_multiccd: bool = True
    """Enable multiple contacts for convex collision pairs, as in CAP."""


def configure_syringe_physics(env_cfg):
    """Apply CAP's 50 Hz, ten-substep Newton tool-sort profile."""
    from isaaclab_newton.physics import NewtonCfg, NewtonShapeCfg

    env_cfg.sim.dt = 0.02
    env_cfg.sim.render_interval = 1
    env_cfg.sim.gravity = (0.0, 0.0, -9.81)
    env_cfg.sim.use_newton_actuators = True
    env_cfg.decimation = 1
    env_cfg.scene.replicate_physics = False
    env_cfg.sim.physics = NewtonCfg(
        solver_cfg=SyringeSolverCfg(
            solver="newton",
            disable_sensors=True,
            integrator="implicitfast",
            nconmax=5000,
            njmax=5000,
            iterations=100,
            ls_iterations=50,
            impratio=20.0,
            cone="elliptic",
            use_mujoco_contacts=True,
        ),
        default_shape_cfg=NewtonShapeCfg(ke=60000.0, kd=500.0, gap=0.002),
        num_substeps=10,
        use_cuda_graph=True,
        debug_mode=False,
    )
    return env_cfg


@dataclass
class SyringeSortEnvironmentCfg(ArenaEnvironmentCfg):
    """Configure the syringe environment and an optional episode timeout."""

    enable_cameras: bool = False
    episode_length_s: float | None = None
    variant: str = "single"


class SyringeSortEnvironment(ArenaEnvironmentFactory[SyringeSortEnvironmentCfg]):
    """Pick a syringe from its tray and release it into the sharps container."""

    name = "syringe_single_newton"
    _legacy_argparse_cfg_type = SyringeSortEnvironmentCfg

    def build(self, cfg: SyringeSortEnvironmentCfg):
        from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg

        from isaaclab_arena.environment_spec.arena_env_graph_spec import ArenaEnvGraphSpec

        from .cameras import configure_syringe_cameras

        assert cfg.variant in ("single", "both", "cluttered")
        spec = ArenaEnvGraphSpec.from_yaml(str(Path(__file__).with_name(f"syringe_{cfg.variant}.yaml")))
        arena_env = spec.to_arena_env(enable_cameras=cfg.enable_cameras)

        # TODO(alexmillane) [berkley-cap-align-embodiments]: Remove these per-task custom
        # embodiment configurations once the upstream repo has done it.
        configure_syringe_cameras(arena_env.embodiment.camera_config)
        gripper = arena_env.embodiment.scene_config.robot.actuators["robotiq_driver"]
        gripper.stiffness, gripper.damping = 20.0, 1.0
        arena_env.embodiment.action_config.gripper_action = JointPositionActionCfg(
            asset_name="robot",
            joint_names=["left_driver_joint"],
            preserve_order=True,
            use_default_offset=False,
            scale=0.8,
            offset=0.0,
        )
        if cfg.variant in ("both", "cluttered"):
            arena_env.placer_params.random_yaw_init = False
            arena_env.placer_params.allow_best_loss_fallbacks = False
        if cfg.variant == "cluttered":
            arena_env.placer_params.solver_params.clearance_m = 0.015
            arena_env.placer_params.max_placement_attempts = 30
        if cfg.episode_length_s is not None:
            assert cfg.episode_length_s > 0
            arena_env.task.episode_length_s = cfg.episode_length_s
        arena_env.env_cfg_callback = configure_syringe_physics
        return arena_env


@dataclass
class SyringeBothEnvironmentCfg(SyringeSortEnvironmentCfg):
    """Select the randomized two-syringe benchmark."""

    variant: str = "both"


class SyringeBothEnvironment(ArenaEnvironmentFactory[SyringeBothEnvironmentCfg]):
    """Dispose of both the red-cap and bare syringes."""

    name = "syringe_both_newton"
    _legacy_argparse_cfg_type = SyringeBothEnvironmentCfg

    def build(self, cfg: SyringeBothEnvironmentCfg):
        assert cfg.variant == "both"
        return SyringeSortEnvironment().build(cfg)


@dataclass
class SyringeClutteredEnvironmentCfg(SyringeSortEnvironmentCfg):
    """Select the randomized four-syringe benchmark."""

    variant: str = "cluttered"


class SyringeClutteredEnvironment(ArenaEnvironmentFactory[SyringeClutteredEnvironmentCfg]):
    """Dispose of all four syringes from the cluttered tray."""

    name = "syringe_cluttered_newton"
    _legacy_argparse_cfg_type = SyringeClutteredEnvironmentCfg

    def build(self, cfg: SyringeClutteredEnvironmentCfg):
        assert cfg.variant == "cluttered"
        return SyringeSortEnvironment().build(cfg)
