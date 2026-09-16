# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Registered Isaac Cap USB-C insertion environments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from isaaclab_arena.environments.arena_environment_factory import ArenaEnvironmentCfg, ArenaEnvironmentFactory

if TYPE_CHECKING:
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment


@dataclass
class UsbcInsertionEasyEnvironmentCfg(ArenaEnvironmentCfg):
    """Configure the bimanual held-bulkhead USB-C task."""

    use_tiled_cameras: bool = False
    use_instanceable_meshes: bool = False


@dataclass
class UsbcInsertionMediumEnvironmentCfg(ArenaEnvironmentCfg):
    """Configure the single-arm bolted-port precision USB-C task."""

    use_tiled_cameras: bool = False


class UsbcInsertionEasyEnvironment(ArenaEnvironmentFactory[UsbcInsertionEasyEnvironmentCfg]):
    """Build the bimanual YAM variant from its environment graph."""

    name = "vabar_contact_rich_insertion__usbc_insertion_easy"
    _legacy_argparse_cfg_type = UsbcInsertionEasyEnvironmentCfg
    scene_spec = Path(__file__).with_name("usbc_easy.yaml")

    def build(self, cfg: UsbcInsertionEasyEnvironmentCfg) -> IsaacLabArenaEnvironment:
        from isaaclab_arena.environment_spec.arena_env_graph_spec import ArenaEnvGraphSpec
        from isaaclab_arena_environments.isaac_cap import register_components

        from .physics import configure_easy_usbc_physics

        register_components()
        spec = ArenaEnvGraphSpec.from_yaml(self.scene_spec)
        spec.embodiment.params.update(
            use_tiled_cameras=cfg.use_tiled_cameras,
            use_instanceable_meshes=cfg.use_instanceable_meshes,
        )
        arena_environment = spec.to_arena_env(enable_cameras=cfg.enable_cameras)
        arena_environment.env_cfg_callback = configure_easy_usbc_physics
        return arena_environment


class UsbcInsertionMediumEnvironment(ArenaEnvironmentFactory[UsbcInsertionMediumEnvironmentCfg]):
    """Build the single-arm precision variant from its environment graph."""

    name = "vabar_contact_rich_insertion__usbc_insertion_medium"
    _legacy_argparse_cfg_type = UsbcInsertionMediumEnvironmentCfg
    scene_spec = Path(__file__).with_name("usbc_medium.yaml")

    def build(self, cfg: UsbcInsertionMediumEnvironmentCfg) -> IsaacLabArenaEnvironment:
        from isaaclab_arena.environment_spec.arena_env_graph_spec import ArenaEnvGraphSpec
        from isaaclab_arena_environments.isaac_cap import register_components

        from .physics import configure_medium_usbc_physics

        register_components()
        spec = ArenaEnvGraphSpec.from_yaml(self.scene_spec)
        spec.embodiment.params["use_tiled_cameras"] = cfg.use_tiled_cameras
        arena_environment = spec.to_arena_env(enable_cameras=cfg.enable_cameras)
        arena_environment.env_cfg_callback = configure_medium_usbc_physics
        return arena_environment
