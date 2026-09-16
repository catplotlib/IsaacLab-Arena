# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Industrial FR3 syringe tool-sorting environment (syringe_easy)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from isaaclab_arena.assets.register import register_environment
from isaaclab_arena.environments.arena_environment_factory import ArenaEnvironmentCfg, ArenaEnvironmentFactory
from isaaclab_arena_environments.industrial_tool_sort_environment import configure_industrial_tool_sort_physics

if TYPE_CHECKING:
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment

# Placement surface (table) frame origin in world coordinates matches the workcell table pose.
_SHARPS_CONTAINER_POSITION = (0.1, 0.28, 0.8)
_INSTRUMENT_TRAY_POSITION = (0.1, -0.25, 0.8)
# Syringe rests on the instrument tray; it settles onto the tray surface during the first steps.
_SYRINGE_POSITION = (0.1, -0.25, 0.84)


@dataclass
class IndustrialSyringeSortEnvironmentCfg(ArenaEnvironmentCfg):
    """Configure the industrial FR3 syringe tool-sorting environment."""

    embodiment: str = "industrial_fr3_robotiq_2f85"
    teleop_device: str | None = None


@register_environment
class IndustrialSyringeSortEnvironment(ArenaEnvironmentFactory[IndustrialSyringeSortEnvironmentCfg]):
    """Pick the syringe off the instrument tray and drop it into the sharps container."""

    name = "vabar_tool_sort__syringe_easy_newton"
    _legacy_argparse_cfg_type = IndustrialSyringeSortEnvironmentCfg

    def build(self, cfg: IndustrialSyringeSortEnvironmentCfg) -> IsaacLabArenaEnvironment:
        """Build the workcell, tray, syringe, sharps container, task, and Newton profile."""
        from isaaclab_arena.assets.hdr_image_library import EmptyWarehouseHDRRobolab
        from isaaclab_arena.assets.object_base import ObjectType
        from isaaclab_arena.assets.object_reference import ObjectReference
        from isaaclab_arena.embodiments.industrial_fr3.config import GRIPPER_JOINT_NAME
        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.scene.scene import Scene
        from isaaclab_arena.utils.pose import Pose
        from isaaclab_arena_cap.tasks.objects_centered_in_regions_task import ObjectsCenteredInRegionsTask

        embodiment = self.asset_registry.get_asset_by_name(cfg.embodiment)(
            enable_cameras=cfg.enable_cameras,
            initial_pose=Pose(position_xyz=(-0.5, -0.1, 0.912)),
        )

        background = self.asset_registry.get_asset_by_name("industrial__fr3_workcell_table")()
        background.set_initial_pose(Pose(position_xyz=(-0.5, -0.1, 0.912)))
        table = ObjectReference(
            name="table",
            prim_path="{ENV_REGEX_NS}/industrial__fr3_workcell_table/placement_surface",
            parent_asset=background,
            object_type=ObjectType.BASE,
        )

        sharps_container = self.asset_registry.get_asset_by_name("industrial__tool_sort_bin")(
            instance_name="sharps_container",
            side="destination",
            appearance="syringe",
            initial_pose=Pose(position_xyz=_SHARPS_CONTAINER_POSITION),
        )
        instrument_tray = self.asset_registry.get_asset_by_name("vabar_tool_sort__instrument_tray")(
            instance_name="instrument_tray",
            initial_pose=Pose(position_xyz=_INSTRUMENT_TRAY_POSITION),
        )
        syringe = self.asset_registry.get_asset_by_name("vabar_tool_sort__syringe")(
            instance_name="syringe_0",
            initial_pose=Pose(position_xyz=_SYRINGE_POSITION),
        )

        shadow_receiver = self.asset_registry.get_asset_by_name("industrial__hdr_shadow_receiver")()
        light = self.asset_registry.get_asset_by_name("light")(hdr=EmptyWarehouseHDRRobolab())
        scene = Scene(
            assets=[
                background,
                shadow_receiver,
                sharps_container,
                instrument_tray,
                syringe,
                light,
                table,
            ]
        )

        task = ObjectsCenteredInRegionsTask(
            object_list=[syringe],
            region_list=[sharps_container],
            bounds_xyzxyz=[(0.053, -0.2055, -0.016, 0.142, -0.0395, 0.134)],
            gripper_joint_name=GRIPPER_JOINT_NAME,
            consecutive_success_steps=50,
            episode_length_s=228.0,
            task_description=(
                "pick up the syringe from the instrument tray and drop it through the hole "
                "in the top of the yellow sharps disposal box"
            ),
        )
        teleop_device = (
            self.device_registry.get_device_by_name(cfg.teleop_device)() if cfg.teleop_device is not None else None
        )
        return IsaacLabArenaEnvironment(
            name=self.name,
            embodiment=embodiment,
            scene=scene,
            task=task,
            teleop_device=teleop_device,
            env_cfg_callback=configure_industrial_tool_sort_physics,
        )
