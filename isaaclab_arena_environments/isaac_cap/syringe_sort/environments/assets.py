# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Syringe manipulands and fixtures for the shared CAP FR3 workcell."""

import isaaclab.sim as sim_utils

from isaaclab_arena.assets.object import Object
from isaaclab_arena.assets.object_type import ObjectType
from isaaclab_arena.utils.pose import Pose
from isaaclab_arena_environments.isaac_cap.assets import CAP_ASSET_ROOT

SYRINGE_ASSET_ROOT = f"{CAP_ASSET_ROOT}/syringe_disposal/assets"


class SyringeRedCap(Object):
    """Rigid red-cap syringe manipuland from the CAP benchmark."""

    name = "syringe"
    tags = ["object", "graspable"]
    usd_path = f"{SYRINGE_ASSET_ROOT}/vabar_tool_sort__syringe/vabar_tool_sort__syringe.usda"

    def __init__(self, instance_name: str = "syringe_0", initial_pose: dict | None = None, **kwargs):
        super().__init__(
            name=instance_name,
            tags=self.tags,
            usd_path=self.usd_path,
            object_type=ObjectType.RIGID,
            initial_pose=Pose.from_dict(initial_pose),
            **kwargs,
        )


class SyringeWhiteCap(Object):
    """Bare white-cap syringe, the second scored object in the both variant."""

    name = "syringe_blank"
    tags = ["object", "graspable"]
    usd_path = f"{SYRINGE_ASSET_ROOT}/vabar_tool_sort__syringe_blank/vabar_tool_sort__syringe_blank.usda"

    def __init__(self, instance_name: str = "syringe_blank", initial_pose: dict | None = None, **kwargs):
        super().__init__(
            name=instance_name,
            tags=self.tags,
            usd_path=self.usd_path,
            object_type=ObjectType.RIGID,
            initial_pose=Pose.from_dict(initial_pose),
            **kwargs,
        )


class InstrumentTray(Object):
    """Instrument tray retaining the authored cavity colliders, optionally fixed."""

    name = "instrument_tray"
    tags = ["object", "container"]
    usd_path = f"{SYRINGE_ASSET_ROOT}/vabar_tool_sort__instrument_tray/vabar_tool_sort__instrument_tray.usda"

    def __init__(
        self,
        instance_name: str = "instrument_tray",
        initial_pose: dict | None = None,
        fixed: bool = True,
        **kwargs,
    ):
        super().__init__(
            name=instance_name,
            tags=self.tags,
            usd_path=self.usd_path,
            object_type=ObjectType.RIGID,
            initial_pose=Pose.from_dict(initial_pose),
            collision_mode="mesh",
            spawn_cfg_addon={"rigid_props": sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=fixed)},
            **kwargs,
        )


class SharpsContainer(InstrumentTray):
    """Fixed sharps container with the authored aperture and interior."""

    name = "sharps_container"
    usd_path = f"{SYRINGE_ASSET_ROOT}/industrial__tool_sort_bin/bin2_syringe.usda"

    def __init__(self, instance_name: str = "sharps_container", **kwargs):
        super().__init__(instance_name=instance_name, **kwargs)
