# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Object registrations for the CAP industrial syringe tool-sort environment.

The USD assets live in the shared vendored tree under
``isaaclab_arena/assets/industrial_tool_sort`` (see ``scripts/vendor_tool_sort_assets``);
the compartmented ``industrial__tool_sort_bin`` and the FR3 workcell/robot assets are
registered by the core libraries and reused here.
"""

from __future__ import annotations

import isaaclab.sim as sim_utils

from isaaclab_arena.assets.object import Object
from isaaclab_arena.assets.object_base import ObjectType
from isaaclab_arena.assets.object_library import _INDUSTRIAL_TOOL_SORT_ASSET_ROOT, IndustrialToolSortObject
from isaaclab_arena.assets.register import register_asset
from isaaclab_arena.utils.pose import Pose


@register_asset
class IndustrialToolSortSyringe(IndustrialToolSortObject):
    """Syringe manipuland from the industrial tool-sorting benchmark."""

    name = "vabar_tool_sort__syringe"
    usd_path = str(_INDUSTRIAL_TOOL_SORT_ASSET_ROOT / name / f"{name}.usda")


@register_asset
class IndustrialToolSortInstrumentTray(Object):
    """Kinematic instrument tray that presents tools for industrial sorting."""

    name = "vabar_tool_sort__instrument_tray"
    tags = ["object", "container", "industrial", "tool_sort"]

    def __init__(
        self,
        instance_name: str = "instrument_tray",
        initial_pose: Pose | None = None,
        **kwargs,
    ):
        super().__init__(
            name=instance_name,
            tags=self.tags,
            usd_path=str(_INDUSTRIAL_TOOL_SORT_ASSET_ROOT / self.name / f"{self.name}.usda"),
            object_type=ObjectType.RIGID,
            initial_pose=initial_pose,
            collision_mode="mesh",
            spawn_cfg_addon={
                "rigid_props": sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            },
            **kwargs,
        )
