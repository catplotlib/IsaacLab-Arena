# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Syringe manipulands and fixtures for the shared CAP FR3 workcell."""

from isaaclab_arena.assets.object_library import LibraryObject
from isaaclab_arena.utils.pose import Pose
from isaaclab_arena_environments.isaac_cap.assets import CAP_ASSET_ROOT

SYRINGE_ASSET_ROOT = f"{CAP_ASSET_ROOT}/syringe_disposal/assets"


class SyringeRedCap(LibraryObject):
    """Red-cap syringe."""

    name = "syringe"
    tags = ["object", "graspable"]
    usd_path = f"{SYRINGE_ASSET_ROOT}/vabar_tool_sort__syringe/vabar_tool_sort__syringe.usda"


class SyringeWhiteCap(LibraryObject):
    """White-cap syringe."""

    name = "syringe_blank"
    tags = ["object", "graspable"]
    usd_path = f"{SYRINGE_ASSET_ROOT}/vabar_tool_sort__syringe_blank/vabar_tool_sort__syringe_blank.usda"


class InstrumentTray(LibraryObject):
    """Instrument tray with the authored cavity colliders."""

    name = "instrument_tray"
    tags = ["object", "container"]
    usd_path = f"{SYRINGE_ASSET_ROOT}/vabar_tool_sort__instrument_tray/vabar_tool_sort__instrument_tray.usda"

    def __init__(self, initial_pose: Pose | None = None, **kwargs):
        super().__init__(initial_pose=initial_pose, collision_mode="mesh", **kwargs)


class SharpsContainer(InstrumentTray):
    """Sharps container with the authored aperture and interior."""

    name = "sharps_container"
    usd_path = f"{SYRINGE_ASSET_ROOT}/industrial__tool_sort_bin/bin2_syringe.usda"
