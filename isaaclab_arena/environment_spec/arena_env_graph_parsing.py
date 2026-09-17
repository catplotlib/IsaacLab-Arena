# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Parse serialized graph parameters into runtime constructor arguments."""

from typing import Any


def parse_asset_params(params: dict[str, Any]) -> dict[str, Any]:
    """Convert an asset's serialized initial pose without modifying its graph parameters."""
    from isaaclab_arena.utils.pose import Pose

    parsed = dict(params)
    if isinstance(parsed.get("initial_pose"), dict):
        parsed["initial_pose"] = Pose.from_dict(parsed["initial_pose"])
    return parsed
