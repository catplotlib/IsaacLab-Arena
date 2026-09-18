# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Locally staged assets for the CAP easy tool-sorting environments."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import isaaclab.sim as sim_utils

from isaaclab_arena.assets.nucleus import ARENA_NUCLEUS_DIR
from isaaclab_arena.assets.object import Object
from isaaclab_arena.assets.object_type import ObjectType
from isaaclab_arena.relations.collision_mode import CollisionMode
from isaaclab_arena.utils.pose import Pose

LOCAL_TOOL_SORT_ASSET_ROOT = Path(__file__).resolve().parents[3] / "__assets" / "cap_envs" / "tool_sorting" / "assets"
"""Ignored local asset tree used until the assets are uploaded."""

PUBLISHED_TOOL_SORT_ASSET_ROOT = (
    f"{ARENA_NUCLEUS_DIR}/Arena/assets/object_library/temp_newton_envs/cap_envs/tool_sorting/assets"
)
"""Final public asset location after the local tree is uploaded and mirrored."""

TOOL_SORT_ASSET_ROOT = os.environ.get("ARENA_TOOL_SORT_ASSET_ROOT", str(LOCAL_TOOL_SORT_ASSET_ROOT))
"""Active asset root; override to test the Nucleus upload before changing the default."""

TOOL_NAMES = (
    "vabar_tool_sort__adjustable_wrench",
    "vabar_tool_sort__battery",
    "vabar_tool_sort__breadboard",
    "vabar_tool_sort__combination_pliers",
    "vabar_tool_sort__cutting_pliers",
    "vabar_tool_sort__flashlight",
    "vabar_tool_sort__insulating_tape",
    "vabar_tool_sort__multimeter",
    "vabar_tool_sort__safety_glasses",
    "vabar_tool_sort__slotted_screwdriver",
    "vabar_tool_sort__tape_measure",
    "vabar_tool_sort__wire_spool",
)
"""Registry names required by the three easy levels."""

_BIN_APPEARANCES = frozenset({"default", "bench", "electrical", "wiring"})


def _normalize_pose(initial_pose: Pose | Mapping[str, Sequence[float]] | None) -> Pose | None:
    """Normalize graph pose mappings to Arena poses."""
    if initial_pose is None or isinstance(initial_pose, Pose):
        return initial_pose
    return Pose(
        position_xyz=tuple(float(value) for value in initial_pose["position_xyz"]),
        rotation_xyzw=tuple(float(value) for value in initial_pose["rotation_xyzw"]),
    )


def _make_tool_factory(registry_name: str):
    """Create one rigid tool factory for ``registry_name``."""

    def factory(
        instance_name: str | None = None,
        initial_pose: Pose | Mapping[str, Sequence[float]] | None = None,
        **_ignored: Any,
    ) -> Object:
        name = instance_name or registry_name
        tool = Object(
            name=name,
            prim_path=f"{{ENV_REGEX_NS}}/{name}",
            object_type=ObjectType.RIGID,
            usd_path=f"{TOOL_SORT_ASSET_ROOT}/{registry_name}/{registry_name}.usda",
            initial_pose=_normalize_pose(initial_pose),
            tags=["object", "graspable", "industrial", "tool_sort"],
        )
        tool.disable_reset_pose()
        return tool

    factory.__name__ = f"make_{registry_name}"
    factory.name = registry_name
    factory.tags = ("object", "graspable", "industrial", "tool_sort")
    factory.object_type = ObjectType.RIGID
    return factory


def make_industrial_tool_sort_bin(
    instance_name: str = "tool_sort_bin",
    side: str = "destination",
    appearance: str = "default",
    initial_pose: Pose | Mapping[str, Sequence[float]] | None = None,
    **_ignored: Any,
) -> Object:
    """Create a kinematic source or compartmented destination bin."""
    assert side in {"source", "destination"}, f"Invalid tool-sort bin side: {side!r}"
    assert appearance in _BIN_APPEARANCES, f"Invalid tool-sort bin appearance: {appearance!r}"
    leaf = "bin1.usda" if side == "source" else f"bin2_{appearance}.usda"
    bin_object = Object(
        name=instance_name,
        prim_path=f"{{ENV_REGEX_NS}}/{instance_name}",
        object_type=ObjectType.RIGID,
        usd_path=f"{TOOL_SORT_ASSET_ROOT}/industrial__tool_sort_bin/{leaf}",
        initial_pose=_normalize_pose(initial_pose),
        spawn_cfg_addon={
            "rigid_props": sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        },
        tags=["object", "container", "industrial", "tool_sort"],
    )
    bin_object.collision_mode = CollisionMode.MESH
    return bin_object


make_industrial_tool_sort_bin.name = "industrial__tool_sort_bin"
make_industrial_tool_sort_bin.tags = ("object", "container", "industrial", "tool_sort")
make_industrial_tool_sort_bin.object_type = ObjectType.RIGID

TOOL_SORT_ASSET_ENTRY_POINTS = {
    **{name: _make_tool_factory(name) for name in TOOL_NAMES},
    make_industrial_tool_sort_bin.name: make_industrial_tool_sort_bin,
}
"""Asset factories registered by the Isaac CAP entry point."""
