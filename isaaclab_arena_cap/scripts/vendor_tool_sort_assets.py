#!/usr/bin/env python3
# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
"""One-off host-side vendoring of industrial tool-sort USD assets.

Copies the USD dependency closure (ASCII-reachable ``@ref@`` files only, matching
the repo's closure test) from the Isaac-cap source tree into the gitignored
``isaaclab_arena/assets/industrial_tool_sort/`` directory. Point ``ARENA_CAP_SOURCE_ROOT``
at the Isaac-cap ``industrial_benchmark`` assets directory (defaults to a sibling checkout).
"""

from __future__ import annotations

import os
import re
import shutil
import sys
from pathlib import Path

_DEFAULT_SOURCE = Path("/home/alex/trunk/Isaac-cap/industrial_benchmark/src/industrial_benchmark/assets")
SOURCE_ROOT = Path(os.environ.get("ARENA_CAP_SOURCE_ROOT") or _DEFAULT_SOURCE)
# Repo root is three levels up: isaaclab_arena_cap/scripts/<this file>.
DEST_ROOT = Path(__file__).resolve().parents[2] / "isaaclab_arena" / "assets" / "industrial_tool_sort"

# Entry points are relative to SOURCE_ROOT and mirror the layout under DEST_ROOT.
ENTRYPOINTS = [
    "industrial__fr3_workcell_table/industrial__fr3_workcell_table.usda",
    "industrial__hdr_shadow_receiver/industrial__hdr_shadow_receiver.usda",
    "industrial__tool_sort_bin/bin1.usda",
    "industrial__tool_sort_bin/bin2_default.usda",
    "industrial__tool_sort_bin/bin2_syringe.usda",
    "vabar_tool_sort__hammer/vabar_tool_sort__hammer.usda",
    "vabar_tool_sort__drill/vabar_tool_sort__drill.usda",
    "vabar_tool_sort__round_nut/vabar_tool_sort__round_nut.usda",
    "vabar_tool_sort__clamp/vabar_tool_sort__clamp.usda",
    "vabar_tool_sort__syringe/vabar_tool_sort__syringe.usda",
    "vabar_tool_sort__instrument_tray/vabar_tool_sort__instrument_tray.usda",
    # The robot embodiment loads its USD from the same vendored tree.
    "industrial__fr3_robotiq_2f85/franka_fr3_robotiq_2f85.usda",
]

ALLOWED_SUFFIXES = {".usd", ".usda", ".usdc", ".png"}
# Material/texture file types referenced from *inside* binary crates (which the ASCII
# ``@ref@`` walk cannot see). Copied wholesale from each crate's sibling ``textures/``
# and ``materials/`` dirs; the pxr-based closure test is the authoritative completeness check.
_MATERIAL_SIBLING_DIRS = ("textures", "materials")
_MATERIAL_SUFFIXES = {".png", ".jpg", ".jpeg", ".exr", ".hdr", ".mdl", ".usd", ".usda", ".usdc"}


def compute_closure() -> set[Path]:
    reachable: set[Path] = set()
    pending = [(SOURCE_ROOT / entry).resolve() for entry in ENTRYPOINTS]
    while pending:
        usd_file = pending.pop().resolve()
        assert usd_file.is_relative_to(SOURCE_ROOT.resolve()), usd_file
        assert usd_file.exists(), f"Missing dependency: {usd_file}"
        if usd_file in reachable:
            continue
        reachable.add(usd_file)
        if usd_file.suffix not in {".usd", ".usda"}:
            continue
        try:
            contents = usd_file.read_text()
        except UnicodeDecodeError:
            continue
        for reference in re.findall(r"@([^@]+)@", contents):
            if "://" in reference:
                continue
            pending.append(usd_file.parent / reference)
    return reachable


def main() -> int:
    reachable = compute_closure()
    suffixes = {path.suffix for path in reachable}
    disallowed = suffixes - ALLOWED_SUFFIXES
    if disallowed:
        print(f"ERROR: reachable closure contains disallowed suffixes: {sorted(disallowed)}")
        for path in sorted(reachable):
            if path.suffix in disallowed:
                print(f"  {path.relative_to(SOURCE_ROOT)}")
        return 1

    # Copy the ASCII-reachable USD layer graph, then the material/texture siblings of every
    # crate directory (binary-crate references the ASCII walk cannot follow).
    to_copy: set[Path] = set(reachable)
    for source_path in reachable:
        for sibling in _MATERIAL_SIBLING_DIRS:
            sibling_dir = source_path.parent / sibling
            if not sibling_dir.is_dir():
                continue
            for asset_file in sibling_dir.rglob("*"):
                if asset_file.is_file() and asset_file.suffix.lower() in _MATERIAL_SUFFIXES:
                    to_copy.add(asset_file.resolve())

    copied = 0
    for source_path in sorted(to_copy):
        relative = source_path.relative_to(SOURCE_ROOT)
        dest_path = DEST_ROOT / relative
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, dest_path)
        copied += 1
    print(f"Vendored {copied} files into {DEST_ROOT}")
    print(f"Suffixes: {sorted({p.suffix for p in to_copy})}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
