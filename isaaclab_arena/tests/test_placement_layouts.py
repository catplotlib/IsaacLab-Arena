# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0


"""Companion layout schema and correlated reset selection."""

import pytest

from isaaclab_arena.relations.placement_layouts import PlacementLayouts


@pytest.mark.parametrize("duplicate", ["object", "pose_field"])
def test_layout_yaml_rejects_duplicate_keys(tmp_path, duplicate):
    import yaml

    pose = "- position_xyz: [1, 0, 0]\n  rotation_xyzw: [0, 0, 0, 1]\n"
    data = "cup:\n" + pose
    if duplicate == "object":
        data += "cup:\n" + pose.replace("[1, 0, 0]", "[2, 0, 0]")
    else:
        data += "  position_xyz: [2, 0, 0]\n"
    path = tmp_path / "poses.yaml"
    path.write_text(data)
    with pytest.raises(yaml.constructor.ConstructorError, match="Duplicate key"):
        PlacementLayouts.from_yaml(path)


@pytest.mark.parametrize(
    "value",
    [None, {"position_xyz": [0, 0, 0], "rotation_xyzw": [0, 0, 0, 1], "extra": 1}],
)
def test_layout_yaml_rejects_missing_or_extra_pose_fields(tmp_path, value):
    import yaml

    path = tmp_path / "poses.yaml"
    path.write_text(yaml.safe_dump({"cup": [value]}))
    with pytest.raises(AssertionError, match="object 'cup', layout 0"):
        PlacementLayouts.from_yaml(path)


def test_layout_yaml_rejects_empty_or_incomplete_layouts(tmp_path):
    import yaml

    pose = {"position_xyz": [0, 0, 0], "rotation_xyzw": [0, 0, 0, 1]}
    path = tmp_path / "poses.yaml"
    for layouts in ({}, {"cup": []}, {"cup": [pose], "bowl": [pose, pose]}):
        path.write_text(yaml.safe_dump(layouts))
        with pytest.raises(AssertionError, match="must contain objects|same nonzero number of poses"):
            PlacementLayouts.from_yaml(path)
