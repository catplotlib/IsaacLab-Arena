# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0


"""Named object poses for complete, reusable environment layouts."""

from __future__ import annotations

import yaml
from dataclasses import dataclass
from pathlib import Path

from isaaclab_arena.utils.pose import Pose


@dataclass
class PlacementLayouts:
    """L complete layouts for N named objects, expressed in environment frame E.

    The same list index selects one complete layout across every object.
    """

    poses: dict[str, list[Pose]]
    """N object names mapped to L poses each; positions have shape (3,), quaternions (4,)."""

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Require complete layouts containing finite poses and unit quaternions."""
        assert self.poses, "A placement cache must contain objects"
        assert all(isinstance(name, str) and name for name in self.poses), "Object names must be nonempty strings"
        counts = {len(poses) for poses in self.poses.values()}
        assert len(counts) == 1 and next(iter(counts)) > 0, "All objects must have the same nonzero number of poses"
        for poses in self.poses.values():
            for pose in poses:
                # Pose construction checks dimensions; cache poses also require finite values and unit quaternions.
                Pose.from_dict(pose.to_dict())

    @property
    def num_layouts(self) -> int:
        """Number of complete layouts."""
        return len(next(iter(self.poses.values())))

    def write_yaml(self, path: str | Path) -> None:
        """Write the layouts as ordinary YAML, refusing to overwrite an existing file."""
        self.validate()
        data = {name: [pose.to_dict() for pose in poses] for name, poses in self.poses.items()}
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as stream:
            yaml.safe_dump(data, stream, sort_keys=False)
