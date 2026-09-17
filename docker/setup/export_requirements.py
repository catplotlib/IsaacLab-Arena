# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Export Docker requirements from the project's existing package metadata."""

import argparse
import re
import tomllib
from pathlib import Path


def requirements(metadata: dict, group: str) -> list[str]:
    """Return runtime or contributor requirements, preserving declared constraints.

    Args:
        metadata: Parsed project metadata.
        group: Docker dependency group, runtime or dev.

    Returns:
        Requirements for the selected group.
    """
    project = metadata["project"]
    # The existing dev extra mixes user tools with the debugger. Keep its
    # declarations authoritative; only classify the contributor-only package.
    contributor_packages = {"debugpy"}
    selected = list(project["dependencies"]) if group == "runtime" else []
    for requirement in project["optional-dependencies"]["dev"]:
        name = re.match(r"[A-Za-z0-9._-]+", requirement).group(0).lower().replace("_", "-")
        if (name in contributor_packages) == (group == "dev"):
            selected.append(requirement)
    return selected


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metadata", type=Path)
    parser.add_argument("group", choices=("runtime", "dev"))
    args = parser.parse_args()
    print("\n".join(requirements(tomllib.loads(args.metadata.read_text()), args.group)))
