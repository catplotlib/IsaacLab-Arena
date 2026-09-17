# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Summarize `docker buildx history logs --progress=rawjson` on stdin."""

import json
import re
import sys
from datetime import datetime


def summarize(stream):
    """Return a Markdown report of cache imports and Dockerfile steps."""
    vertices = {}
    for line in stream:
        for vertex in json.loads(line).get("vertexes", []):
            vertices.setdefault(vertex["digest"], {}).update(vertex)

    def status(vertex):
        if vertex.get("error"):
            return "FAILED"
        if vertex.get("cached"):
            return "CACHED"
        return "RAN" if vertex.get("completed") else "INCOMPLETE"

    def seconds(vertex):
        if not vertex.get("started") or not vertex.get("completed"):
            return "—"
        elapsed = datetime.fromisoformat(vertex["completed"].replace("Z", "+00:00")) - datetime.fromisoformat(
            vertex["started"].replace("Z", "+00:00")
        )
        return f"{elapsed.total_seconds():.1f}"

    imports = [v for v in vertices.values() if v["name"].startswith("importing cache manifest from ")]
    steps = [v for v in vertices.values() if re.match(r"\[[^]]+ \d+/\d+\] (RUN|COPY|ADD|WORKDIR)\b", v["name"])]
    lines = ["### Docker cache usage", ""]
    for vertex in imports:
        result = vertex.get("error") or ("imported" if vertex.get("completed") else "incomplete")
        lines.append(f"- `{vertex['name']}`: {result}")
    if not imports:
        lines.append("- No registry cache import was recorded.")
    hits = sum(status(v) == "CACHED" for v in steps)
    lines += [
        "",
        f"**{hits}/{len(steps)} Dockerfile steps CACHED.** Base-image and frontend downloads are excluded.",
        "Cache import success alone does not mean any build step was reused.",
        "",
        "| Step | Result | Seconds |",
        "| --- | --- | ---: |",
    ]
    for vertex in steps:
        name = vertex["name"].replace("|", "\\|").replace("`", "'")
        lines.append(f"| `{name}` | {status(vertex)} | {seconds(vertex)} |")
    lines += ["", "Step durations can overlap and include fetching cached layers; they are not total build time."]
    if not steps:
        lines += ["", "No Dockerfile steps found; inspect the raw build record before drawing conclusions."]
    return "\n".join(lines)


if __name__ == "__main__":
    print(summarize(sys.stdin))
