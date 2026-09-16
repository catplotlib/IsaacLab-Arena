# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Save before/after solver plots for a box placed with and without On overlap.

Run with ``python -m isaaclab_arena_examples.relations.on_overlap_comparison
--output /tmp/on_overlap_comparison.png``. This example runs the relation solver
and geometric validator, without starting a physics simulation.
"""

from __future__ import annotations

import argparse
import matplotlib
import numpy as np
from pathlib import Path

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from isaaclab_arena.relations.object_placer_params import ObjectPlacerParams
from isaaclab_arena.relations.placement_validators import OnRelationValidator
from isaaclab_arena.relations.relation_solver import RelationSolver
from isaaclab_arena.relations.relation_solver_params import RelationSolverParams
from isaaclab_arena.relations.relations import IsAnchor, On
from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox
from isaaclab_arena.utils.pose import Pose
from isaaclab_arena_examples.relations.example_object import ExampleObject


def solve_example(overlap: bool) -> tuple[ExampleObject, ExampleObject, list[dict], list[bool]]:
    """Solve and validate the same initial box placement for one overlap setting.

    Args:
        overlap: Whether the box may extend beyond the support footprint.

    Returns:
        Support, box, before/after positions, and their On validation results.
    """
    support = ExampleObject(
        "support",
        AxisAlignedBoundingBox(min_point=(-0.5, -0.5, 0.0), max_point=(0.5, 0.5, 0.2)),
    )
    support.set_initial_pose(
        Pose(position_xyz=(0.0, 0.0, 0.0), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)), create_reset_event=False
    )
    support.add_relation(IsAnchor())
    box = ExampleObject(
        "box",
        AxisAlignedBoundingBox(min_point=(-0.2, -0.2, 0.0), max_point=(0.2, 0.2, 0.3)),
    )
    # Zero margin in both cases isolates the overlap flag; overlap=True ignores margins anyway.
    box.add_relation(On(support, overlap=overlap, edge_margin_m=0.0, clearance_m=0.0))
    initial = {support: (0.0, 0.0, 0.0), box: (0.55, 0.0, 0.2)}
    solver = RelationSolver(RelationSolverParams(verbose=False))
    final = solver.solve([support, box], [initial])[0]
    positions = [initial, final]
    bboxes = {obj: obj.get_bounding_box() for obj in (support, box)}
    valid = OnRelationValidator(ObjectPlacerParams()).validate_batch(positions, [{}, {}], [bboxes, bboxes], [])
    assert valid[1], f"Solver did not produce a valid On placement for overlap={overlap}"
    return support, box, positions, valid


def draw_view(ax, support: ExampleObject, box: ExampleObject, positions: list[dict], axes: tuple[int, int]) -> None:
    """Draw support, initial box outline, and solved box for one projection.

    Args:
        ax: Matplotlib axes to draw on.
        support: Fixed support asset.
        box: Movable box asset.
        positions: Initial and final placement dictionaries.
        axes: Coordinate indices for the horizontal and vertical plot axes.
    """
    horizontal, vertical = axes
    for obj, pose, color, label, dashed in (
        (support, positions[0][support], "#dce4ec", "Support", False),
        (box, positions[1][box], "#56a6d8", "After solving", False),
        (box, positions[0][box], "#b45309", "Before solving", True),
    ):
        bounds = obj.get_bounding_box().translated(pose)
        minimum = bounds.min_point[0].tolist()
        size = bounds.size[0].tolist()
        ax.add_patch(
            Rectangle(
                (minimum[horizontal], minimum[vertical]),
                size[horizontal],
                size[vertical],
                facecolor="none" if dashed else color,
                edgecolor=color if dashed else "#34465c",
                linestyle="--" if dashed else "-",
                linewidth=2,
                label=label,
                zorder=3 if dashed else 2,
            )
        )

    initial = positions[0][box]
    final = positions[1][box]
    # The box origin is centered in XY but sits on its bottom in Z.
    center_offset = box.get_bounding_box().center[0].tolist()
    start = (initial[horizontal] + center_offset[horizontal], initial[vertical] + center_offset[vertical])
    end = (final[horizontal] + center_offset[horizontal], final[vertical] + center_offset[vertical])
    if not np.allclose(start, end):
        ax.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle": "->", "color": "#15283e", "lw": 2})
    else:
        ax.text(*end, "Unchanged", ha="center", va="center", fontsize=10, zorder=4)

    ax.set_xlim(-0.65, 0.9)
    ax.set_ylim((-0.65, 0.65) if vertical == 1 else (-0.08, 0.65))
    ax.set_aspect("equal")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m) — top view" if vertical == 1 else "Z (m) — side view")
    ax.grid(alpha=0.2, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)


def main() -> None:
    """Write PNG and SVG comparisons to the requested output location."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("on_overlap_comparison.png"), help="Output PNG path.")
    args = parser.parse_args()
    assert args.output.suffix.lower() == ".png", "--output must be a PNG path"

    fig, plots = plt.subplots(2, 2, figsize=(12, 10), gridspec_kw={"height_ratios": [1.3, 0.8]})
    fig.suptitle("Box on a support: containment versus overlap", fontsize=19, fontweight="bold", y=0.97)
    for column, overlap in enumerate((False, True)):
        support, box, positions, valid = solve_example(overlap)
        before, after = ("PASS" if result else "FAIL" for result in valid)
        plots[0, column].set_title(
            f"On(support, overlap={overlap})\nOn validation: before {before} → after {after}",
            fontsize=13,
            pad=15,
        )
        for row, projection in enumerate(((0, 1), (0, 2))):
            draw_view(plots[row, column], support, box, positions, projection)
        initial_x, _, initial_z = positions[0][box]
        final_x, _, final_z = positions[1][box]
        print(
            f"overlap={overlap}: X {initial_x:.3f} → {final_x:.3f} m; "
            f"bottom Z {initial_z:.3f} → {final_z:.3f} m; On validation {before} → {after}"
        )

    handles, labels = plots[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.085), ncol=3, frameon=False)
    fig.text(
        0.5,
        0.043,
        "Same starting pose, zero edge margin and zero vertical clearance in both cases.\n"
        "Geometric placement only: overlap does not guarantee stable support. No physics settling is run.",
        ha="center",
        fontsize=11,
        color="#475569",
    )
    fig.subplots_adjust(left=0.09, right=0.97, top=0.84, bottom=0.17, wspace=0.3, hspace=0.3)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, facecolor="white")
    fig.savefig(args.output.with_suffix(".svg"), facecolor="white")
    plt.close(fig)
    print(f"Saved {args.output} and {args.output.with_suffix('.svg')}")


if __name__ == "__main__":
    main()
