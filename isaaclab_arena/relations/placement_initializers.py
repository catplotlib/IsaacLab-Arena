# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Strategies for seeding the relation solver's optimization variables."""

from __future__ import annotations

import torch
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from isaaclab_arena.relations.relations import On, PositionLimitsBox, get_relation
from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox
from isaaclab_arena.utils.pose import Pose

if TYPE_CHECKING:
    from isaaclab_arena.relations.placement_asset import PlaceableAsset

Position = tuple[float, float, float]
EnvBoundingBoxes = dict["PlaceableAsset", AxisAlignedBoundingBox]


class InitializerBase(ABC):
    """Produces the starting positions the relation solver optimizes from.

    The solver minimizes a hinge loss that is exactly zero once a relation is satisfied, so
    it stops at the first feasible point it reaches. Where a candidate starts therefore decides
    which part of the feasible set it lands in, and an initializer's job is to spread candidates
    over that set rather than to find a solution itself.
    """

    @abstractmethod
    def generate_initial_positions(
        self,
        objects: list[PlaceableAsset],
        anchor_objects: set[PlaceableAsset],
        env_bboxes: EnvBoundingBoxes,
        generator: torch.Generator | None = None,
    ) -> dict[PlaceableAsset, Position]:
        """Return one starting position per object for a single solver candidate.

        Args:
            objects: Every object taking part in the solve, anchors included.
            anchor_objects: The subset of objects that stay at their fixed initial pose.
            env_bboxes: Per-object local bounding boxes for this env, each of shape (1, 3).
            generator: RNG for reproducible sampling. None uses PyTorch's global RNG.

        Returns:
            Starting position for every object in ``objects``.
        """


def get_world_bbox_at_initial_pose(obj: PlaceableAsset, env_bboxes: EnvBoundingBoxes) -> AxisAlignedBoundingBox:
    """Return obj's local bbox translated to its fixed initial pose."""
    initial_pose = obj.get_initial_pose()
    assert isinstance(
        initial_pose, Pose
    ), f"Object '{obj.name}' must have a fixed Pose to use its env bbox, got {type(initial_pose).__name__}."
    return env_bboxes[obj].translated(initial_pose.position_xyz)


def sample_uniform(low: float, high: float, generator: torch.Generator | None = None) -> float:
    """Sample uniformly from [low, high], returning the midpoint when the interval is empty."""
    if low >= high:
        return float((low + high) / 2.0)
    return float(low + (high - low) * torch.rand(1, generator=generator).item())


class AnchorInitializer(InitializerBase):
    """Seeds every object against an anchor's footprint.

    Objects with an ``On`` relation are sampled inside the footprint of the anchor at or above
    their parent; all others start at the first anchor's center. Only one level of ``On``
    indirection is resolved, so a child of a non-anchor parent is seeded across the whole anchor
    rather than across its actual parent. Prefer ``OnTreeInitializer``, which resolves the full
    chain; this class preserves the historical behaviour for comparison and fallback.
    """

    def generate_initial_positions(
        self,
        objects: list[PlaceableAsset],
        anchor_objects: set[PlaceableAsset],
        env_bboxes: EnvBoundingBoxes,
        generator: torch.Generator | None = None,
    ) -> dict[PlaceableAsset, Position]:
        first_anchor = next(obj for obj in objects if obj in anchor_objects)
        anchor_bbox = get_world_bbox_at_initial_pose(first_anchor, env_bboxes)
        center = anchor_bbox.center[0]
        anchor_center = (float(center[0]), float(center[1]), float(center[2]))

        positions: dict[PlaceableAsset, Position] = {}
        for obj in objects:
            if obj in anchor_objects:
                positions[obj] = _fixed_anchor_position(obj)
            elif get_relation(obj, On) is not None:
                parent_bbox = self._get_on_parent_world_bbox(obj, anchor_objects, anchor_bbox, env_bboxes)
                positions[obj] = sample_on_parent(obj, parent_bbox, env_bboxes, generator)
            else:
                positions[obj] = anchor_center
        return positions

    @staticmethod
    def _get_on_parent_world_bbox(
        obj: PlaceableAsset,
        anchor_objects: set[PlaceableAsset],
        anchor_bbox: AxisAlignedBoundingBox,
        env_bboxes: EnvBoundingBoxes,
    ) -> AxisAlignedBoundingBox:
        """Resolve the world bbox of an On relation's parent for initialization purposes.

        If the parent is an anchor, return its world bbox directly. If the parent is a non-anchor
        with its own On(anchor) relation, use the anchor's world bbox as a proxy. Only one level of
        indirection is resolved; deeper chains fall back to anchor_bbox.
        """
        parent = get_relation(obj, On).parent
        if parent in anchor_objects:
            return get_world_bbox_at_initial_pose(parent, env_bboxes)
        for relation in parent.get_relations():
            if isinstance(relation, On) and relation.parent in anchor_objects:
                return get_world_bbox_at_initial_pose(relation.parent, env_bboxes)
        return anchor_bbox


class OnTreeInitializer(InitializerBase):
    """Seeds objects by walking the ``On`` forest from its anchor roots downwards.

    Each object is sampled inside the footprint its own parent was just sampled at, so a child
    lands on its real parent however deep the chain runs, and the sampled interval is intersected
    with any ``PositionLimitsBox`` on the object. The limits intersection matters as much as the
    chain walk: a parent seeded outside its own limits gets dragged across the scene during the
    solve, sweeping its children out of its footprint and piling them against the trailing edge.
    """

    def generate_initial_positions(
        self,
        objects: list[PlaceableAsset],
        anchor_objects: set[PlaceableAsset],
        env_bboxes: EnvBoundingBoxes,
        generator: torch.Generator | None = None,
    ) -> dict[PlaceableAsset, Position]:
        first_anchor = next(obj for obj in objects if obj in anchor_objects)
        fallback_center = get_world_bbox_at_initial_pose(first_anchor, env_bboxes).center[0]
        fallback_position = (float(fallback_center[0]), float(fallback_center[1]), float(fallback_center[2]))

        positions: dict[PlaceableAsset, Position] = {}
        world_bboxes: dict[PlaceableAsset, AxisAlignedBoundingBox] = {}
        for obj in _order_parents_before_children(objects, anchor_objects):
            if obj in anchor_objects:
                positions[obj] = _fixed_anchor_position(obj)
            else:
                on_relation = get_relation(obj, On)
                # An unparented object has no footprint to sample within, so the solver's
                # other relation losses are left to carry it away from the anchor center.
                if on_relation is None:
                    positions[obj] = fallback_position
                else:
                    parent_bbox = world_bboxes[on_relation.parent]
                    positions[obj] = sample_on_parent(
                        obj, parent_bbox, env_bboxes, generator, apply_position_limits=True
                    )
            world_bboxes[obj] = env_bboxes[obj].translated(positions[obj])
        # Restore the caller's ordering; the solver indexes positions by object, but callers
        # compare these dicts and a reordered dict is needlessly confusing to read.
        return {obj: positions[obj] for obj in objects}


def _fixed_anchor_position(obj: PlaceableAsset) -> Position:
    """Return an anchor's fixed spawn position."""
    initial_pose = obj.get_initial_pose()
    assert isinstance(
        initial_pose, Pose
    ), f"Anchor object '{obj.name}' must have a fixed Pose before placement, got {type(initial_pose).__name__}."
    return initial_pose.position_xyz


def _order_parents_before_children(
    objects: list[PlaceableAsset],
    anchor_objects: set[PlaceableAsset],
) -> list[PlaceableAsset]:
    """Return objects ordered so every On parent precedes its children.

    Anchors and unparented objects come first; the rest follow in dependency order.
    """
    ordered: list[PlaceableAsset] = []
    placed: set[PlaceableAsset] = set()
    remaining: list[PlaceableAsset] = []
    for obj in objects:
        on_relation = get_relation(obj, On)
        if obj in anchor_objects or on_relation is None:
            ordered.append(obj)
            placed.add(obj)
        else:
            remaining.append(obj)

    while remaining:
        ready = [obj for obj in remaining if get_relation(obj, On).parent in placed]
        assert ready, (
            "On relations must form a forest rooted at anchors, but no parent could be resolved for "
            f"{[obj.name for obj in remaining]}. Check for a cycle or a parent outside the placement set."
        )
        ordered += ready
        placed.update(ready)
        ready_set = set(ready)
        remaining = [obj for obj in remaining if obj not in ready_set]
    return ordered


def sample_on_parent(
    obj: PlaceableAsset,
    parent_world_bbox: AxisAlignedBoundingBox,
    env_bboxes: EnvBoundingBoxes,
    generator: torch.Generator | None = None,
    apply_position_limits: bool = False,
) -> Position:
    """Sample a position for obj on top of parent_world_bbox.

    X and Y are drawn from the parent's full footprint inset by the child's extents, and Z is set
    so the child's bottom face rests on the parent's top surface plus the relation's clearance.

    The footprint is deliberately not inset by the relation's ``edge_margin_m``. Seeding into the
    margin ring costs nothing — the On loss pulls the object inward from there — while the extra
    area measurably separates crowded surfaces: insetting dropped a 13-object robolab desk from
    74% to 64% valid layouts.

    Args:
        obj: The object being seeded; must carry an ``On`` relation.
        parent_world_bbox: World-space bbox of the parent, shape (1, 3).
        env_bboxes: Per-object local bounding boxes for this env.
        generator: Optional RNG generator for reproducible sampling.
        apply_position_limits: Intersect the sampled X/Y interval with the object's
            PositionLimitsBox, so the seed already satisfies the limits the solver would
            otherwise drag it to.
    """
    on_relation = get_relation(obj, On)
    child_bbox = env_bboxes[obj]

    child_min, child_max = child_bbox.min_point[0], child_bbox.max_point[0]
    if on_relation.overlap:
        # Intersection compares the child's far edge with the parent's near edge.
        child_min, child_max = child_max, child_min
    limits = get_relation(obj, PositionLimitsBox) if apply_position_limits else None

    position_xy: list[float] = []
    for axis, (limit_min, limit_max) in enumerate(_axis_limits(limits)):
        on_low = float(parent_world_bbox.min_point[0, axis]) - float(child_min[axis])
        on_high = float(parent_world_bbox.max_point[0, axis]) - float(child_max[axis])
        if on_low >= on_high:
            # Child does not fit on the parent along this axis; seed at the parent's center.
            position_xy.append(float(parent_world_bbox.center[0, axis]))
            continue
        low = on_low if limit_min is None else max(on_low, limit_min)
        high = on_high if limit_max is None else min(on_high, limit_max)
        if low >= high:
            # Footprint and limits are disjoint. Seed at the footprint point nearest the limits
            # so the solve starts from the shortest reconciliation the two constraints allow.
            position_xy.append(min(max((low + high) / 2.0, on_low), on_high))
            continue
        position_xy.append(sample_uniform(low, high, generator))

    # Convert from child-origin Z to child-bottom Z so the bottom face lands on the parent top.
    z = float(parent_world_bbox.max_point[0, 2] + on_relation.clearance_m - child_bbox.min_point[0, 2])
    return (position_xy[0], position_xy[1], z)


def _axis_limits(limits: PositionLimitsBox | None) -> list[tuple[float | None, float | None]]:
    """Return per-axis (min, max) bounds for X and Y, unbounded when no limits relation applies."""
    if limits is None:
        return [(None, None), (None, None)]
    return [(limits.x_min, limits.x_max), (limits.y_min, limits.y_max)]
