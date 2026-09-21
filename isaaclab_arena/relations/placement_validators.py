# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
import torch
import trimesh
from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import TYPE_CHECKING, ClassVar, cast

from isaaclab.utils.math import quat_apply, quat_apply_inverse, quat_conjugate, quat_mul

from isaaclab_arena.relations.collision_mode import CollisionMode, get_object_collision_mode, object_uses_mesh_collision
from isaaclab_arena.relations.placement_validation import PlacementCheck
from isaaclab_arena.relations.placement_validator_registry import PlacementValidatorRegistry, register_validator
from isaaclab_arena.relations.relation_loss_strategies import (
    SIDE_CONFIGS,
    NotNextToLossStrategy,
    next_to_violations,
    not_next_to_violations,
)
from isaaclab_arena.relations.relations import FaceTo, NextTo, NotNextTo, On, get_relation
from isaaclab_arena.relations.warp_sdf_kernels import has_sdf_sentinel, mesh_sdf
from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox
from isaaclab_arena.utils.pose import Pose
from isaaclab_arena.utils.yaw import centers_in_target_frame, wrap_angle_to_pi, yaw_from_quat_xyzw, yaw_toward_positions

if TYPE_CHECKING:
    from isaaclab_arena.relations.collision_object import CollisionObject
    from isaaclab_arena.relations.object_placer_params import ObjectPlacerParams
    from isaaclab_arena.relations.placement_asset import PlaceableAsset
    from isaaclab_arena.relations.placement_visualizer import PlacementRerunVisualizer
    from isaaclab_arena.relations.warp_mesh_manager import WarpMeshAndSphereCache


class PlacementValidator(ABC):
    """A single build-time placement check evaluated over a batch of candidate layouts.

    Register a concrete validator with @register_validator so build_validators() can discover it.
    """

    check: ClassVar[str]
    """The check name this validator reports; its registry key and result key. Built-ins use a
    PlacementCheck constant; external validators may use any unique string."""

    run_after_inexpensive_checks: bool = False
    """If True, run this validator only on candidates that already pass every required check that does not
    set this flag, so an expensive check (e.g. IK reachability) never runs on a layout rejected on cheaper
    geometry."""

    def __init__(self, params: ObjectPlacerParams, visualizer: PlacementRerunVisualizer | None = None) -> None:
        self._params = params
        self._visualizer = visualizer

    @classmethod
    def is_available(cls, params: ObjectPlacerParams) -> bool:
        """Whether this validator can run for these params at build time; unavailable ones are delisted.

        build_validators drops any registered validator that returns False. Defaults to True.
        """
        return True

    @abstractmethod
    def validate_batch(
        self,
        positions: list[dict[PlaceableAsset, tuple[float, float, float]]],
        orientations: list[dict[PlaceableAsset, float]],
        bboxes: list[dict[PlaceableAsset, AxisAlignedBoundingBox]],
        collision_objects: list[CollisionObject],
    ) -> list[bool]:
        """Return one pass/fail verdict per candidate layout.

        Args:
            positions: Solved (x, y, z) per object, one dict per candidate.
            orientations: Absolute world Z-yaw per object, one dict per candidate (may be empty).
            bboxes: Per-object bboxes for the candidate's env, one dict per candidate, each (1, 3).
            collision_objects: Fixed background obstacles shared across candidates.
        """
        pass

    def validate_measured_poses(
        self,
        poses: dict[PlaceableAsset, Pose],
        bboxes: dict[PlaceableAsset, AxisAlignedBoundingBox],
        collision_objects: list[CollisionObject],
    ) -> bool:
        """Check measured positions, headings and world-axis bounds in one environment frame.

        Validators that reconstruct rotations from solver layouts must override this method.
        """
        positions = {asset: pose.position_xyz for asset, pose in poses.items()}
        orientations = {asset: yaw_from_quat_xyzw(pose.rotation_xyzw) for asset, pose in poses.items()}
        return self.validate_batch([positions], [orientations], [bboxes], collision_objects)[0]


def get_build_time_checks() -> tuple[str, ...]:
    """Registered build-time check names, in registration order."""
    return tuple(PlacementValidatorRegistry().get_all_keys())


def build_validators(
    params: ObjectPlacerParams, visualizer: PlacementRerunVisualizer | None = None
) -> list[PlacementValidator]:
    """Construct the enabled build-time validators in registration order.

    A registered check whose is_available() returns False is delisted; a check named in
    enabled_checks/required_checks that is not registered is likewise dropped rather than
    raising.

    Args:
        params: Placement params injected into each registered validator.
        visualizer: The caller's debug view, injected into each validator so a check can draw its
            own visualization layer on it; None when the run has no view.
    """
    registry = PlacementValidatorRegistry()
    registered_checks = get_build_time_checks()

    enabled_checks = params.enabled_checks
    if enabled_checks is not None:
        registered_checks = tuple(check for check in registered_checks if check in enabled_checks)

    validators: list[PlacementValidator] = []
    for check in registered_checks:
        validator_cls = registry.get_validator_by_name(check)
        if validator_cls.is_available(params):
            validators.append(validator_cls(params, visualizer))
    return validators


@register_validator
class OnRelationValidator(PlacementValidator):
    """Support footprint and height checks for On and ClutterOn relations."""

    check = PlacementCheck.ON_RELATION

    def validate_batch(
        self,
        positions: list[dict[PlaceableAsset, tuple[float, float, float]]],
        orientations: list[dict[PlaceableAsset, float]],
        bboxes: list[dict[PlaceableAsset, AxisAlignedBoundingBox]],
        collision_objects: list[CollisionObject],
    ) -> list[bool]:
        return [self._validate(positions[i], bboxes[i]) for i in range(len(positions))]

    def _validate(
        self,
        positions: dict[PlaceableAsset, tuple[float, float, float]],
        env_bboxes: dict[PlaceableAsset, AxisAlignedBoundingBox],
    ) -> bool:
        """Check support relations against the same bounds as their loss strategies.

        Args:
            positions: Solved positions for each object.
            env_bboxes: Per-object bounds for one environment, each with min/max shape (1, 3).
        """
        for obj in positions:
            for relation in obj.get_relations():
                if not isinstance(relation, On) or relation.parent not in positions:
                    continue
                if not self.validate_relation(
                    relation, positions[obj], positions[relation.parent], env_bboxes[obj], env_bboxes[relation.parent]
                ):
                    if self._params.verbose:
                        print(f"{type(relation).__name__}: '{obj.name}' outside support footprint or height range")
                    return False
        return True

    def validate_relation(
        self,
        relation: On,
        child_position: tuple[float, float, float],
        parent_position: tuple[float, float, float],
        child_bounds: AxisAlignedBoundingBox,
        parent_bounds: AxisAlignedBoundingBox,
    ) -> bool:
        """Check one support relation using rotation-adjusted bounds about each object origin."""
        child_world = child_bounds.translated(child_position)
        parent_world = relation.support_bbox(parent_bounds.translated(parent_position))
        # Preserve scalar height arithmetic at the strict On contact boundary.
        bottom = child_position[2] + float(child_bounds.min_point[0, 2])
        top = parent_position[2] + float(parent_bounds.max_point[0, 2])
        height_fits = relation.accepts_child_bottom(bottom, top, self._params.on_relation_z_tolerance_m)
        return height_fits and self._validate_footprint(relation, child_world, parent_world)

    def _validate_footprint(self, relation: On, child: AxisAlignedBoundingBox, parent: AxisAlignedBoundingBox) -> bool:
        """Check feasible margins and containment or overlap within the support footprint."""
        margin = 0.0 if relation.overlap else relation.edge_margin_m
        return self._margin_fits(child, parent, margin) and self._footprint_fits(relation, child, parent, margin)

    def _margin_fits(self, child: AxisAlignedBoundingBox, parent: AxisAlignedBoundingBox, margin: float) -> bool:
        """Whether the child fits inside the inset support footprint."""
        if margin == 0:
            return True
        free_space = (parent.size - child.size)[0, :2]
        if torch.any(free_space < 2 * margin):
            if self._params.verbose:
                feasible_margin = max(0.0, float(free_space.min()) / 2.0)
                print(f"Support relation: edge_margin_m={margin} exceeds feasible margin {feasible_margin:.3f} m")
            return False
        return True

    @staticmethod
    def _footprint_fits(
        relation: On, child: AxisAlignedBoundingBox, parent: AxisAlignedBoundingBox, margin: float
    ) -> bool:
        """Whether the child is contained by, or overlaps, the support in XY."""
        child_min, child_max = child.min_point[0, :2], child.max_point[0, :2]
        if relation.overlap:
            child_min, child_max = child_max, child_min
        return bool(
            torch.all(child_min >= parent.min_point[0, :2] + margin)
            and torch.all(child_max <= parent.max_point[0, :2] - margin)
        )


@register_validator
class NextToValidator(PlacementValidator):
    """Validate every NextTo relation: child on the requested side within the relation's tolerance_m."""

    check = PlacementCheck.NEXT_TO

    def validate_batch(
        self,
        positions: list[dict[PlaceableAsset, tuple[float, float, float]]],
        orientations: list[dict[PlaceableAsset, float]],
        bboxes: list[dict[PlaceableAsset, AxisAlignedBoundingBox]],
        collision_objects: list[CollisionObject],
    ) -> list[bool]:
        return [self._validate(positions[i], bboxes[i]) for i in range(len(positions))]

    def _validate(
        self,
        positions: dict[PlaceableAsset, tuple[float, float, float]],
        env_bboxes: dict[PlaceableAsset, AxisAlignedBoundingBox],
    ) -> bool:
        """Validate each NextTo relation: child on the requested side, facing edge within the
        relation's tolerance_m of distance_m from the parent edge. Shares next_to_violations with
        NextToLossStrategy; cross_position_ratio is a soft preference and is not gated.

        Args:
            positions: Solved positions for each object.
            env_bboxes: Per-object bboxes for the current env, each with shape (1, 3).
        """
        for obj in positions:
            for rel in obj.get_relations():
                if not isinstance(rel, NextTo):
                    continue
                parent = rel.parent
                if parent not in positions:
                    continue
                cfg = SIDE_CONFIGS[rel.side]
                child_bbox = env_bboxes[obj]
                child_pos = child_bbox.min_point.new_tensor([positions[obj]])
                parent_world = env_bboxes[parent].translated(positions[parent])
                half_plane, distance = next_to_violations(cfg, child_pos, child_bbox, parent_world, rel.distance_m)

                if half_plane.item() > rel.tolerance_m or distance.item() > rel.tolerance_m:
                    if self._params.verbose:
                        print(
                            f"NextTo: '{obj.name}' next_to({parent.name}) violated"
                            f" (side={half_plane.item():.4f}, distance={distance.item():.4f} m;"
                            f" tolerance_m={rel.tolerance_m})"
                        )
                    return False
        return True


@register_validator
class NotNextToValidator(PlacementValidator):
    """Validate every NotNextTo relation: child has cleared the keep-out zone beside the parent."""

    check = PlacementCheck.NOT_NEXT_TO

    def validate_batch(
        self,
        positions: list[dict[PlaceableAsset, tuple[float, float, float]]],
        orientations: list[dict[PlaceableAsset, float]],
        bboxes: list[dict[PlaceableAsset, AxisAlignedBoundingBox]],
        collision_objects: list[CollisionObject],
    ) -> list[bool]:
        return [self._validate(positions[i], bboxes[i]) for i in range(len(positions))]

    def _validate(
        self,
        positions: dict[PlaceableAsset, tuple[float, float, float]],
        env_bboxes: dict[PlaceableAsset, AxisAlignedBoundingBox],
    ) -> bool:
        """Validate each NotNextTo relation: child has cleared the keep-out zone beside the parent
        (within the relation's tolerance_m) via either route — back over the edge or past the
        footprint end. Shares not_next_to_violations with NotNextToLossStrategy, using its margin_m.

        Args:
            positions: Solved positions for each object.
            env_bboxes: Per-object bboxes for the current env, each with shape (1, 3).
        """
        for obj in positions:
            for rel in obj.get_relations():
                if not isinstance(rel, NotNextTo):
                    continue
                parent = rel.parent
                if parent not in positions:
                    continue
                cfg = SIDE_CONFIGS[rel.side]
                margin_m = self._not_next_to_margin(rel)
                child_bbox = env_bboxes[obj]
                child_pos = child_bbox.min_point.new_tensor([positions[obj]])
                parent_world = env_bboxes[parent].translated(positions[parent])
                remaining_side, remaining_cross = not_next_to_violations(
                    cfg, child_pos, child_bbox, parent_world, margin_m
                )

                if min(remaining_side.item(), remaining_cross.item()) > rel.tolerance_m:
                    if self._params.verbose:
                        print(
                            f"NotNextTo: '{obj.name}' not_next_to({parent.name}) violated"
                            f" (remaining_side={remaining_side.item():.4f},"
                            f" remaining_cross={remaining_cross.item():.4f} m;"
                            f" margin_m={margin_m}, tolerance_m={rel.tolerance_m})"
                        )
                    return False
        return True

    def _not_next_to_margin(self, relation: NotNextTo) -> float:
        """Keep-out margin_m from the registered NotNextTo loss strategy (stays in sync with the solver)."""
        strategy = cast(NotNextToLossStrategy, self._params.solver_params.strategies[type(relation)])
        return strategy.margin_m


@register_validator
class FaceToValidator(PlacementValidator):
    """Validate that each FaceTo subject points toward its target in XY."""

    check = PlacementCheck.FACE_TO

    def validate_batch(
        self,
        positions: list[dict[PlaceableAsset, tuple[float, float, float]]],
        orientations: list[dict[PlaceableAsset, float]],
        bboxes: list[dict[PlaceableAsset, AxisAlignedBoundingBox]],
        collision_objects: list[CollisionObject],
    ) -> list[bool]:
        return [self.validate_facing(positions[i], orientations[i], tolerance_rad=1e-4) for i in range(len(positions))]

    def validate_facing(
        self,
        positions: dict[PlaceableAsset, tuple[float, float, float]],
        orientations: dict[PlaceableAsset, float] | None,
        tolerance_rad: float,
    ) -> bool:
        """Require headings within tolerance_rad of the direction to each FaceTo target."""
        for obj in positions:
            face_to = get_relation(obj, FaceTo)
            if face_to is None:
                continue
            subject_position = torch.tensor([positions[obj]])
            target_position = torch.tensor([positions[face_to.parent]])
            target_yaw, direction_is_defined = yaw_toward_positions(subject_position, target_position)
            if not direction_is_defined.item():
                if self._params.verbose:
                    print(f"  FaceTo: '{obj.name}' is too close to its target in XY")
                return False
            if orientations is None or obj not in orientations:
                if self._params.verbose:
                    print(f"  FaceTo: '{obj.name}' has no computed facing yaw")
                return False
            measured_yaw = orientations[obj]
            if (
                not math.isfinite(measured_yaw)
                or abs(wrap_angle_to_pi(measured_yaw - float(target_yaw.item()))) > tolerance_rad
            ):
                if self._params.verbose:
                    print(f"  FaceTo: '{obj.name}' exceeds the facing tolerance ({tolerance_rad:g} rad)")
                return False
        return True


@register_validator
class NoOverlapValidator(PlacementValidator):
    """Validate that no two placed bounding boxes (or collision meshes) intersect.

    Owns the CPU mesh/sphere cache so the AABB→mesh short-circuit stays local: cheap AABB pairs are
    tested first, and only mesh-collision objects fall through to the sphere-to-SDF penetration test.
    """

    check = PlacementCheck.NO_OVERLAP

    def __init__(self, params: ObjectPlacerParams, visualizer: PlacementRerunVisualizer | None = None) -> None:
        super().__init__(params, visualizer)
        self._cpu_mesh_manager: WarpMeshAndSphereCache | None = None
        self._settled_mesh_manager: WarpMeshAndSphereCache | None = None

    def validate_batch(
        self,
        positions: list[dict[PlaceableAsset, tuple[float, float, float]]],
        orientations: list[dict[PlaceableAsset, float]],
        bboxes: list[dict[PlaceableAsset, AxisAlignedBoundingBox]],
        collision_objects: list[CollisionObject],
    ) -> list[bool]:
        return [
            self._validate(positions[i], bboxes[i], orientations[i], collision_objects) for i in range(len(positions))
        ]

    def _validate(
        self,
        positions: dict[PlaceableAsset, tuple[float, float, float]],
        env_bboxes: dict[PlaceableAsset, AxisAlignedBoundingBox],
        orientations: dict[PlaceableAsset, float] | None,
        collision_objects: list[CollisionObject] | None,
    ) -> bool:
        """AABB overlap check, falling through to mesh penetration for mesh-collision objects."""
        use_mesh = self._should_validate_mesh(positions, collision_objects)
        no_overlap = self._validate_no_overlap(
            positions,
            env_bboxes,
            collision_objects=collision_objects,
            skip_mesh_pairs=use_mesh,
        )
        if no_overlap and use_mesh:
            no_overlap = self._validate_no_overlap_mesh(positions, env_bboxes, orientations, collision_objects)
        return no_overlap

    def validate_measured_poses(
        self,
        poses: dict[PlaceableAsset, Pose],
        bboxes: dict[PlaceableAsset, AxisAlignedBoundingBox],
        collision_objects: list[CollisionObject],
        *,
        penetration_tolerance_m: float = 0.0,
    ) -> bool:
        """Check measured full rotations against collision meshes, allowing resting contact.

        Final contact checks use mesh geometry regardless of the solver's collision mode.
        Assets without meshes use conservative box proxies; extraction errors propagate.
        On-linked support pairs are checked by support validation instead.
        """
        from isaaclab_arena.relations.warp_mesh_manager import WarpMeshAndSphereCache
        from isaaclab_arena.utils.usd.helpers import NoCollisionMeshError

        if self._settled_mesh_manager is None:
            # Solver spheres include an extra 1 cm clearance, which would reject resting contact.
            self._settled_mesh_manager = WarpMeshAndSphereCache(
                num_spheres=self._params.solver_params.num_spheres, sphere_radius=0.0, device="cpu"
            )
        mesh_manager = self._settled_mesh_manager
        mesh_manager.reset_sentinel_warning()
        all_poses = dict(poses)
        for obstacle in collision_objects:
            pose = obstacle.get_initial_pose()
            assert isinstance(pose, Pose), f"Collision obstacle '{obstacle.name}' needs a fixed pose"
            all_poses[obstacle] = pose
        rotations = {asset: pose.rotation_xyzw for asset, pose in all_poses.items()}
        meshes, world_bounds, mesh_cache_objects = {}, {}, {}
        for asset, pose in all_poses.items():
            try:
                mesh = mesh_manager.get_collision_mesh_or_raise(asset)
            except NoCollisionMeshError:
                mesh = None
            mesh_cache_objects[asset] = asset if mesh is not None else None
            proxy_bounds = asset.get_bounding_box()
            if mesh is None and (asset.is_anchor or asset in collision_objects):
                initial = asset.get_initial_pose()
                assert isinstance(initial, Pose), f"Fixed collision object '{asset.name}' needs a Pose"
                # Fixed bounds may use parent USD axes, as for ObjectReference.
                proxy_bounds = asset.get_world_bounding_box().translated(tuple(-v for v in initial.position_xyz))
                rotations[asset] = tuple(
                    quat_mul(
                        torch.tensor(pose.rotation_xyzw), quat_conjugate(torch.tensor(initial.rotation_xyzw))
                    ).tolist()
                )
            meshes[asset] = self._collision_mesh_or_aabb_proxy(mesh, proxy_bounds)
            bounds = AxisAlignedBoundingBox(tuple(meshes[asset].bounds[0]), tuple(meshes[asset].bounds[1]))
            world_bounds[asset] = bounds.rotated_by_quat(rotations[asset]).translated(pose.position_xyz)
        positions = {asset: pose.position_xyz for asset, pose in poses.items()}
        pairs = list(self._non_skip_pairs(positions))
        for asset in poses:
            if not asset.is_anchor:
                pairs.extend((asset, obstacle) for obstacle in collision_objects)
        for first, second in pairs:
            if not world_bounds[first].overlaps(world_bounds[second], margin=-penetration_tolerance_m).item():
                continue
            for source, target in ((first, second), (second, first)):
                if self._spheres_penetrate_mesh(
                    source,
                    meshes[source],
                    mesh_cache_objects[source],
                    True,
                    True,
                    torch.tensor(all_poses[source].position_xyz, dtype=torch.float32),
                    target,
                    meshes[target],
                    torch.tensor(all_poses[target].position_xyz, dtype=torch.float32),
                    True,
                    mesh_manager,
                    -penetration_tolerance_m,
                    None,
                    rotations=rotations,
                    target_mesh_is_proxy=mesh_cache_objects[target] is None,
                ):
                    return False
        return True

    def _should_validate_mesh(
        self,
        positions: dict[PlaceableAsset, tuple[float, float, float]],
        collision_objects: list[CollisionObject] | None,
    ) -> bool:
        """Return True when any object in this validation uses mesh collision."""
        default_collision_mode = self._params.solver_params.collision_mode
        if default_collision_mode == CollisionMode.MESH:
            return True
        objects = [*positions.keys(), *(collision_objects or [])]
        return any(get_object_collision_mode(obj, default_collision_mode) == CollisionMode.MESH for obj in objects)

    @staticmethod
    def _collect_skip_pairs(
        positions: dict[PlaceableAsset, tuple[float, float, float]],
    ) -> tuple[set[tuple], set[int]]:
        """Build On-pair skip set and anchor ID set from positioned objects.

        Returns:
            Tuple of (on_pairs, anchor_ids) where on_pairs contains (id(a), id(b))
            tuples for On-linked objects, and anchor_ids contains id() of anchors.
        """
        on_pairs: set[tuple] = set()
        anchor_ids: set[int] = set()
        for obj in positions:
            for rel in obj.get_relations():
                if isinstance(rel, On) and rel.parent in positions:
                    on_pairs.add((id(obj), id(rel.parent)))
                    on_pairs.add((id(rel.parent), id(obj)))
            if obj.is_anchor:
                anchor_ids.add(id(obj))
        return on_pairs, anchor_ids

    def _non_skip_pairs(
        self,
        positions: dict[PlaceableAsset, tuple[float, float, float]],
        skip_mesh_pairs: bool = False,
    ) -> Iterator[tuple[PlaceableAsset, PlaceableAsset]]:
        """Yield non-relation object pairs, optionally skipping pairs handled by mesh collision."""
        on_pairs, anchor_ids = self._collect_skip_pairs(positions)
        mesh_manager = self._get_cpu_mesh_manager() if skip_mesh_pairs else None
        default_collision_mode = self._params.solver_params.collision_mode
        objects = list(positions.keys())
        for i in range(len(objects)):
            for j in range(i + 1, len(objects)):
                a, b = objects[i], objects[j]
                if id(a) in anchor_ids and id(b) in anchor_ids:
                    continue
                if (id(a), id(b)) in on_pairs:
                    continue
                if mesh_manager is not None and (
                    (
                        object_uses_mesh_collision(a, default_collision_mode)
                        and mesh_manager.get_collision_mesh(a) is not None
                    )
                    or (
                        object_uses_mesh_collision(b, default_collision_mode)
                        and mesh_manager.get_collision_mesh(b) is not None
                    )
                ):
                    continue
                yield a, b

    def _validate_no_overlap(
        self,
        positions: dict[PlaceableAsset, tuple[float, float, float]],
        env_bboxes: dict[PlaceableAsset, AxisAlignedBoundingBox],
        collision_objects: list[CollisionObject] | None = None,
        skip_mesh_pairs: bool = False,
    ) -> bool:
        """AABB overlap check on pre-rotated env_bboxes. Skips On-pairs and anchor-anchor pairs."""
        clearance_m = self._params.solver_params.clearance_m
        margin = max(0.0, clearance_m - 1e-6)
        collision_objects = collision_objects or []
        _, anchor_ids = self._collect_skip_pairs(positions)

        for a, b in self._non_skip_pairs(positions, skip_mesh_pairs=skip_mesh_pairs):
            if self._pair_aabb_overlaps(env_bboxes[a], env_bboxes[b], positions[a], positions[b], 0.0, 0.0, margin):
                if self._params.verbose:
                    print(f"  Overlap between '{a.name}' and '{b.name}'")
                return False

        # Placed (non-anchor) objects must also clear the fixed background obstacles.
        # Anchors are fixed scene geometry too, so anchor-vs-background overlap is not gated.
        background_worlds = [(bg, bg.get_world_bounding_box()) for bg in collision_objects]
        mesh_manager = self._get_cpu_mesh_manager() if skip_mesh_pairs else None
        default_collision_mode = self._params.solver_params.collision_mode
        for obj in positions:
            if id(obj) in anchor_ids:
                continue
            obj_world = env_bboxes[obj].translated(positions[obj])
            for background, background_world in background_worlds:
                if (
                    mesh_manager is not None
                    and object_uses_mesh_collision(background, default_collision_mode)
                    and mesh_manager.get_collision_mesh(background) is not None
                ):
                    continue
                if obj_world.overlaps(background_world, margin=margin).item():
                    if self._params.verbose:
                        print(f"  Overlap between '{obj.name}' and background '{background.name}'")
                    return False
        return True

    def _get_cpu_mesh_manager(self) -> WarpMeshAndSphereCache:
        """Return the CPU-device mesh manager, creating it on first call."""
        if self._cpu_mesh_manager is None:
            from isaaclab_arena.relations.warp_mesh_manager import WarpMeshAndSphereCache

            self._cpu_mesh_manager = WarpMeshAndSphereCache(
                num_spheres=self._params.solver_params.num_spheres,
                device="cpu",
            )
        return self._cpu_mesh_manager

    def _validate_no_overlap_mesh(
        self,
        positions: dict[PlaceableAsset, tuple[float, float, float]],
        env_bboxes: dict[PlaceableAsset, AxisAlignedBoundingBox],
        orientations: dict[PlaceableAsset, float] | None = None,
        collision_objects: list[CollisionObject] | None = None,
    ) -> bool:
        """Sphere-to-SDF overlap check; both-meshless pairs fall back to AABB validation."""
        clearance_m = self._params.solver_params.clearance_m
        tolerance = max(0.0, clearance_m - 1e-6)
        mesh_manager = self._get_cpu_mesh_manager()
        mesh_manager.reset_sentinel_warning()
        warned_no_mesh: set[str] = set()
        collision_objects = collision_objects or []
        default_collision_mode = self._params.solver_params.collision_mode

        for a, b in self._non_skip_pairs(positions):
            a_uses_mesh = object_uses_mesh_collision(a, default_collision_mode)
            b_uses_mesh = object_uses_mesh_collision(b, default_collision_mode)
            a_mesh = mesh_manager.get_collision_mesh(a) if a_uses_mesh else None
            b_mesh = mesh_manager.get_collision_mesh(b) if b_uses_mesh else None
            if a_mesh is None and b_mesh is None:
                for obj, uses_mesh, mesh in [(a, a_uses_mesh, a_mesh), (b, b_uses_mesh, b_mesh)]:
                    if uses_mesh and mesh is None and obj.name not in warned_no_mesh:
                        warned_no_mesh.add(obj.name)
                        print(
                            f"  [NoCollision] MESH mode: '{obj.name}' has no collision mesh,"
                            " falling back to AABB validation for this pair"
                        )
                continue

            a_pos = torch.tensor(positions[a], dtype=torch.float32)
            b_pos = torch.tensor(positions[b], dtype=torch.float32)

            if b_mesh is not None and self._spheres_penetrate_mesh(
                a,
                self._collision_mesh_or_aabb_proxy(a_mesh, env_bboxes[a]),
                a if a_mesh is not None else None,
                a_mesh is not None,
                a.is_anchor,
                a_pos,
                b,
                b_mesh,
                b_pos,
                b.is_anchor,
                mesh_manager,
                tolerance,
                orientations,
            ):
                return False
            if a_mesh is not None and self._spheres_penetrate_mesh(
                b,
                self._collision_mesh_or_aabb_proxy(b_mesh, env_bboxes[b]),
                b if b_mesh is not None else None,
                b_mesh is not None,
                b.is_anchor,
                b_pos,
                a,
                a_mesh,
                a_pos,
                a.is_anchor,
                mesh_manager,
                tolerance,
                orientations,
            ):
                return False

        for source in positions:
            if source.is_anchor:
                continue
            source_mesh = (
                mesh_manager.get_collision_mesh(source)
                if object_uses_mesh_collision(source, default_collision_mode)
                else None
            )
            source_pos = torch.tensor(positions[source], dtype=torch.float32)
            for background in collision_objects:
                target_mesh = (
                    mesh_manager.get_collision_mesh(background)
                    if object_uses_mesh_collision(background, default_collision_mode)
                    else None
                )
                if target_mesh is None:
                    continue
                target_pose = background.get_initial_pose()
                assert isinstance(
                    target_pose, Pose
                ), f"Background collision object '{background.name}' must have a fixed Pose in MESH mode."
                target_pos = torch.tensor(target_pose.position_xyz, dtype=torch.float32)
                if self._spheres_penetrate_mesh(
                    source,
                    self._collision_mesh_or_aabb_proxy(source_mesh, env_bboxes[source]),
                    source if source_mesh is not None else None,
                    source_mesh is not None,
                    source.is_anchor,
                    source_pos,
                    background,
                    target_mesh,
                    target_pos,
                    True,
                    mesh_manager,
                    tolerance,
                    orientations,
                ):
                    return False

        return True

    @staticmethod
    def _collision_mesh_or_aabb_proxy(
        mesh: trimesh.Trimesh | None,
        bbox: AxisAlignedBoundingBox,
    ) -> trimesh.Trimesh:
        """Return an object's collision mesh, or a box mesh matching the candidate AABB."""
        if mesh is not None:
            return mesh
        box_mesh = trimesh.creation.box(extents=bbox.size[0].detach().cpu().numpy())
        box_mesh.apply_translation(bbox.center[0].detach().cpu().numpy())
        return box_mesh

    def _spheres_penetrate_mesh(
        self,
        source: PlaceableAsset,
        source_mesh: trimesh.Trimesh,
        source_sphere_cache_obj: PlaceableAsset | None,
        source_applies_yaw: bool,
        source_uses_pose_yaw: bool,
        source_pos: torch.Tensor,
        target: PlaceableAsset | CollisionObject,
        target_mesh: trimesh.Trimesh,
        target_pos: torch.Tensor,
        target_uses_pose_yaw: bool,
        mesh_manager: WarpMeshAndSphereCache,
        tolerance: float,
        orientations: dict[PlaceableAsset, float] | None,
        rotations: dict[CollisionObject, tuple[float, float, float, float]] | None = None,
        target_mesh_is_proxy: bool = False,
    ) -> bool:
        """True if source's spheres penetrate target's mesh or if BVH returns no-face sentinel.

        source_applies_yaw describes whether sphere centers need sampled-yaw rotation.
        *_uses_pose_yaw controls whether fixed anchors/passive obstacles contribute pose yaw.
        """
        spheres = mesh_manager.get_query_spheres(source_mesh, obj=source_sphere_cache_obj)
        warp_mesh = mesh_manager.get_warp_mesh(target_mesh, obj=None if target_mesh_is_proxy else target)
        centers = self._centers_in_target_frame(
            spheres[:, :3],
            source,
            target,
            source_pos,
            target_pos,
            orientations,
            source_applies_yaw=source_applies_yaw,
            source_uses_pose_yaw=source_uses_pose_yaw,
            target_uses_pose_yaw=target_uses_pose_yaw,
            rotations=rotations,
        )
        sdf = mesh_sdf(centers, warp_mesh)
        mesh_manager.warn_sdf_sentinel(sdf)
        if has_sdf_sentinel(sdf):
            return True
        if (sdf < spheres[:, 3] + tolerance).any():
            if self._params.verbose:
                print(f"  Mesh overlap between '{source.name}' and '{target.name}'")
            return True
        return False

    @staticmethod
    def _effective_yaw(
        obj: PlaceableAsset | CollisionObject,
        orientations: dict[PlaceableAsset, float] | None,
        use_pose_yaw: bool,
    ) -> float:
        """Resolve effective Z-yaw from sampled orientations or, when allowed, fixed initial pose."""
        if orientations is not None and obj in orientations:
            return orientations[cast("PlaceableAsset", obj)]
        if not use_pose_yaw:
            return 0.0
        pose = obj.get_initial_pose()
        if not isinstance(pose, Pose):
            return 0.0
        return yaw_from_quat_xyzw(pose.rotation_xyzw)

    @staticmethod
    def _pair_aabb_overlaps(
        a_bbox: AxisAlignedBoundingBox,
        b_bbox: AxisAlignedBoundingBox,
        pos_a: tuple[float, float, float],
        pos_b: tuple[float, float, float],
        yaw_a: float,
        yaw_b: float,
        margin: float,
    ) -> bool:
        """True if two yaw-rotated, world-translated AABBs overlap (with margin)."""
        if yaw_a != 0.0:
            a_bbox = a_bbox.rotated_around_z(yaw_a)
        if yaw_b != 0.0:
            b_bbox = b_bbox.rotated_around_z(yaw_b)
        a_world = a_bbox.translated(pos_a)
        b_world = b_bbox.translated(pos_b)
        return a_world.overlaps(b_world, margin=margin).item()

    @staticmethod
    def _centers_in_target_frame(
        centers_local: torch.Tensor,
        source_obj: PlaceableAsset,
        target_obj: PlaceableAsset | CollisionObject,
        source_pos: torch.Tensor,
        target_pos: torch.Tensor,
        orientations: dict[PlaceableAsset, float] | None,
        source_applies_yaw: bool = True,
        source_uses_pose_yaw: bool = True,
        target_uses_pose_yaw: bool = True,
        rotations: dict[CollisionObject, tuple[float, float, float, float]] | None = None,
    ) -> torch.Tensor:
        """Transform sphere centers using full rotations when supplied, otherwise solver yaws."""
        if rotations is not None:
            assert source_applies_yaw and source_uses_pose_yaw and target_uses_pose_yaw and orientations is None
            source_rotation = torch.tensor(rotations[source_obj], dtype=centers_local.dtype).expand(
                len(centers_local), 4
            )
            target_rotation = torch.tensor(rotations[target_obj], dtype=centers_local.dtype).expand(
                len(centers_local), 4
            )
            centers_world = quat_apply(source_rotation, centers_local) + source_pos - target_pos
            return quat_apply_inverse(target_rotation, centers_world)
        src_yaw = (
            NoOverlapValidator._effective_yaw(source_obj, orientations, source_uses_pose_yaw)
            if source_applies_yaw
            else 0.0
        )
        tgt_yaw = NoOverlapValidator._effective_yaw(target_obj, orientations, target_uses_pose_yaw)
        return centers_in_target_frame(centers_local, src_yaw, tgt_yaw, source_pos - target_pos)
