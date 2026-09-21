# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Physics-filtered recordings of solved object and robot placements."""

from __future__ import annotations

import math
import torch
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from isaaclab.utils.math import quat_conjugate, quat_mul

from isaaclab_arena.relations.bounding_box_helpers import build_per_env_bounding_boxes, has_heterogeneous_objects
from isaaclab_arena.relations.clutter.geometry import (
    ClutterRegion,
    dynamic_rigid_object_keys,
    spawned_geometry_is_fixed,
    spawned_rigid_body_has_gravity,
)
from isaaclab_arena.relations.clutter.settle_params import ClutterSettleParams
from isaaclab_arena.relations.clutter.validation import check_resting_poses
from isaaclab_arena.relations.physics_settle_params import PlacementRecordingParams
from isaaclab_arena.relations.placement_events import get_base_rotation_per_asset, write_layout_to_sim
from isaaclab_arena.relations.placement_layouts import PlacementLayouts
from isaaclab_arena.relations.placement_validation import PlacementCheck
from isaaclab_arena.relations.placement_validators import (
    FaceToValidator,
    NoOverlapValidator,
    OnRelationValidator,
    PlacementValidator,
)
from isaaclab_arena.relations.relations import ClutterOn, On, RandomAroundSolution, get_relation
from isaaclab_arena.utils import physics_settle
from isaaclab_arena.utils.pose import Pose
from isaaclab_arena.utils.scene_snapshot import SceneSnapshot, articulation_link_poses_in_root_frame
from isaaclab_arena.utils.yaw import yaw_from_quat_xyzw

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from isaaclab_arena.relations.placement_asset import PlaceableAsset
    from isaaclab_arena.relations.pooled_object_placer import PooledObjectPlacer
    from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox


@dataclass
class PlacementRecordingResult:
    """Accepted settled layouts and the outcome of each source candidate."""

    layouts: PlacementLayouts
    """Complete final poses keyed by runtime scene name."""
    accepted_indices: list[tuple[int, int]]
    """Source (environment index, queue index) for each output layout, in file order."""
    rejections: dict[tuple[int, int], str]
    """Failure reason for each rejected source (environment index, queue index)."""

    @property
    def attempted(self) -> int:
        """Total source candidates, including solver failures."""
        return len(self.accepted_indices) + len(self.rejections)


def collect_settled_pool_layouts(
    env: ManagerBasedEnv,
    placement_pool: PooledObjectPlacer,
    params: PlacementRecordingParams | None = None,
    render: bool = False,
    scene_assets: list[PlaceableAsset] | None = None,
) -> PlacementRecordingResult:
    """Settle and filter complete layouts without consuming or changing the source pool.

    Scene roots, joints and actuator targets are restored on success or failure.

    Args:
        env: Initialized environment containing the pool's scene assets.
        placement_pool: Solved layouts grouped by absolute environment index.
        params: Physics time, permitted motion and minimum accepted layout count.
        render: Render the offline physics steps.
        scene_assets: Asset definitions for articulations outside the placement pool.

    Returns:
        Final environment-local poses and source indices, plus rejected-candidate reasons.
    """
    env = env.unwrapped
    params = replace(params) if params is not None else PlacementRecordingParams()
    assert placement_pool.num_envs == env.num_envs, "Placement pool and scene must have the same environment count"
    assets = list(placement_pool.objects)
    for asset in scene_assets or []:
        if asset.get_scene_key() in env.scene.articulations and asset not in assets:
            assets.append(asset)
    keys = _recording_keys(env, assets)
    validators = _final_pose_validators(placement_pool)
    anchors = {asset for asset in assets if asset.is_anchor}
    clutter_keys = {asset.get_scene_key() for asset in assets if get_relation(asset, ClutterOn) is not None}
    rotations = get_base_rotation_per_asset(assets)
    boxes = build_per_env_bounding_boxes(assets, env.num_envs)
    snapshot = SceneSnapshot(env, keys)
    initial_links = articulation_link_poses_in_root_frame(env)
    queues = placement_pool.layouts_per_env()
    accepted: dict[str, list[Pose]] = {key: [] for key in keys}
    accepted_indices, rejections = [], {}
    try:
        for index in range(max((len(queue) for queue in queues), default=0)):
            snapshot.restore(env)
            env_ids = []
            for env_id, queue in enumerate(queues):
                if index >= len(queue):
                    continue
                if not queue[index].success:
                    rejections[env_id, index] = "solver validation failed"
                    continue
                write_layout_to_sim(env, env_id, queue[index], anchors, rotations)
                env_ids.append(env_id)
            if not env_ids:
                continue
            env.scene.write_data_to_sim()
            env.sim.forward()
            initial_poses = {key: env.arena_world.get_pose_e(key) for key in keys}
            quiet = _settle(env, env_ids, keys, params, render)
            final_poses = {key: env.arena_world.get_pose_e(key) for key in keys}
            final_links = articulation_link_poses_in_root_frame(env)
            for candidate_index, env_id in enumerate(env_ids):
                reason = _motion_failure(initial_poses, final_poses, env_id, clutter_keys, params)
                if reason is None and not quiet[candidate_index]:
                    reason = "objects did not remain at rest"
                if reason is None:
                    reason = _link_motion_failure(initial_links, final_links, env_id, params)
                if reason is None:
                    reason = _validate_final_pose(
                        env, env_id, boxes.get_bounding_boxes_for_env_id(env_id), validators, placement_pool, params
                    )
                if reason is not None:
                    rejections[env_id, index] = reason
                    print(f"[recording] rejected env {env_id}, layout {index}: {reason}")
                    continue
                _append_layout(accepted, final_poses, env_id)
                accepted_indices.append((env_id, index))
        assert (
            len(accepted_indices) >= params.min_layouts
        ), f"Accepted {len(accepted_indices)} layouts; need {params.min_layouts}. Rejections: {rejections}"
        return PlacementRecordingResult(PlacementLayouts(accepted), accepted_indices, rejections)
    finally:
        snapshot.restore(env)


def _append_layout(accepted: dict[str, list[Pose]], poses: dict[str, torch.Tensor], env_id: int) -> None:
    """Append one complete layout from environment-local xyz/xyzw tensors shaped (N, 7)."""
    for key, values in poses.items():
        pose = values[env_id].tolist()
        accepted[key].append(Pose(tuple(pose[:3]), tuple(pose[3:])))


def _settle(
    env: ManagerBasedEnv, env_ids: list[int], keys: list[str], params: PlacementRecordingParams, render: bool
) -> list[bool]:
    """Require low velocities on every physics step throughout the quiet window."""
    dt = env.sim.get_physics_dt()
    physics_settle.step_physics(env, math.ceil(params.settle_time_s / dt), render=render)
    quiet = [True] * len(env_ids)
    for _ in range(math.ceil(params.quiet_time_s / dt)):
        physics_settle.step_physics(env, 1, render=render)
        current = physics_settle.are_all_objects_settled_per_env(
            env, env_ids, keys, params.lin_vel_thresh, params.ang_vel_thresh
        )
        quiet = [previous and settled for previous, settled in zip(quiet, current, strict=True)]
    return quiet


def _motion_failure(
    initial: dict[str, torch.Tensor],
    final: dict[str, torch.Tensor],
    env_id: int,
    clutter_keys: set[str],
    params: PlacementRecordingParams,
) -> str | None:
    """Reject invalid poses and excessive ordinary-object drift in the environment frame."""
    for key, poses in final.items():
        if not torch.isfinite(poses[env_id]).all():
            return f"{key}: non-finite pose"
        if key not in clutter_keys:
            reason = physics_settle.pose_drift_reason(
                initial[key][env_id], poses[env_id], params.max_translation_m, params.max_rotation_deg
            )
            if reason:
                return f"{key}: {reason}"
    return None


def _link_motion_failure(
    initial: dict[str, torch.Tensor], final: dict[str, torch.Tensor], env_id: int, params: PlacementRecordingParams
) -> str | None:
    """Reject unrecorded joint motion while allowing a recorded articulation root to move."""
    for key, poses in final.items():
        reason = physics_settle.pose_drift_reason(
            initial[key][env_id], poses[env_id], params.max_link_translation_m, params.max_link_rotation_deg
        )
        if reason:
            return f"{key} links: {reason}; joint states are not recorded"
    return None


def _validate_final_pose(
    env: ManagerBasedEnv,
    env_id: int,
    boxes: dict[PlaceableAsset, AxisAlignedBoundingBox],
    validators: list[PlacementValidator],
    pool: PooledObjectPlacer,
    params: PlacementRecordingParams,
) -> str | None:
    """Check ordinary relations, clutter containment and measured collision geometry."""
    poses, positions, orientations, rotated_boxes = {}, {}, {}, {}
    for asset, bounds in boxes.items():
        values = env.arena_world.get_pose_e(asset.get_scene_key())[env_id].cpu()
        if not torch.isfinite(values).all():
            return f"{asset.name}: non-finite pose"
        pose = Pose(tuple(values[:3].tolist()), tuple(values[3:].tolist()))
        poses[asset], positions[asset] = pose, pose.position_xyz
        orientations[asset] = yaw_from_quat_xyzw(pose.rotation_xyzw)
        rotation = values[3:]
        if asset.is_anchor:
            initial = asset.get_initial_pose()
            assert isinstance(initial, Pose), f"Anchor '{asset.name}' needs a fixed pose"
            rotation = quat_mul(rotation, quat_conjugate(torch.tensor(initial.rotation_xyzw)))
        rotated_boxes[asset] = bounds.to(torch.device("cpu")).rotated_by_quat(rotation.reshape(1, 4))
    for validator in validators:
        if isinstance(validator, OnRelationValidator):
            reason = _support_failure(validator, positions, rotated_boxes)
            if reason:
                return reason
        elif isinstance(validator, FaceToValidator):
            if not validator.validate_facing(positions, orientations, math.radians(params.max_facing_error_deg)):
                return "final pose failed face_to"
        elif isinstance(validator, NoOverlapValidator):
            if not validator.validate_measured_poses(
                poses, rotated_boxes, pool.collision_objects, penetration_tolerance_m=params.penetration_tolerance_m
            ):
                return "final pose failed no_overlap"
        elif not validator.validate_measured_poses(poses, rotated_boxes, pool.collision_objects):
            return f"final pose failed {validator.check}"
    return None


def _support_failure(
    validator: OnRelationValidator,
    positions: dict[PlaceableAsset, tuple[float, float, float]],
    boxes: dict[PlaceableAsset, AxisAlignedBoundingBox],
) -> str | None:
    """Use full support containment for settled clutter and ordinary On bounds elsewhere."""
    containment_params = ClutterSettleParams()
    for asset in positions:
        for relation in asset.get_relations():
            if not isinstance(relation, On):
                continue
            parent = relation.parent
            if isinstance(relation, ClutterOn):
                support = boxes[parent].translated(positions[parent])
                lower, upper = support.min_point[0], support.max_point[0]
                region = ClutterRegion(
                    float(lower[0]), float(lower[1]), float(upper[0]), float(upper[1]), float(upper[2])
                )
                verdict = check_resting_poses(boxes[asset].translated(positions[asset]), region, containment_params)
                if not verdict.ok:
                    return f"{asset.name}: {verdict.describe([asset.name])}"
            elif not validator.validate_relation(
                relation, positions[asset], positions[parent], boxes[asset], boxes[parent]
            ):
                return f"{asset.name}: final pose failed on_relation"
    return None


def _recording_keys(env: ManagerBasedEnv, assets: list[PlaceableAsset]) -> list[str]:
    """Require replayable roots, concrete geometry and fixed clutter supports."""
    assert not has_heterogeneous_objects(assets), "Resolve object sets before recording reusable layouts"
    asset_set = set(assets)
    for asset in assets:
        key = asset.get_scene_key()
        for relation in asset.get_relations():
            if isinstance(relation, On):
                assert relation.parent in asset_set, (
                    f"'{key}' support '{relation.parent.name}' must participate in placement; "
                    "mark a fixed support IsAnchor"
                )
        assert (
            get_relation(asset, RandomAroundSolution) is None
        ), f"'{key}': remove RandomAroundSolution before recording for cached replay"
        assert (
            asset.is_anchor or key in env.scene.rigid_objects or key in env.scene.articulations
        ), f"'{key}' needs a writable rigid or articulation root"
        relation = get_relation(asset, ClutterOn)
        if relation is not None:
            assert key in env.scene.rigid_objects, f"Clutter member '{key}' must be a rigid object"
            assert not spawned_geometry_is_fixed(env.scene, key), f"Clutter member '{key}' must be dynamic"
            assert spawned_rigid_body_has_gravity(env.scene, key), f"Clutter member '{key}' needs gravity"
            gravity = env.cfg.sim.gravity
            assert (
                gravity[0] == 0 and gravity[1] == 0 and gravity[2] < 0
            ), "Clutter settling requires downward world-Z gravity"
            assert spawned_geometry_is_fixed(
                env.scene, relation.parent.get_scene_key()
            ), f"Clutter support '{relation.parent.name}' must be static or kinematic"
    dynamic_keys = set(dynamic_rigid_object_keys(env.scene))
    asset_keys = {asset.get_scene_key() for asset in assets}
    missing_articulations = env.scene.articulations.keys() - asset_keys
    assert (
        not missing_articulations
    ), f"Pass scene_assets for unplaced articulations so their geometry can be checked: {missing_articulations}"
    assert (
        not dynamic_keys - asset_keys
    ), "Dynamic neighbors must participate in placement (use IsAnchor for fixed initial placements)"
    keys = sorted(
        dynamic_keys | set(env.scene.articulations) | {asset.get_scene_key() for asset in assets if not asset.is_anchor}
    )
    assert keys, "Recording requires movable placement objects"
    return keys


def _final_pose_validators(pool: PooledObjectPlacer) -> list[PlacementValidator]:
    """Preserve the pool's required checks and add mandatory support and collision checks."""
    validators = {validator.check: validator for validator in pool.validators}
    params = pool.placer_params
    required = set(validators) | (params.enabled_checks or set()) | (params.required_checks or set())
    required |= {PlacementCheck.ON_RELATION, PlacementCheck.NO_OVERLAP}
    validators.setdefault(PlacementCheck.ON_RELATION, OnRelationValidator(params))
    validators.setdefault(PlacementCheck.NO_OVERLAP, NoOverlapValidator(params))
    missing = required - validators.keys() - {PlacementCheck.PHYSICS_SETTLED}
    assert not missing, f"Required final-pose validators are unavailable: {missing}"
    return [validator for check, validator in validators.items() if check in required]
