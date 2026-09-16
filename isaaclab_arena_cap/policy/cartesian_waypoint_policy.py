# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""YAML-driven, frame-aware Cartesian waypoint execution.

Vendored from Isaac-cap (``isaac_cap.cartesian_waypoint_policy``); the plans it
executes originate from BerkeleyAutomation/AUTOLab-GaP-NVIDIA-Collaboration.
"""

from __future__ import annotations

import math
import numpy as np
import os
import yaml
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, ClassVar

from isaaclab_arena.assets.register import register_policy
from isaaclab_arena.policy.policy_base import PolicyBase, PolicyCfg
from isaaclab_arena_cap.policy.cartesian_waypoint_follower import NewtonLmIkSolver

# Relative plan paths resolve against the module's bundled ``plan/`` directory.
_CONFIG_ROOT = Path(os.environ.get("ARENA_GAP_ROOT") or Path(__file__).resolve().parent / "plan")
_FORMAT = "cap-cartesian-waypoints-v1"


@dataclass(frozen=True)
class Pose:
    position: np.ndarray
    quaternion: np.ndarray


@dataclass(frozen=True)
class TargetSpec:
    asset: str
    control_frame: str
    control_frame_offset: Pose
    arm_action: str
    gripper_action: str
    arm_joint_names: tuple[str, ...]
    initial_gripper: float


@dataclass(frozen=True)
class TargetCommand:
    reference_frame: str
    pose: Pose | None
    gripper: float | None


@dataclass(frozen=True)
class Waypoint:
    name: str
    duration: float
    targets: Mapping[str, TargetCommand]
    max_translation_speed: float | None
    max_rotation_speed: float | None
    interpolation: str | None
    gripper_interpolation: str | None


@dataclass(frozen=True)
class WaypointPlan:
    targets: Mapping[str, TargetSpec]
    waypoints: tuple[Waypoint, ...]
    max_translation_speed: float
    max_rotation_speed: float
    interpolation: str
    gripper_interpolation: str


def _number(value: object, field: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0.0):
        qualifier = "positive and " if positive else ""
        raise ValueError(f"{field} must be {qualifier}finite")
    return result


def _vector(value: object, length: int, field: str) -> np.ndarray:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != length:
        raise ValueError(f"{field} must contain {length} numbers")
    return np.asarray([_number(item, field) for item in value], dtype=np.float64)


def _mapping(value: object, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a mapping")
    return value


def _string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def _gripper(value: object, field: str) -> float:
    if isinstance(value, str):
        normalized = value.casefold()
        if normalized == "open":
            return 0.0
        if normalized in {"closed", "close"}:
            return 1.0
    result = _number(value, field)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{field} must be open, closed, or a number from 0 to 1")
    return result


def _interpolation(value: object, field: str) -> str:
    result = _string(value, field)
    if result not in {"linear", "smoothstep"}:
        raise ValueError(f"{field} must be linear or smoothstep")
    return result


def _gripper_interpolation(value: object, field: str) -> str:
    result = _string(value, field)
    if result not in {"step", "linear", "smoothstep"}:
        raise ValueError(f"{field} must be step, linear, or smoothstep")
    return result


def _normalize_quaternion(value: Sequence[float]) -> np.ndarray:
    quaternion = np.asarray(value, dtype=np.float64)
    norm = float(np.linalg.norm(quaternion))
    if quaternion.shape != (4,) or not math.isfinite(norm) or norm < 1.0e-12:
        raise ValueError("quaternion must contain four finite non-zero XYZW values")
    return quaternion / norm


def _quaternion_from_rpy(value: Sequence[float]) -> np.ndarray:
    """Convert roll, pitch, yaw in radians to an XYZW quaternion."""
    roll, pitch, yaw = value
    cr, sr = math.cos(0.5 * roll), math.sin(0.5 * roll)
    cp, sp = math.cos(0.5 * pitch), math.sin(0.5 * pitch)
    cy, sy = math.cos(0.5 * yaw), math.sin(0.5 * yaw)
    return np.array(
        [
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy,
        ],
        dtype=np.float64,
    )


def _pose(value: object, field: str) -> Pose:
    document = _mapping(value, field)
    unknown = set(document) - {"position_xyz", "quaternion_xyzw", "rotation_rpy"}
    if unknown:
        raise ValueError(f"{field} has unsupported fields {sorted(unknown)}")
    rotation_fields = {name for name in ("quaternion_xyzw", "rotation_rpy") if name in document}
    if len(rotation_fields) != 1:
        raise ValueError(f"{field} must contain exactly one of quaternion_xyzw or rotation_rpy")
    if "quaternion_xyzw" in document:
        quaternion = _vector(document["quaternion_xyzw"], 4, f"{field}.quaternion_xyzw")
    else:
        rotation_rpy = _vector(document["rotation_rpy"], 3, f"{field}.rotation_rpy")
        quaternion = _quaternion_from_rpy(rotation_rpy)
    return Pose(
        _vector(document.get("position_xyz"), 3, f"{field}.position_xyz"),
        _normalize_quaternion(quaternion),
    )


def parse_waypoint_plan(document: object) -> WaypointPlan:
    root = _mapping(document, "waypoint plan")
    unknown = set(root) - {"format", "defaults", "targets", "waypoints", "source"}
    if unknown:
        raise ValueError(f"waypoint plan has unsupported fields {sorted(unknown)}")
    if root.get("format") != _FORMAT:
        raise ValueError(f"waypoint plan format must be {_FORMAT!r}")
    defaults = _mapping(root.get("defaults", {}), "defaults")
    unknown_defaults = set(defaults) - {
        "max_translation_speed",
        "max_rotation_speed",
        "interpolation",
        "gripper_interpolation",
    }
    if unknown_defaults:
        raise ValueError(f"defaults has unsupported fields {sorted(unknown_defaults)}")
    translation_speed = _number(
        defaults.get("max_translation_speed", 0.306186218),
        "defaults.max_translation_speed",
        positive=True,
    )
    rotation_speed = _number(
        defaults.get("max_rotation_speed", 3.06186218),
        "defaults.max_rotation_speed",
        positive=True,
    )
    default_interpolation = _interpolation(defaults.get("interpolation", "linear"), "defaults.interpolation")
    default_gripper_interpolation = _gripper_interpolation(
        defaults.get("gripper_interpolation", "step"),
        "defaults.gripper_interpolation",
    )

    raw_targets = _mapping(root.get("targets"), "targets")
    if not raw_targets:
        raise ValueError("targets must not be empty")
    targets: dict[str, TargetSpec] = {}
    for raw_name, raw_target in raw_targets.items():
        name = _string(raw_name, "target name")
        target = _mapping(raw_target, f"targets.{name}")
        allowed = {
            "asset",
            "control_frame",
            "control_frame_offset",
            "arm_action",
            "gripper_action",
            "arm_joint_names",
            "initial_gripper",
        }
        if unknown_target := set(target) - allowed:
            raise ValueError(f"targets.{name} has unsupported fields {sorted(unknown_target)}")
        raw_joints = target.get("arm_joint_names", ())
        if not isinstance(raw_joints, Sequence) or isinstance(raw_joints, (str, bytes)):
            raise ValueError(f"targets.{name}.arm_joint_names must be a sequence")
        joints = tuple(_string(item, f"targets.{name}.arm_joint_names") for item in raw_joints)
        if not joints:
            raise ValueError(f"targets.{name}.arm_joint_names must not be empty")
        targets[name] = TargetSpec(
            asset=_string(target.get("asset"), f"targets.{name}.asset"),
            control_frame=_string(target.get("control_frame"), f"targets.{name}.control_frame"),
            control_frame_offset=(
                _pose(
                    target["control_frame_offset"],
                    f"targets.{name}.control_frame_offset",
                )
                if "control_frame_offset" in target
                else Pose(
                    np.zeros(3),
                    np.array([0.0, 0.0, 0.0, 1.0]),
                )
            ),
            arm_action=_string(target.get("arm_action"), f"targets.{name}.arm_action"),
            gripper_action=_string(target.get("gripper_action"), f"targets.{name}.gripper_action"),
            arm_joint_names=joints,
            initial_gripper=_gripper(
                target.get("initial_gripper", "open"),
                f"targets.{name}.initial_gripper",
            ),
        )

    raw_waypoints = root.get("waypoints")
    if not isinstance(raw_waypoints, Sequence) or isinstance(raw_waypoints, (str, bytes)) or not raw_waypoints:
        raise ValueError("waypoints must be a non-empty sequence")
    waypoints: list[Waypoint] = []
    for index, raw_waypoint in enumerate(raw_waypoints):
        prefix = f"waypoints[{index}]"
        waypoint = _mapping(raw_waypoint, prefix)
        allowed = {
            "name",
            "duration",
            "targets",
            "max_translation_speed",
            "max_rotation_speed",
            "interpolation",
            "gripper_interpolation",
        }
        if unknown_waypoint := set(waypoint) - allowed:
            raise ValueError(f"{prefix} has unsupported fields {sorted(unknown_waypoint)}")
        commands: dict[str, TargetCommand] = {}
        for raw_target_name, raw_command in _mapping(waypoint.get("targets"), f"{prefix}.targets").items():
            target_name = _string(raw_target_name, f"{prefix} target name")
            if target_name not in targets:
                raise ValueError(f"{prefix} references unknown target {target_name!r}")
            command = _mapping(raw_command, f"{prefix}.targets.{target_name}")
            if unknown_command := set(command) - {"reference_frame", "pose", "gripper"}:
                raise ValueError(f"{prefix}.targets.{target_name} has unsupported fields {sorted(unknown_command)}")
            pose = _pose(command["pose"], f"{prefix}.targets.{target_name}.pose") if "pose" in command else None
            gripper = (
                _gripper(command["gripper"], f"{prefix}.targets.{target_name}.gripper")
                if "gripper" in command
                else None
            )
            if pose is None and gripper is None:
                raise ValueError(f"{prefix}.targets.{target_name} commands nothing")
            commands[target_name] = TargetCommand(
                reference_frame=_string(
                    command.get("reference_frame", "world"),
                    f"{prefix}.targets.{target_name}.reference_frame",
                ),
                pose=pose,
                gripper=gripper,
            )
        if not commands:
            raise ValueError(f"{prefix}.targets must not be empty")
        waypoints.append(
            Waypoint(
                name=_string(waypoint.get("name"), f"{prefix}.name"),
                duration=_number(waypoint.get("duration"), f"{prefix}.duration", positive=True),
                targets=commands,
                max_translation_speed=(
                    _number(
                        waypoint["max_translation_speed"],
                        f"{prefix}.max_translation_speed",
                        positive=True,
                    )
                    if "max_translation_speed" in waypoint
                    else None
                ),
                max_rotation_speed=(
                    _number(
                        waypoint["max_rotation_speed"],
                        f"{prefix}.max_rotation_speed",
                        positive=True,
                    )
                    if "max_rotation_speed" in waypoint
                    else None
                ),
                interpolation=(
                    _interpolation(waypoint["interpolation"], f"{prefix}.interpolation")
                    if "interpolation" in waypoint
                    else None
                ),
                gripper_interpolation=(
                    _gripper_interpolation(
                        waypoint["gripper_interpolation"],
                        f"{prefix}.gripper_interpolation",
                    )
                    if "gripper_interpolation" in waypoint
                    else None
                ),
            )
        )
    return WaypointPlan(
        targets,
        tuple(waypoints),
        translation_speed,
        rotation_speed,
        default_interpolation,
        default_gripper_interpolation,
    )


def load_waypoint_plan(path_value: str) -> WaypointPlan:
    path = Path(path_value)
    if not path.is_absolute():
        if any(part in {"", ".", ".."} for part in path.parts):
            raise ValueError("waypoint plan_path must be package-relative")
        path = _CONFIG_ROOT / path
    return parse_waypoint_plan(yaml.safe_load(path.read_text(encoding="utf-8")))


@dataclass
class CartesianWaypointPolicyArgs(PolicyCfg):
    requires_newton_native_actuators: ClassVar[bool] = True
    minimum_newton_njmax: ClassVar[int] = 512
    policy_device: str = "cuda"
    plan_path: str = ""
    terminate_on_exhaustion: bool = True


@register_policy
class CartesianWaypointPolicy(PolicyBase[CartesianWaypointPolicyArgs]):
    """Execute one frame-aware waypoint plan through named robot action terms."""

    name = "cartesian_waypoints"

    def __init__(self, config: CartesianWaypointPolicyArgs):
        super().__init__(config)
        self._config = config
        self.device = config.policy_device
        self.task_description: str | None = None
        self._plan = load_waypoint_plan(config.plan_path)
        self._runtime: dict[str, _TargetRuntime] | None = None
        self._frame_snapshots: dict[str, Pose] = {}
        self._starts: dict[str, Pose] = {}
        self._goals: dict[str, Pose] = {}
        self._gripper_starts: dict[str, float] = {}
        self._gripper_goals: dict[str, float] = {}
        self._waypoint_index = 0
        self._waypoint_step = 0
        self._waypoint_duration = 0.0
        self._termination_mask = None

    def get_action(self, env: Any, observation: dict[str, Any]):
        del observation
        import torch

        unwrapped = env.unwrapped
        if unwrapped.num_envs != 1:
            raise ValueError("Cartesian waypoint policy requires num_envs=1")
        if self._runtime is None:
            self._initialize(unwrapped)
        assert self._runtime is not None
        action = torch.zeros(
            (1, unwrapped.action_manager.total_action_dim),
            dtype=torch.float32,
            device=unwrapped.device,
        )
        if self._waypoint_index >= len(self._plan.waypoints):
            for name, runtime in self._runtime.items():
                runtime.write_action(action, self._goals[name], self._gripper_goals[name])
            return action
        progress = min(
            1.0,
            (self._waypoint_step + 1) * float(unwrapped.step_dt) / self._waypoint_duration,
        )
        waypoint = self._plan.waypoints[self._waypoint_index]
        interpolation = waypoint.interpolation or self._plan.interpolation
        alpha = progress * progress * (3.0 - 2.0 * progress) if interpolation == "smoothstep" else progress
        gripper_interpolation = waypoint.gripper_interpolation or self._plan.gripper_interpolation
        if gripper_interpolation == "step":
            gripper_alpha = 1.0
        elif gripper_interpolation == "smoothstep":
            gripper_alpha = progress * progress * (3.0 - 2.0 * progress)
        else:
            gripper_alpha = progress
        for name, runtime in self._runtime.items():
            pose = interpolate_pose(self._starts[name], self._goals[name], alpha)
            gripper = self._gripper_starts[name] + gripper_alpha * (
                self._gripper_goals[name] - self._gripper_starts[name]
            )
            runtime.write_action(action, pose, gripper)
        self._waypoint_step += 1
        if progress >= 1.0:
            self._starts = dict(self._goals)
            self._gripper_starts = dict(self._gripper_goals)
            self._waypoint_index += 1
            self._waypoint_step = 0
            if self._waypoint_index == len(self._plan.waypoints):
                if self._config.terminate_on_exhaustion:
                    self._termination_mask = torch.ones(1, dtype=torch.bool, device=unwrapped.device)
            else:
                self._prepare_waypoint(self._plan.waypoints[self._waypoint_index])
        return action

    def _initialize(self, env: Any) -> None:
        slices = _action_slices(env.action_manager)
        used_terms: set[str] = set()
        runtime: dict[str, _TargetRuntime] = {}
        for name, spec in self._plan.targets.items():
            for term_name in (spec.arm_action, spec.gripper_action):
                if term_name in used_terms:
                    raise ValueError(f"action term {term_name!r} is bound more than once")
                if term_name not in slices:
                    raise ValueError(f"action term {term_name!r} is not active")
                used_terms.add(term_name)
            runtime[name] = _TargetRuntime.create(env, spec, slices)
        if used_terms != set(slices):
            missing = sorted(set(slices) - used_terms)
            raise ValueError(f"waypoint targets do not bind active action terms {missing}")
        self._runtime = runtime
        self._starts = {name: target.measured_pose() for name, target in runtime.items()}
        self._goals = dict(self._starts)
        self._gripper_starts = {name: spec.initial_gripper for name, spec in self._plan.targets.items()}
        self._gripper_goals = dict(self._gripper_starts)
        references = {
            command.reference_frame
            for waypoint in self._plan.waypoints
            for command in waypoint.targets.values()
            if command.pose is not None
        }
        self._frame_snapshots = {reference: resolve_frame_pose(env, reference) for reference in references}
        self._prepare_waypoint(self._plan.waypoints[0])

    def _prepare_waypoint(self, waypoint: Waypoint) -> None:
        goals = dict(self._starts)
        grippers = dict(self._gripper_starts)
        for name, command in waypoint.targets.items():
            if command.pose is not None:
                goals[name] = compose_pose(self._frame_snapshots[command.reference_frame], command.pose)
            if command.gripper is not None:
                grippers[name] = command.gripper
        translation_speed = waypoint.max_translation_speed or self._plan.max_translation_speed
        rotation_speed = waypoint.max_rotation_speed or self._plan.max_rotation_speed
        duration = waypoint.duration
        for name in self._starts:
            distance = float(np.linalg.norm(goals[name].position - self._starts[name].position))
            angle = quaternion_distance(self._starts[name].quaternion, goals[name].quaternion)
            duration = max(duration, distance / translation_speed, angle / rotation_speed)
        self._goals = goals
        self._gripper_goals = grippers
        self._waypoint_duration = duration

    def get_episode_termination_mask(self):
        return self._termination_mask if self._config.terminate_on_exhaustion else None

    def reset(self, env_ids=None) -> None:
        del env_ids
        self._runtime = None
        self._frame_snapshots = {}
        self._starts = {}
        self._goals = {}
        self._gripper_starts = {}
        self._gripper_goals = {}
        self._waypoint_index = 0
        self._waypoint_step = 0
        self._waypoint_duration = 0.0
        self._termination_mask = None

    def close(self) -> None:
        self._runtime = None


@dataclass
class _TargetRuntime:
    env: Any
    spec: TargetSpec
    arm_slice: slice
    gripper_slice: slice
    solver: Any | None

    @classmethod
    def create(cls, env: Any, spec: TargetSpec, slices: Mapping[str, slice]) -> _TargetRuntime:
        arm_slice = slices[spec.arm_action]
        gripper_slice = slices[spec.gripper_action]
        if gripper_slice.stop - gripper_slice.start != 1:
            raise ValueError(f"gripper action {spec.gripper_action!r} must be one-dimensional")
        if arm_slice.stop - arm_slice.start != len(spec.arm_joint_names):
            raise ValueError("Newton IK arm action dimension does not match arm_joint_names")
        asset = env.scene[spec.asset]
        articulation_name = PurePosixPath(str(asset.cfg.prim_path)).name
        solver = NewtonLmIkSolver.from_runtime(
            arm_joint_names=spec.arm_joint_names,
            end_effector_body_name=spec.control_frame,
            articulation_name=articulation_name,
            asset_prim_path=str(asset.cfg.prim_path),
            link_offset_position=spec.control_frame_offset.position,
            link_offset_quaternion=spec.control_frame_offset.quaternion,
        )
        return cls(env, spec, arm_slice, gripper_slice, solver)

    def measured_pose(self) -> Pose:
        body = resolve_frame_pose(self.env, f"{self.spec.asset}/{self.spec.control_frame}")
        return compose_pose(body, self.spec.control_frame_offset)

    def write_action(self, action: Any, pose: Pose, gripper: float) -> None:
        import torch

        assert self.solver is not None
        values = self.solver.solve(pose.position, pose.quaternion)
        action[:, self.arm_slice] = torch.as_tensor(values, dtype=action.dtype, device=action.device).reshape(1, -1)
        action[:, self.gripper_slice] = gripper


def _action_slices(manager: Any) -> dict[str, slice]:
    names = list(manager.active_terms)
    dimensions = list(manager.action_term_dim)
    if len(names) != len(dimensions):
        raise ValueError("action manager term names and dimensions disagree")
    result: dict[str, slice] = {}
    start = 0
    for name, dimension in zip(names, dimensions, strict=True):
        result[name] = slice(start, start + int(dimension))
        start += int(dimension)
    if start != manager.total_action_dim:
        raise ValueError("action manager term dimensions do not match total_action_dim")
    return result


def resolve_frame_pose(env: Any, reference: str) -> Pose:
    if reference == "world":
        return Pose(np.zeros(3), np.array([0.0, 0.0, 0.0, 1.0]))
    asset_name, separator, body_name = reference.partition("/")
    try:
        asset = env.scene[asset_name]
    except KeyError as error:
        raise ValueError(f"reference frame asset {asset_name!r} is not in the scene") from error
    if separator:
        indices, _names = asset.find_bodies(body_name)
        if len(indices) != 1:
            raise ValueError(f"reference frame {reference!r} is missing or ambiguous")
        index = indices[0]
        return Pose(
            _numpy(asset.data.body_pos_w[0, index]),
            _normalize_quaternion(_numpy(asset.data.body_quat_w[0, index])),
        )
    return Pose(
        _numpy(asset.data.root_pos_w[0]),
        _normalize_quaternion(_numpy(asset.data.root_quat_w[0])),
    )


def compose_pose(parent: Pose, child: Pose) -> Pose:
    return Pose(
        parent.position + rotate_vector(parent.quaternion, child.position),
        _normalize_quaternion(quaternion_multiply(parent.quaternion, child.quaternion)),
    )


def interpolate_pose(start: Pose, goal: Pose, alpha: float) -> Pose:
    return Pose(
        start.position + float(alpha) * (goal.position - start.position),
        slerp(start.quaternion, goal.quaternion, float(alpha)),
    )


def quaternion_conjugate(value: Sequence[float]) -> np.ndarray:
    x, y, z, w = _normalize_quaternion(value)
    return np.array([-x, -y, -z, w])


def quaternion_multiply(left: Sequence[float], right: Sequence[float]) -> np.ndarray:
    x1, y1, z1, w1 = left
    x2, y2, z2, w2 = right
    return np.array([
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    ])


def rotate_vector(quaternion: Sequence[float], vector: Sequence[float]) -> np.ndarray:
    rotated = quaternion_multiply(
        quaternion_multiply(quaternion, np.concatenate((vector, [0.0]))),
        quaternion_conjugate(quaternion),
    )
    return rotated[:3]


def quaternion_distance(left: Sequence[float], right: Sequence[float]) -> float:
    dot = abs(float(np.dot(_normalize_quaternion(left), _normalize_quaternion(right))))
    return 2.0 * math.acos(min(1.0, max(-1.0, dot)))


def slerp(start: Sequence[float], goal: Sequence[float], alpha: float) -> np.ndarray:
    left = _normalize_quaternion(start)
    right = _normalize_quaternion(goal)
    dot = float(np.dot(left, right))
    if dot < 0.0:
        right = -right
        dot = -dot
    dot = min(1.0, max(-1.0, dot))
    if dot > 0.9995:
        return _normalize_quaternion(left + alpha * (right - left))
    angle = math.acos(dot)
    scale = math.sin(angle)
    return _normalize_quaternion(
        math.sin((1.0 - alpha) * angle) / scale * left + math.sin(alpha * angle) / scale * right
    )


def _numpy(value: Any) -> np.ndarray:
    if hasattr(value, "torch"):
        value = value.torch
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float64)
