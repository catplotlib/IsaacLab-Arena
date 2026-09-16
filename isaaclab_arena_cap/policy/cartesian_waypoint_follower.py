# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reusable absolute-pose waypoint tracking for Newton-controlled robots.

Vendored unchanged from Isaac-cap (``isaac_cap.cartesian_waypoint_follower``).
"""

from __future__ import annotations

import math
import numpy as np
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any, Protocol

_DEFAULT_OUTPUT_MAX = (0.05, 0.05, 0.05, 0.5, 0.5, 0.5)
# Newton's tiled LM kernel fails to launch above roughly this size: 77
# coordinates solve, 91 abort with "Warp CUDA error 1: invalid argument"
# from _lm_solve_tiled. The exact ceiling is somewhere between; this bounds
# it at the largest value not observed to fail, so oversized scenes get this
# message instead of an opaque CUDA error.
_LM_COORDINATE_LIMIT = 90


@dataclass(frozen=True)
class CartesianWaypoint:
    """One absolute world-frame tool pose and its gripper command."""

    label: str
    position_xyz: tuple[float, float, float]
    quaternion_xyzw: tuple[float, float, float, float]
    gripper: float
    max_rotation_delta: float | None = None
    max_translation_delta: float | None = None
    minimum_hold_seconds: float | None = None
    timeout_seconds: float | None = None


@dataclass(frozen=True)
class CartesianWaypointStep:
    """Command and progression state produced by one follower update."""

    label: str
    joint_targets: Any
    gripper: float
    advanced: bool
    completed: bool
    waypoint_index: int
    elapsed_steps: int


class CartesianIkSolver(Protocol):
    """Minimal IK interface used by :class:`CartesianWaypointFollower`."""

    def solve(self, position: np.ndarray, quaternion: np.ndarray) -> Any:
        """Return the controlled arm's absolute joint targets."""


def _normalize_quaternion(quaternion: Sequence[float]) -> np.ndarray:
    value = np.asarray(quaternion, dtype=np.float64)
    if value.shape != (4,):
        raise ValueError("quaternion must have four XYZW entries")
    norm = float(np.linalg.norm(value))
    if not math.isfinite(norm) or norm < 1.0e-12:
        raise ValueError("quaternion must be finite and non-zero")
    return value / norm


def _step_toward(origin: np.ndarray, target: np.ndarray, limit: float) -> np.ndarray:
    offset = target - origin
    distance = float(np.linalg.norm(offset))
    if distance <= limit or distance < 1.0e-12:
        return target.copy()
    return origin + offset * (limit / distance)


def _rotation_distance(origin: np.ndarray, target: np.ndarray) -> float:
    return 2.0 * float(np.arccos(np.clip(abs(float(np.dot(origin, target))), -1.0, 1.0)))


def _turn_toward(origin: np.ndarray, target: np.ndarray, limit: float) -> np.ndarray:
    dot = float(np.dot(origin, target))
    equivalent_target = -target if dot < 0.0 else target
    dot = float(np.clip(abs(dot), -1.0, 1.0))
    angle = 2.0 * float(np.arccos(dot))
    if angle <= limit or angle < 1.0e-9:
        return equivalent_target.copy()
    half_angle = float(np.arccos(dot))
    fraction = limit / angle
    denominator = math.sin(half_angle)
    result = (
        math.sin((1.0 - fraction) * half_angle) / denominator * origin
        + math.sin(fraction * half_angle) / denominator * equivalent_target
    )
    return _normalize_quaternion(result)


class UpstreamOscSetpoint:
    """Upstream OSC's stateful rate limit and measured-pose lead clamp."""

    def __init__(
        self,
        *,
        control_dt: float,
        kp: float = 150.0,
        damping_ratio: float = 1.0,
        output_max: Sequence[float] = _DEFAULT_OUTPUT_MAX,
        measured_lead_clamp: bool = True,
    ) -> None:
        if control_dt <= 0.0 or kp <= 0.0 or damping_ratio <= 0.0:
            raise ValueError("OSC timestep and gains must be positive")
        output = np.asarray(output_max, dtype=np.float64)
        if output.shape != (6,) or np.any(output <= 0.0):
            raise ValueError("OSC output_max must contain six positive entries")
        self.control_dt = float(control_dt)
        self.kp = float(kp)
        self.kd = 2.0 * math.sqrt(self.kp) * float(damping_ratio)
        self.output_max = output
        self.position_speed = self.kp / self.kd * float(np.max(output[:3]))
        self.rotation_speed = self.kp / self.kd * float(np.max(output[3:]))
        self.position_lead = float(np.max(output[:3]))
        self.rotation_lead = float(np.max(output[3:]))
        self.measured_lead_clamp = bool(measured_lead_clamp)
        self._position = np.zeros(3, dtype=np.float64)
        self._quaternion = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        self._trajectory_position = self._position.copy()
        self._trajectory_quaternion = self._quaternion.copy()

    def reset(
        self,
        position: Sequence[float],
        quaternion_xyzw: Sequence[float],
    ) -> None:
        value = np.asarray(position, dtype=np.float64)
        if value.shape != (3,) or not np.all(np.isfinite(value)):
            raise ValueError("position must contain three finite entries")
        self._position = value.copy()
        self._trajectory_position = value.copy()
        self._quaternion = _normalize_quaternion(quaternion_xyzw)
        self._trajectory_quaternion = self._quaternion.copy()

    def advance(
        self,
        *,
        measured_position: Sequence[float],
        measured_quaternion: Sequence[float],
        goal_position: Sequence[float],
        goal_quaternion: Sequence[float],
        translation_output_max: float | None = None,
        rotation_output_max: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        measured_position_array = np.asarray(measured_position, dtype=np.float64)
        goal_position_array = np.asarray(goal_position, dtype=np.float64)
        measured_quaternion_array = _normalize_quaternion(measured_quaternion)
        goal_quaternion_array = _normalize_quaternion(goal_quaternion)
        if measured_position_array.shape != (3,) or goal_position_array.shape != (3,):
            raise ValueError("measured and goal positions must contain XYZ triples")

        position_lead = self.position_lead
        position_speed = self.position_speed
        if translation_output_max is not None:
            if translation_output_max <= 0.0:
                raise ValueError("translation_output_max must be positive")
            position_lead = float(translation_output_max)
            position_speed = self.kp / self.kd * position_lead

        rotation_lead = self.rotation_lead
        rotation_speed = self.rotation_speed
        if rotation_output_max is not None:
            if rotation_output_max <= 0.0:
                raise ValueError("rotation_output_max must be positive")
            rotation_lead = float(rotation_output_max)
            rotation_speed = self.kp / self.kd * rotation_lead

        trajectory_position = _step_toward(
            self._trajectory_position,
            goal_position_array,
            position_speed * self.control_dt,
        )
        trajectory_quaternion = _turn_toward(
            self._trajectory_quaternion,
            goal_quaternion_array,
            rotation_speed * self.control_dt,
        )
        if self.measured_lead_clamp:
            position = _step_toward(
                measured_position_array,
                trajectory_position,
                position_lead,
            )
            quaternion = _turn_toward(
                measured_quaternion_array,
                trajectory_quaternion,
                rotation_lead,
            )
            if float(np.linalg.norm(goal_position_array - position)) > float(
                np.linalg.norm(goal_position_array - self._position)
            ):
                position = self._position.copy()
            if _rotation_distance(goal_quaternion_array, quaternion) > _rotation_distance(
                goal_quaternion_array, self._quaternion
            ):
                quaternion = self._quaternion.copy()
        else:
            position = trajectory_position
            quaternion = trajectory_quaternion
        self._trajectory_position = trajectory_position
        self._trajectory_quaternion = trajectory_quaternion
        self._position = position
        self._quaternion = quaternion
        return position.copy(), quaternion.copy()


class CartesianWaypointFollower:
    """Follow an ordered list of absolute Cartesian poses with policy-owned IK."""

    def __init__(
        self,
        *,
        ik_solver: CartesianIkSolver,
        control_dt: float,
        minimum_hold_seconds: float,
        waypoint_timeout_seconds: float,
        position_tolerance: float,
        orientation_tolerance: float,
        max_translation_delta: float = 0.05,
        max_rotation_delta: float = 0.5,
    ) -> None:
        values = (
            control_dt,
            minimum_hold_seconds,
            waypoint_timeout_seconds,
            position_tolerance,
            orientation_tolerance,
            max_translation_delta,
            max_rotation_delta,
        )
        if any(value <= 0.0 for value in values):
            raise ValueError("Cartesian follower limits must be positive")
        if waypoint_timeout_seconds < minimum_hold_seconds:
            raise ValueError("waypoint timeout cannot be shorter than minimum hold")
        self.ik_solver = ik_solver
        self.control_dt = float(control_dt)
        self.minimum_hold_seconds = float(minimum_hold_seconds)
        self.waypoint_timeout_seconds = float(waypoint_timeout_seconds)
        self.position_tolerance = float(position_tolerance)
        self.orientation_tolerance = float(orientation_tolerance)
        self.setpoint = UpstreamOscSetpoint(
            control_dt=control_dt,
            measured_lead_clamp=False,
            output_max=(
                max_translation_delta,
                max_translation_delta,
                max_translation_delta,
                max_rotation_delta,
                max_rotation_delta,
                max_rotation_delta,
            ),
        )
        self._waypoints: tuple[CartesianWaypoint, ...] = ()
        self._index = 0
        self._elapsed_steps = 0
        self._completed = False

    @property
    def waypoints(self) -> tuple[CartesianWaypoint, ...]:
        return self._waypoints

    @property
    def waypoint_index(self) -> int:
        return self._index

    @property
    def elapsed_steps(self) -> int:
        return self._elapsed_steps

    def load(
        self,
        waypoints: Sequence[CartesianWaypoint],
        *,
        measured_position: Sequence[float],
        measured_quaternion: Sequence[float],
    ) -> None:
        loaded = tuple(waypoints)
        if not loaded:
            raise ValueError("Cartesian waypoint list cannot be empty")
        self._waypoints = loaded
        self._index = 0
        self._elapsed_steps = 0
        self._completed = False
        reset_ik = getattr(self.ik_solver, "reset", None)
        if callable(reset_ik):
            reset_ik()
        self.setpoint.reset(measured_position, measured_quaternion)

    def step(
        self,
        *,
        measured_position: Sequence[float],
        measured_quaternion: Sequence[float],
    ) -> CartesianWaypointStep:
        if not self._waypoints or self._completed:
            raise RuntimeError("no active Cartesian waypoint list")
        waypoint = self._waypoints[self._index]
        measured_position_array = np.asarray(measured_position, dtype=np.float64)
        measured_quaternion_array = _normalize_quaternion(measured_quaternion)
        goal_position = np.asarray(waypoint.position_xyz, dtype=np.float64)
        goal_quaternion = _normalize_quaternion(waypoint.quaternion_xyzw)
        setpoint_position, setpoint_quaternion = self.setpoint.advance(
            measured_position=measured_position_array,
            measured_quaternion=measured_quaternion_array,
            goal_position=goal_position,
            goal_quaternion=goal_quaternion,
            translation_output_max=waypoint.max_translation_delta,
            rotation_output_max=waypoint.max_rotation_delta,
        )
        joint_targets = self.ik_solver.solve(setpoint_position, setpoint_quaternion)

        self._elapsed_steps += 1
        minimum_seconds = (
            waypoint.minimum_hold_seconds if waypoint.minimum_hold_seconds is not None else self.minimum_hold_seconds
        )
        timeout_seconds = (
            waypoint.timeout_seconds if waypoint.timeout_seconds is not None else self.waypoint_timeout_seconds
        )
        minimum_steps = max(1, round(minimum_seconds / self.control_dt))
        timeout_steps = max(minimum_steps, round(timeout_seconds / self.control_dt))
        converged = (
            float(np.linalg.norm(goal_position - measured_position_array)) <= self.position_tolerance
            and _rotation_distance(goal_quaternion, measured_quaternion_array) <= self.orientation_tolerance
        )
        advanced = self._elapsed_steps >= timeout_steps or (converged and self._elapsed_steps >= minimum_steps)
        completed = False
        elapsed_steps = self._elapsed_steps
        current_index = self._index
        if advanced:
            if self._index + 1 == len(self._waypoints):
                self._completed = True
                completed = True
            else:
                self._index += 1
                self._elapsed_steps = 0
        return CartesianWaypointStep(
            label=waypoint.label,
            joint_targets=joint_targets,
            gripper=waypoint.gripper,
            advanced=advanced,
            completed=completed,
            waypoint_index=current_index,
            elapsed_steps=elapsed_steps,
        )


def articulation_only_model(
    *,
    robot_usd: str,
    root_position: Sequence[float],
    root_quaternion_xyzw: Sequence[float],
):
    """Build a Newton model holding one arm, placed where the scene put it.

    The IK optimises an arm's joints and nothing else, but Newton sizes its
    tiled solver from the whole model it is handed, ignoring the dof mask that
    says so. In a scene that is mostly loose objects -- ten syringes contribute
    seventy coordinates the solver never touches -- the tile outgrows what the
    kernel can launch, and an arm becomes unsolvable because of what is lying
    next to it. Importing the arm alone keeps the tile proportional to the
    problem: 21 coordinates rather than 91 for the eight-object tool-sort
    scenes.

    The import is placed at the articulation's world root, so targets stay in
    world frame and callers need no frame conversion. The quaternion is xyzw,
    as Warp expects; Isaac Lab reports wxyz, so callers must reorder.
    """
    import newton
    import warp as wp
    from newton._src.utils.import_usd import parse_usd

    position = tuple(float(value) for value in root_position)
    quaternion = tuple(float(value) for value in root_quaternion_xyzw)
    if len(position) != 3 or len(quaternion) != 4:
        raise ValueError("articulation root needs a 3-vector and an xyzw quaternion")
    builder = newton.ModelBuilder()
    parse_usd(
        builder,
        str(robot_usd),
        xform=wp.transform(wp.vec3(*position), wp.quat(*quaternion)),
    )
    return builder.finalize()


def resolve_newton_ik_indices(
    *,
    body_labels: Sequence[Any],
    joint_labels: Sequence[Any],
    joint_q_starts: Sequence[int],
    joint_qd_starts: Sequence[int],
    arm_joint_names: Sequence[str],
    end_effector_body_name: str = "robotiq_base",
    articulation_name: str | None = None,
) -> tuple[int, list[int], list[int]]:
    """Resolve one end-effector body and ordered arm coordinate/DOF indices."""

    if len(joint_q_starts) not in {len(joint_labels), len(joint_labels) + 1}:
        raise ValueError("Newton joint labels and coordinate starts disagree")
    if len(joint_qd_starts) not in {len(joint_labels), len(joint_labels) + 1}:
        raise ValueError("Newton joint labels and DOF starts disagree")

    body_matches = [
        index
        for index, label in enumerate(body_labels)
        if PurePosixPath(str(label)).name == end_effector_body_name
        and (articulation_name is None or articulation_name in PurePosixPath(str(label)).parts)
    ]
    if len(body_matches) != 1:
        raise ValueError(f"Newton body {end_effector_body_name!r} is missing or ambiguous")

    coordinates: list[int] = []
    dofs: list[int] = []
    for name in arm_joint_names:
        matches = [
            index
            for index, label in enumerate(joint_labels)
            if PurePosixPath(str(label)).name == name
            and (articulation_name is None or articulation_name in PurePosixPath(str(label)).parts)
        ]
        if len(matches) != 1:
            raise ValueError(f"Newton arm joint {name!r} is missing or ambiguous")
        joint_index = matches[0]
        coordinates.append(int(joint_q_starts[joint_index]))
        dofs.append(int(joint_qd_starts[joint_index]))
    if len(set(coordinates)) != len(coordinates):
        raise ValueError("Newton arm joints do not map to unique coordinates")
    if len(set(dofs)) != len(dofs):
        raise ValueError("Newton arm joints do not map to unique DOFs")
    return body_matches[0], coordinates, dofs


def build_newton_ik_dof_mask(*, joint_dof_count: int, arm_dofs: Sequence[int]) -> np.ndarray:
    """Build the upstream arm-only LM solve mask in Newton DOF space."""

    mask = np.zeros(int(joint_dof_count), dtype=bool)
    indices = np.asarray(arm_dofs, dtype=np.int64)
    if len({int(index) for index in indices}) != len(indices):
        raise ValueError("Newton arm DOF indices must be unique")
    if np.any(indices < 0) or np.any(indices >= joint_dof_count):
        raise ValueError("Newton arm DOF index is outside the model")
    mask[indices] = True
    return mask


class NewtonLmIkSolver:
    """Eight-iteration analytic Newton LM IK matching the upstream routine."""

    def __init__(
        self,
        model: Any,
        *,
        arm_joint_names: Sequence[str],
        iterations: int = 8,
        rotation_weight: float = 0.6,
        end_effector_body_name: str = "robotiq_base",
        articulation_name: str | None = None,
        link_offset_position: Sequence[float] = (0.0, 0.0, 0.0),
        link_offset_quaternion: Sequence[float] = (0.0, 0.0, 0.0, 1.0),
        runtime_model: Any | None = None,
        seed_coordinates: Sequence[int] | None = None,
    ) -> None:
        import newton
        import warp as wp

        if int(model.joint_coord_count) > _LM_COORDINATE_LIMIT:
            raise ValueError(
                f"Newton model has {model.joint_coord_count} coordinates; "
                f"analytic LM is limited to {_LM_COORDINATE_LIMIT}"
            )
        simple_joint_types = {
            int(newton.JointType.REVOLUTE),
            int(newton.JointType.PRISMATIC),
            int(newton.JointType.BALL),
            int(newton.JointType.FREE),
            int(newton.JointType.FIXED),
        }
        exotic = {int(value) for value in model.joint_type.numpy()} - simple_joint_types
        if exotic:
            raise ValueError(f"Newton model has unsupported IK joint types {sorted(exotic)}")
        body_index, arm_coordinates, arm_dofs = resolve_newton_ik_indices(
            body_labels=model.body_label,
            joint_labels=model.joint_label,
            joint_q_starts=model.joint_q_start.numpy().tolist(),
            joint_qd_starts=model.joint_qd_start.numpy().tolist(),
            arm_joint_names=arm_joint_names,
            end_effector_body_name=end_effector_body_name,
            articulation_name=articulation_name,
        )
        if iterations <= 0 or rotation_weight <= 0.0:
            raise ValueError("Newton LM iterations and rotation weight must be positive")

        self.model = model
        self.runtime_model = model if runtime_model is None else runtime_model
        self.arm_coordinates = arm_coordinates
        self.arm_dofs = arm_dofs
        if self.runtime_model is model:
            self.runtime_arm_coordinates = arm_coordinates
        else:
            _, self.runtime_arm_coordinates, _ = resolve_newton_ik_indices(
                body_labels=self.runtime_model.body_label,
                joint_labels=self.runtime_model.joint_label,
                joint_q_starts=self.runtime_model.joint_q_start.numpy().tolist(),
                joint_qd_starts=self.runtime_model.joint_qd_start.numpy().tolist(),
                arm_joint_names=arm_joint_names,
                end_effector_body_name=end_effector_body_name,
                articulation_name=articulation_name,
            )
        self.iterations = int(iterations)
        self._wp = wp
        self._target_position = wp.zeros(1, dtype=wp.vec3, device=model.device)
        self._target_rotation = wp.zeros(1, dtype=wp.vec4, device=model.device)
        offset_position = np.asarray(link_offset_position, dtype=np.float64)
        if offset_position.shape != (3,) or not np.all(np.isfinite(offset_position)):
            raise ValueError("IK link offset position must contain three finite values")
        offset_quaternion = _normalize_quaternion(link_offset_quaternion)
        self._position_objective = newton.ik.IKObjectivePosition(
            link_index=body_index,
            link_offset=wp.vec3(*offset_position),
            target_positions=self._target_position,
        )
        self._rotation_objective = newton.ik.IKObjectiveRotation(
            link_index=body_index,
            link_offset_rotation=wp.quat(*offset_quaternion),
            target_rotations=self._target_rotation,
            weight=float(rotation_weight),
        )
        limit_objective = newton.ik.IKObjectiveJointLimit(
            joint_limit_lower=model.joint_limit_lower,
            joint_limit_upper=model.joint_limit_upper,
        )
        self._solver = newton.ik.IKSolver(
            model=model,
            n_problems=1,
            objectives=[
                self._position_objective,
                self._rotation_objective,
                limit_objective,
            ],
            lambda_initial=0.1,
            jacobian_mode=newton.ik.IKJacobianType.ANALYTIC,
            joint_dof_mask=wp.array(
                build_newton_ik_dof_mask(
                    joint_dof_count=int(model.joint_dof_count),
                    arm_dofs=arm_dofs,
                ),
                dtype=wp.bool,
                device=model.device,
            ),
        )
        self._joint_q = wp.zeros(
            (1, int(model.joint_coord_count)),
            dtype=wp.float32,
            device=model.device,
        )
        wp.copy(
            self._joint_q.reshape((int(model.joint_coord_count),)),
            model.joint_q,
        )

        self._seed_coordinates = None if seed_coordinates is None else list(seed_coordinates)
        self._seeded = False

    @classmethod
    def from_runtime(
        cls,
        *,
        arm_joint_names: Sequence[str],
        end_effector_body_name: str = "robotiq_base",
        articulation_name: str | None = None,
        asset_prim_path: str | None = None,
        link_offset_position: Sequence[float] = (0.0, 0.0, 0.0),
        link_offset_quaternion: Sequence[float] = (0.0, 0.0, 0.0, 1.0),
        robot_usd: str | None = None,
        root_position: Sequence[float] | None = None,
        root_quaternion_xyzw: Sequence[float] | None = None,
    ) -> NewtonLmIkSolver:
        from isaaclab_newton.physics import NewtonManager

        runtime_model = NewtonManager.get_model()
        if runtime_model is None or NewtonManager.get_state_0() is None:
            raise RuntimeError("Newton model and state must exist before constructing IK")
        model = runtime_model
        if asset_prim_path is not None:
            from isaaclab import cloner
            from isaaclab.sim import SimulationContext

            sim = SimulationContext.instance()
            if sim is None:
                raise RuntimeError("Simulation context must exist before constructing IK")
            source_path, _, _asset_suffix = cloner.query.path_to_source(sim.get_clone_plan(), asset_prim_path)
            prototype_builder = NewtonManager._cl_protos.get(source_path)
            if prototype_builder is not None:
                model = prototype_builder.finalize(device=runtime_model.device)
        seed_coordinates: list[int] | None = None
        if model is runtime_model and robot_usd is not None:
            # No prototype to borrow: a scene that does not replicate physics
            # never builds one, which is every tool-sort scene. Import the arm
            # on its own instead, so the solver is still sized by the arm and
            # not by whatever is lying next to it.
            if root_position is None or root_quaternion_xyzw is None:
                raise ValueError("a reduced IK model needs the articulation root pose")
            model = articulation_only_model(
                robot_usd=robot_usd,
                root_position=root_position,
                root_quaternion_xyzw=root_quaternion_xyzw,
            )
            _body, seed_coordinates, _dofs = resolve_newton_ik_indices(
                body_labels=runtime_model.body_label,
                joint_labels=runtime_model.joint_label,
                joint_q_starts=runtime_model.joint_q_start.numpy().tolist(),
                joint_qd_starts=runtime_model.joint_qd_start.numpy().tolist(),
                arm_joint_names=arm_joint_names,
                end_effector_body_name=end_effector_body_name,
                articulation_name=articulation_name,
            )
        return cls(
            model,
            arm_joint_names=arm_joint_names,
            end_effector_body_name=end_effector_body_name,
            # An imported arm carries the USD's own prim names rather than the
            # scene's, so the runtime articulation name matches nothing there --
            # and it holds one arm, so there is nothing to disambiguate.
            articulation_name=(None if seed_coordinates is not None else articulation_name),
            link_offset_position=link_offset_position,
            link_offset_quaternion=link_offset_quaternion,
            runtime_model=runtime_model,
            seed_coordinates=seed_coordinates,
        )

    def reset(self) -> None:
        """Discard the previous commanded solution before a new trajectory."""

        self._seeded = False

    def solve(self, position: np.ndarray, quaternion: np.ndarray):
        from isaaclab_newton.physics import NewtonManager

        state = NewtonManager.get_state_0()
        if state is None or NewtonManager.get_model() is not self.runtime_model:
            raise RuntimeError("Newton runtime changed after IK construction")
        self._target_position.assign(np.asarray(position, dtype=np.float32).reshape(1, 3))
        self._target_rotation.assign(np.asarray(quaternion, dtype=np.float32).reshape(1, 4))
        self._position_objective.set_target_positions(self._target_position)
        self._rotation_objective.set_target_rotations(self._target_rotation)
        if not self._seeded:
            prototype_seed = self._wp.to_torch(self._joint_q)
            runtime_joint_q = self._wp.to_torch(state.joint_q)
            prototype_seed[0, self.arm_coordinates] = runtime_joint_q[self.runtime_arm_coordinates]
            self._seeded = True
        self._solver.step(
            self._joint_q,
            self._joint_q,
            iterations=self.iterations,
        )
        return self._wp.to_torch(self._joint_q)[0, self.arm_coordinates]
