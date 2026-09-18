# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Reusable receiver-frame success definition for USB-C insertion."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import MISSING
from typing import Any

import isaaclab.envs.mdp as mdp
from isaaclab.managers import TerminationTermCfg
from isaaclab.utils.configclass import configclass

from isaaclab_arena.assets.asset import Asset
from isaaclab_arena.metrics.metric_base import MetricBase
from isaaclab_arena.metrics.success_rate import SuccessRateMetric
from isaaclab_arena.tasks.predicates.composite import CompositePredicate
from isaaclab_arena.tasks.predicates.gripper import parallel_jaw_gripper_released
from isaaclab_arena.tasks.predicates.spatial import (
    depth_in_range,
    end_effector_distance_from_object_exceeds_threshold,
    tilt_axis_aligned,
    velocity_below_threshold,
    xy_in_proximity,
)
from isaaclab_arena.tasks.task_base import TaskBase
from isaaclab_arena.tasks.terminations import SuccessMode


@configclass
class TerminationsCfg:
    """Timeout and seated-connector success terms."""

    time_out: TerminationTermCfg = TerminationTermCfg(func=mdp.time_out, time_out=True)
    success: TerminationTermCfg = MISSING


class UsbcInsertionTask(TaskBase):
    """Require a seated, slow plug with optional release and hand withdrawal."""

    def __init__(
        self,
        plug: Asset,
        receiver: Asset,
        *,
        receiver_mouth_offset_xyz: tuple[float, float, float],
        receiver_axis: tuple[float, float, float],
        subject_tip_offset_xyz: tuple[float, float, float],
        depth_min: float,
        lateral_max: float,
        speed_max: float,
        depth_max: float | None = None,
        subject_axis: tuple[float, float, float] = (0.0, 0.0, 1.0),
        tilt_max: float | None = None,
        allow_antiparallel_axes: bool = False,
        work_hand: Mapping[str, Any] | None = None,
        withdrawal_distance_min: float | None = None,
        require_released: bool = False,
        consecutive_success_steps: int = 1,
        episode_length_s: float = 120.0,
        task_description: str | None = None,
    ) -> None:
        """Configure geometry and thresholds supplied by one connector variant."""
        for name, vector in (
            ("receiver_mouth_offset_xyz", receiver_mouth_offset_xyz),
            ("receiver_axis", receiver_axis),
            ("subject_tip_offset_xyz", subject_tip_offset_xyz),
            ("subject_axis", subject_axis),
        ):
            assert len(vector) == 3 and all(
                math.isfinite(value) for value in vector
            ), f"{name} must contain three finite values."
        assert math.dist(receiver_axis, (0.0, 0.0, 0.0)) > 0.0, "receiver_axis must be non-zero."
        assert math.dist(subject_axis, (0.0, 0.0, 0.0)) > 0.0, "subject_axis must be non-zero."
        assert depth_min > 0.0 and math.isfinite(depth_min), "depth_min must be positive and finite."
        assert depth_max is None or depth_max >= depth_min, "depth_max must not be less than depth_min."
        assert lateral_max > 0.0 and math.isfinite(lateral_max), "lateral_max must be positive and finite."
        assert speed_max > 0.0 and math.isfinite(speed_max), "speed_max must be positive and finite."
        assert not require_released or work_hand is not None, "Release checking requires work_hand."
        assert (work_hand is None) == (
            withdrawal_distance_min is None
        ), "work_hand and withdrawal_distance_min must be set together."
        if work_hand is not None:
            assert (
                math.isfinite(withdrawal_distance_min) and withdrawal_distance_min > 0.0
            ), "Withdrawal distance must be positive and finite."
            for name in ("robot_name", "gripper_joint_name", "ee_frame_name"):
                assert isinstance(work_hand[name], str) and work_hand[name], f"Invalid work_hand {name}."
            target_frame_name = work_hand.get("target_frame_name")
            assert target_frame_name is None or (
                isinstance(target_frame_name, str) and target_frame_name
            ), "Invalid work_hand target_frame_name."
            assert math.isfinite(work_hand["jaw_gap_at_zero_joint_m"]), "Invalid work_hand jaw gap."
            assert (
                math.isfinite(work_hand["grasp_width_m"]) and work_hand["grasp_width_m"] > 0.0
            ), "Invalid work_hand grasp width."
            assert (
                math.isfinite(work_hand.get("release_clearance_m", 1.5e-3))
                and work_hand.get("release_clearance_m", 1.5e-3) >= 0.0
            ), "Invalid work_hand release clearance."
        assert tilt_max is None or 0.0 <= tilt_max <= math.pi, "tilt_max must be in [0, pi]."
        assert isinstance(consecutive_success_steps, int) and not isinstance(
            consecutive_success_steps, bool
        ), "consecutive_success_steps must be an integer."
        assert consecutive_success_steps > 0, "consecutive_success_steps must be positive."
        assert episode_length_s > 0.0 and math.isfinite(
            episode_length_s
        ), "episode_length_s must be positive and finite."

        super().__init__(
            episode_length_s=episode_length_s,
            task_description=task_description or "Insert the USB-C plug into the receiver.",
        )
        self.plug = plug
        self.receiver = receiver

        mating_params = {
            "subject_name": plug.name,
            "receiver_name": receiver.name,
            "subject_offset_xyz": tuple(subject_tip_offset_xyz),
            "target_offset_xyz": tuple(receiver_mouth_offset_xyz),
            "receiver_axis": tuple(receiver_axis),
        }
        predicates = [
            TerminationTermCfg(
                func=depth_in_range,
                params={
                    **mating_params,
                    "depth_min": depth_min,
                    "depth_max": depth_max,
                },
            ),
            TerminationTermCfg(
                func=xy_in_proximity,
                params={**mating_params, "tolerance_xy": lateral_max},
            ),
        ]
        if tilt_max is not None:
            predicates.append(
                TerminationTermCfg(
                    func=tilt_axis_aligned,
                    params={
                        "subject_name": plug.name,
                        "receiver_name": receiver.name,
                        "subject_axis": tuple(subject_axis),
                        "receiver_axis": tuple(receiver_axis),
                        "max_tilt_rad": tilt_max,
                        "allow_antiparallel": allow_antiparallel_axes,
                    },
                )
            )
        predicates.append(
            TerminationTermCfg(
                func=velocity_below_threshold,
                params={
                    "subject_name": plug.name,
                    "linear_velocity_threshold": speed_max,
                },
            )
        )
        if work_hand is not None:
            if require_released:
                predicates.append(
                    TerminationTermCfg(
                        func=parallel_jaw_gripper_released,
                        params={
                            "robot_name": work_hand["robot_name"],
                            "gripper_joint_name": work_hand["gripper_joint_name"],
                            "jaw_gap_at_zero_joint_m": work_hand["jaw_gap_at_zero_joint_m"],
                            "grasp_width_m": work_hand["grasp_width_m"],
                            "release_clearance_m": work_hand.get("release_clearance_m", 1.5e-3),
                        },
                    )
                )
            predicates.append(
                TerminationTermCfg(
                    func=end_effector_distance_from_object_exceeds_threshold,
                    params={
                        "subject_name": plug.name,
                        "ee_frame_name": work_hand["ee_frame_name"],
                        "target_frame_name": work_hand.get("target_frame_name"),
                        "distance_threshold_m": withdrawal_distance_min,
                    },
                )
            )
        self.termination_cfg = TerminationsCfg(
            success=TerminationTermCfg(
                func=CompositePredicate,
                params={
                    "predicates": predicates,
                    "mode": SuccessMode.ALL,
                    "consecutive_steps": consecutive_success_steps,
                },
            )
        )

    def get_scene_cfg(self) -> Any:
        return None

    def get_termination_cfg(self) -> Any:
        return self.termination_cfg

    def get_events_cfg(self) -> Any:
        return None

    def get_mimic_env_cfg(self, arm_mode) -> Any:
        return None

    def get_metrics(self) -> list[MetricBase]:
        return [SuccessRateMetric()]
