# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import math
from dataclasses import dataclass


@dataclass
class PhysicsSettleParams:
    """Configuration for the in-sim physics settle check."""

    num_steps: int = 5
    """Number of env steps to advance before reading back object state in the settle check. The settle
    check converts this to ``num_steps * decimation`` physics substeps internally."""

    lin_vel_thresh: float = 0.1
    """Max per-object linear speed (m/s) after settling. Above this the layout is considered unsettled."""

    ang_vel_thresh: float = 0.1
    """Max per-object angular speed (rad/s) after settling. Above this the layout is considered unsettled."""


@dataclass
class PlacementRecordingParams:
    """Physics time, motion limits and minimum yield for settled layout recordings."""

    settle_time_s: float = 4.0
    """Simulated seconds before the quiet window, independent of control decimation."""
    quiet_time_s: float = 1.0
    """Simulated seconds that must remain below the velocity limits."""
    lin_vel_thresh: float = 0.1
    """Maximum root linear speed in metres/second throughout the quiet window."""
    ang_vel_thresh: float = 0.1
    """Maximum root angular speed in radians/second throughout the quiet window."""
    max_translation_m: float = 0.1
    """Maximum ordinary-object displacement from the solved pose; clutter is exempt."""
    max_rotation_deg: float = 15.0
    """Maximum ordinary-object rotation from the solved pose; clutter is exempt."""
    max_facing_error_deg: float = 2.0
    """Maximum final FaceTo heading error, distinct from rotation since release."""
    max_link_translation_m: float = 0.002
    """Maximum unrecorded link displacement relative to its articulation root."""
    max_link_rotation_deg: float = 2.0
    """Maximum unrecorded link rotation relative to its articulation root."""
    penetration_tolerance_m: float = 0.002
    """Allowed contact penetration when checking the final collision geometry."""
    min_layouts: int = 1
    """Minimum accepted layouts; fail without writing when fewer survive filtering."""

    def __post_init__(self) -> None:
        assert self.min_layouts > 0, "min_layouts must be positive"
        for name, value in (("settle_time_s", self.settle_time_s), ("quiet_time_s", self.quiet_time_s)):
            assert math.isfinite(value) and value > 0, f"{name} must be finite and positive"
        for name, value in (
            ("lin_vel_thresh", self.lin_vel_thresh),
            ("ang_vel_thresh", self.ang_vel_thresh),
            ("max_translation_m", self.max_translation_m),
            ("max_rotation_deg", self.max_rotation_deg),
            ("max_facing_error_deg", self.max_facing_error_deg),
            ("max_link_translation_m", self.max_link_translation_m),
            ("max_link_rotation_deg", self.max_link_rotation_deg),
            ("penetration_tolerance_m", self.penetration_tolerance_m),
        ):
            assert math.isfinite(value) and value >= 0, f"{name} must be finite and non-negative"
