# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""CAP syringe-sorting environment, without policy implementations."""

from .. import register_components

register_components()

from .environments.environment import SyringeSortEnvironment, SyringeSortEnvironmentCfg  # noqa: E402

__all__ = ["SyringeSortEnvironment", "SyringeSortEnvironmentCfg"]
