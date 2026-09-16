# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Import the module's component libraries so their ``@register_*`` decorators fire.

Hooked into ``isaaclab_arena.assets.registries.ensure_assets_registered`` so the
CAP assets, tasks, and policies register alongside the core libraries. Environment
factories register separately via ``isaaclab_arena_cap.environments`` (auto-imported
by ``ensure_environments_registered``).
"""

import isaaclab_arena_cap.assets.object_library  # noqa: F401
import isaaclab_arena_cap.policy.cartesian_waypoint_policy  # noqa: F401
import isaaclab_arena_cap.tasks.objects_centered_in_regions_task  # noqa: F401
