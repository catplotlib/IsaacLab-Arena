# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from isaaclab_arena.tasks.composite_task_base import CompositeTaskBase


class SequentialTaskBase(CompositeTaskBase):
    """Combine a flat list of tasks that ProgressTracker advances in order.

    The next subtask starts on the following control step. Task success is
    reported on the same step that its completion and final-state requirements are met.
    """

    subtasks_are_sequential: bool = True
