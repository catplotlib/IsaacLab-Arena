# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from collections.abc import Sequence

from isaaclab.envs import ManagerBasedRLEnv

from isaaclab_arena.environments.arena_world import ArenaWorld
from isaaclab_arena.environments.isaaclab_arena_manager_based_env_cfg import (
    IsaacLabArenaManagerBasedRLEnvCfg,
    apply_arena_global_settings,
)
from isaaclab_arena.metrics.metric_data import MetricsDataCollection
from isaaclab_arena.metrics.metrics_manager import MetricsManager
from isaaclab_arena.recording.episode_recorder_manager import EpisodeRecorderManager
from isaaclab_arena.tasks.predicates.object_settling import ObjectInitialRestPoseRecorder
from isaaclab_arena.variations.episode_conditions import EpisodeConditionScheduler
from isaaclab_arena.variations.variation_recorder import VariationRecorder


class IsaacLabArenaManagerBasedRLEnv(ManagerBasedRLEnv):
    """Arena extension to ManagerBasedRLEnv that adds additional Arena-specific functionality."""

    cfg: IsaacLabArenaManagerBasedRLEnvCfg

    def __init__(
        self,
        cfg: IsaacLabArenaManagerBasedRLEnvCfg,
        render_mode: str | None = None,
        variation_recorder: VariationRecorder | None = None,
        episode_condition_scheduler: EpisodeConditionScheduler | None = None,
        **kwargs,
    ):
        apply_arena_global_settings()
        self._arena_world: ArenaWorld | None = None
        self._object_initial_rest_pose_recorder = ObjectInitialRestPoseRecorder(
            num_envs=cfg.scene.num_envs, device=cfg.sim.device
        )
        self._variation_recorder = variation_recorder
        self._episode_condition_scheduler = episode_condition_scheduler
        if variation_recorder is not None:
            # Bind so run-time variation draws can be attributed to the current episode index.
            variation_recorder.bind_env(self)
        # Per-env count of completed episodes; advanced in ``_reset_idx``.
        self._episode_counts: dict[int, int] = {}
        # The initial reset touches every env before any episode has run; skip it.
        self._first_reset = True
        super().__init__(cfg=cfg, render_mode=render_mode, **kwargs)

    @property
    def arena_world(self) -> ArenaWorld:
        """The environment's live Arena scene queries and cached geometry."""
        assert self._arena_world is not None, "ArenaWorld is unavailable before managers are loaded."
        return self._arena_world

    @property
    def variation_recorder(self) -> VariationRecorder | None:
        """The recorder of variation samples, or ``None`` if the env was not built with one."""
        return self._variation_recorder

    @property
    def episode_condition_scheduler(self) -> EpisodeConditionScheduler | None:
        """The active direct-JSONL replay scheduler, if configured."""
        return self._episode_condition_scheduler

    @property
    def replay_complete(self) -> bool:
        """Whether every configured replay condition has completed."""
        return self._episode_condition_scheduler is not None and self._episode_condition_scheduler.all_done

    @property
    def replay_total_conditions(self) -> int | None:
        """Return the replay condition count, or ``None`` outside replay mode."""
        if self._episode_condition_scheduler is None:
            return None
        return self._episode_condition_scheduler.total_conditions

    def get_recorded_variation_samples(self, variation_key, env_ids):
        """Return replayed values and active env IDs, or signal normal sampling."""
        if self._episode_condition_scheduler is None:
            return None, env_ids
        return self._episode_condition_scheduler.samples_for(variation_key, env_ids)

    def get_active_condition_provenance(self, env_id: int) -> dict:
        """Return replay provenance for the active episode in one slot."""
        if self._episode_condition_scheduler is None:
            return {}
        return self._episode_condition_scheduler.active_provenance(env_id)

    @property
    def object_initial_rest_pose_recorder(self) -> ObjectInitialRestPoseRecorder:
        """The recorder of initial object rest poses. Used when object_settled predicate is enabled by task progress tracking."""
        return self._object_initial_rest_pose_recorder

    @property
    def episode_recorder(self) -> EpisodeRecorderManager:
        """The per-episode recorder."""
        return self.episode_recorder_manager

    def load_managers(self) -> None:
        assert self._arena_world is None, "ArenaWorld is already initialized."
        self._arena_world = ArenaWorld(self.scene)
        super().load_managers()
        if self._episode_condition_scheduler is not None:
            self._install_replay_termination_mask()
        self.metrics_manager = MetricsManager(self.cfg.metrics, self)
        self.episode_recorder_manager = EpisodeRecorderManager(self.cfg.episode_recorders, self)

    def _install_replay_termination_mask(self) -> None:
        """Prevent parked replay slots from creating further recorder/reset cycles."""
        scheduler = self._episode_condition_scheduler
        assert scheduler is not None
        termination_manager = self.termination_manager
        compute_terminations = termination_manager.compute

        def compute_active_terminations():
            compute_terminations()
            if scheduler.parked_env_ids:
                parked = torch.tensor(scheduler.parked_env_ids, device=self.device, dtype=torch.long)
                termination_manager._terminated_buf[parked] = False
                termination_manager._truncated_buf[parked] = False
                termination_manager._term_dones[parked] = False
            return termination_manager.dones

        termination_manager.compute = compute_active_terminations

    def get_language_instruction(self) -> str | None:
        """Return the language instruction that is passed to the policy."""
        return self.cfg.task_description

    def get_episode_index(self, env_id: int) -> int:
        """Return the index of the current episode in ``env_id``."""
        return self._episode_counts.get(env_id, 0)

    def _advance_episode_indices(self, env_ids: Sequence[int]) -> None:
        """Advance the per-env episode counter for each episode in ``env_ids``."""
        for env_id in env_ids:
            env_id = int(env_id)
            self._episode_counts[env_id] = self._episode_counts.get(env_id, 0) + 1

    def _reset_idx(self, env_ids: Sequence[int]) -> None:
        # The initial reset touches every env before any episode has run; nothing to record or count.
        if self._first_reset:
            self._first_reset = False
            if self._episode_condition_scheduler is not None:
                self._episode_condition_scheduler.assign_initial(env_ids)
            super()._reset_idx(env_ids)
            return

        if self._episode_condition_scheduler is not None:
            active_env_ids = [
                int(env_id) for env_id in env_ids if self._episode_condition_scheduler.active_provenance(int(env_id))
            ]
            if active_env_ids:
                self.episode_recorder_manager.record_pre_reset(active_env_ids)
                self._advance_episode_indices(active_env_ids)
                self._episode_condition_scheduler.complete_and_reassign(active_env_ids)
            # Always let Isaac Lab clear/reset terminated slots. Variation terms filter parked
            # slots, so they neither draw nor record another condition.
            super()._reset_idx(env_ids)
            return

        # Runs recorder before super() so the just-finished episode is still intact.
        self.episode_recorder_manager.record_pre_reset(env_ids)
        # Advance before super() so reset-mode variation draws are tagged with the episode they begin.
        self._advance_episode_indices(env_ids)
        super()._reset_idx(env_ids)

    def compute_metrics(self) -> MetricsDataCollection:
        """Compute all registered metrics.

        Returns:
            A MetricsDataCollection instance.
        """
        return self.metrics_manager.compute()
