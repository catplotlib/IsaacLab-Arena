# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Load and schedule variation conditions directly from episode-results JSONL."""

from __future__ import annotations

import json
import torch
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from isaaclab_arena.variations.variation_base import BuildTimeVariationBase, RunTimeVariationBase, VariationBase

_PLACEMENT_KEY = "scene.relation_placement"


@dataclass(frozen=True)
class RecordedEpisodeCondition:
    """One ordered run-time variation condition from an episode-results row."""

    condition_id: str
    runtime_variations: dict[str, Any]
    source_episode: dict[str, Any]


@dataclass(frozen=True)
class EpisodeConditions:
    """Validated build-time and ordered run-time samples loaded from JSONL."""

    source_path: str
    build_time_variations: dict[str, Any]
    episodes: tuple[RecordedEpisodeCondition, ...]

    @classmethod
    def load(
        cls,
        path: str | Path,
        variations: dict[str, list[VariationBase]],
    ) -> EpisodeConditions:
        """Load replay conditions and validate them against enabled variations."""
        path = Path(path).expanduser().resolve()
        assert path.is_file(), f"Episode conditions JSONL does not exist: {path}"

        variations_by_key: dict[str, VariationBase] = {}
        for host_name, host_variations in variations.items():
            for variation in host_variations:
                if not variation.enabled:
                    continue
                variation.bind_host(host_name)
                variations_by_key[variation.qualified_name] = variation

        rows: list[dict[str, Any]] = []
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as error:
                    raise AssertionError(f"Invalid JSON on line {line_number} of {path}: {error.msg}") from error
                assert isinstance(row, dict), f"Episode conditions line {line_number} must contain a JSON object."
                assert isinstance(
                    row.get("variations", {}), dict
                ), f"Episode conditions line {line_number} field 'variations' must be an object."
                rows.append(row)
        assert rows, f"Episode conditions JSONL is empty: {path}"

        expected_keys = set(variations_by_key)
        for line_number, row in enumerate(rows, start=1):
            recorded_keys = set(row.get("variations", {}))
            unknown_keys = recorded_keys - expected_keys - {_PLACEMENT_KEY}
            assert (
                not unknown_keys
            ), f"Episode conditions line {line_number} contains unknown variation keys: {sorted(unknown_keys)}."
            missing_keys = expected_keys - recorded_keys
            assert (
                not missing_keys
            ), f"Episode conditions line {line_number} is missing enabled variation keys: {sorted(missing_keys)}."

        build_time_keys = {
            key for key, variation in variations_by_key.items() if isinstance(variation, BuildTimeVariationBase)
        }
        runtime_keys = {
            key for key, variation in variations_by_key.items() if isinstance(variation, RunTimeVariationBase)
        }
        assert build_time_keys | runtime_keys == expected_keys

        first_variations = rows[0].get("variations", {})
        build_time_variations = {key: first_variations[key] for key in build_time_keys}
        for line_number, row in enumerate(rows[1:], start=2):
            row_variations = row.get("variations", {})
            changed = [key for key, value in build_time_variations.items() if row_variations[key] != value]
            assert not changed, (
                f"Build-time variation values changed within {path} on line {line_number}: {sorted(changed)}. "
                "Replay requires one effective environment build per JSONL file."
            )

        episodes = []
        condition_ids: set[str] = set()
        for index, row in enumerate(rows):
            condition_id = str(row.get("condition_id", f"condition_{index:06d}"))
            assert condition_id not in condition_ids, f"Duplicate replay condition_id '{condition_id}' in {path}."
            condition_ids.add(condition_id)
            source_episode = {key: row[key] for key in ("job_name", "env_id", "episode_in_env", "seed") if key in row}
            episodes.append(
                RecordedEpisodeCondition(
                    condition_id=condition_id,
                    runtime_variations={key: row["variations"][key] for key in runtime_keys},
                    source_episode=source_episode,
                )
            )

        return cls(
            source_path=str(path),
            build_time_variations=build_time_variations,
            episodes=tuple(episodes),
        )


class EpisodeConditionScheduler:
    """Assign recorded episode conditions globally in JSONL line order."""

    def __init__(self, conditions: EpisodeConditions) -> None:
        self.conditions = conditions
        self._next_index = 0
        self._active_by_env: dict[int, RecordedEpisodeCondition] = {}
        self._completed_condition_ids: list[str] = []
        self._parked_env_ids: set[int] = set()

    @property
    def total_conditions(self) -> int:
        """Return the authoritative replay episode count."""
        return len(self.conditions.episodes)

    @property
    def completed_conditions(self) -> int:
        """Return the number of replay conditions completed so far."""
        return len(self._completed_condition_ids)

    @property
    def all_done(self) -> bool:
        """Whether every recorded condition has completed."""
        return self.completed_conditions == self.total_conditions

    @property
    def parked_env_ids(self) -> tuple[int, ...]:
        """Return slots that must no longer terminate or consume samples."""
        return tuple(sorted(self._parked_env_ids))

    def assign_initial(self, env_ids: Sequence[int]) -> None:
        """Assign the first conditions to slots during the initial reset."""
        assert not self._active_by_env and self._next_index == 0, "Replay conditions were already assigned."
        for env_id in env_ids:
            self._assign_next(int(env_id))

    def complete_and_reassign(self, env_ids: Sequence[int]) -> list[int]:
        """Complete active conditions and assign the next FIFO entries."""
        completed_env_ids = []
        for env_id_value in env_ids:
            env_id = int(env_id_value)
            condition = self._active_by_env.pop(env_id, None)
            if condition is None:
                continue
            self._completed_condition_ids.append(condition.condition_id)
            completed_env_ids.append(env_id)
            self._assign_next(env_id)
        return completed_env_ids

    def _assign_next(self, env_id: int) -> None:
        """Assign the next condition to one slot, or park it."""
        if self._next_index >= self.total_conditions:
            self._parked_env_ids.add(env_id)
            return
        condition = self.conditions.episodes[self._next_index]
        self._next_index += 1
        self._active_by_env[env_id] = condition
        self._parked_env_ids.discard(env_id)

    def samples_for(
        self,
        variation_key: str,
        env_ids: torch.Tensor,
    ) -> tuple[list[Any], torch.Tensor]:
        """Return replay samples for active slots in ``env_ids``."""
        values = []
        selected_env_ids = []
        for env_id_value in env_ids.tolist():
            env_id = int(env_id_value)
            condition = self._active_by_env.get(env_id)
            if condition is None:
                continue
            assert (
                variation_key in condition.runtime_variations
            ), f"Replay condition '{condition.condition_id}' has no run-time sample for '{variation_key}'."
            values.append(condition.runtime_variations[variation_key])
            selected_env_ids.append(env_id)
        selected = torch.as_tensor(selected_env_ids, device=env_ids.device, dtype=env_ids.dtype)
        return values, selected

    def active_provenance(self, env_id: int) -> dict[str, Any]:
        """Return provenance for the condition currently assigned to ``env_id``."""
        condition = self._active_by_env.get(int(env_id))
        if condition is None:
            return {}
        return {
            "condition_id": condition.condition_id,
            "replay_source": self.conditions.source_path,
            "source_episode": condition.source_episode,
        }
