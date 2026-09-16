# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for direct episode-results variation replay."""

import json
import torch
from dataclasses import field

import pytest
from isaaclab.managers import EventTermCfg
from isaaclab.utils.configclass import configclass

from isaaclab_arena.variations.episode_conditions import EpisodeConditions, EpisodeConditionScheduler
from isaaclab_arena.variations.uniform_sampler import UniformSamplerCfg
from isaaclab_arena.variations.variation_base import BuildTimeVariationBase, RunTimeVariationBase, VariationBaseCfg


@configclass
class _TestVariationCfg(VariationBaseCfg):
    sampler_cfg: UniformSamplerCfg = field(default_factory=lambda: UniformSamplerCfg(low=[0.0], high=[1.0]))


class _BuildVariation(BuildTimeVariationBase):
    def __init__(self):
        super().__init__(_TestVariationCfg(enabled=True), "build")
        self.applied = None

    def sample(self):
        return self.sampler.sample(1)

    def apply_sample(self, sample):
        self.applied = sample


class _RuntimeVariation(RunTimeVariationBase):
    def __init__(self):
        super().__init__(_TestVariationCfg(enabled=True), "runtime")

    def build_event_cfg(self):
        return "runtime", EventTermCfg(func=lambda env, env_ids: None, mode="reset")


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _variations():
    return {"asset": [_BuildVariation(), _RuntimeVariation()]}


def test_loads_jsonl_and_ignores_relation_placement(tmp_path):
    path = tmp_path / "episode_results.jsonl"
    _write_jsonl(
        path,
        [
            {
                "env_id": 1,
                "episode_in_env": 4,
                "variations": {
                    "asset.build": [0.25],
                    "asset.runtime": [0.5],
                    "scene.relation_placement": {"poses": {}},
                },
            },
            {
                "condition_id": "existing-id",
                "env_id": 0,
                "episode_in_env": 8,
                "variations": {
                    "asset.build": [0.25],
                    "asset.runtime": [0.75],
                },
            },
        ],
    )

    conditions = EpisodeConditions.load(path, _variations())

    assert conditions.build_time_variations == {"asset.build": [0.25]}
    assert [episode.condition_id for episode in conditions.episodes] == ["condition_000000", "existing-id"]
    assert [episode.runtime_variations for episode in conditions.episodes] == [
        {"asset.runtime": [0.5]},
        {"asset.runtime": [0.75]},
    ]


@pytest.mark.parametrize(
    "rows,match",
    [
        ([], "empty"),
        ([{"variations": {"asset.build": [0.25]}}], "missing enabled"),
        (
            [
                {"variations": {"asset.build": [0.25], "asset.runtime": [0.5]}},
                {"variations": {"asset.build": [0.5], "asset.runtime": [0.75]}},
            ],
            "Build-time variation values changed",
        ),
        (
            [{"variations": {"asset.build": [0.25], "asset.runtime": [0.5], "asset.unknown": 1}}],
            "unknown variation",
        ),
    ],
)
def test_rejects_invalid_episode_results(tmp_path, rows, match):
    path = tmp_path / "episode_results.jsonl"
    _write_jsonl(path, rows)
    with pytest.raises(AssertionError, match=match):
        EpisodeConditions.load(path, _variations())


def test_scheduler_assigns_global_fifo_and_parks_exhausted_slots(tmp_path):
    path = tmp_path / "episode_results.jsonl"
    _write_jsonl(
        path,
        [{"variations": {"asset.build": [0.25], "asset.runtime": [value]}} for value in (0.1, 0.2, 0.3)],
    )
    scheduler = EpisodeConditionScheduler(EpisodeConditions.load(path, _variations()))
    scheduler.assign_initial([0, 1])

    values, env_ids = scheduler.samples_for("asset.runtime", torch.tensor([0, 1, 2]))
    assert values == [[0.1], [0.2]]
    assert env_ids.tolist() == [0, 1]

    assert scheduler.complete_and_reassign([1]) == [1]
    values, env_ids = scheduler.samples_for("asset.runtime", torch.tensor([0, 1]))
    assert values == [[0.1], [0.3]]
    assert env_ids.tolist() == [0, 1]

    assert scheduler.complete_and_reassign([0]) == [0]
    assert scheduler.active_provenance(0) == {}
    assert not scheduler.all_done
    assert scheduler.complete_and_reassign([1]) == [1]
    assert scheduler.all_done
    assert scheduler.completed_conditions == 3
