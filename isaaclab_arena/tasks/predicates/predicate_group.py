# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Combine predicate results while managing child predicate lifecycles."""

from __future__ import annotations

import torch
from collections.abc import Sequence

from isaaclab.managers import ManagerTermBase, TerminationTermCfg

from isaaclab_arena.tasks.predicates.consecutive import ConsecutivePredicate
from isaaclab_arena.tasks.terminations import SuccessMode, combine_success_results


# TODO(xinjieyao, 2026-09-14): To be removed once progress tracking handles the lifecycle of predicates.
# NOTE(xinjieyao, 2026-09-14): Progress tracking does not support PredicateGroup because it does not
# propagate per-environment active masks to nested stateful predicates.
class PredicateGroup(ConsecutivePredicate):
    """Combine child results, optionally requiring consecutive successful evaluations."""

    def __init__(self, cfg: TerminationTermCfg, env):
        cfg.params.setdefault("consecutive_steps", 1)
        super().__init__(cfg, env)
        self.predicates = cfg.params["predicates"]
        assert self.predicates, "PredicateGroup requires at least one predicate."
        self.results = torch.zeros(
            (len(self.predicates), env.num_envs),
            dtype=torch.bool,
            device=env.device,
        )

    def __call__(
        self,
        env,
        predicates: list[TerminationTermCfg],
        mode: SuccessMode | str = SuccessMode.ALL,
        k: int | None = None,
        consecutive_steps: int = 1,
        active_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # These arguments mirror TerminationTermCfg.params for manager signature validation.
        del predicates, consecutive_steps
        self.results = torch.stack(
            [predicate.func(env, **predicate.params) for predicate in self.predicates],
            dim=0,
        )
        passed = combine_success_results(self.results, mode=mode, k=k)
        return self._update_consecutive_and_get_completion_mask(passed, active_mask=active_mask)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Reset managed children and cached results for selected environments."""
        super().reset(env_ids)
        if env_ids is None:
            env_ids = slice(None)
        self.results[:, env_ids] = False
        reset_predicates: list[ManagerTermBase] = []
        for predicate_cfg in self.predicates:
            predicate = predicate_cfg.func
            if isinstance(predicate, ManagerTermBase) and not any(
                predicate is existing for existing in reset_predicates
            ):
                predicate.reset(env_ids)
                reset_predicates.append(predicate)
