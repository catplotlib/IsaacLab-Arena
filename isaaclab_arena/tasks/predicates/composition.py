# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Managed composition for stateless and stateful task predicates."""

from __future__ import annotations

import torch
from collections.abc import Sequence

from isaaclab.managers import ManagerTermBase, TerminationTermCfg

from isaaclab_arena.tasks.terminations import SuccessMode, combine_success_results


class ConsecutivePredicate(ManagerTermBase):
    """Pass after one child predicate remains true for a fixed number of steps."""

    def __init__(self, cfg: TerminationTermCfg, env):
        super().__init__(cfg, env)
        self.predicate = cfg.params["predicate"]
        steps = cfg.params["steps"]
        assert (
            isinstance(steps, int) and not isinstance(steps, bool) and steps > 0
        ), f"ConsecutivePredicate steps must be a positive integer, got {steps!r}."
        self.count = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

    def __call__(
        self,
        env,
        predicate: TerminationTermCfg,
        steps: int,
    ) -> torch.Tensor:
        # These arguments mirror TerminationTermCfg.params for manager signature validation;
        # the manager-resolved child config is retained on this instance.
        del predicate
        passed = self.predicate.func(env, **self.predicate.params)
        self.count = torch.where(passed, self.count + 1, torch.zeros_like(self.count))
        return self.count >= steps

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Clear counters and reset a managed child for the selected environments."""
        if env_ids is None:
            env_ids = slice(None)
        self.count[env_ids] = 0
        if isinstance(self.predicate.func, ManagerTermBase):
            self.predicate.func.reset(env_ids)


class PredicateGroup(ManagerTermBase):
    """Manage child predicate lifecycles and combine their current results."""

    def __init__(self, cfg: TerminationTermCfg, env):
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
    ) -> torch.Tensor:
        # This argument mirrors TerminationTermCfg.params for manager signature validation.
        del predicates
        self.results = torch.stack(
            [predicate.func(env, **predicate.params) for predicate in self.predicates],
            dim=0,
        )
        return combine_success_results(self.results, mode=mode, k=k)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Reset managed children and cached results for selected environments."""
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
