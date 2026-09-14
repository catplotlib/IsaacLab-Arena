# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Managed temporal predicates and predicate composition."""

from __future__ import annotations

import torch
from collections.abc import Sequence

from isaaclab.managers import ManagerTermBase, TerminationTermCfg

from isaaclab_arena.tasks.terminations import SuccessMode, check_success


class ConsecutivePredicate(ManagerTermBase):
    """Base for predicates that must remain true for consecutive evaluations."""

    def __init__(self, cfg: TerminationTermCfg, env):
        super().__init__(cfg, env)
        consecutive_steps = cfg.params["consecutive_steps"]
        assert (
            isinstance(consecutive_steps, int) and not isinstance(consecutive_steps, bool) and consecutive_steps > 0
        ), f"consecutive_steps must be a positive integer, got {consecutive_steps!r}."
        self._required_consecutive_steps = consecutive_steps
        self.consecutive_true_steps = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        """Current consecutive true-result count for each parallel environment."""

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Clear consecutive-result counters for selected environments."""

        ids = slice(None) if env_ids is None else env_ids
        self.consecutive_true_steps[ids] = 0

    def _update_consecutive(self, passed: torch.Tensor) -> torch.Tensor:
        """Update per-environment streaks and return which have reached the required length."""

        passed = torch.as_tensor(passed, dtype=torch.bool, device=self.device).reshape(-1)
        assert passed.shape == (
            self.num_envs,
        ), f"Predicate returned shape {tuple(passed.shape)}; expected ({self.num_envs},)."
        next_count = torch.clamp(self.consecutive_true_steps + 1, max=self._required_consecutive_steps)
        self.consecutive_true_steps = torch.where(passed, next_count, torch.zeros_like(self.consecutive_true_steps))
        return self.consecutive_true_steps >= self._required_consecutive_steps


# TODO(xinjieyao, 2026-09-14): To be removed once progress tracking handles the lifecycle of predicates.
class PredicateGroup(ManagerTermBase):
    """Combine predicates while forwarding resets to managed predicate terms."""

    def __init__(self, cfg: TerminationTermCfg, env):
        super().__init__(cfg, env)
        predicates = cfg.params["predicates"]
        assert predicates, "PredicateGroup requires at least one predicate."

    def __call__(
        self,
        env,
        predicates: list[TerminationTermCfg],
        mode: SuccessMode | str = SuccessMode.ALL,
        k: int | None = None,
    ) -> torch.Tensor:
        return check_success(env, predicates=predicates, mode=mode, k=k)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Reset each unique managed predicate for the selected environments."""

        reset_predicates: list[ManagerTermBase] = []
        for predicate_cfg in self.cfg.params["predicates"]:
            predicate = predicate_cfg.func
            if isinstance(predicate, ManagerTermBase) and not any(
                predicate is existing for existing in reset_predicates
            ):
                predicate.reset(env_ids)
                reset_predicates.append(predicate)
