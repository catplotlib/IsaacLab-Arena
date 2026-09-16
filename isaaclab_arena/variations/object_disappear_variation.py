# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Variation that removes an object from the scene with a given probability.

Typical use is thinning out distractor clutter, so a policy sees a different subset of
non-task objects from episode to episode.

Each resetting environment draws independently, so parallel envs disagree and an object that was
gone last episode can be back this one. Coming back relies on something else restoring the pose:
relation placement, or the object's own pose reset event. An object with neither stays away once
it has drawn "gone".

Parking is realized in a reset event rather than a spawn pose because relation placement rewrites
every non-anchor object's pose on reset, and per-object pose events restore their own. Variation
events are composed after both, so the park is what survives.
"""

from __future__ import annotations

import torch
from dataclasses import field
from typing import TYPE_CHECKING

from isaaclab.managers import EventTermCfg, SceneEntityCfg
from isaaclab.utils.configclass import configclass

from isaaclab_arena.terms.events import set_object_pose
from isaaclab_arena.utils.pose import Pose
from isaaclab_arena.variations.bernoulli_sampler import BernoulliSampler, BernoulliSamplerCfg
from isaaclab_arena.variations.variation_base import RunTimeVariationBase, VariationBaseCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

# Fraction of env_spacing to park at. Environment origins are laid out one env_spacing apart, so
# staying under half keeps the object inside its own cell.
_PARK_CELL_FRACTION = 0.45


@configclass
class ObjectDisappearVariationCfg(VariationBaseCfg):
    """Configuration for :class:`ObjectDisappearVariation`."""

    away_position_xyz: tuple[float, float, float] | None = None
    """Env-local position to park a disappeared object at, or ``None`` to derive one.

    The derived position sits near the edge of the environment's own cell, which is as far from the
    workspace as an object can go without straying into a neighbour's cameras. Override it only if
    the scene has somewhere better, and keep any override above the ground collider: parking below
    the ground makes the solver depenetrate the object, launching it back up into the scene.
    """

    sampler_cfg: BernoulliSamplerCfg = field(default_factory=BernoulliSamplerCfg)
    """Probability that the object disappears, drawn per environment on every reset."""


def get_away_pose(env: ManagerBasedEnv, away_position_xyz: tuple[float, float, float] | None) -> Pose:
    """Return the pose to park a disappeared object at, deriving one when not configured."""
    if away_position_xyz is None:
        return Pose(position_xyz=(0.0, _PARK_CELL_FRACTION * env.scene.cfg.env_spacing, 0.0))
    # Re-tupled because Hydra overrides arrive as lists.
    return Pose(position_xyz=tuple(away_position_xyz))


def hold_object_away(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    away_position_xyz: tuple[float, float, float] | None,
    sampler: BernoulliSampler,
) -> None:
    """Reset event that parks the resetting envs which drew "gone" out of the way."""
    if env_ids is None or len(env_ids) == 0:
        return
    env_ids = torch.as_tensor(env_ids, device=env.device).reshape(-1)
    disappeared = torch.as_tensor(sampler.sample(num_samples=len(env_ids), env_ids=env_ids), device=env.device)
    away_env_ids = env_ids[disappeared]
    if len(away_env_ids) > 0:
        set_object_pose(env, away_env_ids, asset_cfg=asset_cfg, pose=get_away_pose(env, away_position_xyz))


class ObjectDisappearVariation(RunTimeVariationBase):
    """Remove an object from the scene with a per-env, per-reset probability.

    Args:
        asset_name: Scene-entity name of the target object. Holding the name rather than the object
            keeps the asset's variation list free of a back-reference, which ``cfg.validate()``
            would otherwise follow in circles.
        cfg: Tunable parameters. Defaults to a 50% chance of disappearing.
        name: Identifier under which this variation is registered on the asset.
            Defaults to ``"disappear"``.
    """

    cfg: ObjectDisappearVariationCfg

    def __init__(
        self,
        asset_name: str,
        cfg: ObjectDisappearVariationCfg | None = None,
        name: str = "disappear",
    ):
        super().__init__(cfg=cfg if cfg is not None else ObjectDisappearVariationCfg(), name=name)
        self.asset_name = asset_name

    def build_event_cfg(self) -> tuple[str, EventTermCfg]:
        assert self._sampler is not None, f"ObjectDisappearVariation on '{self.asset_name}': sampler not set."
        return (
            f"{self.asset_name}_{self.name}",
            EventTermCfg(
                func=hold_object_away,
                mode="reset",
                params={
                    "asset_cfg": SceneEntityCfg(self.asset_name),
                    "away_position_xyz": self.cfg.away_position_xyz,
                    "sampler": self._sampler,
                },
            ),
        )
