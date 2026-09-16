# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Variation that removes an object from the scene with a given probability.

Typical use is thinning out distractor clutter, so a policy sees a different subset of
non-task objects from run to run.

The draw happens once at build time, so an object is either present or gone for the whole run.
Realizing it needs a reset event rather than a spawn pose: relation placement rewrites every
non-anchor object's pose on reset, and per-object pose events restore their own. Variation events
are composed after both, so the teleport is what survives.
"""

from __future__ import annotations

from dataclasses import field
from typing import TYPE_CHECKING

from isaaclab.managers import EventTermCfg, SceneEntityCfg
from isaaclab.utils.configclass import configclass

from isaaclab_arena.terms.events import set_object_pose
from isaaclab_arena.utils.pose import Pose
from isaaclab_arena.variations.bernoulli_sampler import BernoulliSamplerCfg
from isaaclab_arena.variations.variation_base import RunTimeVariationBase, VariationBaseCfg

if TYPE_CHECKING:
    import torch

    from isaaclab.envs import ManagerBasedEnv


@configclass
class ObjectDisappearVariationCfg(VariationBaseCfg):
    """Configuration for :class:`ObjectDisappearVariation`."""

    away_position_xyz: tuple[float, float, float] = (0.0, 0.0, -10.0)
    """Env-local position to hold a disappeared object at, far enough out to never be seen or touched.

    The object free-falls from here for the rest of the episode; every reset puts it back.
    """

    sampler_cfg: BernoulliSamplerCfg = field(default_factory=BernoulliSamplerCfg)
    """Probability that the object disappears."""


def hold_object_away(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    pose: Pose,
    disappeared: bool,
) -> None:
    """Reset event that pins a disappeared object to ``pose``, overriding earlier placement writes."""
    if not disappeared:
        return
    set_object_pose(env, env_ids, asset_cfg=asset_cfg, pose=pose)


class ObjectDisappearVariation(RunTimeVariationBase):
    """Remove an object from the scene with a build-time sampled probability.

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
        self._disappeared = False

    def _prepare_at_build_time(self) -> None:
        """Draw once, so the reset event below replays one decision for the whole run."""
        assert self.sampler is not None, "ObjectDisappearVariation: sampler not set."
        self._disappeared = self.sampler.sample(num_samples=1)[0]

    def build_event_cfg(self) -> tuple[str, EventTermCfg]:
        return (
            f"{self.asset_name}_{self.name}",
            EventTermCfg(
                func=hold_object_away,
                mode="reset",
                params={
                    "asset_cfg": SceneEntityCfg(self.asset_name),
                    # Re-tupled because Hydra overrides arrive as lists.
                    "pose": Pose(position_xyz=tuple(self.cfg.away_position_xyz)),
                    "disappeared": self._disappeared,
                },
            ),
        )
