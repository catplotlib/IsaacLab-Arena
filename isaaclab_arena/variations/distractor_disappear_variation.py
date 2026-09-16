# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Build-time variation that makes a distractor object disappear with some probability.

Sampled once before env-cfg composition: with probability ``cfg.sampler_cfg.probability`` the
object's initial pose is overridden to ``cfg.away_position_xyz``, a position far outside the
scene, so it never renders or interacts near the task. Otherwise the object's existing initial
pose is left untouched.
"""

from __future__ import annotations

from dataclasses import field
from typing import TYPE_CHECKING

from isaaclab.utils.configclass import configclass

from isaaclab_arena.utils.pose import Pose
from isaaclab_arena.variations.bernoulli_sampler import BernoulliSamplerCfg
from isaaclab_arena.variations.variation_base import BuildTimeVariationBase, VariationBaseCfg

if TYPE_CHECKING:
    from isaaclab_arena.assets.object_base import ObjectBase


@configclass
class DistractorDisappearVariationCfg(VariationBaseCfg):
    """Configuration for :class:`DistractorDisappearVariation`."""

    away_position_xyz: tuple[float, float, float] = (0.0, 0.0, -10.0)
    """World position the object is teleported to when it disappears."""

    sampler_cfg: BernoulliSamplerCfg = field(default_factory=lambda: BernoulliSamplerCfg(probability=0.5))
    """Probability that the object disappears."""


class DistractorDisappearVariation(BuildTimeVariationBase):
    """Make a distractor object disappear (teleport far outside the scene) at build time.

    Args:
        obj: The distractor object to mutate. A reference is captured; ``apply`` mutates it.
        cfg: Tunable parameters. Defaults to a 50% disappear probability.
        name: Identifier under which this variation is registered on the asset.
            Defaults to ``"distractor_disappear"``.
    """

    cfg: DistractorDisappearVariationCfg

    def __init__(
        self,
        obj: ObjectBase,
        cfg: DistractorDisappearVariationCfg | None = None,
        name: str = "distractor_disappear",
    ):
        super().__init__(cfg=cfg if cfg is not None else DistractorDisappearVariationCfg(), name=name)
        self._object = obj

    def _realize_at_build_time(self) -> None:
        assert self.sampler is not None, "DistractorDisappearVariation: sampler not set."
        disappears = self.sampler.sample(num_samples=1)[0]
        if disappears:
            self._object.set_initial_pose(Pose(position_xyz=self.cfg.away_position_xyz))
