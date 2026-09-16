# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Variation abstract base classes.

A :class:`VariationBase` pairs a target asset with a sampler and a hook that
realises one tweak to the scene. Variations attach to any
:class:`~isaaclab_arena.assets.asset.Asset` and start disabled. Concrete
variations subclass one of two flavors:

* :class:`RunTimeVariationBase` — realised via an event term during simulation
  (e.g. per-reset randomization).
* :class:`BuildTimeVariationBase` — sampled once and applied to asset configs
  before the env cfg is composed (e.g. picking a dome-light HDR).
"""

from __future__ import annotations

import torch
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import field
from typing import Any

from isaaclab.managers import EventTermCfg
from isaaclab.utils.configclass import configclass

from isaaclab_arena.variations.sampler_base import SamplerBase, SamplerBaseCfg


@configclass
class VariationBaseCfg:
    """Base configclass for :class:`VariationBase` instances."""

    enabled: bool = False
    """Whether the variation is applied. Opt in via :meth:`VariationBase.enable` or a cfg override."""

    sampler_cfg: SamplerBaseCfg = field(default_factory=SamplerBaseCfg)
    """Declarative sampler driving this variation. Subclasses set a concrete default."""


class VariationBase(ABC):
    """Variation base class.

    This class only enforces that the variation has a name, a config, a sampler,
    a way to enable and disable it, and a way to apply a new config.

    """

    cfg: VariationBaseCfg
    """The configclass instance holding this variation's tunable parameters."""

    name: str
    """Identifier under which this variation is registered on its asset."""

    def __init__(self, cfg: VariationBaseCfg, name: str):
        self.name = name
        self._qualified_name: str | None = None
        self._sampler: SamplerBase | None = None
        self._sample_listeners: list[Callable[[Any, Any], None]] = []
        self.apply_cfg(cfg)

    @property
    def enabled(self) -> bool:
        """Whether this variation is active and should be built into ``events_cfg``."""
        return self.cfg.enabled

    def enable(self) -> None:
        """Mark this variation as active."""
        self.cfg.enabled = True

    def disable(self) -> None:
        """Mark this variation as inactive."""
        self.cfg.enabled = False

    @property
    def sampler(self) -> SamplerBase | None:
        """The sampler driving this variation, or ``None`` if not yet set."""
        return self._sampler

    @property
    def qualified_name(self) -> str:
        """Return the stable ``<host>.<variation>`` recording key."""
        assert self._qualified_name is not None, f"Variation '{self.name}' is not bound to a host."
        return self._qualified_name

    def bind_host(self, host_name: str) -> None:
        """Bind this variation to its stable recording host."""
        qualified_name = f"{host_name}.{self.name}"
        assert self._qualified_name in (
            None,
            qualified_name,
        ), f"Variation '{self.name}' is already bound as '{self._qualified_name}', not '{qualified_name}'."
        self._qualified_name = qualified_name

    def add_sample_listener(self, listener: Callable[[Any, torch.Tensor | None], None]) -> None:
        """Subscribe ``listener`` (called as ``listener(sample, env_ids)``) to this variation's samples.

        Listeners are stored on the variation, so ``apply_cfg`` re-binds them onto the
        rebuilt sampler and they survive cfg/sampler swaps.
        """
        self._sample_listeners.append(listener)
        if self._sampler is not None:
            self._sampler.add_listener(listener)

    def _notify_recorded_sample(self, sample: Any, env_ids: torch.Tensor | None) -> None:
        """Notify listeners about a replayed sample without drawing from the sampler."""
        for listener in self._sample_listeners:
            listener(sample, env_ids)

    def deserialize_samples(self, values: list[Any]) -> torch.Tensor:
        """Convert recorded numeric values into the sampler's batched tensor shape."""
        assert self.sampler is not None, f"Variation '{self.name}' has no sampler."
        assert hasattr(
            self.sampler, "shape_per_sample"
        ), f"Variation '{self.name}' must override deserialize_samples for sampler type {type(self.sampler).__name__}."
        expected_shape = (len(values), *tuple(self.sampler.shape_per_sample))
        if not values:
            return torch.empty(expected_shape, dtype=torch.float32)
        sample = torch.as_tensor(values, dtype=torch.float32)
        assert tuple(sample.shape) == expected_shape, (
            f"Recorded sample for variation '{self.qualified_name}' has shape {tuple(sample.shape)}; "
            f"expected {expected_shape}."
        )
        return sample

    def _prepare_at_build_time(self) -> None:
        """Configure prerequisites required before environment construction. Default: no-op.

        A run-time variation overrides this when its later event needs a build-time
        precondition (e.g. forcing its camera untiled so per-env edits take effect).
        """

    def _realize_at_build_time(self) -> None:
        """Sample and mutate the bound asset config(s) in place during construction. Default: no-op.

        A build-time variation realises its whole effect here; a run-time variation leaves it a no-op.
        """

    def configure_at_build_time(self) -> None:
        """Run this variation's build-time preparation and realization, once per env build."""
        self._prepare_at_build_time()
        self._realize_at_build_time()

    def apply_cfg(self, cfg: VariationBaseCfg) -> None:
        """Apply new ``cfg``.

        Replaces ``cfg`` and rebuilds ``sampler`` from ``cfg.sampler_cfg``, re-binding any
        variation-owned sample listeners onto the new sampler. Subclasses with extra derived
        state should override and call ``super().apply_cfg(cfg)`` first.

        Args:
            cfg: A cfg of the ``VariationBaseCfg`` subclass this variation accepts.
        """
        self.cfg = cfg
        assert isinstance(
            cfg.sampler_cfg, SamplerBaseCfg
        ), f"cfg.sampler_cfg must be a SamplerBaseCfg; got {type(cfg.sampler_cfg).__name__}."
        self._sampler = cfg.sampler_cfg.build()
        # Re-bind variation-owned listeners so a cfg/sampler swap doesn't drop subscriptions.
        for listener in self._sample_listeners:
            self._sampler.add_listener(listener)


class RunTimeVariationBase(VariationBase):
    """Variation realised at run time via an ``EventTermCfg``.

    Use when the underlying property can be flipped during simulation (e.g.
    visual color, initial pose, mass).
    """

    @abstractmethod
    def build_event_cfg(self) -> tuple[str, EventTermCfg]:
        """Return the ``(name, cfg)`` event term that realises this variation."""
        ...

    def resolve_samples(
        self,
        env: Any,
        env_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return sampled or replayed values and the env IDs they apply to."""
        replay = getattr(env, "get_recorded_variation_samples", None)
        if replay is None:
            assert self.sampler is not None, f"Variation '{self.qualified_name}' has no sampler."
            return self.sampler.sample(num_samples=len(env_ids), env_ids=env_ids), env_ids

        values, replay_env_ids = replay(self.qualified_name, env_ids)
        if values is None:
            assert self.sampler is not None, f"Variation '{self.qualified_name}' has no sampler."
            return self.sampler.sample(num_samples=len(env_ids), env_ids=env_ids), env_ids

        sample = self.deserialize_samples(values)
        self._notify_recorded_sample(sample, replay_env_ids)
        return sample, replay_env_ids


class BuildTimeVariationBase(VariationBase):
    """Variation sampled once and applied before the env is built.

    Use for properties that can't change in-flight: HDR maps, USD swaps,
    spawner params baked into a config. Subclasses hold references to the
    asset(s) they mutate and realise the effect in ``_realize_at_build_time``.
    """

    def sample(self) -> Any:
        """Draw one build-time sample."""
        raise NotImplementedError(f"{type(self).__name__}.sample() is not implemented.")

    def apply_sample(self, sample: Any) -> None:
        """Apply one batched build-time sample to the target configuration."""
        raise NotImplementedError(f"{type(self).__name__}.apply_sample() is not implemented.")

    def _realize_at_build_time(self) -> None:
        """Draw and apply one build-time sample."""
        self.apply_sample(self.sample())

    def apply_recorded_sample(self, value: Any) -> None:
        """Deserialize and apply one recorded build-time sample."""
        sample = self.deserialize_samples([value])
        self._notify_recorded_sample(sample, None)
        self.apply_sample(sample)
