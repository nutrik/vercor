"""Carry composed marine albedo through native JCM forcing selection.

The wrapper owns no model state and preserves native ForcingData copy/select
behavior. SPEEDY's forcing term still diagnoses land albedo and all other
boundary conditions. Only marine and combined albedo are replaced; native
diagnostic schemas stay unchanged so integration carries remain stable PyTrees.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
from jcm.date import DEFAULT_CALENDAR
from jcm.forcing import ForcingData
from jcm.physics.speedy.speedy_terms import SpeedyForcing


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class CoupledJCMForcing:
    """Native forcing plus instantaneous marine albedo in native (x, y) layout."""

    base: ForcingData
    marine_surface_albedo: jax.Array

    def __getattr__(self, name: str) -> Any:
        """Delegate native attributes without recursing during reconstruction."""
        return getattr(object.__getattribute__(self, "base"), name)

    def copy(
        self, *, marine_surface_albedo: jax.Array | None = None, **native_updates: Any
    ) -> CoupledJCMForcing:
        """Replace native fields while preserving the marine extension."""
        return CoupledJCMForcing(
            self.base.copy(**native_updates),
            (
                self.marine_surface_albedo
                if marine_surface_albedo is None
                else marine_surface_albedo
            ),
        )

    def select(self, date: Any, calendar: str = DEFAULT_CALENDAR) -> CoupledJCMForcing:
        """Select native time series/solar geometry and retain current albedo."""
        return CoupledJCMForcing(
            self.base.select(date, calendar=calendar), self.marine_surface_albedo
        )


class CoupledSpeedyForcing(SpeedyForcing):
    """Use the native forcing term and apply precomposed marine albedo once."""

    def __call__(
        self, state: Any, diagnostics: dict[str, Any], forcing: Any, terrain: Any
    ) -> tuple[Any, dict[str, Any]]:
        """Return native tendencies/diagnostics with only the two albedos replaced."""
        wrapped = isinstance(forcing, CoupledJCMForcing)
        native_forcing = forcing.base if wrapped else forcing
        tendencies, result = super().__call__(
            state, diagnostics, native_forcing, terrain
        )
        if not wrapped:
            return tendencies, result
        radiation = result["_mod_radcon"]
        marine = forcing.marine_surface_albedo
        combined = marine + terrain.fmask * (radiation.alb_l - marine)
        return tendencies, {
            **result,
            "_mod_radcon": radiation.copy(alb_s=marine, albsfc=combined),
        }
