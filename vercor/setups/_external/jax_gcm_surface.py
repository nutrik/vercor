"""Carry composed marine albedo through native JCM forcing selection.

The wrapper owns no model state and preserves native ForcingData copy/select
behavior. SPEEDY's forcing term still diagnoses land albedo and all other
boundary conditions. Only marine and combined albedo are replaced; native
diagnostic schemas stay unchanged so integration carries remain stable PyTrees.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import jax
import jax.numpy as jnp
from jcm.forcing import ForcingData, TimeSeries
from jcm.physics.speedy.speedy_terms import SpeedyForcing

from vercor._numerical_safety import require_active_finite


def blend_jcm_forcing_field(
    native: Any, coupled: jax.Array, prescribed_fraction: jax.Array
) -> Any:
    """Add a prescribed marine contribution while retaining native time labels.

    ``coupled`` is already normalized by total marine area; the dimensionless
    fraction and both values use native (longitude, latitude) layout. Affine
    blending commutes with native interpolation and leaves its time policy intact.
    """
    values = native.values if isinstance(native, TimeSeries) else native
    values = jnp.asarray(values, dtype=coupled.dtype)
    require_active_finite(
        values,
        active_mask=prescribed_fraction > 0,
        owner="JCM prescribed marine forcing",
    )
    blended = coupled + prescribed_fraction * jnp.where(
        prescribed_fraction > 0, values, 0
    )
    return (
        replace(native, values=blended) if isinstance(native, TimeSeries) else blended
    )


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class CoupledJCMForcing:
    """Native forcing plus instantaneous marine albedo in native (x, y) layout."""

    base: ForcingData
    marine_surface_albedo: jax.Array
    native_background: ForcingData | None = None
    prescribed_marine_fraction: jax.Array | float = 0.0

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
            self.native_background,
            self.prescribed_marine_fraction,
        )

    def select(self, date: Any) -> CoupledJCMForcing:
        """Select native time series/solar geometry and retain current albedo."""
        return CoupledJCMForcing(
            self.base.select(date),
            self.marine_surface_albedo,
            (
                None
                if self.native_background is None
                else self.native_background.select(date)
            ),
            self.prescribed_marine_fraction,
        )


class CoupledSpeedyForcing(SpeedyForcing):
    """Use the native forcing term and apply precomposed marine albedo once."""

    def __call__(
        self, state: Any, diagnostics: dict[str, Any], forcing: Any, terrain: Any
    ) -> tuple[Any, dict[str, Any]]:
        """Return native tendencies/diagnostics with only the two albedos replaced."""
        wrapped = isinstance(forcing, CoupledJCMForcing)
        native_forcing = forcing.base if wrapped else forcing
        # Diagnose native albedo with the original prescribed ice, then blend
        # it once. Downstream native fluxes still receive the applied SST/ice.
        if wrapped and forcing.native_background is not None:
            native_forcing = native_forcing.copy(
                sice_am=forcing.native_background.sice_am
            )
        tendencies, result = super().__call__(
            state, diagnostics, native_forcing, terrain
        )
        if not wrapped:
            return tendencies, result
        radiation = result["_mod_radcon"]
        marine = (
            forcing.marine_surface_albedo
            + forcing.prescribed_marine_fraction * radiation.alb_s
        )
        combined = marine + terrain.fmask * (radiation.alb_l - marine)
        return tendencies, {
            **result,
            "_forcing_2d": forcing.base,
            "_mod_radcon": radiation.copy(alb_s=marine, albsfc=combined),
        }
