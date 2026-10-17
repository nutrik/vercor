"""Total-area ocean boundary rates from one native Veris substep.

Native thermodynamics diagnoses actual downward mass/salt input in kg/m2/s,
while residual Qnet is upward W/m2. Ice stress is per ice-covered face area;
combine it with complementary open-water stress using the retained momentum
AreaW/AreaS, mask native coasts, then shift west/south to canonical east/north.
The runtime integrates these complete products in time, separately from the
final instantaneous diagnostics. This adapter assumes the shared ocean grid;
it does not establish staggered conservation for different remapping grids.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp

from vercor.dtypes import DTypePolicy, as_jax_real_array
from vercor.setups._external.veris_forcing import VerisForcingPack
from vercor.types import RuntimeArray


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class VerisSurfaceFluxes:
    """Downward heat/mass/salt rates and east/north total stresses in (y, x)."""

    sea_surface_heat_flux: RuntimeArray
    sea_surface_mass_flux: RuntimeArray
    sea_surface_salt_mass_flux: RuntimeArray
    ocean_u_stress: RuntimeArray
    ocean_v_stress: RuntimeArray

    def to_fields(self) -> dict[str, RuntimeArray]:
        """Return canonical component fields without copying traced arrays."""
        return {field.name: getattr(self, field.name) for field in fields(self)}


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class VerisIntervalBudget:
    """Independent downward external mass (kg/m2/s) and heat (W/m2) means."""

    external_mass_flux: RuntimeArray
    external_heat_flux: RuntimeArray


@partial(jax.jit, static_argnames=("dtype",))
def veris_surface_fluxes(
    state: Any,
    diagnostics: Any,
    forcing_pack: VerisForcingPack,
    *,
    dtype: DTypePolicy,
) -> VerisSurfaceFluxes:
    """Convert haloed instantaneous native rates to complete physical fluxes."""

    def physical(value: Any) -> jax.Array:
        return as_jax_real_array(value, dtype)[2:-2, 2:-2]

    wet = physical(state.iceMask) > 0
    west_wet = (physical(state.iceMaskU) > 0) & wet & jnp.roll(wet, 1, axis=0)
    south_wet = (
        (physical(state.iceMaskV) > 0)
        & wet
        & jnp.concatenate((jnp.zeros_like(wet[:, :1]), wet[:, :-1]), axis=1)
    )
    west_area = physical(state.AreaW)
    south_area = physical(state.AreaS)
    tau_u = jnp.where(
        west_wet,
        west_area * physical(diagnostics.OceanStressU)
        + (1 - west_area) * as_jax_real_array(forcing_pack.open_water_u_stress, dtype),
        0.0,
    )
    tau_v = jnp.where(
        south_wet,
        south_area * physical(diagnostics.OceanStressV)
        + (1 - south_area) * as_jax_real_array(forcing_pack.open_water_v_stress, dtype),
        0.0,
    )
    east = jnp.roll(tau_u, -1, axis=0)
    north = jnp.concatenate((tau_v[:, 1:], jnp.zeros_like(tau_v[:, :1])), axis=1)
    return VerisSurfaceFluxes(
        jnp.where(wet, -physical(state.Qnet), 0.0).T,
        jnp.where(wet, physical(diagnostics.ocean_mass_flux), 0.0).T,
        jnp.where(wet, physical(diagnostics.ocean_salt_mass_flux), 0.0).T,
        east.T,
        north.T,
    )
