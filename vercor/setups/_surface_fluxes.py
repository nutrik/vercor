"""Shared CESM ocean bulk diagnostics on (latitude, longitude) tracer cells.

Inputs retain colocated model-level meteorology and tracer-centered ocean
currents. Fluxes follow VerCOR's downward-positive heat/mass convention;
evaporation is negative for ocean water loss. tref (K) and qref (kg/kg) use
the configured CESM reference height, normally 2 m. This open-ocean similarity
diagnosis is also used beneath prescribed ice; it is not an ERA5 observation.
Actual air winds at 10 m retain ocean-current translation and share that solve.
"""

from __future__ import annotations

from collections.abc import Mapping
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp

from vercor.dtypes import DTypePolicy, as_jax_real_array
from vercor.fluxes.bulk_formula_cesm import _compute_ocean_surface_fluxes_with_winds
from vercor.physics import PhysicalConstants
from vercor.types import RuntimeArray


class SurfaceFluxes(NamedTuple):
    """Traced stress, heat, restoring, evaporation and reference-air diagnostics."""

    taux: jax.Array
    tauy: jax.Array
    qnet: jax.Array
    qnec: jax.Array
    evap: jax.Array
    tref: jax.Array
    qref: jax.Array
    u_velocity_10m: jax.Array
    v_velocity_10m: jax.Array


@partial(jax.jit, static_argnames=("dtype",))
def compute_surface_fluxes(
    runtime_fields: Mapping[str, RuntimeArray],
    *,
    sea_surface_temperature: RuntimeArray,
    u_ocean: RuntimeArray,
    v_ocean: RuntimeArray,
    ocean_mask: RuntimeArray,
    constants: PhysicalConstants,
    dtype: DTypePolicy,
) -> SurfaceFluxes:
    """Reuse the CESM bulk kernel and existing Veros heat/restoring equations."""
    bulk = _compute_ocean_surface_fluxes_with_winds(
        constants,
        as_jax_real_array(ocean_mask, dtype),
        as_jax_real_array(runtime_fields["model_level_height"], dtype),
        as_jax_real_array(runtime_fields["air_u_velocity"], dtype),
        as_jax_real_array(runtime_fields["air_v_velocity"], dtype),
        as_jax_real_array(runtime_fields["air_potential_temperature"], dtype),
        as_jax_real_array(runtime_fields["air_specific_humidity"], dtype),
        as_jax_real_array(runtime_fields["air_density"], dtype),
        as_jax_real_array(runtime_fields["air_temperature"], dtype),
        as_jax_real_array(u_ocean, dtype),
        as_jax_real_array(v_ocean, dtype),
        as_jax_real_array(sea_surface_temperature, dtype),
    )
    senf, latf, lwup, evap, taux, tauy, tref, qref, *_, dqfldt = bulk.fluxes
    qnet = (
        as_jax_real_array(runtime_fields["net_shortwave_radiation_flux"], dtype)
        + as_jax_real_array(runtime_fields["downward_longwave_radiation_flux"], dtype)
        + lwup
        + senf
        + latf
    )
    qnec = -jnp.where(dqfldt <= -1e10, 0.0, dqfldt)
    return SurfaceFluxes(
        *(
            as_jax_real_array(value, dtype)
            for value in (
                taux,
                tauy,
                qnet,
                qnec,
                evap,
                tref,
                qref,
                bulk.u_velocity_10m,
                bulk.v_velocity_10m,
            )
        )
    )
