"""Coupled stress, heat, and virtual salt fluxes on the Veros surface T grid."""

from __future__ import annotations

from collections.abc import Mapping
from functools import partial

import jax
import jax.numpy as jnp

from vercor.dtypes import DTypePolicy, as_jax_real_array
from vercor._numerical_safety import replace_missing_nan
from vercor.setups._surface_fluxes import compute_surface_fluxes
from vercor.physics import PhysicalConstants
from vercor.types import RuntimeArray

from veros.state import VerosState


@partial(jax.jit, static_argnames=("dtype",))
def _compute_virtual_salt_flux(
    constants: PhysicalConstants,
    rain_rate: object,
    snow_rate: object,
    evaporation: object,
    surface_mask: object,
    *,
    dtype: DTypePolicy | None = None,
) -> jax.Array:
    """Convert downward-positive kg/m²/s to m (g/kg)/s on the ocean T grid.

    CESM bulk evaporation is negative when water leaves the ocean. With a
    fixed reference salinity, ``-S_ref * (rain + snow + evaporation) / rho_fw``
    freshens under precipitation and salinifies under evaporation, without
    changing ocean volume. This is the freshwater term of the rigid-lid
    conversion in https://github.com/nutrik/veros/pull/81.
    Missing inactive land operands are neutralized before arithmetic for AD.
    """

    mask = as_jax_real_array(surface_mask, dtype)
    rain = jnp.where(mask > 0.0, as_jax_real_array(rain_rate, dtype), 0.0)
    snow = jnp.where(mask > 0.0, as_jax_real_array(snow_rate, dtype), 0.0)
    evap = jnp.where(mask > 0.0, as_jax_real_array(evaporation, dtype), 0.0)
    return (
        -as_jax_real_array(constants.surface_salinity_reference, dtype)
        / as_jax_real_array(constants.freshwater_density, dtype)
        * (rain + snow + evap)
        * mask
    )


def compute_fluxes(
    veros_state: VerosState,
    runtime_fields: Mapping[str, RuntimeArray],
    constants: PhysicalConstants,
    dtype: DTypePolicy,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Return atmosphere-ocean fluxes as five runtime-dtype (nLat, nLon) arrays.

    Outputs are zonal/meridional stress in N/m², net heat in W/m², the SST
    restoring coefficient in W/(m² K), and virtual salt flux in m (g/kg)/s.
    Rain and water-equivalent snow inputs are downward-positive kg/m²/s.
    Bulk evaporation has the same signed mass-flux convention. Native ocean
    velocity/temperature and wet masks are converted from the haloed Veros
    grid before evaluating the CESM bulk formula and freshwater conversion.
    """

    vs = veros_state.variables

    u_tgrid = 0.5 * (
        as_jax_real_array(vs.u[1:, 2:-2, -1, vs.tau], dtype)
        + as_jax_real_array(vs.u[:-1, 2:-2, -1, vs.tau], dtype)
    )
    v_tgrid = 0.5 * (
        as_jax_real_array(vs.v[2:-2, 1:, -1, vs.tau], dtype)
        + as_jax_real_array(vs.v[2:-2, :-1, -1, vs.tau], dtype)
    )

    temp = as_jax_real_array(vs.temp[2:-2, 2:-2, -1, vs.tau], dtype).T + 273.15

    bulk = compute_surface_fluxes(
        runtime_fields,
        sea_surface_temperature=temp,
        u_ocean=u_tgrid[1:-2, :].T,
        v_ocean=v_tgrid[:, 1:-2].T,
        ocean_mask=as_jax_real_array(vs.maskT[2:-2, 2:-2, -1], dtype).T,
        constants=constants,
        dtype=dtype,
    )
    salt_flux = _compute_virtual_salt_flux(
        constants,
        as_jax_real_array(runtime_fields["rain_rate"], dtype),
        as_jax_real_array(runtime_fields["snow_rate"], dtype),
        as_jax_real_array(bulk.evap, dtype),
        as_jax_real_array(vs.maskT[2:-2, 2:-2, -1], dtype).T,
        dtype=dtype,
    )

    return (
        as_jax_real_array(bulk.taux, dtype),
        as_jax_real_array(bulk.tauy, dtype),
        as_jax_real_array(bulk.qnet, dtype),
        as_jax_real_array(bulk.qnec, dtype),
        as_jax_real_array(salt_flux, dtype),
    )


@partial(jax.jit, static_argnames=("dtype",))
def compute_coupled_fluxes(
    fields: Mapping[str, RuntimeArray],
    constants: PhysicalConstants,
    dtype: DTypePolicy,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Convert total downward heat/mass/salt and canonical face stresses once.

    Salinity forcing is (-S_ref*M + 1000*J)/rho_fw in (g/kg)*m/s.
    M is total bulk mass, J actual salt mass, both kg/m2/s. Fixed-reference
    Boussinesq virtual salt differs from physical inventory at other salinities.
    Missing inactive values are neutralized before parameter arithmetic; the
    component's receive boundary rejects missing values on its active domain.
    Native tracer kernels own rho_0, heat capacity, timestep and layer depth.
    """

    def clean(name: str) -> jax.Array:
        return replace_missing_nan(
            as_jax_real_array(fields[name], dtype),
            owner=f"Veros total feedback '{name}'",
        )

    mass = clean("sea_surface_mass_flux")
    salt = clean("sea_surface_salt_mass_flux")
    virtual = (
        -as_jax_real_array(constants.surface_salinity_reference, dtype) * mass
        + as_jax_real_array(1000.0, dtype) * salt
    ) / as_jax_real_array(constants.freshwater_density, dtype)
    return (
        clean("ocean_u_stress"),
        clean("ocean_v_stress"),
        clean("sea_surface_heat_flux"),
        virtual,
    )


__all__ = ["compute_fluxes", "compute_coupled_fluxes"]
