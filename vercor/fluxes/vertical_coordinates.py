from __future__ import annotations

import jax
import jax.numpy as jnp

from vercor.dtypes import as_jax_real_array
from vercor.fluxes.utilities import _virtual_temperature_from_specific_humidity
from vercor.types import RuntimeArray


def compute_pressure_levels(
    reference_pressure: RuntimeArray | float,
    top_pressure: RuntimeArray | float,
    sigma_levels: RuntimeArray,
    normalized_surface_pressure: RuntimeArray,
) -> jax.Array:
    """Compute pressure levels from sigma levels and normalized surface pressure."""

    p0 = as_jax_real_array(reference_pressure)
    p_top = as_jax_real_array(top_pressure)
    sigma = as_jax_real_array(sigma_levels)
    nps = as_jax_real_array(normalized_surface_pressure)

    if p_top.ndim != 0:
        raise ValueError("top_pressure must be a scalar array")
    if sigma.ndim != 1:
        raise ValueError("sigma_levels must be a 1D array")

    ps = as_jax_real_array(nps * p0)[jnp.newaxis, :, :]
    p_top_bcast = jnp.broadcast_to(p_top, ps.shape)
    return p_top_bcast + sigma[:, jnp.newaxis, jnp.newaxis] * (ps - p_top_bcast)


def get_altitudes_sigma_levels(
    temperature: RuntimeArray,
    pressure: RuntimeArray,
    specific_humidity: RuntimeArray,
    *,
    z0: RuntimeArray | float = 0.0,
    g: float = 9.80665,
    Rd: float = 287.05,
    Rv: float = 461.5,
) -> jax.Array:
    """Compute geometric altitude on pressure levels with the hypsometric equation."""

    temperature_array = as_jax_real_array(temperature)
    pressure_array = as_jax_real_array(pressure)
    humidity_array = as_jax_real_array(specific_humidity)

    if (
        temperature_array.ndim != 3
        or pressure_array.ndim != 3
        or humidity_array.ndim != 3
    ):
        raise ValueError(
            "temperature, pressure, specific_humidity must all be 3D: (nlev, nlat, nlon)"
        )
    if (
        temperature_array.shape != pressure_array.shape
        or temperature_array.shape != humidity_array.shape
    ):
        raise ValueError(
            "temperature, pressure, specific_humidity must have identical shapes"
        )

    nlev, nlat, nlon = temperature_array.shape
    eps = Rv / Rd
    virtual_temperature = _virtual_temperature_from_specific_humidity(
        temperature_array,
        humidity_array,
        eps - 1.0,
    )
    log_pressure_ratio = jnp.log(pressure_array[:-1, :, :] / pressure_array[1:, :, :])
    mean_virtual_temperature = 0.5 * (
        virtual_temperature[:-1, :, :] + virtual_temperature[1:, :, :]
    )
    dz = (Rd / g) * mean_virtual_temperature * log_pressure_ratio

    altitude = jnp.empty_like(temperature_array)
    z0_array = as_jax_real_array(z0)
    if z0_array.ndim == 0:
        altitude = altitude.at[0, :, :].set(z0_array)
    elif z0_array.shape == (nlat, nlon):
        altitude = altitude.at[0, :, :].set(z0_array)
    elif z0_array.shape == (nlev, nlat, nlon):
        altitude = altitude.at[0, :, :].set(z0_array[0, :, :])
    else:
        raise ValueError("z0 must be a scalar, (nlat,nlon), or (nlev,nlat,nlon)")

    return altitude.at[1:, :, :].set(altitude[0:1, :, :] + jnp.cumsum(dz, axis=0))
