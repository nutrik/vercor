from __future__ import annotations

import jax
import jax.numpy as jnp

from vercor.dtypes import as_jax_real_array
from vercor.settings import VercorSettings
from vercor.types import RuntimeArray


def _virtual_temperature_from_specific_humidity(
    temperature: RuntimeArray,
    specific_humidity: RuntimeArray,
    virtual_temperature_correction: float,
) -> jax.Array:
    """Return virtual temperature for specific humidity in kg/kg."""

    temperature_array = as_jax_real_array(temperature)
    specific_humidity_array = as_jax_real_array(specific_humidity)
    return temperature_array * (
        1.0 + virtual_temperature_correction * specific_humidity_array
    )


def compute_hybrid_pressure_levels(
    sp: RuntimeArray,
    hya: RuntimeArray,
    hyb: RuntimeArray,
) -> jax.Array:
    """Compute ECMWF-style hybrid-sigma pressure levels."""

    sp_array = as_jax_real_array(sp)
    hya_array = as_jax_real_array(hya)
    hyb_array = as_jax_real_array(hyb)
    return (
        hya_array[jnp.newaxis, jnp.newaxis, :]
        + hyb_array[jnp.newaxis, jnp.newaxis, :] * sp_array[:, :, jnp.newaxis]
    )


def get_altitudes_hybrid_sigma_levels(
    settings: VercorSettings,
    t: RuntimeArray,
    q: RuntimeArray,
    ph: RuntimeArray,
) -> jax.Array:
    """Compute geometric altitudes at ECMWF-IFS hybrid-sigma full levels."""

    return compute_hybrid_sigma_full_level_altitudes(
        t,
        q,
        ph,
        earth_radius=settings.earth_radius,
        gravity=settings.gravity,
        rdair=settings.rdair,
        zvir=settings.zvir,
    )


def compute_hybrid_sigma_full_level_altitudes(
    t: RuntimeArray,
    q: RuntimeArray,
    ph: RuntimeArray,
    *,
    earth_radius: float,
    gravity: float,
    rdair: float,
    zvir: float,
) -> jax.Array:
    """Return bottom-to-top hybrid-sigma full-level geometric altitudes."""

    ph_array = as_jax_real_array(ph)
    virtual_temperature = _virtual_temperature_from_specific_humidity(t, q, zvir)

    lower_half_pressure = ph_array[:, :, :-1]
    upper_half_pressure = ph_array[:, :, 1:]
    zero_lower_half_pressure = lower_half_pressure == 0.0
    safe_lower_half_pressure = jnp.where(
        zero_lower_half_pressure,
        0.1,
        lower_half_pressure,
    )

    dlog_p = jnp.log(upper_half_pressure / safe_lower_half_pressure)
    alpha_general = 1.0 - (
        safe_lower_half_pressure
        / (upper_half_pressure - safe_lower_half_pressure)
        * dlog_p
    )
    alpha = jnp.where(zero_lower_half_pressure, jnp.log(2.0), alpha_general)

    moist_temperature_rd = virtual_temperature * rdair
    half_level_geopotential_increment = jnp.flip(
        moist_temperature_rd * dlog_p,
        axis=2,
    )
    half_level_geopotential = jnp.cumsum(half_level_geopotential_increment, axis=2)

    padded_half_level_geopotential = jnp.pad(
        half_level_geopotential,
        ((0, 0), (0, 0), (1, 0)),
    )
    full_level_geopotential = (
        jnp.flip(moist_temperature_rd * alpha, axis=2)
        + padded_half_level_geopotential[:, :, :-1]
    )
    geopotential_height = full_level_geopotential / gravity
    return earth_radius * geopotential_height / (earth_radius - geopotential_height)


def compute_sigma_pressure_levels(
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
