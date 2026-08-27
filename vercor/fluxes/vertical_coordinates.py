"""Differentiable pressure and altitude kernels for atmospheric coordinates."""

from __future__ import annotations

from typing import TypeAlias, cast

import jax
import jax.numpy as jnp

from vercor.dtypes import as_jax_real_array
from vercor.physics import PhysicalConstants
from vercor.types import RuntimeArray

ScalarPhysicsValue: TypeAlias = float | RuntimeArray


def _virtual_temperature_from_specific_humidity(
    temperature: RuntimeArray,
    specific_humidity: RuntimeArray,
    virtual_temperature_correction: ScalarPhysicsValue,
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
    """Compute ECMWF-style hybrid-sigma pressures ``A + B * ps``.

    ``sp`` has shape ``(nlat, nlon)``. The pressure-valued ``hya`` coefficients
    and dimensionless ``hyb`` coefficients are one-dimensional and share the
    model's top-to-bottom level order. The returned array has shape
    ``(nlat, nlon, nlevel)`` in that same order.
    """

    sp_array = as_jax_real_array(sp)
    hya_array = as_jax_real_array(hya)
    hyb_array = as_jax_real_array(hyb)

    if sp_array.ndim != 2:
        raise ValueError("surface_pressure must be a 2D array")
    if hya_array.ndim != 1 or hyb_array.ndim != 1:
        raise ValueError("hybrid A and B coefficients must be 1D arrays")
    if hya_array.shape != hyb_array.shape:
        raise ValueError("hybrid A and B coefficients must have identical shapes")

    return (
        hya_array[jnp.newaxis, jnp.newaxis, :]
        + hyb_array[jnp.newaxis, jnp.newaxis, :] * sp_array[:, :, jnp.newaxis]
    )


def get_altitudes_hybrid_sigma_levels(
    constants: PhysicalConstants,
    t: RuntimeArray,
    q: RuntimeArray,
    ph: RuntimeArray,
) -> jax.Array:
    """Compute ECMWF-IFS hybrid-sigma full-level heights above ground.

    Temperature and specific humidity in kg/kg have shape
    ``(nlat, nlon, nlevel)``. ``ph`` contains the bounding interface pressures
    with shape ``(nlat, nlon, nlevel + 1)``. All inputs are ordered
    top-to-bottom; returned geometric heights are ordered bottom-to-top. The
    integration uses zero surface geopotential, so the result is height above
    ground rather than altitude above mean sea level.

    The discretization follows IFS equations 2.20--2.23. In particular, it uses
    the model's interface coefficients instead of approximating a full-level
    height directly from the arithmetic midpoint pressure.
    """

    return compute_hybrid_sigma_full_level_altitudes(
        t,
        q,
        ph,
        earth_radius=constants.earth_radius,
        gravity=constants.gravity,
        rdair=constants.dry_air_gas_constant,
        zvir=constants.water_vapor_mass_ratio_correction,
    )


def compute_hybrid_sigma_full_level_altitudes(
    t: RuntimeArray,
    q: RuntimeArray,
    ph: RuntimeArray,
    *,
    earth_radius: ScalarPhysicsValue,
    gravity: ScalarPhysicsValue,
    rdair: ScalarPhysicsValue,
    zvir: ScalarPhysicsValue,
) -> jax.Array:
    """Return bottom-to-top IFS hybrid full-level geometric AGL heights.

    ``t`` and ``q`` must be top-to-bottom ``(nlat, nlon, nlevel)`` arrays;
    ``ph`` must contain their ``nlevel + 1`` top-to-bottom interface pressures.
    The hydrostatic integration starts from zero surface geopotential. A zero
    top interface uses the IFS ``0.1 Pa`` logarithm and ``log(2)`` full-level
    limit. Physics constants are explicit so they remain JAX-traced.
    """

    temperature = as_jax_real_array(t)
    specific_humidity = as_jax_real_array(q)
    half_level_pressure = as_jax_real_array(ph)

    if (
        temperature.ndim != 3
        or specific_humidity.ndim != 3
        or half_level_pressure.ndim != 3
    ):
        raise ValueError(
            "temperature, specific_humidity, and half_level_pressure must be 3D"
        )
    if temperature.shape != specific_humidity.shape:
        raise ValueError("temperature and specific_humidity must have identical shapes")
    expected_half_level_shape = (
        *temperature.shape[:2],
        temperature.shape[2] + 1,
    )
    if half_level_pressure.shape != expected_half_level_shape:
        raise ValueError(
            "half_level_pressure must match the horizontal shape and contain "
            "one more vertical level"
        )

    virtual_temperature = _virtual_temperature_from_specific_humidity(
        temperature,
        specific_humidity,
        zvir,
    )

    upper_interface_pressure = half_level_pressure[:, :, :-1]
    lower_interface_pressure = half_level_pressure[:, :, 1:]
    zero_upper_interface_pressure = upper_interface_pressure == 0.0
    safe_upper_interface_pressure = jnp.where(
        zero_upper_interface_pressure,
        0.1,
        upper_interface_pressure,
    )

    dlog_p = jnp.log(lower_interface_pressure / safe_upper_interface_pressure)
    alpha_general = 1.0 - (
        safe_upper_interface_pressure
        / (lower_interface_pressure - safe_upper_interface_pressure)
        * dlog_p
    )
    alpha = jnp.where(zero_upper_interface_pressure, jnp.log(2.0), alpha_general)

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
    return cast(
        jax.Array,
        earth_radius * geopotential_height / (earth_radius - geopotential_height),
    )


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


def _compute_surface_nearest_sigma_level_altitude(
    temperature: RuntimeArray,
    sigma_level: RuntimeArray | float,
    specific_humidity: RuntimeArray,
    *,
    g: float = 9.80665,
    Rd: float = 287.05,
    Rv: float = 461.5,
) -> jax.Array:
    """Return one sigma-center altitude above its surface pressure."""

    virtual_temperature = _virtual_temperature_from_specific_humidity(
        temperature,
        specific_humidity,
        Rv / Rd - 1.0,
    )
    sigma = as_jax_real_array(sigma_level)
    return cast(jax.Array, -(Rd / g) * virtual_temperature * jnp.log(sigma))


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
    """Compute surface-up geometric altitudes with the hypsometric equation.

    Temperature and specific humidity in kg/kg are sampled at the supplied full
    levels. The first pressure level is anchored at ``z0``; subsequent levels
    must be ordered upward with decreasing pressure.
    """

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
