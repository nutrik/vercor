"""Map native SPEEDY model-level air, surface fluxes, and diagnosed phases."""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from vercor._numerical_safety import replace_missing_nan
from vercor.dtypes import DTypePolicy, as_jax_real_array
from vercor.fluxes.vertical_coordinates import (
    _compute_surface_nearest_sigma_level_altitude,
    compute_sigma_pressure_levels,
)

REFERENCE_SURFACE_TEMPERATURE = 273.15 + 15.0
REFERENCE_AIR_DENSITY = 1.2
REFERENCE_MODEL_LEVEL_HEIGHT = 50.0
COLD_SURFACE_TEMPERATURE_THRESHOLD = 250.0
JAXGCM_INPUT_GRID_FIELD_NAMES = (
    "land_surface_temperature",
    "sea_surface_temperature",
    "soil_moisture",
)
JAXGCM_OUTPUT_GRID_FIELD_NAMES = (
    "u_velocity",
    "v_velocity",
    "temperature",
    "specific_humidity",
    "sensible_heat_flux",
    "latent_heat_flux",
    "net_shortwave_radiation_flux",
    "downward_longwave_radiation_flux",
    "density",
    "potential_temperature",
    "model_level_height",
    "rain_rate",
    "snow_rate",
)
JAXGCM_REQUIRED_GRID_FIELD_NAMES = (
    *JAXGCM_INPUT_GRID_FIELD_NAMES,
    "total_surface_temperature",
    *JAXGCM_OUTPUT_GRID_FIELD_NAMES,
)


@jax.jit
def cleanup_surface_temperature_fields(
    land_surface_temperature: object,
    sea_surface_temperature: object,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    land_surface_temperature_array = replace_missing_nan(
        as_jax_real_array(land_surface_temperature),
        owner="JCM land surface temperature",
    )
    sea_surface_temperature_array = replace_missing_nan(
        as_jax_real_array(sea_surface_temperature),
        owner="JCM sea surface temperature",
    )
    total_surface_temperature = (
        land_surface_temperature_array + sea_surface_temperature_array
    )
    cold_surface_cells = total_surface_temperature < COLD_SURFACE_TEMPERATURE_THRESHOLD
    return (
        land_surface_temperature_array,
        sea_surface_temperature_array,
        total_surface_temperature,
        cold_surface_cells,
    )


@jax.jit
def prepare_surface_temperature_forcing(
    total_surface_temperature: object,
    land_fraction_mask: object,
) -> tuple[jax.Array, jax.Array]:
    """Build finite JCM land and sea temperatures from one surface field.

    JCM applies ``land_fraction_mask`` when combining its land and sea fluxes,
    so fractional coastal cells require the complete temperature in both
    forcing arrays. Scaling a temperature by its surface fraction is
    physically invalid and can make JCM's reverse-mode surface-flux derivative
    non-finite. A reference value fills only cells where the corresponding
    surface is absent.
    """

    total_surface_temperature_array = as_jax_real_array(total_surface_temperature)
    land_fraction_mask_array = as_jax_real_array(land_fraction_mask)

    land_surface_temperature = jnp.where(
        land_fraction_mask_array > 0.0,
        total_surface_temperature_array,
        REFERENCE_SURFACE_TEMPERATURE,
    )
    sea_surface_temperature = jnp.where(
        land_fraction_mask_array < 1.0,
        total_surface_temperature_array,
        REFERENCE_SURFACE_TEMPERATURE,
    )

    return land_surface_temperature, sea_surface_temperature


@partial(jax.jit, static_argnames=("dtype",))
def map_jcm_output_fields(
    latvap: float,
    reference_pressure: float,
    sigma_levels: object,
    mwdair: float,
    rgas: float,
    potential_temperature_reference_pressure: float,
    cappa: float,
    surface_sensible_heat_flux: object,
    surface_evaporation: object,
    downward_longwave_radiation_flux: object,
    net_shortwave_radiation_flux: object,
    normalized_surface_pressure: object,
    u_wind: object,
    v_wind: object,
    temperature: object,
    specific_humidity: object,
    *,
    large_scale_precipitation: object,
    convective_precipitation: object,
    temperature_2m: object,
    snow_temperature: object,
    rain_temperature: object,
    dtype: DTypePolicy,
) -> dict[str, jax.Array]:
    """Map native SPEEDY fields to the coupler's (latitude, longitude) grid.

    ``precls`` and ``precnv`` are downward-positive g/m²/s. Sum them before
    regridding and convert once to kg/m²/s. On the native (longitude,
    latitude) grid, partition the sum using near-surface ``surface_flux.t0``
    (``temperature_2m``, K), then transpose both phases for exchange. The
    snow fraction is sigmoid(4 * (midpoint - T) / threshold_span), matching
    the midpoint and midpoint slope of the calibrated linear-transition
    model in Bedard-Therrien et al. (2025), doi:10.5194/hess-29-1135-2025,
    Appendix C. The ordered thresholds are in K with a positive span.
    The sigmoid retains phase tails outside that interval; it is a smooth
    approximation, not the paper's piecewise-linear model or a new fit.

    SPEEDY's t0 is a near-surface diagnostic without a specified 2 m height.
    All exported air fields retain the bottom sigma-center meaning required
    by the bulk fluxes and ``model_level_height``; t0 is used only for phase
    diagnosis. Radiation and turbulent heat fluxes are surface quantities.
    """

    latvap_array = as_jax_real_array(latvap, dtype)
    reference_pressure_array = as_jax_real_array(reference_pressure, dtype)
    sigma_levels_array = as_jax_real_array(sigma_levels, dtype)
    mwdair_array = as_jax_real_array(mwdair, dtype)
    rgas_array = as_jax_real_array(rgas, dtype)
    potential_temperature_reference_pressure_array = as_jax_real_array(
        potential_temperature_reference_pressure, dtype
    )
    cappa_array = as_jax_real_array(cappa, dtype)
    normalized_surface_pressure_array = as_jax_real_array(
        normalized_surface_pressure, dtype
    ).T
    temperature_array = as_jax_real_array(temperature, dtype)
    specific_humidity_array = as_jax_real_array(specific_humidity, dtype)

    u_velocity_bottom = as_jax_real_array(u_wind, dtype)[-1, :, :].T
    v_velocity_bottom = as_jax_real_array(v_wind, dtype)[-1, :, :].T
    temperature_bottom = temperature_array[-1, :, :].T
    specific_humidity_bottom = specific_humidity_array[-1, :, :].T / 1000.0
    temperature_2m = as_jax_real_array(temperature_2m, dtype)
    snow_temperature_array = as_jax_real_array(snow_temperature, dtype)
    rain_temperature_array = as_jax_real_array(rain_temperature, dtype)
    total_precipitation = (
        as_jax_real_array(large_scale_precipitation, dtype)
        + as_jax_real_array(convective_precipitation, dtype)
    ) / 1000.0
    threshold_span = rain_temperature_array - snow_temperature_array
    midpoint = 0.5 * (rain_temperature_array + snow_temperature_array)
    # A logistic has midpoint derivative -k/4; LT has derivative -1/span.
    snow_fraction = jax.nn.sigmoid(4.0 * (midpoint - temperature_2m) / threshold_span)
    snow_precipitation = total_precipitation * snow_fraction
    rain_rate = (total_precipitation - snow_precipitation).T
    snow_rate = snow_precipitation.T

    sensible_heat_flux = -jnp.sum(
        as_jax_real_array(surface_sensible_heat_flux, dtype), axis=2
    ).T
    latent_heat_flux = -jnp.sum(
        as_jax_real_array(surface_evaporation, dtype) / 1e3 * latvap_array,
        axis=2,
    ).T
    net_shortwave_radiation_flux_surface = as_jax_real_array(
        net_shortwave_radiation_flux, dtype
    ).T
    downward_longwave_radiation_flux_surface = as_jax_real_array(
        downward_longwave_radiation_flux, dtype
    ).T

    pressure = compute_sigma_pressure_levels(
        reference_pressure_array,
        as_jax_real_array(0.0, dtype),
        sigma_levels_array,
        normalized_surface_pressure_array,
    )

    pressure_bottom = pressure[-1, :, :]
    density_bottom = mwdair_array / rgas_array * pressure_bottom / temperature_bottom
    potential_temperature_bottom = (
        temperature_bottom
        * (potential_temperature_reference_pressure_array / pressure_bottom)
        ** cappa_array
    )

    model_level_height = _compute_surface_nearest_sigma_level_altitude(
        temperature_bottom,
        sigma_levels_array[-1],
        specific_humidity_bottom,
    )

    return {
        "u_velocity": u_velocity_bottom,
        "v_velocity": v_velocity_bottom,
        "temperature": temperature_bottom,
        "specific_humidity": specific_humidity_bottom,
        "sensible_heat_flux": sensible_heat_flux,
        "latent_heat_flux": latent_heat_flux,
        "net_shortwave_radiation_flux": net_shortwave_radiation_flux_surface,
        "downward_longwave_radiation_flux": downward_longwave_radiation_flux_surface,
        "pressure": pressure,
        "density": density_bottom,
        "potential_temperature": potential_temperature_bottom,
        "model_level_height": model_level_height,
        "rain_rate": rain_rate,
        "snow_rate": snow_rate,
    }


__all__ = [
    "COLD_SURFACE_TEMPERATURE_THRESHOLD",
    "JAXGCM_INPUT_GRID_FIELD_NAMES",
    "JAXGCM_OUTPUT_GRID_FIELD_NAMES",
    "JAXGCM_REQUIRED_GRID_FIELD_NAMES",
    "REFERENCE_AIR_DENSITY",
    "REFERENCE_MODEL_LEVEL_HEIGHT",
    "REFERENCE_SURFACE_TEMPERATURE",
    "cleanup_surface_temperature_fields",
    "map_jcm_output_fields",
    "prepare_surface_temperature_forcing",
]
