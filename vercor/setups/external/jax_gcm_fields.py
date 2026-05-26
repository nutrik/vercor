from __future__ import annotations

import jax
import jax.numpy as jnp

from vercor.dtypes import as_jax_real_array
from vercor.fluxes.vertical_coordinates import (
    compute_sigma_pressure_levels,
    get_altitudes_sigma_levels,
)

REFERENCE_SURFACE_TEMPERATURE = 273.15 + 15.0
COLD_SURFACE_TEMPERATURE_THRESHOLD = 250.0
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
)
JAXGCM_REQUIRED_GRID_FIELD_NAMES = (
    "land_surface_temperature",
    "sea_surface_temperature",
    "total_surface_temperature",
    *JAXGCM_OUTPUT_GRID_FIELD_NAMES,
)


@jax.jit
def _cleanup_surface_temperature_fields(
    land_surface_temperature: object,
    sea_surface_temperature: object,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    land_surface_temperature_array = jnp.nan_to_num(
        as_jax_real_array(land_surface_temperature)
    )
    sea_surface_temperature_array = jnp.nan_to_num(
        as_jax_real_array(sea_surface_temperature)
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
def _prepare_surface_temperature_forcing(
    total_surface_temperature: object,
    land_fraction_mask: object,
) -> tuple[jax.Array, jax.Array]:
    total_surface_temperature_array = as_jax_real_array(total_surface_temperature)
    land_fraction_mask_array = as_jax_real_array(land_fraction_mask)

    land_surface_temperature = (
        total_surface_temperature_array * land_fraction_mask_array
    )
    sea_surface_temperature = total_surface_temperature_array * (
        1.0 - land_fraction_mask_array
    )

    land_surface_temperature = jnp.where(
        land_surface_temperature == 0.0,
        REFERENCE_SURFACE_TEMPERATURE,
        land_surface_temperature,
    )
    sea_surface_temperature = jnp.where(
        sea_surface_temperature == 0.0,
        REFERENCE_SURFACE_TEMPERATURE,
        sea_surface_temperature,
    )

    return land_surface_temperature, sea_surface_temperature


@jax.jit
def _map_jcm_output_fields(
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
) -> dict[str, jax.Array]:
    u_velocity = as_jax_real_array(u_wind)[-1, :, :].T
    v_velocity = as_jax_real_array(v_wind)[-1, :, :].T
    temperature_2m = as_jax_real_array(temperature)[-1, :, :].T
    specific_humidity_2m = as_jax_real_array(specific_humidity)[-1, :, :].T / 1000.0

    sensible_heat_flux = -jnp.sum(
        as_jax_real_array(surface_sensible_heat_flux), axis=2
    ).T
    latent_heat_flux = -jnp.sum(
        as_jax_real_array(surface_evaporation) / 1e3 * latvap,
        axis=2,
    ).T
    net_shortwave_radiation_flux_2m = as_jax_real_array(net_shortwave_radiation_flux).T
    downward_longwave_radiation_flux_2m = as_jax_real_array(
        downward_longwave_radiation_flux
    ).T

    pressure = compute_sigma_pressure_levels(
        as_jax_real_array(reference_pressure),
        as_jax_real_array(0.0),
        as_jax_real_array(sigma_levels),
        as_jax_real_array(normalized_surface_pressure).T,
    )

    density = (
        as_jax_real_array(mwdair)
        / as_jax_real_array(rgas)
        * pressure[-1, ...]
        / temperature_2m
    )
    potential_temperature = temperature_2m * (
        as_jax_real_array(potential_temperature_reference_pressure) / pressure[-1, ...]
    ) ** as_jax_real_array(cappa)

    model_level_height = get_altitudes_sigma_levels(
        as_jax_real_array(temperature).transpose((0, 2, 1))[::-1, :, :],
        pressure[::-1, :, :],
        as_jax_real_array(specific_humidity).transpose((0, 2, 1))[::-1, :, :] / 1000.0,
    )[1, :, :]

    return {
        "u_velocity": u_velocity,
        "v_velocity": v_velocity,
        "temperature": temperature_2m,
        "specific_humidity": specific_humidity_2m,
        "sensible_heat_flux": sensible_heat_flux,
        "latent_heat_flux": latent_heat_flux,
        "net_shortwave_radiation_flux": net_shortwave_radiation_flux_2m,
        "downward_longwave_radiation_flux": downward_longwave_radiation_flux_2m,
        "pressure": pressure,
        "density": density,
        "potential_temperature": potential_temperature,
        "model_level_height": model_level_height,
    }


__all__ = [
    "COLD_SURFACE_TEMPERATURE_THRESHOLD",
    "JAXGCM_OUTPUT_GRID_FIELD_NAMES",
    "JAXGCM_REQUIRED_GRID_FIELD_NAMES",
    "REFERENCE_SURFACE_TEMPERATURE",
    "_cleanup_surface_temperature_fields",
    "_map_jcm_output_fields",
    "_prepare_surface_temperature_forcing",
]
