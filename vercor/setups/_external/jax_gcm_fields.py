from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from vercor.dtypes import DTypePolicy, as_jax_real_array
from vercor.fluxes.vertical_coordinates import (
    compute_sigma_pressure_levels,
    get_altitudes_sigma_levels,
)

REFERENCE_SURFACE_TEMPERATURE = 273.15 + 15.0
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
    dtype: DTypePolicy,
) -> dict[str, jax.Array]:
    latvap_array = as_jax_real_array(latvap, dtype)
    reference_pressure_array = as_jax_real_array(reference_pressure, dtype)
    sigma_levels_array = as_jax_real_array(sigma_levels, dtype)
    mwdair_array = as_jax_real_array(mwdair, dtype)
    rgas_array = as_jax_real_array(rgas, dtype)
    potential_temperature_reference_pressure_array = as_jax_real_array(
        potential_temperature_reference_pressure, dtype
    )
    cappa_array = as_jax_real_array(cappa, dtype)
    temperature_array = as_jax_real_array(temperature, dtype)
    specific_humidity_array = as_jax_real_array(specific_humidity, dtype)

    u_velocity = as_jax_real_array(u_wind, dtype)[-1, :, :].T
    v_velocity = as_jax_real_array(v_wind, dtype)[-1, :, :].T
    temperature_2m = temperature_array[-1, :, :].T
    specific_humidity_2m = specific_humidity_array[-1, :, :].T / 1000.0

    sensible_heat_flux = -jnp.sum(
        as_jax_real_array(surface_sensible_heat_flux, dtype), axis=2
    ).T
    latent_heat_flux = -jnp.sum(
        as_jax_real_array(surface_evaporation, dtype) / 1e3 * latvap_array,
        axis=2,
    ).T
    net_shortwave_radiation_flux_2m = as_jax_real_array(
        net_shortwave_radiation_flux, dtype
    ).T
    downward_longwave_radiation_flux_2m = as_jax_real_array(
        downward_longwave_radiation_flux, dtype
    ).T

    pressure = compute_sigma_pressure_levels(
        reference_pressure_array,
        as_jax_real_array(0.0, dtype),
        sigma_levels_array,
        as_jax_real_array(normalized_surface_pressure, dtype).T,
    )

    density = mwdair_array / rgas_array * pressure[-1, :, :] / temperature_2m
    potential_temperature = (
        temperature_2m
        * (potential_temperature_reference_pressure_array / pressure[-1, :, :])
        ** cappa_array
    )

    model_level_height = get_altitudes_sigma_levels(
        temperature_array.transpose((0, 2, 1))[::-1, :, :],
        pressure[::-1, :, :],
        specific_humidity_array.transpose((0, 2, 1))[::-1, :, :] / 1000.0,
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
    "JAXGCM_INPUT_GRID_FIELD_NAMES",
    "JAXGCM_OUTPUT_GRID_FIELD_NAMES",
    "JAXGCM_REQUIRED_GRID_FIELD_NAMES",
    "REFERENCE_SURFACE_TEMPERATURE",
    "cleanup_surface_temperature_fields",
    "map_jcm_output_fields",
    "prepare_surface_temperature_forcing",
]
