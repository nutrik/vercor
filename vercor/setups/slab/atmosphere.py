from collections.abc import Mapping
from typing import Any

import jax
import jax.numpy as jnp

from vercor.components import Component
from vercor.dtypes import as_jax_real_array
from vercor.grid import RectilinearGrid

_REFERENCE_SURFACE_TEMPERATURE = 273.15 + 15.0
_ATMOSPHERE_INPUTS = ("sea_surface_temperature",)
_ATMOSPHERE_OUTPUTS = (
    "temperature_2m",
    "sensible_heat_flux",
    "latent_heat_flux",
    "u_velocity_10m",
    "v_velocity_10m",
)
_ATMOSPHERE_DEFAULT_FIELDS = {
    "temperature_2m": _REFERENCE_SURFACE_TEMPERATURE,
    "sensible_heat_flux": 0.0,
    "latent_heat_flux": 0.0,
    "u_velocity_10m": 0.0,
    "v_velocity_10m": 0.0,
}


@jax.jit
def _default_sea_surface_temperature(temperature_2m: object) -> jax.Array:
    return jnp.full_like(
        as_jax_real_array(temperature_2m),
        _REFERENCE_SURFACE_TEMPERATURE,
    )


@jax.jit
def _bulk_flux_step(
    temperature_2m: object,
    sea_surface_temperature: object,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    temperature_2m_array = as_jax_real_array(temperature_2m)
    sea_surface_temperature_array = as_jax_real_array(sea_surface_temperature)
    delta_temperature = temperature_2m_array - sea_surface_temperature_array
    sensible_heat_flux = -10.0 * delta_temperature
    latent_heat_flux = -0.5 * sensible_heat_flux
    updated_temperature_2m = temperature_2m_array - 0.01 * delta_temperature
    return sensible_heat_flux, latent_heat_flux, updated_temperature_2m


@jax.jit
def _surface_wind_10m(
    latitude: object,
    longitude: object,
) -> tuple[jax.Array, jax.Array]:
    latitude_array = as_jax_real_array(latitude)
    longitude_array = as_jax_real_array(longitude) - 180.0
    latitudes, longitudes = jnp.meshgrid(latitude_array, longitude_array, indexing="ij")
    u_velocity_10m = jnp.cos(jnp.deg2rad(latitudes))
    v_velocity_10m = 0.5 * jnp.sin(jnp.deg2rad(longitudes))
    return u_velocity_10m, v_velocity_10m


def make_slab_atmosphere(grid: RectilinearGrid, name: str = "ATM") -> Component:
    """Return a toy slab atmosphere component factory instance."""

    def step(fields: Mapping[str, Any]) -> Mapping[str, Any]:
        temperature_2m = fields["temperature_2m"]
        sea_surface_temperature = fields.get(
            "sea_surface_temperature",
            _default_sea_surface_temperature(temperature_2m),
        )

        sensible_heat_flux, latent_heat_flux, updated_temperature_2m = _bulk_flux_step(
            temperature_2m,
            sea_surface_temperature,
        )
        u_velocity_10m, v_velocity_10m = _surface_wind_10m(
            grid.latitude,
            grid.longitude,
        )
        return {
            "u_velocity_10m": u_velocity_10m,
            "v_velocity_10m": v_velocity_10m,
            "sensible_heat_flux": sensible_heat_flux,
            "latent_heat_flux": latent_heat_flux,
            "temperature_2m": updated_temperature_2m,
        }

    return Component.from_step(
        name=name,
        grid=grid,
        step=step,
        inputs=_ATMOSPHERE_INPUTS,
        outputs=_ATMOSPHERE_OUTPUTS,
        defaults=_ATMOSPHERE_DEFAULT_FIELDS,
    )
