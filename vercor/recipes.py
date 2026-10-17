"""Bundled exchange field recipes."""

from __future__ import annotations

from vercor.fields import ExchangeField as _ExchangeField, vector as _vector

ATMOSPHERE_TO_VEROS_FORCING_FIELDS: tuple[_ExchangeField, ...] = (
    _vector("u_velocity", "v_velocity"),
    "specific_humidity",
    "model_level_height",
    "density",
    "potential_temperature",
    "temperature",
    "net_shortwave_radiation_flux",
    "downward_longwave_radiation_flux",
    "rain_rate",
    "snow_rate",
)
"""Stable atmosphere-to-Veros forcing fields used by coupled setup scripts."""


ATMOSPHERE_TO_VERIS_FORCING_FIELDS: tuple[_ExchangeField, ...] = (
    _vector("u_velocity", "v_velocity"),
    _vector("u_velocity_10m", "v_velocity_10m"),
    "specific_humidity",
    "model_level_height",
    "density",
    "potential_temperature",
    "temperature",
    "surface_pressure",
    "net_shortwave_radiation_flux",
    "downward_shortwave_radiation_flux",
    "downward_longwave_radiation_flux",
    "rain_rate",
    "snow_rate",
)
"""Colocated bulk inputs and separate observed 10 m winds for native Veris."""


OCEAN_TO_VERIS_FORCING_FIELDS: tuple[_ExchangeField, ...] = (
    "sea_surface_temperature",
    "sea_surface_salinity",
    _vector("ocean_u_velocity", "ocean_v_velocity"),
    "sea_surface_height",
)
"""Prescribed tracer surfaces and native face velocities on the shared ocean grid."""


VERIS_TO_OCEAN_FLUX_FIELDS: tuple[_ExchangeField, ...] = (
    "sea_surface_heat_flux",
    "sea_surface_mass_flux",
    "sea_surface_salt_mass_flux",
    _vector("ocean_u_stress", "ocean_v_stress"),
)
"""Total-area interval means: downward W/m2, kg/m2/s, kg salt/m2/s and Pa."""


ATMOSPHERE_TO_DATA_OCEAN_FIELDS: tuple[_ExchangeField, ...] = (
    _vector("u_velocity", "v_velocity"),
    "specific_humidity",
    "temperature",
    "model_level_height",
    "net_shortwave_radiation_flux",
    "downward_longwave_radiation_flux",
)
"""Atmosphere fields imported by data-backed ocean setup scripts."""


ATMOSPHERE_TO_OCEAN_STATE_FIELDS: tuple[_ExchangeField, ...] = (
    _vector("u_velocity", "v_velocity"),
    "specific_humidity",
    "model_level_height",
    "density",
    "potential_temperature",
    "temperature",
)
"""Atmosphere state fields imported by ocean examples."""


ATMOSPHERE_TO_OCEAN_RADIATION_FIELDS: tuple[_ExchangeField, ...] = (
    "net_shortwave_radiation_flux",
    "downward_longwave_radiation_flux",
)
"""Atmosphere radiation fields imported by ocean examples."""


ATMOSPHERE_TO_LAND_STATE_FIELDS: tuple[_ExchangeField, ...] = (
    "specific_humidity",
    "model_level_height",
    "potential_temperature",
)
"""Atmosphere state fields imported by land data setup examples."""


ATMOSPHERE_TO_LAND_BASIC_FIELDS: tuple[_ExchangeField, ...] = (
    "temperature",
    "specific_humidity",
)
"""Basic atmosphere near-surface fields imported by land examples."""


ATMOSPHERE_TO_LAND_RADIATION_FIELDS: tuple[_ExchangeField, ...] = (
    "net_shortwave_radiation_flux",
    "downward_longwave_radiation_flux",
)
"""Atmosphere radiation fields imported by land setup examples."""


JCM_LAND_TO_ATMOSPHERE_FIELDS: tuple[_ExchangeField, ...] = (
    "soil_moisture",
    "land_surface_temperature",
)
"""JCM land fields imported by the atmosphere adapter."""


LAND_TO_ATMOSPHERE_SURFACE_FIELDS: tuple[_ExchangeField, ...] = (
    "land_surface_temperature",
)
"""Land surface fields imported by atmosphere adapters."""


OCEAN_TO_ATMOSPHERE_SURFACE_FIELDS: tuple[_ExchangeField, ...] = (
    "sea_surface_temperature",
)
"""Ocean surface fields imported by atmosphere adapters."""


ATMOSPHERE_TO_JCM_LAND_FLUX_FIELDS: tuple[_ExchangeField, ...] = (
    "latent_heat_flux",
    "sensible_heat_flux",
)
"""Atmosphere flux fields imported by JCM land."""


SLAB_ATMOSPHERE_TO_OCEAN_FLUX_FIELDS: tuple[_ExchangeField, ...] = (
    "latent_heat_flux",
    "sensible_heat_flux",
)
"""Atmosphere flux fields imported by slab ocean components."""


JCM_ATMOSPHERE_TO_SLAB_OCEAN_FIELDS: tuple[_ExchangeField, ...] = (
    *ATMOSPHERE_TO_DATA_OCEAN_FIELDS,
    *SLAB_ATMOSPHERE_TO_OCEAN_FLUX_FIELDS,
)
"""JCM atmosphere fields imported by slab ocean setup scripts."""


SLAB_ATMOSPHERE_TO_LAND_FLUX_FIELDS: tuple[_ExchangeField, ...] = ("latent_heat_flux",)
"""Atmosphere flux fields imported by slab land components."""


SLAB_ATMOSPHERE_TO_OCEAN_FIELDS: tuple[_ExchangeField, ...] = (
    _vector("u_velocity_10m", "v_velocity_10m"),
    "sensible_heat_flux",
    "latent_heat_flux",
)
"""Toy slab atmosphere fields imported by slab ocean examples."""


LAND_TO_ATMOSPHERE_SOIL_FIELDS: tuple[_ExchangeField, ...] = ("soil_moisture",)
"""Land soil fields imported by slab atmosphere examples."""


OCEAN_TO_SEAICE_SURFACE_FIELDS: tuple[_ExchangeField, ...] = (
    "sea_surface_temperature",
)
"""Ocean surface fields imported by sea-ice components."""


SEAICE_TO_OCEAN_FIELDS: tuple[_ExchangeField, ...] = ("ice_fraction",)
"""Sea-ice fields imported by ocean components."""

__all__ = [
    "ATMOSPHERE_TO_DATA_OCEAN_FIELDS",
    "ATMOSPHERE_TO_JCM_LAND_FLUX_FIELDS",
    "ATMOSPHERE_TO_LAND_BASIC_FIELDS",
    "ATMOSPHERE_TO_LAND_RADIATION_FIELDS",
    "ATMOSPHERE_TO_LAND_STATE_FIELDS",
    "ATMOSPHERE_TO_OCEAN_RADIATION_FIELDS",
    "ATMOSPHERE_TO_OCEAN_STATE_FIELDS",
    "ATMOSPHERE_TO_VEROS_FORCING_FIELDS",
    "ATMOSPHERE_TO_VERIS_FORCING_FIELDS",
    "JCM_ATMOSPHERE_TO_SLAB_OCEAN_FIELDS",
    "JCM_LAND_TO_ATMOSPHERE_FIELDS",
    "LAND_TO_ATMOSPHERE_SOIL_FIELDS",
    "LAND_TO_ATMOSPHERE_SURFACE_FIELDS",
    "OCEAN_TO_ATMOSPHERE_SURFACE_FIELDS",
    "OCEAN_TO_SEAICE_SURFACE_FIELDS",
    "OCEAN_TO_VERIS_FORCING_FIELDS",
    "SEAICE_TO_OCEAN_FIELDS",
    "SLAB_ATMOSPHERE_TO_LAND_FLUX_FIELDS",
    "SLAB_ATMOSPHERE_TO_OCEAN_FIELDS",
    "SLAB_ATMOSPHERE_TO_OCEAN_FLUX_FIELDS",
    "VERIS_TO_OCEAN_FLUX_FIELDS",
]
