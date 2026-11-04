"""Bundled exchange field recipes."""

from __future__ import annotations

from vercor.fields import ExchangeField as _ExchangeField, vector as _vector

ATMOSPHERE_TO_OCEAN_BULK_FORCING_FIELDS: tuple[_ExchangeField, ...] = (
    _vector("air_u_velocity", "air_v_velocity"),
    "air_specific_humidity",
    "model_level_height",
    "air_density",
    "air_potential_temperature",
    "air_temperature",
    "net_shortwave_radiation_flux",
    "downward_longwave_radiation_flux",
    "rain_rate",
    "snow_rate",
)
"""Colocated atmospheric state, radiation and precipitation for ocean bulk fluxes."""


OBSERVED_10M_ATMOSPHERE_TO_SEAICE_FORCING_FIELDS: tuple[_ExchangeField, ...] = (
    _vector("air_u_velocity", "air_v_velocity"),
    _vector("u_velocity_10m", "v_velocity_10m"),
    "air_specific_humidity",
    "model_level_height",
    "air_density",
    "air_potential_temperature",
    "air_temperature",
    "surface_pressure",
    "net_shortwave_radiation_flux",
    "downward_shortwave_radiation_flux",
    "downward_longwave_radiation_flux",
    "rain_rate",
    "snow_rate",
)
"""Colocated bulk inputs and separate observed 10 m winds for sea-ice forcing."""

MODEL_LEVEL_ATMOSPHERE_TO_SEAICE_FORCING_FIELDS: tuple[_ExchangeField, ...] = (
    _vector("air_u_velocity", "air_v_velocity"),
    "air_specific_humidity",
    "model_level_height",
    "air_density",
    "air_potential_temperature",
    "air_temperature",
    "surface_pressure",
    "downward_shortwave_radiation_flux",
    "downward_longwave_radiation_flux",
    "rain_rate",
    "snow_rate",
    "open_water_albedo",
)
"""Model-level air and raw radiation; derive actual 10 m winds with the bulk solve."""


OCEAN_TO_SEAICE_FORCING_FIELDS: tuple[_ExchangeField, ...] = (
    "sea_surface_temperature",
    "sea_surface_salinity",
    _vector("ocean_u_velocity", "ocean_v_velocity"),
    "sea_surface_height",
)
"""Tracer surfaces and canonical east/north face velocities on the shared ocean grid."""


SEAICE_TO_OCEAN_TOTAL_FLUX_FIELDS: tuple[_ExchangeField, ...] = (
    "sea_surface_heat_flux",
    "sea_surface_mass_flux",
    "sea_surface_salt_mass_flux",
    _vector("ocean_u_stress", "ocean_v_stress"),
)
"""Total-area interval means: downward W/m2, kg/m2/s, kg salt/m2/s and Pa."""

SEAICE_TO_ATMOSPHERE_SURFACE_FIELDS: tuple[_ExchangeField, ...] = (
    "ice_surface_temperature",
    "ice_surface_albedo",
)
"""Current per-ice-area properties; weight by source ice coverage before remapping."""

SEAICE_COVERAGE_FIELDS: tuple[_ExchangeField, ...] = ("ice_fraction",)
"""Ice concentration; use binary wet support to avoid a second concentration factor."""

OCEAN_TO_ATMOSPHERE_OPEN_WATER_FIELDS: tuple[_ExchangeField, ...] = (
    "open_water_temperature_contribution",
)
"""Post-ocean (1-A)*SST; use binary source-flux support before conservative remapping."""


ATMOSPHERE_TO_OCEAN_STATE_FIELDS: tuple[_ExchangeField, ...] = (
    _vector("air_u_velocity", "air_v_velocity"),
    "air_specific_humidity",
    "model_level_height",
    "air_density",
    "air_potential_temperature",
    "air_temperature",
)
"""Atmosphere state displayed through explicit data-gallery diagnostic imports."""


ATMOSPHERE_TO_OCEAN_RADIATION_FIELDS: tuple[_ExchangeField, ...] = (
    "net_shortwave_radiation_flux",
    "downward_longwave_radiation_flux",
)
"""Atmosphere radiation displayed through explicit data-gallery diagnostic imports."""


ATMOSPHERE_TO_LAND_STATE_FIELDS: tuple[_ExchangeField, ...] = (
    "air_specific_humidity",
    "model_level_height",
    "air_potential_temperature",
)
"""Atmosphere state displayed by the land data-gallery diagnostics."""


ATMOSPHERE_TO_LAND_RADIATION_FIELDS: tuple[_ExchangeField, ...] = (
    "net_shortwave_radiation_flux",
    "downward_longwave_radiation_flux",
)
"""Atmosphere radiation displayed by the land data-gallery diagnostics."""


LAND_TO_ATMOSPHERE_STATE_FIELDS: tuple[_ExchangeField, ...] = (
    "soil_moisture",
    "land_surface_temperature",
)
"""Land temperature and soil moisture imported by atmosphere adapters."""


LAND_TO_ATMOSPHERE_SURFACE_FIELDS: tuple[_ExchangeField, ...] = (
    "land_surface_temperature",
)
"""Land surface fields imported by atmosphere adapters."""


OCEAN_TO_ATMOSPHERE_SURFACE_FIELDS: tuple[_ExchangeField, ...] = (
    "sea_surface_temperature",
)
"""Ocean surface fields imported by atmosphere adapters."""


ATMOSPHERE_TO_OCEAN_TURBULENT_HEAT_FLUX_FIELDS: tuple[_ExchangeField, ...] = (
    "latent_heat_flux",
    "sensible_heat_flux",
)
"""Atmospheric turbulent heat fluxes consumed by ocean surface components."""


ATMOSPHERE_TO_LAND_LATENT_HEAT_FLUX_FIELDS: tuple[_ExchangeField, ...] = (
    "latent_heat_flux",
)
"""Atmospheric latent heat flux consumed by land surface components."""


OCEAN_TO_SEAICE_SURFACE_FIELDS: tuple[_ExchangeField, ...] = (
    "sea_surface_temperature",
)
"""Ocean surface fields imported by sea-ice components."""


__all__ = [
    "ATMOSPHERE_TO_LAND_RADIATION_FIELDS",
    "ATMOSPHERE_TO_LAND_STATE_FIELDS",
    "ATMOSPHERE_TO_OCEAN_RADIATION_FIELDS",
    "ATMOSPHERE_TO_OCEAN_STATE_FIELDS",
    "ATMOSPHERE_TO_OCEAN_BULK_FORCING_FIELDS",
    "OBSERVED_10M_ATMOSPHERE_TO_SEAICE_FORCING_FIELDS",
    "MODEL_LEVEL_ATMOSPHERE_TO_SEAICE_FORCING_FIELDS",
    "LAND_TO_ATMOSPHERE_STATE_FIELDS",
    "LAND_TO_ATMOSPHERE_SURFACE_FIELDS",
    "OCEAN_TO_ATMOSPHERE_SURFACE_FIELDS",
    "OCEAN_TO_ATMOSPHERE_OPEN_WATER_FIELDS",
    "OCEAN_TO_SEAICE_SURFACE_FIELDS",
    "OCEAN_TO_SEAICE_FORCING_FIELDS",
    "ATMOSPHERE_TO_LAND_LATENT_HEAT_FLUX_FIELDS",
    "ATMOSPHERE_TO_OCEAN_TURBULENT_HEAT_FLUX_FIELDS",
    "SEAICE_TO_OCEAN_TOTAL_FLUX_FIELDS",
    "SEAICE_TO_ATMOSPHERE_SURFACE_FIELDS",
    "SEAICE_COVERAGE_FIELDS",
]
