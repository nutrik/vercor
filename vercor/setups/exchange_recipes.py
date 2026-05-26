from __future__ import annotations

from typing import TypeAlias

ExchangeField: TypeAlias = str | tuple[str, str]


ATMOSPHERE_TO_VEROS_FORCING_FIELDS: tuple[ExchangeField, ...] = (
    ("u_velocity", "v_velocity"),
    "specific_humidity",
    "model_level_height",
    "density",
    "potential_temperature",
    "temperature",
    "net_shortwave_radiation_flux",
    "downward_longwave_radiation_flux",
)
"""Stable atmosphere-to-Veros forcing fields used by coupled setup scripts."""


ATMOSPHERE_TO_DATA_OCEAN_FIELDS: tuple[ExchangeField, ...] = (
    ("u_velocity", "v_velocity"),
    "specific_humidity",
    "temperature",
    "model_level_height",
    "net_shortwave_radiation_flux",
    "downward_longwave_radiation_flux",
)
"""Atmosphere fields imported by data-backed or slab ocean setup scripts."""


ATMOSPHERE_TO_OCEAN_STATE_FIELDS: tuple[ExchangeField, ...] = (
    ("u_velocity", "v_velocity"),
    "specific_humidity",
    "model_level_height",
    "density",
    "potential_temperature",
    "temperature",
)
"""Atmosphere state fields imported by ocean examples."""


ATMOSPHERE_TO_OCEAN_RADIATION_FIELDS: tuple[ExchangeField, ...] = (
    "net_shortwave_radiation_flux",
    "downward_longwave_radiation_flux",
)
"""Atmosphere radiation fields imported by ocean examples."""


ATMOSPHERE_TO_LAND_STATE_FIELDS: tuple[ExchangeField, ...] = (
    "specific_humidity",
    "model_level_height",
    "potential_temperature",
)
"""Atmosphere state fields imported by land data setup examples."""


ATMOSPHERE_TO_LAND_BASIC_FIELDS: tuple[ExchangeField, ...] = (
    "temperature",
    "specific_humidity",
)
"""Basic atmosphere near-surface fields imported by land examples."""


ATMOSPHERE_TO_LAND_RADIATION_FIELDS: tuple[ExchangeField, ...] = (
    "net_shortwave_radiation_flux",
    "downward_longwave_radiation_flux",
)
"""Atmosphere radiation fields imported by land setup examples."""


JCM_LAND_TO_ATMOSPHERE_FIELDS: tuple[ExchangeField, ...] = (
    "soil_moisture",
    "land_surface_temperature",
)
"""JCM land fields imported by the atmosphere adapter."""


LAND_TO_ATMOSPHERE_SURFACE_FIELDS: tuple[ExchangeField, ...] = (
    "land_surface_temperature",
)
"""Land surface fields imported by atmosphere adapters."""


OCEAN_TO_ATMOSPHERE_SURFACE_FIELDS: tuple[ExchangeField, ...] = (
    "sea_surface_temperature",
)
"""Ocean surface fields imported by atmosphere adapters."""


ATMOSPHERE_TO_JCM_LAND_FLUX_FIELDS: tuple[ExchangeField, ...] = (
    "latent_heat_flux",
    "sensible_heat_flux",
)
"""Atmosphere flux fields imported by JCM land."""


SLAB_ATMOSPHERE_TO_OCEAN_FLUX_FIELDS: tuple[ExchangeField, ...] = (
    "latent_heat_flux",
    "sensible_heat_flux",
)
"""Atmosphere flux fields imported by slab ocean components."""


SLAB_ATMOSPHERE_TO_LAND_FLUX_FIELDS: tuple[ExchangeField, ...] = ("latent_heat_flux",)
"""Atmosphere flux fields imported by slab land components."""


SLAB_ATMOSPHERE_TO_OCEAN_FIELDS: tuple[ExchangeField, ...] = (
    ("u_velocity_10m", "v_velocity_10m"),
    "sensible_heat_flux",
    "latent_heat_flux",
)
"""Toy slab atmosphere fields imported by slab ocean examples."""


LAND_TO_ATMOSPHERE_SOIL_FIELDS: tuple[ExchangeField, ...] = ("soil_moisture",)
"""Land soil fields imported by slab atmosphere examples."""


OCEAN_TO_SEAICE_SURFACE_FIELDS: tuple[ExchangeField, ...] = ("sea_surface_temperature",)
"""Ocean surface fields imported by sea-ice components."""


SEAICE_TO_OCEAN_FIELDS: tuple[ExchangeField, ...] = ("ice_fraction",)
"""Sea-ice fields imported by ocean components."""
