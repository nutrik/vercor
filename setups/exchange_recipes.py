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


JCM_LAND_TO_ATMOSPHERE_FIELDS: tuple[ExchangeField, ...] = (
    "soil_moisture",
    "land_surface_temperature",
)
"""JCM land fields imported by the atmosphere adapter."""


ATMOSPHERE_TO_JCM_LAND_FLUX_FIELDS: tuple[ExchangeField, ...] = (
    "latent_heat_flux",
    "sensible_heat_flux",
)
"""Atmosphere flux fields imported by JCM land."""
