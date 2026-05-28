"""Lightweight CAMulator runtime field contract ownership."""

from __future__ import annotations

CAMULATOR_RUNTIME_FIELD_NAMES = (
    "specific_humidity",
    "net_shortwave_radiation_flux",
    "downward_longwave_radiation_flux",
    "sea_surface_temperature",
    "land_surface_temperature",
    "u_velocity",
    "v_velocity",
    "temperature",
    "potential_temperature",
    "density",
    "latent_heat_flux",
    "sensible_heat_flux",
    "model_level_height",
    "total_surface_temperature",
    "temperature_3d",
    "specific_humidity_3d",
)


def camulator_runtime_field_defaults(value: float = 0.0) -> dict[str, float]:
    """Return scalar defaults for all CAMulator runtime exchange fields."""

    return {field_name: value for field_name in CAMULATOR_RUNTIME_FIELD_NAMES}


__all__ = [
    "CAMULATOR_RUNTIME_FIELD_NAMES",
    "camulator_runtime_field_defaults",
]
