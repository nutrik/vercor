from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import TypeAlias

from vercor._field_names import freeze_name_sequence as _freeze_name_sequence


@dataclass(frozen=True)
class VectorField:
    """Two named runtime fields that should be transferred as one vector."""

    u: str
    v: str

    def __post_init__(self) -> None:
        """Validate vector component names."""

        if not isinstance(self.u, str) or not isinstance(self.v, str):
            raise TypeError("VectorField component names must be strings")
        if not self.u or not self.v:
            raise ValueError("VectorField component names must be non-empty")

    def __iter__(self) -> Iterable[str]:
        """Iterate over vector component names in transfer order."""

        return iter((self.u, self.v))


ExchangeField: TypeAlias = str | VectorField


COMMON_FIELD_NAMES: tuple[str, ...] = (
    "specific_humidity",
    "temperature",
    "temperature_2m",
    "potential_temperature",
    "sea_surface_temperature",
    "open_water_temperature_contribution",
    "land_surface_temperature",
    "model_level_height",
    "u_velocity",
    "v_velocity",
    "u_velocity_10m",
    "v_velocity_10m",
    "surface_pressure",
    "pressure",
    "density",
    "ice_fraction",
    "ice_surface_temperature",
    "ice_surface_albedo",
    "marine_surface_albedo",
    "marine_ice_fraction",
    "open_water_albedo",
    "downward_shortwave_radiation_flux",
    "land_sensible_heat_flux",
    "land_latent_heat_flux",
    "soil_moisture",
    "sensible_heat_flux",
    "latent_heat_flux",
    "net_shortwave_radiation_flux",
    "downward_longwave_radiation_flux",
    "rain_rate",
    "snow_rate",
)


def vector(u: str, v: str) -> VectorField:
    """Return a vector-field declaration for exchange recipes."""

    return VectorField(u, v)


def _normalize_field_items(
    fields: Sequence[ExchangeField],
) -> tuple[ExchangeField, ...]:
    """Validate and freeze exchange field declarations."""

    normalized: list[ExchangeField] = []
    for field in _freeze_name_sequence(fields, label="Exchange.fields"):
        if isinstance(field, VectorField):
            normalized.append(field)
            continue
        if isinstance(field, str):
            if not field:
                raise ValueError("Exchange field names must be non-empty")
            normalized.append(field)
            continue
        if isinstance(field, tuple):
            raise TypeError(
                "Tuple vector field declarations are unsupported; use "
                "vercor.fields.vector('u_field', 'v_field') to create a VectorField."
            )
        raise TypeError(
            "Exchange fields must be strings or VectorField instances, got "
            f"{type(field).__name__}."
        )
    return tuple(normalized)


def _flatten_field_items(fields: Sequence[ExchangeField]) -> list[str]:
    """Return scalar field names from scalar and vector field declarations."""

    flattened: list[str] = []
    for field in _normalize_field_items(fields):
        if isinstance(field, VectorField):
            flattened.extend((field.u, field.v))
        else:
            flattened.append(field)
    return flattened


__all__ = [
    "COMMON_FIELD_NAMES",
    "ExchangeField",
    "VectorField",
    "vector",
]
