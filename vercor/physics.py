"""Canonical traced physical constants used by VerCOR physics kernels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import jax
import jax.numpy as jnp

from vercor.dtypes import (
    DTypePolicy as _DTypePolicy,
    as_jax_real_array as _as_jax_real_array,
)
from vercor._pytree import PyTreeNodeMixin as _PyTreeNodeMixin
from vercor.types import RuntimeArray as _RuntimeArray

PhysicalValue = float | _RuntimeArray


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True, kw_only=True)
class PhysicalConstants(_PyTreeNodeMixin):
    """Immutable JAX-traced physical values shared by coupled components.

    Values use SI units except the documented native g/kg salinity reference,
    and intentionally carry no dtype policy. The enclosing
    :class:`vercor.runtime.RuntimeOptions` remains the sole owner of precision.
    Every field is a PyTree child so callers can differentiate physics outputs
    with respect to any configured constant in forward or reverse mode.

    Attributes:
        earth_radius: Mean Earth radius in m.
        gravity: Gravitational acceleration in m/s^2.
        air_density: Reference air density in kg/m^3.
        dry_air_gas_constant: Specific gas constant of dry air in J/(kg K).
        dry_air_specific_heat: Specific heat of dry air in J/(kg K).
        water_vapor_mass_ratio_correction: Dimensionless virtual-temperature
            correction for water vapor.
        reference_pressure: Reference pressure for potential temperature in Pa.
        dry_air_molecular_weight: Molecular weight of dry air in kg/kmol; this
            is a molar mass, not a specific gas constant.
        water_vapor_specific_heat: Specific heat of water vapor in J/(kg K).
        water_vapor_specific_heat_ratio_correction: Dimensionless moist-air
            heat-capacity correction.
        dry_air_kappa: Dimensionless dry-air gas-constant to heat-capacity ratio.
        ice_latent_heat_of_fusion: Sea-ice latent heat of fusion added to the
            vaporization heat in atmosphere-ice fluxes, in J/kg.
        universal_gas_constant: Universal molar gas constant in J/(kmol K),
            paired with molecular weight when computing density.
        ocean_minimum_wind_speed: Minimum wind speed over ocean in m/s.
        ice_minimum_wind_speed: Minimum wind speed over sea ice in m/s.
        von_karman_constant: Dimensionless von Karman constant.
        stefan_boltzmann_constant: Stefan-Boltzmann constant in W/(m^2 K^4).
        ocean_emissivity: Dimensionless long-wave ocean emissivity.
        ice_emissivity: Dimensionless long-wave sea-ice emissivity.
        snow_emissivity: Dimensionless long-wave snow emissivity.
        latent_heat_of_vaporization: Latent heat of vaporization in J/kg.
        freshwater_latent_heat_of_fusion: Freshwater latent heat of fusion in
            J/kg, distinct from the sea-ice flux value above.
        bulk_aerodynamic_resistance: Dimensionless bulk aerodynamic resistance.
        reference_height: Momentum reference height in m.
        air_temperature_reference_height: Air-temperature and humidity
            reference height in m.
        freshwater_density: Water-equivalent freshwater density in kg/m^3.
        surface_salinity_reference: Fixed virtual-salt-flux reference salinity
            in g/kg, matching the numerical salinity units used by Veros.
        precipitation_snow_temperature: Lower precipitation-phase threshold
            in K (272.77 K, or -0.38 degrees C).
        precipitation_rain_temperature: Upper precipitation-phase threshold
            in K (278.15 K, or 5 degrees C), greater than the snow threshold.
            Both thresholds come from the calibrated linear-transition model
            in Bedard-Therrien et al. (2025), doi:10.5194/hess-29-1135-2025.
            JCM phase diagnosis uses a sigmoid with the same midpoint and
            midpoint slope, so these are not hard rain/snow cutoffs.
    """

    pytree_children = (
        "earth_radius",
        "gravity",
        "air_density",
        "dry_air_gas_constant",
        "dry_air_specific_heat",
        "water_vapor_mass_ratio_correction",
        "reference_pressure",
        "dry_air_molecular_weight",
        "water_vapor_specific_heat",
        "water_vapor_specific_heat_ratio_correction",
        "dry_air_kappa",
        "ice_latent_heat_of_fusion",
        "universal_gas_constant",
        "ocean_minimum_wind_speed",
        "ice_minimum_wind_speed",
        "von_karman_constant",
        "stefan_boltzmann_constant",
        "ocean_emissivity",
        "ice_emissivity",
        "snow_emissivity",
        "latent_heat_of_vaporization",
        "freshwater_latent_heat_of_fusion",
        "bulk_aerodynamic_resistance",
        "reference_height",
        "air_temperature_reference_height",
        "freshwater_density",
        "surface_salinity_reference",
        "precipitation_snow_temperature",
        "precipitation_rain_temperature",
    )

    earth_radius: PhysicalValue = 6.371e6
    gravity: PhysicalValue = 9.81
    air_density: PhysicalValue = 1.3
    dry_air_gas_constant: PhysicalValue = 287.042
    dry_air_specific_heat: PhysicalValue = 1.00464e3
    water_vapor_mass_ratio_correction: PhysicalValue = 0.608
    reference_pressure: PhysicalValue = 1e5
    dry_air_molecular_weight: PhysicalValue = 28.966
    water_vapor_specific_heat: PhysicalValue = 1.810e3
    water_vapor_specific_heat_ratio_correction: PhysicalValue = 0.802
    dry_air_kappa: PhysicalValue = 0.286
    ice_latent_heat_of_fusion: PhysicalValue = 3.337e5
    universal_gas_constant: PhysicalValue = 8314.47
    ocean_minimum_wind_speed: PhysicalValue = 0.5
    ice_minimum_wind_speed: PhysicalValue = 1.0
    von_karman_constant: PhysicalValue = 0.4
    stefan_boltzmann_constant: PhysicalValue = 5.67e-8
    ocean_emissivity: PhysicalValue = 0.97
    ice_emissivity: PhysicalValue = 0.97
    snow_emissivity: PhysicalValue = 0.99
    latent_heat_of_vaporization: PhysicalValue = 2.501e6
    freshwater_latent_heat_of_fusion: PhysicalValue = 3.34e5
    bulk_aerodynamic_resistance: PhysicalValue = 0.1
    reference_height: PhysicalValue = 10.0
    air_temperature_reference_height: PhysicalValue = 2.0
    freshwater_density: PhysicalValue = 1000.0
    surface_salinity_reference: PhysicalValue = 34.7
    precipitation_snow_temperature: PhysicalValue = 272.77
    precipitation_rain_temperature: PhysicalValue = 278.15

    def __post_init__(self) -> None:
        """Defensively normalize physical leaves while retaining JAX tracers."""

        for name in self.pytree_children:
            value = getattr(self, name)
            ndim = getattr(value, "ndim", None)
            if ndim is not None and ndim != 0:
                raise TypeError(f"{name} must be a scalar physical value")
            dtype = getattr(value, "dtype", None)
            if dtype is not None and not (
                jnp.issubdtype(dtype, jnp.integer)
                or jnp.issubdtype(dtype, jnp.floating)
            ):
                raise TypeError(f"{name} must be a real numeric scalar")
            item = getattr(value, "item", None)
            if (
                ndim is not None
                and not isinstance(value, (jax.Array, jax.core.Tracer))
                and callable(item)
            ):
                object.__setattr__(self, name, item())
            elif ndim is None and (
                isinstance(value, bool) or not isinstance(value, (int, float))
            ):
                raise TypeError(f"{name} must be a real numeric scalar")


def _physical_constants_for_dtype(
    constants: PhysicalConstants,
    dtype: _DTypePolicy,
) -> PhysicalConstants:
    """Cast traced constants at the runtime precision boundary."""

    return cast(
        PhysicalConstants,
        jax.tree_util.tree_map(
            lambda value: _as_jax_real_array(value, dtype),
            constants,
        ),
    )


__all__ = ["PhysicalConstants"]
