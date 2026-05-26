from __future__ import annotations

from typing import Any, TypeAlias

PhysicalConstantSetting: TypeAlias = tuple[Any, str, str]


PHYSICAL_CONSTANT_SETTINGS: dict[str, PhysicalConstantSetting] = {
    "earth_radius": (6.371e6, "Earth radius", "m"),
    "gravity": (9.81, "Acceleration due to gravity", "m/s^2"),
    "rhoAir": (1.3, "Density of air", "kg/m^3"),
    "rdair": (287.042, "Dry air gas constant", "J/(K*kg)"),
    "cpdair": (
        1.00464e3,
        "Specific heat capacity of dry air",
        "J/(kg*K)",
    ),
    "zvir": (
        0.608,
        "Dry-air water-vapor molecular mass ratio correction",
        "-",
    ),
    "p0": (1e5, "Reference pressure for potential temperature", "Pa"),
    "mwdair": (28.966, "Molecular weight of dry air", "kg/kmole"),
    "cpwv": (1.810e3, "Specific heat of water vapor", "J/(kg*K)"),
    "cpvir": (0.802, "Specific heat of vaporization ratio correction", "-"),
    "cappa": (0.286, "Dry air gas constant over heat capacity", "-"),
    "latice": (3.337e5, "Latent heat of fusion", "J/kg"),
    "rgas": (8314.47, "Ideal gas constant", "J/(K*kmole)"),
    "umin_ocean": (
        0.5,
        "Minimum atmospheric wind speed over ocean surface",
        "m/s",
    ),
    "umin_ice": (
        1.0,
        "Minimum atmospheric wind speed over ice surface",
        "m/s",
    ),
    "karman": (0.4, "von Karman constant", "-"),
    "stefBoltz": (5.67e-8, "Stefan-Boltzmann constant", "W/(m^2*K^4)"),
    "ocean_emissivity": (0.97, "Long-wave emissivity of ocean surface", "-"),
    "ice_emissivity": (0.97, "Long-wave emissivity of sea ice", "-"),
    "snow_emissivity": (0.99, "Long-wave emissivity of snow", "-"),
    "latvap": (2.501e6, "Latent heat of vaporization", "J/kg"),
    "latfresh": (3.34e5, "Latent heat of fusion", "J/kg"),
    "gamma_blk": (0.1, "Bulk aerodynamic resistance", "-"),
    "zref": (10.0, "Reference height", "m"),
    "ztref": (2.0, "Reference height for air temperature", "m"),
}
