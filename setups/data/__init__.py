"""Data setup adapters with optional dependencies loaded on demand."""

from __future__ import annotations

from typing import Any

from setups._lazy_imports import LazyExport, lazy_export_names, resolve_lazy_export

_LAZY_EXPORTS = {
    "camulator_land": LazyExport("camulator_land"),
    "era5_atmosphere": LazyExport("era5_atmosphere"),
    "era5_land": LazyExport("era5_land"),
    "era5_ocean": LazyExport("era5_ocean"),
    "erainterim_ocean": LazyExport("erainterim_ocean"),
    "jcm_land": LazyExport("jcm_land"),
    "make_camulator_land": LazyExport("camulator_land", "make_camulator_land"),
    "make_era5_atmosphere": LazyExport("era5_atmosphere", "make_era5_atmosphere"),
    "make_era5_land": LazyExport("era5_land", "make_era5_land"),
    "make_era5_ocean": LazyExport("era5_ocean", "make_era5_ocean"),
    "make_erainterim_ocean": LazyExport(
        "erainterim_ocean",
        "make_erainterim_ocean",
    ),
    "make_jcm_land": LazyExport("jcm_land", "make_jcm_land"),
}

__all__ = lazy_export_names(_LAZY_EXPORTS)


def __getattr__(name: str) -> Any:
    """Load data setup adapters only when their exports are requested."""

    return resolve_lazy_export(__name__, _LAZY_EXPORTS, name)


def __dir__() -> list[str]:
    """Return package exports without importing optional adapters."""

    return __all__
