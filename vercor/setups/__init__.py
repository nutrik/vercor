"""Public setup adapter factories for VerCOR examples and applications."""

from __future__ import annotations

from typing import Any

from vercor.setups._lazy_imports import (
    LazyExport,
    lazy_export_names,
    resolve_lazy_export,
)
from vercor.setups.coupler_helpers import (
    ExchangeSpec,
    add_exchange_specs,
    add_exchanges,
    build_coupler,
    build_exchanges,
)
from vercor.setups.slab import (
    make_slab_atmosphere,
    make_slab_land,
    make_slab_ocean,
    make_slab_seaice,
)

_LAZY_EXPORTS = {
    "make_camulator_gcm": LazyExport("external.camulator", "make_camulator_gcm"),
    "make_camulator_land": LazyExport(
        "external.camulator_land",
        "make_camulator_land",
    ),
    "make_era5_atmosphere": LazyExport("data.era5_atmosphere", "make_era5_atmosphere"),
    "make_era5_land": LazyExport("data.era5_land", "make_era5_land"),
    "make_era5_ocean": LazyExport("data.era5_ocean", "make_era5_ocean"),
    "make_erainterim_ocean": LazyExport(
        "data.erainterim_ocean",
        "make_erainterim_ocean",
    ),
    "make_jax_gcm": LazyExport("external.jax_gcm", "make_jax_gcm"),
    "make_jcm_land": LazyExport("data.jcm_land", "make_jcm_land"),
    "make_veros_gcm": LazyExport("external.veros_gcm", "make_veros_gcm"),
}

__all__ = [
    "ExchangeSpec",
    "add_exchange_specs",
    "add_exchanges",
    "build_coupler",
    "build_exchanges",
    "make_slab_atmosphere",
    "make_slab_land",
    "make_slab_ocean",
    "make_slab_seaice",
    *lazy_export_names(_LAZY_EXPORTS),
]


def __getattr__(name: str) -> Any:
    """Load optional setup factories only when requested."""

    return resolve_lazy_export(__name__, _LAZY_EXPORTS, name)


def __dir__() -> list[str]:
    """Return package exports without importing optional adapters."""

    return __all__
