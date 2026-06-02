"""External setup adapters with optional dependencies loaded on demand."""

from __future__ import annotations

from typing import Any

from vercor.setups._lazy_imports import (
    LazyExport,
    lazy_export_names,
    resolve_lazy_export,
)

_LAZY_EXPORTS = {
    "camulator_land": LazyExport("camulator_land"),
    "jax_gcm": LazyExport("jax_gcm"),
    "make_camulator_gcm": LazyExport("camulator", "make_camulator_gcm"),
    "make_camulator_land": LazyExport("camulator_land", "make_camulator_land"),
    "make_jax_gcm": LazyExport("jax_gcm", "make_jax_gcm"),
    "make_veros_gcm": LazyExport("veros_gcm", "make_veros_gcm"),
}

__all__ = lazy_export_names(_LAZY_EXPORTS)


def __getattr__(name: str) -> Any:
    """Load external setup adapters only when their exports are requested."""

    return resolve_lazy_export(__name__, _LAZY_EXPORTS, name)


def __dir__() -> list[str]:
    """Return package exports without importing optional adapters."""

    return __all__
