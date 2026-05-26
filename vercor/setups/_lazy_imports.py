"""Lazy import helpers for setup package export surfaces."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Any, Mapping


@dataclass(frozen=True)
class LazyExport:
    """Describe one lazily resolved package export."""

    module: str
    attribute: str | None = None


def resolve_lazy_export(
    package: str,
    exports: Mapping[str, LazyExport],
    name: str,
) -> Any:
    """Resolve one lazy package export or raise ``AttributeError``."""

    try:
        export = exports[name]
    except KeyError as error:
        raise AttributeError(f"module {package!r} has no attribute {name!r}") from error

    module = import_module(f"{package}.{export.module}")
    if export.attribute is None:
        return module
    return getattr(module, export.attribute)


def lazy_export_names(exports: Mapping[str, LazyExport]) -> list[str]:
    """Return stable export names for ``__all__`` and ``__dir__``."""

    return sorted(exports)
