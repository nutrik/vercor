"""Output helpers for runtime fields and GCM period-average files."""

from __future__ import annotations

from typing import Any

_RUNTIME_EXPORTS = {
    "output_masks_for_component",
    "write_coupler_runtime_outputs",
    "write_runtime_component_view_to_netcdf",
}

__all__ = [
    "output_masks_for_component",
    "write_coupler_runtime_outputs",
    "write_runtime_component_view_to_netcdf",
]


def __getattr__(name: str) -> Any:
    """Load runtime-output helpers only when top-level exports are requested."""

    if name in _RUNTIME_EXPORTS:
        from vercor.output import runtime as _runtime

        return getattr(_runtime, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Return package exports without importing runtime internals."""

    return sorted({*globals(), *__all__})
