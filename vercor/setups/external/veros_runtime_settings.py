"""Explicit Veros host-runtime configuration used by the Veros adapter."""

from __future__ import annotations

from typing import Any


def _set_runtime_setting(runtime_settings: Any, name: str, value: Any) -> None:
    """Set one Veros runtime setting, tolerating already-applied locked values."""

    try:
        setattr(runtime_settings, name, value)
    except RuntimeError:
        if getattr(runtime_settings, name, None) == value:
            return
        raise


def configure_veros_runtime() -> None:
    """Configure Veros for the NumPy host runtime used by ``VerosGCM``."""

    from veros import runtime_settings  # type: ignore[import]

    _set_runtime_setting(runtime_settings, "backend", "numpy")
    _set_runtime_setting(runtime_settings, "force_overwrite", True)
    _set_runtime_setting(runtime_settings, "diskless_mode", True)
