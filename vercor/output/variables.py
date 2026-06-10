"""Shared output-variable containers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class OutputVariable:
    """Array values with NetCDF dimension names and variable attributes."""

    dims: tuple[str, ...]
    values: Any
    attrs: Mapping[str, Any] = field(default_factory=dict)


__all__ = ["OutputVariable"]
