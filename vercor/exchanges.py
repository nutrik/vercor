"""Public exchange declarations for coupled component field transfers."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import typing as _typing
import vercor.fields as _fields
import vercor.regridding as _regridding
from vercor.fields import _normalize_field_items
from vercor.regridding import bilinear as _bilinear


@dataclass(frozen=True, init=False)
class Exchange:
    """Declare one stable route and the area meaning of its transferred fields.

    ``legacy`` preserves static scalar/vector behavior. Explicit ``intensive``
    quantities are covered-source means; ``source_flux`` and ``target_flux``
    are contributions per target grid-cell area using source or target coverage.
    Routes touching dynamic masks must choose an explicit quantity.
    """

    source: str
    target: str
    fields: Sequence[_fields.ExchangeField]
    route_id: str
    regridder_factory: _regridding.RegridderFactory
    quantity: _typing.Literal["legacy", "intensive", "source_flux", "target_flux"]

    def __init__(
        self,
        source: str,
        target: str,
        fields: Sequence[_fields.ExchangeField],
        *,
        route_id: str | None = None,
        regridder_factory: _regridding.RegridderFactory = _bilinear,
        quantity: _typing.Literal[
            "legacy", "intensive", "source_flux", "target_flux"
        ] = "legacy",
    ) -> None:
        """Create an exchange with an explicit or endpoint-derived route ID."""

        for label, value in (("source", source), ("target", target)):
            if not isinstance(value, str):
                raise TypeError(f"{label} must be a string")
            if not value.strip():
                raise ValueError(f"{label} must be a non-empty string")
        if route_id is not None and not isinstance(route_id, str):
            raise TypeError("route_id must be a string or None")
        normalized_route_id = f"{source}->{target}" if route_id is None else route_id
        if not normalized_route_id.strip():
            raise ValueError("route_id must be a non-empty string")
        if not callable(regridder_factory):
            raise TypeError("regridder_factory must be callable")
        if not isinstance(quantity, str):
            raise TypeError("quantity must be a string")
        if quantity not in ("legacy", "intensive", "source_flux", "target_flux"):
            raise ValueError(
                "quantity must be 'legacy', 'intensive', 'source_flux', or 'target_flux'"
            )

        object.__setattr__(self, "source", source)
        object.__setattr__(self, "target", target)
        object.__setattr__(self, "fields", _normalize_field_items(fields))
        object.__setattr__(self, "route_id", normalized_route_id)
        object.__setattr__(self, "regridder_factory", regridder_factory)
        object.__setattr__(self, "quantity", quantity)

    def __str__(self) -> str:
        return (
            f"{self.__class__.__name__}:\n"
            f"├── Route ID: {self.route_id}\n"
            f"├── Source component: {self.source}\n"
            f"└── Target component: {self.target}\n"
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(route_id={self.route_id!r}, "
            f"source={self.source!r}, target={self.target!r}, fields={self.fields!r})"
        )


__all__ = [
    "Exchange",
]
