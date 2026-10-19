"""Public topology policy contracts for optional exchange-map patching."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Literal, Protocol, runtime_checkable

from vercor.components import Component as _Component
from vercor.exchanges import Exchange as _Exchange
from vercor.jax_logging import LoggerLike as _LoggerLike
from vercor.types import RuntimeArray as _RuntimeArray


@dataclass(frozen=True)
class TopologyContext:
    """Public read-only context supplied to topology policies."""

    components: Mapping[str, _Component]
    exchanges: Sequence[_Exchange]
    logger: _LoggerLike


@dataclass(frozen=True)
class ExchangeTopologyPatch:
    """Static binary route restrictions keyed by stable exchange route IDs.

    Fractional coverage belongs to component declarations and is resolved by
    the runtime; topology can only restrict target computational support.
    """

    binary_masks: Mapping[str, _RuntimeArray] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Freeze caller-provided patch mappings."""

        object.__setattr__(
            self, "binary_masks", MappingProxyType(dict(self.binary_masks))
        )


@runtime_checkable
class TopologyPolicy(Protocol):
    """Public protocol for setup-specific exchange-topology customization."""

    def build(self, context: TopologyContext) -> ExchangeTopologyPatch:
        """Return exchange mask updates for this topology."""
        ...


@dataclass(frozen=True)
class SurfaceMaskPolicy:
    """Policy for the bundled atmosphere/ocean/land surface-mask topology."""

    mode: Literal["auto", "required", "disabled"] = "auto"
    atmosphere: str = "ATM"
    ocean: str = "OCN"
    land: str = "LND"

    def __post_init__(self) -> None:
        """Validate surface-mask policy values."""

        if self.mode not in ("auto", "required", "disabled"):
            raise ValueError("mode must be one of 'auto', 'required', 'disabled'")
        for role, name in (
            ("atmosphere", self.atmosphere),
            ("ocean", self.ocean),
            ("land", self.land),
        ):
            if not isinstance(name, str) or not name:
                raise ValueError(f"{role} component name must be a non-empty string")

    def build(self, context: TopologyContext) -> ExchangeTopologyPatch:
        """Return exchange mask updates for bundled surface exchanges."""

        from vercor._runtime.surface_masks import (
            build_surface_mask_topology_patch,
            should_apply_surface_mask_policy,
        )

        if not should_apply_surface_mask_policy(
            context.components,
            context.exchanges,
            self,
        ):
            return ExchangeTopologyPatch()

        return build_surface_mask_topology_patch(context, self)


__all__ = [
    "ExchangeTopologyPatch",
    "SurfaceMaskPolicy",
    "TopologyContext",
    "TopologyPolicy",
]
