from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from vercor.types import RuntimeArray


@dataclass(frozen=True, slots=True)
class RuntimeTopologyMaps:
    """Read-only grouped exchange topology maps used by runtime dispatch."""

    regridders: Mapping[str, Any]
    binary_masks: Mapping[str, RuntimeArray]

    def __post_init__(self) -> None:
        """Copy and freeze topology mappings at the runtime boundary."""

        object.__setattr__(
            self,
            "regridders",
            MappingProxyType(dict(self.regridders)),
        )
        object.__setattr__(
            self,
            "binary_masks",
            MappingProxyType(dict(self.binary_masks)),
        )


__all__ = ["RuntimeTopologyMaps"]
