from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

import vercor.runtime.exchange_topology as _exchange_topology
import vercor.runtime.surface_masks as _surface_masks
from vercor.exchange import Exchange
from vercor.jax_logging import LoggerLike
from vercor.runtime.topology_state import (
    ExchangeTopologyState,
    RuntimeTopologyMaps,
)
from vercor.settings import VercorSettings

if TYPE_CHECKING:
    from vercor.components.base import Component


def build_exchange_topology(
    *,
    components: Mapping[str, "Component"],
    exchanges: Sequence[Exchange],
    settings: VercorSettings,
    logger: LoggerLike,
    topology_maps: RuntimeTopologyMaps | None = None,
) -> ExchangeTopologyState:
    """Build exchange regridders, masks, and surface topology state."""

    surface_masks = _surface_masks.create_surface_exchange_masks(
        components,
        logger=logger,
    )
    _surface_masks.validate_land_mask_consistency(
        components,
        surface_masks,
    )
    logger.info(" LND <--> ATM & OCN <--> ATM masks initialization complete")

    initialized_maps = _exchange_topology.build_exchange_topology_maps(
        components=components,
        exchanges=exchanges,
        topology_maps=topology_maps,
        settings=settings,
        logger=logger,
    )
    _surface_masks.apply_surface_exchange_masks(
        initialized_maps,
        surface_masks=surface_masks,
    )
    logger.info(" Exchange masks patching complete")
    return ExchangeTopologyState(
        topology_maps=initialized_maps,
        surface_masks=surface_masks,
    )


__all__ = ["build_exchange_topology"]
