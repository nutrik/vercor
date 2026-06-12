from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

from vercor.dtypes import jax_ones
from vercor.exchange import Exchange
from vercor.jax_logging import LoggerLike
from vercor.runtime.topology_state import RuntimeTopologyMaps
from vercor.settings import VercorSettings

if TYPE_CHECKING:
    from vercor.components.base import Component


def build_exchange_topology_maps(
    *,
    components: Mapping[str, "Component"],
    exchanges: Sequence[Exchange],
    settings: VercorSettings,
    logger: LoggerLike,
    topology_maps: RuntimeTopologyMaps | None = None,
) -> RuntimeTopologyMaps:
    """Build exchange regridders and identity masks for configured topology."""

    if topology_maps is None:
        initialized_maps = RuntimeTopologyMaps.empty()
    else:
        initialized_maps = RuntimeTopologyMaps(
            regridders=dict(topology_maps.regridders),
            binary_masks=dict(topology_maps.binary_masks),
            fractional_masks=dict(topology_maps.fractional_masks),
        )

    for exchange in exchanges:
        key = (exchange.source, exchange.destination, exchange.interpolation_type)

        if key not in initialized_maps.regridders:
            initialized_maps.regridders[key] = exchange.create(
                components[exchange.source].grid,
                components[exchange.destination].grid,
            )
            initialized_maps.binary_masks[key] = jax_ones(
                components[exchange.destination].grid.shape,
                settings,
            )
            initialized_maps.fractional_masks[key] = jax_ones(
                components[exchange.destination].grid.shape,
                settings,
            )
        else:
            logger.warning(
                f" Regridder for exchange {exchange.name} already exists, skipping creation"
            )

    return initialized_maps


__all__ = ["build_exchange_topology_maps"]
