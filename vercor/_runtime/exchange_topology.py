from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

from vercor.dtypes import DTypePolicy, as_jax_real_array, jax_ones
from vercor.exceptions import CouplerError
from vercor.exchanges import Exchange
from vercor.fields import VectorField
from vercor.jax_logging import LoggerLike
from vercor.regridding import Regridder, VectorRegridder
from vercor._runtime.topology_state import RuntimeTopologyMaps

if TYPE_CHECKING:
    from vercor.components._adapter import _ComponentBinding


def build_exchange_topology_maps(
    *,
    components: Mapping[str, "_ComponentBinding"],
    exchanges: Sequence[Exchange],
    dtype: DTypePolicy,
    logger: LoggerLike,
    topology_maps: RuntimeTopologyMaps | None = None,
) -> RuntimeTopologyMaps:
    """Build exchange regridders and identity masks for configured topology."""

    regridders = {} if topology_maps is None else dict(topology_maps.regridders)
    binary_masks = {} if topology_maps is None else dict(topology_maps.binary_masks)
    fractional_masks = (
        {} if topology_maps is None else dict(topology_maps.fractional_masks)
    )
    configured_route_ids: set[str] = set()

    for exchange in exchanges:
        route_id = exchange.route_id
        if route_id in configured_route_ids:
            raise CouplerError(
                f"Duplicate exchange route ID {route_id!r}; route IDs must be unique."
            )
        configured_route_ids.add(route_id)
        if route_id not in regridders:
            regridder = exchange.regridder_factory(
                components[exchange.source].grid, components[exchange.target].grid
            )
            needs_scalar = any(isinstance(field, str) for field in exchange.fields)
            needs_vector = any(
                isinstance(field, VectorField) for field in exchange.fields
            )
            if needs_scalar and not isinstance(regridder, Regridder):
                raise CouplerError(
                    f"Exchange route '{route_id}' requires a Regridder capability."
                )
            if needs_vector and not isinstance(regridder, VectorRegridder):
                raise CouplerError(
                    f"Exchange route '{route_id}' requires a VectorRegridder capability."
                )
            regridders[route_id] = regridder
            target_grid = components[exchange.target].grid
            target_active_mask = (
                jax_ones(target_grid.shape, dtype)
                if target_grid.binary_mask is None
                else as_jax_real_array(target_grid.binary_mask, dtype)
            )
            binary_masks[route_id] = target_active_mask
            fractional_masks[route_id] = target_active_mask

    return RuntimeTopologyMaps(
        regridders=regridders,
        binary_masks=binary_masks,
        fractional_masks=fractional_masks,
    )


__all__ = ["build_exchange_topology_maps"]
