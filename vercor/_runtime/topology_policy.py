from __future__ import annotations

from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import TYPE_CHECKING

import jax.numpy as jnp
from jax.errors import TracerBoolConversionError

from vercor.components import Component
from vercor.exceptions import CouplerError
from vercor.exchanges import Exchange
from vercor.jax_logging import LoggerLike
from vercor._runtime.topology_state import RuntimeTopologyMaps
from vercor.topology import (
    ExchangeTopologyPatch,
    TopologyContext,
    TopologyPolicy,
)
from vercor.types import RuntimeArray

if TYPE_CHECKING:
    from vercor.components._adapter import _ComponentBinding


def build_topology_context(
    *,
    components: Mapping[str, "_ComponentBinding"],
    exchanges: Sequence[Exchange],
    logger: LoggerLike,
) -> TopologyContext:
    """Return public topology-policy context from private runtime inputs."""

    public_components = MappingProxyType(
        {name: component._component for name, component in components.items()}
    )
    return TopologyContext(
        components=public_components,
        exchanges=tuple(exchanges),
        logger=logger,
    )


def apply_topology_policy(
    topology_maps: RuntimeTopologyMaps,
    *,
    components: Mapping[str, "_ComponentBinding"],
    exchanges: Sequence[Exchange],
    logger: LoggerLike,
    policy: TopologyPolicy | None,
) -> RuntimeTopologyMaps:
    """Apply an optional public topology policy to private runtime maps."""

    if policy is None:
        return topology_maps

    context = build_topology_context(
        components=components,
        exchanges=exchanges,
        logger=logger,
    )
    patch = policy.build(context)
    if not isinstance(patch, ExchangeTopologyPatch):
        raise CouplerError(
            f"Topology policy {policy.__class__.__qualname__}.build(...) must "
            "return ExchangeTopologyPatch; "
            f"got {type(patch).__name__}."
        )
    prepared_maps = _apply_exchange_topology_patch(
        topology_maps,
        patch,
        components=context.components,
        exchanges=context.exchanges,
    )
    logger.info("Exchange topology policy patching complete")
    return prepared_maps


def _apply_exchange_topology_patch(
    topology_maps: RuntimeTopologyMaps,
    patch: ExchangeTopologyPatch,
    *,
    components: Mapping[str, Component],
    exchanges: Sequence[Exchange],
) -> RuntimeTopologyMaps:
    """Apply public topology mask patches to runtime topology maps."""

    binary_masks = dict(topology_maps.binary_masks)
    route_targets = {exchange.route_id: exchange.target for exchange in exchanges}
    for key, value in patch.binary_masks.items():
        binary_masks[key] = _validate_patch_item(
            topology_maps,
            components,
            route_targets,
            key,
            value,
        )
    return RuntimeTopologyMaps(
        regridders=topology_maps.regridders,
        binary_masks=binary_masks,
    )


def _validate_patch_item(
    topology_maps: RuntimeTopologyMaps,
    components: Mapping[str, Component],
    route_targets: Mapping[str, str],
    key: str,
    value: object,
) -> RuntimeArray:
    """Validate one public topology patch item before map replacement."""

    if key not in topology_maps.regridders:
        raise CouplerError(
            f"Topology policy binary mask route ID {key!r} does not match a "
            "configured route ID."
        )
    try:
        mask_array = jnp.asarray(value)
    except (TypeError, ValueError) as exc:
        raise CouplerError(
            f"Topology policy binary mask for key {key!r} must be a "
            "concrete numeric or bool array."
        ) from exc
    is_real_numeric = (
        jnp.issubdtype(mask_array.dtype, jnp.bool_)
        or jnp.issubdtype(mask_array.dtype, jnp.integer)
        or jnp.issubdtype(mask_array.dtype, jnp.floating)
    )
    if not is_real_numeric:
        raise CouplerError(
            f"Topology policy binary mask for key {key!r} must be a "
            "concrete numeric or bool array."
        )
    target_name = route_targets[key]
    target_shape = components[target_name].grid.shape
    mask_shape = mask_array.shape
    if mask_shape != target_shape:
        raise CouplerError(
            f"Topology policy binary mask for key {key!r} has shape "
            f"{mask_shape}, expected {target_shape} for target component "
            f"{target_name!r}."
        )
    try:
        all_finite = bool(jnp.all(jnp.isfinite(mask_array)))
        all_binary = bool(jnp.all(jnp.logical_or(mask_array == 0, mask_array == 1)))
    except TracerBoolConversionError as exc:
        raise CouplerError(
            f"Topology policy binary mask for key {key!r} must be a "
            "concrete numeric or bool array."
        ) from exc
    if not all_finite:
        raise CouplerError(
            f"Topology policy binary mask for key {key!r} must contain "
            "only finite values."
        )
    if not all_binary:
        raise CouplerError(
            f"Topology policy binary mask for key {key!r} must contain only "
            "values in {0, 1}."
        )
    target_support = components[target_name].grid.binary_mask
    if target_support is not None and bool(
        jnp.any((mask_array > 0) & (jnp.asarray(target_support) == 0))
    ):
        raise CouplerError(
            f"Topology policy binary mask for route {key!r} expands target computational support."
        )
    reference = topology_maps.binary_masks[key]
    return jnp.asarray(mask_array, dtype=jnp.asarray(reference).dtype)


__all__ = [
    "apply_topology_policy",
    "build_topology_context",
]
