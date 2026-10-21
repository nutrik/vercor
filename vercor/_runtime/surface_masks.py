from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Protocol

import jax.numpy as jnp

from vercor.exceptions import ComponentError, CouplerError
from vercor.grid_masks import (
    check_remap_conservation,
    check_land_ocean_fraction_sum,
    compute_ocean_land_masks_on_atmosphere_grid,
)
from vercor.grid_geometry import grids_identical
from vercor.jax_logging import LoggerLike
from vercor._regridders.conservative import ConservativeRectilinearRegridder
from vercor.types import RuntimeArray
from vercor.topology import ExchangeTopologyPatch, SurfaceMaskPolicy, TopologyContext

if TYPE_CHECKING:
    from vercor.exchanges import Exchange
    from vercor.grids import RectilinearGrid


class _SurfaceRoleComponent(Protocol):
    @property
    def name(self) -> str:
        """Return the registered component name."""
        ...

    @property
    def grid(self) -> "RectilinearGrid":
        """Return the component grid."""
        ...


def should_apply_surface_mask_policy(
    components: Mapping[str, _SurfaceRoleComponent],
    exchanges: Sequence["Exchange"],
    policy: SurfaceMaskPolicy | None,
) -> bool:
    """Return whether configured exchanges should use the surface-mask policy."""

    if policy is None or policy.mode == "disabled":
        return False
    if policy.mode == "required":
        return True
    role_names = (policy.atmosphere, policy.ocean, policy.land)
    if not all(name in components for name in role_names):
        return False
    return any(
        exchange.target == policy.atmosphere
        and exchange.source in (policy.ocean, policy.land)
        for exchange in exchanges
    )


def _require_surface_role(
    components: Mapping[str, _SurfaceRoleComponent],
    role_name: str,
) -> _SurfaceRoleComponent:
    """Return a surface-role component with a policy-oriented error."""

    try:
        component = components[role_name]
    except KeyError as exc:
        raise CouplerError(
            f"Surface mask policy requires role component {role_name!r} to be registered"
        ) from exc
    if component.name != role_name:
        raise CouplerError(
            f"Surface mask policy requires role component {role_name!r} to be registered"
        )
    return component


def create_surface_exchange_masks(
    components: Mapping[str, _SurfaceRoleComponent],
    *,
    policy: SurfaceMaskPolicy,
    logger: LoggerLike,
) -> tuple[RuntimeArray, RuntimeArray, RuntimeArray]:
    """Create atmosphere-grid ocean/land masks required by exchange setup."""

    land_component = _require_surface_role(components, policy.land)
    atmosphere_component = _require_surface_role(components, policy.atmosphere)
    ocean_component = _require_surface_role(components, policy.ocean)

    if not grids_identical(land_component.grid, atmosphere_component.grid):
        raise CouplerError(
            "Land and atmospheric components must use identical horizontal grids"
        )

    regridder = ConservativeRectilinearRegridder(
        ocean_component.grid,
        atmosphere_component.grid,
    )

    ocean_binary_mask = ocean_component.grid.binary_mask
    if ocean_binary_mask is None:
        raise ComponentError(
            f"Ocean component {ocean_component.name} has no binary mask defined"
        )

    (
        ocean_fraction_on_atmosphere_grid,
        land_fraction_on_atmosphere_grid,
        land_binary_mask_on_atmosphere_grid,
    ) = compute_ocean_land_masks_on_atmosphere_grid(ocean_binary_mask, regridder)

    check_remap_conservation(
        regridder,
        ocean_binary_mask,
        ocean_fraction_on_atmosphere_grid,
        logger=logger,
    )

    check_land_ocean_fraction_sum(
        land_fraction_on_atmosphere_grid, ocean_fraction_on_atmosphere_grid
    )
    return (
        ocean_fraction_on_atmosphere_grid,
        land_fraction_on_atmosphere_grid,
        land_binary_mask_on_atmosphere_grid,
    )


def validate_land_mask_consistency(
    components: Mapping[str, _SurfaceRoleComponent],
    land_binary_mask_on_atmosphere_grid: RuntimeArray,
    *,
    policy: SurfaceMaskPolicy,
) -> None:
    """Validate that a component land mask matches the remapped exchange mask."""

    land_component = _require_surface_role(components, policy.land)
    land_mask_from_component = land_component.grid.binary_mask
    if land_mask_from_component is not None:
        component_mask = jnp.asarray(land_mask_from_component)
        remapped_mask = jnp.asarray(land_binary_mask_on_atmosphere_grid)
        if component_mask.shape != remapped_mask.shape:
            raise CouplerError(
                "Land binary mask read from component does not match atmospheric grid shape"
            )
        if not bool(jnp.all(component_mask == remapped_mask)):
            mismatch = int(jnp.count_nonzero(component_mask != remapped_mask))
            raise CouplerError(
                "Land binary mask created from remapped ocean mask does not match component-provided mask "
                f"(mismatched points: {mismatch})"
            )


def build_surface_mask_topology_patch(
    context: TopologyContext,
    policy: SurfaceMaskPolicy,
) -> ExchangeTopologyPatch:
    """Return topology patches from temporary derived surface masks."""

    (
        _,
        _,
        land_binary_mask_on_atmosphere_grid,
    ) = create_surface_exchange_masks(
        context.components,
        policy=policy,
        logger=context.logger,
    )
    validate_land_mask_consistency(
        context.components,
        land_binary_mask_on_atmosphere_grid,
        policy=policy,
    )
    binary_masks = {}
    for exchange in context.exchanges:
        source = exchange.source
        destination = exchange.target
        route_id = exchange.route_id
        if source == policy.land and destination == policy.atmosphere:
            binary_masks[route_id] = land_binary_mask_on_atmosphere_grid
    return ExchangeTopologyPatch(
        binary_masks=binary_masks,
    )


__all__ = [
    "build_surface_mask_topology_patch",
    "create_surface_exchange_masks",
    "should_apply_surface_mask_policy",
    "validate_land_mask_consistency",
]
