from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

import jax.numpy as jnp

from vercor.exceptions import ComponentError, CouplerError
from vercor.grid_masks import (
    check_remap_conservation,
    check_total_lnd_ocn_mask_sum,
    compute_ocn_lnd_masks_on_atm_grid,
)
from vercor.grid_geometry import grids_identical
from vercor.jax_logging import LoggerLike
from vercor.regridders import ConservativeRectilinearRegridder
from vercor.runtime.component_topology import require_component
from vercor.runtime.topology_state import RuntimeTopologyMaps, SurfaceExchangeMasks

if TYPE_CHECKING:
    from vercor.components.base import Component


def create_surface_exchange_masks(
    components: Mapping[str, "Component"],
    *,
    logger: LoggerLike,
) -> SurfaceExchangeMasks:
    """Create atmosphere-grid ocean/land masks required by exchange setup."""

    land_component = require_component(components, "LND")
    atmosphere_component = require_component(components, "ATM")
    ocean_component = require_component(components, "OCN")

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
        ocn_fmask_on_atm_grid,
        lnd_fmask_on_atm_grid,
        lnd_bmask_on_atm_grid,
    ) = compute_ocn_lnd_masks_on_atm_grid(ocean_binary_mask, regridder)

    check_remap_conservation(
        regridder,
        ocean_binary_mask,
        ocn_fmask_on_atm_grid,
        logger=logger,
    )

    check_total_lnd_ocn_mask_sum(lnd_fmask_on_atm_grid, ocn_fmask_on_atm_grid)
    return SurfaceExchangeMasks(
        ocn_fmask_on_atm_grid=ocn_fmask_on_atm_grid,
        lnd_fmask_on_atm_grid=lnd_fmask_on_atm_grid,
        lnd_bmask_on_atm_grid=lnd_bmask_on_atm_grid,
    )


def validate_land_mask_consistency(
    components: Mapping[str, "Component"],
    surface_masks: SurfaceExchangeMasks,
) -> None:
    """Validate that a component land mask matches the remapped exchange mask."""

    land_component = require_component(components, "LND")
    lnd_mask_from_component = land_component.grid.binary_mask
    if lnd_mask_from_component is not None:
        component_mask = jnp.asarray(lnd_mask_from_component)
        remapped_mask = jnp.asarray(surface_masks.lnd_bmask_on_atm_grid)
        if component_mask.shape != surface_masks.lnd_bmask_on_atm_grid.shape:
            raise CouplerError(
                "Land binary mask read from component does not match atmospheric grid shape"
            )
        if not bool(jnp.all(component_mask == remapped_mask)):
            mismatch = int(jnp.count_nonzero(component_mask != remapped_mask))
            raise CouplerError(
                "Land binary mask created from remapped ocean mask does not match component-provided mask "
                f"(mismatched points: {mismatch})"
            )


def apply_surface_exchange_masks(
    topology_maps: RuntimeTopologyMaps,
    *,
    surface_masks: SurfaceExchangeMasks,
) -> RuntimeTopologyMaps:
    """Patch special land/ocean masks onto bilinear atmosphere exchanges."""

    for key in topology_maps.binary_masks.keys():
        source, destination, interp_type = key
        if "bilinear" in interp_type:
            if source == "OCN" and destination == "ATM":
                topology_maps.fractional_masks[key] = (
                    surface_masks.ocn_fmask_on_atm_grid
                )
            elif source == "LND" and destination == "ATM":
                topology_maps.binary_masks[key] = surface_masks.lnd_bmask_on_atm_grid
                topology_maps.fractional_masks[key] = (
                    surface_masks.lnd_fmask_on_atm_grid
                )
    return topology_maps


__all__ = [
    "apply_surface_exchange_masks",
    "create_surface_exchange_masks",
    "validate_land_mask_consistency",
]
