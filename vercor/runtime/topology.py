from __future__ import annotations

from collections.abc import Mapping, MutableMapping, Sequence
from dataclasses import dataclass

import jax.numpy as jnp

from vercor.components.base import Component
from vercor.dtypes import jax_ones
from vercor.exceptions import ComponentError, CouplerError
from vercor.exchange import Exchange
from vercor.grid_masks import (
    check_remap_conservation,
    check_total_lnd_ocn_mask_sum,
    compute_ocn_lnd_masks_on_atm_grid,
)
from vercor.grid_geometry import grids_identical
from vercor.jax_logging import LoggerLike
from vercor.regridders import (
    BilinearRectilinearRegridder,
    ConservativeRectilinearRegridder,
)
from vercor.settings import VercorSettings
from vercor.types import RuntimeArray

RuntimeRegridder = BilinearRectilinearRegridder | ConservativeRectilinearRegridder
VALID_TOPOLOGY_COMPONENT_NAMES = ("ATM", "OCN", "LND", "ICE")


@dataclass(frozen=True)
class ExchangeTopologyState:
    """Exchange regridders, masks, and derived land/ocean topology arrays."""

    regridders: dict[tuple[str, str, str], RuntimeRegridder]
    binary_masks: dict[tuple[str, str, str], RuntimeArray]
    fractional_masks: dict[tuple[str, str, str], RuntimeArray]
    ocn_fmask_on_atm_grid: RuntimeArray
    lnd_fmask_on_atm_grid: RuntimeArray
    lnd_bmask_on_atm_grid: RuntimeArray


def validate_component_topology_names(components: dict[str, Component]) -> None:
    """Validate registered component names supported by the default topology."""

    for name in components:
        if name not in VALID_TOPOLOGY_COMPONENT_NAMES:
            allowed = ", ".join(VALID_TOPOLOGY_COMPONENT_NAMES)
            raise ComponentError(f"Incorrect component name: {name}, must be {allowed}")


def get_component(allcomponents: dict[str, Component], types: str) -> Component:
    """Return the registered component with the requested VerCOR component name."""

    components: list[Component] = [
        component for component in allcomponents.values() if component.name == types
    ]
    if len(components) > 1:
        raise CouplerError(
            f"Multiple {components[0].name} components registered; only one supported"
        )
    if not components:
        raise CouplerError(f"No component of types ({types}) registered")
    return components[0]


def create_exchange_masks(
    components: dict[str, Component],
    *,
    logger: LoggerLike,
) -> tuple[RuntimeArray, RuntimeArray, RuntimeArray]:
    """Create atmosphere-grid ocean/land masks required by exchange setup."""

    land_component = get_component(components, "LND")
    atmosphere_component = get_component(components, "ATM")
    ocean_component = get_component(components, "OCN")

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
    return ocn_fmask_on_atm_grid, lnd_fmask_on_atm_grid, lnd_bmask_on_atm_grid


def validate_land_mask_consistency(
    components: dict[str, Component],
    lnd_bmask_on_atm_grid: RuntimeArray,
) -> None:
    """Validate that a component land mask matches the remapped exchange mask."""

    land_component = get_component(components, "LND")
    lnd_mask_from_component = land_component.grid.binary_mask
    if lnd_mask_from_component is not None:
        component_mask = jnp.asarray(lnd_mask_from_component)
        remapped_mask = jnp.asarray(lnd_bmask_on_atm_grid)
        if component_mask.shape != lnd_bmask_on_atm_grid.shape:
            raise CouplerError(
                "Land binary mask read from component does not match atmospheric grid shape"
            )
        if not bool(jnp.all(component_mask == remapped_mask)):
            mismatch = int(jnp.count_nonzero(component_mask != remapped_mask))
            raise CouplerError(
                "Land binary mask created from remapped ocean mask does not match component-provided mask "
                f"(mismatched points: {mismatch})"
            )


def initialize_regridders_and_masks(
    *,
    components: dict[str, Component],
    exchanges: Sequence[Exchange],
    regridders: MutableMapping[tuple[str, str, str], RuntimeRegridder],
    binary_masks: MutableMapping[tuple[str, str, str], RuntimeArray],
    fractional_masks: MutableMapping[tuple[str, str, str], RuntimeArray],
    settings: VercorSettings,
    logger: LoggerLike,
) -> None:
    """Initialize exchange regridders and identity masks for configured topology."""

    for exchange in exchanges:
        key = (exchange.source, exchange.destination, exchange.interpolation_type)

        if key not in regridders:
            regridders[key] = exchange.create(
                components[exchange.source].grid,
                components[exchange.destination].grid,
            )
            binary_masks[key] = jax_ones(
                components[exchange.destination].grid.shape,
                settings,
            )
            fractional_masks[key] = jax_ones(
                components[exchange.destination].grid.shape,
                settings,
            )
        else:
            logger.warning(
                f" Regridder for exchange {exchange.name} already exists, skipping creation"
            )


def patch_exchange_masks(
    *,
    binary_masks: MutableMapping[tuple[str, str, str], RuntimeArray],
    fractional_masks: MutableMapping[tuple[str, str, str], RuntimeArray],
    ocn_fmask_on_atm_grid: RuntimeArray,
    lnd_bmask_on_atm_grid: RuntimeArray,
    lnd_fmask_on_atm_grid: RuntimeArray,
) -> None:
    """Patch special land/ocean masks onto bilinear atmosphere exchanges."""

    for key in binary_masks.keys():
        source, destination, interp_type = key
        if "bilinear" in interp_type:
            if source == "OCN" and destination == "ATM":
                fractional_masks[key] = ocn_fmask_on_atm_grid
            elif source == "LND" and destination == "ATM":
                binary_masks[key] = lnd_bmask_on_atm_grid
                fractional_masks[key] = lnd_fmask_on_atm_grid


def build_exchange_topology(
    *,
    components: dict[str, Component],
    exchanges: Sequence[Exchange],
    settings: VercorSettings,
    logger: LoggerLike,
    regridders: Mapping[tuple[str, str, str], RuntimeRegridder] | None = None,
    binary_masks: Mapping[tuple[str, str, str], RuntimeArray] | None = None,
    fractional_masks: Mapping[tuple[str, str, str], RuntimeArray] | None = None,
) -> ExchangeTopologyState:
    """Build exchange regridders and masks as an explicit topology state."""

    (
        ocn_fmask_on_atm_grid,
        lnd_fmask_on_atm_grid,
        lnd_bmask_on_atm_grid,
    ) = create_exchange_masks(components, logger=logger)
    validate_land_mask_consistency(
        components,
        lnd_bmask_on_atm_grid,
    )
    logger.info(" LND <--> ATM & OCN <--> ATM masks initialization complete")

    topology_regridders = dict(regridders or {})
    topology_binary_masks = dict(binary_masks or {})
    topology_fractional_masks = dict(fractional_masks or {})
    initialize_regridders_and_masks(
        components=components,
        exchanges=exchanges,
        regridders=topology_regridders,
        binary_masks=topology_binary_masks,
        fractional_masks=topology_fractional_masks,
        settings=settings,
        logger=logger,
    )
    patch_exchange_masks(
        binary_masks=topology_binary_masks,
        fractional_masks=topology_fractional_masks,
        ocn_fmask_on_atm_grid=ocn_fmask_on_atm_grid,
        lnd_bmask_on_atm_grid=lnd_bmask_on_atm_grid,
        lnd_fmask_on_atm_grid=lnd_fmask_on_atm_grid,
    )
    logger.info(" Exchange masks patching complete")
    return ExchangeTopologyState(
        regridders=topology_regridders,
        binary_masks=topology_binary_masks,
        fractional_masks=topology_fractional_masks,
        ocn_fmask_on_atm_grid=ocn_fmask_on_atm_grid,
        lnd_fmask_on_atm_grid=lnd_fmask_on_atm_grid,
        lnd_bmask_on_atm_grid=lnd_bmask_on_atm_grid,
    )
