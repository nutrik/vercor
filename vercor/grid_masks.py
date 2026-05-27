from __future__ import annotations

from typing import Any, cast

import jax.numpy as jnp

from vercor.dtypes import as_jax_real_array
from vercor.exceptions import RegridderError
from vercor.grid import RectilinearGrid
from vercor.interpolators.conservative_remap_rectilinear import (
    ConservativeRectilinearRemapper,
)
from vercor.jax_logging import LoggerLike, get_default_logger
from vercor.regridders.conservative import ConservativeRectilinearRegridder
from vercor.types import RuntimeArray


def compute_land_mask(ocean_fractional_mask: Any) -> Any:
    """Compute land binary mask from an ocean fractional mask."""

    fminval = 0.001
    fmaxval = 1.0
    land_binary_mask = 1.0 - as_jax_real_array(ocean_fractional_mask)
    land_binary_mask = jnp.where(land_binary_mask > fmaxval, 1.0, land_binary_mask)
    land_binary_mask = jnp.where(land_binary_mask < fminval, 0.0, land_binary_mask)
    return cast(Any, jnp.where(land_binary_mask != 0.0, 1, 0))


def compute_ocn_lnd_masks_on_atm_grid(
    ocean_binary_mask: RuntimeArray, regridder: ConservativeRectilinearRegridder
) -> tuple[RuntimeArray, RuntimeArray, RuntimeArray]:
    """Compute ocean and land fractional and binary masks on the atmosphere grid."""

    ocean_bmask = as_jax_real_array(ocean_binary_mask)
    ocn_fmask_on_atm_grid = jnp.clip(
        as_jax_real_array(regridder(ocean_bmask)),
        0.0,
        1.0,
    )
    lnd_fmask_on_atm_grid = 1.0 - ocn_fmask_on_atm_grid
    lnd_bmask_on_atm_grid = compute_land_mask(ocn_fmask_on_atm_grid)

    return ocn_fmask_on_atm_grid, lnd_fmask_on_atm_grid, lnd_bmask_on_atm_grid


def check_total_lnd_ocn_mask_sum(
    lnd_fmask_on_atm_grid: RuntimeArray, ocn_fmask_on_atm_grid: RuntimeArray
) -> None:
    """Validate that land and ocean fractional masks sum to one."""

    fmask_sum = as_jax_real_array(lnd_fmask_on_atm_grid) + as_jax_real_array(
        ocn_fmask_on_atm_grid
    )
    min_fsum = float(jnp.min(fmask_sum))
    max_fsum = float(jnp.max(fmask_sum))
    if not bool(
        jnp.isclose(min_fsum, 1.0, atol=1e-3) & jnp.isclose(max_fsum, 1.0, atol=1e-3)
    ):
        raise RegridderError(
            "Fractional land and ocean masks on atmospheric grid must sum to approx. 1 everywhere "
            f"(minimum sum {min_fsum}, maximum sum {max_fsum})"
        )


def check_remap_conservation(
    regridder: ConservativeRectilinearRegridder,
    ocean_binary_mask_on_ocn_grid: RuntimeArray,
    ocn_fmask_on_atm_grid: RuntimeArray,
    logger: LoggerLike | None = None,
) -> None:
    """Validate conservative ocean-mask remapping mass conservation when grids are comparable."""

    do_not_check_mass = False
    log = logger if logger is not None else get_default_logger()

    if regridder.interpolator is not None and isinstance(
        regridder.interpolator, ConservativeRectilinearRemapper
    ):
        src_lat = regridder.interpolator.src_lat_b
        dst_lat = regridder.interpolator.dst_lat_b
        if bool((src_lat[-1] != dst_lat[-1]) | (src_lat[0] != dst_lat[0])):
            do_not_check_mass = True
            log.warning(
                "Skipping mass conservation check for regridding ocean mask to atmospheric grid "
                "due to different latitude bounds."
            )

        src_total_mass = regridder.interpolator.get_src_total_mass(
            ocean_binary_mask_on_ocn_grid
        )
        dst_total_mass = regridder.interpolator.get_dst_total_mass(
            ocn_fmask_on_atm_grid
        )

        if not do_not_check_mass and not bool(
            jnp.isclose(src_total_mass, dst_total_mass, atol=1e-6)
        ):
            raise RegridderError(
                "Regridding ocean binary mask to atmospheric grid does not conserve total mass "
                f"(source mass: {src_total_mass}, destination mass: {dst_total_mass})"
            )


def create_lnd_mask_from_ocn(
    atm_lat: RuntimeArray, atm_lon: RuntimeArray, ocn_grid: RectilinearGrid
) -> tuple[RuntimeArray, RuntimeArray]:
    """Create land binary and fractional masks from an ocean-grid binary mask."""

    atmosphere_grid = RectilinearGrid(
        name="ATM",
        longitude=atm_lon,
        latitude=atm_lat,
    )

    regridder = ConservativeRectilinearRegridder(
        ocn_grid,
        atmosphere_grid,
    )

    ocean_binary_mask = as_jax_real_array(ocn_grid.binary_mask)

    (
        ocn_fmask_on_atm_grid,
        lnd_fmask_on_atm_grid,
        lnd_bmask_on_atm_grid,
    ) = compute_ocn_lnd_masks_on_atm_grid(ocean_binary_mask, regridder)

    check_remap_conservation(regridder, ocean_binary_mask, ocn_fmask_on_atm_grid)
    check_total_lnd_ocn_mask_sum(lnd_fmask_on_atm_grid, ocn_fmask_on_atm_grid)

    return lnd_bmask_on_atm_grid, lnd_fmask_on_atm_grid
