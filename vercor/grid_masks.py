from __future__ import annotations

from typing import Any, cast

import jax.numpy as jnp

from vercor.dtypes import as_jax_real_array as _as_jax_real_array
from vercor.exceptions import RegridderError as _RegridderError
from vercor.grids import RectilinearGrid as _RectilinearGrid
from vercor._interpolators.conservative_remap_rectilinear import (
    ConservativeRectilinearRemapper as _ConservativeRectilinearRemapper,
)
from vercor.jax_logging import (
    LoggerLike as _LoggerLike,
    get_default_logger as _get_default_logger,
)
from vercor._regridders.conservative import (
    ConservativeRectilinearRegridder as _ConservativeRectilinearRegridder,
)
from vercor.regridding import Regridder as _Regridder
from vercor.types import RuntimeArray as _RuntimeArray

__all__ = [
    "check_remap_conservation",
    "check_land_ocean_fraction_sum",
    "compute_land_mask",
    "compute_ocean_land_masks_on_atmosphere_grid",
    "create_land_mask_from_ocean",
]

_FRACTIONAL_MASK_MINIMUM = 0.001


def compute_land_mask(ocean_fractional_mask: Any) -> Any:
    """Compute land binary mask from an ocean fractional mask."""

    maximum_fraction = 1.0
    land_binary_mask = 1.0 - _as_jax_real_array(ocean_fractional_mask)
    land_binary_mask = jnp.where(
        land_binary_mask > maximum_fraction, 1.0, land_binary_mask
    )
    land_binary_mask = jnp.where(
        land_binary_mask < _FRACTIONAL_MASK_MINIMUM,
        0.0,
        land_binary_mask,
    )
    return cast(Any, jnp.where(land_binary_mask != 0.0, 1, 0))


def compute_ocean_land_masks_on_atmosphere_grid(
    ocean_binary_mask: _RuntimeArray, regridder: _Regridder
) -> tuple[_RuntimeArray, _RuntimeArray, _RuntimeArray]:
    """Compute ocean and land fractional and binary masks on the atmosphere grid."""

    ocean_binary_mask_array = _as_jax_real_array(ocean_binary_mask)
    remapped_ocean_fraction = jnp.clip(
        _as_jax_real_array(regridder.regrid(ocean_binary_mask_array)),
        0.0,
        1.0,
    )
    ocean_fraction_on_atmosphere_grid = jnp.where(
        remapped_ocean_fraction < _FRACTIONAL_MASK_MINIMUM,
        0.0,
        remapped_ocean_fraction,
    )
    ocean_fraction_on_atmosphere_grid = jnp.where(
        ocean_fraction_on_atmosphere_grid > 1.0 - _FRACTIONAL_MASK_MINIMUM,
        1.0,
        ocean_fraction_on_atmosphere_grid,
    )
    land_fraction_on_atmosphere_grid = 1.0 - ocean_fraction_on_atmosphere_grid
    land_binary_mask_on_atmosphere_grid = compute_land_mask(
        ocean_fraction_on_atmosphere_grid
    )

    return (
        ocean_fraction_on_atmosphere_grid,
        land_fraction_on_atmosphere_grid,
        land_binary_mask_on_atmosphere_grid,
    )


def check_land_ocean_fraction_sum(
    land_fraction_on_atmosphere_grid: _RuntimeArray,
    ocean_fraction_on_atmosphere_grid: _RuntimeArray,
) -> None:
    """Validate that land and ocean fractional masks sum to one."""

    fraction_sum = _as_jax_real_array(
        land_fraction_on_atmosphere_grid
    ) + _as_jax_real_array(ocean_fraction_on_atmosphere_grid)
    minimum_fraction_sum = float(jnp.min(fraction_sum))
    maximum_fraction_sum = float(jnp.max(fraction_sum))
    if not bool(
        jnp.isclose(minimum_fraction_sum, 1.0, atol=1e-3)
        & jnp.isclose(maximum_fraction_sum, 1.0, atol=1e-3)
    ):
        raise _RegridderError(
            "Fractional land and ocean masks on atmospheric grid must sum to approx. 1 everywhere "
            f"(minimum sum {minimum_fraction_sum}, maximum sum {maximum_fraction_sum})"
        )


def _rectilinear_cell_areas(
    lon_bounds: _RuntimeArray,
    lat_bounds: _RuntimeArray,
    *,
    radius: float,
    flip_lat: bool,
) -> _RuntimeArray:
    """Return spherical rectilinear cell areas matching a field's latitude order."""

    dlon = jnp.abs(jnp.diff(jnp.deg2rad(_as_jax_real_array(lon_bounds))))
    sin_lat = jnp.sin(jnp.deg2rad(_as_jax_real_array(lat_bounds)))
    dsinlat = jnp.abs(jnp.diff(sin_lat))
    areas = (radius**2) * dsinlat[:, None] * dlon[None, :]
    return areas[::-1, :] if flip_lat else areas


def _total_mass_on_rectilinear_bounds(
    field: _RuntimeArray,
    lon_bounds: _RuntimeArray,
    lat_bounds: _RuntimeArray,
    *,
    radius: float,
    flip_lat: bool,
) -> float:
    """Return total finite scalar mass on rectilinear spherical bounds."""

    areas = _rectilinear_cell_areas(
        lon_bounds,
        lat_bounds,
        radius=radius,
        flip_lat=flip_lat,
    )
    return float(jnp.nansum(_as_jax_real_array(field) * areas))


def check_remap_conservation(
    regridder: _Regridder,
    ocean_binary_mask_on_ocean_grid: _RuntimeArray,
    ocean_fraction_on_atmosphere_grid: _RuntimeArray,
    logger: _LoggerLike | None = None,
) -> None:
    """Validate conservative ocean-mask remapping mass conservation when grids are comparable."""

    conservative_regridder = cast(_ConservativeRectilinearRegridder, regridder)
    do_not_check_mass = False
    log = logger if logger is not None else _get_default_logger()

    if conservative_regridder.interpolator is not None and isinstance(
        conservative_regridder.interpolator, _ConservativeRectilinearRemapper
    ):
        src_lat = conservative_regridder.interpolator.src_lat_b
        dst_lat = conservative_regridder.interpolator.dst_lat_b
        if bool((src_lat[-1] != dst_lat[-1]) | (src_lat[0] != dst_lat[0])):
            do_not_check_mass = True
            log.warning(
                "Skipping mass conservation check for regridding ocean mask to atmospheric grid "
                "due to different latitude bounds."
            )

        src_total_mass = _total_mass_on_rectilinear_bounds(
            ocean_binary_mask_on_ocean_grid,
            conservative_regridder.interpolator.src_lon_b,
            conservative_regridder.interpolator.src_lat_b,
            radius=conservative_regridder.interpolator.radius,
            flip_lat=conservative_regridder.interpolator._s_lat_flip,
        )
        dst_total_mass = _total_mass_on_rectilinear_bounds(
            ocean_fraction_on_atmosphere_grid,
            conservative_regridder.interpolator.dst_lon_b,
            conservative_regridder.interpolator.dst_lat_b,
            radius=conservative_regridder.interpolator.radius,
            flip_lat=conservative_regridder.interpolator._d_lat_flip,
        )

        if not do_not_check_mass and not bool(
            jnp.isclose(src_total_mass, dst_total_mass, atol=1e-6)
        ):
            raise _RegridderError(
                "Regridding ocean binary mask to atmospheric grid does not conserve total mass "
                f"(source mass: {src_total_mass}, destination mass: {dst_total_mass})"
            )


def create_land_mask_from_ocean(
    atmosphere_latitude: _RuntimeArray,
    atmosphere_longitude: _RuntimeArray,
    ocean_grid: _RectilinearGrid,
) -> tuple[_RuntimeArray, _RuntimeArray]:
    """Create land binary and fractional masks from an ocean-grid binary mask."""

    atmosphere_grid = _RectilinearGrid(
        name="atmosphere-mask-grid",
        longitude=atmosphere_longitude,
        latitude=atmosphere_latitude,
    )

    regridder = _ConservativeRectilinearRegridder(
        ocean_grid,
        atmosphere_grid,
    )

    ocean_binary_mask = _as_jax_real_array(ocean_grid.binary_mask)

    (
        ocean_fraction_on_atmosphere_grid,
        land_fraction_on_atmosphere_grid,
        land_binary_mask_on_atmosphere_grid,
    ) = compute_ocean_land_masks_on_atmosphere_grid(ocean_binary_mask, regridder)

    check_remap_conservation(
        regridder, ocean_binary_mask, ocean_fraction_on_atmosphere_grid
    )
    check_land_ocean_fraction_sum(
        land_fraction_on_atmosphere_grid, ocean_fraction_on_atmosphere_grid
    )

    return land_binary_mask_on_atmosphere_grid, land_fraction_on_atmosphere_grid
