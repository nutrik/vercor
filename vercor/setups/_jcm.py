from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from vercor._host_arrays import transposed_host_array
from vercor.components import Component, DataComponent
from vercor.grids import RectilinearGrid
from vercor.dtypes import as_jax_real_array
from vercor.grid_masks import _surface_fractions_from_ocean
from vercor.setups.config import JCMLandAtmosphereConfig


@dataclass(frozen=True)
class JCMLandAtmosphereSetup:
    """JCM setup components plus generated setup objects used by runnable scripts."""

    land: DataComponent
    atmosphere: Component
    coords: Any
    terrain: Any
    forcing: Any


@dataclass(frozen=True)
class JCMInputs:
    """Generated JCM coordinate, terrain, and forcing inputs."""

    coords: Any
    terrain: Any
    forcing: Any


def load_jcm_inputs(
    *,
    resolution: int = 31,
    input_data_directory: Path | None = None,
) -> JCMInputs:
    """Load JCM coordinate, terrain, and forcing inputs."""

    from vercor.setups._external.jax_gcm_tools import load_jcm_coords_terrain_forcing

    coords, terrain, forcing = load_jcm_coords_terrain_forcing(
        resolution=resolution,
        input_data_directory=input_data_directory,
    )
    return JCMInputs(coords=coords, terrain=terrain, forcing=forcing)


def _load_jcm_factories() -> tuple[
    Callable[..., DataComponent],
    Callable[..., Component],
]:
    """Return optional JCM setup factories from their owning modules."""

    from vercor.setups._data.jcm_land import make_jcm_land
    from vercor.setups._external.jax_gcm import make_jcm_atmosphere

    return make_jcm_land, make_jcm_atmosphere


def make_jcm_land_atmosphere(
    ocn_grid: RectilinearGrid,
    *,
    inputs: JCMInputs | None = None,
    config: JCMLandAtmosphereConfig | None = None,
) -> JCMLandAtmosphereSetup:
    """Create paired JCM land and atmosphere setup components for an ocean grid."""

    make_jcm_land, make_jcm_atmosphere = _load_jcm_factories()
    if config is None:
        config = JCMLandAtmosphereConfig()
    atmosphere_config = config.atmosphere
    jcm_inputs = load_jcm_inputs() if inputs is None else inputs
    coords = jcm_inputs.coords
    terrain = jcm_inputs.terrain
    forcing = jcm_inputs.forcing
    land = make_jcm_land(
        coords,
        forcing,
        ocn_grid,
        name=config.land_name,
        output=config.land_output,
        native_land_fraction=as_jax_real_array(terrain.fmask).T,
    )

    # JAXGCM expects the terrain mask in host/transposed JCM layout.
    if land.grid.binary_mask is None:
        raise ValueError("JCM land grid requires a binary mask for terrain patching")
    terrain = terrain.copy(
        fmask=transposed_host_array(
            as_jax_real_array(land.spec.initial_fields["land_fraction"])
        )
    )
    _, prescribed_marine_fraction, _ = _surface_fractions_from_ocean(
        land.grid, ocn_grid, as_jax_real_array(jcm_inputs.terrain.fmask).T
    )

    atmosphere = make_jcm_atmosphere(
        coords,
        terrain,
        config=replace(
            atmosphere_config,
            forcing_data=(
                forcing
                if atmosphere_config.forcing_data is None
                else atmosphere_config.forcing_data
            ),
        ),
        prescribed_marine_fraction=prescribed_marine_fraction,
    )
    return JCMLandAtmosphereSetup(
        land=land,
        atmosphere=atmosphere,
        coords=coords,
        terrain=terrain,
        forcing=forcing,
    )
