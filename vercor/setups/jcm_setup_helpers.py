from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from vercor.host_arrays import transposed_host_array
from vercor.components import Component, DataComponent
from vercor.grid import RectilinearGrid


@dataclass(frozen=True)
class JCMLandAtmosphereSetup:
    """JCM setup components plus generated setup objects used by runnable scripts."""

    land: DataComponent
    atmosphere: Component
    coords: Any
    terrain: Any
    forcing: Any


def generate_jcm_coords_forcing_topography_files(*args: Any, **kwargs: Any) -> Any:
    """Load JCM setup data through the optional JCM dependency boundary."""

    from vercor.setups.external.jax_gcm_tools import (
        generate_jcm_coords_forcing_topography_files as _generate,
    )

    return _generate(*args, **kwargs)


def make_jcm_land(*args: Any, **kwargs: Any) -> DataComponent:
    """Create a JCM land component through the optional JCM dependency boundary."""

    from vercor.setups.data.jcm_land import make_jcm_land as _make_jcm_land

    return _make_jcm_land(*args, **kwargs)


def make_jax_gcm(*args: Any, **kwargs: Any) -> Component:
    """Create a JAXGCM component through the optional JCM dependency boundary."""

    from vercor.setups.external.jax_gcm import make_jax_gcm as _make_jax_gcm

    return _make_jax_gcm(*args, **kwargs)


def build_jcm_land_atmosphere_components(
    ocn_grid: RectilinearGrid,
    *,
    custom_parameters: Mapping[str, float] | None = None,
    do_spinup: bool = True,
    jitted: bool = True,
    output_frequency: str | None = "month",
) -> JCMLandAtmosphereSetup:
    """Create paired JCM land and atmosphere setup components for an ocean grid."""

    coords, terrain, forcing = generate_jcm_coords_forcing_topography_files()
    land = make_jcm_land(coords, forcing, ocn_grid)

    # JAXGCM expects the terrain mask in host/transposed JCM layout.
    if land.grid.binary_mask is None:
        raise ValueError("JCM land grid requires a binary mask for terrain patching")
    terrain.fmask = transposed_host_array(land.grid.binary_mask)

    atmosphere_kwargs: dict[str, Any] = {
        "forcing_data": forcing,
        "do_spinup": do_spinup,
        "jitted": jitted,
        "output_frequency": output_frequency,
    }
    if custom_parameters is not None:
        atmosphere_kwargs["custom_parameters"] = dict(custom_parameters)

    atmosphere = make_jax_gcm(coords, terrain, **atmosphere_kwargs)
    return JCMLandAtmosphereSetup(
        land=land,
        atmosphere=atmosphere,
        coords=coords,
        terrain=terrain,
        forcing=forcing,
    )
