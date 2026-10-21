"""Standalone sea ice paired with one prescribed or prognostic Veros ocean."""

from __future__ import annotations

from dataclasses import dataclass

from vercor.components import Component, DataComponent
from vercor.output import OutputSpec
from vercor.setups.config import VerisConfig, VerosConfig
from vercor.setups._data.veros_ocean import make_veros_data_ocean
from vercor.setups._external.veros_data import load_veros_data_snapshot
from vercor.setups._external.veris import load_standalone_veris, make_veris


@dataclass(frozen=True)
class VerisVerosDataSetup:
    """Two independently owned components sharing one exact ocean grid."""

    seaice: Component
    ocean: DataComponent


def make_veris_verosdata(
    *,
    config: VerisConfig | None = None,
    ocean_name: str = "OCN",
    ocean_output: OutputSpec | None = None,
) -> VerisVerosDataSetup:
    """Initialize Veros geometry once, retaining only prescribed arrays and Veris."""
    config = VerisConfig() if config is None else config
    if not isinstance(ocean_name, str) or not ocean_name:
        raise ValueError("ocean_name must be a non-empty string")
    if ocean_name == config.name:
        raise ValueError("sea-ice and ocean names must be distinct")
    native = load_standalone_veris()
    snapshot = load_veros_data_snapshot()
    geometry = native.OceanGeometry(**snapshot.geometry)
    return VerisVerosDataSetup(
        seaice=make_veris(snapshot.grid, geometry, config=config),
        ocean=make_veros_data_ocean(snapshot, name=ocean_name, output=ocean_output),
    )


@dataclass(frozen=True)
class VerisVerosSetup:
    """Prognostic sea ice and ocean sharing one exact native wet grid."""

    seaice: Component
    ocean: Component


def make_veris_veros(
    *,
    config: VerisConfig | None = None,
    ocean_config: VerosConfig | None = None,
) -> VerisVerosSetup:
    """Initialize one prognostic ocean and reuse its exact geometry for sea ice.

    Total feedback mode is required. Native initialization happens once; the
    setup-only ocean owner retains the evolving ocean, while Veris owns its
    independent frozen payload. No second ocean or climatology proxy is kept.
    """
    config = VerisConfig() if config is None else config
    ocean_config = (
        VerosConfig(surface_forcing="surface_fluxes")
        if ocean_config is None
        else ocean_config
    )
    if (
        ocean_config.surface_forcing != "surface_fluxes"
        or ocean_config.setup != "global_4deg"
    ):
        raise ValueError("prognostic Veris requires coupled global_4deg Veros")
    if ocean_config.name == config.name:
        raise ValueError("sea-ice and ocean names must be distinct")
    native = load_standalone_veris()
    from vercor.setups._external.veros_gcm import (
        _initialize_veros_owner,
        _make_veros_component,
    )
    from vercor.setups._external.veros_data import _snapshot_veros_geometry

    owner = _initialize_veros_owner(ocean_config)
    grid, geometry = _snapshot_veros_geometry(owner._veros_state)
    owner.grid = grid
    return VerisVerosSetup(
        seaice=make_veris(grid, native.OceanGeometry(**geometry), config=config),
        ocean=_make_veros_component(owner, ocean_config),
    )
