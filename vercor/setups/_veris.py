"""Paired standalone sea ice and prescribed ocean from one Veros setup snapshot."""

from __future__ import annotations

from dataclasses import dataclass

from vercor.components import Component, DataComponent
from vercor.output import OutputSpec
from vercor.setups.config import VerisConfig
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
