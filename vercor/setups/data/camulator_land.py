"""Compatibility facade for the CAMulator land host-runtime adapter."""

from vercor.setups.external.camulator_land import (
    _CAMulatorLandState,
    _prepare_camulator_land_surface_temperature,
    make_camulator_land,
)

__all__ = [
    "_CAMulatorLandState",
    "_prepare_camulator_land_surface_temperature",
    "make_camulator_land",
]
