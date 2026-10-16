"""Lazy factory for standalone Veris on an explicitly supplied ocean wet grid."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from vercor._host_arrays import array_to_host
from vercor.components import (
    CallableComponent,
    Component,
    ComponentMaskSpec,
    ComponentSpec,
    LifecycleHooks,
)
from vercor.fields import _flatten_field_items
from vercor.grids import RectilinearGrid
from vercor.recipes import (
    ATMOSPHERE_TO_VERIS_FORCING_FIELDS,
    OCEAN_TO_VERIS_FORCING_FIELDS,
)
from vercor.setups.config import VerisConfig


def load_standalone_veris() -> Any:
    """Require maintained standalone entry points rather than the legacy plugin."""
    message = (
        "Standalone Veris requires initialize_experiment/advance and OceanGeometry; "
        "install the approved model with `pip install 'vercor[veris]'`."
    )
    try:
        import veris
    except ImportError as error:
        raise ImportError(message) from error
    if not all(
        callable(getattr(veris, name, None))
        for name in (
            "Experiment",
            "Execution",
            "Grid",
            "OceanGeometry",
            "initialize_experiment",
            "advance",
        )
    ):
        raise ImportError(message)
    return veris


def make_veris(
    grid: RectilinearGrid, geometry: Any, *, config: VerisConfig | None = None
) -> Component:
    """Wrap native sea ice on the ocean's identical physical grid and binary mask.

    Geometry uses native physical (x, y[, z]) arrays without halos. The factory
    copies geometry; setup alone initializes native state and static resources.
    Runtime ice_fraction is native Area, independent of wet-cell forcing support.
    """
    native = load_standalone_veris()
    if not isinstance(grid, RectilinearGrid):
        raise TypeError("grid must be RectilinearGrid")
    if not isinstance(geometry, native.OceanGeometry):
        raise TypeError("geometry must be native Veris OceanGeometry")
    mask = array_to_host(geometry.maskT)
    if mask.ndim != 3 or mask.shape[:2] != grid.shape[::-1] or mask.shape[2] < 1:
        raise ValueError("Veris geometry must match the physical ocean grid shape")
    surface = mask[:, :, -1].T
    if grid.binary_mask is None or not bool(((surface == 0) | (surface == 1)).all()):
        raise ValueError("Veris requires the binary ocean mask")
    if not bool((surface == array_to_host(grid.binary_mask)).all()):
        raise ValueError("Veris geometry surface mask must equal the ocean grid mask")
    config = VerisConfig() if config is None else config
    from vercor.setups._external.veris_runtime import VerisSetupState

    owner = VerisSetupState(config, deepcopy(geometry))
    return CallableComponent(
        config.name,
        grid,
        owner.step,
        spec=ComponentSpec(
            inputs=tuple(
                _flatten_field_items(
                    (
                        *ATMOSPHERE_TO_VERIS_FORCING_FIELDS,
                        *OCEAN_TO_VERIS_FORCING_FIELDS,
                    )
                )
            ),
            outputs=("ice_fraction",),
            initial_fields={"ice_fraction": 0.0},
            masks=ComponentMaskSpec(fractional_field="ice_fraction"),
            lifecycle=LifecycleHooks(setup=owner.setup),
            output=config.output,
        ),
    )
