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
from vercor.output import OutputSpec
from vercor.recipes import (
    ATMOSPHERE_TO_VERIS_FORCING_FIELDS,
    OCEAN_TO_VERIS_FORCING_FIELDS,
    VERIS_TO_OCEAN_FLUX_FIELDS,
    VERIS_TO_JCM_SURFACE_FIELDS,
)
from vercor.setups.config import VerisConfig


def load_standalone_veris() -> Any:
    """Require maintained standalone entry points rather than the legacy plugin."""
    message = (
        "Standalone Veris requires initialize_experiment/advance, spherical OceanGeometry, "
        "physical-parameter AD, current surface albedo and corrected ocean budgets; "
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
            "diagnose_surface_albedo",
        )
    ):
        raise ImportError(message)
    try:
        from veris.model.physical_constants import PhysicalConstants
        from veris.model.diagnostics import DIAGNOSTICS
        from dataclasses import fields

        supported = (
            callable(getattr(PhysicalConstants, "with_overrides", None))
            and callable(getattr(PhysicalConstants, "base_parameter_names", None))
            and {"latitude", "latitude_edges", "radius"}
            <= {field.name for field in fields(veris.OceanGeometry)}
            and {
                "physics_status",
                "ocean_mass_flux",
                "ocean_salt_mass_flux",
                "external_mass_flux",
                "external_heat_flux",
            }
            <= DIAGNOSTICS.keys()
        )
    except (ImportError, TypeError) as error:
        raise ImportError(message) from error
    if not supported:
        raise ImportError(message)
    return veris


def make_veris(
    grid: RectilinearGrid, geometry: Any, *, config: VerisConfig | None = None
) -> Component:
    """Wrap native sea ice on the ocean's identical physical grid and binary mask.

    Geometry uses native west/south faces and physical (x, y[, z]) arrays
    without halos. Incoming canonical currents use Veros east/north faces.
    The factory
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
    from vercor.setups._external.veris_output import (
        veris_output_provider,
        write_veris_snapshot_output,
    )

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
            outputs=(
                "ice_fraction",
                *_flatten_field_items(VERIS_TO_JCM_SURFACE_FIELDS),
                *_flatten_field_items(VERIS_TO_OCEAN_FLUX_FIELDS),
            ),
            initial_fields={
                name: 0.0
                for name in (
                    "ice_fraction",
                    *_flatten_field_items(VERIS_TO_JCM_SURFACE_FIELDS),
                    *_flatten_field_items(VERIS_TO_OCEAN_FLUX_FIELDS),
                )
            },
            masks=ComponentMaskSpec(fractional_field="ice_fraction"),
            lifecycle=LifecycleHooks(setup=owner.setup),
            output=OutputSpec(
                provider=(
                    veris_output_provider()
                    if config.output.provider is None
                    else config.output.provider
                ),
                period=config.output.period,
                snapshot_writer=(
                    write_veris_snapshot_output
                    if config.output.snapshot_writer is None
                    else config.output.snapshot_writer
                ),
            ),
        ),
    )
