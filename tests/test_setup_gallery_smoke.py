from __future__ import annotations

import ast
from datetime import datetime
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest

from tests.assertions import assert_allclose_compact
from vercor import Clock, Coupler, Exchange, RectilinearGrid, RuntimeOptions
from vercor.components import (
    CallableComponent,
    ComponentSpec,
    DataComponent,
    LifecycleHooks,
    SetupResult,
    StepResult,
)
from vercor.fields import _flatten_field_items
from vercor.dtypes import DTypePolicy
from vercor.regridding import bilinear, conservative
from vercor.topology import SurfaceMaskPolicy
from vercor.recipes import (
    ATMOSPHERE_TO_OCEAN_BULK_FORCING_FIELDS,
)


@pytest.mark.fast_always
@pytest.mark.parametrize(
    "names", [("ATM", "OCN", "LND"), ("air-model", "water-model", "terrain-model")]
)
def test_host_workflow_preserves_payload_precision_and_surface_masks(
    names: tuple[str, str, str],
) -> None:
    """Advance host payloads through the generic atmosphere/ocean/land boundary."""

    ocean_mask = np.asarray([[1.0, 0.0], [0.0, 1.0]])
    land_mask = 1.0 - ocean_mask
    longitude = np.asarray([0.0, 90.0])
    latitude = np.asarray([-45.0, 45.0])
    ocean_grid = RectilinearGrid(
        "ocean-grid",
        longitude=longitude,
        latitude=latitude,
        binary_mask=ocean_mask,
    )
    atmosphere_grid = RectilinearGrid(
        "atmosphere-grid",
        longitude=longitude,
        latitude=latitude,
    )
    land_grid = RectilinearGrid(
        "land-grid",
        longitude=longitude,
        latitude=latitude,
        binary_mask=land_mask,
    )

    ocean = DataComponent(
        names[1],
        ocean_grid,
        fields={
            "sea_surface_temperature": np.asarray([[280.0, np.nan], [np.nan, 284.0]])
        },
        spec=ComponentSpec(
            inputs=tuple(_flatten_field_items(ATMOSPHERE_TO_OCEAN_BULK_FORCING_FIELDS))
        ),
    )
    land = DataComponent(
        names[2],
        land_grid,
        fields={
            "land_surface_temperature": np.asarray([[np.nan, 282.0], [286.0, np.nan]])
        },
    )

    atmosphere_output_names = tuple(
        dict.fromkeys(
            _flatten_field_items(
                (*_flatten_field_items(ATMOSPHERE_TO_OCEAN_BULK_FORCING_FIELDS),)
            )
        )
    )
    atmosphere_fields = {
        name: np.full(atmosphere_grid.shape, index + 1.0)
        for index, name in enumerate(atmosphere_output_names)
    }
    native_steps: list[int] = []

    def setup_atmosphere(component: Any, context: Any) -> SetupResult:
        _ = component
        assert context.dtype.enable_x64 is False
        return SetupResult(payload=np.zeros((), dtype=np.float32))

    def step_atmosphere(
        fields: Any,
        context: Any,
        payload: Any | None,
    ) -> StepResult:
        _ = fields, context
        assert isinstance(payload, np.ndarray)
        assert payload.dtype == np.float32
        assert bool(np.isfinite(payload).all())
        native_steps.append(int(payload.item()))
        return StepResult(payload=np.asarray(payload + 1.0, dtype=np.float32))

    atmosphere = CallableComponent(
        names[0],
        atmosphere_grid,
        step_atmosphere,
        spec=ComponentSpec(
            inputs=("sea_surface_temperature", "land_surface_temperature"),
            outputs=atmosphere_output_names,
            initial_fields=atmosphere_fields,
            execution="host",
            lifecycle=LifecycleHooks(setup=setup_atmosphere),
        ),
    )

    coupler = Coupler(
        clock=Clock(start=datetime(1981, 1, 3), dt_seconds=21600.0, steps=2),
        components=(ocean, land, atmosphere),
        exchanges=(
            Exchange(
                atmosphere.name,
                ocean.name,
                ATMOSPHERE_TO_OCEAN_BULK_FORCING_FIELDS,
                quantity="intensive",
                regridder_factory=bilinear,
            ),
            Exchange(
                ocean.name,
                atmosphere.name,
                ("sea_surface_temperature",),
                quantity="source_flux",
                regridder_factory=conservative,
            ),
            Exchange(
                land.name,
                atmosphere.name,
                ("land_surface_temperature",),
                quantity="source_flux",
                regridder_factory=conservative,
            ),
        ),
        run_order=(ocean.name, land.name, atmosphere.name),
        runtime=RuntimeOptions(
            dtype=DTypePolicy(enable_x64=False),
            topology=SurfaceMaskPolicy(
                atmosphere=atmosphere.name, ocean=ocean.name, land=land.name
            ),
        ),
        log_level="error",
    )
    final_state = coupler.run()

    assert native_steps == [0, 1]
    topology = coupler._ensure_prepared().topology_maps
    assert_allclose_compact(
        topology.binary_masks[f"{atmosphere.name}->{ocean.name}"], ocean_mask
    )
    assert_allclose_compact(
        topology.binary_masks[f"{land.name}->{atmosphere.name}"], land_mask
    )

    views = final_state.components((ocean.name, land.name, atmosphere.name))
    for view in views.values():
        assert view.grid is not None
        assert view.grid.longitude.dtype == jnp.float32
        assert view.grid.latitude.dtype == jnp.float32
        if view.grid.binary_mask is not None:
            assert view.grid.binary_mask.dtype == jnp.float32
        for _, _, value in view.iter_fields("state", "received", "sent"):
            assert value.dtype == jnp.float32

    assert_allclose_compact(
        views[atmosphere.name].field("sea_surface_temperature", scope="received"),
        np.asarray([[280.0, 0.0], [0.0, 284.0]], dtype=np.float32),
    )
    assert_allclose_compact(
        views[atmosphere.name].field("land_surface_temperature", scope="received"),
        np.asarray([[0.0, 282.0], [286.0, 0.0]], dtype=np.float32),
    )


@pytest.mark.fast_always
def test_veris_gallery_uses_public_import_owners() -> None:
    """Keep the runnable gallery usable as an ordinary copied setup script."""
    path = Path("vercor/setups/gallery/veris_dataveros_era5.py")
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.ImportFrom)
            and node.module
            and node.module.startswith("vercor")
        ):
            assert not any(part.startswith("_") for part in node.module.split("."))
