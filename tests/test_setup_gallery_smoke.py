from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, cast

import jax.numpy as jnp
import numpy as np
import pytest
import torch

from tests.assertions import assert_allclose_compact
from vercor import Clock, Coupler, RectilinearGrid
from vercor.components import (
    CallableComponent,
    ComponentSpec,
    DataComponent,
    LifecycleHooks,
    SetupResult,
    StepResult,
)
from vercor.fields import _flatten_field_items
from vercor.output import OutputTarget
from vercor.recipes import (
    ATMOSPHERE_TO_LAND_RADIATION_FIELDS,
    ATMOSPHERE_TO_VEROS_FORCING_FIELDS,
)
from vercor.setups.gallery import run_camulator_with_veros


@pytest.mark.fast_always
def test_camulator_veros_gallery_composes_through_fake_native_boundary(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Exercise the unavailable CAMulator gallery through a finite host fake."""

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
        "OCN",
        ocean_grid,
        fields={
            "sea_surface_temperature": np.asarray([[280.0, np.nan], [np.nan, 284.0]])
        },
        spec=ComponentSpec(
            inputs=tuple(_flatten_field_items(ATMOSPHERE_TO_VEROS_FORCING_FIELDS))
        ),
    )
    land = DataComponent(
        "LND",
        land_grid,
        fields={
            "land_surface_temperature": np.asarray([[np.nan, 282.0], [286.0, np.nan]])
        },
        spec=ComponentSpec(
            inputs=tuple(_flatten_field_items(ATMOSPHERE_TO_LAND_RADIATION_FIELDS))
        ),
    )

    atmosphere_output_names = tuple(
        dict.fromkeys(
            _flatten_field_items(
                (
                    *_flatten_field_items(ATMOSPHERE_TO_VEROS_FORCING_FIELDS),
                    *ATMOSPHERE_TO_LAND_RADIATION_FIELDS,
                )
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
        return SetupResult(payload=torch.zeros((), dtype=torch.float32))

    def step_atmosphere(
        fields: Any,
        context: Any,
        payload: Any | None,
    ) -> StepResult:
        _ = fields, context
        assert isinstance(payload, torch.Tensor)
        assert bool(torch.all(torch.isfinite(payload)))
        native_steps.append(int(payload.item()))
        return StepResult(payload=payload + 1.0)

    atmosphere = CallableComponent(
        "ATM",
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

    monkeypatch.setattr(run_camulator_with_veros, "make_veros_gcm", lambda **_: ocean)
    monkeypatch.setattr(
        run_camulator_with_veros,
        "make_camulator_gcm",
        lambda **_: atmosphere,
    )
    monkeypatch.setattr(
        run_camulator_with_veros,
        "make_camulator_land",
        lambda *args, **kwargs: land,
    )

    real_clock = Clock
    real_coupler = Coupler
    captured: dict[str, Any] = {}

    monkeypatch.setattr(
        run_camulator_with_veros,
        "Clock",
        lambda *args, **kwargs: real_clock(
            start=datetime(1981, 1, 3),
            dt_seconds=21600.0,
            steps=2,
            calendar="noleap",
        ),
    )
    monkeypatch.setattr(
        run_camulator_with_veros,
        "OutputTarget",
        lambda _: OutputTarget(
            tmp_path,
            write_period=False,
            write_final_fields=False,
            write_snapshots=False,
        ),
    )

    class _RecordingCoupler:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.coupler = real_coupler(*args, **kwargs)
            captured["coupler"] = self.coupler

        def run(self, *, output: OutputTarget) -> Any:
            final_state = self.coupler.run(output=output)
            captured["final_state"] = final_state
            return final_state

    monkeypatch.setattr(run_camulator_with_veros, "Coupler", _RecordingCoupler)

    run_camulator_with_veros.run_setup(loglevel="error", float_type="float32")

    assert native_steps == [0, 1]
    coupler = cast(Coupler, captured["coupler"])
    topology = coupler._ensure_prepared().topology_maps
    assert_allclose_compact(topology.binary_masks["ATM->OCN"], ocean_mask)
    assert_allclose_compact(topology.fractional_masks["ATM->OCN"], ocean_mask)
    assert_allclose_compact(topology.binary_masks["ATM->LND"], land_mask)
    assert_allclose_compact(topology.fractional_masks["ATM->LND"], land_mask)

    final_state = captured["final_state"]
    views = final_state.components(("OCN", "LND", "ATM"))
    for view in views.values():
        assert view.grid is not None
        assert view.grid.longitude.dtype == jnp.float32
        assert view.grid.latitude.dtype == jnp.float32
        if view.grid.binary_mask is not None:
            assert view.grid.binary_mask.dtype == jnp.float32
        for _, _, value in view.iter_fields("state", "received", "sent"):
            assert value.dtype == jnp.float32

    assert_allclose_compact(
        views["ATM"].field("sea_surface_temperature", scope="received"),
        np.asarray([[280.0, 0.0], [0.0, 284.0]], dtype=np.float32),
    )
    assert_allclose_compact(
        views["ATM"].field("land_surface_temperature", scope="received"),
        np.asarray([[0.0, 282.0], [286.0, 0.0]], dtype=np.float32),
    )
