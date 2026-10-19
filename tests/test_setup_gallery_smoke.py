from __future__ import annotations

import ast
import io
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import jax.numpy as jnp
import numpy as np
import pytest
import torch
import xarray as xr

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
    ATMOSPHERE_TO_VEROS_FORCING_FIELDS,
)
from vercor.setups.gallery import run_camulator_with_veros


@pytest.mark.fast_always
def test_camulator_veros_gallery_composes_through_fake_native_boundary(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Exercise the unavailable CAMulator gallery through a finite host fake."""

    # Actual file-backed initial diagnostics; model construction remains fake.
    diagnostic_path = tmp_path / "native-initial.nc"
    xr.Dataset(
        {
            name: (("time", "latitude", "longitude"), np.full((1, 2, 2), value))
            for name, value in (
                ("TS", 280.0),
                ("FSNS", 21600 * 100.0),
                ("FLNS", -21600 * 3.0),
            )
        },
        coords={"time": [datetime(1981, 1, 3)]},
    ).to_netcdf(diagnostic_path, engine="h5netcdf")
    monkeypatch.setenv("VERCOR_CAMULATOR_WATER_VAPOR_VARIABLE", "VAPOR")
    monkeypatch.setenv("VERCOR_CAMULATOR_INITIAL_DIAGNOSTICS", str(diagnostic_path))
    monkeypatch.setattr(
        run_camulator_with_veros,
        "open",
        lambda path: io.StringIO(
            "data:\n  variables: [U, V, T, VAPOR]\npredict:\n  start_datetime: '1981-01-03 00:00:00'\n"
        ),
        raising=False,
    )

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
    )

    atmosphere_output_names = tuple(
        dict.fromkeys(
            _flatten_field_items(
                (*_flatten_field_items(ATMOSPHERE_TO_VEROS_FORCING_FIELDS),)
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

    def make_atmosphere(*, config: Any) -> Any:
        assert config.water_vapor_variable == "VAPOR"
        assert config.initial_diagnostics["time"] == datetime(1981, 1, 3)
        assert_allclose_compact(config.initial_diagnostics["TS"], 280.0)
        assert_allclose_compact(config.initial_diagnostics["FSNS"], 21600 * 100.0)
        assert_allclose_compact(config.initial_diagnostics["FLNS"], -21600 * 3.0)
        return atmosphere

    monkeypatch.setattr(run_camulator_with_veros, "make_camulator_gcm", make_atmosphere)
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
            start=kwargs["start"],
            dt_seconds=21600.0,
            steps=2,
            calendar=kwargs["calendar"],
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
    assert_allclose_compact(topology.binary_masks["LND->ATM"], land_mask)

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


@pytest.mark.fast_always
def test_veris_gallery_uses_public_import_owners() -> None:
    """Keep the runnable gallery usable as an ordinary copied setup script."""
    path = Path("vercor/setups/gallery/run_veris_dataveros_era5.py")
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.ImportFrom)
            and node.module
            and node.module.startswith("vercor")
        ):
            assert not any(part.startswith("_") for part in node.module.split("."))
