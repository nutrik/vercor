"""Verify native metadata/staggering and shared period/snapshot coordination."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import xarray as xr

from tests.test_veris_component import small_coupler, small_ocean
from tests.test_veris_rollouts import native_payload
from vercor.components import CallableComponent, ComponentSpec
from vercor.exceptions import ComponentError
from vercor.output import OutputContext, OutputSpec, OutputTarget, PeriodOutput
from vercor.state import ComponentState
from vercor.grids import RectilinearGrid
from vercor.setups._external.veris_runtime import VerisPayload


@pytest.mark.fast_always
@pytest.mark.parametrize("with_edges", [False, True])
def test_tagged_provider_preserves_native_axes_units_and_faces(
    with_edges: bool,
) -> None:
    from vercor.setups._external.veris_output import veris_output_provider

    grid, _ = small_ocean()
    assert grid.longitude_edges is not None and grid.latitude_edges is not None
    expected_x = np.asarray(grid.longitude_edges)[1:]
    expected_y = np.asarray(grid.latitude_edges)[1:]
    if not with_edges:
        grid = RectilinearGrid(
            "inferred",
            longitude=grid.longitude,
            latitude=grid.latitude,
            binary_mask=grid.binary_mask,
        )
    tagged = jnp.arange(42.0).reshape(7, 6)
    state = SimpleNamespace(
        Area=0.5 * jnp.ones_like(tagged),
        hIceMean=0.25 * jnp.ones_like(tagged),
        hSnowMean=tagged,
        TSurf=260.0 + tagged,
        uIce=100 + tagged,
        vIce=200 + tagged,
    )
    diagnostics = SimpleNamespace(
        **{
            name: 300 + tagged
            for name in (
                "IcePenetSW",
                "OceanStressU",
                "OceanStressV",
                "EmPmR",
                "forc_salt_surface",
            )
        }
    )
    payload = VerisPayload(state, diagnostics)
    component = CallableComponent("CICE", grid, lambda f: {}, spec=ComponentSpec())
    context = OutputContext(
        component,
        ComponentState("CICE", grid),
        payload,
        1,
        datetime(2000, 1, 1),
        timedelta(seconds=600),
    )
    frame = veris_output_provider().sample(context)
    np.testing.assert_array_equal(
        frame.variables["hSnowMean"].values, tagged[2:-2, 2:-2].T
    )
    np.testing.assert_array_equal(
        frame.variables["uIce"].values, (100 + tagged)[2:-2, 2:-2].T
    )
    np.testing.assert_array_equal(frame.variables["hIceMean"].values, 0.25)
    assert frame.variables["Area"].dims == ("y_center", "x_center")
    assert frame.variables["uIce"].dims == ("y_center", "x_face")
    assert frame.variables["vIce"].dims == ("y_face", "x_center")
    for name, units in {
        "Area": "1",
        "hIceMean": "m",
        "TSurf": "K",
        "uIce": "m s-1",
        "OceanStressU": "N m-2",
        "EmPmR": "kg m-2 s-1",
        "forc_salt_surface": "unknown",
    }.items():
        assert frame.variables[name].attrs["units"] == units
    np.testing.assert_array_equal(frame.coordinates["x_center"].values, grid.longitude)
    np.testing.assert_array_equal(frame.coordinates["y_center"].values, grid.latitude)
    np.testing.assert_array_equal(frame.coordinates["x_face"].values, expected_x)
    np.testing.assert_array_equal(frame.coordinates["y_face"].values, expected_y)
    assert set(frame.coordinates) == {
        "time",
        "x_center",
        "y_center",
        "x_face",
        "y_face",
    }


@pytest.mark.fast_always
def test_provider_requires_native_payload() -> None:
    from vercor.setups._external.veris_output import veris_output_provider

    grid, _ = small_ocean()
    component = CallableComponent("CICE", grid, lambda f: {}, spec=ComponentSpec())
    context = OutputContext(
        component,
        ComponentState("CICE", grid),
        None,
        0,
        datetime(2000, 1, 1),
        timedelta(seconds=600),
    )
    with pytest.raises(ValueError, match="payload"):
        veris_output_provider().sample(context)


@pytest.mark.slow
@pytest.mark.parametrize(
    "variables", [(), ("Area", "hIceMean"), ("unknown_native_variable",)]
)
def test_shared_period_selection_and_snapshot_metadata(
    tmp_path: Path, variables: tuple[str, ...]
) -> None:
    from vercor.setups import VerisConfig

    config = VerisConfig(
        model_timestep=timedelta(seconds=300),
        custom_parameters={"nEVPsteps": 8},
        initial_conditions={"Area": 0.5, "hIceMean": 0.25, "TSurf": 260.0},
        output=OutputSpec(period=PeriodOutput(frequency="step", variables=variables)),
    )
    coupler, component, _, _ = small_coupler(config=config)
    if variables == ("unknown_native_variable",):
        with pytest.raises(
            ComponentError, match="unknown output variable.*unknown_native_variable"
        ):
            coupler.run(output=OutputTarget(tmp_path))
        return
    final = coupler.run(output=OutputTarget(tmp_path))
    payload = native_payload(final)
    periods = sorted(tmp_path.glob("cice.averages.*.nc"))
    assert len(periods) == 2
    with xr.open_dataset(periods[-1]) as ds:
        assert (
            set(ds.data_vars) == set(variables)
            if variables
            else len(ds.data_vars) == 11
        )
        np.testing.assert_allclose(
            ds["Area"][0], final.component("CICE").field("ice_fraction")
        )
        np.testing.assert_allclose(
            ds["hIceMean"][0], payload.state.hIceMean[2:-2, 2:-2].T
        )
        assert ds["hIceMean"].attrs["units"] == "m"
        # Shared period files label the period start; snapshots represent the
        # final post-step state at the end of the second interval.
        assert ds.time.values[0] == np.datetime64("2000-01-01T00:10:00")
        if "uIce" in ds:
            assert ds["uIce"].dims == ("time", "y_center", "x_face")
    with xr.open_dataset(tmp_path / "cice.snapshot.nc") as ds:
        assert ds.time.values[0] == np.datetime64("2000-01-01T00:20:00")
        np.testing.assert_allclose(
            ds["Area"][0], final.component("CICE").field("ice_fraction")
        )
        assert ds["forc_salt_surface"].attrs["units"] == "unknown"
        assert ds["OceanStressV"].dims == ("time", "y_face", "x_center")
    with xr.open_dataset(tmp_path / "cice.runtime_fields.nc") as ds:
        np.testing.assert_allclose(ds["fmask_FORCING_CICE"], component.grid.binary_mask)


@pytest.mark.fast_always
def test_custom_output_owners_are_preserved() -> None:
    from vercor.setups import VerisConfig, make_veris

    grid, geometry = small_ocean()

    class Provider:
        def sample(self, context: OutputContext) -> Any:
            raise AssertionError("unexpected output")

    def snapshot(context: Any) -> None:
        raise AssertionError("unexpected snapshot")

    provider = Provider()
    component = make_veris(
        grid,
        geometry,
        config=VerisConfig(
            output=OutputSpec(provider=provider, snapshot_writer=snapshot)
        ),
    )
    assert component.spec.output.provider is provider
    assert component.spec.output.snapshot_writer is snapshot


@pytest.mark.slow
def test_output_disabled_outer_jit_performs_no_io(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from vercor.setups import VerisConfig

    class ForbiddenProvider:
        def sample(self, context: OutputContext) -> Any:
            raise AssertionError("provider called during output-free rollout")

    def forbidden_snapshot(context: Any) -> None:
        raise AssertionError("snapshot called during output-free rollout")

    base = VerisConfig(
        model_timestep=timedelta(seconds=300),
        custom_parameters={"nEVPsteps": 8},
        initial_conditions={"Area": 0.5, "hIceMean": 0.25},
    )
    config = replace(
        base,
        output=OutputSpec(
            provider=ForbiddenProvider(),
            period=PeriodOutput(frequency="step"),
            snapshot_writer=forbidden_snapshot,
        ),
    )
    coupler, _, _, _ = small_coupler(config=config, steps=1)
    initial = coupler.initial_state()
    monkeypatch.chdir(tmp_path)
    final = jax.jit(coupler.run)(initial)
    assert np.isfinite(np.asarray(final.component("CICE").field("ice_fraction"))).all()
    assert not tuple(tmp_path.iterdir())
