"""Bundled slab and data factories support opt-in configurable period output."""

from __future__ import annotations

from datetime import datetime
from inspect import Parameter, signature
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import h5netcdf
import jax.numpy as jnp
import numpy as np
import pytest

from tests._coverage_support import make_test_grid
from vercor.clock import Clock
from vercor.components import (
    CallableComponent,
    ComponentSpec,
    DataComponent,
    TransferPolicy,
)
from vercor.coupler import Coupler
from vercor.dtypes import DTypePolicy
from vercor.exchanges import Exchange
from vercor.output import OutputSpec, OutputTarget, PeriodOutput
from vercor.runtime import RuntimeOptions
from vercor._runtime.time import build_runtime_step_info
from vercor.setups import (
    make_era5_atmosphere,
    make_era5_land,
    make_era5_ocean,
    make_erainterim_ocean,
    make_slab_atmosphere,
    make_slab_land,
    make_slab_ocean,
    make_slab_seaice,
)
from vercor.setups._data import jcm_land as jcm_land_module
from vercor.setups._data._component_helpers import (
    time_interpolated_data_component,
)

_SLAB_FACTORIES = (
    make_slab_atmosphere,
    make_slab_land,
    make_slab_ocean,
    make_slab_seaice,
)
_DATA_FACTORIES = (
    make_era5_atmosphere,
    make_era5_land,
    make_era5_ocean,
    make_erainterim_ocean,
)


def _assert_output_disabled_by_default(component: Any) -> None:
    assert component.spec.output == OutputSpec()


@pytest.mark.parametrize(
    "factory",
    _SLAB_FACTORIES,
)
def test_all_bundled_slab_factories_disable_output_by_default(factory: Any) -> None:
    _assert_output_disabled_by_default(factory(make_test_grid(name="slab-output")))


@pytest.mark.parametrize("factory", _SLAB_FACTORIES)
def test_slab_factory_accepts_keyword_only_output_spec(factory: Any) -> None:
    output_parameter = signature(factory).parameters["output"]
    assert output_parameter.kind is Parameter.KEYWORD_ONLY
    assert output_parameter.default is None
    custom_output = OutputSpec(period=PeriodOutput(frequency="month"))

    component = factory(
        make_test_grid(name="configured-slab"),
        output=custom_output,
    )

    assert component.spec.output is custom_output
    assert component.spec.transfer == TransferPolicy("current")


def test_bundled_output_rejects_invalid_override() -> None:
    with pytest.raises(TypeError, match="output must be OutputSpec or None"):
        make_slab_atmosphere(
            make_test_grid(name="invalid-output"),
            output=cast(Any, "month"),
        )


def test_shared_data_factory_disables_output_by_default() -> None:
    grid = make_test_grid(name="data-output")
    component = time_interpolated_data_component(
        name="DATA",
        grid=grid,
        fields={"air_temperature": jnp.full(grid.shape, 280.0)},
        inputs=("forcing",),
        outputs=("air_temperature",),
        initial_fields={"forcing": 1.0},
    )

    _assert_output_disabled_by_default(component)


def test_shared_data_factory_accepts_output_spec() -> None:
    grid = make_test_grid(name="configured-data-output")
    custom_output = OutputSpec(period=PeriodOutput(frequency="month"))

    component = time_interpolated_data_component(
        name="DATA",
        grid=grid,
        fields={"air_temperature": jnp.full(grid.shape, 280.0)},
        outputs=("air_temperature",),
        output=custom_output,
    )

    assert component.spec.output is custom_output


@pytest.mark.parametrize("factory", _DATA_FACTORIES)
def test_public_data_factory_accepts_keyword_only_output_spec(factory: Any) -> None:
    output_parameter = signature(factory).parameters["output"]
    assert output_parameter.kind is Parameter.KEYWORD_ONLY
    assert output_parameter.default is None
    custom_output = OutputSpec(period=PeriodOutput(frequency="month"))

    component = factory(output=custom_output)

    assert component.spec.output is custom_output


@pytest.mark.parametrize("factory", _DATA_FACTORIES)
def test_public_data_factory_disables_output_by_default(factory: Any) -> None:
    component = factory()

    _assert_output_disabled_by_default(component)


def _make_test_jcm_land(
    monkeypatch: pytest.MonkeyPatch,
    output: OutputSpec | None = None,
) -> Any:
    grid = make_test_grid(name="jcm-land-output")
    monkeypatch.setattr(
        jcm_land_module,
        "create_land_mask_from_ocean",
        lambda **kwargs: (
            jnp.ones(grid.shape),
            jnp.zeros(grid.shape),
        ),
    )
    coords = SimpleNamespace(
        horizontal=SimpleNamespace(
            longitudes=jnp.deg2rad(jnp.asarray([0.0, 180.0])),
            latitudes=jnp.deg2rad(jnp.asarray([-45.0, 45.0])),
        )
    )
    forcing = SimpleNamespace(
        stl_am=jnp.full(grid.shape, 280.0),
        soilw_am=jnp.full(grid.shape, 0.25),
    )

    if output is None:
        return jcm_land_module.make_jcm_land(
            cast(Any, coords),
            cast(Any, forcing),
            grid,
        )
    return jcm_land_module.make_jcm_land(
        cast(Any, coords),
        cast(Any, forcing),
        grid,
        output=output,
    )


def test_direct_jcm_land_data_factory_disables_output_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    component = _make_test_jcm_land(monkeypatch)
    _assert_output_disabled_by_default(component)


def test_direct_jcm_land_data_factory_accepts_output_spec(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    custom_output = OutputSpec(period=PeriodOutput(frequency="month"))

    component = _make_test_jcm_land(monkeypatch, custom_output)

    assert component.spec.output is custom_output


def test_slab_period_file_contains_declared_outputs_only(tmp_path: Path) -> None:
    grid = make_test_grid(name="slab-period")
    component = make_slab_atmosphere(
        grid,
        name="ATM",
        output=OutputSpec(period=PeriodOutput(frequency="step")),
    )
    coupler = Coupler(
        Clock(datetime(2000, 1, 1), dt_seconds=3600.0, steps=1),
        components=(component,),
        run_order=(component.name,),
        log_level="WARNING",
    )

    coupler.run(
        output=OutputTarget(
            tmp_path,
            write_final_fields=False,
            write_snapshots=False,
        )
    )

    with h5netcdf.File(tmp_path / "atm.averages.2000-01-01.nc", "r") as dataset:
        field_variables = set(dataset.variables) - {"time", "latitude", "longitude"}
        assert field_variables == set(component.spec.outputs)


def test_slab_omitted_output_writes_no_period_files(tmp_path: Path) -> None:
    grid = make_test_grid(name="disabled-slab-period")
    component = make_slab_atmosphere(grid)
    coupler = Coupler(
        Clock(datetime(2000, 1, 1), dt_seconds=3600.0, steps=1),
        components=(component,),
        run_order=(component.name,),
        log_level="WARNING",
    )

    coupler.run(
        output=OutputTarget(
            tmp_path,
            write_final_fields=False,
            write_snapshots=False,
        )
    )

    assert not tuple(tmp_path.glob("*.averages.*.nc"))


def test_slab_omitted_output_still_writes_final_fields(tmp_path: Path) -> None:
    grid = make_test_grid(name="final-fields-without-period")
    component = make_slab_atmosphere(grid)
    coupler = Coupler(
        Clock(datetime(2000, 1, 1), dt_seconds=3600.0, steps=1),
        components=(component,),
        run_order=(component.name,),
        log_level="WARNING",
    )

    coupler.run(
        output=OutputTarget(
            tmp_path,
            write_snapshots=False,
        )
    )

    assert not tuple(tmp_path.glob("*.averages.*.nc"))
    assert (tmp_path / "atm.runtime_fields.nc").is_file()


def test_slab_month_output_averages_coupler_step_samples(tmp_path: Path) -> None:
    grid = make_test_grid(name="monthly-slab-period")
    start = datetime(2000, 1, 30)
    dt_seconds = 86_400.0

    def run_without_output(steps: int) -> np.ndarray:
        component = make_slab_atmosphere(grid, output=OutputSpec())
        coupler = Coupler(
            Clock(start, dt_seconds=dt_seconds, steps=steps),
            components=(component,),
            run_order=(component.name,),
            log_level="WARNING",
        )
        return np.asarray(
            coupler.run().component(component.name).field("temperature_2m")
        )

    expected_mean = 0.5 * (run_without_output(1) + run_without_output(2))
    component = make_slab_atmosphere(
        grid,
        output=OutputSpec(
            period=PeriodOutput(
                frequency="month",
                variables=("temperature_2m",),
            )
        ),
    )
    coupler = Coupler(
        Clock(start, dt_seconds=dt_seconds, steps=2),
        components=(component,),
        run_order=(component.name,),
        log_level="WARNING",
    )

    coupler.run(
        output=OutputTarget(
            tmp_path,
            write_final_fields=False,
            write_snapshots=False,
        )
    )

    with h5netcdf.File(tmp_path / "atm.averages.2000-01.nc", "r") as dataset:
        assert set(dataset.variables) == {
            "time",
            "latitude",
            "longitude",
            "temperature_2m",
        }
        np.testing.assert_allclose(
            dataset.variables["temperature_2m"][0],
            expected_mean,
        )


def test_data_period_file_contains_declared_outputs_only(tmp_path: Path) -> None:
    grid = make_test_grid(name="data-period")
    component = time_interpolated_data_component(
        name="DATA",
        grid=grid,
        fields={"air_temperature": jnp.full((12, *grid.shape), 280.0)},
        inputs=("forcing",),
        outputs=("air_temperature",),
        initial_fields={"forcing": 1.0},
        output=OutputSpec(period=PeriodOutput(frequency="step")),
    )
    coupler = Coupler(
        Clock(datetime(2000, 1, 1), dt_seconds=3600.0, steps=1),
        components=(component,),
        run_order=(component.name,),
        log_level="WARNING",
    )

    coupler.run(
        output=OutputTarget(
            tmp_path,
            write_final_fields=False,
            write_snapshots=False,
        )
    )

    with h5netcdf.File(tmp_path / "data.averages.2000-01-01.nc", "r") as dataset:
        assert dataset.variables["air_temperature"].shape == (1, *grid.shape)
        np.testing.assert_allclose(dataset.variables["air_temperature"][0], 280.0)
        assert "forcing" not in dataset.variables


@pytest.mark.parametrize("backend", ["host", "jax"])
def test_linear_data_month_output_averages_exact_exported_slices(
    backend: str,
    tmp_path: Path,
) -> None:
    grid = make_test_grid(name=f"linear-output-{backend}")
    forcing = jnp.arange(12.0, dtype=jnp.float64)[
        :, jnp.newaxis, jnp.newaxis
    ] * jnp.ones((12, *grid.shape), dtype=jnp.float64)
    component = time_interpolated_data_component(
        name="DATA",
        grid=grid,
        fields={"air_temperature": forcing},
        outputs=("air_temperature",),
        output=OutputSpec(
            period=PeriodOutput(
                frequency="month",
                variables=("air_temperature",),
            )
        ),
    )
    clock = Clock(
        datetime(2001, 1, 1),
        dt_seconds=86_400.0,
        steps=59,
        calendar="noleap",
    )
    coupler = Coupler(
        clock,
        components=(component,),
        run_order=(component.name,),
        runtime=RuntimeOptions(backend=cast(Any, backend)),
        log_level="WARNING",
    )

    coupler.run(
        output=OutputTarget(
            tmp_path,
            write_final_fields=False,
            write_snapshots=False,
        )
    )

    metadata = build_runtime_step_info(clock)
    record_values = np.arange(12.0)
    selected = (
        np.asarray(metadata.monthly_weight_left)
        * record_values[np.asarray(metadata.monthly_index_left)]
        + np.asarray(metadata.monthly_weight_right)
        * record_values[np.asarray(metadata.monthly_index_right)]
    )
    expected = (float(np.mean(selected[:31])), float(np.mean(selected[31:])))
    paths = sorted(tmp_path.glob("data.averages.*.nc"))
    actual = []
    for path in paths:
        with h5netcdf.File(path, "r") as dataset:
            values = np.asarray(dataset.variables["air_temperature"])
            assert values.shape == (1, *grid.shape)
            actual.append(float(np.mean(values)))

    assert [path.name for path in paths] == [
        "data.averages.2001-01.nc",
        "data.averages.2001-02.nc",
    ]
    np.testing.assert_allclose(actual, expected)
    assert not np.isclose(actual[0], actual[1])


@pytest.mark.fast_always
@pytest.mark.parametrize("enable_x64", [False, True])
def test_linear_data_period_output_matches_sent_field_precision(
    enable_x64: bool, tmp_path: Path
) -> None:
    grid = make_test_grid(name="linear-output-precision")
    forcing = np.arange(12.0)[:, None, None] * np.ones((12, *grid.shape))
    source = time_interpolated_data_component(
        name="DATA",
        grid=grid,
        fields={"air_temperature": forcing},
        outputs=("air_temperature",),
        output=OutputSpec(period=PeriodOutput("step")),
    )
    target = CallableComponent(
        "SINK", grid, lambda fields: {}, spec=ComponentSpec(inputs=("air_temperature",))
    )
    coupler = Coupler(
        Clock(datetime(2001, 2, 1), 86400.0, 1),
        components=(source, target),
        exchanges=(
            Exchange("DATA", "SINK", ("air_temperature",), quantity="intensive"),
        ),
        run_order=("DATA", "SINK"),
        runtime=RuntimeOptions(dtype=DTypePolicy(enable_x64=enable_x64)),
        log_level="WARNING",
    )

    result = coupler.run(
        output=OutputTarget(tmp_path, write_final_fields=False, write_snapshots=False)
    )

    sent = np.asarray(result.component("DATA").field("air_temperature", scope="sent"))
    path = next(tmp_path.glob("data.averages.*.nc"))
    with h5netcdf.File(path, "r") as dataset:
        output = np.asarray(dataset.variables["air_temperature"])[0]
    expected_dtype = np.dtype(np.float64 if enable_x64 else np.float32)
    assert sent.dtype == expected_dtype
    assert output.dtype == expected_dtype
    np.testing.assert_array_equal(output, sent)


def test_daily_data_month_output_averages_exact_exported_slices(
    tmp_path: Path,
) -> None:
    grid = make_test_grid(name="daily-output")
    forcing = jnp.arange(365.0, dtype=jnp.float64)[
        :, jnp.newaxis, jnp.newaxis
    ] * jnp.ones((365, *grid.shape), dtype=jnp.float64)
    component = DataComponent(
        "DATA",
        grid,
        {"air_temperature": forcing},
        spec=ComponentSpec(
            outputs=("air_temperature",),
            transfer=TransferPolicy("daily"),
            output=OutputSpec(
                period=PeriodOutput(
                    frequency="month",
                    variables=("air_temperature",),
                )
            ),
        ),
    )
    coupler = Coupler(
        Clock(
            datetime(2001, 1, 1),
            dt_seconds=86_400.0,
            steps=59,
            calendar="noleap",
        ),
        components=(component,),
        run_order=(component.name,),
        runtime=RuntimeOptions(backend="jax"),
        log_level="WARNING",
    )

    coupler.run(
        output=OutputTarget(
            tmp_path,
            write_final_fields=False,
            write_snapshots=False,
        )
    )

    paths = sorted(tmp_path.glob("data.averages.*.nc"))
    actual = []
    for path in paths:
        with h5netcdf.File(path, "r") as dataset:
            values = np.asarray(dataset.variables["air_temperature"])
            assert values.shape == (1, *grid.shape)
            actual.append(float(np.mean(values)))

    assert [path.name for path in paths] == [
        "data.averages.2001-01.nc",
        "data.averages.2001-02.nc",
    ]
    np.testing.assert_allclose(actual, (15.0, 44.5))


def test_custom_components_remain_period_output_opt_in(tmp_path: Path) -> None:
    grid = make_test_grid(name="custom-output")
    component = CallableComponent(
        "CUSTOM",
        grid,
        lambda fields: {"air_temperature": fields["air_temperature"] + 1.0},
        spec=ComponentSpec(
            outputs=("air_temperature",),
            initial_fields={"air_temperature": 280.0},
        ),
    )
    coupler = Coupler(
        Clock(datetime(2000, 1, 1), dt_seconds=3600.0, steps=1),
        components=(component,),
        run_order=(component.name,),
        log_level="WARNING",
    )

    assert component.spec.output.period is None
    coupler.run(
        output=OutputTarget(
            tmp_path,
            write_final_fields=False,
            write_snapshots=False,
        )
    )
    assert not tuple(tmp_path.glob("*.averages.*.nc"))
