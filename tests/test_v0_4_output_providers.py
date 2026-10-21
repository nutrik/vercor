"""VerCOR 0.4 unified output-provider and run-level coordinator contracts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields
from datetime import datetime
import inspect
from pathlib import Path
from types import MappingProxyType
from typing import Any, cast, get_type_hints

import h5netcdf
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests._coverage_support import make_test_grid
from vercor.clock import Clock
from vercor.components import (
    CallableComponent,
    ComponentSpec,
    LifecycleHooks,
    SetupResult,
    StepContext,
    StepResult,
)
from vercor.coupler import Coupler
from vercor.exceptions import ComponentError, CouplerError
import vercor.output as output_api
import vercor.output._runtime as output_runtime
import vercor.output._session as output_session
import vercor.runtime as runtime
from vercor.state import RunState


def _api(name: str) -> Any:
    """Return a required 0.4 output symbol with a focused RED failure."""

    assert hasattr(output_api, name), f"vercor.output is missing 0.4 symbol {name}"
    return getattr(output_api, name)


def _component(
    *,
    name: str = "model",
    output: Any | None = None,
    execution: str = "jax",
) -> CallableComponent:
    grid = make_test_grid(name=f"output-{name}")

    def step(
        fields: Mapping[str, Any],
        context: StepContext,
    ) -> Mapping[str, Any]:
        _ = context
        return {
            "temperature": fields["temperature"] + 1.0,
            "salinity": fields["salinity"] + 2.0,
        }

    return CallableComponent(
        name,
        grid,
        step,
        spec=ComponentSpec(
            outputs=("temperature", "salinity"),
            initial_fields={"temperature": 0.0, "salinity": 10.0},
            execution=cast(Any, execution),
            output=output,
        ),
    )


def _coupler(
    component: Any,
    *,
    steps: int = 2,
    dt_seconds: float = 86_400.0,
    backend: Any = "jax",
) -> Coupler:
    return Coupler(
        Clock(datetime(2000, 1, 1), dt_seconds, steps),
        components=(component,),
        run_order=(component.name,),
        runtime=runtime.RuntimeOptions(backend=backend),
        log_level="WARNING",
    )


def _period_spec(
    *,
    provider: Any | None = None,
    variables: tuple[str, ...] = (),
    frequency: str = "step",
    snapshot_writer: Any | None = None,
) -> Any:
    return _api("OutputSpec")(
        provider=provider,
        period=_api("PeriodOutput")(
            frequency=frequency,
            variables=variables,
        ),
        snapshot_writer=snapshot_writer,
    )


def _target(directory: Path, **kwargs: Any) -> Any:
    return _api("OutputTarget")(directory, **kwargs)


def _read(path: Path, name: str) -> np.ndarray:
    with h5netcdf.File(path, "r") as dataset:
        return np.asarray(dataset.variables[name])


def test_output_module_exports_exact_v0_4_contract() -> None:
    assert output_api.__all__ == [
        "OutputContext",
        "OutputFrame",
        "OutputProvider",
        "OutputSpec",
        "OutputTarget",
        "OutputVariable",
        "PeriodOutput",
        "SnapshotContext",
        "SnapshotWriter",
    ]
    assert not hasattr(output_api, "OutputConfig")
    assert "vercor.components.contracts" not in Path(
        "vercor/output/__init__.py"
    ).read_text(encoding="utf-8")
    assert not hasattr(output_api, "OutputFrequency")


def test_output_contract_signatures_are_minimal() -> None:
    provider = _api("OutputProvider")
    spec = _api("OutputSpec")
    target = _api("OutputTarget")

    assert list(inspect.signature(provider.sample).parameters) == ["self", "context"]
    assert list(inspect.signature(spec).parameters) == [
        "provider",
        "period",
        "snapshot_writer",
    ]
    assert list(inspect.signature(target).parameters) == [
        "directory",
        "write_period",
        "write_final_fields",
        "write_snapshots",
    ]
    assert list(inspect.signature(Coupler.run).parameters) == [
        "self",
        "state",
        "output",
    ]
    assert not hasattr(Coupler, "write_outputs")


def test_output_frame_and_configuration_own_immutable_snapshots(tmp_path: Path) -> None:
    variable = _api("OutputVariable")(("x",), jnp.asarray([1.0]))
    variables = {"temperature": variable}
    coordinates = {"x": variable}
    metadata = {"title": "native"}
    frame = _api("OutputFrame")(
        variables,
        coordinates=coordinates,
        metadata=metadata,
    )
    period_variables = ["temperature", "temperature"]
    period = _api("PeriodOutput")(variables=period_variables)
    target = _target(tmp_path)

    variables["other"] = variable
    coordinates.clear()
    metadata["title"] = "changed"
    period_variables.append("other")

    assert isinstance(frame.variables, MappingProxyType)
    assert tuple(frame.variables) == ("temperature",)
    assert tuple(frame.coordinates) == ("x",)
    assert frame.metadata == {"title": "native"}
    assert period.variables == ("temperature",)
    assert target.directory == tmp_path
    assert target.write_period
    assert target.write_final_fields
    assert target.write_snapshots


@pytest.mark.parametrize(
    ("factory", "kwargs", "error"),
    [
        ("OutputFrame", {"variables": []}, "variables must be a mapping"),
        ("OutputFrame", {"variables": {"x": object()}}, "OutputVariable"),
        ("OutputSpec", {"provider": object()}, "provider.*sample"),
        ("OutputSpec", {"snapshot_writer": object()}, "snapshot_writer"),
        ("OutputTarget", {"directory": 3}, "directory"),
        ("PeriodOutput", {"variables": "temperature"}, "sequence"),
    ],
)
def test_output_configuration_rejects_invalid_nested_values(
    factory: str,
    kwargs: dict[str, Any],
    error: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=error):
        _api(factory)(**kwargs)


def test_output_none_performs_no_io_or_provider_sampling(tmp_path: Path) -> None:
    class ExplodingProvider:
        def sample(self, context: Any) -> Any:
            _ = context
            raise AssertionError("disabled provider was sampled")

    component = _component(output=_period_spec(provider=ExplodingProvider()))
    coupler = _coupler(component, steps=1)

    final = coupler.run(output=None)

    assert float(final.component("model").field("temperature")[0, 0]) == 1.0
    assert not tuple(tmp_path.iterdir())


def test_runtime_default_provider_writes_selected_period_variables(
    tmp_path: Path,
) -> None:
    component = _component(output=_period_spec(variables=("salinity",)))

    _coupler(component, steps=2).run(
        output=_target(
            tmp_path,
            write_final_fields=False,
            write_snapshots=False,
        )
    )

    paths = sorted(tmp_path.glob("model.averages.*.nc"))
    assert len(paths) == 2
    with h5netcdf.File(paths[0], "r") as dataset:
        assert "salinity" in dataset.variables
        assert "temperature" not in dataset.variables
    np.testing.assert_allclose(_read(paths[1], "salinity"), 14.0)


@pytest.mark.fast_always
@pytest.mark.parametrize("axis_values", [[0.0, 1.0], (0.0, 1.0)])
def test_custom_provider_writes_array_like_and_scalar_coordinates(
    axis_values: Any, tmp_path: Path
) -> None:
    """Accepted public coordinate values normalize at the I/O boundary."""

    class Provider:
        def sample(self, context: Any) -> Any:
            _ = context
            return _api("OutputFrame")(
                {"temperature": _api("OutputVariable")(("x",), [280.0, 281.0])},
                coordinates={
                    "x": _api("OutputVariable")(("x",), axis_values, {"units": "m"}),
                    "height": _api("OutputVariable")((), 2.0, {"units": "m"}),
                },
            )

    component = _component(output=_period_spec(provider=Provider()))
    _coupler(component, steps=1).run(
        output=_target(tmp_path, write_final_fields=False, write_snapshots=False)
    )

    path = next(tmp_path.glob("model.averages.*.nc"))
    with h5netcdf.File(path, "r") as dataset:
        np.testing.assert_array_equal(dataset.variables["x"], [0.0, 1.0])
        assert dataset.variables["x"].attrs["units"] == "m"
        assert dataset.variables["height"].shape == ()
        assert float(np.asarray(dataset.variables["height"])) == 2.0
        assert dataset.variables["height"].attrs["units"] == "m"
        np.testing.assert_array_equal(
            dataset.variables["temperature"], [[280.0, 281.0]]
        )


def test_custom_provider_receives_public_context_and_uses_same_filtering(
    tmp_path: Path,
) -> None:
    contexts: list[Any] = []

    class NativeProvider:
        def sample(self, context: Any) -> Any:
            contexts.append(context)
            value = context.state.field("temperature")
            variable = _api("OutputVariable")(
                ("native_y", "native_x"),
                value + 100.0,
                {"units": "K"},
            )
            dropped = _api("OutputVariable")(
                ("native_y", "native_x"),
                value - 100.0,
            )
            return _api("OutputFrame")(
                {"native_temperature": variable, "drop_me": dropped},
                coordinates={
                    "native_y": _api("OutputVariable")(
                        ("native_y",), jnp.arange(value.shape[0])
                    ),
                    "native_x": _api("OutputVariable")(
                        ("native_x",), jnp.arange(value.shape[1])
                    ),
                },
                metadata={"source": "third-party"},
            )

    provider = NativeProvider()
    component = _component(
        output=_period_spec(provider=provider, variables=("native_temperature",))
    )

    _coupler(component, steps=1).run(
        output=_target(
            tmp_path,
            write_final_fields=False,
            write_snapshots=False,
        )
    )

    assert contexts
    context = contexts[-1]
    assert context.component is component
    assert context.state.name == "model"
    assert context.payload is None
    assert context.step == 0
    assert context.time == datetime(2000, 1, 2)
    path = tmp_path / "model.averages.2000-01-01.nc"
    with h5netcdf.File(path, "r") as dataset:
        assert tuple(dataset.variables["native_temperature"].dimensions) == (
            "time",
            "native_y",
            "native_x",
        )
        assert "drop_me" not in dataset.variables
        assert dataset.attrs["source"] == "third-party"


def test_custom_provider_unknown_selection_is_rejected(tmp_path: Path) -> None:
    class NativeProvider:
        def sample(self, context: Any) -> Any:
            return _api("OutputFrame")(
                {
                    "native": _api("OutputVariable")(
                        ("nlat", "nlon"), context.state.field("temperature")
                    )
                }
            )

    component = _component(
        output=_period_spec(provider=NativeProvider(), variables=("missing",))
    )
    with pytest.raises(ComponentError, match="component 'model'.*missing"):
        _coupler(component, steps=1).run(
            output=_target(
                tmp_path,
                write_final_fields=False,
                write_snapshots=False,
            )
        )


def test_custom_backend_period_output_remains_core_owned(tmp_path: Path) -> None:
    class RecordingBackend:
        def __init__(self) -> None:
            self.calls = 0

        def execute(
            self,
            state: RunState,
            *,
            context: runtime.ExecutionContext,
            chunk: runtime.ExecutionChunk,
            driver: runtime.RuntimeDriver,
        ) -> RunState:
            self.calls += 1
            for plan in chunk.steps:
                state = driver.run_step(state, plan)
            return state

    backend = RecordingBackend()
    component = _component(output=_period_spec(variables=("temperature",)))

    final = _coupler(component, steps=2, backend=backend).run(
        output=_target(
            tmp_path,
            write_final_fields=False,
            write_snapshots=False,
        )
    )

    assert backend.calls == 2
    assert float(final.component("model").field("temperature")[0, 0]) == 2.0
    paths = sorted(tmp_path.glob("model.averages.*.nc"))
    assert len(paths) == 2
    np.testing.assert_allclose(_read(paths[0], "temperature"), 1.0)
    np.testing.assert_allclose(_read(paths[1], "temperature"), 2.0)


def test_period_filenames_are_safe_and_collision_free(tmp_path: Path) -> None:
    first = _component(name="first/model", output=_period_spec())
    second = _component(name="first model", output=_period_spec())
    coupler = Coupler(
        Clock(datetime(2000, 1, 1), 3_600.0, 2),
        components=(first, second),
        run_order=(first.name, second.name),
        log_level="WARNING",
    )

    coupler.run(
        output=_target(
            tmp_path,
            write_final_fields=False,
            write_snapshots=False,
        )
    )

    paths = tuple(tmp_path.glob("*.nc"))
    assert len(paths) == 4
    assert len({path.name for path in paths}) == 4
    assert all("/" not in path.name for path in paths)
    assert all(path.parent == tmp_path for path in paths)


def test_output_target_writes_final_fields_and_snapshots_in_one_run(
    tmp_path: Path,
) -> None:
    snapshot_contexts: list[Any] = []

    def write_snapshot(context: Any) -> None:
        snapshot_contexts.append(context)
        context.output_path.write_text("snapshot")

    component = _component(output=_api("OutputSpec")(snapshot_writer=write_snapshot))

    final = _coupler(component, steps=1).run(output=_target(tmp_path))

    assert float(final.component("model").field("temperature")[0, 0]) == 1.0
    final_fields_path = tmp_path / "model.runtime_fields.nc"
    assert final_fields_path.is_file()
    np.testing.assert_allclose(_read(final_fields_path, "state_temperature"), 1.0)
    np.testing.assert_allclose(_read(final_fields_path, "state_salinity"), 12.0)
    assert (tmp_path / "model.snapshot.nc").read_text() == "snapshot"
    assert snapshot_contexts[0].component is component
    assert snapshot_contexts[0].state.name == "model"
    assert snapshot_contexts[0].time == datetime(2000, 1, 2)


def test_final_fields_preserve_leading_dimensions(tmp_path: Path) -> None:
    grid = make_test_grid(name="three-dimensional-output")
    values = jnp.arange(3 * 2 * 2, dtype=float).reshape((3, 2, 2))
    component = CallableComponent(
        "profile-model",
        grid,
        lambda fields: {"pressure": fields["pressure"]},
        spec=ComponentSpec(
            outputs=("pressure",),
            initial_fields={"pressure": values},
        ),
    )

    _coupler(component, steps=0).run(
        output=_target(
            tmp_path,
            write_period=False,
            write_snapshots=False,
        )
    )

    with h5netcdf.File(tmp_path / "profile-model.runtime_fields.nc", "r") as dataset:
        pressure = dataset.variables["state_pressure"]
        assert pressure.dimensions == ("pressure_dim_0", "nlat", "nlon")
        np.testing.assert_allclose(np.asarray(pressure), values)


def test_final_and_snapshot_filenames_are_safe_and_collision_free(
    tmp_path: Path,
) -> None:
    snapshot_paths: list[Path] = []

    def snapshot(context: Any) -> None:
        snapshot_paths.append(context.output_path)
        context.output_path.write_text(context.component.name)

    first = _component(
        name="first/model",
        output=_api("OutputSpec")(snapshot_writer=snapshot),
    )
    second = _component(
        name="first model",
        output=_api("OutputSpec")(snapshot_writer=snapshot),
    )
    coupler = Coupler(
        Clock(datetime(2000, 1, 1), 3_600.0, 0),
        components=(first, second),
        run_order=(),
        log_level="WARNING",
    )

    coupler.run(output=_target(tmp_path, write_period=False))

    final_paths = tuple(tmp_path.glob("*.runtime_fields.nc"))
    assert len(final_paths) == 2
    assert len(snapshot_paths) == 2
    assert len({path.name for path in (*final_paths, *snapshot_paths)}) == 4
    assert all(path.parent == tmp_path for path in (*final_paths, *snapshot_paths))


def test_output_target_flags_disable_each_output_kind(tmp_path: Path) -> None:
    called: list[bool] = []
    component = _component(
        output=_period_spec(snapshot_writer=lambda context: called.append(True))
    )

    _coupler(component, steps=1).run(
        output=_target(
            tmp_path,
            write_period=False,
            write_final_fields=False,
            write_snapshots=False,
        )
    )

    assert not called
    assert not tuple(tmp_path.iterdir())


def test_any_enabled_output_rejects_traced_state_before_io(tmp_path: Path) -> None:
    component = _component(output=_api("OutputSpec")())
    coupler = _coupler(component, steps=1)
    state = coupler.initial_state()
    target = _target(
        tmp_path,
        write_period=False,
        write_snapshots=False,
    )

    with pytest.raises(CouplerError, match="Output is an I/O workflow"):
        jax.jit(lambda value: coupler.run(value, output=target))(state)


def test_period_accumulation_is_immutable_and_preserves_nan_counts() -> None:
    import vercor.output._session as session_module

    assert tuple(field.name for field in fields(session_module._OutputSchema)) == (
        "component",
        "provider",
        "period",
    )
    accumulator_type = getattr(session_module, "_OutputAccumulator", None)
    assert accumulator_type is not None, "missing unified immutable accumulator"
    variable = _api("OutputVariable")
    empty = accumulator_type.zeros_from_frame(
        _api("OutputFrame")({"x": variable(("x",), jnp.asarray([0.0, 0.0]))}),
        selected=("x",),
    )
    first = empty.add_frame(
        _api("OutputFrame")({"x": variable(("x",), jnp.asarray([1.0, jnp.nan]))})
    )
    second = first.add_frame(
        _api("OutputFrame")({"x": variable(("x",), jnp.asarray([3.0, 5.0]))})
    )

    np.testing.assert_allclose(empty.sum_values[0], [0.0, 0.0])
    np.testing.assert_allclose(first.sum_values[0], [1.0, 0.0])
    np.testing.assert_allclose(second.mean_frame().variables["x"].values, [2.0, 5.0])


def test_daily_output_accumulates_multiple_samples_and_resets_windows(
    tmp_path: Path,
) -> None:
    component = _component(
        output=_period_spec(
            variables=("temperature",),
            frequency="day",
        )
    )

    _coupler(component, steps=4, dt_seconds=43_200.0).run(
        output=_target(
            tmp_path,
            write_final_fields=False,
            write_snapshots=False,
        )
    )

    paths = sorted(tmp_path.glob("model.averages.*.nc"))
    assert len(paths) == 2
    np.testing.assert_allclose(_read(paths[0], "temperature"), 1.5)
    np.testing.assert_allclose(_read(paths[1], "temperature"), 3.5)


def test_components_with_independent_period_cadences_share_one_session(
    tmp_path: Path,
) -> None:
    each_step = _component(
        name="step-model",
        output=_period_spec(variables=("temperature",), frequency="step"),
    )
    daily = _component(
        name="daily-model",
        output=_period_spec(variables=("temperature",), frequency="day"),
    )
    coupler = Coupler(
        Clock(datetime(2000, 1, 1), 43_200.0, 4),
        components=(each_step, daily),
        run_order=(each_step.name, daily.name),
        log_level="WARNING",
    )

    coupler.run(
        output=_target(
            tmp_path,
            write_final_fields=False,
            write_snapshots=False,
        )
    )

    assert len(tuple(tmp_path.glob("step-model.averages.*.nc"))) == 4
    daily_paths = sorted(tmp_path.glob("daily-model.averages.*.nc"))
    assert len(daily_paths) == 2
    np.testing.assert_allclose(_read(daily_paths[0], "temperature"), 1.5)


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize(
    ("selection", "expected"),
    [
        ((), ("salinity", "temperature")),
        (("salinity",), ("salinity",)),
        (("salinity", "salinity"), ("salinity",)),
    ],
)
def test_variable_filtering_is_identical_for_runtime_and_native_providers(
    native: bool,
    selection: tuple[str, ...],
    expected: tuple[str, ...],
    tmp_path: Path,
) -> None:
    class NativeProvider:
        def sample(self, context: Any) -> Any:
            return _api("OutputFrame")(
                {
                    "temperature": _api("OutputVariable")(
                        ("nlat", "nlon"), context.state.field("temperature")
                    ),
                    "salinity": _api("OutputVariable")(
                        ("nlat", "nlon"), context.state.field("salinity")
                    ),
                }
            )

    component = _component(
        output=_period_spec(
            provider=NativeProvider() if native else None,
            variables=selection,
        )
    )
    case_dir = tmp_path / f"case-{native}-{len(selection)}"
    _coupler(component, steps=1).run(
        output=_target(
            case_dir,
            write_final_fields=False,
            write_snapshots=False,
        )
    )

    with h5netcdf.File(case_dir / "model.averages.2000-01-01.nc", "r") as dataset:
        actual = tuple(
            sorted(
                name
                for name in dataset.variables
                if name in {"temperature", "salinity"}
            )
        )
    assert actual == expected


def test_disabled_output_never_samples_or_snapshots_even_in_output_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ExplodingProvider:
        def sample(self, context: Any) -> Any:
            _ = context
            raise AssertionError("disabled provider sampled")

    def explode_snapshot(context: Any) -> None:
        _ = context
        raise AssertionError("disabled snapshot called")

    component = _component(
        output=_period_spec(
            provider=ExplodingProvider(),
            snapshot_writer=explode_snapshot,
        )
    )
    monkeypatch.chdir(tmp_path)
    coupler = _coupler(component, steps=1)

    coupler.run(output=None)
    coupler.run(
        output=_target(
            tmp_path / "disabled",
            write_period=False,
            write_final_fields=False,
            write_snapshots=False,
        )
    )

    assert not tuple(tmp_path.iterdir())


@pytest.mark.parametrize("kind", ["period", "final", "snapshot"])
def test_each_enabled_io_kind_rejects_traced_state(kind: str, tmp_path: Path) -> None:
    snapshot = (lambda context: None) if kind == "snapshot" else None
    component = _component(
        output=(
            _period_spec(snapshot_writer=snapshot)
            if kind == "period"
            else _api("OutputSpec")(snapshot_writer=snapshot)
        )
    )
    coupler = _coupler(component, steps=1)
    state = coupler.initial_state()
    target = _target(
        tmp_path,
        write_period=kind == "period",
        write_final_fields=kind == "final",
        write_snapshots=kind == "snapshot",
    )

    with pytest.raises(CouplerError, match="Output is an I/O workflow"):
        jax.jit(lambda value: coupler.run(value, output=target))(state)


def test_all_disabled_target_remains_jit_and_gradient_compatible(
    tmp_path: Path,
) -> None:
    class ExplodingProvider:
        def sample(self, context: Any) -> Any:
            _ = context
            raise AssertionError("disabled provider sampled")

    component = _component(output=_period_spec(provider=ExplodingProvider()))
    coupler = _coupler(component, steps=1)
    state = coupler.initial_state()
    target = _target(
        tmp_path,
        write_period=False,
        write_final_fields=False,
        write_snapshots=False,
    )

    compiled = jax.jit(lambda value: coupler.run(value, output=target))(state)

    def objective(value: jax.Array) -> jax.Array:
        seeded = state.replace_fields("model", {"temperature": jnp.full((2, 2), value)})
        return jnp.sum(
            coupler.run(seeded, output=target).component("model").field("temperature")
        )

    assert float(compiled.component("model").field("temperature")[0, 0]) == 1.0
    assert float(jax.grad(objective)(jnp.asarray(2.0))) == 4.0
    assert not tuple(tmp_path.iterdir())


@pytest.mark.parametrize(
    "failure",
    [
        "return",
        "sample_error",
        "empty",
        "variables",
        "dimensions",
        "shape",
        "attrs",
        "coordinates",
        "coordinate_shape",
        "coordinate_rank",
        "coordinate_dtype",
        "coordinate_attrs",
        "metadata",
        "variable_dtype",
        "sample_dimension",
        "time_dimension",
        "dimension_order",
    ],
)
def test_provider_return_and_cross_sample_schema_are_validated(
    failure: str,
    tmp_path: Path,
) -> None:
    class InvalidProvider:
        def sample(self, context: Any) -> Any:
            if failure == "return":
                return {"temperature": context.state.field("temperature")}
            if failure == "sample_error":
                raise ValueError("native extraction failed")
            if failure == "empty":
                return _api("OutputFrame")({})

            dims = (
                ("cell",)
                if failure == "dimensions" and context.step == 1
                else ("nlat", "nlon")
            )
            values = context.state.field("temperature")
            if dims == ("cell",):
                values = values.reshape(-1)
            elif failure == "shape" and context.step == 1:
                values = jnp.ones((2, 3))
            attrs = {"units": "degC" if context.step == 0 else "K"}
            if failure != "attrs":
                attrs = {"units": "K"}
            coordinate_dim = (
                "row" if failure == "coordinates" and context.step == 1 else "nlat"
            )
            coordinate_values = (
                jnp.arange(3.0)
                if failure == "coordinate_shape" and context.step == 1
                else jnp.arange(2.0)
            )
            if failure == "coordinate_rank":
                coordinate_values = jnp.ones((2, 2))
            elif failure == "coordinate_dtype" and context.step == 1:
                coordinate_values = jnp.arange(2, dtype=jnp.int32)
            coordinate_attrs = {
                "units": (
                    "radians"
                    if failure == "coordinate_attrs" and context.step == 1
                    else "degrees_north"
                )
            }
            metadata = {
                "source": (
                    "changed"
                    if failure == "metadata" and context.step == 1
                    else "stable"
                )
            }
            time_dimension = (
                "valid_time"
                if failure == "time_dimension" and context.step == 1
                else "time"
            )
            dimension_order = (
                ("time", "nlat", "nlon")
                if failure == "dimension_order" and context.step == 1
                else None
            )
            sample_dimension = (
                "sample"
                if failure == "sample_dimension" and context.step == 1
                else None
            )
            variables = {
                "temperature": _api("OutputVariable")(
                    dims,
                    (
                        values.astype(jnp.float32)
                        if failure == "variable_dtype" and context.step == 1
                        else values.astype(jnp.float64)
                    ),
                    attrs,
                )
            }
            if failure == "variables" and context.step == 1:
                variables["new_variable"] = _api("OutputVariable")(
                    ("nlat", "nlon"), values
                )
            return _api("OutputFrame")(
                variables,
                coordinates={
                    "latitude": _api("OutputVariable")(
                        (coordinate_dim,), coordinate_values, coordinate_attrs
                    )
                },
                metadata=metadata,
                sample_dimension=sample_dimension,
                time_dimension=time_dimension,
                dimension_order=dimension_order,
            )

    component = _component(output=_period_spec(provider=InvalidProvider()))
    with pytest.raises(
        ComponentError,
        match=(
            "OutputFrame|native extraction failed|no variables|variables changed|"
            "dimensions changed|"
            "shape changed|dtype changed|attributes changed|dimension count|"
            "coordinate schema changed|metadata "
            "changed|sample dimension changed|time dimension changed|dimension order "
            "changed"
        ),
    ):
        _coupler(component, steps=2).run(
            output=_target(
                tmp_path,
                write_final_fields=False,
                write_snapshots=False,
            )
        )


def test_period_writer_errors_include_component_and_filename(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    component = _component(output=_period_spec())

    def fail_write(**kwargs: Any) -> None:
        raise ValueError(f"bad dataset at {kwargs['output']}")

    monkeypatch.setattr(output_session, "write_netcdf_dataset", fail_write)

    with pytest.raises(
        ComponentError,
        match=r"component 'model'.*model\.averages\.2000-01-01\.nc.*bad dataset",
    ):
        _coupler(component, steps=1).run(
            output=_target(
                tmp_path,
                write_final_fields=False,
                write_snapshots=False,
            )
        )


def test_period_directory_errors_include_component_and_filename(tmp_path: Path) -> None:
    component = _component(output=_period_spec())
    blocked_directory = tmp_path / "not-a-directory"
    blocked_directory.write_text("blocked")

    with pytest.raises(
        ComponentError,
        match=r"component 'model'.*model\.averages\.2000-01-01\.nc",
    ):
        _coupler(component, steps=1).run(
            output=_target(
                blocked_directory,
                write_final_fields=False,
                write_snapshots=False,
            )
        )


def test_final_field_writer_errors_include_component_and_filename(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    component = _component(output=_api("OutputSpec")())

    def fail_write(*args: Any, **kwargs: Any) -> None:
        raise ValueError("final writer exploded")

    monkeypatch.setattr(
        output_runtime,
        "write_runtime_component_view_to_netcdf",
        fail_write,
    )

    with pytest.raises(
        ComponentError,
        match=r"component 'model'.*model\.runtime_fields\.nc.*final writer exploded",
    ):
        _coupler(component, steps=0).run(
            output=_target(
                tmp_path,
                write_period=False,
                write_snapshots=False,
            )
        )


def test_snapshot_writer_errors_include_component_and_filename(tmp_path: Path) -> None:
    def fail_snapshot(context: Any) -> None:
        raise ValueError("snapshot exploded")

    component = _component(output=_api("OutputSpec")(snapshot_writer=fail_snapshot))

    with pytest.raises(
        ComponentError,
        match=r"component 'model'.*model\.snapshot\.nc.*snapshot exploded",
    ):
        _coupler(component, steps=0).run(
            output=_target(
                tmp_path,
                write_period=False,
                write_final_fields=False,
            )
        )


def test_output_frame_rejects_duplicate_dimension_order() -> None:
    with pytest.raises(ValueError, match="dimension_order.*unique"):
        _api("OutputFrame")(
            {"temperature": _api("OutputVariable")(("x",), jnp.ones((2,)))},
            dimension_order=("time", "time", "x"),
        )


def test_provider_samples_post_step_payload_and_end_time(tmp_path: Path) -> None:
    grid = make_test_grid(name="payload-output")
    contexts: list[Any] = []

    class PayloadProvider:
        def sample(self, context: Any) -> Any:
            contexts.append(context)
            return _api("OutputFrame")(
                {"payload_value": _api("OutputVariable")((), context.payload)}
            )

    def setup(component: Any, context: Any) -> SetupResult:
        _ = component, context
        return SetupResult(payload=jnp.asarray(0.0))

    def step(
        fields: Mapping[str, Any],
        context: StepContext,
        payload: Any,
    ) -> StepResult:
        _ = context
        return StepResult(
            {"temperature": fields["temperature"] + 1.0},
            payload=payload + 1.0,
        )

    component = CallableComponent(
        "payload-model",
        grid,
        step,
        spec=ComponentSpec(
            outputs=("temperature",),
            initial_fields={"temperature": 0.0},
            lifecycle=LifecycleHooks(setup=setup),
            output=_period_spec(provider=PayloadProvider()),
        ),
    )

    _coupler(component, steps=2).run(
        output=_target(
            tmp_path,
            write_final_fields=False,
            write_snapshots=False,
        )
    )

    assert [float(context.payload) for context in contexts] == [1.0, 2.0]
    assert [context.time for context in contexts] == [
        datetime(2000, 1, 2),
        datetime(2000, 1, 3),
    ]
    paths = sorted(tmp_path.glob("payload-model.averages.*.nc"))
    assert [path.name for path in paths] == [
        "payload-model.averages.2000-01-01.nc",
        "payload-model.averages.2000-01-02.nc",
    ]
    np.testing.assert_allclose(_read(paths[0], "payload_value"), [1.0])
    np.testing.assert_allclose(_read(paths[1], "payload_value"), [2.0])


def test_sample_dimension_reduces_every_native_sample_in_period_window(
    tmp_path: Path,
) -> None:
    class MultiSampleProvider:
        def sample(self, context: Any) -> Any:
            value = context.state.field("temperature")
            return _api("OutputFrame")(
                {
                    "temperature": _api("OutputVariable")(
                        ("sample", "nlat", "nlon"),
                        jnp.stack((value, value + 2.0)),
                    )
                },
                sample_dimension="sample",
            )

    component = _component(
        output=_period_spec(
            provider=MultiSampleProvider(),
            frequency="day",
        )
    )

    _coupler(component, steps=2, dt_seconds=43_200.0).run(
        output=_target(
            tmp_path,
            write_final_fields=False,
            write_snapshots=False,
        )
    )

    path = tmp_path / "model.averages.2000-01-01.nc"
    with h5netcdf.File(path, "r") as dataset:
        assert dataset.variables["temperature"].dimensions == (
            "time",
            "nlat",
            "nlon",
        )
        np.testing.assert_allclose(dataset.variables["temperature"][:], 2.5)


def test_output_signatures_pin_defaults_kinds_and_public_annotations() -> None:
    run_signature = inspect.signature(Coupler.run)
    target_signature = inspect.signature(_api("OutputTarget"))

    assert run_signature.parameters["state"].default is None
    assert run_signature.parameters["output"].kind is inspect.Parameter.KEYWORD_ONLY
    assert run_signature.parameters["output"].default is None
    assert "OutputTarget" in str(run_signature.parameters["output"].annotation)
    assert "'_" not in str(run_signature.parameters["state"].annotation)
    assert "'_" not in str(run_signature.parameters["output"].annotation)
    assert "'_" not in str(run_signature.return_annotation)
    assert (
        target_signature.parameters["directory"].kind
        is inspect.Parameter.POSITIONAL_OR_KEYWORD
    )
    for name in ("write_period", "write_final_fields", "write_snapshots"):
        assert target_signature.parameters[name].kind is inspect.Parameter.KEYWORD_ONLY
        assert target_signature.parameters[name].default is True

    expected_parameters = {
        "OutputContext": ("component", "state", "payload", "step", "time", "dt"),
        "OutputFrame": (
            "variables",
            "coordinates",
            "metadata",
            "sample_dimension",
            "time_dimension",
            "dimension_order",
        ),
        "OutputSpec": ("provider", "period", "snapshot_writer"),
        "OutputTarget": (
            "directory",
            "write_period",
            "write_final_fields",
            "write_snapshots",
        ),
        "OutputVariable": ("dims", "values", "attrs"),
        "PeriodOutput": ("frequency", "variables"),
        "SnapshotContext": (
            "component",
            "state",
            "payload",
            "output_path",
            "time",
            "logger",
        ),
    }
    for owner_name, parameters in expected_parameters.items():
        owner = _api(owner_name)
        signature = inspect.signature(owner)
        assert tuple(signature.parameters) == parameters
        assert "'_" not in str(signature)
        for hint in get_type_hints(owner).values():
            assert "vercor._" not in str(hint)
