"""Private provider normalization and immutable run-level output sessions."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
import re
from typing import TYPE_CHECKING, Any, cast

import jax
import jax.numpy as jnp

from vercor.calendar import ModelDateTime
from vercor.clock import Clock
from vercor.dtypes import as_jax_index_array
from vercor.exceptions import ComponentError, CouplerError
from vercor.jax_logging import LoggerLike
from vercor.output import (
    JaxOutputContext,
    JaxOutputProvider,
    OutputContext,
    OutputFrame,
    OutputTarget,
    OutputVariable,
    PeriodOutput,
)
from vercor.output._dataset import grid_field_dims, time_coordinate_variable
from vercor.output._netcdf import write_netcdf_dataset
from vercor.output._period import (
    _sample_sum_and_counts,
    period_mean_sample_to_output_variable,
    should_write_period_output,
)
from vercor._runtime.field_transfer import select_runtime_field
from vercor._runtime.time import RuntimeStepInfo
from vercor._pytree import PyTreeNodeMixin
from vercor.state import ComponentState
from vercor.types import RuntimeArray

if TYPE_CHECKING:
    from vercor.components._adapter import _ComponentBinding
    from vercor.state import RunState

_Time = datetime | ModelDateTime
_ClockStep = tuple[int, _Time, timedelta]


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class _OutputAccumulator(PyTreeNodeMixin):
    """The sole immutable sum/count accumulator used by output coordination."""

    pytree_children = ("sum_values", "counts", "coordinate_values")
    pytree_aux_data = (
        "names",
        "select_all",
        "dims",
        "dtypes",
        "attrs",
        "coordinate_names",
        "coordinate_dims",
        "coordinate_shapes",
        "coordinate_dtypes",
        "coordinate_attrs",
        "metadata",
        "sample_dimension",
        "time_dimension",
        "dimension_order",
    )

    names: tuple[str, ...]
    select_all: bool
    dims: tuple[tuple[str, ...], ...]
    dtypes: tuple[str, ...]
    attrs: tuple[tuple[tuple[str, Any], ...], ...]
    sum_values: tuple[jax.Array, ...]
    counts: tuple[jax.Array, ...]
    coordinate_names: tuple[str, ...]
    coordinate_dims: tuple[tuple[str, ...], ...]
    coordinate_shapes: tuple[tuple[int, ...], ...]
    coordinate_dtypes: tuple[str, ...]
    coordinate_values: tuple[Any, ...]
    coordinate_attrs: tuple[tuple[tuple[str, Any], ...], ...]
    metadata: tuple[tuple[str, Any], ...]
    sample_dimension: str | None
    time_dimension: str
    dimension_order: tuple[str, ...] | None

    @classmethod
    def zeros_from_frame(
        cls,
        frame: OutputFrame,
        *,
        selected: Sequence[str] = (),
    ) -> "_OutputAccumulator":
        """Create an empty shape-stable accumulator from one provider frame."""

        names = _selected_names(frame, selected)
        dims: list[tuple[str, ...]] = []
        attrs: list[tuple[tuple[str, Any], ...]] = []
        dtypes: list[str] = []
        sums: list[jax.Array] = []
        counts: list[jax.Array] = []
        for name in names:
            sample_dims, sum_values, sample_counts = _sample_sum_and_counts(
                name,
                frame.variables[name],
                summation_dim=frame.sample_dimension,
            )
            dims.append(sample_dims)
            dtypes.append(_value_dtype(frame.variables[name].values))
            attrs.append(tuple(frame.variables[name].attrs.items()))
            sums.append(jnp.zeros_like(sum_values))
            counts.append(jnp.zeros_like(sample_counts))
        coordinate_names, coordinate_dims, coordinate_values, coordinate_attrs = (
            _coordinate_parts(frame)
        )
        return cls(
            names=names,
            select_all=not tuple(selected),
            dims=tuple(dims),
            dtypes=tuple(dtypes),
            attrs=tuple(attrs),
            sum_values=tuple(sums),
            counts=tuple(counts),
            coordinate_names=coordinate_names,
            coordinate_dims=coordinate_dims,
            coordinate_shapes=_coordinate_shapes(coordinate_values),
            coordinate_dtypes=_coordinate_dtypes(coordinate_values),
            coordinate_values=coordinate_values,
            coordinate_attrs=coordinate_attrs,
            metadata=tuple(frame.metadata.items()),
            sample_dimension=frame.sample_dimension,
            time_dimension=frame.time_dimension,
            dimension_order=frame.dimension_order,
        )

    def add_frame(self, frame: OutputFrame) -> "_OutputAccumulator":
        """Return a new accumulator containing one selected provider frame."""

        selected = _selected_names(frame, self.names)
        if selected != self.names or (
            self.select_all and tuple(frame.variables) != self.names
        ):
            raise ValueError("Output provider variables changed across samples.")
        if frame.sample_dimension != self.sample_dimension:
            raise ValueError("Output provider sample dimension changed across samples.")
        if frame.time_dimension != self.time_dimension:
            raise ValueError("Output provider time dimension changed across samples.")
        if frame.dimension_order != self.dimension_order:
            raise ValueError("Output provider dimension order changed across samples.")
        if tuple(frame.metadata.items()) != self.metadata:
            raise ValueError("Output provider metadata changed across samples.")
        coordinate_names, coordinate_dims, coordinate_values, coordinate_attrs = (
            _coordinate_parts(frame)
        )
        if (
            coordinate_names,
            coordinate_dims,
            coordinate_attrs,
        ) != (
            self.coordinate_names,
            self.coordinate_dims,
            self.coordinate_attrs,
        ) or (
            _coordinate_shapes(coordinate_values) != self.coordinate_shapes
            or _coordinate_dtypes(coordinate_values) != self.coordinate_dtypes
        ):
            raise ValueError(
                "Output provider coordinate schema changed across samples."
            )
        sums: list[jax.Array] = []
        counts: list[jax.Array] = []
        for index, name in enumerate(self.names):
            dims, sample_sum, sample_counts = _sample_sum_and_counts(
                name,
                frame.variables[name],
                summation_dim=self.sample_dimension,
            )
            if dims != self.dims[index]:
                raise ValueError(f"Output variable {name!r} dimensions changed.")
            if sample_sum.shape != self.sum_values[index].shape:
                raise ValueError(f"Output variable {name!r} shape changed.")
            if _value_dtype(frame.variables[name].values) != self.dtypes[index]:
                raise ValueError(f"Output variable {name!r} dtype changed.")
            if tuple(frame.variables[name].attrs.items()) != self.attrs[index]:
                raise ValueError(f"Output variable {name!r} attributes changed.")
            sums.append(self.sum_values[index] + sample_sum)
            counts.append(self.counts[index] + sample_counts)
        return replace(
            self,
            sum_values=tuple(sums),
            counts=tuple(counts),
            coordinate_shapes=_coordinate_shapes(coordinate_values),
            coordinate_dtypes=_coordinate_dtypes(coordinate_values),
            coordinate_values=coordinate_values,
        )

    def reset(self) -> "_OutputAccumulator":
        """Return an empty accumulator with unchanged static schema."""

        return replace(
            self,
            sum_values=tuple(jnp.zeros_like(value) for value in self.sum_values),
            counts=tuple(jnp.zeros_like(value) for value in self.counts),
        )

    def mean_frame(self) -> OutputFrame:
        """Return the finite-count mean as one immutable provider frame."""

        variables = {}
        for index, name in enumerate(self.names):
            mean_dtype = jnp.result_type(self.sum_values[index].dtype, jnp.float32)
            denominator = jnp.where(self.counts[index] > 0, self.counts[index], 1)
            finite_means = self.sum_values[index] / denominator
            mean_values = jnp.where(
                self.counts[index] > 0,
                finite_means,
                jnp.full(self.sum_values[index].shape, jnp.nan, dtype=mean_dtype),
            )
            variables[name] = OutputVariable(
                self.dims[index],
                mean_values,
                dict(self.attrs[index]),
            )
        return OutputFrame(
            variables,
            coordinates={
                name: OutputVariable(dims, values, dict(attrs))
                for name, dims, values, attrs in zip(
                    self.coordinate_names,
                    self.coordinate_dims,
                    self.coordinate_values,
                    self.coordinate_attrs,
                    strict=True,
                )
            },
            metadata=dict(self.metadata),
            time_dimension=self.time_dimension,
            dimension_order=self.dimension_order,
        )


def _selected_names(frame: OutputFrame, selected: Sequence[str]) -> tuple[str, ...]:
    """Apply the one provider-independent variable selection rule."""

    names = tuple(selected) or tuple(frame.variables)
    if not names:
        raise ValueError("Output provider returned no variables.")
    missing = next((name for name in names if name not in frame.variables), None)
    if missing is not None:
        available = ", ".join(frame.variables) or "<none>"
        raise KeyError(f"unknown output variable {missing!r}; available: {available}")
    return names


def _coordinate_parts(
    frame: OutputFrame,
) -> tuple[
    tuple[str, ...],
    tuple[tuple[str, ...], ...],
    tuple[Any, ...],
    tuple[tuple[tuple[str, Any], ...], ...],
]:
    """Split non-time coordinate values from their static PyTree schema."""

    coordinates = tuple(
        (name, variable)
        for name, variable in frame.coordinates.items()
        if name != frame.time_dimension
    )
    return (
        tuple(name for name, _ in coordinates),
        tuple(variable.dims for _, variable in coordinates),
        tuple(variable.values for _, variable in coordinates),
        tuple(tuple(variable.attrs.items()) for _, variable in coordinates),
    )


def _coordinate_shapes(values: tuple[Any, ...]) -> tuple[tuple[int, ...], ...]:
    """Return coordinate shapes without moving their values out of PyTree leaves."""

    return tuple(tuple(jnp.asarray(value).shape) for value in values)


def _coordinate_dtypes(values: tuple[Any, ...]) -> tuple[str, ...]:
    """Return stable coordinate dtypes for cross-sample schema validation."""

    return tuple(_value_dtype(value) for value in values)


def _value_dtype(value: Any) -> str:
    """Return a hashable dtype token without moving array values to the host."""

    return str(jnp.asarray(value).dtype)


class _RuntimeFieldProvider:
    """Default provider exposing time-selected declared component outputs."""

    def __init__(
        self,
        component: "_ComponentBinding",
        step_infos: RuntimeStepInfo,
    ) -> None:
        self._component = component
        self._step_infos = step_infos

    def sample(self, context: OutputContext) -> OutputFrame:
        """Return time-selected runtime fields using the ordinary public context."""
        return self.sample_jax(
            JaxOutputContext(
                context.component,
                context.state,
                context.payload,
                context.step,
                context.dt.total_seconds(),
            )
        )

    def sample_jax(self, context: JaxOutputContext) -> OutputFrame:
        """Select fields with the traced absolute index and fixed grid schema."""
        step_info = cast(
            RuntimeStepInfo,
            jax.tree_util.tree_map(
                lambda value: value[context.step],
                self._step_infos,
            ),
        )
        variables = {}
        for name in self._component.spec.outputs:
            values = select_runtime_field(
                context.state.field(name, scope="state"),
                self._component.spec.transfer,
                step_info,
                field_name=name,
                masks=self._component.spec.masks,
            )
            variables[name] = OutputVariable(
                grid_field_dims(
                    name,
                    tuple(values.shape),
                    self._component.grid.shape,
                ),
                values,
                {"component": self._component.name, "field_name": name},
            )
        return OutputFrame(
            variables,
            coordinates={
                "latitude": OutputVariable(("nlat",), self._component.grid.latitude),
                "longitude": OutputVariable(("nlon",), self._component.grid.longitude),
            },
        )


@dataclass(frozen=True)
class _OutputSchema:
    component: "_ComponentBinding"
    provider: Any
    period: PeriodOutput


@dataclass(frozen=True)
class _OutputBoundary:
    stop_step: int
    due_schema_indices: tuple[int, ...]
    period_starts: tuple[_Time, ...]
    output_filenames: tuple[str, ...]


@dataclass(frozen=True)
class _OutputPlan:
    schemas: tuple[_OutputSchema, ...]
    boundaries: tuple[_OutputBoundary, ...]
    target: OutputTarget


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class _OutputSession(PyTreeNodeMixin):
    """Immutable per-run optional accumulator bundle."""

    accumulators: tuple[_OutputAccumulator | None, ...]
    pytree_children = ("accumulators",)
    pytree_aux_data = ()

    def accumulate(
        self,
        plan: _OutputPlan,
        state: "RunState",
        *,
        step: int,
        time: _Time,
        dt: timedelta,
    ) -> "_OutputSession":
        """Sample every provider's supported capability after one concrete step."""
        return self._accumulate(
            plan, state, step=step, dt_seconds=dt.total_seconds(), host_time=(time, dt)
        )

    def accumulate_jax(
        self,
        plan: _OutputPlan,
        state: "RunState",
        *,
        step: int | RuntimeArray,
        dt_seconds: float,
    ) -> "_OutputSession":
        """Accumulate one pure numeric sample inside a fixed-shape JAX scan."""
        return self._accumulate(plan, state, step=step, dt_seconds=dt_seconds)

    def _accumulate(
        self,
        plan: _OutputPlan,
        state: "RunState",
        *,
        step: int | RuntimeArray,
        dt_seconds: float,
        host_time: tuple[_Time, timedelta] | None = None,
    ) -> "_OutputSession":
        """Share provider selection, frame checks and reduction on every backend."""
        accumulated = []
        for schema, accumulator in zip(plan.schemas, self.accumulators, strict=True):
            current = _accumulate_provider_frame(
                schema,
                accumulator,
                lambda: _sample_provider(
                    schema, state, step=step, dt_seconds=dt_seconds, host_time=host_time
                ),
            )
            if host_time is None:
                current = replace(
                    current,
                    coordinate_values=tuple(
                        jnp.asarray(value) for value in current.coordinate_values
                    ),
                )
            accumulated.append(current)
        return _OutputSession(tuple(accumulated))


def _sample_provider(
    schema: _OutputSchema,
    state: "RunState",
    *,
    step: int | RuntimeArray,
    dt_seconds: float,
    host_time: tuple[_Time, timedelta] | None,
) -> OutputFrame:
    """Prefer numeric extraction per provider, independently of runtime backend.

    Concrete host sampling does not trace component payloads or coerce their
    coordinates. Only ordinary providers require calendar time and a Python
    step index; scan eligibility is checked before abstract preparation.
    """
    runtime_state = state._component_state(schema.component.name)
    view = ComponentState._from_runtime(
        schema.component.name, schema.component.grid, runtime_state
    )
    provider = schema.provider
    if isinstance(provider, JaxOutputProvider):
        return provider.sample_jax(
            JaxOutputContext(
                schema.component._component,
                view,
                runtime_state.payload,
                as_jax_index_array(step),
                dt_seconds,
            )
        )
    if host_time is None:
        raise TypeError("scanned output requires a JaxOutputProvider")
    time, dt = host_time
    return cast(
        OutputFrame,
        provider.sample(
            OutputContext(
                schema.component._component,
                view,
                runtime_state.payload,
                cast(int, step),
                time,
                dt,
            )
        ),
    )


def _accumulate_provider_frame(
    schema: _OutputSchema,
    accumulator: _OutputAccumulator | None,
    sample: Callable[[], OutputFrame],
) -> _OutputAccumulator:
    """Apply the same frame validation and reduction to both sampling lanes."""
    try:
        frame = sample()
        if not isinstance(frame, OutputFrame):
            raise TypeError(f"must return OutputFrame; got {type(frame).__name__}")
        current = (
            accumulator
            if accumulator is not None
            else _OutputAccumulator.zeros_from_frame(
                frame,
                selected=schema.period.variables,
            )
        )
        return current.add_frame(frame)
    except (KeyError, TypeError, ValueError) as exc:
        raise ComponentError(
            f"Output provider for component {schema.component.name!r}: {exc}"
        ) from exc


def has_period_output(
    components: Mapping[str, "_ComponentBinding"],
    target: OutputTarget | None,
) -> bool:
    """Return whether the explicit target enables any declared period output."""

    return bool(
        target is not None
        and target.write_period
        and any(
            component.spec.output.period is not None
            for component in components.values()
        )
    )


def validate_output_run_state_not_traced(state: "RunState") -> None:
    """Reject any enabled I/O run carrying traced state leaves."""

    if any(
        isinstance(leaf, jax.core.Tracer) for leaf in jax.tree_util.tree_leaves(state)
    ):
        raise CouplerError(
            "Output is an I/O workflow and cannot run with traced RunState leaves. "
            "Differentiated or outer-jitted runs must pass output=None."
        )


def build_output_plan(
    components: Mapping[str, "_ComponentBinding"],
    clock: Clock,
    target: OutputTarget,
    *,
    step_infos: RuntimeStepInfo,
    clock_steps: Sequence[_ClockStep] | None = None,
) -> _OutputPlan:
    """Normalize component providers and allocate all period filenames."""

    schemas: list[_OutputSchema] = []
    for component in components.values():
        period = component.spec.output.period
        if period is None:
            continue
        provider = component.spec.output.provider or _RuntimeFieldProvider(
            component,
            step_infos,
        )
        schemas.append(_OutputSchema(component, provider, period))
    boundaries = _output_boundaries(
        tuple(schemas),
        clock,
        clock_steps=clock_steps,
    )
    return _OutputPlan(tuple(schemas), boundaries, target)


def initial_output_session(plan: _OutputPlan) -> _OutputSession:
    """Return a lazy immutable accumulator slot for every provider."""

    return _OutputSession(tuple(None for _ in plan.schemas))


def write_output_boundary(
    plan: _OutputPlan,
    session: _OutputSession,
    boundary: _OutputBoundary,
    *,
    logger: LoggerLike | None,
) -> _OutputSession:
    """Write due means and reset only the completed accumulator windows."""

    accumulators = list(session.accumulators)
    for index, period_start, filename in zip(
        boundary.due_schema_indices,
        boundary.period_starts,
        boundary.output_filenames,
        strict=True,
    ):
        schema = plan.schemas[index]
        output_path = plan.target.directory / filename
        accumulator = accumulators[index]
        try:
            plan.target.directory.mkdir(parents=True, exist_ok=True)
            if accumulator is None:
                raise ValueError("Period output requires at least one sample.")
            frame = accumulator.mean_frame()
            variables = {
                name: period_mean_sample_to_output_variable(
                    variable,
                    time_dim=frame.time_dimension,
                    dimension_order=frame.dimension_order,
                )
                for name, variable in frame.variables.items()
            }
            coordinates = dict(frame.coordinates)
            coordinates[frame.time_dimension] = time_coordinate_variable(
                period_start,
                time_dim=frame.time_dimension,
            )
            write_netcdf_dataset(
                output=str(output_path),
                coordinate_variables=coordinates,
                data_variables=variables,
                global_attrs=dict(frame.metadata) or None,
                logger=logger,
            )
        except Exception as exc:
            raise ComponentError(
                "Period output for component "
                f"{schema.component.name!r} at {str(output_path)!r}: {exc}"
            ) from exc
        accumulators[index] = accumulator.reset()
    return _OutputSession(tuple(accumulators))


def _output_boundaries(
    schemas: tuple[_OutputSchema, ...],
    clock: Clock,
    *,
    clock_steps: Sequence[_ClockStep] | None,
) -> tuple[_OutputBoundary, ...]:
    steps = tuple(clock.iter()) if clock_steps is None else tuple(clock_steps)
    if not steps:
        return ()

    window_starts = [steps[0][1] for _ in schemas]
    raw: list[tuple[int, tuple[int, ...], tuple[_Time, ...], tuple[str, ...]]] = []
    for step, time, dt in steps:
        due = tuple(
            index
            for index, schema in enumerate(schemas)
            if should_write_period_output(schema.period, time=time, dt=dt)
        )
        if due:
            period_starts = tuple(window_starts[index] for index in due)
            bases = tuple(
                f"{_safe_token(schemas[index].component.name)}.averages."
                f"{_period_filename_date(period_start, schemas[index].period)}.nc"
                for index, period_start in zip(due, period_starts, strict=True)
            )
            raw.append((step + 1, due, period_starts, bases))
            next_window_start = time + dt
            for index in due:
                window_starts[index] = next_window_start
    counts = Counter(filename for *_, filenames in raw for filename in filenames)
    used: set[str] = set()
    result: list[_OutputBoundary] = []
    request = 0
    for stop, due, period_starts, filenames in raw:
        allocated: list[str] = []
        for schema_index, period_start, filename in zip(
            due,
            period_starts,
            filenames,
            strict=True,
        ):
            candidate = filename
            if counts[filename] > 1:
                stem = filename[:-3]
                candidate = (
                    f"{stem}T{period_start.hour:02d}{period_start.minute:02d}"
                    f"{period_start.second:02d}.{period_start.microsecond:06d}."
                    f"step{stop - 1:08d}."
                    f"schema{schema_index:04d}.nc"
                )
            collision = 0
            while candidate in used:
                stem = candidate[:-3]
                candidate = f"{stem}.record{request:08d}.collision{collision:04d}.nc"
                collision += 1
            used.add(candidate)
            allocated.append(candidate)
            request += 1
        result.append(_OutputBoundary(stop, due, period_starts, tuple(allocated)))
    return tuple(result)


def _period_filename_date(period_start: _Time, period: PeriodOutput) -> str:
    """Return a filename date token at the configured cadence's precision."""

    format_by_frequency = {
        "step": "%Y-%m-%d",
        "day": "%Y-%m-%d",
        "month": "%Y-%m",
        "year": "%Y",
    }
    return period_start.strftime(format_by_frequency[period.frequency])


def _safe_token(value: str) -> str:
    """Return a deterministic path-safe component token."""

    token = re.sub(r"[^A-Za-z0-9_-]+", "-", value).strip("-_").lower()
    return token or "component"


__all__: list[str] = []
