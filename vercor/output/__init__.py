"""Stable public output contracts for components and run-level I/O."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, Protocol, TypeAlias, runtime_checkable

from vercor.calendar import ModelDateTime as _ModelDateTime
from vercor.components._protocol import Component as _Component
from vercor.jax_logging import LoggerLike as _LoggerLike
from vercor._field_names import freeze_name_sequence as _freeze_name_sequence
from vercor.state import ComponentState as _ComponentState
from vercor.types import RuntimeArray as _RuntimeArray

_OutputFrequency: TypeAlias = Literal["step", "day", "month", "year"]


def _frozen_mapping(
    values: Mapping[str, Any] | None,
    *,
    label: str,
) -> Mapping[str, Any]:
    """Return an insertion-ordered read-only snapshot of one mapping."""

    if values is not None and not isinstance(values, Mapping):
        raise TypeError(f"{label} must be a mapping or None")
    snapshot = dict(values or {})
    invalid = next(
        (name for name in snapshot if not isinstance(name, str) or not name),
        None,
    )
    if invalid is not None:
        raise TypeError(f"{label} names must be non-empty strings")
    return MappingProxyType(snapshot)


def _canonical_metadata_value(value: Any, *, label: str) -> Any:
    """Return one NetCDF attribute value with safe static equality semantics."""

    if not isinstance(value, (str, bytes)) and hasattr(value, "tolist"):
        try:
            unpacked = value.tolist()
        except Exception as exc:
            raise TypeError(f"{label} values must be concrete metadata") from exc
        return _canonical_metadata_value(unpacked, label=label)
    if not isinstance(value, (str, bytes)) and hasattr(value, "item"):
        try:
            return _canonical_metadata_value(value.item(), label=label)
        except Exception as exc:
            raise TypeError(f"{label} values must be concrete metadata") from exc
    if isinstance(value, (list, tuple)):
        return tuple(_canonical_metadata_value(item, label=label) for item in value)
    if value is None or isinstance(value, (str, bytes, bool, int, float, complex)):
        return value
    raise TypeError(
        f"{label} values must be scalar or concrete array-like NetCDF metadata"
    )


def _frozen_metadata(
    values: Mapping[str, Any] | None,
    *,
    label: str,
) -> Mapping[str, Any]:
    """Freeze and canonicalize metadata for writing and PyTree schemas."""

    snapshot = _frozen_mapping(values, label=label)
    return MappingProxyType(
        {
            name: _canonical_metadata_value(value, label=label)
            for name, value in snapshot.items()
        }
    )


@dataclass(frozen=True)
class OutputVariable:
    """Array values together with NetCDF dimensions and variable metadata."""

    dims: tuple[str, ...]
    values: Any
    attrs: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate dimension names and freeze variable metadata."""

        if not isinstance(self.dims, tuple) or not all(
            isinstance(dim, str) and dim for dim in self.dims
        ):
            raise TypeError("OutputVariable.dims must be a tuple of non-empty names")
        if len(set(self.dims)) != len(self.dims):
            raise ValueError("OutputVariable.dims must be unique")
        value_rank = _value_rank(self.values)
        if value_rank != len(self.dims):
            raise ValueError(
                "OutputVariable dimension count "
                f"{len(self.dims)} does not match value rank {value_rank}"
            )
        object.__setattr__(
            self,
            "attrs",
            _frozen_metadata(self.attrs, label="OutputVariable.attrs"),
        )


def _value_rank(values: Any) -> int:
    """Return array rank without importing a host-array implementation."""

    shape = getattr(values, "shape", None)
    if shape is not None:
        return len(shape)
    if isinstance(values, (list, tuple)):
        return 1 if not values else 1 + _value_rank(values[0])
    return 0


@dataclass(frozen=True, init=False)
class OutputFrame:
    """One provider sample containing variables, coordinates, and metadata.

    Variable and coordinate dimensions and per-variable attributes are carried
    by :class:`OutputVariable`. ``metadata`` contains dataset-global attrs.
    Every mapping is snapshotted so a sampled frame cannot change underneath
    the immutable output session. ``sample_dimension`` identifies an in-frame
    axis reduced into the period mean. ``time_dimension`` names the leading
    coordinate added to a written mean. ``dimension_order`` optionally gives
    the preferred NetCDF order; dimensions absent from it retain their sampled
    order.
    """

    variables: Mapping[str, OutputVariable]
    coordinates: Mapping[str, OutputVariable]
    metadata: Mapping[str, Any]
    sample_dimension: str | None
    time_dimension: str
    dimension_order: tuple[str, ...] | None

    def __init__(
        self,
        variables: Mapping[str, OutputVariable],
        *,
        coordinates: Mapping[str, OutputVariable] | None = None,
        metadata: Mapping[str, Any] | None = None,
        sample_dimension: str | None = None,
        time_dimension: str = "time",
        dimension_order: Sequence[str] | None = None,
    ) -> None:
        """Create and validate one immutable provider frame."""

        frozen_variables = _frozen_mapping(variables, label="variables")
        frozen_coordinates = _frozen_mapping(coordinates, label="coordinates")
        invalid = next(
            (
                name
                for name, value in (
                    *frozen_variables.items(),
                    *frozen_coordinates.items(),
                )
                if not isinstance(value, OutputVariable)
            ),
            None,
        )
        if invalid is not None:
            raise TypeError(f"OutputFrame entry {invalid!r} must be OutputVariable")
        overlap = set(frozen_variables).intersection(frozen_coordinates)
        if overlap:
            raise ValueError(
                f"OutputFrame name {sorted(overlap)[0]!r} is both a variable and coordinate"
            )
        object.__setattr__(self, "variables", frozen_variables)
        object.__setattr__(self, "coordinates", frozen_coordinates)
        object.__setattr__(
            self,
            "metadata",
            _frozen_metadata(metadata, label="metadata"),
        )
        if sample_dimension is not None and (
            not isinstance(sample_dimension, str) or not sample_dimension
        ):
            raise TypeError("sample_dimension must be a non-empty string or None")
        if not isinstance(time_dimension, str) or not time_dimension:
            raise TypeError("time_dimension must be a non-empty string")
        normalized_order = (
            None
            if dimension_order is None
            else _freeze_name_sequence(
                dimension_order,
                label="OutputFrame.dimension_order",
            )
        )
        if normalized_order is not None and not all(
            isinstance(dim, str) and dim for dim in normalized_order
        ):
            raise TypeError("dimension_order entries must be non-empty strings")
        if normalized_order is not None and len(set(normalized_order)) != len(
            normalized_order
        ):
            raise ValueError("dimension_order entries must be unique")
        object.__setattr__(self, "sample_dimension", sample_dimension)
        object.__setattr__(self, "time_dimension", time_dimension)
        object.__setattr__(self, "dimension_order", normalized_order)


@dataclass(frozen=True)
class OutputContext:
    """Public post-step component view supplied to an output provider.

    ``step`` is zero-based. ``time`` is the end of that step and therefore the
    model time represented by ``state`` and ``payload``.
    """

    component: _Component
    state: _ComponentState
    payload: Any | None
    step: int
    time: datetime | _ModelDateTime
    dt: timedelta


@runtime_checkable
class OutputProvider(Protocol):
    """Structural extension point that samples one component output frame."""

    def sample(self, context: OutputContext) -> OutputFrame:
        """Return one output frame for the supplied post-step state."""
        ...


@dataclass(frozen=True)
class JaxOutputContext:
    """Numeric post-step view for pure, fixed-schema output sampling.

    ``step`` is an absolute zero-based int32 clock index, traced inside JAX
    scans and concrete on host runs. ``dt_seconds`` is the static coupling
    interval in seconds. Calendar formatting belongs to host output; numeric
    sampling must not perform I/O or mutate native state.
    """

    component: _Component
    state: _ComponentState
    payload: Any | None
    step: int | _RuntimeArray
    dt_seconds: float


@runtime_checkable
class JaxOutputProvider(OutputProvider, Protocol):
    """Optional numeric sampling capability used independently of the backend.

    Numeric frames have a fixed schema, metadata, shapes and dtypes across the
    run. Numeric values and coordinates may evolve; Python time coordinates
    are omitted. Ordinary providers need only implement :class:`OutputProvider`.
    Period output prefers ``sample_jax`` wherever this capability is present;
    all-numeric JAX runs additionally accumulate samples inside scans.
    """

    def sample_jax(self, context: JaxOutputContext) -> OutputFrame:
        """Return a pure numeric frame from the supplied post-step state."""
        ...


@dataclass(frozen=True, init=False)
class PeriodOutput:
    """Mean-output cadence and uniform provider-variable selection policy.

    An empty ``variables`` tuple selects every variable returned by the
    provider. A non-empty tuple selects that ordered subset, duplicate names
    are removed, and unknown names fail with a component-scoped error.
    """

    frequency: _OutputFrequency
    variables: tuple[str, ...]

    def __init__(
        self,
        frequency: Literal["step", "day", "month", "year"] = "step",
        variables: Sequence[str] = (),
    ) -> None:
        """Validate cadence and freeze a de-duplicated variable selection."""

        if frequency not in ("step", "day", "month", "year"):
            raise ValueError("frequency must be one of 'step', 'day', 'month', 'year'")
        normalized = tuple(
            dict.fromkeys(
                _freeze_name_sequence(
                    variables,
                    label="PeriodOutput.variables",
                )
            )
        )
        if not all(isinstance(variable, str) and variable for variable in normalized):
            raise ValueError("variables entries must be non-empty strings")
        object.__setattr__(self, "frequency", frequency)
        object.__setattr__(self, "variables", normalized)


@dataclass(frozen=True)
class SnapshotContext:
    """Public payload passed to component snapshot writers.

    ``time`` is the model time represented by the final state: the clock start
    for a zero-step run and the end of the last step otherwise. The coordinator
    allocates the collision-safe ``output_path`` before invoking the writer.
    """

    component: _Component
    state: _ComponentState
    payload: Any | None
    output_path: Path
    time: datetime | _ModelDateTime
    logger: _LoggerLike | None


SnapshotWriter: TypeAlias = Callable[[SnapshotContext], None]


@dataclass(frozen=True)
class OutputSpec:
    """Declare a component provider, period policy, and optional snapshot."""

    provider: OutputProvider | None = None
    period: PeriodOutput | None = None
    snapshot_writer: SnapshotWriter | None = None

    def __post_init__(self) -> None:
        """Validate nested component output policy immediately."""

        if self.provider is not None and not callable(
            getattr(self.provider, "sample", None)
        ):
            raise TypeError("provider must define callable sample(context) or be None")
        if self.period is not None and not isinstance(self.period, PeriodOutput):
            raise TypeError("period must be PeriodOutput or None")
        if self.snapshot_writer is not None and not callable(self.snapshot_writer):
            raise TypeError("snapshot_writer must be callable or None")


@dataclass(frozen=True, init=False)
class OutputTarget:
    """Enable selected run-level outputs beneath one directory.

    Passing no target to :meth:`vercor.Coupler.run` performs no I/O regardless
    of component output declarations.
    """

    directory: Path
    write_period: bool
    write_final_fields: bool
    write_snapshots: bool

    def __init__(
        self,
        directory: str | Path,
        *,
        write_period: bool = True,
        write_final_fields: bool = True,
        write_snapshots: bool = True,
    ) -> None:
        """Create a validated run-level I/O target."""

        if not isinstance(directory, (str, Path)):
            raise TypeError("directory must be a path-like string or Path")
        for name, value in (
            ("write_period", write_period),
            ("write_final_fields", write_final_fields),
            ("write_snapshots", write_snapshots),
        ):
            if not isinstance(value, bool):
                raise TypeError(f"{name} must be bool")
        object.__setattr__(self, "directory", Path(directory))
        object.__setattr__(self, "write_period", write_period)
        object.__setattr__(self, "write_final_fields", write_final_fields)
        object.__setattr__(self, "write_snapshots", write_snapshots)

    @property
    def enabled(self) -> bool:
        """Return whether this target requests any I/O."""

        return self.write_period or self.write_final_fields or self.write_snapshots


# Dataclasses preserve postponed field annotations in their generated
# signatures. Replace only the generated context annotations with their public
# runtime objects so introspection never exposes private import aliases.
OutputContext.__init__.__annotations__.update(
    {
        "component": _Component,
        "state": _ComponentState,
        "time": datetime | _ModelDateTime,
    }
)
SnapshotContext.__init__.__annotations__.update(
    {
        "component": _Component,
        "state": _ComponentState,
        "time": datetime | _ModelDateTime,
        "logger": _LoggerLike | None,
    }
)


__all__ = [
    "JaxOutputContext",
    "JaxOutputProvider",
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
