"""JAX-backed streaming accumulators for period-average output variables."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import jax
import jax.numpy as jnp

from vercor.dtypes import as_jax_real_array, jax_index_dtype


@dataclass(frozen=True)
class PeriodAverageSample:
    """One JAX-backed output variable sample with dimension metadata."""

    dims: tuple[str, ...]
    values: Any
    attrs: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AccumulatedPeriodVariable:
    """Running sum and per-element finite-value counts for one variable."""

    dims: tuple[str, ...]
    sum_values: jax.Array
    counts: jax.Array
    attrs: Mapping[str, Any] = field(default_factory=dict)

    def mean_sample(self) -> PeriodAverageSample:
        """Return the current period mean, preserving ``nanmean`` semantics."""

        mean_dtype = jnp.result_type(self.sum_values.dtype, jnp.float32)
        denominator = jnp.where(self.counts > 0, self.counts, 1)
        finite_means = self.sum_values / denominator
        mean_values = jnp.where(
            self.counts > 0,
            finite_means,
            jnp.full(self.sum_values.shape, jnp.nan, dtype=mean_dtype),
        )
        return PeriodAverageSample(
            dims=self.dims,
            values=mean_values,
            attrs=dict(self.attrs),
        )


class PeriodAverageAccumulator:
    """Accumulate named output variables as running sums and finite counts."""

    def __init__(self) -> None:
        """Create an empty period-average accumulator."""

        self._variables: dict[str, AccumulatedPeriodVariable] = {}

    @property
    def empty(self) -> bool:
        """Return whether no variable samples have been accumulated."""

        return not self._variables

    @property
    def variables(self) -> Mapping[str, AccumulatedPeriodVariable]:
        """Return a read-only view of accumulated variables."""

        return MappingProxyType(self._variables)

    def clear(self) -> None:
        """Reset the accumulator for the next averaging period."""

        self._variables.clear()

    def add_samples(
        self,
        samples: Mapping[str, PeriodAverageSample],
        *,
        summation_dim: str | None = None,
    ) -> None:
        """Add named samples, optionally reducing one dimension as summands."""

        if not samples:
            raise ValueError(
                "Period average accumulation requires at least one sample."
            )

        variable_names = tuple(samples.keys())
        if self._variables and variable_names != tuple(self._variables.keys()):
            raise ValueError("Period average variables changed across samples.")

        for name, sample in samples.items():
            self._add_sample(name, sample, summation_dim=summation_dim)

    def mean_samples(self) -> dict[str, PeriodAverageSample]:
        """Return period means for all variables in insertion order."""

        if not self._variables:
            raise ValueError("Period average output requires at least one sample.")
        return {
            name: variable.mean_sample() for name, variable in self._variables.items()
        }

    def _add_sample(
        self,
        name: str,
        sample: PeriodAverageSample,
        *,
        summation_dim: str | None,
    ) -> None:
        dims, sum_values, counts = _sample_sum_and_counts(
            name,
            sample,
            summation_dim=summation_dim,
        )
        existing = self._variables.get(name)
        if existing is None:
            self._variables[name] = AccumulatedPeriodVariable(
                dims=dims,
                sum_values=sum_values.copy(),
                counts=counts.copy(),
                attrs=dict(sample.attrs),
            )
            return

        if existing.dims != dims:
            raise ValueError(f"Period average variable {name!r} dimensions changed.")
        if existing.sum_values.shape != sum_values.shape:
            raise ValueError(f"Period average variable {name!r} shape changed.")

        self._variables[name] = AccumulatedPeriodVariable(
            dims=existing.dims,
            sum_values=existing.sum_values + sum_values,
            counts=existing.counts + counts,
            attrs=existing.attrs,
        )


def _sample_sum_and_counts(
    name: str,
    sample: PeriodAverageSample,
    *,
    summation_dim: str | None,
) -> tuple[tuple[str, ...], jax.Array, jax.Array]:
    dims = sample.dims
    if not isinstance(dims, tuple) or not all(isinstance(dim, str) for dim in dims):
        raise ValueError(f"Period average variable {name!r} has invalid dimensions.")

    try:
        values = as_jax_real_array(sample.values)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Period average variable {name!r} must contain numeric values."
        ) from exc

    if values.ndim != len(dims):
        raise ValueError(
            f"Period average variable {name!r} has shape {values.shape} "
            f"but dimensions {dims}."
        )

    try:
        finite = jnp.isfinite(values)
    except TypeError as exc:
        raise ValueError(
            f"Period average variable {name!r} must contain numeric values."
        ) from exc

    sum_values = jnp.where(finite, values, 0.0)
    counts = finite.astype(jax_index_dtype())

    if summation_dim is None:
        return dims, sum_values, counts

    if dims.count(summation_dim) != 1:
        raise ValueError(
            f"Period average variable {name!r} must include one "
            f"{summation_dim!r} dimension."
        )
    axis = dims.index(summation_dim)
    reduced_dims = dims[:axis] + dims[axis + 1 :]  # noqa: E203
    return (
        reduced_dims,
        jnp.sum(sum_values, axis=axis),
        jnp.sum(counts, axis=axis),
    )


__all__ = [
    "AccumulatedPeriodVariable",
    "PeriodAverageAccumulator",
    "PeriodAverageSample",
]
