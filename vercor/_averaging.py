"""Shared sum/count arithmetic, independent of sampling clocks and field names.

Diagnostic output excludes nonfinite samples with per-cell integer counts.
Mandatory native samples use strict sums and one scalar count: invalid physics
must propagate to its owner's validation. Both reductions store only totals and
counts and divide once at their owner's interval boundary. Units are unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from vercor.dtypes import jax_index_dtype


@partial(jax.jit, static_argnames=("axis",))
def finite_sum_and_count(
    values: ArrayLike, *, axis: int | None = None
) -> tuple[jax.Array, jax.Array]:
    """Return finite totals and int32 counts, optionally reducing one axis.

    With no axis each cell contributes zero or one sample. Missing cells never
    enter arithmetic, including its derivatives; an empty reduction has count 0.
    """
    values = jnp.asarray(values)
    finite = jnp.isfinite(values)
    total = jnp.where(finite, values, 0)
    count = finite.astype(jax_index_dtype())
    if axis is None:
        return total, count
    return jnp.sum(total, axis=axis), jnp.sum(count, axis=axis, dtype=jax_index_dtype())


@jax.jit
def mean_from_sum_and_count(total: ArrayLike, count: ArrayLike) -> jax.Array:
    """Divide totals by broadcastable counts/weights, returning NaN at zero.

    Scalar native counts and duration weights avoid redundant field-sized
    arrays. The safe denominator keeps zero-support derivatives finite; counts
    and physical duration weights are validated by the sampling owner.
    """
    total = jnp.asarray(total)
    count = jnp.asarray(count)
    denominator = jnp.where(count > 0, count, 1)
    return jnp.where(count > 0, total / denominator, jnp.nan)


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class SampleMean:
    """Strict streaming mean of an array PyTree with equal-duration samples.

    Each leaf retains its native shape and dtype; one int32 count serves the
    whole tree. Callers diagnose nonlinear fields before adding them and keep
    endpoint continuation separately. Creating a new instance starts a window.
    """

    total: Any
    count: jax.Array

    @classmethod
    def from_sample(cls, sample: Any) -> SampleMean:
        """Start a window with one mapped post-step sample, without a history."""
        return cls(jax.tree.map(jnp.asarray, sample), jnp.asarray(1, jax_index_dtype()))

    @jax.jit
    def add(self, sample: Any) -> SampleMean:
        """Include a required sample without discarding invalid native values."""
        return SampleMean(jax.tree.map(jnp.add, self.total, sample), self.count + 1)

    @jax.jit
    def mean(self) -> Any:
        """Return the tree mean at the sampling owner's interval boundary."""
        return jax.tree.map(
            lambda total: mean_from_sum_and_count(total, self.count), self.total
        )
