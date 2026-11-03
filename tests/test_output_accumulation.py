"""Batched output arithmetic preserves heterogeneous numeric and AD contracts."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests.assertions import assert_allclose_compact

pytestmark = pytest.mark.fast_always


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_batched_output_accumulation_keeps_axes_counts_and_missing_cells(
    dtype: Any,
) -> None:
    from vercor.output._period import _accumulate_output_samples

    samples = (
        jnp.asarray([2, jnp.nan, jnp.inf], dtype),
        jnp.asarray([[2, jnp.nan], [6, 10]], dtype),
        jnp.asarray([[3, jnp.nan], [9, 12]], dtype),
        jnp.asarray([8, jnp.nan], dtype),
    )
    totals = tuple(jnp.ones(shape, dtype) for shape in ((3,), (2,), (2,), (2,)))
    counts = tuple(jnp.ones(x.shape, jnp.int32) for x in totals)
    reduced_counts = (None, None, None, jnp.asarray([4, 0], jnp.int32))
    result, result_counts = _accumulate_output_samples(
        totals, counts, samples, reduced_counts, axes=(None, 0, 1, None)
    )
    expected = ([3, 1, 1], [9, 11], [4, 22], [9, 1])
    expected_counts = ([2, 1, 1], [3, 2], [2, 3], [5, 1])
    for total, count, value, number in zip(
        result, result_counts, expected, expected_counts, strict=True
    ):
        assert_allclose_compact(total, value)
        np.testing.assert_array_equal(count, number)
        assert total.dtype == dtype
        assert count.dtype == jnp.int32


def test_batched_output_accumulation_jvp_vjp_ignore_missing_samples() -> None:
    from vercor.output._period import _accumulate_output_samples

    def objective(x: jax.Array) -> jax.Array:
        totals, _ = _accumulate_output_samples(
            (jnp.zeros(3), jnp.zeros(3)),
            (jnp.zeros(3, jnp.int32), jnp.zeros(3, jnp.int32)),
            (jnp.stack((x, 3 * x)), 2 * x),
            (None, jnp.asarray([1, 1, 0], jnp.int32)),
            axes=(0, None),
        )
        return jnp.sum(totals[0]) + jnp.sum(totals[1])

    values = jnp.asarray([2.0, 5.0, jnp.nan])
    tangent = jnp.asarray([1.0, 1.0, 0.0])
    primal, forward = jax.jit(lambda x: jax.jvp(objective, (x,), (tangent,)))(values)
    assert_allclose_compact(primal, 42)
    assert_allclose_compact(forward, 12)
    assert_allclose_compact(jax.jit(jax.grad(objective))(values), [6, 6, 0])
