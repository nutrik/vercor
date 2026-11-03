"""Shared mean arithmetic preserves missing-output and strict-native contracts."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests.assertions import assert_allclose_compact

pytestmark = pytest.mark.fast_always


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_finite_reduction_counts_cells_and_handles_empty_support(dtype: Any) -> None:
    from vercor._averaging import finite_sum_and_count, mean_from_sum_and_count

    samples = jnp.asarray([[2, jnp.nan, jnp.inf], [6, 10, -jnp.inf]], dtype=dtype)
    total, count = jax.jit(lambda x: finite_sum_and_count(x, axis=0))(samples)
    assert_allclose_compact(total, [8, 10, 0])
    np.testing.assert_array_equal(count, [2, 1, 0])
    assert total.dtype == dtype
    assert count.dtype == jnp.int32
    result = mean_from_sum_and_count(total, count)
    np.testing.assert_allclose(result, [4, 10, np.nan], equal_nan=True)
    assert result.dtype == dtype
    empty_total, empty_count = finite_sum_and_count(jnp.empty((0, 3), dtype), axis=0)
    np.testing.assert_array_equal(empty_count, 0)
    assert np.isnan(mean_from_sum_and_count(empty_total, empty_count)).all()


def test_finite_reduction_without_axis_keeps_each_sample() -> None:
    from vercor._averaging import finite_sum_and_count

    total, count = finite_sum_and_count(jnp.asarray([2.0, jnp.nan, -4.0]))
    assert_allclose_compact(total, [2, 0, -4])
    np.testing.assert_array_equal(count, [1, 0, 1])


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_streaming_native_tree_mean_keeps_scalar_count_endpoint_and_reset(
    dtype: Any,
) -> None:
    from vercor._averaging import SampleMean

    def sample(value: float) -> dict[str, Any]:
        x = jnp.full((2, 3), value, dtype=dtype)
        return {"state": x, "diagnostic": (x**2,)}

    initial = SampleMean.from_sample(sample(2))
    initial_size = sum(x.nbytes for x in jax.tree.leaves(initial))
    current = jax.jit(lambda x: x.add(sample(4)))(initial)
    current = current.add(sample(6))
    assert current.count.shape == ()
    assert current.count.dtype == jnp.int32
    assert sum(x.nbytes for x in jax.tree.leaves(current)) == initial_size
    assert_allclose_compact(current.mean()["state"], 4)
    assert_allclose_compact(current.mean()["diagnostic"][0], 56 / 3, rtol=1e-6)
    assert current.mean()["state"].dtype == dtype
    assert_allclose_compact(initial.mean()["state"], 2)
    assert_allclose_compact(SampleMean.from_sample(sample(8)).mean()["state"], 8)


def test_native_missing_sample_is_not_silently_discarded() -> None:
    from vercor._averaging import SampleMean

    mean = SampleMean.from_sample({"field": jnp.asarray([jnp.nan, jnp.inf])})
    result = mean.add({"field": jnp.asarray([4.0, 8.0])}).mean()["field"]
    assert np.isnan(result[0])
    assert np.isinf(result[1])


def test_shared_means_support_jvp_vjp_with_missing_output_cells() -> None:
    from vercor._averaging import (
        SampleMean,
        finite_sum_and_count,
        mean_from_sum_and_count,
    )

    def objective(x: jax.Array) -> jax.Array:
        samples = jnp.stack((x, 3 * x))
        total, count = finite_sum_and_count(samples, axis=0)
        output = mean_from_sum_and_count(total, count)
        native = SampleMean.from_sample(x).add(3 * x).mean()
        # Exclude the missing cell before summation, as an active-mask consumer does.
        return jnp.sum(jnp.where(count > 0, output, 0)) + jnp.sum(native[:2])

    value = jnp.asarray([2.0, 5.0, jnp.nan])
    tangent = jnp.asarray([1.0, 1.0, 0.0])
    primal, forward = jax.jit(lambda x: jax.jvp(objective, (x,), (tangent,)))(value)
    reverse = jax.grad(objective)(value)
    assert_allclose_compact(primal, 28)
    assert_allclose_compact(forward, 8)
    assert_allclose_compact(reverse, [4, 4, 0])
