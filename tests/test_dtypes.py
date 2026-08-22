from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from tests.assertions import assert_finite_jvp_vjp
import vercor.dtypes as dtypes_module
from vercor.dtypes import (
    DTypePolicy,
    as_jax_index_array,
    as_jax_real_array,
    jax_arange,
    jax_full,
    jax_index_dtype,
    jax_ones,
    jax_real_dtype,
    jax_zeros,
)


def test_dtype_policy_disable_x64_maps_real_arrays_to_float32() -> None:
    policy = DTypePolicy(enable_x64=False)

    assert jax_real_dtype(policy) == jnp.float32
    assert np.dtype(jax_real_dtype(policy)) == np.dtype(np.float32)
    assert as_jax_real_array([1.0, 2.0], policy).dtype == jnp.float32
    assert jax_zeros((2, 3), policy).dtype == jnp.float32
    assert jax_ones((2, 3), policy).dtype == jnp.float32
    assert jax_full((2, 3), 1.5, policy).dtype == jnp.float32
    assert jax_arange(0.0, 3.0, 1.0, policy).dtype == jnp.float32


def test_dtype_policy_enable_x64_maps_real_arrays_to_float64() -> None:
    policy = DTypePolicy(enable_x64=True)

    assert jax_real_dtype(policy) == jnp.float64
    assert np.dtype(jax_real_dtype(policy)) == np.dtype(np.float64)
    assert as_jax_real_array([1.0, 2.0], policy).dtype == jnp.float64
    assert jax_zeros((2, 3), policy).dtype == jnp.float64
    assert jax_ones((2, 3), policy).dtype == jnp.float64
    assert jax_full((2, 3), 1.5, policy).dtype == jnp.float64
    assert jax_arange(0.0, 3.0, 1.0, policy).dtype == jnp.float64


def test_dtypes_module_does_not_export_unused_copy_helper() -> None:
    assert not hasattr(dtypes_module, "jax_real_array_copy")


def test_dtypes_module_does_not_export_numpy_dtype_helpers() -> None:
    assert not hasattr(dtypes_module, "numpy_real_dtype")
    assert not hasattr(dtypes_module, "numpy_index_dtype")


def test_index_dtype_is_int32_for_both_real_precision_modes() -> None:
    for enable_x64 in (False, True):
        policy = DTypePolicy(enable_x64=enable_x64)

        assert jax_index_dtype(policy) == jnp.int32
        assert np.dtype(jax_index_dtype(policy)) == np.dtype(np.int32)
        assert as_jax_index_array([0, 1, 2], policy).dtype == jnp.int32
        compiled_indices = jax.jit(lambda value: as_jax_index_array(value, policy))(
            jnp.asarray([0, 1, 2], dtype=jnp.int32)
        )
        assert compiled_indices.dtype == jnp.int32


def test_numpy_and_jax_helpers_agree_on_dtype_policy() -> None:
    policy = DTypePolicy(enable_x64=True)

    assert np.dtype(jax_real_dtype(policy)) == np.dtype(np.float64)
    assert np.dtype(jax_index_dtype(policy)) == np.dtype(np.int32)


def test_unconfigured_real_conversion_preserves_existing_array_dtype() -> None:
    source = jnp.asarray([1.0, 2.0], dtype=jnp.float32)

    assert as_jax_real_array(source).dtype == jnp.float32


def test_real_dtype_normalization_has_finite_jit_jvp_and_vjp() -> None:
    policy = DTypePolicy(enable_x64=True)
    assert_finite_jvp_vjp(
        lambda floating_leaf: jnp.sum(as_jax_real_array(floating_leaf, policy) ** 2),
        jnp.asarray([1.0, 2.0], dtype=jnp.float64),
        jnp.ones(2, dtype=jnp.float64),
    )
