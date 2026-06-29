from __future__ import annotations

import jax.numpy as jnp
import numpy as np

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
    numpy_index_dtype,
    numpy_real_dtype,
)
from vercor.settings import VercorSettings


def test_settings_disable_x64_maps_real_arrays_to_float32() -> None:
    settings = VercorSettings(enable_x64=False)

    assert settings.dtype_policy == DTypePolicy(enable_x64=False)
    assert jax_real_dtype(settings) == jnp.float32
    assert numpy_real_dtype(settings) == np.dtype(np.float32)
    assert as_jax_real_array([1.0, 2.0], settings).dtype == jnp.float32
    assert jax_zeros((2, 3), settings).dtype == jnp.float32
    assert jax_ones((2, 3), settings).dtype == jnp.float32
    assert jax_full((2, 3), 1.5, settings).dtype == jnp.float32
    assert jax_arange(0.0, 3.0, 1.0, settings).dtype == jnp.float32


def test_settings_enable_x64_maps_real_arrays_to_float64() -> None:
    settings = VercorSettings(enable_x64=True)

    assert settings.dtype_policy == DTypePolicy(enable_x64=True)
    assert jax_real_dtype(settings) == jnp.float64
    assert numpy_real_dtype(settings) == np.dtype(np.float64)
    assert as_jax_real_array([1.0, 2.0], settings).dtype == jnp.float64
    assert jax_zeros((2, 3), settings).dtype == jnp.float64
    assert jax_ones((2, 3), settings).dtype == jnp.float64
    assert jax_full((2, 3), 1.5, settings).dtype == jnp.float64
    assert jax_arange(0.0, 3.0, 1.0, settings).dtype == jnp.float64


def test_dtypes_module_does_not_export_unused_copy_helper() -> None:
    assert not hasattr(dtypes_module, "jax_real_array_copy")


def test_dtype_policy_reads_updated_settings_value() -> None:
    settings = VercorSettings(enable_x64=False)

    settings.set_value("enable_x64", True)

    assert settings.dtype_policy == DTypePolicy(enable_x64=True)


def test_index_dtype_is_int32_for_both_real_precision_modes() -> None:
    for enable_x64 in (False, True):
        settings = VercorSettings(enable_x64=enable_x64)

        assert jax_index_dtype(settings) == jnp.int32
        assert numpy_index_dtype(settings) == np.dtype(np.int32)
        assert as_jax_index_array([0, 1, 2], settings).dtype == jnp.int32


def test_numpy_and_jax_helpers_agree_on_dtype_policy() -> None:
    policy = DTypePolicy(enable_x64=True)

    assert np.dtype(jax_real_dtype(policy)) == numpy_real_dtype(policy)
    assert np.dtype(jax_index_dtype(policy)) == numpy_index_dtype(policy)


def test_unconfigured_real_conversion_preserves_existing_array_dtype() -> None:
    source = jnp.asarray([1.0, 2.0], dtype=jnp.float32)

    assert as_jax_real_array(source).dtype == jnp.float32
