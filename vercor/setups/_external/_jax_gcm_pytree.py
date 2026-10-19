from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from vercor.dtypes import DTypePolicy


def tree_as_runtime_dtype(tree: Any, policy: DTypePolicy) -> Any:
    """Normalize floating JAXGCM leaves without changing integer semantics."""

    complex_dtype = jnp.complex128 if policy.enable_x64 else jnp.complex64

    def normalize(value: Any) -> Any:
        try:
            array = jnp.asarray(value)
        except (TypeError, ValueError):
            return value
        if jnp.issubdtype(array.dtype, jnp.floating):
            return array.astype(policy.jax_real)
        if jnp.issubdtype(array.dtype, jnp.complexfloating):
            return array.astype(complex_dtype)
        return array

    return jax.tree_util.tree_map(normalize, tree)


def tree_mean(tree: Any, axis: int | list[int]) -> Any:
    """Average inexact leaves and retain the final categorical sample."""

    def reduce(value: Any) -> Any:
        array = jnp.asarray(value)
        if jnp.issubdtype(array.dtype, jnp.inexact):
            return jnp.mean(array, axis=axis)
        axes = (axis,) if isinstance(axis, int) else tuple(axis)
        reduced = array
        normalized_axes = sorted(
            (selected_axis % array.ndim for selected_axis in axes),
            reverse=True,
        )
        for selected_axis in normalized_axes:
            reduced = jnp.take(reduced, -1, axis=selected_axis)
        return reduced

    return jax.tree_util.tree_map(reduce, tree)


__all__ = [
    "tree_as_runtime_dtype",
    "tree_mean",
]
