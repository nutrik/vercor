from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from vercor.dtypes import jax_real_dtype


def tree_as_real_dtype(tree: Any, policy: Any = None) -> Any:
    """Cast every JAXGCM PyTree leaf to the configured VerCOR real dtype."""

    return jax.tree_util.tree_map(
        lambda arr: arr.astype(jax_real_dtype(policy)),
        tree,
    )


def tree_mean(tree: Any, axis: int | list[int]) -> Any:
    """Average every JAXGCM PyTree leaf along one or more axes."""

    return jax.tree_util.tree_map(lambda arr: jnp.mean(arr, axis=axis), tree)


def tree_unwrap_leading_dims(obj: Any, first_n_dim: int = 2) -> Any:
    """Flatten the leading dimensions of each JAXGCM PyTree leaf."""

    def _unwrap(arr: jnp.ndarray) -> jnp.ndarray:
        new_shape = (-1,) + arr.shape[first_n_dim:]
        return jnp.reshape(arr, new_shape)

    return jax.tree_util.tree_map(_unwrap, obj)


def tree_stack(objs: list[Any]) -> Any:
    """Stack matching JAXGCM PyTree leaves from a sequence of objects."""

    return jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *objs)


__all__ = [
    "tree_as_real_dtype",
    "tree_mean",
    "tree_stack",
    "tree_unwrap_leading_dims",
]
