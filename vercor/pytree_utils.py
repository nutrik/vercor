from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from vercor.dtypes import jax_real_dtype


def asfloat(tree: Any, policy: Any = None) -> Any:
    """Cast every leaf in a PyTree to VerCOR's configured real dtype."""

    return jax.tree_util.tree_map(lambda arr: arr.astype(jax_real_dtype(policy)), tree)


def mean_leaf(tree: Any, axis: int | list[int]) -> Any:
    """Return a PyTree with ``jnp.mean`` applied to every leaf."""

    return jax.tree_util.tree_map(lambda arr: jnp.mean(arr, axis=axis), tree)


def unwrap_leading_dims(obj: Any, first_n_dim: int = 2) -> Any:
    """Flatten the leading dimensions of every array leaf in a PyTree."""

    def _unwrap(arr: jnp.ndarray) -> jnp.ndarray:
        new_shape = (-1,) + arr.shape[first_n_dim:]
        return jnp.reshape(arr, new_shape)

    return jax.tree_util.tree_map(_unwrap, obj)


def stack_objects(objs: list[Any]) -> Any:
    """Stack a list of same-structure PyTrees leafwise."""

    return jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *objs)


def concat_objects(objs: list[Any], axis: int) -> Any:
    """Concatenate a list of same-structure PyTrees leafwise."""

    return jax.tree_util.tree_map(lambda *xs: jnp.concatenate(xs, axis=axis), *objs)
