from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from vercor.dtypes import as_jax_real_array
from vercor.field_layout import canonicalize_time_last_surface_field


def positive_binary_mask(values: ArrayLike) -> jax.Array:
    """Return a binary mask with ones where ``values`` are strictly positive."""

    return jnp.where(as_jax_real_array(values) > 0.0, 1.0, 0.0)


def canonicalize_surface_field(field: ArrayLike) -> jax.Array:
    """Convert a surface field to VerCOR's trailing-grid layout.

    Two-dimensional ``(nLon, nLat)`` fields become ``(nLat, nLon)``. Time-last
    ``(nLon, nLat, nTime)`` fields become ``(nTime, nLat, nLon)``.
    """

    field_array = as_jax_real_array(field)
    if field_array.ndim == 2:
        return field_array.T
    if field_array.ndim == 3:
        return canonicalize_time_last_surface_field(field_array)
    raise ValueError(
        "Expected a surface field with shape (nLon, nLat) or " "(nLon, nLat, nTime)."
    )


def mask_time_last_surface_field(
    surface_field: ArrayLike,
    binary_mask: ArrayLike,
) -> jax.Array:
    """Apply a 2D binary mask to a time-last surface field.

    ``surface_field`` is converted from ``(nLon, nLat, nTime)`` to VerCOR's
    canonical ``(nTime, nLat, nLon)`` layout before masking. Masked cells are
    represented as NaN so downstream diagnostics can ignore missing ocean or
    land values.
    """

    canonical_field = canonicalize_time_last_surface_field(surface_field)
    mask = as_jax_real_array(binary_mask) > 0.0
    return jnp.where(mask[jnp.newaxis, ...], canonical_field, jnp.nan)
