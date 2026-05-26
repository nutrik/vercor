from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from vercor.dtypes import as_jax_real_array
from vercor.field_layout import canonicalize_time_last_surface_field


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
