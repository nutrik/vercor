from __future__ import annotations

from typing import cast

from jax import Array, lax
import jax.numpy as jnp

from vercor.dtypes import jax_full
import vercor.interpolators._bilinear_geometry as _geometry


def valid_scalar_source_mask(source_values: Array, src_mask: Array | None) -> Array:
    """Return source points available for scalar interpolation or extrapolation."""

    finite_source = jnp.isfinite(source_values)
    if src_mask is None:
        return finite_source
    return jnp.asarray(src_mask, dtype=bool) & finite_source


def extrapolate_scalar_field(
    *,
    source_values: Array,
    valid_source_mask: Array,
    target_shape: tuple[int, ...],
    target_lon_flat: Array,
    target_lat_flat: Array,
    source_lon_flat: Array,
    source_lat_flat: Array,
    mode: str | None,
    idw_k: int,
    idw_eps: float,
    fill_value: float,
) -> Array:
    """Fill target points from nearest or IDW valid source values."""

    if mode is None:
        return jax_full(target_shape, fill_value)

    valid = valid_scalar_source_mask(source_values, valid_source_mask).reshape(-1)
    values = source_values.reshape(-1)
    fill_flat = jnp.full((target_lon_flat.size,), fill_value, dtype=values.dtype)

    def no_valid(_: None) -> Array:
        return fill_flat

    def compute_extrapolated(_: None) -> Array:
        distances = _geometry.great_circle_distance_rad(
            target_lon_flat[:, None],
            target_lat_flat[:, None],
            source_lon_flat[None, :],
            source_lat_flat[None, :],
        )
        masked_distances = jnp.where(valid[None, :], distances, jnp.inf)

        if mode == "nearest":
            idx = jnp.argmin(masked_distances, axis=1)
            return cast(Array, values[idx])

        if mode == "idw":
            k = min(idw_k, values.size)
            neg_distances = -masked_distances
            top_neg, idx = lax.top_k(neg_distances, k)
            dist_k = -top_neg
            val_k = values[idx]
            weights = jnp.where(
                jnp.isfinite(dist_k),
                1.0 / (dist_k + idw_eps),
                0.0,
            )
            wsum = jnp.sum(weights, axis=1)
            return cast(
                Array,
                jnp.where(
                    wsum > 0.0,
                    jnp.sum(weights * val_k, axis=1) / wsum,
                    fill_value,
                ),
            )

        raise ValueError("extrapolation_mode must be 'nearest', 'idw', or None")

    flat = lax.cond(jnp.any(valid), compute_extrapolated, no_valid, operand=None)
    return cast(Array, flat.reshape(target_shape))
