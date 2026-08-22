"""Private, transform-safe numerical validation and masking primitives."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from vercor.exceptions import CouplerError


def _broadcast_active_mask(
    values: jax.Array,
    active_mask: Any | None,
    owner: str,
) -> jax.Array:
    if active_mask is None:
        return jnp.ones(values.shape, dtype=bool)
    mask = jnp.asarray(active_mask) > 0
    if mask.ndim > values.ndim or (
        mask.ndim > 0 and values.shape[-mask.ndim :] != mask.shape
    ):
        raise CouplerError(
            f"{owner} active mask shape {mask.shape} is not a trailing suffix "
            f"of value shape {values.shape}."
        )
    reshaped = jnp.reshape(mask, (1,) * (values.ndim - mask.ndim) + mask.shape)
    return jnp.broadcast_to(reshaped, values.shape)


def _raise_for_invalid_count(count: Any, owner: str, requirement: str) -> None:
    count_value = int(count)
    if count_value:
        raise CouplerError(
            f"{owner} contains {count_value} value(s) violating {requirement}."
        )


def _require_zero(count: jax.Array, owner: str, requirement: str) -> None:
    if isinstance(count, jax.core.Tracer):
        jax.debug.callback(
            lambda concrete: _raise_for_invalid_count(
                concrete,
                owner,
                requirement,
            ),
            count,
        )
        return
    _raise_for_invalid_count(count, owner, requirement)


def require_active_finite(
    values: Any,
    *,
    active_mask: Any | None,
    owner: str,
) -> None:
    """Reject active NaN/inf and reject infinity outside the active domain."""

    array = jnp.asarray(values)
    active = _broadcast_active_mask(array, active_mask, owner)
    valid = jnp.isfinite(array) | (~active & jnp.isnan(array))
    invalid_count = jnp.count_nonzero(~valid)
    requirement = "finite values in the active domain and no infinity elsewhere"
    _require_zero(invalid_count, owner, requirement)


def require_strictly_positive(values: Any, *, owner: str) -> None:
    """Reject non-finite, zero, or negative numerical values."""

    array = jnp.asarray(values)
    invalid_count = jnp.count_nonzero(~jnp.isfinite(array) | (array <= 0))
    _require_zero(invalid_count, owner, "strictly positive finite values")


def replace_missing_nan(
    values: Any,
    *,
    owner: str,
    fill_value: float = 0.0,
) -> jax.Array:
    """Replace missing-data NaNs while rejecting infinities."""

    array = jnp.asarray(values)
    finite_locations = ~jnp.isnan(array)
    require_active_finite(array, active_mask=finite_locations, owner=owner)
    fill_array = jnp.asarray(fill_value)
    require_active_finite(fill_array, active_mask=None, owner=f"{owner} fill value")
    return jnp.where(
        finite_locations, array, jnp.asarray(fill_array, dtype=array.dtype)
    )


def safe_masked_divide(
    numerator: Any,
    denominator: Any,
    *,
    where: Any,
    inactive_value: Any,
) -> jax.Array:
    """Divide only finite-neutralized active operands and mask the result."""

    condition = jnp.asarray(where, dtype=bool)
    numerator_array = jnp.asarray(numerator)
    denominator_array = jnp.asarray(denominator)
    require_active_finite(
        numerator_array,
        active_mask=condition,
        owner="safe numerator",
    )
    require_active_finite(
        denominator_array,
        active_mask=condition,
        owner="safe denominator",
    )
    safe_numerator = jnp.where(condition, numerator_array, 0.0)
    safe_denominator = jnp.where(condition, denominator_array, 1.0)
    quotient = safe_numerator / safe_denominator
    require_active_finite(quotient, active_mask=condition, owner="safe quotient")
    return jnp.where(condition, quotient, inactive_value)
