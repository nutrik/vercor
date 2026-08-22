"""Behavioral contracts for transform-safe numerical checks."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from jax.errors import JaxRuntimeError

from tests.assertions import assert_array_equal_compact, assert_finite_jvp_vjp
from vercor._numerical_safety import (
    replace_missing_nan,
    require_active_finite,
    require_strictly_positive,
    safe_masked_divide,
)
from vercor.exceptions import CouplerError


def test_active_finite_contract_broadcasts_trailing_grid_mask() -> None:
    values = jnp.asarray(
        [[[1.0, jnp.nan], [2.0, jnp.nan]], [[3.0, jnp.nan], [4.0, jnp.nan]]]
    )
    mask = jnp.asarray([[1.0, 0.0], [1.0, 0.0]])
    require_active_finite(values, active_mask=mask, owner="test field")


@pytest.mark.parametrize("bad_value", [jnp.nan, jnp.inf, -jnp.inf])
def test_active_finite_contract_rejects_active_nonfinite_values(
    bad_value: float,
) -> None:
    with pytest.raises(CouplerError, match="test field.*active domain"):
        require_active_finite(
            jnp.asarray([[bad_value, 2.0]]),
            active_mask=jnp.asarray([[1.0, 0.0]]),
            owner="test field",
        )


def test_active_finite_contract_rejects_inactive_infinity_but_allows_nan() -> None:
    require_active_finite(
        jnp.asarray([[1.0, jnp.nan]]),
        active_mask=jnp.asarray([[1.0, 0.0]]),
        owner="test field",
    )
    with pytest.raises(CouplerError, match="infinity"):
        require_active_finite(
            jnp.asarray([[1.0, jnp.inf]]),
            active_mask=jnp.asarray([[1.0, 0.0]]),
            owner="test field",
        )


def test_active_finite_contract_rejects_non_trailing_mask() -> None:
    with pytest.raises(CouplerError, match="test field.*trailing suffix"):
        require_active_finite(
            jnp.ones((2, 3, 4)),
            active_mask=jnp.ones((2, 3)),
            owner="test field",
        )


def test_active_finite_contract_reports_compiled_failure() -> None:
    checked_sum = jax.jit(
        lambda values: (
            require_active_finite(values, active_mask=None, owner="compiled field"),
            jnp.sum(values),
        )[1]
    )
    with pytest.raises(JaxRuntimeError, match="compiled field.*active domain"):
        checked_sum(jnp.asarray([1.0, jnp.nan])).block_until_ready()


@pytest.mark.parametrize("bad_value", [0.0, -1.0, jnp.nan, jnp.inf, -jnp.inf])
def test_strictly_positive_rejects_non_positive_or_non_finite_values(
    bad_value: float,
) -> None:
    with pytest.raises(CouplerError, match="strictly positive finite"):
        require_strictly_positive(jnp.asarray([1.0, bad_value]), owner="test scale")


def test_strictly_positive_reports_compiled_failure() -> None:
    checked_sum = jax.jit(
        lambda values: (
            require_strictly_positive(values, owner="compiled scale"),
            jnp.sum(values),
        )[1]
    )
    with pytest.raises(JaxRuntimeError, match="compiled scale.*strictly positive"):
        checked_sum(jnp.asarray([1.0, 0.0])).block_until_ready()


def test_replace_missing_nan_fills_nan_and_preserves_finite_values() -> None:
    replaced = replace_missing_nan(
        jnp.asarray([1.5, jnp.nan, -2.0]), owner="test field", fill_value=7.0
    )
    assert_array_equal_compact(replaced, jnp.asarray([1.5, 7.0, -2.0]))


@pytest.mark.parametrize("bad_value", [jnp.inf, -jnp.inf])
def test_replace_missing_nan_rejects_infinity(bad_value: float) -> None:
    with pytest.raises(CouplerError, match="test field"):
        replace_missing_nan(jnp.asarray([1.0, bad_value]), owner="test field")


def test_safe_masked_divide_has_finite_jvp_and_vjp() -> None:
    mask = jnp.asarray([True, False])

    def objective(values: jax.Array) -> jax.Array:
        divided = safe_masked_divide(
            values,
            jnp.asarray([2.0, 0.0]),
            where=mask,
            inactive_value=jnp.nan,
        )
        return jnp.nansum(divided)

    assert_finite_jvp_vjp(
        objective,
        jnp.asarray([4.0, jnp.nan]),
        jnp.asarray([1.0, 0.0]),
    )
