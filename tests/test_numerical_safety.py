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
    safe_masked_divide,
)
from vercor.exceptions import CouplerError


@pytest.mark.fast_always
@pytest.mark.parametrize("check", ["numeric", "state"])
def test_successful_compiled_validation_does_not_call_host(
    monkeypatch: pytest.MonkeyPatch, check: str
) -> None:
    """Valid numerical state must not transfer predicates to Python."""
    import vercor._numerical_safety as safety
    import vercor._runtime.state_validation as validation

    observed: list[int] = []
    if check == "numeric":
        original = safety._raise_for_invalid_count

        def record(count: object, owner: str, requirement: str) -> None:
            observed.append(int(count))  # type: ignore[call-overload]
            original(count, owner, requirement)

        monkeypatch.setattr(safety, "_raise_for_invalid_count", record)

        def validate(values: jax.Array) -> jax.Array:
            require_active_finite(values, active_mask=None, owner="profile field")
            return values * 2

    else:
        original_predicate = validation._raise_if_false

        def record_predicate(predicate: object, message: str) -> None:
            observed.append(int(predicate))  # type: ignore[call-overload]
            original_predicate(predicate, message)

        monkeypatch.setattr(validation, "_raise_if_false", record_predicate)

        def validate(values: jax.Array) -> jax.Array:
            validation._require_true(jnp.all(jnp.isfinite(values)), "profile field")
            return values * 2

    compiled = jax.jit(validate)
    assert_array_equal_compact(compiled(jnp.asarray([1.0, 2.0])), [2.0, 4.0])
    jax.effects_barrier()
    assert observed == []
    with pytest.raises(JaxRuntimeError, match="profile field"):
        compiled(jnp.asarray([1.0, jnp.nan])).block_until_ready()
    assert len(observed) == 1


@pytest.mark.fast_always
@pytest.mark.parametrize("transform", ["scan", "vmap"])
def test_compiled_validation_rejects_later_invalid_values(transform: str) -> None:
    """Skipping successful callbacks must still check later transformed values."""

    def checked(value: jax.Array) -> jax.Array:
        require_active_finite(value, active_mask=None, owner="later field")
        return value

    def scanned(values: jax.Array) -> jax.Array:
        def step(total: jax.Array, value: jax.Array) -> tuple[jax.Array, None]:
            return total + checked(value), None

        return jax.lax.scan(step, jnp.asarray(0.0), values)[0]

    transformed = scanned if transform == "scan" else jax.vmap(checked)
    compiled = jax.jit(transformed)
    valid = jnp.asarray([1.0, 2.0, 3.0])
    assert_array_equal_compact(compiled(valid), 6.0 if transform == "scan" else valid)
    with pytest.raises(JaxRuntimeError, match="later field.*active domain"):
        compiled(jnp.asarray([1.0, 2.0, jnp.nan])).block_until_ready()


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
    def checked_sum(values: jax.Array) -> jax.Array:
        require_active_finite(values, active_mask=None, owner="compiled field")
        return jnp.sum(values)

    with pytest.raises(JaxRuntimeError, match="compiled field.*active domain"):
        jax.jit(checked_sum)(jnp.asarray([1.0, jnp.nan])).block_until_ready()


def test_replace_missing_nan_fills_nan_and_preserves_finite_values() -> None:
    replaced = replace_missing_nan(
        jnp.asarray([1.5, jnp.nan, -2.0]), owner="test field", fill_value=7.0
    )
    assert_array_equal_compact(replaced, jnp.asarray([1.5, 7.0, -2.0]))


@pytest.mark.parametrize("bad_value", [jnp.inf, -jnp.inf])
def test_replace_missing_nan_rejects_infinity(bad_value: float) -> None:
    with pytest.raises(CouplerError, match="test field"):
        replace_missing_nan(jnp.asarray([1.0, bad_value]), owner="test field")


@pytest.mark.parametrize("bad_fill_value", [jnp.nan, jnp.inf, -jnp.inf])
def test_replace_missing_nan_rejects_non_finite_fill_value(
    bad_fill_value: float,
) -> None:
    with pytest.raises(CouplerError, match="test field fill value"):
        replace_missing_nan(
            jnp.asarray([1.0, jnp.nan]),
            owner="test field",
            fill_value=bad_fill_value,
        )


@pytest.mark.parametrize("operand", ["numerator", "denominator"])
@pytest.mark.parametrize("bad_value", [jnp.inf, -jnp.inf])
def test_safe_masked_divide_rejects_infinite_inactive_operands(
    operand: str,
    bad_value: float,
) -> None:
    numerator = jnp.asarray([4.0, bad_value if operand == "numerator" else jnp.nan])
    denominator = jnp.asarray([2.0, bad_value if operand == "denominator" else 0.0])
    with pytest.raises(CouplerError, match=f"safe {operand}.*infinity"):
        safe_masked_divide(
            numerator,
            denominator,
            where=jnp.asarray([True, False]),
            inactive_value=jnp.nan,
        )


def test_safe_masked_divide_rejects_non_finite_active_quotient() -> None:
    with pytest.raises(CouplerError, match="safe quotient.*active domain"):
        safe_masked_divide(
            jnp.asarray([1.0, jnp.nan]),
            jnp.asarray([0.0, 0.0]),
            where=jnp.asarray([True, False]),
            inactive_value=jnp.nan,
        )


def test_safe_masked_divide_broadcasts_scalar_numerator() -> None:
    divided = safe_masked_divide(
        4.0,
        jnp.asarray([2.0, 0.0]),
        where=jnp.asarray([True, False]),
        inactive_value=jnp.nan,
    )
    assert_array_equal_compact(divided, jnp.asarray([2.0, jnp.nan]))


def test_safe_masked_divide_broadcasts_scalar_denominator() -> None:
    divided = safe_masked_divide(
        jnp.asarray([4.0, jnp.nan]),
        2.0,
        where=jnp.asarray([True, False]),
        inactive_value=jnp.nan,
    )
    assert_array_equal_compact(divided, jnp.asarray([2.0, jnp.nan]))


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


@pytest.mark.parametrize("bad_value", [jnp.inf, -jnp.inf])
def test_safe_masked_divide_rejects_infinite_inactive_output(
    bad_value: float,
) -> None:
    with pytest.raises(CouplerError, match="safe masked divide output.*infinity"):
        safe_masked_divide(
            jnp.asarray([4.0, jnp.nan]),
            jnp.asarray([2.0, 0.0]),
            where=jnp.asarray([True, False]),
            inactive_value=bad_value,
        )


@pytest.mark.parametrize("bad_value", [jnp.inf, -jnp.inf])
def test_safe_masked_divide_reports_compiled_infinite_inactive_output(
    bad_value: float,
) -> None:
    def divide(inactive_value: jax.Array) -> jax.Array:
        return safe_masked_divide(
            jnp.asarray([4.0, jnp.nan]),
            jnp.asarray([2.0, 0.0]),
            where=jnp.asarray([True, False]),
            inactive_value=inactive_value,
        )

    with pytest.raises(
        JaxRuntimeError,
        match="safe masked divide output.*infinity",
    ):
        jax.jit(divide)(jnp.asarray(bad_value)).block_until_ready()
