# No-NaN Differentiation Policy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enforce fail-fast active-domain finiteness throughout VerCOR and verify finite primal, JIT, JVP, and VJP behavior for every implemented differentiable component and numerical subsystem.

**Architecture:** A dependency-light private numerical-safety module supplies transform-safe assertions, missing-NaN replacement, and masked division. Existing preparation, component, transfer, and exchange owners call it at their handoff boundaries, while source-level arithmetic repairs prevent inactive masked branches from poisoning derivatives. Focused component tests and composed Coupler tests provide executable audit evidence without changing the public API.

**Tech Stack:** Python 3.12+, JAX, jaxtyping-compatible annotations, NumPy, pytest, pytest-xdist, pytest-cov, Black, flake8, mypy, Flit.

**Spec:** `docs/superpowers/specs/2026-08-22-no-nan-differentiation-policy-design.md`

## Global Constraints

- Active field values, forward tangents, and reverse cotangents must be finite.
- Inactive values may be finite or NaN; positive and negative infinity are invalid everywhere.
- A component grid mask is active where it is greater than zero; an absent grid mask means every grid cell is active.
- Exchange handoffs are active where their route fractional mask is greater than zero.
- Leading field dimensions broadcast over the trailing two-dimensional grid mask.
- Active non-finite values fail immediately and are never silently replaced, clipped, or scaled.
- Missing-data NaNs may be replaced only in explicitly documented adapter merge operations; infinities must remain errors.
- No global `jax_debug_nans`, numerical fudge factor, public runtime switch, registry, or public API expansion is allowed.
- Physics values remain JAX-traced, precision remains owned by `RuntimeOptions.dtype`, and runtime state remains immutable.
- JAX code must not branch in Python on traced values.
- Optional JCM, Veros, CAMulator, Torch, TensorFlow, and CREDIT imports remain lazy.
- Tests precede production changes, and every commit follows a passing full unit suite as required by `AGENTS.md`.
- `PROGRESS.md`, `DESIGN.md`, and `DEPENDENCIES.md` must match the implemented result before completion.

## File Structure

- Create `vercor/_numerical_safety.py`: sole dependency-light owner of active-domain checks, strict-positive checks, missing-NaN replacement, and differentiation-safe masked division.
- Create `tests/test_numerical_safety.py`: focused behavioral contract for the private numerical-safety owner.
- Modify `tests/assertions.py`: add one scalar-objective helper that checks primal/JIT/JVP/VJP finiteness and the JVP/VJP inner-product identity.
- Modify `vercor/_runtime/state_validation.py`: validate all runtime field stores with component- or route-specific activity masks.
- Modify `vercor/components/runtime_execution.py`: validate fields returned by component steps before they can propagate.
- Modify `vercor/_runtime/field_transfer.py`: validate time-selected sent fields.
- Modify `vercor/_runtime/exchange_dispatch.py`: validate raw regridding outputs and mask scalar transfers without evaluating `NaN * 0` as a propagated value.
- Modify `vercor/_interpolators/bilinear_rectilinear.py`, `vercor/_interpolators/_bilinear_extrapolation.py`, and `vercor/_interpolators/conservative_remap_rectilinear.py`: replace masked zero-denominator divisions with the shared safe primitive.
- Modify `vercor/setups/_external/jax_gcm_fields.py`, `vercor/setups/_external/veros_state.py`, and `vercor/setups/_external/camulator_fields.py`: replace broad `nan_to_num` cleanup with NaN-only replacement that rejects infinities; guard CAMulator zero-variance normalization.
- Modify the existing numerical/component test files named in Tasks 3-5: add explicit transform coverage beside each owner rather than creating a parallel component test hierarchy.
- Modify `docs/api-architecture-review.md`, `DEPENDENCIES.md`, `DESIGN.md`, and `PROGRESS.md`: keep the private inventory, dependency order, stable design, and audit evidence synchronized.

---

### Task 1: Build the Numerical-Safety Foundation

**Files:**
- Create: `vercor/_numerical_safety.py`
- Create: `tests/test_numerical_safety.py`
- Modify: `tests/assertions.py`
- Modify: `docs/api-architecture-review.md`
- Modify: `DEPENDENCIES.md`
- Modify: `PROGRESS.md`

**Interfaces:**
- Consumes: `vercor.exceptions.CouplerError`, `jax.Array`, `jax.debug.callback`, and JAX array primitives.
- Produces: `require_active_finite(values: Any, *, active_mask: Any | None, owner: str) -> None`.
- Produces: `require_strictly_positive(values: Any, *, owner: str) -> None`.
- Produces: `replace_missing_nan(values: Any, *, owner: str, fill_value: float = 0.0) -> jax.Array`.
- Produces: `safe_masked_divide(numerator: Any, denominator: Any, *, where: Any, inactive_value: Any) -> jax.Array`.
- Produces for tests: `assert_finite_jvp_vjp(objective: Callable[[Any], jax.Array], primal: Any, tangent: Any, *, rtol: float = 1e-6, atol: float = 1e-8) -> None`.

- [ ] **Step 1: Add the reusable transform assertion and focused failing contract tests**

Add this helper to `tests/assertions.py`; it intentionally accepts a scalar-output objective so VJP cotangent construction stays unambiguous:

```python
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp


def assert_finite_jvp_vjp(
    objective: Callable[[Any], jax.Array],
    primal: Any,
    tangent: Any,
    *,
    rtol: float = 1e-6,
    atol: float = 1e-8,
) -> None:
    """Assert one scalar objective has finite, mutually consistent JVP and VJP."""

    eager_value = objective(primal)
    jitted_value = jax.jit(objective)(primal)
    jvp_value, forward = jax.jvp(objective, (primal,), (tangent,))
    vjp_value, pullback = jax.vjp(objective, primal)
    (reverse,) = pullback(jnp.ones_like(vjp_value))

    leaves = jax.tree_util.tree_leaves(
        (eager_value, jitted_value, jvp_value, forward, reverse)
    )
    assert all(bool(jnp.all(jnp.isfinite(leaf))) for leaf in leaves)
    tangent_leaves = jax.tree_util.tree_leaves(tangent)
    reverse_leaves = jax.tree_util.tree_leaves(reverse)
    reverse_projection = sum(
        (
            jnp.vdot(tangent_leaf, reverse_leaf)
            for tangent_leaf, reverse_leaf in zip(
                tangent_leaves,
                reverse_leaves,
                strict=True,
            )
        ),
        start=jnp.asarray(0.0, dtype=jnp.asarray(forward).dtype),
    )
    assert_allclose_compact(
        forward,
        reverse_projection,
        rtol=rtol,
        atol=atol,
        equal_nan=False,
    )
```

Create `tests/test_numerical_safety.py` with behavior-first tests for:

```python
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


def test_active_finite_contract_reports_compiled_failure() -> None:
    checked_sum = jax.jit(
        lambda values: (
            require_active_finite(values, active_mask=None, owner="compiled field"),
            jnp.sum(values),
        )[1]
    )
    with pytest.raises(JaxRuntimeError, match="compiled field.*active domain"):
        checked_sum(jnp.asarray([1.0, jnp.nan])).block_until_ready()


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
```

Also test that a mask whose shape is not a trailing suffix raises a contextual `CouplerError`, `require_strictly_positive` rejects zero/negative/non-finite values in eager and compiled execution, and `replace_missing_nan` replaces NaN but rejects both infinities.

- [ ] **Step 2: Run the focused tests and record the intended RED result**

Run:

```bash
/Users/romannuterman/miniforge3/envs/scipy/bin/python -m pytest tests/test_numerical_safety.py -q -n0 --tb=short
```

Expected: collection fails because `vercor._numerical_safety` does not exist.

- [ ] **Step 3: Implement the private numerical-safety owner**

Create `vercor/_numerical_safety.py` with the following structure and exact public-within-private-module functions:

```python
from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from vercor.exceptions import CouplerError


def _broadcast_active_mask(values: jax.Array, active_mask: Any | None, owner: str) -> jax.Array:
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
    return jnp.where(finite_locations, array, jnp.asarray(fill_value, dtype=array.dtype))


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
    safe_numerator = jnp.where(condition, numerator_array, 0.0)
    safe_denominator = jnp.where(condition, denominator_array, 1.0)
    quotient = safe_numerator / safe_denominator
    return jnp.where(condition, quotient, inactive_value)
```

- [ ] **Step 4: Update the private inventory and dependency order in the same change**

Add `vercor._numerical_safety` to the section-5 private module code block and the “Foundations and numerical implementations” list in `docs/api-architecture-review.md`. Add `vercor/_numerical_safety.py` to dependency layer 2 in `DEPENDENCIES.md`, after the layer-1 `exceptions.py` owner it consumes and before all numerical/runtime callers.

- [ ] **Step 5: Run focused GREEN and architecture checks**

Run:

```bash
/Users/romannuterman/miniforge3/envs/scipy/bin/python -m pytest tests/test_numerical_safety.py tests/test_api_architecture_review.py::test_documented_private_inventory_matches_all_nonpublic_modules -q -n0 --tb=short
```

Expected: all selected tests pass; no callback failure escapes the tests that intentionally catch it.

- [ ] **Step 6: Record the foundation RED/GREEN evidence**

Add a dated top note to `PROGRESS.md` naming the new private owner, the exact focused RED reason, the focused GREEN count, and the private-inventory check. Mark runtime integration and component audits as still pending.

- [ ] **Step 7: Run the required pre-commit regression gates**

Run:

```bash
/Users/romannuterman/miniforge3/envs/scipy/bin/python -m pytest tests/ -q --fast --tb=short
/Users/romannuterman/miniforge3/envs/scipy/bin/python -m pytest tests/ -q --tb=short
git diff --check
```

Expected: the fast and complete suites pass with only the documented third-party warnings; the whitespace check exits zero.

- [ ] **Step 8: Commit the foundation**

```bash
git add vercor/_numerical_safety.py tests/test_numerical_safety.py tests/assertions.py docs/api-architecture-review.md DEPENDENCIES.md PROGRESS.md
git commit -m "feat: add transform-safe numerical policy"
```

---

### Task 2: Enforce Finiteness at Runtime Handoffs

**Files:**
- Modify: `vercor/_runtime/state_validation.py`
- Modify: `vercor/components/runtime_execution.py`
- Modify: `vercor/_runtime/field_transfer.py`
- Modify: `vercor/_runtime/exchange_dispatch.py`
- Modify: `tests/test_coupler_runtime.py`
- Modify: `tests/test_runtime_state.py`
- Modify: `tests/test_runtime_exchange.py`
- Modify: `PROGRESS.md`

**Interfaces:**
- Consumes: Task 1 `require_active_finite` contract.
- Produces: unchanged private runtime call signatures, except helper-only internal arguments where needed.
- Guarantees: initial/foreign state, raw regridding results, component outputs, and selected sent fields fail before the next numerical owner can consume active non-finite data.

- [ ] **Step 1: Write RED tests for initial state and component-step failures**

In `tests/test_coupler_runtime.py`, add a small `CallableComponent` fixture with a two-cell grid mask `[[1.0, 0.0]]`. Add tests equivalent to:

```python
def test_initial_state_rejects_active_nan_but_allows_inactive_missing_nan() -> None:
    invalid = _masked_scalar_component(jnp.asarray([[jnp.nan, jnp.nan]]))
    with pytest.raises(CouplerError, match="Component 'SAFE'.*field 'value'.*active domain"):
        _single_component_coupler(invalid).initial_state()

    valid = _masked_scalar_component(jnp.asarray([[2.0, jnp.nan]]))
    state = _single_component_coupler(valid).initial_state()
    assert np.isfinite(np.asarray(state.component("SAFE").field("value"))[0, 0])
    assert np.isnan(np.asarray(state.component("SAFE").field("value"))[0, 1])


def test_component_step_fails_at_first_active_nonfinite_output() -> None:
    component = _masked_scalar_component(
        jnp.asarray([[2.0, jnp.nan]]),
        step=lambda fields: {"value": fields["value"].at[0, 0].set(jnp.nan)},
        execution="host",
    )
    with pytest.raises(
        CouplerError,
        match="Component 'SAFE' step output field 'value'.*active domain",
    ):
        _single_component_coupler(component).run()
```

Add a compiled version using `execution="jax"` that expects `JaxRuntimeError` with the same owner text after `.block_until_ready()`.

- [ ] **Step 2: Write RED tests for time-selected send fields**

In `tests/test_runtime_state.py`, extend the existing monthly and daily transfer tests with:

```python
def test_runtime_send_rejects_nonfinite_selected_active_field() -> None:
    component = _RuntimeSendComponent(TransferPolicy("daily"))
    component.grid = make_test_grid(binary_mask=jnp.asarray([[1.0, 0.0], [1.0, 0.0]]))
    forcing = jnp.asarray(
        [
            [[1.0, jnp.nan], [2.0, jnp.nan]],
            [[jnp.nan, jnp.nan], [3.0, jnp.nan]],
        ]
    )
    state = _runtime_send_state(forcing)
    with pytest.raises(CouplerError, match="sent field 'temperature'.*active domain"):
        send_runtime_fields(
            component,
            state,
            _daily_step_info(index=1),
            contract=ExchangeContract(sends=("temperature",)),
        )
```

Use the file's existing fixtures and constructors rather than mutating a frozen grid or component if those helpers expose immutable objects. The asserted selected slice must contain one active NaN and only inactive NaNs elsewhere.

- [ ] **Step 3: Write RED exchange tests for scalar and vector route activity**

In `tests/test_runtime_exchange.py`, introduce `_NonFiniteRegridder` and add:

```python
def test_exchange_rejects_nonfinite_active_regridding_output() -> None:
    regridder = _NonFiniteRegridder(jnp.asarray([[1.0, jnp.nan]]))
    state, exchange = _one_field_exchange_state(mask=jnp.asarray([[1.0, 1.0]]))
    with pytest.raises(CouplerError, match="exchange 'OCN->ATM'.*temperature.*active domain"):
        dispatch_component_exchanges(state, "ATM", (exchange,), {exchange.route_id: regridder})


def test_exchange_neutralizes_inactive_nan_without_poisoning_jvp_or_vjp() -> None:
    mask = jnp.asarray([[1.0, 0.0]])

    def objective(source: jax.Array) -> jax.Array:
        received = _dispatch_with_passthrough_regridder(source, mask)
        return jnp.sum(received)

    assert_finite_jvp_vjp(
        objective,
        jnp.asarray([[2.0, jnp.nan]]),
        jnp.asarray([[1.0, 0.0]]),
    )
    assert_allclose_compact(
        _dispatch_with_passthrough_regridder(jnp.asarray([[2.0, jnp.nan]]), mask),
        np.asarray([[2.0, 0.0]]),
        equal_nan=False,
    )
```

Add the corresponding vector-output rejection using both vector components and the route mask.

- [ ] **Step 4: Run runtime RED tests**

Run:

```bash
/Users/romannuterman/miniforge3/envs/scipy/bin/python -m pytest tests/test_coupler_runtime.py tests/test_runtime_state.py tests/test_runtime_exchange.py -q -n0 --tb=short -k "nonfinite or active_nan or inactive_nan"
```

Expected: the new cases fail because field stores and handoffs do not yet call the numerical-safety owner, and scalar exchange still evaluates `NaN * 0`.

- [ ] **Step 5: Add route-aware validation to runtime state checking**

In `vercor/_runtime/state_validation.py`, derive the unique inbound route mask for every `(target_component, flattened_field_name)` from `exchanges` and `runtime_state._fractional_mask(exchange.route_id)`. Fan-in is already rejected, so each received field has at most one route mask.

After existing name/shape/dtype checks, validate stores as follows:

```python
for store_name in ("fields", "received", "sent"):
    store = getattr(component_state, store_name)
    for field_name in store.field_names:
        inbound_mask = received_masks.get((component.name, field_name))
        active_mask = (
            inbound_mask
            if store_name in {"fields", "received"} and inbound_mask is not None
            else component.grid.binary_mask
        )
        require_active_finite(
            store.get(field_name),
            active_mask=active_mask,
            owner=(
                f"Component '{component.name}' runtime {store_name} "
                f"field '{field_name}'"
            ),
        )
```

Keep schema validation and author lifecycle validation in their existing owners. Do not add route metadata to public component contracts.

- [ ] **Step 6: Validate component results and selected outgoing fields**

In `vercor/components/runtime_execution.py`, assign `apply_step_result(...)` to `updated_state`, validate every name in `component.spec.outputs` with `component.grid.binary_mask`, and then return `updated_state`:

```python
for field_name in component.spec.outputs:
    require_active_finite(
        updated_state.fields.get(field_name),
        active_mask=component.grid.binary_mask,
        owner=f"Component '{component.name}' step output field '{field_name}'",
    )
```

In `vercor/_runtime/field_transfer.py`, build the selected-field mapping before `set_many`, validate each selected value against `component.grid.binary_mask`, then update the sent store. Keep monthly/daily selection logic unchanged.

- [ ] **Step 7: Validate raw regridding output and neutralize inactive scalar cells**

In `vercor/_runtime/exchange_dispatch.py`, pass the exchange route ID and fractional mask into both scalar and vector primitives. Validate each raw scalar/vector result with the positive fractional mask. For scalar dispatch, use:

```python
active = jnp.asarray(fractional_mask) > 0
weighted = jnp.asarray(regridded) * jnp.asarray(fractional_mask)
received_updates[field_name] = jnp.where(active, weighted, 0.0)
```

Compute `weighted` from a finite-neutralized regridded operand so the inactive branch never evaluates `NaN * 0`; use `jnp.where(active, regridded, 0.0) * fractional_mask`. Vector behavior remains numerically unscaled, but raw `u` and `v` outputs are validated against the route activity mask before storage.

- [ ] **Step 8: Run focused GREEN plus runtime architecture contracts**

Run:

```bash
/Users/romannuterman/miniforge3/envs/scipy/bin/python -m pytest tests/test_coupler_runtime.py tests/test_runtime_state.py tests/test_runtime_exchange.py tests/test_component_boundary_contracts.py tests/test_runtime_facade_boundaries.py -q -n0 --tb=short
```

Expected: all selected tests pass with unchanged public/runtime ownership assertions.

- [ ] **Step 9: Record runtime-boundary RED/GREEN evidence**

Update the active `PROGRESS.md` note with the exact initial-state, step-output, send, scalar-exchange, and vector-exchange RED failures and focused GREEN count. Record that inactive NaNs remain accepted while inactive infinity and active non-finite values fail.

- [ ] **Step 10: Run the required pre-commit regression gates**

Run:

```bash
/Users/romannuterman/miniforge3/envs/scipy/bin/python -m pytest tests/ -q --fast --tb=short
/Users/romannuterman/miniforge3/envs/scipy/bin/python -m pytest tests/ -q --tb=short
git diff --check
```

Expected: fast and complete suites pass; whitespace is clean.

- [ ] **Step 11: Commit runtime enforcement**

```bash
git add vercor/_runtime/state_validation.py vercor/components/runtime_execution.py vercor/_runtime/field_transfer.py vercor/_runtime/exchange_dispatch.py tests/test_coupler_runtime.py tests/test_runtime_state.py tests/test_runtime_exchange.py PROGRESS.md
git commit -m "feat: reject active nonfinite runtime fields"
```

---

### Task 3: Remove Masked-Division Hazards from Regridding

**Files:**
- Modify: `vercor/_interpolators/bilinear_rectilinear.py`
- Modify: `vercor/_interpolators/_bilinear_extrapolation.py`
- Modify: `vercor/_interpolators/conservative_remap_rectilinear.py`
- Modify: `tests/test_bilinear_rectilinear_interpolator.py`
- Modify: `tests/test_conservative_rectilinear_remapper.py`
- Modify: `PROGRESS.md`

**Interfaces:**
- Consumes: Task 1 `safe_masked_divide` and `tests.assertions.assert_finite_jvp_vjp`.
- Produces: unchanged interpolator/remapper constructor and application APIs with finite derivatives when missing support is inactive.

- [ ] **Step 1: Add explicit JVP/VJP tests for zero-support bilinear cells**

In `tests/test_bilinear_rectilinear_interpolator.py`, test scalar and vector paths with a source mask that removes every supporting corner and a target mask that marks the target inactive. The objective must use `jnp.nansum` so the intentionally missing output does not enter the scalar loss:

```python
def test_zero_support_bilinear_cells_have_finite_jvp_and_vjp() -> None:
    interpolator = BilinearRectilinearInterpolator(
        lon_src=jnp.asarray([0.0, 1.0]),
        lat_src=jnp.asarray([0.0, 1.0]),
        lon_tgt=jnp.asarray([0.5]),
        lat_tgt=jnp.asarray([0.5]),
        src_mask=jnp.zeros((2, 2), dtype=bool),
        tgt_mask=jnp.zeros((1, 1), dtype=bool),
    )
    objective = lambda source: jnp.nansum(interpolator.apply_scalar(source))
    assert_finite_jvp_vjp(
        objective,
        jnp.ones((2, 2)),
        jnp.ones((2, 2)),
    )
```

Add the vector equivalent with one scalar objective summing both returned arrays.

- [ ] **Step 2: Add explicit JVP/VJP tests for IDW and conservative zero support**

Add an IDW case with fewer valid values than `idw_k`, and in `tests/test_conservative_rectilinear_remapper.py` add a fully masked `normalize="fracarea"` case. Assert intentional output NaNs remain present and use `assert_finite_jvp_vjp` on a `jnp.nansum` objective.

- [ ] **Step 3: Demonstrate why the raw internal divisions violate the source policy**

Add the source assertion to `tests/test_bilinear_rectilinear_interpolator.py`, reading the three files with `Path.read_text`, and require that the production files no longer contain these exact raw patterns:

```python
assert "num / wsum" not in bilinear_source
assert "jnp.sum(weights * val_k, axis=1) / wsum" not in extrapolation_source
assert "weighted_sum / norm" not in conservative_source
```

Run:

```bash
/Users/romannuterman/miniforge3/envs/scipy/bin/python -m pytest tests/test_bilinear_rectilinear_interpolator.py tests/test_conservative_rectilinear_remapper.py -q -n0 --tb=short -k "zero_support or idw"
```

Expected: the source-policy assertions fail on the three current raw masked divisions.

- [ ] **Step 4: Replace only the masked divisions with the shared safe primitive**

In scalar bilinear interpolation:

```python
out = safe_masked_divide(
    num,
    wsum,
    where=wsum > 0.0,
    inactive_value=jnp.nan,
)
```

Use the same pattern for vector `vt3`, broadcasting the condition with `[..., None]`. In IDW extrapolation, safely divide the weighted sum by `wsum`, using `fill_value` as the inactive value. In conservative `fracarea`/conservation output, safely divide by `norm` where `norm > 1e-15`, using NaN outside supported cells.

Do not alter interpolation weights, source/target mask conventions, extrapolation choices, conservation normalization, or public defaults.

- [ ] **Step 5: Run complete regridding GREEN tests**

Run:

```bash
/Users/romannuterman/miniforge3/envs/scipy/bin/python -m pytest tests/test_bilinear_rectilinear_interpolator.py tests/test_bilinear_rectilinear_regridder.py tests/test_conservative_rectilinear_remapper.py tests/test_conservative_rectilinear_regridder.py tests/test_runtime_exchange.py -q -n0 --tb=short
```

Expected: all scalar, vector, extrapolation, conservation, missing-data, JIT, JVP, and VJP tests pass.

- [ ] **Step 6: Record the regridding audit evidence**

Update `PROGRESS.md` with scalar, vector, IDW, and conservative masked-division RED/GREEN results and the explicit JVP/VJP outcomes. Record that interpolation values and missing-target NaN semantics did not change.

- [ ] **Step 7: Run the required pre-commit regression gates**

Run:

```bash
/Users/romannuterman/miniforge3/envs/scipy/bin/python -m pytest tests/ -q --fast --tb=short
/Users/romannuterman/miniforge3/envs/scipy/bin/python -m pytest tests/ -q --tb=short
git diff --check
```

Expected: fast/full suites and whitespace pass.

- [ ] **Step 8: Commit regridding safety**

```bash
git add vercor/_interpolators/bilinear_rectilinear.py vercor/_interpolators/_bilinear_extrapolation.py vercor/_interpolators/conservative_remap_rectilinear.py tests/test_bilinear_rectilinear_interpolator.py tests/test_conservative_rectilinear_remapper.py PROGRESS.md
git commit -m "fix: make masked regridding derivatives finite"
```

---

### Task 4: Audit Every Bundled Component and Adapter

**Files:**
- Modify: `vercor/setups/_external/jax_gcm_fields.py`
- Modify: `vercor/setups/_external/veros_state.py`
- Modify: `vercor/setups/_external/camulator_fields.py`
- Modify: `tests/test_slab_kernels.py`
- Modify: `tests/test_data_component_kernels.py`
- Modify: `tests/test_fluxes_utilities.py`
- Modify: `tests/test_helpers_coverage.py`
- Modify: `tests/test_tools_assets_and_regridding.py`
- Modify: `tests/test_dtypes.py`
- Modify: `tests/test_camulator_component_kernels.py`
- Modify: `tests/test_external_components_coverage.py`
- Modify: `tests/test_coupler_runtime.py`
- Modify: `PROGRESS.md`

**Interfaces:**
- Consumes: Task 1 `replace_missing_nan`, `require_active_finite`, `require_strictly_positive`, and `assert_finite_jvp_vjp`.
- Produces: the existing JCM, Veros, CAMulator, slab, data, flux, and vertical-coordinate interfaces with explicit finite transform evidence.
- Guarantees: broad adapter cleanup never converts infinity into a finite maximum value.

- [ ] **Step 1: Extend slab and data kernel tests to explicit JVP/VJP coverage**

In `tests/test_slab_kernels.py`, retain the existing closed-form assertions and add one scalar objective per private kernel:

```python
assert_finite_jvp_vjp(
    lambda sst: jnp.sum(_bulk_flux_step(jnp.full_like(sst, 288.0), sst)[0]),
    jnp.full((2, 2), 285.0),
    jnp.ones((2, 2)),
)
assert_finite_jvp_vjp(
    lambda sst: jnp.sum(
        _advance_sea_surface_temperature(
            sst, jnp.full_like(sst, 10.0), jnp.full_like(sst, 5.0),
            3600.0, 1025.0, 3990.0, 30.0, 1.0 / (30.0 * 86400.0), 288.15,
        )
    ),
    jnp.full((2, 2), 288.0),
    jnp.ones((2, 2)),
)
```

Add equivalent objectives for `_update_soil_moisture` and `_diagnose_ice_fraction`. In `tests/test_data_component_kernels.py`, add transform assertions for ERA5 atmosphere field mapping, shared masked surface fields using zero tangent on inactive NaNs, ERA5 ocean mask construction through a differentiable temperature field, ERA-Interim field assembly, and JCM land time-slice selection.

- [ ] **Step 2: Extend flux and vertical-coordinate tests to explicit VJP evidence**

In `tests/test_fluxes_utilities.py`, use `assert_finite_jvp_vjp` for:

- `qsat` over 260-300 K;
- `qsat_august_eqn` over positive pressure/temperature;
- `compute_air_density` and `compute_potential_temperature` over positive fields;
- `get_altitudes_hybrid_sigma_levels` including the zero top half-level case;
- `get_altitudes_sigma_levels` over strictly decreasing positive pressure; and
- scalar objectives over all leaves returned by `compute_ocean_surface_fluxes` and `shr_flux_atmIce`.

For tuple outputs, the objective is the sum of `jnp.sum(value)` for each returned array. Keep existing finite-difference comparisons as an independent check.

In `tests/test_helpers_coverage.py`, add a finite scalar JVP/VJP objective for `centers_to_edges` on valid latitude and longitude centers. In `tests/test_tools_assets_and_regridding.py`, add finite transform objectives for `compute_land_mask`, `compute_ocn_lnd_masks_on_atm_grid`, and the area-weighted global mean using valid fractional masks and fields. In `tests/test_dtypes.py`, differentiate an objective whose sole argument is a floating leaf normalized through the dtype policy, and assert finite JIT/JVP/VJP results. Test integer-leaf preservation separately without passing integer leaves through `jax.jvp`.

- [ ] **Step 3: Write RED tests proving adapter cleanup rejects infinity**

Add parameterized tests beside the current JCM, Veros, and CAMulator cleanup tests:

```python
@pytest.mark.parametrize("bad_value", [jnp.inf, -jnp.inf])
def test_jcm_surface_cleanup_rejects_infinity(bad_value: float) -> None:
    with pytest.raises(CouplerError, match="JCM.*surface temperature.*infinity"):
        cleanup_surface_temperature_fields(
            jnp.asarray([[bad_value, jnp.nan]]),
            jnp.asarray([[jnp.nan, 280.0]]),
        )


def test_veros_surface_forcing_rejects_infinity() -> None:
    with pytest.raises(CouplerError, match="Veros.*surface forcing.*infinity"):
        prepare_surface_forcing_fields(
            jnp.asarray([[jnp.inf]]),
            jnp.zeros((1, 1)),
            jnp.zeros((1, 1)),
            jnp.zeros((1, 1)),
            True,
        )


def test_camulator_surface_forcing_rejects_infinity_and_zero_variance() -> None:
    with pytest.raises(CouplerError, match="CAMulator.*surface temperature.*infinity"):
        prepare_camulator_surface_forcing(
            jnp.asarray([[jnp.inf, 280.0]]),
            jnp.asarray([[jnp.nan, jnp.nan]]),
            jnp.zeros((1, 2)),
        )
    with pytest.raises(CouplerError, match="standard deviation.*strictly positive"):
        prepare_camulator_surface_forcing(
            jnp.full((2, 2), 280.0),
            jnp.zeros((2, 2)),
            jnp.zeros((2, 2)),
        )
```

Add compiled variants expecting `JaxRuntimeError` with the same owner fragments.

- [ ] **Step 4: Run the adapter RED cases**

Run:

```bash
/Users/romannuterman/miniforge3/envs/scipy/bin/python -m pytest tests/test_camulator_component_kernels.py tests/test_external_components_coverage.py -q -n0 --tb=short -k "rejects_infinity or zero_variance"
```

Expected: infinity cases fail because `nan_to_num` converts infinity to finite extrema; the constant CAMulator case fails because it returns NaNs instead of raising contextually.

- [ ] **Step 5: Replace broad cleanup with NaN-only replacement and guard normalization**

In JCM cleanup, replace both `jnp.nan_to_num` calls with:

```python
land_surface_temperature_array = replace_missing_nan(
    as_jax_real_array(land_surface_temperature),
    owner="JCM land surface temperature",
)
sea_surface_temperature_array = replace_missing_nan(
    as_jax_real_array(sea_surface_temperature),
    owner="JCM sea surface temperature",
)
```

In Veros `_prepare`, accept an `owner` argument and use `replace_missing_nan(field_jax, owner=owner)` before transpose/axis expansion. Supply `Veros taux surface forcing`, `Veros tauy surface forcing`, `Veros qnet surface forcing`, and `Veros qnec surface forcing` as labels.

In CAMulator surface preparation, replace `nan_to_num` for `sst` and `skt` with `replace_missing_nan`. Compute `mean` and `standard_deviation` with ordinary `jnp.mean`/`jnp.std` after replacement, call:

```python
require_strictly_positive(
    standard_deviation,
    owner="CAMulator surface-temperature standard deviation",
)
```

then divide. Validate both returned arrays with `require_active_finite(..., active_mask=None, ...)`. Do not add an epsilon or select an arbitrary normalized fallback.

- [ ] **Step 6: Add explicit JVP/VJP coverage to optional adapters**

Use `assert_finite_jvp_vjp` on valid, nonconstant representative inputs for:

- JCM `cleanup_surface_temperature_fields`, `prepare_surface_temperature_forcing`, and a scalar objective over `map_jcm_output_fields`;
- Veros `update_veros_interior`, `extract_surface_temperature`, and `prepare_surface_forcing_fields`;
- CAMulator `prepare_camulator_surface_forcing`, `prepare_camulator_dynamic_forcing_chunk`, `prepare_camulator_sst_input`, and a scalar objective over `map_camulator_prediction_arrays`.

In `tests/test_coupler_runtime.py::test_real_jax_gcm_runtime_seeds_and_advances_jcm_2_carry`, retain `value_and_grad` and add an explicit `jax.vjp(full_coupler_loss, coupled_surface_temperature)` pullback. Assert the returned cotangent is finite and agrees with the existing forward tangent under the scalar inner-product identity.

Veros and CAMulator native host calls remain primal-only. Their existing runtime tests must additionally assert every returned runtime field is finite before host application.

- [ ] **Step 7: Run complete component and adapter GREEN tests**

Run:

```bash
/Users/romannuterman/miniforge3/envs/scipy/bin/python -m pytest tests/test_slab_kernels.py tests/test_data_component_kernels.py tests/test_fluxes_utilities.py tests/test_helpers_coverage.py tests/test_tools_assets_and_regridding.py tests/test_dtypes.py tests/test_camulator_component_kernels.py tests/test_external_components_coverage.py tests/test_coupler_runtime.py -q -n0 --tb=short
```

Expected: every implemented bundled kernel has finite eager/JIT/JVP/VJP evidence where differentiable; host-only paths have finite primal evidence; infinity and zero-variance failures are contextual.

- [ ] **Step 8: Record the bundled-component audit evidence**

Update the active `PROGRESS.md` entry with the exact slab, data, geometry/mask, dtype, flux/vertical, JCM, Veros, and CAMulator focused counts. Record infinity-cleanup and zero-variance RED/GREEN results and distinguish real optional-model tests from fake-boundary tests.

- [ ] **Step 9: Run the required pre-commit regression gates**

Run:

```bash
/Users/romannuterman/miniforge3/envs/scipy/bin/python -m pytest tests/ -q --fast --tb=short
/Users/romannuterman/miniforge3/envs/scipy/bin/python -m pytest tests/ -q --tb=short
git diff --check
```

Expected: fast/full suites and whitespace pass with only known third-party warnings.

- [ ] **Step 10: Commit the bundled-component audit**

```bash
git add vercor/setups/_external/jax_gcm_fields.py vercor/setups/_external/veros_state.py vercor/setups/_external/camulator_fields.py tests/test_slab_kernels.py tests/test_data_component_kernels.py tests/test_fluxes_utilities.py tests/test_helpers_coverage.py tests/test_tools_assets_and_regridding.py tests/test_dtypes.py tests/test_camulator_component_kernels.py tests/test_external_components_coverage.py tests/test_coupler_runtime.py PROGRESS.md
git commit -m "fix: enforce finite bundled component numerics"
```

---

### Task 5: Verify Composed Workflows and Record the Audit

**Files:**
- Modify: `tests/test_coupler_runtime.py`
- Modify: `tests/test_v0_4_workflow_execution.py`
- Modify: `DESIGN.md`
- Modify: `PROGRESS.md`

**Interfaces:**
- Consumes: Tasks 1-4 and the approved design spec.
- Produces: executable end-to-end no-NaN evidence and durable project documentation.

- [ ] **Step 1: Make every existing differentiable Coupler regression exercise explicit VJP**

For each of these existing scalar loss tests, retain its current `jax.grad`/JVP assertions and add `jax.vjp`:

- `test_run_supports_jit_grad_and_jvp`;
- `test_mixed_grid_slab_coupler_runs_with_real_regridders_under_jit_grad_and_jvp`;
- daily/monthly forcing replay tests;
- `test_jax_gcm_runs_inside_runtime_under_jit_and_grad`;
- `test_real_jax_gcm_runtime_seeds_and_advances_jcm_2_carry`; and
- `test_data_forcing_replays_into_jax_gcm_runtime_under_jit_grad_and_jvp`.

Use this exact scalar pattern:

```python
value, pullback = jax.vjp(loss, primal)
(reverse_vjp,) = pullback(jnp.ones_like(value))
assert np.all(np.isfinite(np.asarray(reverse_vjp)))
assert_allclose_compact(
    jnp.vdot(tangent_seed, reverse_vjp),
    forward_tangent,
    rtol=1e-5,
    atol=1e-8,
    equal_nan=False,
)
```

Do not weaken existing reference-value, calendar-selection, or reverse-gradient checks.

- [ ] **Step 2: Strengthen workflow-level derivative evidence**

In `tests/test_v0_4_workflow_execution.py`, update both output-free differentiation tests to call `jax.vjp` explicitly and verify:

```python
assert np.isfinite(np.asarray(primal))
assert np.isfinite(np.asarray(tangent))
assert np.isfinite(np.asarray(reverse_vjp))
assert_allclose_compact(tangent, reverse_vjp, equal_nan=False)
```

Add one inactive-missing-data Coupler objective whose initial state contains NaN only outside the grid mask and whose tangent seed is zero there. Verify eager, outer JIT, JVP, and VJP all retain a finite active objective and finite derivative projection.

- [ ] **Step 3: Run the composed audit tests**

Run:

```bash
/Users/romannuterman/miniforge3/envs/scipy/bin/python -m pytest tests/test_coupler_runtime.py tests/test_v0_4_workflow_execution.py tests/test_numerical_safety.py -q -n0 --tb=short
```

Expected: all composed slab, forcing, regridding, payload, JCM, inactive-NaN, JIT, JVP, and VJP paths pass.

- [ ] **Step 4: Update stable design and progress evidence**

Add a concise invariant to `DESIGN.md` sections 1, 6, and 10:

```markdown
- Active runtime fields are finite at preparation, exchange, component-step,
  and send boundaries. NaN remains a missing-data sentinel only outside the
  applicable component or route mask; infinity is always invalid.
- Masked numerical kernels neutralize inactive operands before nonlinear
  arithmetic so output-free JVP and VJP cannot inherit inactive-branch NaNs.
```

Add a dated top entry to `PROGRESS.md` containing a compact matrix with rows for slab, data, flux/vertical, bilinear, conservative, exchange/runtime, JCM, Veros, CAMulator, and composed Coupler paths. Columns are `Primal/JIT`, `JVP`, `VJP`, `Inactive NaN`, and `Fail-fast`; use `pass`, `primal-only`, or `not applicable` based on actual test evidence. Record each first-source numerical repair and the exact final gate counts. Do not claim optional real-model coverage for a dependency or data path that was skipped.

- [ ] **Step 5: Run formatting and static verification**

Run:

```bash
/Users/romannuterman/miniforge3/envs/scipy/bin/python -m black vercor tests
/Users/romannuterman/miniforge3/envs/scipy/bin/python -m flake8 . --count --max-line-length=120 --statistics
/Users/romannuterman/miniforge3/envs/scipy/bin/python -m mypy vercor tests
/Users/romannuterman/miniforge3/envs/scipy/bin/python -m compileall -q vercor tests
git diff --check
```

Expected: Black exits zero, flake8 reports zero findings, mypy reports no issues, compileall exits zero, and whitespace is clean.

- [ ] **Step 6: Run focused, fast, full, and coverage verification**

Run:

```bash
/Users/romannuterman/miniforge3/envs/scipy/bin/python -m pytest tests/test_numerical_safety.py tests/test_runtime_exchange.py tests/test_slab_kernels.py tests/test_data_component_kernels.py tests/test_fluxes_utilities.py tests/test_camulator_component_kernels.py tests/test_external_components_coverage.py tests/test_coupler_runtime.py tests/test_v0_4_workflow_execution.py -q -n0 --tb=short
/Users/romannuterman/miniforge3/envs/scipy/bin/python -m pytest tests/ -q --fast --tb=short
/Users/romannuterman/miniforge3/envs/scipy/bin/python -m pytest tests/ -q --tb=short
/Users/romannuterman/miniforge3/envs/scipy/bin/python -m pytest tests/ -q --cov=vercor --cov-branch --tb=short
```

Expected: every command passes; branch coverage remains at or above the configured 90% floor. Copy exact test counts, warnings, statement count, branch count, and coverage percentage into `PROGRESS.md`, then rerun the focused progress/documentation contracts if the recorded numbers change executable evidence.

- [ ] **Step 7: Build and inspect fresh package artifacts without touching existing `dist/`**

Create and use one fresh temporary build directory:

```bash
VERCOR_NAN_BUILD_DIR="$(mktemp -d /private/tmp/vercor-no-nan-build.XXXXXX)"
/Users/romannuterman/miniforge3/envs/scipy/bin/python -m build --outdir "$VERCOR_NAN_BUILD_DIR"
/Users/romannuterman/miniforge3/envs/scipy/bin/python -m twine check "$VERCOR_NAN_BUILD_DIR/vercor-0.4.4-py3-none-any.whl" "$VERCOR_NAN_BUILD_DIR/vercor-0.4.4.tar.gz"
/Users/romannuterman/miniforge3/envs/scipy/bin/python -m pytest tests/test_distribution_boundaries.py tests/test_production_numpy_boundaries.py -q -n0 --tb=short
```

Expected: exactly the 0.4.4 wheel and source distribution are built, Twine accepts both, and artifact/boundary tests pass. Record the printed temporary path in `PROGRESS.md`; never remove or overwrite an unknown directory.

- [ ] **Step 8: Review requirements and the complete diff**

Run:

```bash
git status --short --branch
git diff --stat 54b29bde1d1fd1c8337ef7f779299e54cf9af5b7
git diff --check 54b29bde1d1fd1c8337ef7f779299e54cf9af5b7
git diff 54b29bde1d1fd1c8337ef7f779299e54cf9af5b7 -- vercor tests DESIGN.md DEPENDENCIES.md PROGRESS.md docs/api-architecture-review.md
```

Review every acceptance criterion in the spec against a production owner and a test. Confirm no public export changed, no `nan_to_num` remains in differentiable computational adapters, no active value is silently filled, and all intentional NaNs are confined to masked/output-only semantics.

- [ ] **Step 9: Commit final audit evidence after fresh full verification**

After any review correction, rerun Black, flake8, mypy, compileall, focused tests, fast tests, the full suite, coverage, and `git diff --check`. Then commit:

```bash
git add DESIGN.md PROGRESS.md tests/test_coupler_runtime.py tests/test_v0_4_workflow_execution.py
git commit -m "test: verify no-NaN differentiation policy"
```

- [ ] **Step 10: Verify the committed branch state**

Run:

```bash
git status --short --branch
git log --oneline --decorate -6
git diff --check main...HEAD
```

Expected: the branch is clean, the design and implementation commits are visible, and the committed range has no whitespace errors. Do not push, create a PR, tag, publish, or upload artifacts without separate authorization.
