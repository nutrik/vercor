"""Focused coverage for the 0.4 component adapters and private runtime bridge."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import jax.numpy as jnp
import jax
import numpy as np
import pytest

from tests._coverage_support import make_test_grid
from vercor.clock import Clock
from vercor.components import (
    CallableComponent,
    ComponentMaskSpec,
    ComponentSpec,
    DataComponent,
    LifecycleHooks,
    PrefillResult,
    SetupResult,
    StepContext,
    StepResult,
    TransferPolicy,
)
from vercor.coupler import Coupler
from vercor.dtypes import DTypePolicy
from vercor.exceptions import ComponentError
from vercor.runtime import RuntimeOptions
from vercor._runtime.contracts import ExchangeContract
from vercor._runtime.field_transfer import select_runtime_field
from vercor._runtime.time import initial_runtime_step_info


def _coupler(component: Any, *, steps: int = 1) -> Coupler:
    return Coupler(
        Clock(datetime(2000, 1, 1), 60.0, steps),
        components=(component,),
        run_order=(component.name,),
    )


@pytest.mark.parametrize("arity", [1, 2, 3])
def test_callable_component_adapts_supported_step_arities(arity: int) -> None:
    def one(fields: Any) -> Any:
        return {"value": fields["value"] + 1.0}

    def two(fields: Any, context: Any) -> Any:
        return {"value": fields["value"] + context.dt_seconds / 60.0}

    def three(fields: Any, context: Any, payload: Any) -> Any:
        _ = context
        return {"value": fields["value"] + payload}

    setup = None
    step = (one, two, three)[arity - 1]
    if arity == 3:
        setup = LifecycleHooks(
            setup=lambda component, context: SetupResult(payload=jnp.asarray(1.0))
        )
    component = CallableComponent(
        "MODEL",
        make_test_grid(),
        step,
        spec=ComponentSpec(
            outputs=("value",),
            initial_fields={"value": 0.0},
            lifecycle=setup,
        ),
    )

    result = _coupler(component).run()

    np.testing.assert_array_equal(
        result.component("MODEL").field("value"), np.ones((2, 2))
    )


@pytest.mark.parametrize(
    "step",
    [lambda: {}, lambda a, b, c, d: {}, lambda fields, *, required: {}],
)
def test_callable_component_rejects_invalid_step_signatures(step: Any) -> None:
    with pytest.raises(ComponentError, match="1, 2, or 3 positional"):
        CallableComponent("MODEL", make_test_grid(), step)


def test_callable_component_preparation_retains_only_normalized_step() -> None:
    component = CallableComponent("MODEL", make_test_grid(), lambda fields: {})
    prepared = _coupler(component)._ensure_prepared().components["MODEL"]

    assert not hasattr(component, "_author_step")
    assert not hasattr(prepared, "_author_step")
    assert (
        prepared.step(
            {},
            StepContext(dt_seconds=60.0, time=datetime(2000, 1, 1)),
        )
        == {}
    )


def test_step_result_can_preserve_replace_and_clear_payload() -> None:
    def step(fields: Any, context: Any, payload: Any) -> StepResult:
        _ = fields, context
        if payload == 1:
            return StepResult(payload=2)
        return StepResult(payload=None)

    component = CallableComponent(
        "MODEL",
        make_test_grid(),
        step,
        spec=ComponentSpec(
            execution="host",
            lifecycle=LifecycleHooks(
                setup=lambda component, context: SetupResult(payload=1)
            ),
        ),
    )

    state = _coupler(component).run()
    assert state._component_state("MODEL").payload == 2

    state = _coupler(component).run(state)
    assert state._component_state("MODEL").payload is None


def test_setup_and_prefill_reject_wrong_result_types() -> None:
    setup_component = CallableComponent(
        "SETUP",
        make_test_grid(),
        lambda fields: {},
        spec=ComponentSpec(
            lifecycle=LifecycleHooks(setup=lambda component, context: {})  # type: ignore[arg-type, return-value]
        ),
    )
    with pytest.raises(ComponentError, match="setup must return SetupResult"):
        _coupler(setup_component).initial_state()

    prefill_component = CallableComponent(
        "PREFILL",
        make_test_grid(),
        lambda fields: {},
        spec=ComponentSpec(
            lifecycle=LifecycleHooks(prefill=lambda component, context: {})  # type: ignore[arg-type, return-value]
        ),
    )
    with pytest.raises(ComponentError, match="prefill must return PrefillResult"):
        _coupler(prefill_component).initial_state()


def test_prefill_result_populates_declared_runtime_fields() -> None:
    component = CallableComponent(
        "MODEL",
        make_test_grid(),
        lambda fields: {},
        spec=ComponentSpec(
            outputs=("value",),
            lifecycle=LifecycleHooks(
                prefill=lambda component, context: PrefillResult(
                    fields={"value": jnp.full((2, 2), 4.0)}
                )
            ),
        ),
    )

    state = _coupler(component).initial_state()
    np.testing.assert_array_equal(
        state.component("MODEL").field("value"), np.full((2, 2), 4.0)
    )


def test_step_result_rejects_shape_changes() -> None:
    component = CallableComponent(
        "MODEL",
        make_test_grid(),
        lambda fields: {"value": jnp.zeros((3, 3))},
        spec=ComponentSpec(outputs=("value",)),
    )

    with pytest.raises(ComponentError, match="invalid step field update"):
        _coupler(component).run()


def _prepared_prefill_binding(result: PrefillResult) -> Any:
    component = CallableComponent(
        "MODEL",
        make_test_grid(),
        lambda fields: {},
        spec=ComponentSpec(
            inputs=("incoming",),
            outputs=("outgoing",),
            initial_fields={"outgoing": 0.0},
            lifecycle=LifecycleHooks(prefill=lambda component, context: result),
        ),
    )
    coupler = Coupler(
        Clock(datetime(2000, 1, 1), 60.0, 1),
        components=(component,),
        run_order=(component.name,),
        runtime=RuntimeOptions(dtype=DTypePolicy(enable_x64=False)),
    )
    return coupler._ensure_prepared().components["MODEL"]


@pytest.mark.parametrize("store_name", ["received", "sent"])
def test_prefill_rejects_fields_absent_from_exchange_contract(
    store_name: str,
) -> None:
    result = PrefillResult(**{store_name: {"secret": 1.0}})
    binding = _prepared_prefill_binding(result)

    with pytest.raises(
        ComponentError,
        match=rf"prefill {store_name}.*'secret'.*exchange contract",
    ):
        binding._prefill_runtime_state_fields(
            dict(binding._data),
            {},
            {},
            ExchangeContract(receives=("incoming",), sends=("outgoing",)),
        )


@pytest.mark.parametrize("store_name", ["received", "sent"])
def test_prefill_rejects_non_grid_runtime_store_shapes(store_name: str) -> None:
    field_name = "incoming" if store_name == "received" else "outgoing"
    result = PrefillResult(**{store_name: {field_name: np.zeros((3, 3))}})
    binding = _prepared_prefill_binding(result)

    with pytest.raises(
        ComponentError,
        match=rf"prefill {store_name} field '{field_name}'.*shape \(3, 3\).*\(2, 2\)",
    ):
        binding._prefill_runtime_state_fields(
            dict(binding._data),
            {},
            {},
            ExchangeContract(receives=("incoming",), sends=("outgoing",)),
        )


def test_prefill_normalizes_all_store_scalars_and_dtypes_before_update() -> None:
    binding = _prepared_prefill_binding(
        PrefillResult(
            fields={"outgoing": np.int64(3)},
            received={"incoming": np.int64(4)},
            sent={"outgoing": np.int64(5)},
        )
    )
    data = dict(binding._data)
    received: dict[str, Any] = {}
    sent: dict[str, Any] = {}

    binding._prefill_runtime_state_fields(
        data,
        received,
        sent,
        ExchangeContract(receives=("incoming",), sends=("outgoing",)),
    )

    for mapping, field_name, expected in (
        (data, "outgoing", 3.0),
        (received, "incoming", 4.0),
        (sent, "outgoing", 5.0),
    ):
        value = mapping[field_name]
        assert value.shape == binding.grid.shape
        assert value.dtype == jnp.float32
        np.testing.assert_array_equal(value, np.full((2, 2), expected))


def test_prefill_rejects_nonnumeric_store_values_with_component_error() -> None:
    binding = _prepared_prefill_binding(PrefillResult(received={"incoming": object()}))

    with pytest.raises(
        ComponentError,
        match="prefill received field 'incoming'.*real numeric",
    ):
        binding._prefill_runtime_state_fields(
            dict(binding._data),
            {},
            {},
            ExchangeContract(receives=("incoming",), sends=("outgoing",)),
        )


@pytest.mark.fast_always
@pytest.mark.parametrize(
    ("mode", "shape"),
    [
        ("linear", (2, 2)),
        ("linear", (1, 2, 2)),
        ("linear", (13, 2, 2)),
        ("linear", (11, 3, 2, 2)),
        ("daily", (2, 2)),
        ("daily", (1, 2, 2)),
        ("daily", (366, 2, 2)),
        ("daily", (364, 3, 2, 2)),
    ],
)
def test_time_selected_outputs_reject_invalid_record_layouts_before_run(
    mode: str, shape: tuple[int, ...]
) -> None:
    """Output-only data must fail before bad takes become missing samples."""
    component = DataComponent(
        "FORCING",
        make_test_grid(),
        {"temperature": np.ones(shape)},
        spec=ComponentSpec(transfer=TransferPolicy(mode)),  # type: ignore[arg-type]
    )

    with pytest.raises(
        ComponentError,
        match=rf"FORCING.*temperature.*{mode}.*(12|365).*time-first",
    ):
        _coupler(component).initial_state()


@pytest.mark.fast_always
@pytest.mark.parametrize(("mode", "records"), [("linear", 12), ("daily", 365)])
@pytest.mark.parametrize("levels", [False, True])
def test_setup_can_seed_time_selected_outputs_with_static_declared_masks(
    mode: str, records: int, levels: bool
) -> None:
    shape = (records, 3, 2, 2) if levels else (records, 2, 2)
    component = CallableComponent(
        "FORCING",
        make_test_grid(),
        lambda fields: {},
        spec=ComponentSpec(
            outputs=("temperature", "wet", "fraction"),
            lifecycle=LifecycleHooks(
                setup=lambda component, context: SetupResult(
                    fields={
                        "temperature": np.full(shape, 280.0),
                        "wet": 1.0,
                        "fraction": 0.5,
                    }
                )
            ),
            transfer=TransferPolicy(mode),  # type: ignore[arg-type]
            masks=ComponentMaskSpec(binary_field="wet", fractional_field="fraction"),
        ),
    )

    state = _coupler(component).run()

    assert state.component("FORCING").field("temperature").shape == shape
    np.testing.assert_array_equal(state.component("FORCING").field("fraction"), 0.5)


@pytest.mark.fast_always
@pytest.mark.parametrize("mode", ["linear", "daily"])
@pytest.mark.parametrize("phase", ["setup", "prefill"])
def test_lifecycle_cannot_install_invalid_time_selected_output(
    mode: str, phase: str
) -> None:
    fields = {"temperature": np.ones((1, 2, 2))}
    hooks = (
        LifecycleHooks(setup=lambda component, context: SetupResult(fields=fields))
        if phase == "setup"
        else LifecycleHooks(
            prefill=lambda component, context: PrefillResult(fields=fields)
        )
    )
    records = 12 if mode == "linear" else 365
    component = DataComponent(
        "FORCING",
        make_test_grid(),
        {"temperature": np.ones((records, 2, 2))},
        spec=ComponentSpec(
            lifecycle=hooks,
            transfer=TransferPolicy(mode),  # type: ignore[arg-type]
        ),
    )

    with pytest.raises(ComponentError, match=rf"FORCING.*temperature.*{mode}"):
        _coupler(component).initial_state()


@pytest.mark.fast_always
@pytest.mark.parametrize("mode", ["linear", "daily"])
def test_time_selector_rejects_incomplete_climatology(mode: str) -> None:
    clock = Clock(datetime(2001, 2, 1), 86400.0, 1)

    with pytest.raises(ComponentError, match=rf"temperature.*{mode}.*time-first"):
        select_runtime_field(
            np.ones((1, 2, 2)),
            TransferPolicy(mode),  # type: ignore[arg-type]
            initial_runtime_step_info(clock),
            field_name="temperature",
        )


@pytest.mark.fast_always
@pytest.mark.parametrize("field_dtype", [jnp.float32, jnp.float64])
def test_linear_time_selector_preserves_field_precision_and_derivatives(
    field_dtype: Any,
) -> None:
    assert jax.config.read("jax_enable_x64")
    forcing = jnp.ones((12, 2, 2), dtype=field_dtype)
    step_info = initial_runtime_step_info(Clock(datetime(2001, 2, 1), 86400.0, 1))

    def select(values: Any) -> Any:
        return select_runtime_field(values, TransferPolicy("linear"), step_info)

    selected, tangent = jax.jvp(jax.jit(select), (forcing,), (jnp.ones_like(forcing),))
    _, pullback = jax.vjp(select, forcing)

    assert selected.dtype == field_dtype
    assert tangent.dtype == field_dtype
    np.testing.assert_allclose(selected, 1.0)
    np.testing.assert_allclose(tangent, 1.0)
    np.testing.assert_allclose(
        jnp.sum(pullback(jnp.ones_like(selected))[0], axis=0), 1.0
    )


@pytest.mark.parametrize("shape", [(2, 2), (3, 2, 2), (4, 3, 2, 2)])
def test_current_outputs_preserve_static_and_level_layouts(
    shape: tuple[int, ...],
) -> None:
    component = DataComponent(
        "FORCING", make_test_grid(), {"temperature": np.full(shape, 280.0)}
    )

    assert (
        _coupler(component).run().component("FORCING").field("temperature").shape
        == shape
    )
