from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
import importlib
import importlib.util
from pathlib import Path
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import xarray as xr

import vercor.components as components_module
import vercor.components.base as base_module
from vercor.components._contracts import merge_component_outputs
from vercor.components._lifecycle import ComponentLifecycleHooks
from tests._coverage_support import DummyComponent, make_test_grid
from tests.assertions import assert_allclose_compact
from vercor.forcing_data import read_forcing
from vercor.setups.data.era5_atmosphere import make_era5_atmosphere
from vercor.clock import Clock
from vercor.runtime.contexts import ComponentInitContext, RuntimeStepContext
from vercor.coupler import Coupler
from vercor.exceptions import ComponentError, CouplerError
from vercor.forcing_data import ComponentForcingData
from vercor.output import write_runtime_component_view_to_netcdf
from vercor.run_sequence import RunSequence
from vercor.runtime import (
    RuntimeComponentContract,
    RuntimeComponentState,
    RuntimeFieldStore,
)
from vercor.runtime.component_state import create_runtime_component_state
from vercor.runtime.field_transfer import (
    receive_runtime_fields,
    send_runtime_fields,
)
from vercor.runtime.validation import (
    check_not_empty_import_export_lists,
    check_valid_exchange_field_names,
    validate_component_runtime_contract_fields,
)
from vercor.runtime.time import scalar_runtime_step_info
from vercor.runtime.views import RuntimeComponentView
from vercor.settings import VercorSettings
from vercor.types import RuntimeArray


class _RuntimeOnlyComponent(base_module.Component):
    def step_runtime_state(
        self,
        component_state: RuntimeComponentState,
        context: RuntimeStepContext,
    ) -> RuntimeComponentState:
        data = component_state.data.set(
            "temperature",
            component_state.data.get("temperature") + context.dt_seconds,
        )
        return component_state.with_data(data)


class _MissingSetupComponent(base_module.Component):
    def __init__(self) -> None:
        pass

    def step_runtime_state(
        self,
        component_state: RuntimeComponentState,
        context: RuntimeStepContext,
    ) -> RuntimeComponentState:
        _ = context
        return component_state


class _HostStepOnlyComponent(base_module.HostRuntimeComponent):
    def step_host_runtime_state(
        self,
        component_state: RuntimeComponentState,
        context: RuntimeStepContext,
    ) -> RuntimeComponentState:
        _ = context
        return component_state


def test_component_runtime_execution_policy_helpers_detect_host_components() -> None:
    assert importlib.util.find_spec("vercor.components._runtime_execution") is not None
    runtime_execution = cast(
        Any,
        importlib.import_module("vercor.components._runtime_execution"),
    )
    pure_component = _RuntimeOnlyComponent(name="ATM", grid=make_test_grid())
    host_component = _HostStepOnlyComponent(name="OCN", grid=make_test_grid())

    assert runtime_execution.component_requires_host_runtime(pure_component) is False
    assert runtime_execution.component_requires_host_runtime(host_component) is True
    assert runtime_execution.host_component_names(
        {"ATM": pure_component, "OCN": host_component}
    ) == ["OCN"]


def test_component_runtime_execution_policy_steps_selected_runtime_path() -> None:
    assert importlib.util.find_spec("vercor.components._runtime_execution") is not None
    runtime_execution = cast(
        Any,
        importlib.import_module("vercor.components._runtime_execution"),
    )

    class PureMarkerComponent(base_module.Component):
        def step_runtime_state(
            self,
            component_state: RuntimeComponentState,
            context: RuntimeStepContext,
        ) -> RuntimeComponentState:
            _ = context
            return component_state.with_data(
                component_state.data.set(
                    "marker",
                    component_state.data.get("marker") + 1.0,
                )
            )

    class HostMarkerComponent(base_module.HostRuntimeComponent):
        def step_host_runtime_state(
            self,
            component_state: RuntimeComponentState,
            context: RuntimeStepContext,
        ) -> RuntimeComponentState:
            _ = context
            return component_state.with_data(
                component_state.data.set(
                    "marker",
                    component_state.data.get("marker") + 2.0,
                )
            )

    grid = make_test_grid()
    context = RuntimeStepContext(dt_seconds=1.0, settings=VercorSettings())
    state = RuntimeComponentState(
        data=RuntimeFieldStore.from_mapping({"marker": jnp.asarray(0.0)}),
        incoming=RuntimeFieldStore.empty(),
        outgoing=RuntimeFieldStore.empty(),
    )

    pure_state = runtime_execution.step_component_runtime_state(
        PureMarkerComponent(name="ATM", grid=grid),
        state,
        context,
        allow_host_runtime=False,
    )
    host_state = runtime_execution.step_component_runtime_state(
        HostMarkerComponent(name="OCN", grid=grid),
        state,
        context,
        allow_host_runtime=True,
    )

    assert_allclose_compact(pure_state.data.get("marker"), np.asarray(1.0))
    assert_allclose_compact(host_state.data.get("marker"), np.asarray(2.0))
    with pytest.raises(ComponentError, match="host-backed"):
        runtime_execution.step_component_runtime_state(
            HostMarkerComponent(name="LND", grid=grid),
            state,
            context,
            allow_host_runtime=False,
        )


@pytest.mark.fast_always
def test_active_component_requires_explicit_runtime_step() -> None:
    class MissingRuntimeStep(base_module.Component):
        pass

    with pytest.raises(TypeError, match="step_runtime_state"):
        MissingRuntimeStep(name="ATM", grid=make_test_grid())  # type: ignore[abstract]


@pytest.mark.fast_always
def test_host_runtime_component_requires_explicit_host_step() -> None:
    class MissingHostStep(base_module.HostRuntimeComponent):
        pass

    with pytest.raises(TypeError, match="step_host_runtime_state"):
        MissingHostStep(name="ATM", grid=make_test_grid())  # type: ignore[abstract]


@pytest.mark.fast_always
def test_data_component_uses_explicit_noop_runtime_step() -> None:
    class StaticForcingComponent(base_module.DataComponent):
        pass

    grid = make_test_grid(name="data")
    component = StaticForcingComponent(name="OCN", grid=grid)
    component.data["sea_surface_temperature"] = jnp.full(grid.shape, 280.0)
    contract = RuntimeComponentContract(exports=("sea_surface_temperature",))
    state = create_runtime_component_state(
        component,
        prefill_missing=True,
        contract=contract,
    )

    stepped = component.step_runtime_state(
        state,
        RuntimeStepContext(dt_seconds=60.0, settings=VercorSettings()),
    )

    assert stepped is state
    sent = send_runtime_fields(component, stepped, contract=contract)
    assert_allclose_compact(
        sent.outgoing.get("sea_surface_temperature"),
        np.full(grid.shape, 280.0),
    )


@pytest.mark.fast_always
def test_data_component_seeds_canonical_fields() -> None:
    grid = make_test_grid(name="factory-data")
    component = components_module.data_component(
        name="OBS",
        grid=grid,
        fields={"temperature": jnp.full(grid.shape, 281.0)},
    )

    assert isinstance(component, base_module.DataComponent)
    assert_allclose_compact(
        component.data["temperature"],
        np.full(grid.shape, 281.0),
    )


@pytest.mark.fast_always
def test_convenience_factories_delegate_to_authoring_facade() -> None:
    grid = make_test_grid(name="author-factories")

    data_component = components_module.data_component(
        name="OBS",
        grid=grid,
        fields={"temperature": 281.0},
    )
    assert isinstance(data_component, base_module.DataComponent)
    assert_allclose_compact(
        data_component.data["temperature"],
        np.full(grid.shape, 281.0),
    )

    def step(
        fields: Mapping[str, RuntimeArray],
        context: base_module.ComponentStepContext,
        payload: Any | None,
    ) -> Mapping[str, RuntimeArray]:
        _ = payload
        return {
            "temperature": fields["temperature"] + fields["forcing"],
            "tendency": fields["tendency"] + context.dt_seconds,
        }

    differentiable = components_module.differentiable_component(
        name="ATM",
        grid=grid,
        step=step,
        inputs=("forcing",),
        outputs=("temperature", "tendency"),
        default_fields={"temperature": 280.0, "forcing": 2.0},
    )
    assert isinstance(differentiable, base_module.Component)
    assert differentiable.field_spec.inputs == ("forcing",)
    assert differentiable.field_spec.outputs == ("temperature", "tendency")

    state = create_runtime_component_state(
        differentiable,
        prefill_missing=True,
        contract=RuntimeComponentContract(),
    )
    stepped = differentiable.step_runtime_state(
        state,
        RuntimeStepContext(dt_seconds=3.0, settings=VercorSettings()),
    )
    assert_allclose_compact(
        stepped.data.get("temperature"),
        np.full(grid.shape, 282.0),
    )
    assert_allclose_compact(
        stepped.data.get("tendency"),
        np.full(grid.shape, 3.0),
    )

    host = components_module.host_component(
        name="HOST",
        grid=grid,
        step=step,
        outputs=("temperature",),
        default_fields={"temperature": 1.0, "forcing": 4.0, "tendency": 0.0},
    )
    assert isinstance(host, base_module.HostRuntimeComponent)
    host_state = create_runtime_component_state(
        host,
        prefill_missing=True,
        contract=RuntimeComponentContract(),
    )
    host_stepped = host.step_host_runtime_state(
        host_state,
        RuntimeStepContext(dt_seconds=5.0, settings=VercorSettings()),
    )
    assert_allclose_compact(
        host_stepped.data.get("temperature"),
        np.full(grid.shape, 5.0),
    )


@pytest.mark.fast_always
def test_legacy_wrapper_entrypoints_are_removed() -> None:
    assert not hasattr(base_module.Component, "wrap")
    assert not hasattr(base_module.DataComponent, "wrap")
    assert not hasattr(base_module.HostRuntimeComponent, "wrap")
    assert not hasattr(base_module, "make_data_component")
    assert not hasattr(base_module, "make_differentiable_component")
    assert not hasattr(base_module, "make_host_component")


@pytest.mark.fast_always
def test_from_fields_and_from_model_facade_expand_scalar_defaults() -> None:
    grid = make_test_grid(name="facade")

    data_component = base_module.DataComponent.from_fields(
        name="OBS",
        grid=grid,
        fields={"temperature": 281.0},
    )
    assert isinstance(data_component, base_module.DataComponent)
    assert_allclose_compact(
        data_component.data["temperature"],
        np.full(grid.shape, 281.0),
    )

    def step(
        fields: Mapping[str, RuntimeArray],
        context: RuntimeStepContext,
        payload: Any | None,
    ) -> Mapping[str, RuntimeArray]:
        _ = payload
        return {
            "temperature": fields["temperature"] + fields["forcing"],
            "tendency": fields["tendency"] + context.dt_seconds,
        }

    component = base_module.Component.from_model(
        name="ATM",
        grid=grid,
        step=step,
        outputs=("temperature", "tendency"),
        default_fields={"temperature": 280.0, "forcing": 2.0},
    )
    state = create_runtime_component_state(
        component,
        prefill_missing=True,
        contract=RuntimeComponentContract(),
    )

    component.validate_runtime_state(state, RuntimeComponentContract())
    assert_allclose_compact(state.data.get("temperature"), np.full(grid.shape, 280.0))
    assert_allclose_compact(state.data.get("forcing"), np.full(grid.shape, 2.0))
    assert_allclose_compact(state.data.get("tendency"), np.zeros(grid.shape))

    stepped = component.step_runtime_state(
        state,
        RuntimeStepContext(dt_seconds=3.0, settings=VercorSettings()),
    )
    assert_allclose_compact(
        stepped.data.get("temperature"),
        np.full(grid.shape, 282.0),
    )
    assert_allclose_compact(
        stepped.data.get("tendency"),
        np.full(grid.shape, 3.0),
    )


@pytest.mark.fast_always
def test_data_component_from_fields_normalizes_author_fields_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    grid = make_test_grid(name="facade-normalize-once")
    real_normalize = base_module._normalize_author_field_values
    call_count = 0

    def counting_normalize(*args: Any, **kwargs: Any) -> dict[str, RuntimeArray] | None:
        nonlocal call_count
        call_count += 1
        return real_normalize(*args, **kwargs)

    monkeypatch.setattr(
        base_module,
        "_normalize_author_field_values",
        counting_normalize,
    )

    component = base_module.DataComponent.from_fields(
        name="OBS",
        grid=grid,
        fields={"temperature": 281.0},
    )

    assert call_count == 1
    assert component.field_spec.outputs == ("temperature",)
    assert_allclose_compact(
        component.data["temperature"],
        np.full(grid.shape, 281.0),
    )


@pytest.mark.fast_always
def test_callable_facade_accepts_one_two_and_three_argument_steps() -> None:
    grid = make_test_grid(name="flex-step")

    def fields_only(fields: Mapping[str, RuntimeArray]) -> Mapping[str, RuntimeArray]:
        return {"temperature": fields["temperature"] + 1.0}

    def fields_and_context(
        fields: Mapping[str, RuntimeArray],
        context: RuntimeStepContext,
    ) -> Mapping[str, RuntimeArray]:
        return {"temperature": fields["temperature"] + context.dt_seconds}

    def fields_context_payload(
        fields: Mapping[str, RuntimeArray],
        context: RuntimeStepContext,
        payload: Any | None,
    ) -> Mapping[str, RuntimeArray]:
        assert isinstance(payload, Mapping)
        return {
            "temperature": (
                fields["temperature"] + context.dt_seconds + payload["offset"]
            )
        }

    components = (
        components_module.differentiable_component(
            name="ONE",
            grid=grid,
            step=fields_only,
            outputs=("temperature",),
            default_fields={"temperature": 280.0},
        ),
        components_module.differentiable_component(
            name="TWO",
            grid=grid,
            step=fields_and_context,
            outputs=("temperature",),
            default_fields={"temperature": 280.0},
        ),
        components_module.differentiable_component(
            name="THREE",
            grid=grid,
            step=fields_context_payload,
            payload={"offset": 3.0},
            outputs=("temperature",),
            default_fields={"temperature": 280.0},
        ),
    )

    for component, expected_temperature in zip(
        components,
        (281.0, 282.0, 285.0),
        strict=True,
    ):
        state = create_runtime_component_state(
            component,
            prefill_missing=True,
            contract=RuntimeComponentContract(),
        )
        stepped = component.step_runtime_state(
            state,
            RuntimeStepContext(dt_seconds=2.0, settings=VercorSettings()),
        )
        assert_allclose_compact(
            stepped.data.get("temperature"),
            np.full(grid.shape, expected_temperature),
        )


@pytest.mark.fast_always
def test_callable_facade_rejects_unsupported_step_signature() -> None:
    grid = make_test_grid(name="bad-step")

    def too_many_arguments(
        fields: Mapping[str, RuntimeArray],
        context: RuntimeStepContext,
        payload: Any | None,
        extra: object,
    ) -> Mapping[str, RuntimeArray]:
        _ = fields, context, payload, extra
        return {}

    with pytest.raises(
        ComponentError,
        match="step callable.*1, 2, or 3 positional arguments",
    ):
        components_module.differentiable_component(
            name="ATM",
            grid=grid,
            step=too_many_arguments,
        )


@pytest.mark.fast_always
def test_callable_facade_rejects_removed_legacy_field_seed_keyword() -> None:
    grid = make_test_grid(name="removed-legacy-field-seed")

    def step(fields: Mapping[str, RuntimeArray]) -> Mapping[str, RuntimeArray]:
        return {"temperature": fields["temperature"]}

    removed_keyword = "initial" + "_fields"
    with pytest.raises(TypeError, match=removed_keyword):
        cast(Any, components_module.differentiable_component)(
            name="ATM",
            grid=grid,
            step=step,
            **{removed_keyword: {"temperature": 280.0}},
            outputs=("temperature",),
        )


@pytest.mark.fast_always
def test_seed_declared_defaults_and_field_names_expose_author_state() -> None:
    grid = make_test_grid(name="declared-defaults")
    component = _RuntimeOnlyComponent(name="ATM", grid=grid)
    component.declare_fields(
        outputs=("temperature", "humidity"),
        default_fields={"temperature": 280.0, "humidity": 0.5},
    )

    returned = component.seed_declared_defaults()

    assert returned is component
    assert component.field_names == ("temperature", "humidity")
    assert_allclose_compact(component.data["temperature"], np.full(grid.shape, 280.0))
    assert_allclose_compact(component.data["humidity"], np.full(grid.shape, 0.5))


@pytest.mark.fast_always
def test_base_initialize_seeds_declared_defaults() -> None:
    grid = make_test_grid(name="base-initialize")
    component = _RuntimeOnlyComponent(name="ATM", grid=grid)
    component.declare_fields(
        outputs=("temperature", "humidity"),
        default_fields={"temperature": 280.0, "humidity": 0.5},
    )

    component.initialize(
        ComponentInitContext(
            start=datetime(2000, 1, 1),
            dt_seconds=60.0,
            logger=cast(Any, None),
            settings=VercorSettings(),
            run_sequence=RunSequence(order=["ATM"]),
        )
    )

    assert component.field_names == ("temperature", "humidity")
    assert_allclose_compact(component.data["temperature"], np.full(grid.shape, 280.0))
    assert_allclose_compact(component.data["humidity"], np.full(grid.shape, 0.5))


@pytest.mark.fast_always
def test_update_settings_is_chainable() -> None:
    component = _RuntimeOnlyComponent(name="ATM", grid=make_test_grid())

    returned = component.update_settings(
        apply_time_interpolation=True,
        get_field_time_slice=True,
    )

    assert returned is component
    assert component.settings.apply_time_interpolation
    assert component.settings.get_field_time_slice


@pytest.mark.fast_always
def test_grid_field_defaults_expands_default_value_and_overrides() -> None:
    grid = make_test_grid(name="grid-defaults")
    component = _RuntimeOnlyComponent(name="ATM", grid=grid)

    defaults = component.grid_field_defaults(
        ("temperature", "humidity", "pressure"),
        value=0.0,
        overrides={"temperature": 280.0, "humidity": np.full(grid.shape, 0.5)},
    )

    assert tuple(defaults) == ("temperature", "humidity", "pressure")
    assert_allclose_compact(defaults["temperature"], np.full(grid.shape, 280.0))
    assert_allclose_compact(defaults["humidity"], np.full(grid.shape, 0.5))
    assert_allclose_compact(defaults["pressure"], np.zeros(grid.shape))


@pytest.mark.fast_always
def test_apply_step_result_updates_fields_and_payload() -> None:
    grid = make_test_grid(name="apply-step-result")
    component = _RuntimeOnlyComponent(name="ATM", grid=grid)
    component.seed_field("temperature", 280.0)
    state = create_runtime_component_state(
        component,
        contract=RuntimeComponentContract(),
    )

    updated = component.apply_step_result(
        state,
        base_module.ComponentStepResult(
            fields={"temperature": jnp.full(grid.shape, 281.0)},
            payload={"counter": 1},
        ),
    )

    assert_allclose_compact(updated.data.get("temperature"), np.full(grid.shape, 281.0))
    assert updated.runtime_payload == {"counter": 1}


@pytest.mark.fast_always
def test_data_component_seeding_updates_declared_outputs() -> None:
    grid = make_test_grid(name="data-outputs")
    component = components_module.data_component(
        name="OBS",
        grid=grid,
        fields={"temperature": 281.0},
    )

    component.seed_field("humidity", 0.5)
    component.seed_fields({"pressure": 101325.0})

    assert component.field_spec.outputs == ("temperature", "humidity", "pressure")
    assert component.field_names == ("temperature", "humidity", "pressure")


@pytest.mark.fast_always
def test_merge_component_outputs_is_pure_and_preserves_contract_details() -> None:
    field_spec = base_module.ComponentFieldSpec(
        inputs=("forcing",),
        outputs=("temperature",),
        default_fields={"temperature": 280.0},
    )

    merged = merge_component_outputs(field_spec, ("humidity", "temperature"))

    assert merged is not field_spec
    assert field_spec.outputs == ("temperature",)
    assert merged.inputs == ("forcing",)
    assert merged.outputs == ("temperature", "humidity")
    assert merged.default_fields == {"temperature": 280.0}


@pytest.mark.fast_always
def test_data_component_seeding_preserves_inputs_and_defaults() -> None:
    grid = make_test_grid(name="data-contract-preserve")
    component = base_module.DataComponent(name="DATA", grid=grid)
    component.declare_fields(
        inputs=("forcing",),
        default_fields={"temperature": 280.0},
    )

    component.seed_fields({"humidity": 0.5})

    assert component.field_spec.inputs == ("forcing",)
    assert component.field_spec.outputs == ("humidity",)
    assert "temperature" in component.field_spec.default_fields


@pytest.mark.fast_always
def test_factory_lifecycle_hooks_are_stored_in_single_private_container() -> None:
    grid = make_test_grid(name="lifecycle-container")
    events: list[str] = []

    def step(fields: Mapping[str, RuntimeArray]) -> Mapping[str, RuntimeArray]:
        return {"temperature": fields["temperature"] + 1.0}

    def initialize(component: Any, context: ComponentInitContext) -> None:
        _ = context
        events.append(f"initialize:{component.name}")

    def create_runtime_payload(component: Any) -> dict[str, int]:
        events.append(f"payload:{component.name}")
        return {"counter": 1}

    def prefill(
        component: Any,
        data: dict[str, RuntimeArray],
        incoming: dict[str, RuntimeArray],
        outgoing: dict[str, RuntimeArray],
        contract: RuntimeComponentContract,
    ) -> None:
        _ = incoming, outgoing, contract
        events.append(f"prefill:{component.name}")
        component.prefill_runtime_fields(data, outputs=("temperature",))

    def validate(
        component: Any,
        state: RuntimeComponentState,
        contract: RuntimeComponentContract,
    ) -> None:
        _ = contract
        events.append(f"validate:{component.name}")
        component.require_runtime_fields(state, "temperature")

    factories = (
        components_module.data_component(
            name="DATA",
            grid=grid,
            initialize=initialize,
            create_runtime_payload=create_runtime_payload,
            prefill_runtime_state_fields=prefill,
            validate_runtime_state=validate,
        ),
        components_module.differentiable_component(
            name="ATM",
            grid=grid,
            step=step,
            initialize=initialize,
            create_runtime_payload=create_runtime_payload,
            prefill_runtime_state_fields=prefill,
            validate_runtime_state=validate,
        ),
        components_module.host_component(
            name="HOST",
            grid=grid,
            step=step,
            initialize=initialize,
            create_runtime_payload=create_runtime_payload,
            prefill_runtime_state_fields=prefill,
            validate_runtime_state=validate,
        ),
    )

    for component in factories:
        assert isinstance(component._lifecycle_hooks, ComponentLifecycleHooks)
        assert not hasattr(component, "_initialize_hook")
        assert not hasattr(component, "_create_runtime_payload_hook")
        component.initialize(
            ComponentInitContext(
                start=datetime(2000, 1, 1),
                dt_seconds=60.0,
                logger=cast(Any, None),
                settings=VercorSettings(),
                run_sequence=RunSequence(order=[component.name]),
            )
        )
        state = create_runtime_component_state(
            component,
            prefill_missing=True,
            contract=RuntimeComponentContract(),
        )
        component.validate_runtime_state(state, RuntimeComponentContract())

    assert events == [
        "initialize:DATA",
        "prefill:DATA",
        "payload:DATA",
        "validate:DATA",
        "initialize:ATM",
        "prefill:ATM",
        "payload:ATM",
        "validate:ATM",
        "initialize:HOST",
        "prefill:HOST",
        "payload:HOST",
        "validate:HOST",
    ]


@pytest.mark.fast_always
def test_seed_helpers_accept_scalar_author_values_and_expose_field_spec() -> None:
    grid = make_test_grid(name="scalar-seed")
    component = _RuntimeOnlyComponent(name="ATM", grid=grid)

    returned_spec = component.declare_fields(
        inputs=("forcing",),
        outputs=("temperature",),
        default_fields={"pressure": 101325.0},
    )
    assert component.field_spec == returned_spec
    assert component.field_spec.inputs == ("forcing",)
    assert component.field_spec.outputs == ("temperature",)
    assert not hasattr(component.field_spec, "required_fields")
    assert "pressure" in component.field_spec.default_fields
    with pytest.raises(AttributeError):
        component.field_spec = base_module.ComponentFieldSpec()  # type: ignore[misc]

    component.seed_field("temperature", 280.0)
    component.seed_fields({"humidity": 0.5, "forcing": jnp.ones(grid.shape)})
    state = create_runtime_component_state(
        component,
        prefill_missing=True,
        contract=RuntimeComponentContract(),
    )

    assert_allclose_compact(state.data.get("temperature"), np.full(grid.shape, 280.0))
    assert_allclose_compact(state.data.get("humidity"), np.full(grid.shape, 0.5))
    assert_allclose_compact(state.data.get("forcing"), np.ones(grid.shape))
    assert_allclose_compact(state.data.get("pressure"), np.full(grid.shape, 101325.0))


@pytest.mark.fast_always
def test_seeded_component_arrays_follow_float32_policy_with_global_x64_enabled() -> (
    None
):
    grid = make_test_grid(name="seeded-policy")
    component = base_module.DataComponent.from_fields(
        name="DATA",
        grid=grid,
        fields={
            "temperature": jnp.asarray([[280.0, 281.0], [282.0, 283.0]]),
        },
        settings=VercorSettings(enable_x64=False),
    )

    assert component.data["temperature"].dtype == jnp.float32


@pytest.mark.fast_always
def test_required_fields_declaration_api_is_removed() -> None:
    grid = make_test_grid(name="removed-required-fields")

    def step(fields: Mapping[str, RuntimeArray]) -> Mapping[str, RuntimeArray]:
        return {"temperature": fields["temperature"]}

    rejected_callables: tuple[tuple[Any, dict[str, Any]], ...] = (
        (base_module.ComponentFieldSpec, {}),
        (
            base_module.Component.from_model,
            {"name": "ATM", "grid": grid, "step": step},
        ),
        (
            base_module.HostRuntimeComponent.from_model,
            {"name": "HOST", "grid": grid, "step": step},
        ),
        (
            components_module.differentiable_component,
            {"name": "ATM", "grid": grid, "step": step},
        ),
        (
            components_module.host_component,
            {"name": "HOST", "grid": grid, "step": step},
        ),
    )
    for callable_factory, kwargs in rejected_callables:
        with pytest.raises(TypeError, match="required_fields"):
            cast(Any, callable_factory)(**kwargs, required_fields=("humidity",))

    component = _RuntimeOnlyComponent(name="ATM", grid=grid)
    with pytest.raises(TypeError, match="required_fields"):
        cast(Any, component.declare_fields)(required_fields=("humidity",))


@pytest.mark.fast_always
def test_from_model_inputs_validate_missing_fields_without_zero_prefill() -> None:
    grid = make_test_grid(name="facade-inputs")

    def step(
        fields: Mapping[str, RuntimeArray],
        context: RuntimeStepContext,
        payload: Any | None,
    ) -> Mapping[str, RuntimeArray]:
        _ = context, payload
        return {"temperature": fields["temperature"] + fields["forcing"]}

    component = base_module.Component.from_model(
        name="ATM",
        grid=grid,
        step=step,
        inputs=("forcing",),
        outputs=("temperature",),
        default_fields={"temperature": 280.0},
    )
    state = create_runtime_component_state(
        component,
        prefill_missing=True,
        contract=RuntimeComponentContract(),
    )

    with pytest.raises(
        CouplerError,
        match="Runtime missing required data field 'forcing' for component 'ATM'",
    ):
        component.validate_runtime_state(state, RuntimeComponentContract())


@pytest.mark.fast_always
def test_host_runtime_component_from_model_uses_author_friendly_names() -> None:
    grid = make_test_grid(name="facade-host")

    def step(
        fields: Mapping[str, RuntimeArray],
        context: RuntimeStepContext,
        payload: Any | None,
    ) -> Mapping[str, RuntimeArray]:
        _ = payload
        return {"temperature": fields["temperature"] + context.dt_seconds}

    component = base_module.HostRuntimeComponent.from_model(
        name="HOST",
        grid=grid,
        step=step,
        outputs=("temperature",),
        default_fields={"temperature": 1.0},
    )
    state = create_runtime_component_state(
        component,
        prefill_missing=True,
        contract=RuntimeComponentContract(),
    )

    stepped = component.step_host_runtime_state(
        state,
        RuntimeStepContext(dt_seconds=5.0, settings=VercorSettings()),
    )
    assert_allclose_compact(
        stepped.data.get("temperature"),
        np.full(grid.shape, 6.0),
    )


@pytest.mark.fast_always
def test_subclasses_can_declare_fields_with_author_spec() -> None:
    grid = make_test_grid(name="declared")

    class DeclaredComponent(base_module.Component):
        def __init__(self, name: str, grid: Any) -> None:
            super().__init__(name, grid)
            self.declare_fields(
                base_module.ComponentFieldSpec(
                    inputs=("forcing",),
                    outputs=("temperature",),
                    default_fields={"temperature": 280.0},
                )
            )

        def step_runtime_state(
            self,
            component_state: RuntimeComponentState,
            context: RuntimeStepContext,
        ) -> RuntimeComponentState:
            _ = context
            return self.with_runtime_fields(
                component_state,
                {
                    "temperature": (
                        self.runtime_field(component_state, "temperature")
                        + self.runtime_field(component_state, "forcing")
                    )
                },
            )

    component = DeclaredComponent(name="ATM", grid=grid)
    missing_input_state = create_runtime_component_state(
        component,
        prefill_missing=True,
        contract=RuntimeComponentContract(),
    )
    with pytest.raises(
        CouplerError,
        match="Runtime missing required data field 'forcing' for component 'ATM'",
    ):
        component.validate_runtime_state(
            missing_input_state, RuntimeComponentContract()
        )

    contract = RuntimeComponentContract(imports=("forcing",))
    state = create_runtime_component_state(
        component,
        prefill_missing=True,
        contract=contract,
    )
    component.validate_runtime_state(state, contract)
    assert_allclose_compact(state.data.get("temperature"), np.full(grid.shape, 280.0))
    assert_allclose_compact(state.data.get("forcing"), np.zeros(grid.shape))


@pytest.mark.fast_always
def test_runtime_field_optional_helpers_return_defaults() -> None:
    grid = make_test_grid(name="field-defaults")
    component = _RuntimeOnlyComponent(name="ATM", grid=grid)
    component.seed_field("temperature", jnp.full(grid.shape, 280.0))
    state = create_runtime_component_state(
        component,
        contract=RuntimeComponentContract(),
    )

    assert component.has_runtime_field(state, "temperature")
    assert not component.has_runtime_field(state, "missing")
    assert_allclose_compact(
        component.runtime_field_or(state, "temperature", 1.0),
        np.full(grid.shape, 280.0),
    )
    assert_allclose_compact(
        component.runtime_field_or(state, "missing", 2.0),
        np.full(grid.shape, 2.0),
    )
    assert_allclose_compact(
        component.runtime_field_or_zeros_like(state, "missing", "temperature"),
        np.zeros(grid.shape),
    )


@pytest.mark.fast_always
def test_callable_component_prefills_and_validates_declared_fields() -> None:
    grid = make_test_grid(name="facade-prefill")

    def step(
        fields: Mapping[str, RuntimeArray],
        context: RuntimeStepContext,
        payload: Any | None,
    ) -> Mapping[str, RuntimeArray]:
        _ = payload
        return {
            "temperature": fields["temperature"] + fields["wind"] + context.dt_seconds,
            "wind": fields["wind"],
        }

    component = base_module.Component.from_model(
        name="ATM",
        grid=grid,
        step=step,
        outputs=("temperature", "wind"),
        default_fields={"temperature": jnp.full(grid.shape, 280.0)},
    )
    assert not hasattr(component, "_required_fields")
    assert not hasattr(component, "_prefill_fields")
    assert not hasattr(component, "_field_defaults")
    state = create_runtime_component_state(
        component,
        prefill_missing=True,
        contract=RuntimeComponentContract(),
    )

    component.validate_runtime_state(state, RuntimeComponentContract())
    assert_allclose_compact(state.data.get("temperature"), np.full(grid.shape, 280.0))
    assert_allclose_compact(state.data.get("wind"), np.zeros(grid.shape))

    stepped = component.step_runtime_state(
        state,
        RuntimeStepContext(dt_seconds=2.0, settings=VercorSettings()),
    )
    assert_allclose_compact(
        stepped.data.get("temperature"),
        np.full(grid.shape, 282.0),
    )


@pytest.mark.fast_always
def test_callable_component_reports_missing_declared_inputs() -> None:
    grid = make_test_grid(name="facade-required")

    def step(
        fields: Mapping[str, RuntimeArray],
        context: RuntimeStepContext,
        payload: Any | None,
    ) -> Mapping[str, RuntimeArray]:
        _ = context, payload
        return {"temperature": fields["temperature"]}

    component = base_module.Component.from_model(
        name="ATM",
        grid=grid,
        step=step,
        inputs=("temperature",),
    )
    state = create_runtime_component_state(
        component,
        prefill_missing=True,
        contract=RuntimeComponentContract(),
    )

    with pytest.raises(
        CouplerError,
        match="Runtime missing required data field 'temperature' for component 'ATM'",
    ):
        component.validate_runtime_state(state, RuntimeComponentContract())


@pytest.mark.fast_always
def test_component_seed_default_helpers_and_required_field_validator() -> None:
    grid = make_test_grid(name="seed-defaults")
    component = _RuntimeOnlyComponent(name="ATM", grid=grid)

    component.seed_zero_field("temperature")
    component.seed_zero_fields(("u_velocity", "v_velocity"))
    component.seed_constant_field("humidity", 0.5)
    state = create_runtime_component_state(
        component,
        contract=RuntimeComponentContract(),
    )

    component.require_runtime_fields(
        state,
        "temperature",
        "u_velocity",
        "v_velocity",
        "humidity",
    )
    assert_allclose_compact(state.data.get("temperature"), np.zeros(grid.shape))
    assert_allclose_compact(state.data.get("u_velocity"), np.zeros(grid.shape))
    assert_allclose_compact(state.data.get("v_velocity"), np.zeros(grid.shape))
    assert_allclose_compact(state.data.get("humidity"), np.full(grid.shape, 0.5))

    with pytest.raises(
        CouplerError,
        match="Runtime missing required data field 'missing' for component 'ATM'",
    ):
        component.require_runtime_fields(state, "missing")


@pytest.mark.fast_always
def test_required_field_validator_accepts_time_dependent_canonical_data() -> None:
    grid = make_test_grid(name="time-dependent-required")
    component = _RuntimeOnlyComponent(name="OCN", grid=grid)
    monthly_sst = jnp.zeros((12, *grid.shape), dtype=jnp.float64)
    state = RuntimeComponentState(
        data=RuntimeFieldStore.from_mapping({"sea_surface_temperature": monthly_sst}),
        incoming=RuntimeFieldStore.empty(),
        outgoing=RuntimeFieldStore.empty(),
    )

    component.require_runtime_fields(state, "sea_surface_temperature")

    bad_state = RuntimeComponentState(
        data=RuntimeFieldStore.from_mapping(
            {"bad_metadata": jnp.zeros((3,), dtype=jnp.float64)}
        ),
        incoming=RuntimeFieldStore.empty(),
        outgoing=RuntimeFieldStore.empty(),
    )
    with pytest.raises(
        CouplerError,
        match="bad_metadata.*canonical grid-field layout",
    ):
        component.require_runtime_fields(bad_state, "bad_metadata")


@pytest.mark.fast_always
def test_data_component_rejects_non_grid_fields_early() -> None:
    grid = make_test_grid(name="factory-layout")

    with pytest.raises(
        ComponentError,
        match="data field 'bad_metadata'.*canonical grid-field layout",
    ):
        components_module.data_component(
            name="OBS",
            grid=grid,
            fields={"bad_metadata": jnp.zeros((3,), dtype=jnp.float64)},
        )


@pytest.mark.fast_always
def test_component_helpers_seed_and_update_runtime_fields() -> None:
    grid = make_test_grid(name="helper-fields")
    component = _RuntimeOnlyComponent(name="ATM", grid=grid)
    component.seed_field("temperature", jnp.ones(grid.shape))
    component.seed_fields({"humidity": jnp.full(grid.shape, 0.5)})
    state = create_runtime_component_state(
        component,
        contract=RuntimeComponentContract(),
    )

    fields = component.runtime_fields(state)
    assert set(fields) == {"temperature", "humidity"}
    assert_allclose_compact(
        component.runtime_field(state, "humidity"),
        np.full(grid.shape, 0.5),
    )

    updated = component.with_runtime_fields(
        state,
        {"temperature": jnp.full(grid.shape, 284.0)},
    )

    assert_allclose_compact(
        updated.data.get("temperature"),
        np.full(grid.shape, 284.0),
    )
    assert_allclose_compact(
        updated.data.get("humidity"),
        np.full(grid.shape, 0.5),
    )

    base_source = Path("vercor/components/base.py").read_text(encoding="utf-8")
    runtime_fields_source = Path("vercor/components/_runtime_fields.py").read_text(
        encoding="utf-8"
    )
    assert "component_state.data.to_mapping()" not in base_source
    assert "component_state.data.replace_many(fields)" not in base_source
    assert "validate_runtime_component_data_field" not in base_source
    assert "component_state.data.to_mapping()" in runtime_fields_source
    assert "component_state.data.replace_many(fields)" in runtime_fields_source
    assert "from vercor.runtime.validation import" not in runtime_fields_source


@pytest.mark.fast_always
def test_differentiable_component_applies_callable_field_updates() -> None:
    grid = make_test_grid(name="factory-active")

    def step(
        fields: Mapping[str, RuntimeArray],
        context: RuntimeStepContext,
        payload: Any | None,
    ) -> Mapping[str, RuntimeArray]:
        assert payload is None
        return {"temperature": fields["temperature"] + context.dt_seconds}

    component = components_module.differentiable_component(
        name="ATM",
        grid=grid,
        step=step,
        outputs=("temperature",),
        default_fields={"temperature": jnp.ones(grid.shape)},
    )
    state = create_runtime_component_state(
        component,
        prefill_missing=True,
        contract=RuntimeComponentContract(),
    )

    stepped = component.step_runtime_state(
        state,
        RuntimeStepContext(dt_seconds=3.0, settings=VercorSettings()),
    )

    assert_allclose_compact(
        stepped.data.get("temperature"),
        np.full(grid.shape, 4.0),
    )


@pytest.mark.fast_always
def test_callable_component_preserves_and_replaces_payload() -> None:
    grid = make_test_grid(name="factory-payload")

    def preserve_payload(
        fields: Mapping[str, RuntimeArray],
        context: RuntimeStepContext,
        payload: Any | None,
    ) -> Mapping[str, RuntimeArray]:
        _ = context
        assert isinstance(payload, Mapping)
        return {"temperature": fields["temperature"] + payload["offset"]}

    preserve_component = components_module.differentiable_component(
        name="ATM",
        grid=grid,
        payload={"offset": jnp.asarray(2.0)},
        step=preserve_payload,
        outputs=("temperature",),
        default_fields={"temperature": jnp.ones(grid.shape)},
    )
    preserve_state = create_runtime_component_state(
        preserve_component,
        prefill_missing=True,
        contract=RuntimeComponentContract(),
    )
    preserved = preserve_component.step_runtime_state(
        preserve_state,
        RuntimeStepContext(dt_seconds=1.0, settings=VercorSettings()),
    )

    assert preserved.runtime_payload is preserve_state.runtime_payload
    assert_allclose_compact(
        preserved.data.get("temperature"),
        np.full(grid.shape, 3.0),
    )

    def replace_payload(
        fields: Mapping[str, RuntimeArray],
        context: RuntimeStepContext,
        payload: Any | None,
    ) -> base_module.ComponentStepResult:
        _ = context
        assert isinstance(payload, Mapping)
        return base_module.ComponentStepResult(
            fields={"temperature": fields["temperature"] + 1.0},
            payload={"offset": payload["offset"] + 1.0},
        )

    replace_component = components_module.differentiable_component(
        name="ATM",
        grid=grid,
        payload={"offset": jnp.asarray(2.0)},
        step=replace_payload,
        outputs=("temperature",),
        default_fields={"temperature": jnp.ones(grid.shape)},
    )
    replace_state = create_runtime_component_state(
        replace_component,
        prefill_missing=True,
        contract=RuntimeComponentContract(),
    )
    replaced = replace_component.step_runtime_state(
        replace_state,
        RuntimeStepContext(dt_seconds=1.0, settings=VercorSettings()),
    )

    assert replaced.runtime_payload is not replace_state.runtime_payload
    assert_allclose_compact(
        replaced.data.get("temperature"),
        np.full(grid.shape, 2.0),
    )
    assert isinstance(replaced.runtime_payload, Mapping)
    assert_allclose_compact(replaced.runtime_payload["offset"], np.asarray(3.0))


@pytest.mark.fast_always
def test_host_component_runs_through_coupler_host_runtime() -> None:
    grid = make_test_grid(name="factory-host")

    def step(
        fields: Mapping[str, RuntimeArray],
        context: RuntimeStepContext,
        payload: Any | None,
    ) -> Mapping[str, RuntimeArray]:
        _ = payload
        return {"temperature": fields["temperature"] + context.dt_seconds}

    component = components_module.host_component(
        name="HOST",
        grid=grid,
        step=step,
        outputs=("temperature",),
        default_fields={"temperature": jnp.ones(grid.shape)},
    )
    coupler = Coupler(clock=Clock(start=datetime(2000, 1, 1), dt_seconds=5.0, steps=1))
    coupler.register(component)
    coupler.set_components_run_sequence(RunSequence(order=["HOST"]))

    final_state = coupler.run()

    assert_allclose_compact(
        final_state.get_component_state("HOST").data.get("temperature"),
        np.full(grid.shape, 6.0),
    )


@pytest.mark.fast_always
def test_callable_component_rejects_unseeded_field_updates() -> None:
    grid = make_test_grid(name="factory-missing")

    def step(
        fields: Mapping[str, RuntimeArray],
        context: RuntimeStepContext,
        payload: Any | None,
    ) -> Mapping[str, RuntimeArray]:
        _ = fields, context, payload
        return {"created_during_step": jnp.zeros(grid.shape)}

    component = components_module.differentiable_component(
        name="ATM",
        grid=grid,
        step=step,
        default_fields={"temperature": jnp.ones(grid.shape)},
    )
    state = create_runtime_component_state(
        component,
        prefill_missing=True,
        contract=RuntimeComponentContract(),
    )

    with pytest.raises(
        ComponentError,
        match="created_during_step.*missing from runtime data.*seed_field",
    ):
        component.step_runtime_state(
            state,
            RuntimeStepContext(dt_seconds=1.0, settings=VercorSettings()),
        )


@pytest.mark.fast_always
def test_era5_atmosphere_uses_data_component_runtime_contract() -> None:

    assert callable(make_era5_atmosphere)
    assert issubclass(base_module.DataComponent, base_module.Component)


@pytest.mark.fast_always
def test_component_setup_validation_reports_missing_required_attributes() -> None:
    component = _MissingSetupComponent()
    contract = RuntimeComponentContract(exports=("temperature",))

    with pytest.raises(
        ComponentError,
        match="missing required setup attribute.*name.*grid.*data.*settings",
    ):
        create_runtime_component_state(component, contract=contract)


@pytest.mark.fast_always
def test_component_data_layout_validation_accepts_canonical_grid_fields() -> None:
    grid = make_test_grid(
        name="layout",
        longitude=np.asarray([0.0, 1.0, 2.0]),
        latitude=np.asarray([-1.0, 1.0]),
    )
    component = DummyComponent(name="ATM", grid=grid)
    component.data = {
        "snapshot_2d": jnp.zeros(grid.shape, dtype=jnp.float64),
        "time_surface_3d": jnp.zeros((12, *grid.shape), dtype=jnp.float64),
        "level_snapshot_3d": jnp.zeros((4, *grid.shape), dtype=jnp.float64),
        "time_level_4d": jnp.zeros((12, 4, *grid.shape), dtype=jnp.float64),
    }

    state = create_runtime_component_state(
        component, contract=RuntimeComponentContract()
    )

    assert state.data.get("snapshot_2d").shape == grid.shape
    assert state.data.get("time_surface_3d").shape == (12, *grid.shape)
    assert state.data.get("level_snapshot_3d").shape == (4, *grid.shape)
    assert state.data.get("time_level_4d").shape == (12, 4, *grid.shape)


@pytest.mark.fast_always
def test_component_data_layout_validation_rejects_non_grid_data_fields() -> None:
    grid = make_test_grid(
        name="layout",
        longitude=np.asarray([0.0, 1.0, 2.0]),
        latitude=np.asarray([-1.0, 1.0]),
    )
    component = DummyComponent(name="ATM", grid=grid)
    component.data = {
        "legacy_monthly_temperature": jnp.zeros((3, 2, 12), dtype=jnp.float64),
        "hyai": jnp.zeros((4,), dtype=jnp.float64),
    }

    with pytest.raises(
        ComponentError,
        match=(
            "Component 'ATM' data field 'legacy_monthly_temperature'.*"
            r"shape \(3, 2, 12\).*canonical.*"
            r"\(nTime, nLat, nLon\)"
        ),
    ):
        create_runtime_component_state(component, contract=RuntimeComponentContract())


@pytest.mark.fast_always
def test_host_component_rejects_scanned_runtime_with_clear_error() -> None:
    grid = make_test_grid(name="host")
    component = _HostStepOnlyComponent(name="ATM", grid=grid)
    component.data["temperature"] = jnp.ones(grid.shape)
    coupler = Coupler(
        clock=Clock(start=datetime(2000, 1, 1), dt_seconds=3600.0, steps=1)
    )
    coupler.components = {"ATM": component}
    coupler.run_sequence = RunSequence(order=["ATM"])
    state = coupler.create_runtime_state()

    with pytest.raises(ComponentError, match="host-backed.*Coupler.run"):
        coupler._run_scanned_runtime(state)


@pytest.mark.fast_always
def test_removed_component_api_stays_absent() -> None:
    component = DummyComponent(name="ATM", grid=make_test_grid())

    assert not hasattr(components_module, "Shared")
    assert not hasattr(components_module, "TimedNamedArray")
    assert not hasattr(components_module, "ComponentInitContext")
    assert not hasattr(components_module, "RuntimeStepContext")
    assert not hasattr(base_module, "Shared")
    assert not hasattr(base_module, "TimedNamedArray")
    assert not hasattr(base_module, "write_shared_to_netcdf")
    assert not hasattr(base_module, "write_runtime_component_to_netcdf")
    assert not hasattr(base_module, "write_runtime_component_view_to_netcdf")
    assert not hasattr(base_module, "ComponentForcingData")
    assert not hasattr(components_module, "ComponentForcingData")
    assert not hasattr(components_module, "Atmosphere")
    assert not hasattr(components_module, "Ocean")
    assert not hasattr(components_module, "SeaIce")
    assert not hasattr(components_module, "Land")
    assert not hasattr(components_module, "ERA5Atmosphere")
    assert not hasattr(components_module, "ERA5Ocean")
    assert not hasattr(components_module, "ERAInterimOcean")
    assert not hasattr(components_module, "ERA5Land")
    assert not hasattr(components_module, "JCMLand")
    assert not hasattr(components_module, "JAXGCM")
    assert not hasattr(components_module, "VerosGCM")
    assert not hasattr(components_module, "CAMulatorGCM")
    assert not hasattr(components_module, "CAMulatorLand")
    assert not hasattr(components_module, "write_runtime_component_to_netcdf")
    assert not hasattr(components_module, "write_runtime_component_view_to_netcdf")
    assert not hasattr(component, "incoming_fields")
    assert not hasattr(component, "outgoing_fields")
    assert not hasattr(component, "commit_runtime_state")
    assert not hasattr(component, "merge_incoming_outgoing_fields")
    assert not hasattr(component, "get")
    assert not hasattr(component, "step")
    assert not hasattr(component, "to_runtime_component_state")
    assert not hasattr(component, "receive_runtime_fields")
    assert not hasattr(component, "send_runtime_fields")
    assert not hasattr(component, "check_not_empty_import_export_lists")
    assert not hasattr(component, "check_valid_exchange_field_names")
    assert not hasattr(component, "_validate_runtime_grid_data_field")
    assert not hasattr(component, "_sync_data_from_runtime_state")


def test_runtime_state_creation_receive_and_send() -> None:
    grid = make_test_grid(name="atm")
    component = _RuntimeOnlyComponent(name="ATM", grid=grid)
    contract = RuntimeComponentContract(
        imports=("temperature",),
        exports=("sensible_heat_flux",),
    )
    component.data["sensible_heat_flux"] = jnp.full(grid.shape, 2.0)

    state = create_runtime_component_state(
        component,
        prefill_missing=True,
        contract=contract,
    )
    assert set(state.incoming.field_names) == {"temperature"}
    assert set(state.outgoing.field_names) == {"sensible_heat_flux"}
    assert isinstance(state.incoming.get("temperature"), jax.Array)

    incoming = state.incoming.set("temperature", jnp.full(grid.shape, 5.0))
    state = receive_runtime_fields(
        state.with_incoming(incoming),
        contract,
    )
    assert_allclose_compact(state.data.get("temperature"), np.full(grid.shape, 5.0))

    stepped = component.step_runtime_state(
        state,
        RuntimeStepContext(
            dt_seconds=3.0,
            settings=VercorSettings(),
        ),
    )
    assert_allclose_compact(stepped.data.get("temperature"), np.full(grid.shape, 8.0))

    sent = send_runtime_fields(component, stepped, contract=contract)
    assert_allclose_compact(
        sent.outgoing.get("sensible_heat_flux"),
        np.full(grid.shape, 2.0),
    )


def test_component_validation_and_runtime_receive_delegate() -> None:
    component = DummyComponent(name="ATM", grid=make_test_grid())

    with pytest.raises(ComponentError, match="no fields to import"):
        check_not_empty_import_export_lists(component, RuntimeComponentContract())

    import_only = RuntimeComponentContract(imports=("temperature",))
    with pytest.raises(ComponentError, match="no fields to export"):
        check_not_empty_import_export_lists(component, import_only)

    overlapping = RuntimeComponentContract(
        imports=("temperature",),
        exports=("temperature",),
    )
    with pytest.raises(ComponentError, match="overlapping fields"):
        check_not_empty_import_export_lists(component, overlapping)

    invalid = RuntimeComponentContract(
        imports=("temperature",),
        exports=("not_supported",),
    )
    with pytest.raises(ComponentError, match="not a recognized exchange variable"):
        check_valid_exchange_field_names(component, invalid)

    contract = RuntimeComponentContract(
        imports=("temperature",),
        exports=("sensible_heat_flux",),
    )
    state = create_runtime_component_state(
        component,
        prefill_missing=True,
        contract=contract,
    )
    state = state.with_incoming(
        state.incoming.set("temperature", np.ones(component.grid.shape))
    )
    received = receive_runtime_fields(state, contract)
    assert_allclose_compact(
        received.data.get("temperature"), np.ones(component.grid.shape)
    )


def test_runtime_validation_uses_component_grid_shape_without_shape_argument() -> None:
    grid = make_test_grid(
        name="atm",
        longitude=np.asarray([0.0, 1.0, 2.0]),
        latitude=np.asarray([-1.0, 1.0]),
    )
    component = DummyComponent(name="ATM", grid=grid)
    contract = RuntimeComponentContract(
        imports=("temperature",),
        exports=("sensible_heat_flux",),
    )
    valid_state = RuntimeComponentState(
        data=RuntimeFieldStore.from_mapping(
            {
                "temperature": jnp.ones(grid.shape),
                "sensible_heat_flux": jnp.zeros(grid.shape),
            }
        ),
        incoming=RuntimeFieldStore.from_mapping({"temperature": jnp.ones(grid.shape)}),
        outgoing=RuntimeFieldStore.from_mapping(
            {"sensible_heat_flux": jnp.zeros(grid.shape)}
        ),
    )

    validate_component_runtime_contract_fields(component, valid_state, contract)
    component.validate_runtime_state(valid_state, contract)

    bad_state = valid_state.with_incoming(
        RuntimeFieldStore.from_mapping({"temperature": jnp.ones((1, 3))})
    )
    with pytest.raises(
        CouplerError,
        match=r"has shape \(1, 3\), expected \(2, 3\)",
    ):
        validate_component_runtime_contract_fields(component, bad_state, contract)


def test_send_runtime_fields_updates_outgoing_store() -> None:
    grid = make_test_grid()
    component = DummyComponent(name="ATM", grid=grid)
    timestamp = datetime(2000, 1, 1)
    contract = RuntimeComponentContract(exports=("temperature",))
    component.data["temperature"] = jnp.full(grid.shape, 1.0)

    component_state = send_runtime_fields(
        component,
        create_runtime_component_state(component, contract=RuntimeComponentContract()),
        contract=contract,
    )
    assert_allclose_compact(
        component_state.outgoing.get("temperature"),
        np.full(grid.shape, 1.0),
    )
    assert isinstance(component_state.outgoing.get("temperature"), jax.Array)

    runtime_coupler = Coupler(
        clock=Clock(start=datetime(2000, 1, 1), dt_seconds=3600.0, steps=1)
    )
    monthly = jnp.zeros((12, *grid.shape), dtype=jnp.float64)
    monthly = monthly.at[0].set(jnp.asarray([[1.0, 2.0], [3.0, 4.0]]))
    component.settings.apply_time_interpolation = True
    component.settings.get_field_time_slice = False
    component.data["temperature"] = monthly
    component_state = send_runtime_fields(
        component,
        create_runtime_component_state(component, contract=RuntimeComponentContract()),
        scalar_runtime_step_info(
            timestamp, runtime_coupler.clock, runtime_coupler.settings
        ),
        contract=contract,
    )
    assert_allclose_compact(
        component_state.outgoing.get("temperature"),
        np.asarray(monthly[0]),
    )

    runtime_coupler = Coupler(
        clock=Clock(start=datetime(2000, 1, 3), dt_seconds=3600.0, steps=1)
    )
    daily = jnp.arange(5 * 2 * 2, dtype=jnp.float64).reshape((5, *grid.shape))
    component.settings.apply_time_interpolation = False
    component.settings.get_field_time_slice = True
    component.data["temperature"] = daily
    component_state = send_runtime_fields(
        component,
        create_runtime_component_state(component, contract=RuntimeComponentContract()),
        scalar_runtime_step_info(
            runtime_coupler.clock.start,
            runtime_coupler.clock,
            runtime_coupler.settings,
        ),
        contract=contract,
    )
    assert_allclose_compact(
        component_state.outgoing.get("temperature"),
        np.asarray(daily[2]),
    )


def test_component_forcing_data_read_and_runtime_write_round_trip(
    tmp_path: Path,
) -> None:
    path = tmp_path / "forcing.nc"
    source = np.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    xr.Dataset({"foo": (("x", "y"), source)}).to_netcdf(path)

    reader = ComponentForcingData()
    reader.DATA_FILES = {"sample": str(path)}

    normal_read = reader._read_forcing("foo", "sample")
    flipped_read = reader._read_forcing("foo", "sample", flip_y=True)
    functional_read = read_forcing({"sample": str(path)}, "foo", "sample")
    functional_flipped_read = read_forcing(
        {"sample": str(path)},
        "foo",
        "sample",
        flip_y=True,
    )

    assert isinstance(normal_read, jax.Array)
    assert isinstance(flipped_read, jax.Array)
    assert_allclose_compact(normal_read, source.T)
    assert_allclose_compact(functional_read, normal_read)
    assert_allclose_compact(
        flipped_read,
        np.flip(source.T, axis=1),
    )
    assert_allclose_compact(functional_flipped_read, flipped_read)

    with pytest.raises(KeyError, match="Provided 'where' key 'missing'"):
        reader._read_forcing("foo", "missing")
    with pytest.raises(KeyError, match="Provided 'where' key 'missing'"):
        read_forcing({"sample": str(path)}, "foo", "missing")

    with pytest.raises(KeyError, match="Provided 'where' key 'sample'"):
        reader._read_forcing("bar", "sample")
    with pytest.raises(KeyError, match="Provided 'where' key 'sample'"):
        read_forcing({"sample": str(path)}, "bar", "sample")

    broken = tmp_path / "broken.nc"
    broken.write_text("not-a-netcdf-file", encoding="utf-8")
    reader.DATA_FILES["broken"] = str(broken)

    with pytest.raises(RuntimeError, match="Error reading variable 'foo'"):
        reader._read_forcing("foo", "broken")
    with pytest.raises(RuntimeError, match="Error reading variable 'foo'"):
        read_forcing({"broken": str(broken)}, "foo", "broken")

    state = RuntimeComponentState(
        data=RuntimeFieldStore.empty(),
        incoming=RuntimeFieldStore.from_mapping(
            {"temperature": jnp.asarray([[10.0, 11.0], [12.0, 13.0]])}
        ),
        outgoing=RuntimeFieldStore.from_mapping(
            {"humidity": jnp.asarray([[0.1, 0.2], [0.3, 0.4]])}
        ),
    )
    output = tmp_path / "runtime.nc"

    write_runtime_component_view_to_netcdf(
        RuntimeComponentView.from_component_state("ATM", make_test_grid(), state),
        output,
        masks={"fmask_OCN_ATM_bilinear": jnp.ones((2, 2))},
    )

    with xr.open_dataset(output) as dataset:
        assert_allclose_compact(
            dataset["incoming_temperature"].values,
            state.incoming.get("temperature"),
        )
        assert_allclose_compact(
            dataset["outgoing_humidity"].values,
            state.outgoing.get("humidity"),
        )
        assert_allclose_compact(dataset["latitude"].values, np.asarray([-1.0, 1.0]))
        assert_allclose_compact(dataset["longitude"].values, np.asarray([0.0, 1.0]))
        assert dataset["incoming_temperature"].attrs["component"] == "ATM"
        assert dataset["incoming_temperature"].attrs["runtime_store"] == "incoming"
        assert "fmask_OCN_ATM_bilinear" in dataset

    view_output = tmp_path / "runtime-view.nc"
    write_runtime_component_view_to_netcdf(
        RuntimeComponentView.from_component_state(
            "ATM",
            make_test_grid(),
            state,
        ),
        view_output,
    )
    with xr.open_dataset(view_output) as dataset:
        assert dataset["outgoing_humidity"].attrs["component"] == "ATM"
