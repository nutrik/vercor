from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import datetime
from inspect import Parameter, signature
from types import MappingProxyType
from typing import Any, cast, get_args, get_origin, get_type_hints

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import vercor.components as component_api
from tests._coverage_support import make_test_grid
from vercor.clock import Clock
from vercor.coupler import Coupler
from vercor.dtypes import DTypePolicy
from vercor.exceptions import ComponentError
from vercor.runtime import RuntimeOptions


def _api(name: str) -> Any:
    value = getattr(component_api, name, None)
    assert value is not None, f"vercor.components.{name} is missing"
    return value


def _one_step_coupler(component: Any) -> Coupler:
    return Coupler(
        Clock(datetime(2000, 1, 1), 60.0, 1),
        components=(component,),
        run_order=(component.name,),
    )


class _MutablePayload:
    """Opaque mutable leaf used to prove host payload ownership."""

    def __init__(self, counter: int = 0) -> None:
        self.counter = counter


def test_component_is_the_structural_protocol() -> None:
    component_type = _api("Component")
    component_spec = _api("ComponentSpec")
    component_grid = make_test_grid()

    class StructuralModel:
        name = "model"
        spec = component_spec(outputs=("temperature",))
        grid = component_grid

        def step(
            self,
            fields: dict[str, Any],
            context: Any,
            payload: Any = None,
        ) -> dict[str, Any]:
            _ = context, payload
            return {"temperature": fields["temperature"] + 1.0}

    model = StructuralModel()
    assert getattr(component_type, "_is_protocol", False)
    assert isinstance(model, component_type)


def test_public_constructor_signatures_match_the_v0_4_contract() -> None:
    component_spec = signature(_api("ComponentSpec"))
    callable_component = signature(_api("CallableComponent"))
    data_component = signature(_api("DataComponent"))

    assert tuple(component_spec.parameters) == (
        "inputs",
        "outputs",
        "initial_fields",
        "execution",
        "lifecycle",
        "transfer",
        "output",
        "masks",
    )
    assert tuple(callable_component.parameters) == (
        "name",
        "grid",
        "step",
        "spec",
    )
    assert tuple(data_component.parameters) == ("name", "grid", "fields", "spec")
    for constructor in (component_spec, callable_component, data_component):
        assert (
            constructor.parameters["spec"].kind is Parameter.KEYWORD_ONLY
            if "spec" in constructor.parameters
            else True
        )
    assert component_spec.parameters["execution"].kind is Parameter.KEYWORD_ONLY


def test_component_spec_deduplicates_names_and_freezes_initial_fields() -> None:
    component_spec = _api("ComponentSpec")
    source = {"temperature": np.ones((2, 2))}

    spec = component_spec(
        inputs=("forcing", "forcing"),
        outputs=("temperature", "temperature"),
        initial_fields=source,
    )
    source["temperature"][...] = 9.0
    source["other"] = np.zeros((2, 2))

    assert spec.inputs == ("forcing",)
    assert spec.outputs == ("temperature",)
    assert isinstance(spec.initial_fields, MappingProxyType)
    assert tuple(spec.initial_fields) == ("temperature",)
    np.testing.assert_array_equal(spec.initial_fields["temperature"], np.ones((2, 2)))
    with pytest.raises(TypeError):
        spec.initial_fields["other"] = np.zeros((2, 2))  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        spec.execution = "host"


@pytest.mark.parametrize("mode", ["current", "linear", "daily"])
def test_transfer_policy_accepts_exact_explicit_modes(mode: str) -> None:
    transfer_policy = _api("TransferPolicy")

    assert transfer_policy(mode).time_selection == mode


def test_nested_policies_and_callbacks_validate_immediately() -> None:
    component_spec = _api("ComponentSpec")
    lifecycle_hooks = _api("LifecycleHooks")
    transfer_policy = _api("TransferPolicy")

    with pytest.raises(ValueError, match="time_selection"):
        transfer_policy("nearest")
    with pytest.raises(TypeError, match="lifecycle.*LifecycleHooks"):
        component_spec(lifecycle=object())
    with pytest.raises(TypeError, match="transfer.*TransferPolicy"):
        component_spec(transfer=object())
    with pytest.raises(TypeError, match="setup.*callable"):
        lifecycle_hooks(setup=object())
    with pytest.raises(TypeError, match="prefill.*callable"):
        lifecycle_hooks(prefill=object())
    with pytest.raises(TypeError, match="validate.*callable"):
        lifecycle_hooks(validate=object())
    with pytest.raises(TypeError, match="setup.*exactly.*component, context"):
        lifecycle_hooks(setup=lambda component: None)
    with pytest.raises(TypeError, match="prefill.*exactly.*component, context"):
        lifecycle_hooks(prefill=lambda component, context, extra: None)


def test_result_and_context_mappings_are_frozen_snapshots() -> None:
    step_result = _api("StepResult")
    setup_result = _api("SetupResult")
    prefill_result = _api("PrefillResult")
    prefill_context = _api("PrefillContext")
    source = {"temperature": jnp.ones((2, 2))}

    values = (
        step_result(fields=source),
        setup_result(fields=source),
        prefill_result(fields=source, received=source, sent=source),
        prefill_context(fields=source, received=source, sent=source),
    )
    source["other"] = jnp.zeros((2, 2))

    for value in values:
        for name in ("fields", "received", "sent"):
            mapping = getattr(value, name, None)
            if mapping is None:
                continue
            assert isinstance(mapping, MappingProxyType)
            assert "other" not in mapping
            with pytest.raises(TypeError):
                mapping["other"] = jnp.zeros((2, 2))  # type: ignore[index]


def test_result_payloads_are_registered_shape_stable_pytrees() -> None:
    setup_result = _api("SetupResult")
    value = setup_result(
        fields={"temperature": jnp.ones((2, 2))},
        payload={"counter": jnp.asarray(2.0)},
    )

    leaves, tree = jax.tree_util.tree_flatten(value)
    restored = jax.tree_util.tree_unflatten(tree, leaves)

    assert len(leaves) == 2
    assert isinstance(restored.fields, MappingProxyType)
    np.testing.assert_array_equal(restored.fields["temperature"], np.ones((2, 2)))
    assert float(restored.payload["counter"]) == 2.0


def test_callable_component_uses_declared_outputs_and_setup_payload() -> None:
    callable_component = _api("CallableComponent")
    component_spec = _api("ComponentSpec")
    lifecycle_hooks = _api("LifecycleHooks")
    setup_result = _api("SetupResult")
    step_result = _api("StepResult")

    def setup(component: Any, context: Any) -> Any:
        _ = component, context
        return setup_result(payload={"increment": jnp.asarray(2.0)})

    def step(fields: Any, context: Any, payload: Any) -> Any:
        _ = context
        return step_result(
            fields={"temperature": fields["temperature"] + payload["increment"]}
        )

    component = callable_component(
        "model",
        make_test_grid(),
        step,
        spec=component_spec(
            outputs=("temperature",), lifecycle=lifecycle_hooks(setup=setup)
        ),
    )
    coupler = _one_step_coupler(component)
    initial = coupler.initial_state()
    np.testing.assert_array_equal(
        initial.component("model").field("temperature"), np.zeros((2, 2))
    )
    final = coupler.run(initial)
    np.testing.assert_array_equal(
        final.component("model").field("temperature"), np.full((2, 2), 2.0)
    )
    payload = cast(Any, final._component_state("model").payload)
    assert float(payload["increment"]) == 2.0


def test_data_component_composes_an_immutable_field_declaration() -> None:
    data_component = _api("DataComponent")
    values = np.arange(4.0).reshape(2, 2)
    component = data_component("forcing", make_test_grid(), {"sst": values})

    assert component.spec.outputs == ("sst",)
    values[...] = -1.0
    state = _one_step_coupler(component).initial_state()
    np.testing.assert_array_equal(
        state.component("forcing").field("sst"), np.arange(4.0).reshape(2, 2)
    )


def test_prepared_runtime_binding_is_frozen_and_owns_immutable_fields() -> None:
    data_component = _api("DataComponent")
    coupler = _one_step_coupler(
        data_component("forcing", make_test_grid(), {"sst": np.ones((2, 2))})
    )

    binding = coupler._ensure_prepared().components["forcing"]

    assert isinstance(binding._data, MappingProxyType)
    with pytest.raises(TypeError):
        binding._data["other"] = jnp.zeros((2, 2))  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        binding.grid = make_test_grid("replacement")  # type: ignore[misc]


def test_structural_component_runs_from_its_spec_and_step() -> None:
    component_type = _api("Component")
    component_spec = _api("ComponentSpec")
    component_grid = make_test_grid()

    class StructuralModel:
        name = "model"
        grid = component_grid
        spec = component_spec(
            outputs=("temperature",), initial_fields={"temperature": 3.0}
        )

        def step(self, fields: Any, context: Any, payload: Any = None) -> Any:
            _ = context, payload
            return {"temperature": fields["temperature"] * 2.0}

    model = StructuralModel()
    assert isinstance(model, component_type)

    final = _one_step_coupler(model).run()
    np.testing.assert_array_equal(
        final.component("model").field("temperature"), np.full((2, 2), 6.0)
    )


def test_host_execution_is_a_component_spec_capability() -> None:
    callable_component = _api("CallableComponent")
    component_spec = _api("ComponentSpec")
    component = callable_component(
        "host-model",
        make_test_grid(),
        lambda fields: {},
        spec=component_spec(execution="host"),
    )

    from vercor.components.runtime_execution import host_component_names

    assert host_component_names({component.name: component}) == ["host-model"]


def test_undeclared_initial_fields_fail_with_a_focused_error() -> None:
    component_spec = _api("ComponentSpec")

    with pytest.raises(ComponentError, match="initial field 'secret'.*declared"):
        component_spec(initial_fields={"secret": 1.0})


def test_undeclared_setup_fields_fail_with_a_focused_error() -> None:
    callable_component = _api("CallableComponent")
    component_spec = _api("ComponentSpec")
    lifecycle_hooks = _api("LifecycleHooks")
    setup_result = _api("SetupResult")
    component = callable_component(
        "model",
        make_test_grid(),
        lambda fields: {},
        spec=component_spec(
            lifecycle=lifecycle_hooks(
                setup=lambda component, context: setup_result(fields={"secret": 1.0})
            )
        ),
    )

    with pytest.raises(ComponentError, match="setup.*field 'secret'.*declared"):
        _one_step_coupler(component).initial_state()


def test_undeclared_step_updates_fail_with_a_focused_error() -> None:
    callable_component = _api("CallableComponent")
    component = callable_component(
        "model", make_test_grid(), lambda fields: {"secret": 1.0}
    )

    with pytest.raises(ComponentError, match="step.*field 'secret'.*declared output"):
        _one_step_coupler(component).run()


@pytest.mark.parametrize(
    ("keyword", "value"),
    (
        ("inputs", None),
        ("inputs", "temperature"),
        ("outputs", 7),
        ("outputs", ("temperature", "")),
        ("outputs", ("temperature", 3)),
    ),
)
def test_component_spec_rejects_invalid_field_name_iterables(
    keyword: str,
    value: Any,
) -> None:
    component_spec = _api("ComponentSpec")

    with pytest.raises(TypeError, match="iterable of non-empty strings"):
        component_spec(**{keyword: value})


@pytest.mark.parametrize(
    ("constructor", "keyword"),
    (
        ("ComponentSpec", "initial_fields"),
        ("SetupResult", "fields"),
        ("StepResult", "fields"),
        ("PrefillResult", "received"),
    ),
)
def test_component_mapping_arguments_reject_non_mappings(
    constructor: str,
    keyword: str,
) -> None:
    with pytest.raises(TypeError, match="mapping"):
        _api(constructor)(**{keyword: [("temperature", 1.0)]})


def test_setup_payload_copy_owns_nested_numpy_array_leaves() -> None:
    setup_result = _api("SetupResult")
    lifecycle_hooks = _api("LifecycleHooks")
    callable_component = _api("CallableComponent")
    component_spec = _api("ComponentSpec")
    source = np.asarray([1.0, 2.0])

    component = callable_component(
        "model",
        make_test_grid(),
        lambda fields: {},
        spec=component_spec(
            lifecycle=lifecycle_hooks(
                setup=lambda component, context: setup_result(
                    payload={"nested": (source,)}
                )
            )
        ),
    )
    coupler = _one_step_coupler(component)
    state = coupler.initial_state()
    source[...] = -1.0

    payload = cast(Any, state._component_state("model").payload)
    np.testing.assert_array_equal(
        payload["nested"][0],
        np.asarray([1.0, 2.0]),
    )


def test_initial_states_receive_independent_setup_payload_trees() -> None:
    setup_result = _api("SetupResult")
    lifecycle_hooks = _api("LifecycleHooks")
    callable_component = _api("CallableComponent")
    component_spec = _api("ComponentSpec")
    source = np.asarray([1.0, 2.0])
    component = callable_component(
        "model",
        make_test_grid(),
        lambda fields: {},
        spec=component_spec(
            lifecycle=lifecycle_hooks(
                setup=lambda component, context: setup_result(
                    payload={"nested": [source]}
                )
            )
        ),
    )
    coupler = _one_step_coupler(component)

    first = cast(Any, coupler.initial_state()._component_state("model").payload)
    second = cast(Any, coupler.initial_state()._component_state("model").payload)
    first["nested"][0][0] = 99.0

    assert first is not second
    assert first["nested"] is not second["nested"]
    np.testing.assert_array_equal(second["nested"][0], np.asarray([1.0, 2.0]))
    np.testing.assert_array_equal(source, np.asarray([1.0, 2.0]))


def test_host_in_place_payload_mutation_cannot_change_input_run_state() -> None:
    setup_result = _api("SetupResult")
    step_result = _api("StepResult")
    lifecycle_hooks = _api("LifecycleHooks")
    callable_component = _api("CallableComponent")
    component_spec = _api("ComponentSpec")

    def step(fields: Any, context: Any, payload: Any) -> Any:
        _ = fields, context
        payload["counter"][0] += 1.0
        return step_result(payload=payload)

    component = callable_component(
        "model",
        make_test_grid(),
        step,
        spec=component_spec(
            execution="host",
            lifecycle=lifecycle_hooks(
                setup=lambda component, context: setup_result(
                    payload={"counter": np.asarray([1.0])}
                )
            ),
        ),
    )
    coupler = _one_step_coupler(component)
    initial = coupler.initial_state()

    final = coupler.run(initial)

    initial_payload = cast(Any, initial._component_state("model").payload)
    final_payload = cast(Any, final._component_state("model").payload)
    np.testing.assert_array_equal(initial_payload["counter"], np.asarray([1.0]))
    np.testing.assert_array_equal(final_payload["counter"], np.asarray([2.0]))


def test_opaque_mutable_host_payload_is_independent_per_state_and_step() -> None:
    setup_result = _api("SetupResult")
    step_result = _api("StepResult")
    lifecycle_hooks = _api("LifecycleHooks")
    callable_component = _api("CallableComponent")
    component_spec = _api("ComponentSpec")

    def step(fields: Any, context: Any, payload: _MutablePayload) -> Any:
        _ = fields, context
        payload.counter += 1
        return step_result(payload=payload)

    component = callable_component(
        "model",
        make_test_grid(),
        step,
        spec=component_spec(
            execution="host",
            lifecycle=lifecycle_hooks(
                setup=lambda component, context: setup_result(payload=_MutablePayload())
            ),
        ),
    )
    coupler = _one_step_coupler(component)
    first = coupler.initial_state()
    second = coupler.initial_state()

    final = coupler.run(first)

    first_payload = cast(_MutablePayload, first._component_state("model").payload)
    second_payload = cast(_MutablePayload, second._component_state("model").payload)
    final_payload = cast(_MutablePayload, final._component_state("model").payload)
    assert first_payload is not second_payload
    assert first_payload.counter == 0
    assert second_payload.counter == 0
    assert final_payload.counter == 1


def test_validation_cannot_mutate_initial_or_supplied_state_payload() -> None:
    setup_result = _api("SetupResult")
    lifecycle_hooks = _api("LifecycleHooks")
    callable_component = _api("CallableComponent")
    component_spec = _api("ComponentSpec")
    validation_payloads: list[_MutablePayload] = []

    def validate(component: Any, context: Any) -> None:
        _ = component
        payload = cast(_MutablePayload, context.payload)
        validation_payloads.append(payload)
        payload.counter += 1

    component = callable_component(
        "model",
        make_test_grid(),
        lambda fields: {},
        spec=component_spec(
            execution="host",
            lifecycle=lifecycle_hooks(
                setup=lambda component, context: setup_result(
                    payload=_MutablePayload()
                ),
                validate=validate,
            ),
        ),
    )
    coupler = _one_step_coupler(component)

    initial = coupler.initial_state()
    initial_payload = cast(_MutablePayload, initial._component_state("model").payload)
    assert initial_payload.counter == 0

    final = coupler.run(initial)
    final_payload = cast(_MutablePayload, final._component_state("model").payload)

    assert len(validation_payloads) >= 2
    assert all(payload.counter == 1 for payload in validation_payloads)
    assert all(payload is not initial_payload for payload in validation_payloads)
    assert initial_payload.counter == 0
    assert final_payload.counter == 0


def test_public_component_metadata_annotations_resolve_to_protocol() -> None:
    component_type = _api("Component")
    from vercor.output import SnapshotContext
    from vercor.topology import TopologyContext

    snapshot_hints = get_type_hints(SnapshotContext)
    topology_hints = get_type_hints(TopologyContext)
    coupler_hints = get_type_hints(cast(Any, Coupler.components).fget)

    assert snapshot_hints["component"] is component_type
    assert get_origin(topology_hints["components"]) is not None
    assert component_type in get_args(topology_hints["components"])
    assert component_type in get_args(coupler_hints["return"])


@pytest.mark.parametrize(
    ("payload", "expected_kind"),
    (
        pytest.param(None, "clear", id="clear"),
        pytest.param({"counter": jnp.asarray(2.0)}, "replace", id="replace"),
    ),
)
def test_step_result_pytree_roundtrip_distinguishes_payload_updates(
    payload: Any,
    expected_kind: str,
) -> None:
    step_result = _api("StepResult")
    result = step_result(fields={"value": jnp.asarray(1.0)}, payload=payload)
    leaves, tree = jax.tree_util.tree_flatten(result)
    restored = jax.tree_util.tree_unflatten(tree, leaves)

    assert float(restored.fields["value"]) == 1.0
    if expected_kind == "clear":
        assert restored.payload is None
    else:
        assert float(restored.payload["counter"]) == 2.0


def test_step_result_pytree_roundtrip_preserves_omitted_payload() -> None:
    step_result = _api("StepResult")
    result = step_result(fields={"value": jnp.asarray(1.0)})
    leaves, tree = jax.tree_util.tree_flatten(result)
    restored = jax.tree_util.tree_unflatten(tree, leaves)

    assert len(leaves) == 1
    assert restored.payload is result.payload


def test_jax_runtime_differentiates_through_setup_and_replaced_payload() -> None:
    setup_result = _api("SetupResult")
    step_result = _api("StepResult")
    lifecycle_hooks = _api("LifecycleHooks")
    callable_component = _api("CallableComponent")
    component_spec = _api("ComponentSpec")

    def simulate(scale: Any) -> Any:
        component = callable_component(
            "model",
            make_test_grid(),
            lambda fields, context, payload: step_result(
                fields={"value": fields["value"] + payload["factor"]},
                payload={"factor": payload["factor"] * 2.0},
            ),
            spec=component_spec(
                outputs=("value",),
                lifecycle=lifecycle_hooks(
                    setup=lambda component, context: setup_result(
                        payload={"factor": scale}
                    )
                ),
            ),
        )
        final = _one_step_coupler(component).run()
        return jnp.sum(final.component("model").field("value"))

    primal, tangent = jax.jvp(simulate, (jnp.asarray(3.0),), (jnp.asarray(1.0),))

    assert float(primal) == 12.0
    assert float(tangent) == 4.0
    assert float(jax.grad(simulate)(jnp.asarray(3.0))) == 4.0


def test_jax_scan_rejects_payload_pytree_structure_changes_actionably() -> None:
    setup_result = _api("SetupResult")
    step_result = _api("StepResult")
    lifecycle_hooks = _api("LifecycleHooks")
    callable_component = _api("CallableComponent")
    component_spec = _api("ComponentSpec")
    component = callable_component(
        "model",
        make_test_grid(),
        lambda fields, context, payload: step_result(payload=None),
        spec=component_spec(
            lifecycle=lifecycle_hooks(
                setup=lambda component, context: setup_result(
                    payload={"counter": jnp.asarray(0.0)}
                )
            )
        ),
    )

    with pytest.raises(
        ComponentError,
        match="payload PyTree structure.*execution='host'",
    ):
        _one_step_coupler(component).run()


@pytest.mark.parametrize("phase", ["initial", "setup"])
def test_initial_and_setup_fields_expand_scalars_and_apply_runtime_dtype(
    phase: str,
) -> None:
    component_spec = _api("ComponentSpec")
    callable_component = _api("CallableComponent")
    lifecycle_hooks = _api("LifecycleHooks")
    setup_result = _api("SetupResult")
    initial_fields = {"value": np.float64(1.5)} if phase == "initial" else None
    lifecycle = (
        lifecycle_hooks(
            setup=lambda component, context: setup_result(fields={"value": 1.5})
        )
        if phase == "setup"
        else None
    )
    component = callable_component(
        "model",
        make_test_grid(),
        lambda fields: {},
        spec=component_spec(
            outputs=("value",),
            initial_fields=initial_fields,
            lifecycle=lifecycle,
        ),
    )
    coupler = Coupler(
        Clock(datetime(2000, 1, 1), 60.0, 1),
        components=(component,),
        run_order=(component.name,),
        runtime=RuntimeOptions(dtype=DTypePolicy(enable_x64=False)),
    )

    value = coupler.initial_state().component("model").field("value")

    assert value.shape == component.grid.shape
    assert value.dtype == jnp.float32
    np.testing.assert_array_equal(value, np.full((2, 2), 1.5))


@pytest.mark.parametrize("phase", ["initial", "setup"])
def test_initial_and_setup_fields_reject_noncanonical_shapes(phase: str) -> None:
    component_spec = _api("ComponentSpec")
    callable_component = _api("CallableComponent")
    lifecycle_hooks = _api("LifecycleHooks")
    setup_result = _api("SetupResult")
    invalid = np.zeros((3, 3))
    initial_fields = {"value": invalid} if phase == "initial" else None
    lifecycle = (
        lifecycle_hooks(
            setup=lambda component, context: setup_result(fields={"value": invalid})
        )
        if phase == "setup"
        else None
    )
    component = callable_component(
        "model",
        make_test_grid(),
        lambda fields: {},
        spec=component_spec(
            outputs=("value",),
            initial_fields=initial_fields,
            lifecycle=lifecycle,
        ),
    )

    with pytest.raises(ComponentError, match="canonical grid-field layout"):
        _one_step_coupler(component).initial_state()
