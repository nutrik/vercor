from __future__ import annotations

from collections.abc import Mapping
import importlib
from dataclasses import FrozenInstanceError, dataclass, fields, is_dataclass
from datetime import datetime
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import jax
import jax.numpy as jnp
import pytest

import vercor.output._runtime as output_runtime_module
from tests._architecture_support import package_import_cycles, source_for
from tests._coverage_support import make_test_grid
from vercor import (
    Clock,
    Coupler,
)
from vercor.components import (
    CallableComponent,
    Component,
    ComponentSpec,
    DataComponent,
    LifecycleHooks,
    SetupResult,
)
from vercor.dtypes import DTypePolicy
from vercor.exceptions import CouplerError
from vercor._runtime.topology_state import RuntimeTopologyMaps
from vercor.output import OutputTarget
from vercor.runtime import RuntimeOptions
from vercor.topology import ExchangeTopologyPatch, TopologyContext


def test_prepared_coupling_owns_single_normalized_runtime_boundary() -> None:
    prepared_module = importlib.import_module("vercor._runtime.prepared")
    PreparedCoupling = prepared_module.PreparedCoupling

    assert is_dataclass(PreparedCoupling)
    assert getattr(PreparedCoupling, "__dataclass_params__").frozen is True
    assert [field.name for field in fields(PreparedCoupling)] == [
        "components",
        "exchanges",
        "run_order",
        "contracts",
        "topology_maps",
        "dispatch_context",
        "clock",
        "constants",
        "runtime",
        "interrupts",
    ]


def test_coupler_owns_one_optional_prepared_coupling() -> None:
    coupler_source = Path("vercor/coupler.py").read_text(encoding="utf-8")

    assert "self._prepared: _PreparedCoupling | None = None" in coupler_source
    assert "def _ensure_prepared(" in coupler_source
    assert "RuntimeInputs" not in coupler_source
    assert "_runtime_initialized" not in coupler_source
    assert "CouplerRuntimeResources" not in coupler_source


@pytest.mark.fast_always
def test_contracts_and_dispatch_are_built_once_across_public_runtime_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared_module = importlib.import_module("vercor._runtime.prepared")
    initialization_module = importlib.import_module("vercor._runtime.initialization")
    calls = {"contracts": 0, "dispatch": 0}
    original_contracts = initialization_module.build_exchange_contracts
    original_dispatch = prepared_module.build_runtime_dispatch_context

    def counting_contracts(*args: Any, **kwargs: Any) -> Any:
        calls["contracts"] += 1
        return original_contracts(*args, **kwargs)

    def counting_dispatch(*args: Any, **kwargs: Any) -> Any:
        calls["dispatch"] += 1
        return original_dispatch(*args, **kwargs)

    monkeypatch.setattr(
        initialization_module,
        "build_exchange_contracts",
        counting_contracts,
    )
    monkeypatch.setattr(
        prepared_module,
        "build_runtime_dispatch_context",
        counting_dispatch,
    )
    monkeypatch.setattr(
        output_runtime_module,
        "write_coupler_runtime_outputs",
        lambda **kwargs: None,
    )

    component = DataComponent(
        "MODEL",
        make_test_grid(name="prepared-once"),
        {"temperature": 280.0},
    )
    coupler = Coupler(
        Clock(start=datetime(2000, 1, 1), dt_seconds=60.0, steps=1),
        components=(component,),
        run_order=("MODEL",),
    )

    initial_state = coupler.initial_state()
    prepared = coupler._prepared
    coupler.run(
        state=initial_state,
        output=OutputTarget(
            ".",
            write_period=False,
            write_final_fields=True,
            write_snapshots=False,
        ),
    )

    assert calls == {"contracts": 1, "dispatch": 1}
    assert coupler._prepared is prepared


@pytest.mark.fast_always
def test_run_output_rejects_incompatible_supplied_state_before_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = Clock(start=datetime(2000, 1, 1), dt_seconds=60.0, steps=1)
    coupler = Coupler(
        clock,
        components=(
            DataComponent(
                "MODEL",
                make_test_grid(name="output-state-model"),
                {"temperature": 280.0},
            ),
        ),
        run_order=("MODEL",),
    )
    foreign_coupler = Coupler(
        clock,
        components=(
            DataComponent(
                "FOREIGN",
                make_test_grid(name="output-state-foreign"),
                {"temperature": 281.0},
            ),
        ),
        run_order=("FOREIGN",),
    )
    foreign_state = foreign_coupler.initial_state()
    output_calls: list[str] = []
    monkeypatch.setattr(
        output_runtime_module,
        "write_coupler_runtime_outputs",
        lambda **kwargs: output_calls.append("write"),
    )

    error: CouplerError | None = None
    try:
        coupler.run(
            foreign_state,
            output=OutputTarget(
                ".",
                write_period=False,
                write_final_fields=True,
                write_snapshots=False,
            ),
        )
    except CouplerError as exc:
        error = exc

    assert output_calls == []
    assert error is not None
    assert "MODEL" in str(error)
    assert "missing MODEL" in str(error)


@pytest.mark.fast_always
def test_coupler_has_no_public_configuration_mutators() -> None:
    component = DataComponent(
        "MODEL",
        make_test_grid(name="prepared-invalidation"),
        {"temperature": 280.0},
    )
    coupler = Coupler(
        Clock(start=datetime(2000, 1, 1), dt_seconds=60.0, steps=1),
        components=(component,),
        run_order=("MODEL",),
    )

    prepared = coupler._ensure_prepared()

    for mutator in ("add_component", "add_exchange", "add_exchanges", "set_run_order"):
        assert not hasattr(coupler, mutator)
    assert coupler._ensure_prepared() is prepared


@pytest.mark.fast_always
@pytest.mark.parametrize(
    "mutation",
    ("name", "grid", "spec", "step"),
)
def test_prepared_binding_is_stable_after_original_component_mutation(
    mutation: str,
) -> None:
    component = DataComponent(
        "MODEL",
        make_test_grid(name="prepared-mutation"),
        {"temperature": 280.0},
    )
    coupler = Coupler(
        Clock(start=datetime(2000, 1, 1), dt_seconds=60.0, steps=1),
        components=(component,),
        run_order=("MODEL",),
    )
    state = coupler.initial_state()
    prepared = coupler._ensure_prepared()

    if mutation == "name":
        component.name = "RENAMED"
    elif mutation == "grid":
        component.grid = make_test_grid(name="prepared-mutation-replacement")
    elif mutation == "spec":
        component.spec = ComponentSpec(
            outputs=("temperature",),
            initial_fields={"temperature": 281.0},
        )
    else:
        cast(Any, component).step = lambda fields, context, payload=None: {}

    final_state = coupler.run(state=state)

    assert coupler._ensure_prepared() is prepared
    assert tuple(final_state.components()) == ("MODEL",)
    assert jnp.all(final_state.component("MODEL").field("temperature") == 280.0)


@pytest.mark.fast_always
@pytest.mark.parametrize(
    "replacement",
    (
        None,
        lambda fields, context, payload=None: {},
    ),
)
def test_v0_4_callable_component_prepared_binding_is_stable(
    replacement: Any,
) -> None:
    component = CallableComponent(
        "MODEL",
        make_test_grid(name="prepared-callable-mutation"),
        lambda fields: fields,
    )
    coupler = Coupler(
        Clock(start=datetime(2000, 1, 1), dt_seconds=60.0, steps=1),
        components=(component,),
        run_order=("MODEL",),
    )
    state = coupler.initial_state()
    prepared = coupler._ensure_prepared()

    cast(Any, component).step = replacement

    final_state = coupler.run(state=state)

    assert coupler._ensure_prepared() is prepared
    assert tuple(final_state.components()) == ("MODEL",)


class _MutablePreparedTopologyPolicy:
    def __init__(self) -> None:
        self.enabled = False

    def build(self, context: TopologyContext) -> ExchangeTopologyPatch:
        _ = context
        return ExchangeTopologyPatch()


class _SlotsPreparedTopologyPolicy:
    __slots__ = ("enabled",)

    def __init__(self) -> None:
        self.enabled = False

    def build(self, context: TopologyContext) -> ExchangeTopologyPatch:
        _ = context
        return ExchangeTopologyPatch()


@pytest.mark.fast_always
def test_nested_topology_policy_mutation_does_not_rebuild_preparation() -> None:
    policy = _MutablePreparedTopologyPolicy()
    component = DataComponent(
        "MODEL",
        make_test_grid(name="prepared-topology-mutation"),
        {"temperature": 280.0},
    )
    coupler = Coupler(
        Clock(start=datetime(2000, 1, 1), dt_seconds=60.0, steps=1),
        components=(component,),
        run_order=("MODEL",),
        runtime=RuntimeOptions(topology=policy),
    )
    coupler.initial_state()
    prepared = coupler._ensure_prepared()

    policy.enabled = True

    assert coupler._ensure_prepared() is prepared


@pytest.mark.fast_always
def test_slots_topology_policy_mutation_does_not_rebuild_preparation() -> None:
    policy = _SlotsPreparedTopologyPolicy()
    component = DataComponent(
        "MODEL",
        make_test_grid(name="prepared-slots-topology-mutation"),
        {"temperature": 280.0},
    )
    coupler = Coupler(
        Clock(start=datetime(2000, 1, 1), dt_seconds=60.0, steps=1),
        components=(component,),
        run_order=("MODEL",),
        runtime=RuntimeOptions(topology=policy),
    )
    coupler.initial_state()
    prepared = coupler._ensure_prepared()

    policy.enabled = True

    assert coupler._ensure_prepared() is prepared


class _FreshSeedStructuralComponent:
    name = "MODEL"

    def __init__(self, seed_value: float = 280.0) -> None:
        self.grid = make_test_grid(name="prepared-fresh-structural-seeds")
        self.spec = ComponentSpec(
            outputs=("temperature",),
            initial_fields={"temperature": seed_value},
        )

    def step(
        self,
        fields: Mapping[str, Any],
        context: Any,
        payload: object | None = None,
    ) -> Mapping[str, Any]:
        _ = fields, context, payload
        return {}


@pytest.mark.fast_always
def test_fresh_value_equivalent_structural_seeds_reuse_preparation() -> None:
    component = _FreshSeedStructuralComponent()
    coupler = Coupler(
        Clock(start=datetime(2000, 1, 1), dt_seconds=60.0, steps=1),
        components=(component,),
        run_order=("MODEL",),
    )
    state = coupler.initial_state()
    prepared = coupler._prepared

    assert prepared is not None
    assert coupler._ensure_prepared() is prepared
    coupler.run(state=state)


@pytest.mark.fast_always
def test_replaced_structural_initial_fields_require_a_new_coupler() -> None:
    component = _FreshSeedStructuralComponent()
    coupler = Coupler(
        Clock(start=datetime(2000, 1, 1), dt_seconds=60.0, steps=1),
        components=(component,),
        run_order=("MODEL",),
    )
    state = coupler.initial_state()
    prepared = coupler._ensure_prepared()

    component.spec = ComponentSpec(
        outputs=("temperature",),
        initial_fields={"temperature": 281.0},
    )

    assert coupler._ensure_prepared() is prepared
    assert jnp.all(state.component("MODEL").field("temperature") == 280.0)

    replacement = Coupler(
        Clock(start=datetime(2000, 1, 1), dt_seconds=60.0, steps=1),
        components=(component,),
        run_order=("MODEL",),
    )
    assert jnp.all(
        replacement.initial_state().component("MODEL").field("temperature") == 281.0
    )


class _TracerSeedStructuralComponent:
    name = "MODEL"

    def __init__(self, seed_value: Any) -> None:
        self.grid = make_test_grid(name="prepared-tracer-structural-seeds")
        self.spec = ComponentSpec(
            outputs=("temperature",),
            initial_fields={"temperature": seed_value},
        )

    def step(
        self,
        fields: Mapping[str, Any],
        context: Any,
        payload: object | None = None,
    ) -> Mapping[str, Any]:
        _ = context, payload
        return {"temperature": 2.0 * fields["temperature"]}


@pytest.mark.fast_always
def test_tracer_derived_fresh_structural_seed_reuses_preparation_under_grad() -> None:
    def objective(seed_value: Any) -> Any:
        component = _TracerSeedStructuralComponent(seed_value)
        coupler = Coupler(
            Clock(start=datetime(2000, 1, 1), dt_seconds=60.0, steps=1),
            components=(component,),
            run_order=("MODEL",),
        )
        state = coupler.initial_state()
        final_state = coupler.run(state=state)
        return jnp.mean(final_state.component("MODEL").field("temperature"))

    assert jnp.isclose(jax.grad(objective)(jnp.asarray(3.0)), 2.0)


class _CountingSetupStructuralComponent:
    name = "MODEL"

    def __init__(self) -> None:
        self.grid = make_test_grid(name="prepared-counted-setup")
        self.setup_calls = 0
        self.spec = ComponentSpec(
            outputs=("temperature",),
            lifecycle=LifecycleHooks(setup=self._setup),
        )

    def _setup(self, component: Any, context: Any) -> SetupResult:
        _ = component, context
        self.setup_calls += 1
        return SetupResult(fields={"temperature": 280.0})

    def step(
        self,
        fields: Mapping[str, Any],
        context: Any,
        payload: object | None = None,
    ) -> Mapping[str, Any]:
        _ = fields, context, payload
        return {}


@pytest.mark.fast_always
def test_prepared_reuse_does_not_reinvoke_or_materialize_setup_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared_module = importlib.import_module("vercor._runtime.prepared")
    dtypes_module = importlib.import_module("vercor.dtypes")
    component = _CountingSetupStructuralComponent()
    coupler = Coupler(
        Clock(start=datetime(2000, 1, 1), dt_seconds=60.0, steps=1),
        components=(component,),
        run_order=("MODEL",),
    )
    state = coupler.initial_state()
    calls_after_preparation = component.setup_calls

    def forbidden_array_materialization(*args: Any, **kwargs: Any) -> Any:
        _ = args, kwargs
        raise AssertionError("prepared validation materialized array values")

    dtype_jax = dtypes_module.jax
    monkeypatch.setattr(
        dtypes_module,
        "jax",
        SimpleNamespace(
            config=dtype_jax.config,
            core=dtype_jax.core,
            device_get=forbidden_array_materialization,
        ),
    )
    monkeypatch.setattr(
        prepared_module,
        "blake2b",
        forbidden_array_materialization,
        raising=False,
    )
    monkeypatch.setattr(
        output_runtime_module,
        "write_coupler_runtime_outputs",
        lambda **kwargs: None,
    )

    prepared = coupler._prepared
    assert prepared is not None
    assert coupler._ensure_prepared() is prepared
    coupler.run(
        state=state,
        output=OutputTarget(
            ".",
            write_period=False,
            write_final_fields=True,
            write_snapshots=False,
        ),
    )

    assert component.setup_calls == calls_after_preparation
    prepared_source = source_for("vercor/_runtime/prepared.py")
    assert "device_get" not in prepared_source
    assert "blake2b" not in prepared_source
    assert ".tobytes" not in prepared_source


def _closure_step_with_mutable_configuration() -> tuple[Any, dict[str, float]]:
    configuration = {"increment": 1.0}

    def step(fields: Mapping[str, Any]) -> Mapping[str, Any]:
        return {"temperature": fields["temperature"] + configuration["increment"]}

    return step, configuration


def _default_step_with_mutable_configuration() -> tuple[Any, dict[str, float]]:
    configuration = {"increment": 1.0}

    def step(
        fields: Mapping[str, Any],
        configured: dict[str, float] = configuration,
    ) -> Mapping[str, Any]:
        return {"temperature": fields["temperature"] + configured["increment"]}

    return step, configuration


def _kwdefault_step_with_mutable_configuration() -> tuple[Any, dict[str, float]]:
    configuration = {"increment": 1.0}

    def step(
        fields: Mapping[str, Any],
        *,
        configured: dict[str, float] = configuration,
    ) -> Mapping[str, Any]:
        return {"temperature": fields["temperature"] + configured["increment"]}

    return step, configuration


_MUTABLE_STEP_FACTORIES = {
    "closure": _closure_step_with_mutable_configuration,
    "defaults": _default_step_with_mutable_configuration,
    "kwdefaults": _kwdefault_step_with_mutable_configuration,
}


@pytest.mark.fast_always
@pytest.mark.parametrize("owner", ("closure", "defaults", "kwdefaults"))
def test_ordinary_step_hidden_configuration_is_not_a_prepared_owner(
    owner: str,
) -> None:
    step, configuration = _MUTABLE_STEP_FACTORIES[owner]()
    component = CallableComponent(
        "MODEL",
        make_test_grid(name=f"prepared-function-{owner}-mutation"),
        step,
        spec=ComponentSpec(
            outputs=("temperature",),
            initial_fields={"temperature": 280.0},
        ),
    )
    coupler = Coupler(
        Clock(start=datetime(2000, 1, 1), dt_seconds=60.0, steps=1),
        components=(component,),
        run_order=("MODEL",),
    )
    coupler.initial_state()
    prepared = coupler._prepared

    configuration["increment"] = 2.0

    assert prepared is not None
    assert coupler._ensure_prepared() is prepared


@dataclass
class _CountingLifecycleValidator:
    calls: int = 0

    def __call__(self, component: Component, context: Any) -> None:
        _ = component, context
        self.calls += 1


@pytest.mark.fast_always
@pytest.mark.parametrize("hook_kind", ("function", "callable-object"))
def test_lifecycle_validation_operational_state_does_not_invalidate_preparation(
    hook_kind: str,
) -> None:
    events: list[str] = []

    def validate(component: Component, context: Any) -> None:
        _ = context
        events.append(component.name)

    callable_validator = _CountingLifecycleValidator()
    hook = validate if hook_kind == "function" else callable_validator
    component = DataComponent(
        "MODEL",
        make_test_grid(name=f"prepared-lifecycle-{hook_kind}"),
        {"temperature": 280.0},
        spec=ComponentSpec(lifecycle=LifecycleHooks(validate=hook)),
    )
    coupler = Coupler(
        Clock(start=datetime(2000, 1, 1), dt_seconds=60.0, steps=1),
        components=(component,),
        run_order=("MODEL",),
    )

    state = coupler.initial_state()
    first = coupler.run(state=state)
    coupler.run(state=first)

    # Initial-state validation plus input and chunk-result validation for each
    # run must not invalidate or rebuild the prepared coupling.
    assert events == (["MODEL"] * 5 if hook_kind == "function" else [])
    assert callable_validator.calls == (5 if hook_kind == "callable-object" else 0)


class _MutableStepCallable:
    def __init__(self, increment: float = 1.0) -> None:
        self.increment = increment

    def __call__(self, fields: Mapping[str, Any]) -> Mapping[str, Any]:
        return {"temperature": fields["temperature"] + self.increment}


@pytest.mark.fast_always
def test_mutable_callable_operational_state_does_not_rebuild_preparation() -> None:
    step = _MutableStepCallable()
    component = CallableComponent(
        "MODEL",
        make_test_grid(name="prepared-callable-object-mutation"),
        step,
        spec=ComponentSpec(
            outputs=("temperature",),
            initial_fields={"temperature": 280.0},
        ),
    )
    coupler = Coupler(
        Clock(start=datetime(2000, 1, 1), dt_seconds=60.0, steps=1),
        components=(component,),
        run_order=("MODEL",),
    )
    coupler.initial_state()
    prepared = coupler._ensure_prepared()

    step.increment = 2.0

    assert coupler._ensure_prepared() is prepared


class _BoundMethodStepModel:
    def __init__(self, increment: float = 1.0) -> None:
        self.increment = increment
        self.self_reference = self

    def step(self, fields: Mapping[str, Any]) -> Mapping[str, Any]:
        return {"temperature": fields["temperature"] + self.increment}


@pytest.mark.fast_always
def test_bound_method_operational_state_does_not_rebuild_preparation() -> None:
    model = _BoundMethodStepModel()
    component = CallableComponent(
        "MODEL",
        make_test_grid(name="prepared-bound-method-owner-mutation"),
        model.step,
        spec=ComponentSpec(
            outputs=("temperature",),
            initial_fields={"temperature": 280.0},
        ),
    )
    coupler = Coupler(
        Clock(start=datetime(2000, 1, 1), dt_seconds=60.0, steps=1),
        components=(component,),
        run_order=("MODEL",),
    )
    coupler.initial_state()
    prepared = coupler._prepared

    assert prepared is not None
    assert coupler._ensure_prepared() is prepared

    model.increment = 2.0

    assert coupler._ensure_prepared() is prepared


def _partial_step(
    fields: Mapping[str, Any],
    *,
    increment: float,
) -> Mapping[str, Any]:
    return {"temperature": fields["temperature"] + increment}


@pytest.mark.fast_always
def test_partial_keyword_state_does_not_rebuild_preparation() -> None:
    step = partial(_partial_step, increment=1.0)
    component = CallableComponent(
        "MODEL",
        make_test_grid(name="prepared-partial-keyword-mutation"),
        step,
        spec=ComponentSpec(
            outputs=("temperature",),
            initial_fields={"temperature": 280.0},
        ),
    )
    coupler = Coupler(
        Clock(start=datetime(2000, 1, 1), dt_seconds=60.0, steps=1),
        components=(component,),
        run_order=("MODEL",),
    )
    coupler.initial_state()
    prepared = coupler._ensure_prepared()

    assert step.keywords is not None
    step.keywords["increment"] = 2.0

    assert coupler._ensure_prepared() is prepared


@pytest.mark.fast_always
def test_coupler_public_configuration_is_read_only() -> None:
    component = DataComponent(
        "MODEL",
        make_test_grid(name="prepared-configuration"),
        {"temperature": 280.0},
    )
    coupler = Coupler(
        Clock(start=datetime(2000, 1, 1), dt_seconds=60.0, steps=1),
        components=(component,),
        run_order=("MODEL",),
    )
    replacement_clock = Clock(
        start=coupler.clock.start,
        dt_seconds=coupler.clock.dt_seconds,
        steps=coupler.clock.steps,
    )
    with pytest.raises(AttributeError, match="clock.*no setter"):
        coupler.clock = replacement_clock  # type: ignore[misc]
    with pytest.raises(AttributeError, match="runtime.*no setter"):
        coupler.runtime = RuntimeOptions()  # type: ignore[misc]
    with pytest.raises(TypeError):
        coupler.components["MODEL"] = component  # type: ignore[index]

    assert coupler.components == {"MODEL": component}
    assert coupler.run_order == ("MODEL",)


@pytest.mark.fast_always
def test_runtime_topology_maps_are_frozen_read_only_views() -> None:
    key = "SRC->DST"
    maps = RuntimeTopologyMaps(
        regridders={key: object()},
        binary_masks={key: jnp.ones((2, 2))},
        fractional_masks={key: jnp.ones((2, 2))},
    )

    assert getattr(RuntimeTopologyMaps, "__dataclass_params__").frozen is True
    with pytest.raises(TypeError):
        maps.regridders[key] = object()  # type: ignore[index]
    with pytest.raises(TypeError):
        maps.binary_masks[key] = jnp.zeros((2, 2))  # type: ignore[index]
    with pytest.raises(TypeError):
        maps.fractional_masks[key] = jnp.zeros((2, 2))  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        maps.regridders = {}  # type: ignore[misc]


@pytest.mark.fast_always
def test_precision_capability_and_allocation_policy_remain_distinct(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dtypes_module = importlib.import_module("vercor.dtypes")

    class FakeConfig:
        def __init__(self, enabled: bool) -> None:
            self.enabled = enabled
            self.updates: list[tuple[str, bool]] = []

        def read(self, name: str) -> bool:
            assert name == "jax_enable_x64"
            return self.enabled

        def update(self, name: str, value: bool) -> None:
            assert name == "jax_enable_x64"
            self.updates.append((name, value))
            self.enabled = value

    fake_config = FakeConfig(enabled=False)
    monkeypatch.setattr(dtypes_module.jax, "config", fake_config)
    x64_component = DataComponent(
        "X64",
        make_test_grid(name="x64-capability"),
        {"temperature": 280.0},
    )
    x64_coupler = Coupler(
        Clock(start=datetime(2000, 1, 1), dt_seconds=60.0, steps=1),
        components=(x64_component,),
        run_order=("X64",),
        runtime=RuntimeOptions(dtype=DTypePolicy(enable_x64=True)),
    )

    x64_coupler.initial_state()

    assert fake_config.updates == [("jax_enable_x64", True)]

    fake_config.enabled = True
    fake_config.updates.clear()
    f32_component = DataComponent(
        "F32",
        make_test_grid(name="f32-policy"),
        {"temperature": 280.0},
    )
    f32_coupler = Coupler(
        Clock(start=datetime(2000, 1, 1), dt_seconds=60.0, steps=1),
        components=(f32_component,),
        run_order=("F32",),
        runtime=RuntimeOptions(dtype=DTypePolicy(enable_x64=False)),
    )

    state = f32_coupler.initial_state()

    assert fake_config.updates == []
    assert state.component("F32").field("temperature").dtype == jnp.float32


def test_runtime_preparation_module_owns_runtime_state_preparation() -> None:
    preparation_path = Path("vercor/_runtime/preparation.py")
    assert preparation_path.exists()

    preparation_source = preparation_path.read_text(encoding="utf-8")
    assert "class PreparedRuntimeState" not in preparation_source
    assert "def runtime_state_from_components(" in preparation_source
    assert "def validate_runtime_state(" in preparation_source
    assert "def create_runtime_state(" in preparation_source
    assert "def prepare_runtime_state(" in preparation_source
    assert "RuntimePreparationInputs" not in preparation_source
    assert "return PreparedRuntimeState" not in preparation_source


@pytest.mark.fast_always
def test_surface_masks_owns_surface_role_lookup() -> None:
    component_topology_path = Path("vercor/_runtime/component_topology.py")
    topology_source = source_for("vercor/_runtime/topology.py")
    surface_masks_source = source_for("vercor/_runtime/surface_masks.py")
    topology_policy_source = source_for("vercor/_runtime/topology_policy.py")
    initialization_source = source_for("vercor/_runtime/initialization.py")

    assert not component_topology_path.exists()
    assert "def validate_component_topology_names(" not in topology_source
    assert "def get_component(" not in topology_source
    assert "def require_component(" not in topology_source
    assert "from vercor._runtime.component_topology import" not in topology_source
    assert "def _require_surface_role(" in surface_masks_source
    assert "from vercor._runtime.component_topology import" not in surface_masks_source
    assert (
        "from vercor._runtime.component_topology import" not in topology_policy_source
    )
    assert "from vercor._runtime.component_topology import" not in initialization_source


@pytest.mark.fast_always
def test_runtime_topology_state_groups_read_only_maps() -> None:
    topology_state_module = importlib.import_module("vercor._runtime.topology_state")
    topology_state_source = source_for("vercor/_runtime/topology_state.py")
    topology_source = source_for("vercor/_runtime/topology.py")

    assert hasattr(topology_state_module, "RuntimeTopologyMaps")
    RuntimeTopologyMaps = topology_state_module.RuntimeTopologyMaps
    assert is_dataclass(RuntimeTopologyMaps)
    assert getattr(RuntimeTopologyMaps, "__dataclass_params__").frozen is True
    assert hasattr(RuntimeTopologyMaps, "__slots__")
    assert not hasattr(RuntimeTopologyMaps, "empty")
    assert [field.name for field in fields(RuntimeTopologyMaps)] == [
        "regridders",
        "binary_masks",
        "fractional_masks",
    ]
    assert "class RuntimeTopologyMaps" in topology_state_source
    assert "class ExchangeTopologyState" not in topology_state_source
    assert "class RuntimeTopologyMaps" not in topology_source
    assert "MappingProxyType" in topology_state_source
    assert not Path("vercor/_runtime/resources.py").exists()


@pytest.mark.fast_always
def test_runtime_topology_policy_boundaries_are_focused() -> None:
    topology_state_module = importlib.import_module("vercor._runtime.topology_state")
    topology_state_source = source_for("vercor/_runtime/topology_state.py")
    exchange_topology_source = source_for("vercor/_runtime/exchange_topology.py")
    surface_masks_source = source_for("vercor/_runtime/surface_masks.py")
    topology_policy_source = source_for("vercor/_runtime/topology_policy.py")
    topology_source = source_for("vercor/_runtime/topology.py")
    assert not hasattr(topology_state_module, "ExchangeTopologyState")
    assert not hasattr(topology_state_module, "SurfaceExchangeMasks")

    assert "RuntimeRegridder =" not in topology_state_source
    assert "BilinearRectilinearRegridder" not in topology_state_source
    assert "ConservativeRectilinearRegridder" not in topology_state_source
    assert "def build_exchange_topology_maps(" in exchange_topology_source
    assert "def create_surface_exchange_masks(" in surface_masks_source
    assert "def validate_land_mask_consistency(" in surface_masks_source
    assert "def apply_surface_exchange_masks(" not in surface_masks_source
    assert "def apply_topology_policy(" in topology_policy_source
    assert "def build_topology_context(" in topology_policy_source

    for marker in (
        "compute_ocn_lnd_masks_on_atm_grid",
        "check_remap_conservation",
        "check_total_lnd_ocn_mask_sum",
        "ConservativeRectilinearRegridder",
        "jax_ones",
        "def create_exchange_masks(",
        "def validate_land_mask_consistency(",
        "def initialize_regridders_and_masks(",
        "def patch_exchange_masks(",
    ):
        assert marker not in topology_source

    assert "import vercor._runtime.exchange_topology as" in topology_source
    assert "import vercor._runtime.topology_policy as" in topology_source
    assert "import vercor._runtime.surface_masks as" not in topology_source
    assert "from vercor._runtime.topology_state import" in topology_source
    assert "isinstance(policy, SurfaceMaskPolicy)" not in topology_policy_source


@pytest.mark.fast_always
def test_prepared_boundary_owns_runtime_initialization_and_dispatch() -> None:
    prepared_module = importlib.import_module("vercor._runtime.prepared")
    prepared_source = source_for("vercor/_runtime/prepared.py")

    assert prepared_module.PreparedCoupling.__module__ == "vercor._runtime.prepared"
    assert is_dataclass(prepared_module.PreparedCoupling)
    assert getattr(prepared_module.PreparedCoupling, "__dataclass_params__").frozen
    assert prepared_module.prepare_coupling.__module__ == "vercor._runtime.prepared"
    assert "initialize_coupler_runtime(" in prepared_source
    assert "build_runtime_dispatch_context(" in prepared_source


@pytest.mark.fast_always
def test_runtime_state_validation_module_owns_runtime_topology_validation() -> None:
    state_validation_path = Path("vercor/_runtime/state_validation.py")
    coupler_state_source = source_for("vercor/_runtime/coupler_state.py")
    preparation_source = source_for("vercor/_runtime/preparation.py")
    facade_source = source_for("vercor/_runtime/facade.py")
    coupler_source = source_for("vercor/coupler.py")

    assert state_validation_path.exists()
    state_validation_source = state_validation_path.read_text(encoding="utf-8")
    assert "def validate_runtime_state(" in state_validation_source
    assert "def validate_runtime_state(" not in coupler_state_source
    assert "from vercor._runtime.state_validation import" in preparation_source
    assert "from vercor._runtime.state_validation import" not in facade_source
    assert "from vercor._runtime.state_validation import" not in coupler_source


def test_runtime_facade_reexports_preparation_without_owning_it() -> None:
    facade_source = source_for("vercor/_runtime/facade.py")
    preparation_source = source_for("vercor/_runtime/preparation.py")

    assert "from vercor._runtime.preparation import" in facade_source
    assert "PreparedRuntimeState" not in facade_source
    assert "PreparedRuntimeState" not in preparation_source
    assert "Protocol" not in preparation_source
    assert "RuntimePreparationInputs" not in preparation_source
    assert "RuntimeInputs" not in preparation_source
    assert "from vercor._runtime.prepared import PreparedCoupling" in preparation_source
    assert "def runtime_state_from_components(" not in facade_source
    assert "def validate_runtime_state(" not in facade_source
    assert "def create_runtime_state(" not in facade_source
    assert "def prepare_runtime_state(" not in facade_source


@pytest.mark.fast_always
def test_runtime_facade_builds_and_executes_the_validated_plan_directly() -> None:
    runner_path = Path("vercor/_runtime/runner.py")
    facade_source = source_for("vercor/_runtime/facade.py")
    backend_source = source_for("vercor/_runtime/backends.py")
    execution_source = source_for("vercor/_runtime/execution.py")
    run_body = facade_source.split("def run(", 1)[1]

    assert not runner_path.exists()
    assert "def execute_jax_chunk(" in backend_source
    assert "def execute_host_chunk(" in backend_source
    assert "def execute_custom_chunk(" in backend_source
    assert "def execute_jax_chunk(" not in facade_source
    assert "def execute_host_chunk(" not in facade_source
    assert "def execute_plan(" in execution_source
    assert "execute_jax_chunk(" in execution_source
    assert "execute_host_chunk(" in execution_source
    assert "execute_custom_chunk(" in execution_source
    assert "vercor._runtime.runner" not in backend_source
    assert "class _JAXScannedBackend" not in backend_source
    assert "class _HostLoopBackend" not in backend_source
    assert "RuntimeRunContext(" in run_body
    assert "build_validated_execution_plan(context)" in run_body
    assert "execute_plan(runtime_state, plan=plan, context=context)" in run_body
    assert "compiled_runtime_cache_key(" not in run_body
    assert "get_or_compile_for_context(" not in facade_source
    assert "context.compiled_runtime_cache_key(" not in facade_source
    assert "get_or_compile(" not in facade_source
    assert facade_source.count("signal_scope(") == 1
    assert "donate_state" not in facade_source
    assert "raise CouplerError(" not in run_body


def test_runtime_package_has_no_top_level_import_cycles() -> None:
    assert package_import_cycles("vercor/_runtime", "vercor._runtime") == []
