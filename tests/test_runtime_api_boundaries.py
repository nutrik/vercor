from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from inspect import signature
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest

import vercor
import vercor.runtime as runtime
import vercor.topology as topology
from tests._coverage_support import make_test_grid
from tests.assertions import assert_allclose_compact
from vercor.clock import Clock
from vercor.components import (
    CallableComponent,
    Component,
    ComponentSpec,
    DataComponent,
    LifecycleHooks,
    SetupContext,
    SetupResult,
    StepContext,
    TransferPolicy,
)
from vercor.coupler import Coupler
from vercor.exchanges import Exchange
from vercor.exceptions import CouplerError
from vercor.output import OutputSpec, OutputTarget, SnapshotContext
from vercor.runtime import ExecutionContext, RuntimeDriver, RuntimeOptions
from vercor.state import RunState


def _clock(steps: int = 1) -> Clock:
    return Clock(
        start=datetime(2000, 1, 1),
        dt_seconds=60.0,
        steps=steps,
    )


@pytest.mark.fast_always
def test_runtime_module_owns_public_runtime_contracts() -> None:
    options = runtime.RuntimeOptions()

    assert runtime.__all__ == [
        "ExecutionBackend",
        "ExecutionChunk",
        "ExecutionContext",
        "ExecutionPlan",
        "RuntimeDriver",
        "RuntimeOptions",
        "SequentialWorkflow",
        "StepPlan",
        "Workflow",
        "WorkflowContext",
    ]
    assert RuntimeOptions is runtime.RuntimeOptions
    assert ExecutionContext is runtime.ExecutionContext
    assert RuntimeDriver is runtime.RuntimeDriver
    assert not hasattr(runtime, "RunState")
    assert not hasattr(runtime, "ComponentState")
    assert not hasattr(vercor, "ExecutionContext")
    assert not hasattr(vercor, "RuntimeDriver")
    assert options.topology is None
    assert not hasattr(options, "model_year_seconds")
    assert "model_year_seconds" not in signature(RuntimeOptions).parameters
    assert "year_in_seconds" not in signature(runtime.RuntimeOptions).parameters
    assert "surface_masks" not in signature(runtime.RuntimeOptions).parameters
    assert not hasattr(runtime, "SurfaceMaskPolicy")
    assert not hasattr(vercor, "SurfaceMaskPolicy")
    assert "RuntimeRunContext" not in str(signature(runtime.ExecutionBackend.execute))


@pytest.mark.fast_always
def test_topology_module_owns_public_topology_contracts() -> None:
    policy = topology.SurfaceMaskPolicy(mode="disabled")
    patch = topology.ExchangeTopologyPatch(binary_masks={"custom": jnp.asarray(1.0)})

    assert topology.__all__ == [
        "ExchangeTopologyPatch",
        "SurfaceMaskPolicy",
        "TopologyContext",
        "TopologyPolicy",
    ]
    assert policy.mode == "disabled"
    assert patch.binary_masks["custom"].shape == ()
    assert "TopologyPolicy" not in vercor.__all__
    assert "SurfaceMaskPolicy" not in vercor.__all__


@pytest.mark.fast_always
def test_custom_execution_backend_receives_public_context_and_driver() -> None:
    grid = make_test_grid(name="custom-backend")

    class StepOnceBackend:
        def __init__(self) -> None:
            self.calls: list[tuple[ExecutionContext, object]] = []

        def execute(
            self,
            state: RunState,
            *,
            context: ExecutionContext,
            chunk: runtime.ExecutionChunk,
            driver: RuntimeDriver,
        ) -> RunState:
            self.calls.append((context, driver))
            assert context.component_names == ("MODEL",)
            assert context.options.backend is self
            return driver.run_step(state, chunk.steps[0])

    backend = StepOnceBackend()
    component = CallableComponent(
        "MODEL",
        grid,
        lambda fields, context: {"temperature": fields["temperature"] + 1.0},
        spec=ComponentSpec(
            inputs=("temperature",),
            outputs=("temperature",),
            initial_fields={"temperature": 280.0},
        ),
    )
    coupler = Coupler(
        _clock(),
        components=(component,),
        run_order=("MODEL",),
        runtime=RuntimeOptions(backend=backend),
    )

    final_state = coupler.run()

    assert len(backend.calls) == 1
    assert_allclose_compact(
        final_state.component("MODEL").field("temperature"),
        jnp.full(grid.shape, 281.0),
    )


@pytest.mark.fast_always
def test_custom_backend_runs_complete_host_exchange_order_from_supplied_state() -> None:
    grid = make_test_grid(name="custom-host-backend")
    observed_steps: list[tuple[str, int, object]] = []

    def step_source(
        fields: Mapping[str, Any],
        context: StepContext,
    ) -> Mapping[str, Any]:
        observed_steps.append(("SRC", int(context.step), context.time))
        return {"flux": fields["flux"] + 1.0}

    def step_target(
        fields: Mapping[str, Any],
        context: StepContext,
    ) -> Mapping[str, Any]:
        observed_steps.append(("DST", int(context.step), context.time))
        return {"total": fields["total"] + fields["flux"]}

    class SequentialBackend:
        def __init__(self) -> None:
            self.received_state: RunState | None = None
            self.driver_calls: list[tuple[int, tuple[str, ...]]] = []

        def execute(
            self,
            state: RunState,
            *,
            context: ExecutionContext,
            chunk: runtime.ExecutionChunk,
            driver: RuntimeDriver,
        ) -> RunState:
            self.received_state = state
            _ = context
            for plan in chunk.steps:
                self.driver_calls.append((plan.step, plan.components))
                state = driver.run_step(state, plan)
            return state

    source = CallableComponent(
        "SRC",
        grid,
        step_source,
        spec=ComponentSpec(
            inputs=("flux",),
            outputs=("flux",),
            initial_fields={"flux": 1.0},
            execution="host",
        ),
    )
    target = CallableComponent(
        "DST",
        grid,
        step_target,
        spec=ComponentSpec(
            inputs=("flux", "total"),
            outputs=("total",),
            initial_fields={"total": 0.0},
            execution="host",
        ),
    )
    backend = SequentialBackend()
    coupler = Coupler(
        _clock(steps=2),
        components=(source, target),
        exchanges=(Exchange("SRC", "DST", ("flux",), quantity="intensive"),),
        run_order=("SRC", "DST"),
        runtime=RuntimeOptions(backend=backend),
    )
    initial_state = coupler.initial_state().replace_fields(
        "SRC",
        {"flux": jnp.full(grid.shape, 10.0)},
    )

    final_state = coupler.run(initial_state)

    assert backend.received_state is not None
    assert_allclose_compact(backend.received_state.component("SRC").field("flux"), 10.0)
    assert_allclose_compact(initial_state.component("SRC").field("flux"), 10.0)
    assert backend.driver_calls == [
        (0, ("SRC", "DST")),
        (1, ("SRC", "DST")),
    ]
    assert observed_steps == [
        ("SRC", 0, datetime(2000, 1, 1, 0, 0)),
        ("DST", 0, datetime(2000, 1, 1, 0, 0)),
        ("SRC", 1, datetime(2000, 1, 1, 0, 1)),
        ("DST", 1, datetime(2000, 1, 1, 0, 1)),
    ]
    assert_allclose_compact(
        final_state.component("SRC").field("flux"),
        jnp.full(grid.shape, 12.0),
    )
    assert_allclose_compact(
        final_state.component("DST").field("total"),
        jnp.full(grid.shape, 23.0),
    )


@pytest.mark.fast_always
@pytest.mark.parametrize(
    ("returned", "actual_type"),
    (
        pytest.param(None, "NoneType", id="none"),
        pytest.param({"state": "invalid"}, "dict", id="mapping"),
        pytest.param(object(), "object", id="object"),
    ),
)
def test_custom_backend_rejects_non_run_state_return(
    returned: object,
    actual_type: str,
) -> None:
    class InvalidReturnBackend:
        def execute(
            self,
            state: RunState,
            *,
            context: ExecutionContext,
            chunk: runtime.ExecutionChunk,
            driver: RuntimeDriver,
        ) -> Any:
            _ = state, context, chunk, driver
            return returned

    grid = make_test_grid(name=f"invalid-backend-{actual_type}")
    coupler = Coupler(
        _clock(),
        components=(DataComponent("MODEL", grid, {"value": 1.0}),),
        run_order=("MODEL",),
        runtime=RuntimeOptions(backend=InvalidReturnBackend()),
    )

    with pytest.raises(
        CouplerError,
        match=rf"InvalidReturnBackend.*return.*RunState.*{actual_type}",
    ):
        coupler.run()


class _ReturnForeignStateBackend:
    def __init__(self, state: RunState) -> None:
        self.state = state

    def execute(
        self,
        state: RunState,
        *,
        context: ExecutionContext,
        chunk: runtime.ExecutionChunk,
        driver: RuntimeDriver,
    ) -> RunState:
        _ = context
        for plan in chunk.steps:
            state = driver.run_step(state, plan)
        return self.state


def _data_state(
    *components: DataComponent,
    run_order: tuple[str, ...],
) -> RunState:
    return Coupler(
        _clock(),
        components=components,
        run_order=run_order,
        runtime=RuntimeOptions(topology=None),
    ).initial_state()


@pytest.mark.fast_always
def test_custom_backend_accepts_structurally_compatible_foreign_run_state() -> None:
    grid = make_test_grid(name="compatible-foreign-state")
    foreign_state = _data_state(
        DataComponent("MODEL", grid, {"value": 9.0}),
        run_order=("MODEL",),
    )
    coupler = Coupler(
        _clock(),
        components=(DataComponent("MODEL", grid, {"value": 1.0}),),
        run_order=("MODEL",),
        runtime=RuntimeOptions(
            topology=None,
            backend=_ReturnForeignStateBackend(foreign_state),
        ),
    )

    result = coupler.run()

    assert result is foreign_state
    assert_allclose_compact(
        result.component("MODEL").field("value"),
        jnp.full(grid.shape, 9.0),
    )


@pytest.mark.fast_always
@pytest.mark.parametrize(
    "case",
    (
        pytest.param("missing", id="missing-component"),
        pytest.param("extra", id="extra-component"),
        pytest.param("extra-field", id="extra-field"),
        pytest.param("shape", id="incompatible-field-shape"),
    ),
)
def test_custom_backend_validates_returned_run_state_schema(case: str) -> None:
    grid = make_test_grid(name=f"custom-backend-schema-{case}")
    if case == "missing":
        foreign_state = _data_state(
            DataComponent("OTHER", grid, {"value": 1.0}),
            run_order=("OTHER",),
        )
        message = "missing.*MODEL"
    elif case == "extra":
        foreign_state = _data_state(
            DataComponent("MODEL", grid, {"value": 1.0}),
            DataComponent("EXTRA", grid, {"value": 2.0}),
            run_order=("MODEL",),
        )
        message = "extra.*EXTRA"
    elif case == "extra-field":
        foreign_state = _data_state(
            DataComponent(
                "MODEL",
                grid,
                {"value": 1.0, "extra_field": 2.0},
            ),
            run_order=("MODEL",),
        )
        message = "MODEL.*fields.*extra_field"
    else:
        wide_grid = make_test_grid(
            name="custom-backend-schema-wide",
            longitude=np.asarray([0.0, 1.0, 2.0]),
        )
        foreign_state = _data_state(
            DataComponent("MODEL", wide_grid, {"value": 1.0}),
            run_order=("MODEL",),
        )
        message = r"MODEL.*runtime grid name.*custom-backend-schema-wide"

    coupler = Coupler(
        _clock(),
        components=(DataComponent("MODEL", grid, {"value": 1.0}),),
        run_order=("MODEL",),
        runtime=RuntimeOptions(
            topology=None,
            backend=_ReturnForeignStateBackend(foreign_state),
        ),
    )

    with pytest.raises(CouplerError, match=message):
        coupler.run()


@pytest.mark.fast_always
def test_runtime_driver_rejects_invalid_dispatch_before_component_step() -> None:
    class InvalidDriverCallBackend:
        def execute(
            self,
            state: RunState,
            *,
            context: ExecutionContext,
            chunk: runtime.ExecutionChunk,
            driver: RuntimeDriver,
        ) -> RunState:
            _ = context
            return driver.run_step(object(), chunk.steps[0])  # type: ignore[arg-type]

    grid = make_test_grid(name="driver-invalid-state")
    coupler = Coupler(
        _clock(steps=2),
        components=(DataComponent("MODEL", grid, {"value": 1.0}),),
        run_order=("MODEL",),
        runtime=RuntimeOptions(backend=InvalidDriverCallBackend()),
    )

    with pytest.raises(CouplerError, match="run_step state.*RunState.*object"):
        coupler.run()


@pytest.mark.fast_always
def test_runtime_driver_uses_each_plan_absolute_step_and_time() -> None:
    grid = make_test_grid(name="driver-jax-step")
    observed_contexts: list[StepContext] = []

    def step_model(
        fields: Mapping[str, Any],
        context: StepContext,
    ) -> Mapping[str, Any]:
        observed_contexts.append(context)
        return {"value": fields["value"] + 1.0}

    class SequentialPlanBackend:
        def execute(
            self,
            state: RunState,
            *,
            context: ExecutionContext,
            chunk: runtime.ExecutionChunk,
            driver: RuntimeDriver,
        ) -> RunState:
            _ = context
            for plan in chunk.steps:
                state = driver.run_step(state, plan)
            return state

    component = CallableComponent(
        "MODEL",
        grid,
        step_model,
        spec=ComponentSpec(
            inputs=("value",),
            outputs=("value",),
            initial_fields={"value": 1.0},
            execution="host",
        ),
    )
    coupler = Coupler(
        _clock(steps=3),
        components=(component,),
        run_order=("MODEL",),
        runtime=RuntimeOptions(backend=SequentialPlanBackend()),
    )

    final_state = coupler.run()

    assert [int(context.step) for context in observed_contexts] == [0, 1, 2]
    assert [context.time for context in observed_contexts] == [
        datetime(2000, 1, 1, 0, 0),
        datetime(2000, 1, 1, 0, 1),
        datetime(2000, 1, 1, 0, 2),
    ]
    assert_allclose_compact(
        final_state.component("MODEL").field("value"),
        jnp.full(grid.shape, 4.0),
    )


@pytest.mark.fast_always
def test_runtime_backends_own_loops_without_a_runner_module() -> None:
    package_root = Path(vercor.__file__).parent
    backend_source = (package_root / "_runtime" / "backends.py").read_text()
    facade_source = (package_root / "_runtime" / "facade.py").read_text()

    assert not (package_root / "_runtime" / "runner.py").exists()
    assert "vercor._runtime.runner" not in backend_source
    assert "class _JAXScannedBackend" not in backend_source
    assert "class _HostLoopBackend" not in backend_source
    for implementation in ("execute_jax_chunk", "execute_host_chunk"):
        assert f"def {implementation}" in backend_source
        assert f"def {implementation}" not in facade_source


@pytest.mark.fast_always
def test_structural_component_can_request_host_runtime_through_spec() -> None:
    setup_contexts: list[SetupContext] = []

    def setup(component: object, context: SetupContext) -> SetupResult:
        _ = component
        setup_contexts.append(context)
        return SetupResult()

    class PlainHostComponent:
        name = "MODEL"

        def __init__(self) -> None:
            self.grid = make_test_grid(name="plain-host")
            self.spec = ComponentSpec(
                inputs=("temperature",),
                outputs=("temperature",),
                initial_fields={"temperature": 280.0},
                execution="host",
                lifecycle=LifecycleHooks(setup=setup),
            )

        def step(
            self,
            fields: Mapping[str, Any],
            context: StepContext,
            payload: object | None = None,
        ) -> Mapping[str, Any]:
            _ = payload
            return {
                "temperature": fields["temperature"] + jnp.asarray(context.step) + 1.0
            }

    component = PlainHostComponent()
    coupler = Coupler(
        _clock(steps=2),
        components=(component,),
        run_order=("MODEL",),
    )

    final_state = coupler.run()

    assert len(setup_contexts) == 1
    assert setup_contexts[0].run_order == ("MODEL",)
    assert not hasattr(component, "initial_fields")
    assert not hasattr(component, "initialize")
    assert_allclose_compact(
        final_state.component("MODEL").field("temperature"),
        jnp.full(component.grid.shape, 283.0),
    )


@pytest.mark.fast_always
def test_transfer_policy_belongs_to_component_spec() -> None:
    policy = TransferPolicy(time_selection="linear")
    grid = make_test_grid(name="import-policy")
    spec = ComponentSpec(outputs=("temperature",), transfer=policy)

    component = DataComponent(
        "OBS",
        grid,
        fields={"temperature": 280.0},
        spec=spec,
    )

    assert not hasattr(spec, "import_policy")
    assert not hasattr(component, "import_policy")
    assert component.spec.transfer is policy
    assert not hasattr(vercor, "FieldImportPolicy")


@pytest.mark.fast_always
def test_coupler_components_exposes_read_only_private_metadata() -> None:
    class PlainComponent:
        name = "MODEL"

        def __init__(self) -> None:
            self.grid = make_test_grid(name="component-info")
            self.spec = ComponentSpec(
                outputs=("temperature",),
                initial_fields={"temperature": 280.0},
            )

        def step(
            self,
            fields: Mapping[str, Any],
            context: StepContext,
            payload: object | None = None,
        ) -> Mapping[str, Any]:
            _ = fields, context, payload
            return {}

    component = PlainComponent()
    coupler = Coupler(
        _clock(),
        components=(component,),
        run_order=("MODEL",),
    )

    info = coupler.components["MODEL"]

    assert not hasattr(vercor, "ComponentInfo")
    assert info is component
    assert info.name == "MODEL"
    assert info.grid.name == "component-info"
    assert info.spec.outputs == ("temperature",)
    assert isinstance(info, Component)
    with pytest.raises(TypeError):
        coupler.components["OTHER"] = info  # type: ignore[index]


@pytest.mark.fast_always
def test_snapshot_writer_receives_original_component(tmp_path: Path) -> None:
    grid = make_test_grid(name="snapshot-component-info")
    seen: list[Any] = []

    def writer(context: SnapshotContext) -> None:
        seen.append(context.component)

    component = CallableComponent(
        "MODEL",
        grid,
        lambda fields: {"temperature": fields["temperature"]},
        spec=ComponentSpec(
            inputs=("temperature",),
            outputs=("temperature",),
            initial_fields={"temperature": 280.0},
            output=OutputSpec(snapshot_writer=writer),
        ),
    )
    coupler = Coupler(
        _clock(),
        components=(component,),
        run_order=("MODEL",),
    )

    coupler.run(output=OutputTarget(tmp_path))

    assert len(seen) == 1
    assert seen[0].name == "MODEL"
    assert seen[0].grid.name == grid.name
    assert seen[0].grid.shape == grid.shape
    assert seen[0].spec is component.spec
    assert seen[0] is component
    assert not hasattr(vercor, "ComponentInfo")
