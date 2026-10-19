from __future__ import annotations

from collections.abc import Mapping
from dataclasses import FrozenInstanceError
from datetime import datetime
import ast
import importlib
from pathlib import Path
import re
from types import SimpleNamespace
from typing import Any, cast

import jax.numpy as jnp
import numpy as np
import pytest
import xarray as xr

import vercor
from tests._coverage_support import make_test_grid
from tests.assertions import assert_allclose_compact
from vercor import Clock, Coupler, Exchange
from vercor.components import (
    CallableComponent,
    ComponentSpec,
    DataComponent,
    LifecycleHooks,
    PrefillContext,
    PrefillResult,
    SetupContext,
    SetupResult,
    StepContext,
    ValidationContext,
)
from vercor.exceptions import ComponentError, CouplerError
from vercor.runtime import (
    ExecutionChunk,
    ExecutionContext,
    RuntimeDriver,
    RuntimeOptions,
)
from vercor.state import ComponentState, RunState
from vercor.topology import ExchangeTopologyPatch, SurfaceMaskPolicy, TopologyContext

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PLUGIN_AUTHORING_GUIDE = PROJECT_ROOT / "docs" / "plugin-authoring.md"
STABLE_EXTENSION_MODULES = {
    "vercor.components",
    "vercor.coupler",
    "vercor.exchanges",
    "vercor.grids",
    "vercor.output",
    "vercor.physics",
    "vercor.regridding",
    "vercor.runtime",
    "vercor.state",
    "vercor.topology",
    "vercor.types",
}
STABLE_ROOT_EXPORTS = {
    "Clock",
    "Coupler",
    "Exchange",
    "RectilinearGrid",
    "RunState",
    "RuntimeOptions",
}


def _assert_stable_extension_imports(source: str, *, owner: str) -> None:
    """Reject imports outside the documented plugin compatibility tier."""

    tree = ast.parse(source, filename=owner)
    for node in ast.walk(tree):
        modules: list[str] = []
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            modules.append(node.module)
        for module in modules:
            if module != "vercor" and not module.startswith("vercor."):
                continue
            assert module == "vercor" or module in STABLE_EXTENSION_MODULES, module
            if isinstance(node, ast.ImportFrom) and module == "vercor":
                assert all(
                    alias.name in STABLE_ROOT_EXPORTS for alias in node.names
                ), module


@pytest.mark.parametrize(
    "source",
    (
        "from vercor.setups import make_slab_ocean",
        "from vercor import OutputSpec",
        "import vercor._runtime",
    ),
)
def test_plugin_authoring_import_policy_rejects_unstable_contracts(
    source: str,
) -> None:
    with pytest.raises(AssertionError):
        _assert_stable_extension_imports(source, owner="unstable-plugin.py")


@pytest.mark.fast_always
def test_plugin_authoring_guide_is_public_only_and_executable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Execute the guide's cumulative public extension example."""

    markdown = PLUGIN_AUTHORING_GUIDE.read_text(encoding="utf-8")
    for heading in (
        "Package and configuration",
        "Structural components and payload state",
        "Regridders and topology",
        "Workflows and execution backends",
        "Output providers",
        "Testing with fakes",
        "Installed example",
    ):
        assert f"## {heading}" in markdown

    snippets = tuple(re.findall(r"```python\n(.*?)```", markdown, flags=re.DOTALL))
    assert snippets
    namespace: dict[str, object] = {}
    monkeypatch.chdir(tmp_path)
    for index, snippet in enumerate(snippets):
        owner = f"plugin-authoring.md:{index}"
        _assert_stable_extension_imports(snippet, owner=owner)
        tree = ast.parse(snippet, filename=owner)
        exec(compile(tree, f"plugin-authoring.md:{index}", "exec"), namespace)

    config = cast(Any, namespace["guide_config"])
    factory = cast(Any, namespace["guide_factory"])
    assembly = cast(Any, namespace["guide_assembly"])
    assert factory.config is config
    assert assembly.run_order == ("FORCING", "MODEL")

    final_state = cast(RunState, namespace["guide_final_state"])
    assert_allclose_compact(
        final_state.component("MODEL").field("temperature"),
        jnp.full((2, 2), 3.0),
    )
    period_files = tuple((tmp_path / "guide-output").glob("*.averages.*.nc"))
    assert len(period_files) == 1
    with xr.open_dataset(period_files[0], engine="h5netcdf") as dataset:
        assert float(dataset["payload_offset"].values[0]) == pytest.approx(2.0)


@pytest.mark.fast_always
def test_runtime_options_own_core_runtime_configuration() -> None:
    runtime = RuntimeOptions(
        topology=SurfaceMaskPolicy(mode="disabled"),
    )

    assert runtime.topology == SurfaceMaskPolicy(mode="disabled")
    assert runtime.dtype.enable_x64 is False
    assert runtime.backend == "auto"
    assert not hasattr(runtime, "model_year_seconds")
    assert vercor.RuntimeOptions is RuntimeOptions
    assert "RuntimeOptions" in vercor.__all__
    assert "SurfaceMaskPolicy" not in vercor.__all__
    assert not hasattr(vercor, "SurfaceMaskPolicy")

    with pytest.raises(ModuleNotFoundError, match="vercor.config"):
        importlib.import_module("vercor.config")
    with pytest.raises(ModuleNotFoundError, match="vercor.setup_config"):
        importlib.import_module("vercor.setup_config")


class _RecordingTopologyPolicy:
    def __init__(
        self,
        *,
        patch: ExchangeTopologyPatch | None = None,
    ) -> None:
        self._patch = ExchangeTopologyPatch() if patch is None else patch
        self.events: list[tuple[str, TopologyContext]] = []

    def build(self, context: TopologyContext) -> ExchangeTopologyPatch:
        self.events.append(("build", context))
        return self._patch


def _topology_policy_coupler(policy: _RecordingTopologyPolicy) -> Coupler:
    grid = make_test_grid(name="topology-policy")
    source = DataComponent(
        "SRC",
        grid,
        {"custom_flux": 1.0},
    )
    target = DataComponent(
        "DST",
        grid,
        {"custom_flux": 0.0},
        spec=ComponentSpec(inputs=("custom_flux",)),
    )
    return Coupler(
        Clock(start=datetime(2000, 1, 1), dt_seconds=60.0, steps=1),
        components=(source, target),
        exchanges=(Exchange("SRC", "DST", ("custom_flux",), quantity="intensive"),),
        run_order=("SRC", "DST"),
        runtime=RuntimeOptions(topology=policy),
    )


@pytest.mark.fast_always
def test_custom_topology_policy_builds_once_and_patches_route_maps() -> None:
    key = "SRC->DST"
    for restriction in (None, jnp.array([[0.0, 1.0], [1.0, 0.0]])):
        policy = _RecordingTopologyPolicy(
            patch=ExchangeTopologyPatch(
                binary_masks={} if restriction is None else {key: restriction}
            )
        )
        coupler = _topology_policy_coupler(policy)
        initial = coupler.initial_state()
        assert [event for event, _ in policy.events] == ["build"]
        expected = jnp.ones((2, 2)) if restriction is None else restriction
        assert_allclose_compact(initial._fractional_mask(key), expected)
        final = coupler.run(initial)
        assert_allclose_compact(
            final.component("DST").field("custom_flux", scope="received"), expected
        )


@pytest.mark.fast_always
@pytest.mark.parametrize(
    "key,value,message",
    [
        ("UNKNOWN", jnp.ones((2, 2)), "UNKNOWN.*configured route ID"),
        ("SRC->DST", jnp.ones((1, 2)), r"SRC->DST.*shape \(1, 2\).*expected \(2, 2\)"),
        ("SRC->DST", np.full((2, 2), "land"), "binary.*concrete numeric or bool"),
        (
            "SRC->DST",
            np.full((2, 2), object(), dtype=object),
            "binary.*concrete numeric or bool",
        ),
        ("SRC->DST", np.array([[0.0, 1.0], [np.nan, 0.0]]), "binary.*finite"),
        ("SRC->DST", np.array([[0.0, 1.0], [np.inf, 0.5]]), "binary.*finite"),
        ("SRC->DST", np.array([[0.0, 1.0], [0.5, 0.0]]), r"binary.*\{0, 1\}"),
        ("SRC->DST", np.array([[0.0, 1.0], [-0.01, 0.5]]), r"binary.*\{0, 1\}"),
        ("SRC->DST", np.array([[0.0, 1.01], [0.25, 0.5]]), r"binary.*\{0, 1\}"),
    ],
)
def test_topology_policy_patch_rejects_invalid_restrictions(
    key: str, value: Any, message: str
) -> None:
    coupler = _topology_policy_coupler(
        _RecordingTopologyPolicy(patch=ExchangeTopologyPatch(binary_masks={key: value}))
    )
    with pytest.raises(CouplerError, match=message):
        coupler.initial_state()


@pytest.mark.fast_always
@pytest.mark.parametrize(
    "binary_mask",
    [
        np.array([[False, True], [True, False]]),
        np.array([[0, 1], [1, 0]], dtype=np.int32),
    ],
)
def test_topology_policy_patch_accepts_binary_restriction_types(
    binary_mask: np.ndarray,
) -> None:
    coupler = _topology_policy_coupler(
        _RecordingTopologyPolicy(
            patch=ExchangeTopologyPatch(binary_masks={"SRC->DST": binary_mask})
        )
    )
    state = coupler.initial_state()
    assert_allclose_compact(state._fractional_mask("SRC->DST"), binary_mask)


@pytest.mark.fast_always
def test_component_spec_freezes_mapping_inputs_and_exposes_execution_policy() -> None:
    initial_fields: dict[str, object] = {"temperature": 280.0}
    spec = ComponentSpec(
        inputs=("temperature", "temperature"),
        outputs=("heat_flux",),
        initial_fields=initial_fields,
        execution="host",
    )
    initial_fields["temperature"] = 999.0

    assert spec.inputs == ("temperature",)
    assert spec.initial_fields["temperature"] == 280.0
    assert spec.execution == "host"
    assert not hasattr(spec, "import_policy")
    with pytest.raises(TypeError):
        spec.initial_fields["temperature"] = 281.0  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        spec.execution = "jax"  # type: ignore[misc]

    grid = make_test_grid(name="execution-precedence")
    capable_host = CallableComponent(
        "CAPABLE_HOST",
        grid,
        lambda fields: fields,
        spec=ComponentSpec(execution="host"),
    )
    explicit_jax = CallableComponent(
        "ENFORCED_HOST",
        grid,
        lambda fields: fields,
        spec=ComponentSpec(execution="jax"),
    )

    assert capable_host.spec.execution == "host"
    assert explicit_jax.spec.execution == "jax"


@pytest.mark.fast_always
def test_structural_component_like_runs_without_private_component_internals() -> None:
    class PlainComponent:
        name = "MODEL"

        def __init__(self) -> None:
            self.grid = make_test_grid(name="plain-component")

            def setup(owner: object, context: SetupContext) -> None:
                assert owner is self
                assert tuple(context.run_order) == ("MODEL",)

            self.spec = ComponentSpec(
                inputs=("heat_flux",),
                outputs=("temperature",),
                initial_fields={"temperature": 280.0, "heat_flux": 1.5},
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
                "temperature": fields["temperature"]
                + fields["heat_flux"]
                * jnp.asarray(context.dt_seconds)
                * jnp.asarray(context.step + 1)
            }

    component = PlainComponent()
    coupler = Coupler(
        Clock(start=datetime(2000, 1, 1), dt_seconds=2.0, steps=2),
        components=(component,),
        run_order=("MODEL",),
        runtime=RuntimeOptions(topology=None),
    )

    final_state = coupler.run()

    assert not hasattr(component, "_data")
    assert_allclose_compact(
        final_state.component("MODEL").field("temperature"),
        jnp.full(component.grid.shape, 289.0),
    )


@pytest.mark.fast_always
@pytest.mark.parametrize(
    ("override", "message"),
    (
        ({"name": ""}, "name.*non-empty string"),
        ({"name": "   "}, "name.*non-empty string"),
        ({"name": 7}, "name.*non-empty string"),
        ({"grid": object()}, "grid.*RectilinearGrid"),
        ({"spec": object()}, "spec.*ComponentSpec"),
        ({"step": None}, "step.*callable"),
    ),
)
def test_structural_component_validation_is_actionable(
    override: dict[str, object],
    message: str,
) -> None:
    candidate = SimpleNamespace(
        name="MODEL",
        grid=make_test_grid(name="invalid-plain-component"),
        spec=ComponentSpec(),
        step=lambda fields, context, payload=None: {},
    )
    for attribute, value in override.items():
        setattr(candidate, attribute, value)

    with pytest.raises(CouplerError, match=message):
        Coupler(
            Clock(start=datetime(2000, 1, 1), dt_seconds=60.0, steps=1),
            components=(candidate,),
            run_order=("MODEL",),
            runtime=RuntimeOptions(topology=None),
        )


@pytest.mark.fast_always
@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("name", "name.*non-empty string"),
        ("grid", "grid.*RectilinearGrid"),
        ("spec", "spec.*ComponentSpec"),
        ("step", "step.*callable"),
    ),
)
def test_vercor_component_validation_uses_the_structural_contract_path(
    mutation: str,
    message: str,
) -> None:
    component = CallableComponent(
        "MODEL",
        make_test_grid(name="invalid-vercor-component"),
        lambda fields: fields,
    )
    if mutation == "name":
        component.name = "   "
    elif mutation == "grid":
        component.grid = cast(Any, object())
    elif mutation == "spec":
        component.spec = cast(Any, object())
    else:
        setattr(component, mutation, None)

    with pytest.raises(CouplerError, match=message):
        Coupler(
            Clock(start=datetime(2000, 1, 1), dt_seconds=60.0, steps=1),
            components=(component,),
            run_order=("MODEL",),
            runtime=RuntimeOptions(topology=None),
        )


@pytest.mark.fast_always
@pytest.mark.parametrize("attribute", ("name", "grid", "spec"))
def test_component_setup_cannot_change_static_identity(attribute: str) -> None:
    class InvalidatingComponent:
        def __init__(self) -> None:
            self.name = "MODEL"
            self.grid = make_test_grid(name="identity-mutation")

            def setup(owner: Any, context: SetupContext) -> None:
                _ = context
                replacements = {
                    "name": "CHANGED",
                    "grid": make_test_grid(name="changed-grid"),
                    "spec": ComponentSpec(),
                }
                setattr(owner, attribute, replacements[attribute])

            self.spec = ComponentSpec(lifecycle=LifecycleHooks(setup=setup))

        def step(
            self,
            fields: Mapping[str, Any],
            context: StepContext,
            payload: object | None = None,
        ) -> Mapping[str, Any]:
            _ = context, payload
            return fields

    component = InvalidatingComponent()
    coupler = Coupler(
        Clock(datetime(2000, 1, 1), dt_seconds=60.0, steps=1),
        components=(component,),
        run_order=("MODEL",),
    )

    with pytest.raises(ComponentError, match=rf"MODEL.*{attribute}.*during setup"):
        coupler.initial_state()


@pytest.mark.fast_always
@pytest.mark.parametrize("phase", ("prefill", "validate"))
@pytest.mark.parametrize("attribute", ("name", "grid", "spec"))
def test_component_runtime_lifecycle_cannot_change_static_identity(
    phase: str,
    attribute: str,
) -> None:
    class InvalidatingComponent:
        def __init__(self) -> None:
            self.name = "MODEL"
            self.grid = make_test_grid(name=f"{phase}-identity-mutation")

            def mutate_identity(owner: Any) -> None:
                replacements = {
                    "name": "CHANGED",
                    "grid": make_test_grid(name=f"changed-{phase}-grid"),
                    "spec": ComponentSpec(),
                }
                setattr(owner, attribute, replacements[attribute])

            def prefill_identity(
                owner: Any,
                context: PrefillContext,
            ) -> PrefillResult:
                _ = context
                mutate_identity(owner)
                return PrefillResult()

            def validate_identity(owner: Any, context: ValidationContext) -> None:
                _ = context
                mutate_identity(owner)

            self.spec = ComponentSpec(
                lifecycle=LifecycleHooks(
                    prefill=prefill_identity if phase == "prefill" else None,
                    validate=validate_identity if phase == "validate" else None,
                )
            )

        def step(
            self,
            fields: Mapping[str, Any],
            context: StepContext,
            payload: object | None = None,
        ) -> Mapping[str, Any]:
            _ = context, payload
            return fields

    component = InvalidatingComponent()
    coupler = Coupler(
        Clock(datetime(2000, 1, 1), dt_seconds=60.0, steps=1),
        components=(component,),
        run_order=("MODEL",),
    )

    with pytest.raises(
        ComponentError,
        match=rf"MODEL.*{attribute}.*during {phase}",
    ):
        coupler.initial_state()

    assert tuple(coupler.components) == ("MODEL",)
    assert coupler._prepared is not None
    assert tuple(coupler._prepared.components) == ("MODEL",)


@pytest.mark.fast_always
def test_structural_lifecycle_hooks_receive_original_object_in_order() -> None:
    events: list[str] = []
    hook_owners: list[object] = []

    def setup_hook(owner: object, context: SetupContext) -> SetupResult:
        hook_owners.append(owner)
        events.append("setup")
        assert context.run_order == ("MODEL",)
        return SetupResult(
            fields={"temperature": 282.0},
            payload={"owner": owner.name},  # type: ignore[attr-defined]
        )

    def prefill_hook(
        owner: object,
        context: PrefillContext,
    ) -> PrefillResult:
        hook_owners.append(owner)
        events.append("prefill")
        assert float(jnp.mean(context.fields["temperature"])) == 282.0
        return PrefillResult()

    def validate_hook(
        owner: object,
        context: ValidationContext,
    ) -> None:
        hook_owners.append(owner)
        events.append("validate")
        assert isinstance(context.state, ComponentState)
        assert context.payload == {"owner": "MODEL"}

    class PlainLifecycleComponent:
        name = "MODEL"

        def __init__(self) -> None:
            self.grid = make_test_grid(name="plain-lifecycle")
            self.spec = ComponentSpec(
                outputs=("temperature",),
                initial_fields={"temperature": 280.0},
                lifecycle=LifecycleHooks(
                    setup=setup_hook,
                    prefill=prefill_hook,
                    validate=validate_hook,
                ),
            )

        def step(
            self,
            fields: Mapping[str, Any],
            context: StepContext,
            payload: object | None = None,
        ) -> Mapping[str, Any]:
            _ = context, payload
            return {"temperature": fields["temperature"]}

    component = PlainLifecycleComponent()
    coupler = Coupler(
        Clock(start=datetime(2000, 1, 1), dt_seconds=60.0, steps=1),
        components=(component,),
        run_order=("MODEL",),
        runtime=RuntimeOptions(topology=None),
    )

    state = coupler.initial_state()

    assert_allclose_compact(
        state.component("MODEL").field("temperature"),
        jnp.full(component.grid.shape, 282.0),
    )
    assert events == [
        "setup",
        "prefill",
        "validate",
    ]
    assert hook_owners == [component, component, component]


@pytest.mark.fast_always
def test_constructor_builds_coupler_from_plain_recipe() -> None:
    grid = make_test_grid(name="coupler-spec")
    forcing = DataComponent("SRC", grid, {"flux": 2.0})
    model = CallableComponent(
        "DST",
        grid,
        lambda fields: {"flux": fields["flux"] + 1.0},
        spec=ComponentSpec(
            inputs=("flux",),
            outputs=("flux",),
            initial_fields={"flux": 0.0},
        ),
    )
    coupler = Coupler(
        Clock(start=datetime(2000, 1, 1), dt_seconds=60.0, steps=1),
        components=(forcing, model),
        exchanges=(Exchange("SRC", "DST", ("flux",), quantity="intensive"),),
        run_order=("SRC", "DST"),
        runtime=RuntimeOptions(topology=None),
    )

    assert isinstance(coupler, Coupler)
    assert coupler.runtime.topology is None
    assert_allclose_compact(
        coupler.run().component("DST").field("flux"),
        jnp.full(grid.shape, 3.0),
    )


@pytest.mark.fast_always
def test_runtime_options_accept_custom_execution_backend() -> None:
    class RecordingBackend:
        def __init__(self) -> None:
            self.calls = 0

        def execute(
            self,
            state: RunState,
            *,
            context: ExecutionContext,
            chunk: ExecutionChunk,
            driver: RuntimeDriver,
        ) -> RunState:
            self.calls += 1
            assert context.component_names == ("MODEL",)
            for plan in chunk.steps:
                state = driver.run_step(state, plan)
            return state.replace_fields(
                "MODEL",
                {"temperature": jnp.full(grid.shape, 301.0)},
            )

    grid = make_test_grid(name="custom-backend")
    backend = RecordingBackend()
    component = DataComponent(
        "MODEL",
        grid,
        {"temperature": 280.0},
    )
    coupler = Coupler(
        Clock(start=datetime(2000, 1, 1), dt_seconds=60.0, steps=1),
        components=(component,),
        run_order=("MODEL",),
        runtime=RuntimeOptions(topology=None, backend=backend),
    )

    final_state = coupler.run()

    assert backend.calls == 1
    assert_allclose_compact(
        final_state.component("MODEL").field("temperature"),
        jnp.full(grid.shape, 301.0),
    )


@pytest.mark.fast_always
@pytest.mark.parametrize(
    "step",
    (
        lambda fields: {},
        lambda fields, context: {},
        lambda fields, context, payload, extra: {},
        lambda fields, context, *, payload: {},
    ),
)
def test_structural_step_requires_protocol_signature_at_preparation(step: Any) -> None:
    candidate = SimpleNamespace(
        name="MODEL",
        grid=make_test_grid(name="invalid-structural-step"),
        spec=ComponentSpec(),
        step=step,
    )
    coupler = Coupler(
        Clock(start=datetime(2000, 1, 1), dt_seconds=60.0, steps=1),
        components=(candidate,),
        run_order=("MODEL",),
        runtime=RuntimeOptions(topology=None),
    )
    with pytest.raises(CouplerError, match="MODEL.*step.*fields.*context.*payload"):
        coupler.initial_state()


@pytest.mark.fast_always
def test_structural_step_ignores_private_callable_convenience_attributes() -> None:
    candidate = SimpleNamespace(
        name="MODEL",
        grid=make_test_grid(name="structural-direct-step"),
        spec=ComponentSpec(outputs=("value",)),
        step=lambda fields, context, payload: {"value": jnp.full((2, 2), 7.0)},
        _normalized_step=lambda fields, context, payload: {
            "value": jnp.full((2, 2), 99.0)
        },
    )
    coupler = Coupler(
        Clock(start=datetime(2000, 1, 1), dt_seconds=60.0, steps=1),
        components=(candidate,),
        run_order=("MODEL",),
        runtime=RuntimeOptions(topology=None),
    )
    assert_allclose_compact(
        coupler.run().component("MODEL").field("value"), jnp.full((2, 2), 7.0)
    )
