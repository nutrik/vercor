"""Exercise VerCOR extension points using only stable public modules."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import FrozenInstanceError, dataclass
from datetime import datetime
import json
from pathlib import Path
from typing import Any

import jax.numpy as jnp

from vercor import Clock
from vercor.components import (
    Component,
    ComponentMaskSpec,
    ComponentSpec,
    DataComponent,
    LifecycleHooks,
    SetupContext,
    SetupResult,
    StepContext,
    StepResult,
)
from vercor.coupler import Coupler
from vercor.exchanges import Exchange
from vercor.grids import RectilinearGrid
from vercor.output import OutputSpec, OutputTarget, PeriodOutput, SnapshotContext
from vercor.regridding import Regridder, RegridderFactory
from vercor.runtime import (
    ExecutionChunk,
    ExecutionContext,
    ExecutionPlan,
    RuntimeDriver,
    RuntimeOptions,
    StepPlan,
    WorkflowContext,
)
from vercor.state import RunState
from vercor.topology import (
    ExchangeTopologyPatch,
    TopologyContext,
)
from vercor.types import RuntimeArray


@dataclass(frozen=True)
class PluginConfig:
    """Plugin-owned immutable construction policy."""

    forcing: float = 1.0
    initial_temperature: float = 0.0
    steps: int = 2


@dataclass
class PluginRegridder:
    """Small structural scalar regridder implemented outside VerCOR."""

    source_grid: RectilinearGrid
    target_grid: RectilinearGrid

    @property
    def has_identical_grids(self) -> bool:
        """Return whether this regridder received the same grid object twice."""

        return self.source_grid is self.target_grid

    def regrid(self, field: RuntimeArray) -> RuntimeArray:
        """Broadcast the source mean over the target grid."""

        values = jnp.asarray(field)
        return jnp.full(self.target_grid.shape, jnp.mean(values), dtype=values.dtype)


class PluginRegridderFactory:
    """Injected factory recording construction of the plugin route."""

    def __init__(self, route_id: str) -> None:
        self.route_id = route_id
        self.calls: list[str] = []

    def __call__(
        self,
        source_grid: RectilinearGrid,
        target_grid: RectilinearGrid,
        **kwargs: Any,
    ) -> Regridder:
        """Return one structural plugin regridder."""

        _ = kwargs
        self.calls.append(self.route_id)
        return PluginRegridder(source_grid, target_grid)


def _setup_original_component(owner: Any, context: SetupContext) -> SetupResult:
    """Set up payload state while retaining the original plugin owner."""

    if not isinstance(owner, StructuralJaxComponent):
        raise TypeError("lifecycle hook did not receive the original component")
    owner.record_setup(context)
    owner.lifecycle_events.append("hook-setup")
    owner.lifecycle_owner_ids.append(id(owner))
    if context.run_order != ("FORCING", "JAX", "HOST"):
        raise ValueError("unexpected run order")
    return SetupResult(payload=jnp.asarray(0, dtype=jnp.int32))


def _write_snapshot(context: SnapshotContext) -> None:
    """Write one public snapshot-context payload for smoke verification."""

    value = float(jnp.asarray(context.state.field("temperature"))[0, 0])
    context.output_path.write_text(
        json.dumps(
            {
                "component": context.component.name,
                "temperature": value,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )


class StructuralJaxComponent:
    """Structural differentiable component implemented outside VerCOR."""

    name = "JAX"

    def __init__(self, config: PluginConfig) -> None:
        self.grid = RectilinearGrid.uniform(
            "plugin-jax-grid",
            nlon=4,
            nlat=3,
            longitude=(0.0, 360.0),
            latitude=(-90.0, 90.0),
        )
        self.spec = ComponentSpec(
            inputs=("forcing",),
            outputs=("temperature", "coverage"),
            initial_fields={
                "temperature": config.initial_temperature,
                "coverage": 0.25,
            },
            masks=ComponentMaskSpec(fractional_field="coverage"),
            lifecycle=LifecycleHooks(setup=_setup_original_component),
            output=OutputSpec(
                period=PeriodOutput(
                    frequency="step",
                    variables=("temperature",),
                ),
                snapshot_writer=_write_snapshot,
            ),
        )
        self.lifecycle_events: list[str] = []
        self.lifecycle_owner_ids: list[int] = []

    def record_setup(self, context: SetupContext) -> None:
        """Record structural user setup invoked by the lifecycle hook."""

        _ = context
        self.lifecycle_events.append("user-setup")

    def step(
        self,
        fields: Mapping[str, RuntimeArray],
        context: StepContext,
        payload: Any | None = None,
    ) -> StepResult:
        """Update fields and replace the public runtime payload."""

        _ = context
        if payload is None:
            raise ValueError("JAX component payload was not initialized")
        payload_value = jnp.asarray(payload, dtype=jnp.int32)
        return StepResult(
            fields={
                "temperature": fields["temperature"]
                + fields["forcing"]
                + payload_value,
                "coverage": fields["coverage"] + 0.125,
            },
            payload=payload_value + 1,
        )


class StructuralHostComponent:
    """Structural host component implemented outside VerCOR."""

    name = "HOST"

    def __init__(self) -> None:
        self.grid = RectilinearGrid.uniform(
            "plugin-host-grid",
            nlon=2,
            nlat=2,
            longitude=(0.0, 360.0),
            latitude=(-90.0, 90.0),
        )
        self.spec = ComponentSpec(
            outputs=("host_value",),
            initial_fields={"host_value": 10.0},
            execution="host",
        )

    def step(
        self,
        fields: Mapping[str, RuntimeArray],
        context: StepContext,
        payload: Any | None = None,
    ) -> Mapping[str, RuntimeArray]:
        """Advance the host field through the public mapping contract."""

        _ = context, payload
        return {"host_value": fields["host_value"] + 2.0}


class SequentialBackend:
    """Sequential custom backend using only the public runtime driver."""

    def __init__(self) -> None:
        self.calls = 0
        self.state_replacement = False

    def execute(
        self,
        state: RunState,
        *,
        context: ExecutionContext,
        chunk: ExecutionChunk,
        driver: RuntimeDriver,
    ) -> RunState:
        """Advance every plan in one core-defined chunk."""

        if not self.state_replacement:
            host_value = state.component("HOST").field("host_value")
            state = state.replace_fields(
                "HOST",
                {"host_value": jnp.full_like(host_value, 11.0)},
            )
            self.state_replacement = True
        self.calls += 1
        _ = context
        for plan in chunk.steps:
            state = driver.run_step(state, plan)
        return state


class PluginWorkflow:
    """Plugin-owned workflow using the constructor order for every step."""

    def __init__(self) -> None:
        self.events: list[str] = []

    def build(self, context: WorkflowContext) -> ExecutionPlan:
        """Build a complete static plan through public workflow contracts."""

        self.events.append("build")
        return ExecutionPlan(
            tuple(
                StepPlan(step=step, components=context.default_order)
                for step in range(context.clock.steps)
            )
        )


class RecordingTopologyPolicy:
    """Custom topology policy returning a non-empty route-ID patch."""

    def __init__(self, route_id: str) -> None:
        self.route_id = route_id
        self.events: list[str] = []
        self.patch_routes: tuple[str, ...] = ()

    def build(self, context: TopologyContext) -> ExchangeTopologyPatch:
        """Patch the configured route with an explicit target-shaped mask."""

        exchange = next(
            exchange
            for exchange in context.exchanges
            if exchange.route_id == self.route_id
        )
        target_shape = context.components[exchange.target].grid.shape
        self.events.append(f"build:{self.route_id}")
        self.patch_routes = (self.route_id,)
        return ExchangeTopologyPatch(
            binary_masks={self.route_id: jnp.ones(target_shape)}
        )


@dataclass(frozen=True)
class PluginAssembly:
    """Components and routes built from one plugin configuration."""

    components: tuple[Component, ...]
    exchanges: tuple[Exchange, ...]
    run_order: tuple[str, ...]


@dataclass(frozen=True)
class PluginFactory:
    """Compose plugin components with an injected regridder factory."""

    config: PluginConfig
    regridder_factory: RegridderFactory
    route_id: str = "plugin-forcing"

    def build(self) -> PluginAssembly:
        """Build a complete immutable plugin assembly."""

        jax_component = StructuralJaxComponent(self.config)
        host_component = StructuralHostComponent()
        forcing_component = DataComponent(
            "FORCING",
            RectilinearGrid.uniform(
                "plugin-forcing-grid",
                nlon=2,
                nlat=2,
                longitude=(0.0, 360.0),
                latitude=(-90.0, 90.0),
            ),
            {"forcing": self.config.forcing},
        )
        components: tuple[Component, ...] = (
            forcing_component,
            jax_component,
            host_component,
        )
        return PluginAssembly(
            components=components,
            exchanges=(
                Exchange(
                    "FORCING",
                    "JAX",
                    ("forcing",),
                    route_id=self.route_id,
                    regridder_factory=self.regridder_factory,
                    quantity="intensive",
                ),
            ),
            run_order=tuple(component.name for component in components),
        )


def run_smoke(output_dir: Path) -> dict[str, object]:
    """Run all required public extension points and return compact evidence."""

    config = PluginConfig(forcing=2.0, initial_temperature=3.0, steps=3)
    config_frozen = False
    try:
        setattr(config, "steps", 3)
    except FrozenInstanceError:
        config_frozen = True
    regridder_factory = PluginRegridderFactory("plugin-forcing")
    factory = PluginFactory(config, regridder_factory)
    assembly = factory.build()
    jax_component = assembly.components[1]
    backend = SequentialBackend()
    workflow = PluginWorkflow()
    topology = RecordingTopologyPolicy(factory.route_id)
    coupler = Coupler(
        Clock(start=datetime(2000, 1, 1), dt_seconds=60.0, steps=config.steps),
        components=assembly.components,
        exchanges=assembly.exchanges,
        run_order=assembly.run_order,
        runtime=RuntimeOptions(
            backend=backend,
            workflow=workflow,
            topology=topology,
        ),
    )

    final_state = coupler.run(output=OutputTarget(output_dir))

    temperature = float(
        jnp.asarray(final_state.component("JAX").field("temperature"))[0, 0]
    )
    host_value = float(
        jnp.asarray(final_state.component("HOST").field("host_value"))[0, 0]
    )
    exchange_forcing = float(
        jnp.asarray(final_state.component("JAX").field("forcing", scope="received"))[
            0, 0
        ]
    )
    snapshot_path = output_dir / "jax.snapshot.nc"
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))

    if not isinstance(jax_component, StructuralJaxComponent):
        raise AssertionError("plugin factory did not retain the structural component")
    if jax_component.lifecycle_events != ["user-setup", "hook-setup"]:
        raise AssertionError("structural lifecycle order was not preserved")
    if jax_component.lifecycle_owner_ids != [id(jax_component)]:
        raise AssertionError("lifecycle hook did not receive the original object")
    if backend.calls != config.steps:
        raise AssertionError("custom backend did not follow period-output chunks")
    if workflow.events != ["build"]:
        raise AssertionError("custom workflow was not invoked exactly once")
    if topology.events != [f"build:{factory.route_id}"]:
        raise AssertionError("custom topology policy was not applied")
    if temperature != 12.0 or host_value != 17.0 or exchange_forcing != 2.0:
        raise AssertionError("sequential backend produced unexpected fields")
    if not backend.state_replacement:
        raise AssertionError("public RunState.replace_fields was not exercised")

    period_files = tuple(
        path.name for path in sorted(output_dir.glob("jax.averages.*.nc"))
    )
    if len(period_files) != config.steps:
        raise AssertionError("period output did not write one file per step")

    return {
        "backend_calls": backend.calls,
        "coverage": float(jnp.mean(final_state.component("JAX").field("coverage"))),
        "config": {
            "forcing": config.forcing,
            "initial_temperature": config.initial_temperature,
            "steps": config.steps,
        },
        "config_frozen": config_frozen,
        "exchange_forcing": exchange_forcing,
        "factory": assembly.run_order,
        "host_value": host_value,
        "lifecycle": tuple(jax_component.lifecycle_events),
        "period_files": period_files,
        "regridder_calls": tuple(regridder_factory.calls),
        "snapshot": snapshot,
        "state_replacement": backend.state_replacement,
        "temperature": temperature,
        "topology": tuple(topology.events),
        "topology_patch_routes": topology.patch_routes,
        "workflow": tuple(workflow.events),
    }


__all__ = [
    "PluginConfig",
    "PluginFactory",
    "PluginRegridder",
    "PluginRegridderFactory",
    "PluginWorkflow",
    "RecordingTopologyPolicy",
    "SequentialBackend",
    "StructuralHostComponent",
    "StructuralJaxComponent",
    "run_smoke",
]
