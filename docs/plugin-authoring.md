# Authoring VerCOR 0.4 plugins

VerCOR `0.4.0` provides a stable extension tier: the six-symbol package root
plus the public owner modules used below. Plugins are ordinary Python packages:
construct structural objects and inject them into `Coupler`. There is no
registry or entry-point discovery.

The Python blocks form one executable example in source order. They import only
the documented public extension tier.

## Package and configuration

Declare the stable floor while accepting compatible `0.4` releases:

```toml
[project]
dependencies = ["vercor>=0.4.0,<0.5"]
```

Keep plugin-owned construction policy immutable and separate from VerCOR's
runtime policy:

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class GuideConfig:
    forcing: float = 2.0
    initial_temperature: float = 0.0
    steps: int = 1
```

## Structural components and payload state

A component implements `name`, `grid`, `spec`, and `step`; inheritance is not
required. Evolving model state is returned as payload instead of being retained
on the author object.

```python
from collections.abc import Mapping
from typing import Any

import jax.numpy as jnp

from vercor import RectilinearGrid
from vercor.components import (
    ComponentSpec,
    LifecycleHooks,
    SetupContext,
    SetupResult,
    StepContext,
    StepResult,
)
from vercor.output import OutputSpec
from vercor.types import RuntimeArray


def guide_setup(component: Any, context: SetupContext) -> SetupResult:
    _ = component, context
    return SetupResult(payload=jnp.asarray(1, dtype=jnp.int32))


class GuideModel:
    name = "MODEL"

    def __init__(
        self,
        grid: RectilinearGrid,
        config: GuideConfig,
        output: OutputSpec | None = None,
    ) -> None:
        self.grid = grid
        self.spec = ComponentSpec(
            inputs=("forcing",),
            outputs=("temperature",),
            initial_fields={"temperature": config.initial_temperature},
            lifecycle=LifecycleHooks(setup=guide_setup),
            output=output,
        )

    def step(
        self,
        fields: Mapping[str, RuntimeArray],
        context: StepContext,
        payload: Any | None = None,
    ) -> StepResult:
        _ = context
        if payload is None:
            raise ValueError("GuideModel payload was not initialized")
        offset = jnp.asarray(payload, dtype=jnp.int32)
        return StepResult(
            fields={
                "temperature": fields["temperature"]
                + fields["forcing"]
                + offset
            },
            payload=offset + 1,
        )
```

## Regridders and topology

Inject a callable factory on each route. Topology patches are keyed by the
route's stable `route_id`, never by source/target name guessing.

```python
from vercor.regridding import Regridder
from vercor.topology import ExchangeTopologyPatch, TopologyContext


@dataclass
class GuideRegridder:
    source_grid: RectilinearGrid
    target_grid: RectilinearGrid

    @property
    def has_identical_grids(self) -> bool:
        return self.source_grid is self.target_grid

    def regrid(self, field: RuntimeArray) -> RuntimeArray:
        values = jnp.asarray(field)
        return jnp.full(self.target_grid.shape, jnp.mean(values), dtype=values.dtype)


class GuideRegridderFactory:
    def __call__(
        self,
        source_grid: RectilinearGrid,
        target_grid: RectilinearGrid,
    ) -> Regridder:
        return GuideRegridder(source_grid, target_grid)


class GuideTopology:
    def __init__(self, route_id: str) -> None:
        self.route_id = route_id

    def build(self, context: TopologyContext) -> ExchangeTopologyPatch:
        exchange = next(
            route for route in context.exchanges if route.route_id == self.route_id
        )
        shape = context.components[exchange.target].grid.shape
        return ExchangeTopologyPatch(
            binary_masks={self.route_id: jnp.ones(shape)}
        )
```

## Workflows and execution backends

Workflows author a complete plan. Backends consume each core-supplied plan once
and in order through `RuntimeDriver`.

```python
from vercor.runtime import (
    ExecutionChunk,
    ExecutionContext,
    ExecutionPlan,
    RuntimeDriver,
    StepPlan,
    WorkflowContext,
)
from vercor.state import RunState


class GuideWorkflow:
    def build(self, context: WorkflowContext) -> ExecutionPlan:
        return ExecutionPlan(
            tuple(
                StepPlan(step=step, components=context.default_order)
                for step in range(context.clock.steps)
            )
        )


class GuideBackend:
    def execute(
        self,
        state: RunState,
        *,
        context: ExecutionContext,
        chunk: ExecutionChunk,
        driver: RuntimeDriver,
    ) -> RunState:
        _ = context
        for plan in chunk.steps:
            state = driver.run_step(state, plan)
        return state
```

## Output providers

Providers sample post-step component state. They return immutable frames;
VerCOR owns cadence, host transfer, filenames, and NetCDF writing.

```python
from vercor.output import (
    OutputContext,
    OutputFrame,
    OutputTarget,
    OutputVariable,
    PeriodOutput,
)


class GuideProvider:
    def sample(self, context: OutputContext) -> OutputFrame:
        if context.payload is None:
            raise ValueError("GuideModel output payload is missing")
        return OutputFrame(
            {
                "temperature": OutputVariable(
                    ("nlat", "nlon"),
                    context.state.field("temperature"),
                    {"units": "K"},
                ),
                "payload_offset": OutputVariable(
                    (),
                    jnp.asarray(context.payload, dtype=jnp.float32),
                    {"long_name": "runtime payload offset"},
                ),
            }
        )
```

## Testing with fakes

Test component kernels directly with a small public context before composing
the integration graph. This catches field and payload mistakes cheaply.

```python
fake_grid = RectilinearGrid.uniform(
    "fake",
    nlon=2,
    nlat=2,
    longitude=(0.0, 360.0),
    latitude=(-90.0, 90.0),
)
fake_model = GuideModel(fake_grid, GuideConfig())
fake_result = fake_model.step(
    {
        "temperature": jnp.zeros(fake_grid.shape),
        "forcing": jnp.full(fake_grid.shape, 2.0),
    },
    StepContext(dt_seconds=60.0, step=0),
    jnp.asarray(1, dtype=jnp.int32),
)
assert bool(jnp.all(jnp.asarray(fake_result.fields["temperature"]) == 3.0))
```

## Installed example

The final assembly uses only public contracts. The repository's independently
built `tests/fixtures/external_extension_test_fixture` wheel runs this same
class of extension outside the checkout and protects the documented extension
tier.

```python
from datetime import datetime

from vercor import Clock, Coupler, Exchange
from vercor.components import Component, DataComponent
from vercor.runtime import RuntimeOptions


@dataclass(frozen=True)
class GuideAssembly:
    components: tuple[Component, ...]
    exchanges: tuple[Exchange, ...]
    run_order: tuple[str, ...]


@dataclass(frozen=True)
class GuideFactory:
    config: GuideConfig
    regridder_factory: GuideRegridderFactory
    route_id: str = "guide-forcing"

    def build(
        self,
        grid: RectilinearGrid,
        output: OutputSpec,
    ) -> GuideAssembly:
        source = DataComponent(
            "FORCING", grid, {"forcing": self.config.forcing}
        )
        model = GuideModel(grid, self.config, output)
        return GuideAssembly(
            components=(source, model),
            exchanges=(
                Exchange(
                    "FORCING",
                    "MODEL",
                    ("forcing",),
                    quantity="intensive",
                    route_id=self.route_id,
                    regridder_factory=self.regridder_factory,
                ),
            ),
            run_order=("FORCING", "MODEL"),
        )


guide_config = GuideConfig()
guide_grid = RectilinearGrid.uniform(
    "guide",
    nlon=2,
    nlat=2,
    longitude=(0.0, 360.0),
    latitude=(-90.0, 90.0),
)
guide_factory = GuideFactory(guide_config, GuideRegridderFactory())
guide_assembly = guide_factory.build(
    guide_grid,
    OutputSpec(
        provider=GuideProvider(),
        period=PeriodOutput(
            frequency="step",
            variables=("temperature", "payload_offset"),
        ),
    ),
)
guide_coupler = Coupler(
    Clock(datetime(2000, 1, 1), dt_seconds=60.0, steps=guide_config.steps),
    components=guide_assembly.components,
    exchanges=guide_assembly.exchanges,
    run_order=guide_assembly.run_order,
    runtime=RuntimeOptions(
        backend=GuideBackend(),
        workflow=GuideWorkflow(),
        topology=GuideTopology(guide_factory.route_id),
    ),
)
guide_final_state = guide_coupler.run(
    output=OutputTarget(
        "guide-output",
        write_final_fields=False,
        write_snapshots=False,
    )
)
```

For pure JAX differentiation or outer JIT compilation, call `run(output=None)`
and keep file output outside the transformed function.
