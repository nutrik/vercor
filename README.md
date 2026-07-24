# VerCOR

[![codecov](https://codecov.io/github/nutrik/vercor/graph/badge.svg?token=A960MY4GXH)](https://codecov.io/github/nutrik/vercor)

Versatile Earth system COupleR (VerCOR) connects atmosphere, ocean, sea-ice,
land, and forcing-data components in one simulation. It is designed for Earth
system researchers who want to combine models, move fields between their grids,
run them on a shared clock, and collect diagnostics and output.

VerCOR is built around [JAX](https://github.com/jax-ml/jax), a numerical
computing library that supports compilation and automatic differentiation.
Output-free JAX workflows can be differentiated end to end, making VerCOR
suitable for sensitivity analysis and gradient-based experiments.

> **Stable release:** Version `0.4.1` is the current release. VerCOR 0.3
> applications must use the
> [0.3-to-0.4 migration guide](docs/migration-0.3-to-0.4.md).

## Key features

- Combine JAX-based and host-side models in one coupled simulation.
- Exchange scalar and vector fields between rectilinear grids.
- Use bilinear or conservative regridding.
- Run on Gregorian, no-leap, or 360-day calendars.
- Use bundled slab atmosphere, ocean, land, and sea-ice components.
- Connect optional JCM, Veros, and CAMulator models.
- Supply static or time-dependent forcing data.
- Keep simulation state immutable and compatible with JAX transformations.
- Write period averages, final fields, and component snapshots when requested.
- Add custom components without inheriting from a VerCOR base class.

## Requirements

- Python 3.12 or 3.13
- A JAX installation supported by your platform

The core package also depends on NumPy, SciPy, h5py, h5netcdf, xarray,
Matplotlib, `jax-datetime`, and `tree-math`. These dependencies are installed
with the package.

JCM and Veros are optional. CAMulator additionally requires NCAR's
[MILES-CREDIT](https://github.com/NCAR/miles-credit). A compatible CREDIT
release has not yet been confirmed or pinned.

## Installation

Install the current stable core package with pip:

```bash
python -m pip install "vercor==0.4.1"
```

Install the relevant optional extra before using a bundled JCM or Veros setup:

```bash
python -m pip install "vercor[jcm]"
python -m pip install "vercor[veros]"
```

The repository also defines optional environments for tests and development
tools:

```bash
python -m pip install "vercor[test]"
python -m pip install "vercor[dev]"
```

No CAMulator installation command is documented yet. Install MILES-CREDIT
separately before using the CAMulator factories.

## Getting started

This example runs a small slab-ocean model for two one-hour steps. A slab model
is a simplified model that represents the ocean as a single mixed layer.

```python
from datetime import datetime

from vercor import Clock, Coupler, RectilinearGrid
from vercor.setups import make_slab_ocean

grid = RectilinearGrid.uniform(
    "demo",
    nlon=2,
    nlat=2,
    longitude=(0.0, 360.0),
    latitude=(-90.0, 90.0),
)

clock = Clock(
    start=datetime(2000, 1, 1),
    dt_seconds=3600.0,
    steps=2,
)

ocean = make_slab_ocean(grid)
coupler = Coupler(clock, components=(ocean,), run_order=("OCN",))
final_state = coupler.run()

sea_surface_temperature = final_state.component("OCN").field(
    "sea_surface_temperature"
)
```

The main objects are:

- `RectilinearGrid`: the longitude-latitude grid used by a component.
- `Clock`: the start time, step length, number of steps, and calendar.
- `Coupler`: the complete simulation, including components, exchanges, and run
  order.
- `RunState`: the immutable state returned by a run.

Create a new `Coupler` when you need different components, exchanges, or runtime
settings. An empty `run_order=()` is valid when you only want setup, validation,
state creation, and output preparation without advancing a component.

## Common usage examples

### Create a custom JAX component

A component only needs a name, grid, component specification, and `step`
method. `ComponentSpec` declares its fields and starting values.

```python
from collections.abc import Mapping
from typing import Any

from vercor.components import ComponentSpec, StepContext
from vercor.types import RuntimeArray


class WarmingModel:
    name = "MODEL"

    def __init__(self, model_grid: RectilinearGrid) -> None:
        self.grid = model_grid
        self.spec = ComponentSpec(
            outputs=("temperature",),
            initial_fields={"temperature": 280.0},
        )

    def step(
        self,
        fields: Mapping[str, RuntimeArray],
        context: StepContext,
        payload: Any | None = None,
    ) -> Mapping[str, RuntimeArray]:
        _ = context, payload
        return {"temperature": fields["temperature"] + 0.25}


model = WarmingModel(grid)
custom_coupler = Coupler(
    clock,
    components=(model,),
    run_order=(model.name,),
)
custom_state = custom_coupler.run()
```

No VerCOR inheritance is required. For a function-based component with less
boilerplate, use `vercor.components.CallableComponent`.

For a complete independently packaged extension, including payload state,
regridders, topology, workflows, backends, and output providers, follow the
[plugin authoring guide](docs/plugin-authoring.md).

Lifecycle hooks can prepare component fields and a payload before the run.
`TransferPolicy` selects current, linearly interpolated, or daily source data.
When a JAX-compiled component replaces its payload, the replacement must keep
the same JAX tree structure, shapes, and data types. Host-side components may
clear or restructure payload data.

### Run a host-side component

Use host execution for a Python model or a model backed by another runtime.
With `backend="auto"`, VerCOR selects the host loop when a scheduled component
declares `execution="host"`.

```python
from vercor import RuntimeOptions
from vercor.components import CallableComponent


def host_step(
    fields: Mapping[str, RuntimeArray],
) -> Mapping[str, RuntimeArray]:
    return {"counter": fields["counter"] + 1.0}


host = CallableComponent(
    "HOST",
    grid,
    host_step,
    spec=ComponentSpec(
        outputs=("counter",),
        initial_fields={"counter": 0.0},
        execution="host",
    ),
)

host_coupler = Coupler(
    clock,
    components=(host,),
    run_order=("HOST",),
    runtime=RuntimeOptions(backend="auto"),
)
host_state = host_coupler.run()
```

Forcing `backend="jax"` rejects host components. Forcing `backend="host"` runs
every scheduled component through the Python loop.

### Couple components on different grids

An `Exchange` moves named fields from one component to another. A regridder
translates the data when the source and target grids differ.

```python
from vercor import Exchange
from vercor.regridding import bilinear

exchange = Exchange(
    source="ATM",
    target="OCN",
    fields=("surface_temperature",),
    regridder_factory=bilinear,
)
```

The field must be declared by both endpoints. Each route has a stable
`route_id`; the default is `"source->target"`. Supply distinct IDs when more
than one route connects the same pair. VerCOR rejects ambiguous cases where
multiple routes write the same target field.

Use `vercor.topology.SurfaceMaskPolicy()` for the bundled atmosphere/ocean/land
surface-mask policy. Leave `RuntimeOptions.topology` as `None` for an ordinary
setup without topology patches.

### Write output

Output is opt-in. Passing an `OutputTarget` enables period files, final fields,
and component snapshots beneath one directory:

```python
from pathlib import Path

from vercor.output import OutputSpec, OutputTarget, PeriodOutput, SnapshotContext


def snapshot_writer(context: SnapshotContext) -> None:
    value = context.state.field("temperature")
    context.output_path.write_text(str(value), encoding="utf-8")


def output_step(
    fields: Mapping[str, RuntimeArray],
) -> Mapping[str, RuntimeArray]:
    return {"temperature": fields["temperature"] + 1.0}


output_model = CallableComponent(
    "OUTPUT",
    grid,
    output_step,
    spec=ComponentSpec(
        outputs=("temperature",),
        initial_fields={"temperature": 280.0},
        output=OutputSpec(
            period=PeriodOutput(frequency="step"),
            snapshot_writer=snapshot_writer,
        ),
    ),
)
output_coupler = Coupler(
    clock,
    components=(output_model,),
    run_order=(output_model.name,),
)

output_directory = Path("output")
output_state = output_coupler.run(output=OutputTarget(output_directory))
```

Disable individual output types when needed:

```python
selected_output_state = output_coupler.run(
    output=OutputTarget(
        output_directory,
        write_period=False,
        write_final_fields=True,
        write_snapshots=False,
    )
)
```

Components configure output with `OutputSpec`. `PeriodOutput` controls the
sampling frequency and selected variables. An empty variable list selects all
variables supplied by the component's output provider; an unknown name is an
error. Providers sample the post-step state at the end-of-step model time.

Bundled slab and data factories default to `OutputSpec()`, so omitting
`output` does not schedule period files or component snapshots. Pass a complete
`OutputSpec` through a factory's keyword-only `output` argument to configure
one component independently:

```python
from vercor.output import OutputSpec, PeriodOutput
from vercor.setups import make_slab_ocean

monthly_ocean = make_slab_ocean(
    grid,
    output=OutputSpec(
        period=PeriodOutput(
            frequency="month",
            variables=("sea_surface_temperature",),
        )
    ),
)
```

The same argument is available on ERA5, ERA-Interim, and direct JCM land
factories. `JCMLandAtmosphereConfig.land_output` configures land independently
in the paired JCM setup. To request the former step cadence explicitly, use
`OutputSpec(period=PeriodOutput(frequency="step"))`.

An omitted declaration still permits run-level final fields through
`OutputTarget.write_final_fields`. JAXGCM, Veros, and CAMulator also default to
no period policy, while retaining their model-specific native snapshot writers.

The generic provider writes only fields declared in `ComponentSpec.outputs`.
Custom and third-party components remain opt-in and must attach their own
`OutputSpec(period=PeriodOutput(...))`. In every case, files are written only
when `Coupler.run` receives an enabled `OutputTarget`.

Use `output=None`, the default, for differentiated or outer-JIT-compiled runs.
This performs no file I/O or output sampling. Enabled output does not accept
traced runtime state.

### Explore complete examples

The [`examples`](examples) directory contains complete configurations for:

- coupled slab atmosphere, ocean, sea-ice, and land components;
- JCM with slab, ERA5, ERA-Interim, or Veros data and models;
- Veros with ERA5 forcing;
- CAMulator with Veros;
- custom component wrappers; and
- runtime profiling.

Some examples require external datasets, model configuration files, weights,
or optional packages. Review each script before running it; the CAMulator
example currently contains machine-specific configuration and checkpoint
paths that must be replaced.

## Configuration

VerCOR keeps configuration in four places:

- `RuntimeOptions` controls numeric precision, execution backend, workflow,
  and topology policy.
- `vercor.physics.PhysicalConstants` contains physical constants used by the
  model and remains visible to JAX differentiation.
- `ComponentSpec` declares one component's input and output fields, initial
  fields, lifecycle hooks, transfer policy, execution mode, and output policy.
- Setup configuration classes such as `JAXGCMConfig`, `VerosConfig`, and
  `CAMulatorConfig` hold settings for bundled model integrations.

The public package root exports `Clock`, `Coupler`, `Exchange`,
`RectilinearGrid`, `RunState`, and `RuntimeOptions`. More specialized APIs live
in their subject modules, including `vercor.components`, `vercor.output`,
`vercor.physics`, `vercor.regridding`, `vercor.runtime`, `vercor.setups`, and
`vercor.topology`.

## Troubleshooting

### The installed package does not provide the 0.4 API

Version `0.4.1` is the current release. Check the installed version and upgrade
to the stable release before following this README:

```bash
python -m pip install --upgrade "vercor==0.4.1"
```

### An optional setup cannot be imported

Install the matching `jcm` or `veros` extra. Optional model libraries are loaded
only when their setup factory is called, so they are not included in the core
installation. CAMulator requires MILES-CREDIT, but an exact compatible release
and installation command are not yet documented.

### A host component fails with the JAX backend

Use `RuntimeOptions(backend="auto")` or `RuntimeOptions(backend="host")`.
The forced JAX backend cannot run components declared with `execution="host"`.

### Output fails inside a differentiated or JIT-compiled call

Run with `output=None`. File output requires host-side work and is intentionally
disabled for traced runtime state.

### A VerCOR 0.3 import or workflow no longer works

Version 0.4.0 does not include a legacy compatibility namespace. Follow the
[migration guide](docs/migration-0.3-to-0.4.md) to update imports and assembly.

## Additional resources

- [API architecture and public-module reference](docs/api-architecture-review.md)
- [Plugin authoring guide](docs/plugin-authoring.md)
- [VerCOR 0.3 to 0.4 migration guide](docs/migration-0.3-to-0.4.md)
- [Design specification](DESIGN.md)
- [Changelog](CHANGELOG.md)
- [License](LICENSE)
