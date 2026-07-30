# VerCOR 0.4 Design Specification

VerCOR is a JAX-first, fully differentiable coupler for composing Earth-system
components. This document describes the implemented stable `0.4.0` architecture.
The exact API inventory and migration decisions are in
`docs/api-architecture-review.md` and `docs/migration-0.3-to-0.4.md`.

## 1. Goals and constraints

- Preserve end-to-end JVP and reverse-mode differentiation for output-free JAX
  workflows.
- Keep physics values explicit, traced, SI-valued inputs.
- Represent runtime state as immutable registered PyTrees.
- Make component, route, workflow, backend, topology, and output ownership
  unambiguous.
- Keep optional JCM, Veros, and CAMulator frameworks lazy.
- Validate author and backend boundaries before corrupt state reaches a physics
  step.
- Keep public contracts small while allowing third-party structural plugins.

The stable release does not provide registries, entry-point discovery, Pydantic
models, fan-in reducers, a public prepared graph, fractional subcycling,
restart files, or a CAMulator dependency pin. Ambiguous target-field fan-in is
rejected.

## 2. Public boundary

The root exports exactly `Clock`, `Coupler`, `Exchange`, `RectilinearGrid`,
`RunState`, and `RuntimeOptions`. Advanced objects live only in canonical public
owner modules. A public module has an explicit `__all__`; private imports are
underscored and public annotations resolve without leaking private names.

The stable extension tier is the six-symbol root plus `vercor.components`,
`vercor.coupler`, `vercor.exchanges`, `vercor.grids`, `vercor.output`,
`vercor.physics`, `vercor.regridding`, `vercor.runtime`, `vercor.state`,
`vercor.topology`, and `vercor.types`. Other existing public manifests remain
available as the current inventory; they are protected by the complete
manifest/signature JSON, but are not independent plugin-workflow promises.

`Coupler(...)` is the only assembly path. It receives the complete clock,
component collection, exchange collection, run order, runtime policy, physical
constants, and logging policy. It copy-owns the collections but retains the
original component author objects, which are treated as immutable
configuration with immutable name, grid, and specification identity.
Reconfiguration constructs a new coupler.

The public state surface is intentionally opaque. `RunState.component(...)`,
`RunState.components(...)`, and `RunState.replace_fields(...)` provide reads and
immutable replacement. Runtime stores, topology maps, alignment metadata, and
prepared bindings remain private.

## 3. Configuration ownership

Configuration has four non-overlapping owners:

1. `PhysicalConstants` is a frozen registered PyTree of traced physics values.
2. `RuntimeOptions` owns static dtype, backend, workflow, and topology policy.
3. `ComponentSpec` owns one component's inputs, outputs, initial fields,
   execution capability, lifecycle, transfer, and output declaration.
4. Frozen setup or plugin dataclasses own model-specific construction policy.

`Clock.calendar` selects the model calendar. `vercor.calendar` owns canonical
year types and durations, and runtime time metadata derives the applicable
duration independently from every timestamp's calendar and year.

`RuntimeOptions.dtype` is the sole precision owner. Preparation normalizes
VerCOR-owned fields, grids, constants, and numeric payload leaves to that
policy. Integer/index arrays stay 32-bit. Physics code never branches in Python
on traced constants.

## 4. Component authoring

`vercor.components.Component` is a runtime-checkable structural protocol with
`name`, `grid`, `spec`, and `step`. Authors may implement it directly.
`CallableComponent` adapts a step callable and `DataComponent` provides seeded
data; there are no other concrete convenience hierarchies.

`LifecycleHooks.setup(component, context)` receives the original author object
and returns `SetupResult(fields, payload)`. Prefill and validation use typed
contexts and immutable result mappings. Scalar initial/setup values expand to
the component grid. Every declared field is normalized and checked before the
runtime state is created. The private binding preserves the original name,
grid, and specification references and revalidates them after setup, prefill,
and validation callbacks.

A mapping step result replaces declared fields and preserves payload. A
`StepResult` may explicitly replace payload. Compiled JAX execution requires
payload PyTree structure, leaf shapes, and dtypes to remain stable; host
execution may clear or restructure payload. Standard containers and NumPy
leaves are defensively owned during preparation. External model state belongs
to that payload from setup through every functional step; it is never retained
as hidden evolving adapter state.

One private adapter performs structural validation, setup, declaration
normalization, payload ownership, and runtime callback dispatch. There is no
component `initialize`, `initial_fields` method, constructor payload, separate
payload factory, output marker, or import-policy property.

## 5. Coupling and topology

Every `Exchange` owns a stable `route_id` and injected `regridder_factory`. The
default ID is `source->target`; collisions require explicit distinct IDs and
fail before setup or factory calls. Routes may use custom field names declared
by both endpoints.

`Regridder` defines scalar transfer. `VectorRegridder` adds vector transfer and
is required for vector fields. Runtime preparation verifies that each factory
result supports the route's field capabilities. `RegridderFactory` is one
runtime-checkable protocol, so static typing and runtime introspection share
the same public callable contract.

A `TopologyPolicy.build(TopologyContext)` returns one
`ExchangeTopologyPatch` keyed by configured route IDs. Patch masks must match
the target grid and contain finite binary or fractional values in their allowed
ranges. The optional `SurfaceMaskPolicy` implements bundled ATM/OCN/LND mask
policy; ordinary setup-agnostic graphs use no topology policy.

The runtime rejects duplicate route IDs and deterministic scalar/vector fan-in
to one target field. Feedback where a component receives and later sends the
same field is valid.

## 6. Preparation and state validation

The private prepared binding is the sole post-lifecycle boundary. It stores
normalized component bindings, routes, contracts, topology, run order, clock,
constants, runtime options, dispatch context, and interrupt controller. It has
no reflective configuration fingerprint and is never public.

Initial and supplied states are checked against the prepared binding. Custom
backend inputs and results receive the same check. Validation covers:

- exact component, store-field, and route names and order;
- grid type, name, centers, edges, and masks;
- array shapes and dtypes;
- payload PyTree structure where compiled execution requires it; and
- finite binary and fractional mask constraints.

Runtime assertions remain transform-safe. A structurally exact foreign state
is accepted; a changed coordinate, dtype, field, payload schema, or mask is not.

## 7. Workflow and execution

A `Workflow.build(WorkflowContext)` returns an `ExecutionPlan` containing
exactly one ascending `StepPlan` per clock step. A step may reorder or omit
registered components but may not reference unknown names or repeat one. The
default `SequentialWorkflow` repeats the constructor run order.

The private coordinator validates the plan, groups compatible consecutive
schedules, and splits at output boundaries. Each `ExecutionChunk` retains
absolute clock indices. Output-free uniform workflows preserve one JIT-wrapped
`jax.lax.scan`; a run-local executor is reused for repeated schedules.

An `ExecutionBackend.execute(...)` receives state, public context, a core-owned
chunk, and `RuntimeDriver`. It must consume every plan exactly once and in order
through `RuntimeDriver.run_step`. The driver rejects forged, repeated,
reordered, skipped, or out-of-chunk plans before dispatch.

`backend="auto"` selects JAX unless a scheduled component requires host
execution. Forced JAX rejects host components; forced host runs all scheduled
components in Python. Host, compiled, and custom execution share receive,
step, send, cancellation, state validation, and output boundaries.

Graceful interruption is scoped around each run. Host paths check at step and
component boundaries. Compiled paths use ordered callbacks and a nonblocking
wakeup descriptor so terminal signals delivered during XLA execution are
observed without depending on logging.

## 8. Output

Output is opt-in. `Coupler.run(output=None)` and an all-disabled `OutputTarget`
perform no provider sampling, host transfer, path creation, or file I/O.
Differentiated and outer-jitted callers use this path.

An `OutputProvider.sample(OutputContext)` returns an immutable `OutputFrame` of
named `OutputVariable` values, coordinates, dimensions, attributes, and
metadata. `PeriodOutput.variables` uses identical empty/subset/unknown behavior
for runtime, JAXGCM, Veros, CAMulator, and third-party providers.
Bundled slab and data factories default to `OutputSpec()` with no period
policy, matching bundled external configuration defaults. They accept a
complete keyword-only `OutputSpec`, allowing cadence, variable selection,
providers, and snapshots to be configured independently per component.

The default runtime-field provider applies the component's `TransferPolicy`
with the exact precomputed metadata for `OutputContext.step`. Period output
therefore samples the same `current`, linearly interpolated monthly, or indexed
daily field exported during that coupling step; internal forcing-record axes
are never emitted as physical output dimensions.

One private session owns:

- provider schema validation and selection;
- immutable JAX sum/count accumulation;
- precomputed cadence boundaries;
- collision-safe filenames and per-schema averaging-window-start timestamps;
- final runtime fields and snapshot contexts;
- host transfer; and
- NetCDF writes.

Providers sample post-step state at end-of-step model time. The first sample
fixes the run schema; later variable, dimension, coordinate, metadata, shape, or
dtype drift fails with component-scoped diagnostics. Output paths include time,
absolute-step, and schema discriminators where needed.

Enabled output rejects traced runtime leaves before provider or writer calls.
Execution backends and model steps never receive an output session and never
choose cadence or paths. JAXGCM, Veros, and CAMulator retain only native state
needed for sampling and final snapshots.

## 9. Bundled setups and plugins

`vercor.setups` is the sole public lazy export table. Attribute access loads a
lightweight factory; JCM/Dinosaur, Veros, CREDIT, Torch, TensorFlow, and runtime
configuration are deferred until invocation. Missing optional dependencies
produce factory-oriented errors without affecting core import.

The setup-gallery command boundary has explicit owners:

- `vercor.cli`: Click presentation, shared discovery, duplicate rejection,
  and exclusive copying.
- `vercor._setup_runner`: private child loading and contract invocation.
- `vercor.setups.gallery`: model-specific setup construction and explicit
  translation of log level and dtype.
- `vercor._logging.config`: standard levels plus `trace == 5`.

`vercor.cli` exposes the installed ``vercor`` command. It discovers the
packaged gallery and direct external directories named by an ``os.pathsep``-
separated ``VERCOR_SETUP_DIR`` list, rejects duplicate template names, and
creates or reuses a requested ``--to`` directory while exclusively copying a
single file without overwrite. ``run`` accepts only local ``.py`` files and
uses the active interpreter to invoke the private child runner. That child
loads the setup without its main guard and requires exactly
``run_setup(*, loglevel, float_type)``; ``None`` means success and an integer
is the process status. CLI choices are lowercase ``trace``, ``debug``,
``info``, ``warning``, and ``error`` (default ``info``), plus ``float64`` and
``float32`` (default ``float64``). Gallery templates translate those choices to
the Coupler log level and ``DTypePolicy``/``RuntimeOptions`` dtype policy.

Bundled slab, data, JCM, Veros, and CAMulator factories return the same
structural components used by external plugins. JAXGCM/Veros spinup is
controlled only by `Spinup.enabled`. CAMulator enabled spinup is rejected
because it is not implemented. CAMulator forcing alignment is explicitly
`strict` or `forcing_start`; both policies require a standard-library
`datetime` clock because no-leap and 360-day CAMulator conversion is not
implemented.

Every slab and data factory accepts a final keyword-only
`output: OutputSpec | None` argument. Omission selects `OutputSpec()`; a
supplied declaration is retained unchanged. Paired JCM construction owns its
land declaration independently as `JCMLandAtmosphereConfig.land_output`, which
also defaults to `OutputSpec()`. The paired atmosphere retains its explicit
historic monthly policy.

The temporarily built external extension test fixture demonstrates the external
boundary with
plugin-owned frozen configuration and factory, structural JAX/host components,
an injected regridder, explicit route, non-empty topology patch, custom
workflow/backend, immutable state replacement, period output, and snapshot
output. It imports no private VerCOR module.

No legacy adapter namespace or executable VerCOR 0.3 evidence ships with
version 0.4.0. VerCOR 0.3-only workflows must migrate directly to the current
contracts.

## 10. Testing and release evidence

Tests are layered across pure numerical kernels, component contracts, routes,
state, workflow/backend behavior, output behavior, optional setup boundaries,
gradients, and installed distributions. `--fast` selects a deterministic
development subset; the full suite runs before commit.

Release gates are Black, strict flake8, mypy, compileall, fast/full pytest,
90% branch coverage, build, installed wheel and source-distribution probes,
external-extension smoke and strict mypy, optional base/JCM/Veros lanes, a
macOS smoke, and `git diff --check`.

Built-artifact tests run outside the checkout and verify origin, metadata,
`py.typed`, the six-symbol root, every canonical owner manifest, central
constructor signatures, removed primary modules, the dependency-free slab, and
the composed external extension. CI builds and uploads only the two VerCOR
distributions once, shares that artifact bundle with all matrix cells, and
builds the fixture separately for each external-extension contract job.
