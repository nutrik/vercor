# VerCOR 0.4 Design Specification

VerCOR is a JAX-first, fully differentiable coupler for composing Earth-system
components. This document describes the implemented stable `0.4.0` architecture.
The exact API inventory and compatibility decisions are in
`docs/api-architecture-review.md`. The completed source/test removal decisions,
retained native adapters and assertion migrations are recorded in
`docs/source-coupling-audit.md`.

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
- Active runtime fields are finite at preparation, exchange, component-step,
  and send boundaries. NaN remains a missing-data sentinel only outside the
  applicable component or route mask; infinity is always invalid.
- Masked numerical kernels neutralize inactive operands before nonlinear
  arithmetic so output-free JVP and VJP cannot inherit inactive-branch NaNs.
- ComponentMaskSpec may reference current binary and fractional runtime fields
  inside immutable grid support. Explicit exchange quantities weight source
  fields before remapping; intensive values normalize covered overlap and
  remain available on wet cells with zero target coverage. Route diagnostics
  refresh at initialization, supplied-state preparation, and every component
  step, and final output records current coverage. Mask fields remain finite
  everywhere; inactive exports may contain NaN until their support reactivates.
  Stored imports retain the coverage of their last dispatch, separately from
  current route diagnostics, so a source can reactivate before its recipient
  runs. Imported mask fields are validated before author physics executes.
  Explicit routes can select `source_coverage="binary"` for total-area fluxes;
  their exported fields require finite binary support in sends and supplied
  states, including wet cells with zero fractional coverage.

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
Replacing a current exported field also replaces its matching outgoing copy,
so a supplied state is coherent before any recipient runs. Record arrays keep
their prior selected export until the source's next time-selected send;
received values and their last-dispatch support retain independent ownership.

Non-current transfer policies require time-first selectable outputs with twelve
monthly records or 365 no-leap daily records. Declared grid-shaped mask fields
remain current. Setup, prefill, and the shared selector enforce this layout;
monthly interpolation preserves the runtime field dtype in sends and output.

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
VerCOR-owned fields, grids, and constants to that policy. Native adapters own
payload precision conversion; the generic binding copies arbitrary payload
leaves without casting opaque native state or categorical values. VerCOR
integer/index arrays stay 32-bit. Physics code never branches in Python
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
Active runtime fields are finite at preparation, exchange, component-step, and
send boundaries. NaN remains a missing-data sentinel only outside the
applicable component or route mask; infinity is always invalid. Masked
numerical kernels neutralize inactive operands before nonlinear arithmetic so
output-free JVP and VJP cannot inherit inactive-branch NaNs.
State validation completes schema, topology, and central route/component field
finiteness checks before invoking component-author lifecycle validation.
JCM, Veris, and CAMulator validate their owned numeric payload/native leaves
at native transition boundaries while preserving each adapter's established
JAX-transform or host-only differentiation boundary. Veros currently relies on
central exported-field validation; hidden deep native-state validation needs
an explicit active-mask policy. The current interface and wrapper audit is
recorded in `docs/interface-wrapper-audit.md`.

## 7. Workflow and execution

A `Workflow.build(WorkflowContext)` returns an `ExecutionPlan` containing
exactly one ascending `StepPlan` per clock step. A step may reorder or omit
registered components but may not reference unknown names or repeat one. The
default `SequentialWorkflow` repeats the constructor run order.

The private coordinator validates the plan, groups compatible consecutive
schedules, and splits at the union of enabled output cadence boundaries when
every provider supports numeric JAX sampling. Each `ExecutionChunk` retains
absolute clock indices. Output-free uniform workflows preserve one JIT-wrapped
`jax.lax.scan`; a run-local executor is reused for repeated schedules. The
fixed-shape scan step is wrapped in `jax.checkpoint`, so reverse-mode rollouts
rematerialize per-step intermediates instead of retaining the complete
trajectory. Scan length, callback order, progress reporting, interrupt checks,
and carry structure are unchanged.

Numeric output uses the same private runtime-step builder, composing each
post-step sample with immutable sum/count accumulation inside a checkpointed
scan. Output-free and output-carry executors retain separate run-local caches
by component schedule. Ordinary providers and host/custom backends preserve
single-step sampling through the same output session. A schedule change also
splits a scanned output window while retaining every provider's partial accumulator.

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
for runtime, JCM, Veros, CAMulator, and third-party providers.
Bundled slab and data factories default to `OutputSpec()` with no period
policy, matching bundled external configuration defaults. They accept a
complete keyword-only `OutputSpec`, allowing cadence, variable selection,
providers, and snapshots to be configured independently per component.

`JaxOutputProvider` is an optional structural capability in addition to ordinary
`OutputProvider.sample`. Period sampling prefers this capability independently
for each provider on every backend, including mixed-component and custom runs.
Its pure `sample_jax(JaxOutputContext)` receives state, payload, absolute int32
step and coupling interval in seconds, with no Python datetime. The step is
traced inside a scan and concrete on the host. Numeric values and coordinates
may change, but names, dimensions, metadata, shapes and dtypes remain fixed.
The default field provider and bundled
JCM, Veris and Veros providers implement it. Their host methods add calendar
timestamps to the same numeric extraction; snapshots retain ordinary host I/O.

Before the first output scan, abstract tracing of the first runtime step and
numeric sample establishes a zero accumulator template. It executes no actual
model step and accumulates no initial-state sample. The first actual post-step
sample remains inside the first scan. The carry stores only current state,
sums, finite counts and latest coordinates, without a trajectory. Completed
periods reset only their due accumulators; incomplete periods are not flushed.
Zero-step runs do not trace or sample providers. An ordinary provider among the
enabled providers retains per-step execution without speculative tracing of its
sampling method; numeric neighbors still use `sample_jax`. Ordinary providers
receive the concrete Python step, exact end-of-step calendar time and interval.
Host and custom backends sample concrete post-step state without abstractly
tracing native payloads. All sampling paths share provider selection, frame
validation, variable selection, finite sum/count accumulation and period resets.
Only scanned carry normalizes coordinate leaves to JAX arrays; host sampling
preserves native coordinate precision.

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
choose cadence or paths. JCM, Veros, and CAMulator retain only native state
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
single file without overwrite. Copy references resolve against both the public
stem shown by ``show-setups`` and the canonical ``.py`` filename; a reference
shared by one template's stem and another's filename is rejected as ambiguous.
Every failure after exclusive creation removes the partial destination.
``run`` accepts only local ``.py`` files and uses the active interpreter with
``-P`` to execute the resolved private runner file from the already imported
package. That child keeps the setup's parent importable while loading and
invoking the setup, restores ``sys.path`` afterward, skips the main guard, and
requires exactly ``run_setup(*, loglevel, float_type)``; ``None`` means success
and an integer is the process status. CLI choices are lowercase ``trace``,
``debug``, ``info``, ``warning``, and ``error`` (default ``info``), plus
``float64`` and ``float32`` (default ``float64``). Gallery templates translate
those choices to the Coupler log level and ``DTypePolicy``/``RuntimeOptions``
dtype policy.

Bundled slab, data, JCM, Veros, Veris, and CAMulator factories return the same
structural components used by external plugins. JCM/Veros spinup is
controlled only by `Spinup.enabled`. CAMulator enabled spinup is rejected
because it is not implemented. CAMulator forcing alignment is explicitly
`strict` or `forcing_start`; both policies require a standard-library
`datetime` clock because no-leap and 360-day CAMulator conversion is not
implemented.

Standalone Veris uses native `initialize_experiment` and pure `advance` with
immutable state/diagnostics and traced physical constants in the component
payload. Frozen setup resources retain numerical configuration, precision and
substep alignment. The lazy `replace_veris_parameters` helper replaces only
independent scalar physical leaves, using native validation/dependent laws,
while preserving fields and native state. VerCOR owns native rhoFresh through
its traced freshwater_density at setup and every transition. Native ocean
initial fields do not depend on rhoFresh, so that owner is applied after host
initialization to preserve outer-JIT setup. Each substep receives fresh forcing. `make_veris_verosdata` snapshots one setup-only Veros global4deg ocean,
sharing its exact tracer grid/wet domain with the data ocean. The snapshot
converts Veros east/north face geometry to native west/south indices and retains
authoritative latitude/edge coordinates and native radius for spherical metrics;
supplied native areas retain their quadrature without rescaling. CESM bulk
`tref/qref` supply ice air temperature/humidity; native `Area` supplies evolving
fractional coverage while ice-free wet cells retain forcing. Output reuses
the shared coordinator; the gallery sends no feedback to prescribed data.

`VerosConfig` selects `setup="global_4deg"` or `setup="acc"` and an
`execution="host"` or `execution="jax"` lane. Host is the compatibility
default and configures stock Veros with the NumPy backend and its best
available solver. Adapter construction deep-copies the complete native state,
substituting an immutable tuple for Veros' non-copyable settings `dict_keys`
view without modifying the source. On every host step, the generic component
runtime ownership boundary copies that normalized payload exactly once before
the adapter mutates it through forcing and model substeps. JAX execution
configures the JAX backend with `scipy_jax`, uses the fork's native PyTree-aware
`VerosState.copy()` at functional mutation boundaries, and requires runtime
capabilities supplied by the approved differentiable fork: native state
copying, differentiable operators, and traced `c_k`/`c_eps` variables.
`execution` is the sole Veros copy and compilation policy; there is no separate
`jitted` option. Runtime settings lock on first Veros use, so conflicting
host/JAX components must run in separate Python processes. The packaged
optional requirement remains `veros>=1.7.0`; fork selection is
capability-gated rather than globally pinned.

Both global Veros execution lanes receive zero-seeded `rain_rate` and
`snow_rate` in kg/m²/s. ERA5 aggregates `crr + lsrr` and `csfr + lssfr` before
monthly interpolation; JCM aggregates `precls + precnv` before remapping and
converts g/m²/s to kg/m²/s. JCM diagnoses phases on its native grid using
SPEEDY `surface_flux.t0`, passed as `near_surface_air_temperature`. This near-surface
diagnostic has no specified 2 m height; it is distinct from bottom-level air.
The snow fraction is `sigmoid(4 * (T_mid - t0) / (T_rain - T_snow))`, with
`T_mid = (T_snow + T_rain) / 2`. The traced `PhysicalConstants` thresholds
`precipitation_snow_temperature=272.77` K and
`precipitation_rain_temperature=278.15` K are the calibrated -0.38/5 degrees C
LT values from Bedard-Therrien et al. (2025), doi:10.5194/hess-29-1135-2025,
Appendix C. They require a positive span. The sigmoid matches LT's midpoint
and midpoint slope but retains phase tails, so it is a smooth approximation,
not a refit or the exact piecewise-linear model. The calibration used boreal
eastern Canadian land stations, not global ocean or SPEEDY validation.
Native snow is total precipitation times its fraction; native rain is the
remainder. Both rates are transposed and then independently interpolated.
Bottom sigma-center temperature, humidity, winds, density, and potential
temperature use `air_temperature`, `air_specific_humidity`, `air_u_velocity`,
`air_v_velocity`, `air_density` and `air_potential_temperature`, with their common
`model_level_height`; only the phase diagnosis uses t0. JCM's `air_pressure_3d`
retains `(sigma_level, latitude, longitude)` pressure in Pa; CAMulator's
`model_level_air_pressure` is the horizontal lowest-level pressure in Pa.
VerCOR computes the rigid-lid virtual salt flux on the ocean grid as
`-S_ref * (rain + snow - E) / rho_fw`, with upward-positive evaporation `E`
from the existing bulk calculation. `PhysicalConstants` owns traced
`surface_salinity_reference=34.7` g/kg and `freshwater_density=1000.0` kg/m³.
The native adapter retains the coupled contribution separately so its forcing
callback can assign `forc_salt_surface` without accumulating restoring across
substeps. `restore_to_climatology_temp` independently enables the SST
correction, while `restore_to_climatology_salt` adds 30-day SSS restoring.
Both default to `False`. The common host/JAX native callback owns both
scalar weights, so the controls also apply during spinup before any coupled
forcing is prepared. Layout preparation preserves the raw `qnec` coefficient;
the temperature weight gates only its contribution to heat forcing. Coupled
heat and freshwater contributions remain active with either switch off.
Ocean/ice masks
and native cyclic halo exchange are applied to the final salt flux. Its
metadata is `m g kg-1 s-1`; no timestep or top-cell-depth factor is applied
before Veros's tracer integration.

The JAX global adapter consumes external atmospheric forcing, disables
streamfunction and diagnostics that would change the payload PyTree, and uses
the fork's finite-derivative calm-stress norm. The JAX ACC adapter retains its
native internal forcing and topology while leaving variable-owned turbulence
parameters at their fork defaults. JCM uses one physical default mapping
everywhere fields are declared, initialized, or prefilled: temperature and
potential temperature are 288.15 K, density is 1.2 kg/m³, and model-level
height is 50 m. These defaults prevent a not-yet-stepped atmosphere from
feeding valid-looking zero divisors and logarithm inputs into the ocean.
After a JCM step, model-level height is the surface-nearest sigma-center
altitude above ground. The adapter anchors the pressure ratio at the surface
and uses the lowest model-level virtual temperature for the unresolved
surface-to-center layer.

ERA5 and CAMulator instead use the distinct ECMWF hybrid-sigma coordinate.
Interface pressure is `A + B * ps`; temperature, specific humidity, and
interfaces enter the height kernel top-to-bottom. The kernel follows the IFS
half-level hydrostatic discretization, integrates upward from zero surface
geopotential, and returns full-level geometric heights above ground in
bottom-to-top order. It therefore uses the bounding interface pressures rather
than treating an arithmetic full-level pressure as an exact hydrostatic
location. Any nonzero pressure component makes the effective pressure ratio
and height depend on local surface pressure. Static layout checks reject
ambiguous ranks, mismatched thermodynamic fields, and missing interfaces before
JAX arithmetic can broadcast them silently.

Here, fully differentiable means finite first-order JVP, VJP, and reverse-mode
gradients for output-free fixed-plan JAX rollouts over physically valid
continuous inputs. Output I/O, setup-time/static values, stock-host execution,
and second derivatives are outside this contract.

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

The 0.4 API does not include legacy compatibility adapters or executable
pre-redesign evidence. Applications use the current contracts directly.

`calendar.YearType` is the sole forcing-year enum owner. Public regridding
factories construct concrete private implementations directly while retaining
their signatures, including `radius_km`. JCM output reads only runtime context
payloads; a missing final snapshot payload raises a component-scoped error
before output is created. Setup constructors own all native runtime policy
attributes. Veris forcing consumers use the single production forcing pack,
which contains native forcing and open-water face stresses.

## 10. Testing and release evidence

Tests are layered across pure numerical kernels, component contracts, routes,
state, workflow/backend behavior, output behavior, optional setup boundaries,
gradients, and installed distributions. `--fast` selects a deterministic
development subset; the full suite runs before commit.

The numerical-safety matrix requires finite active primal/JIT, JVP, and VJP
evidence for differentiable paths and fail-fast primal evidence at host-only
boundaries. It retains inactive NaN only for masked or output-only semantics;
masked kernels neutralize inactive operands before nonlinear arithmetic.

Release gates are Black, strict flake8, mypy, compileall, fast/full pytest,
90% branch coverage, build, installed wheel and source-distribution probes,
external-extension smoke and strict mypy, optional base/JCM/Veros lanes, a
macOS smoke, and `git diff --check`.

CI keeps stock-Veros compatibility separate from differentiable-fork
acceptance. The ordinary quality job and installed-artifact Veros matrix lane
resolve the packaged `veros>=1.7.0` requirement. A dedicated serial job
installs the public differentiable fork without dependencies at immutable commit
`7a8c964cf00b5aa0713c995edd760643a431b3c9`, verifies its PEP 610 commit record
and required runtime capabilities, and runs the multi-step rollout module.
Both lanes collect relative-path branch data without applying a partial-suite
threshold and upload distinct raw artifacts. A downstream job requires both
artifacts, unions their executed lines and arcs with Coverage.py, enforces the
90% project threshold, and sends one deterministic XML report to Codecov.
Release publication depends on that combined gate, so neither stock
compatibility nor the differentiability contract can be bypassed.

Built-artifact tests run outside the checkout and verify origin, metadata,
`py.typed`, the six-symbol root, every canonical owner manifest, central
constructor signatures, removed primary modules, the dependency-free slab, and
the composed external extension. CI builds and uploads only the two VerCOR
distributions once, shares that artifact bundle with all matrix cells, and
builds the fixture separately for each external-extension contract job.

The Veris runtime retains final native diagnostics and separately integrates
complete ocean heat, actual mass/salt and face-stress products over native
substeps. Canonical fields are total-area interval means; two external boundary
means in the payload support independent column budgets. Native inputs/outputs
are checked at every substep with a sticky validity carry and a single coupling
boundary guard, including accumulation overflow. Stress combines retained
momentum AreaW/AreaS with ice traction and complementary shared-bulk open-water
traction before shifting west/south to east/north faces.

Global Veros selects `surface_forcing="bulk_meteorology"` (default) or
`surface_forcing="surface_fluxes"` explicitly. Surface-flux mode consumes the
five canonical rates once, preserves independent native
restoring, and activates a scalar native flag after climatological spinup. The
cold-water proxy applies only while total feedback is inactive. Native pressure/
free-surface execution provides actual SSH, with canonical native surface
velocities and salinity alongside SST. Salt conversion uses fixed S_ref and
actual M/J, separately from physical local-salinity inventory accounting.

The prognostic Veris/Veros factory reuses the sole Veros setup owner and extracts
geometry without copying prescribed climatology proxies. Both components share
one exact grid. Scalar total fluxes use conservative routing; vector stresses
use existing shared-grid identity routing, with binary coverage for both. The
focused calibration gallery fits native h0 through public immutable state and
fields using fixed wet weights, an identifiability guard and JAX value/gradient
evaluation inside bounded SciPy minimization; it adds no generic optimizer API.

Extended Veris validation reuses the prescribed gallery, Coupler.run, native
stored areas and the shared output coordinator. A private index counter measures
actual guarded native transitions; pure device reductions provide scalar budget
rates. Host-only benchmark reporting and segmented comparisons introduce no
alternative integrator or mutable model state. See docs/veris-validation.rst.


## Coupled Veris–Veros–JCM surface assembly

Reusable field recipes use physical atmosphere/ocean/sea-ice names. The concrete
JCM entry points are `make_jcm_atmosphere` and `JCMAtmosphereConfig`; private
`JCMAtmosphereSetupState` and `JCMAtmosphereRuntimePayload` retain their existing
module paths and lifecycle. There are no former-name aliases. The field,
recipe, mask-helper and forcing-mode migration is documented in
`docs/coupling-fields.rst`.

`run_veris_veros_jcm.py` reuses the paired native factories and the sequential
LND → CICE → OCN → ATM workflow. Ice exports current per-ice temperature and
native category/snow albedo, plus current concentration. Ocean exports current
`(1-A)*SST` after its native advance; full SST/SSS/currents/SSH remain available
under ice. Total ocean boundary rates pass through Veris once. Routes, run order,
diagnostic state lookup and topology roles use actual component names, preserving
configurable instance identities.

Explicit routes form source products before conservative remapping:
`R(W*A*Tice)`, `R(W*A*alpha_ice)`, `R(W*A)` and `R(W*(1-A)*SST)`.
The JCM compositor normalizes by static ocean area only. Land temperature and
moisture retain their independent native fields. A frozen forcing wrapper
carries marine albedo through native time/solar selection; a narrow native
SPEEDY forcing subclass replaces only marine and combined albedo diagnostics.
Native concentration remains the real marine ice fraction. All evolving values
are payload/field leaves, with no model or physics-term ice-state mutation.

JCM 3 uses Gregorian native time/step leaves in its immutable payload. Initial
physical diagnostics do not advance the native carry. Humidity is already kg/kg;
evaporation and precipitation diagnostics retain native gram conversions.
`SurfaceMaskPolicy` validates static geometry and restricts binary land support.
Components own fractions; the runtime resolves current overlap and separately
retains last-dispatch coverage for imports. Declared mask fields bypass record
selection in static/daily/monthly data components.

All exchanges require `intensive`, `source_flux`, or `target_flux`. One scalar/
vector dispatcher forms covered source products before remapping. Intensive
values divide by remapped source coverage; source fluxes retain that numerator;
target fluxes multiply the normalized value by target coverage. Preweighted
Veris rates and open-water products use binary source coverage to avoid a second
area factor. See `docs/exchange-coverage.rst` for formulas and the field audit.

JCM's single marine channel uses the area-composed effective temperature;
separate ice/water turbulent channels and long-term climate validation remain
outside this functional integration. Canonical marine output joins the native
provider through the existing shared output coordinator.
