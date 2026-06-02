# VerCOR: Design Specification

A fully differentiable coupler in JAX for different Earth system models written in JAX.

---

## 1. Goals and Non-Goals

### Goals

- **End-to-end differentiable**: exact gradients of any output with respect to any
  coupled models parameters, via JAX reverse-mode AD.
- **No global arrays or mutable state**: all data is passed explicitly via function arguments and return values.
- **Accelerated**: pure functions are JIT-compiled with `jax.jit` decorator, 
single device parallelism via `jax.vmap` where applicable, use `jax.lax.scan` for iterative methods/solvers, 
use `jax.lax.fori_loop`, `jax.numpy.where` and  `jax.lax.cond` etc. to avoid Python control flow.
- **Modularity**: clean, modular code structure for easy maintenance and extension.
- **Documentation**: comprehensive docstrings and usage examples.
- **Testing**: extensive unit tests for correctness and regression prevention.

## 2. Architecture Overview

### Modular design

The codebase is organized into modules corresponding to physical, numerical and different coupled models/components.

Interpolation, exchangers, grids, model components, output routines etc. are all separate modules with well-defined interfaces. 
This allows different agents to work on different components in parallel and makes testing easier.

For bilinear rectilinear interpolation, `BilinearRectilinearInterpolator` is the
public PyTree and user-facing facade. Private helper owners under
`vercor.interpolators` keep lower-level concerns separate:
`_bilinear_geometry` owns spherical geometry and orientation checks,
`_bilinear_weights` owns target-to-source cell lookup and bilinear weights, and
`_bilinear_extrapolation` owns nearest/IDW fill policy plus valid-source mask
normalization. Production code outside `vercor.interpolators` should depend on
the public interpolator or regridder boundary, not these private helpers.

The output module handles all data saving and logging, ensuring a clean separation between computation and I/O.

### Pure functional style

All functions are pure, jitted with `jax.jit` decorator and stateless. No mutable global state. No side effects.
This is critical for JAX compatibility and makes reasoning about the code easier.
Each function takes explicit inputs and returns explicit outputs, which can be easily tested and debugged.

### Compile cache hits and safe buffer donation

To ensure good performance, we need to design the code to maximize JIT cache hits and enable safe buffer donation.
This means avoiding dynamic shapes, using static arguments for control parameters, and ensuring that arrays are not mutated in-place.

**Keep JIT compile keys stable (avoid surprise recompiles):**
- Define model/containers at module top‑level so identities don’t change between runs.
- Mark non‑array metadata as static so it isn’t traced.
- Keep argument pytrees small and consistent (e.g., NamedTuple with fixed fields).
- If you pass constants/flags, make them static args.

**Anti‑pattern to avoid:** constructing fresh containers every call with changing non‑array fields (e.g., dicts with varying keys or dataclasses whose __eq__ changes) → recompiles.

**Donate buffers safely (lower peak memory, speed up):**
Donation lets XLA reuse input buffers for outputs.
- Rule of thumb: donate only arrays you won’t read again after the call.
- Practical boundary: donate at the outer step (not deep internals) so the contract is easy to respect.

**Small patterns that add up**
1) Stable run‑state wrapper
Keep “run‑level” scalars (step, time, dt) in a fixed NamedTuple; keep big arrays (model params) in plain pytrees (tuples/dicts) with stable keys.
2) Keep non‑array metadata out of traces
3) Reduce variant explosion: prefer fixed‑shape boundary tuples over dicts whose keys appear/disappear:
4) Donation audit at the callsite.
5) Deterministic RNG: split once per step at the boundary; don’t split inside inner kernels (helps compile stability).

### Input / Output

All I/O is handled by a dedicated module that reads/writes from/to disk.
The core computational modules are completely decoupled from file formats and storage details.
This allows us to easily swap out the I/O layer if needed, and keeps the core logic clean and focused on the physics.

The output is done in a structured format, such as NetCDF, HDF5, that can be easily read by visualization tools and post-processing scripts.

Model restart files are supported and written in compact HDF5 format using `h5py`.

Current example output snapshots are also written in HDF5. NetCDF output for broader
VerCOR workflows remains future work.

### Data flow: PyTree-based result objects

Shared PyTree mechanics live in `vercor.pytree.PyTreeNodeMixin`. Immutable
classes registered with `@jax.tree_util.register_pytree_node_class` should
inherit from the mixin and declare `pytree_children` for traced fields plus
`pytree_aux_data` for static metadata. The mixin reconstructs objects without
rerunning constructors, and classes with derived static attributes can restore
them in `_pytree_post_unflatten()`.

Every module returns an immutable PyTree container, usually a frozen dataclass,
containing arrays and objects.

No mutable state. No side effects.

```python
@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class RectilinearGrid(PyTreeNodeMixin):
    pytree_children = (
        "longitude",
        "latitude",
        "longitude_edges",
        "latitude_edges",
        "binary_mask",
    )
    pytree_aux_data = ("name",)

    name: str
    longitude: Array
    latitude: Array
    longitude_edges: Array
    latitude_edges: Array
    binary_mask: Array
    ...
```

### Public and runtime API boundary

VerCOR intentionally separates user-facing orchestration objects from the
immutable runtime containers used during traced integration.

- Public orchestration API: `Coupler`, `Exchange`, `RunSequence`, `Clock`,
  grids, regridders, and bundled concrete components are the objects users
  compose when configuring a coupled run.
- Component-author API: `Component`, `DataComponent`, and
  `HostRuntimeComponent` are the stable extension points. Custom adapters should
  use the helper-first authoring layer where possible. The most concise public
  helpers are `data_component()`, `differentiable_component()`, and
  `host_component()`, which delegate to the class-level constructors:
  `DataComponent.from_fields()` for data-only fields, `Component.from_model()`
  for pure callable JAX models, and `HostRuntimeComponent.from_model()` for
  Python host-side models. These helpers use author-facing field names:
  `inputs` declare fields the model reads, `outputs` declare fields the model
  writes, and `default_fields` declare concrete runtime defaults for fields
  the model reads or updates. Scalar default and seeded values expand
  to grid-shaped constants. `ComponentSetupContext` and `ComponentStepContext`
  are public setup and step context payloads passed to author callbacks, with
  canonical ownership in `vercor.components.contexts`.
  Lifecycle hook type aliases (`ComponentInitializeHook`,
  `ComponentCreatePayloadHook`, `ComponentPrefillHook`, and
  `ComponentValidateHook`) are public component-author contracts and are
  reexported from `vercor.components` and `vercor`.
  `ComponentFieldSpec`, `field_spec`, and `declare_fields()` provide the same
  vocabulary and read-only introspection for subclasses. `field_names` exposes
  setup-time seeded field names in insertion order. Subclass constructors can
  use `update_settings(...)` for chainable updates to existing component
  settings and `grid_field_defaults(...)` to build validated grid-shaped
  default-field mappings with scalar expansion and field-specific overrides.
  `DataComponent` seeding
  automatically records seeded fields as declared outputs, so data-only
  components remain introspectable whether fields are declared up front or added
  through helper seeding. The legacy `wrap()` classmethods and
  `make_*_component()` factory functions have been removed; component authors
  should use the helper facade, class-level `from_fields()` / `from_model()`
  constructors, or subclasses with `declare_fields(...)`. `vercor.components`
  and `vercor` reexport the component-author facade.
  `vercor.components.contracts` owns public author-facing contract types and
  context aliases, `vercor.components.base` owns only the abstract
  differentiable `Component` contract, `vercor.components.data` owns
  `DataComponent`, `vercor.components.host` owns `HostRuntimeComponent`, and
  module-level factory helpers live in `vercor.components.factories`.
  Field-name de-duplication lives in private
  `vercor.components._field_names`, and component authoring methods for field
  declarations, setup seeding, and settings updates live in private
  `vercor.components._field_authoring`. Lifecycle hook storage and hook
  installation live behind the typed private `ComponentLifecycleOwner`
  boundary in `vercor.components._lifecycle`; default lifecycle dispatch lives
  in private `vercor.components._lifecycle_api`, and factory-installed hooks are
  stored in one private `ComponentLifecycleHooks` container rather than as
  ad-hoc component attributes. Author-value normalization lives in private
  `vercor.components._contracts`; callable signature adaptation, shared
  callable construction metadata, and shared callable runtime mechanics live in
  private `vercor.components._callable_wrappers`, which carries lifecycle hooks
  as that container and delegates hook precedence/default payload fallback to
  the lifecycle mixin. The concrete callable-backed
  differentiable wrapper is owned by `vercor.components.base`, and the concrete
  callable-backed host wrapper is owned by `vercor.components.host`, keeping
  each runtime kind beside its public abstract base. Private helper modules
  share component-owned structural protocols from
  `vercor.components._protocols` instead of importing the public
  `Component` class for helper annotations, so helper boundaries stay decoupled
  from the concrete base implementation. Runtime-oriented component protocols
  expose only name, grid, settings, and declared field contracts; setup data
  storage stays on concrete component classes. Execution protocols in the same
  private module expose only runtime step methods, and host-runtime detection is
  a runtime-checkable structural protocol rather than a concrete component class
  check. Component-facing
  runtime-field adapters live in private `vercor.components._runtime_fields`,
  component-facing runtime-field convenience methods live in private
  `vercor.components._runtime_access`, and component-facing required-field
  validation lives in private `vercor.components._runtime_validation`.
  Component host/scanned execution policy lives in internal
  `vercor.components.runtime_execution`, and setup validation lives in internal
  `vercor.components.setup_validation`, giving runtime modules explicit
  component-owned bridge modules instead of importing private component
  internals. These internals are not exported from `vercor.components`.
  Subclasses should call
  the base constructor so `name`,
  `grid`, `data`, and a component-owned `VercorSettings` container are available
  during initialization, execution, and finalization. `Component.data` is a
  grid-field store, not a
  general metadata store: all entries must use one of the canonical layouts
  `(nLat, nLon)`, `(nTime, nLat, nLon)`, `(nLev, nLat, nLon)`, or
  `(nTime, nLev, nLat, nLon)`. Setup and runtime-state creation validate this
  contract before traced execution. Subclasses should seed fields with
  `seed_field()` or `seed_fields()` for scalar or array-like author values, or
  with the explicit zero/constant helpers when that reads better, rather than
  mutating `data` directly; step methods
  should read fields with `runtime_field()`, `runtime_fields()`,
  `runtime_field_or()`, or `runtime_field_or_zeros_like()` and return updates
  with `with_runtime_fields()` where possible. These component helpers are
  author-facing adapters in `vercor.components._runtime_fields` over
  `RuntimeFieldStore` membership, mapping, fallback, and existing-field
  replacement mechanics owned by the runtime.
  When a step also needs to replace runtime payload, `apply_step_result()`
  applies either a field mapping or `ComponentStepResult` through the same
  validated update path used by callable wrappers.
  `seed_declared_defaults()` seeds fields from a component's declared
  defaults, and the base `initialize()` hook now does this automatically when
  subclasses do not need custom setup. Prefill hooks should use
  `prefill_runtime_fields()` for ordinary output/default fields. Non-grid
  metadata such as hybrid-level coefficients belongs on component attributes or
  runtime payloads. Factory-created setup adapters should put non-runtime setup
  metadata in
  `Component.setup_metadata` rather than attaching ad-hoc attributes to the
  component object. Examples include forcing-file provenance and diagnostic
  coefficients that should not enter runtime field validation or JAX scan state.
  `Component` for differentiable active models and implement
  `step_runtime_state()`. Use `DataComponent` for forcing/static data adapters
  that intentionally keep the shared no-op runtime step and do not create
  plotting-only runtime fields. Derived diagnostics, such as a combined land/sea
  surface temperature used only for plots, belong in diagnostics or setups.
  Use `HostRuntimeComponent` for non-differentiable adapters and implement
  `step_host_runtime_state()`; host-backed adapters must run through
  `Coupler.run()` so VerCOR can select the Python host runtime path. Optional
  hooks include `initialize()`, `create_runtime_payload()`,
  `prefill_runtime_state_fields()`, and `validate_runtime_state()`. Callable
  wrappers may accept `(fields)`, `(fields, context)`, or
  `(fields, context, payload)` and return either a field-update mapping or
  `ComponentStepResult(fields, payload)` when the runtime payload must
  be replaced. Callable-backed differentiable and host components share
  signature normalization and step-result application helpers, while
  `Component.from_model()` and `HostRuntimeComponent.from_model()` construct
  their own private runtime-kind wrappers directly. Both declare their runtime
  contract with the same `ComponentFieldSpec` path used by subclasses, and
  apply step results through the runtime-owned field replacement helpers.
  Runtime prefill and validation depend only on `inputs`, `outputs`, and
  `default_fields`.
  These helpers still enforce the same stable runtime-state
  contract: updated fields must already exist through seeded data, declared
  outputs/defaults, or exchange prefill, and scanned payload pytrees must keep
  stable shapes and dtypes.
- Internal runtime API: the `vercor.runtime` package owns
  `RuntimeFieldStore`, `RuntimeComponentState`, `RuntimeCouplerState`, runtime
  contexts, dispatch contexts, and runtime helper functions. These containers
  carry immutable arrays and static metadata through JAX tracing. They are
  required for differentiability and stable scan carry structure. Runtime
  field stores live in `vercor.runtime.stores` and own name membership, mapping
  roundtrips, fallback reads, and replacement of existing fields while
  preserving established dtypes. Import/export contract construction lives in
  `vercor.runtime.contracts`, exchange dispatch lives in
  `vercor.runtime.exchange_dispatch`, static dispatch context construction
  lives in `vercor.runtime.dispatch_context`, runtime step metadata lives in
  `vercor.runtime.time`, component state creation lives in
  `vercor.runtime.component_state`, field receive/send mechanics live in
  `vercor.runtime.field_transfer`, component/store runtime validation lives in
  `vercor.runtime.validation`, and configured runtime-state/topology validation
  lives in `vercor.runtime.state_validation`. Runtime coupler-state assembly
  and runtime-contract refresh live in `vercor.runtime.coupler_state`; exchange
  component-name validation and component lookup live in
  `vercor.runtime.component_topology`; runtime topology data contracts live in
  `vercor.runtime.topology_state`, including grouped `RuntimeTopologyMaps`,
  `SurfaceExchangeMasks`, and `ExchangeTopologyState`. Generic exchange
  regridder/identity-mask map construction lives in
  `vercor.runtime.exchange_topology`, while ATM/OCN/LND surface-mask creation,
  validation, and bilinear exchange patching live in
  `vercor.runtime.surface_masks`. `vercor.runtime.topology` remains the
  orchestration boundary that composes those owners and returns the explicit
  topology state for the runtime facade to store. Setup-time component
  precision synchronization, initialization context construction, component
  setup validation, runtime contract validation, and topology handoff live in
  `vercor.runtime.initialization`. Runtime state preparation, the
  `PreparedRuntimeState` bundle, contract refresh for prepared states,
  validation, and initial outgoing-store priming live in
  `vercor.runtime.preparation`; `vercor.runtime.facade` reexports these helpers
  for the coupler-facing runtime boundary but does not own their implementation.
  Shared compiled-runtime aliases live in `vercor.runtime.compilation`. Frozen
  `RuntimeRunContext` execution inputs live in `vercor.runtime.run_context`,
  which also owns context-derived compiled-runtime cache keys.
  `CompiledRuntimeCache` storage and JIT wrapping live in
  `vercor.runtime.cache`. The run context carries the cache owner rather than a
  mutable cache mapping, while the cache owner remains independent of run
  context internals.
  Shared host/scanned progress messages plus traced callbacks live in
  `vercor.runtime.progress`, and the interrupt controller lives in
  `vercor.runtime.interrupts`. Mutable per-coupler runtime resources live in
  `vercor.runtime.resources.CouplerRuntimeResources`, which owns private
  exchange topology maps, refreshed runtime contracts, a compiled-runtime cache
  owner, and the interrupt controller. Runtime facade and preparation code use
  explicit holder accessors and replacement methods rather than raw resource
  dictionaries, and focused tests install synthetic topology through the same
  grouped topology-map replacement helper. Compiled-runtime cache clearing,
  counting, and test inspection delegate through the cache owner rather than
  exposing its dictionary. `Coupler` passes repeated
  runtime inputs through the internal
  `vercor.runtime.facade.RuntimeFacadeInputs` bundle and exposes only
  `clear_runtime_cache()` plus `runtime_cache_entry_count()` as a small public
  profiling facade; there are no private compatibility aliases for individual
  runtime maps or caches.
  Host/scanned runtime loops, run-mode
  selection, compiled scanned dispatch, donation checks, and interrupt
  translation live in `vercor.runtime.runner`. High-level runtime orchestration
  for the public `Coupler` facade lives in `vercor.runtime.facade`: runtime-state
  preparation, dispatch/run context construction, host/scanned execution,
  runtime views, and final output delegation enter through this module instead
  of direct `Coupler` imports of runtime implementation helpers.
  Runtime component metadata and read-only field
  resolution for diagnostics/output live in `vercor.runtime.views`;
  `RuntimeComponentView`, `runtime_field_candidates(...)`, and
  `runtime_field(...)` own data/incoming/outgoing lookup for explicit views and
  compatible runtime states. `Coupler` exposes `runtime_component_view()` and
  `runtime_component_views()` as the public facade for creating those views.
  Final runtime output iteration, output-mask naming/selection, and
  view writing live in `vercor.output`, with `vercor.runtime.facade` validating
  and delegating output writes for `Coupler.finalize()`. `Coupler` delegates to
  the runtime facade and
  remains the public setup/finalization facade rather than the owner of runtime
  adapter mechanics.
  The `vercor.runtime` package initializer does not reexport runtime containers
  or helper functions; internal code should import from the focused owner
  modules listed above.
  payload pytrees carried through `jax.lax.scan` must preserve every leaf's
  shape and dtype between input and output; per-step slices or adapted forcing
  objects should be local values unless they are shape-stable runtime state.
  Internal runtime containers are not exported from the package top level.

### Setup adapters and shared ownership

Reusable concrete adapters live under the canonical packaged namespace
`vercor.setups`. Runnable assembly scripts live under `examples/`; in-repo code
should not depend on a top-level `setups` package. Setup adapters use
`ComponentSetupContext`, `ComponentStepContext`, and plain runtime-array
mappings at their author boundary instead of importing runtime context/store
internals directly. Shared setup orchestration helpers in
`vercor.setups.coupler_helpers` own component registration and compact
`ExchangeSpec` recipes, so examples share source/destination/regridder wiring
instead of repeating raw `Exchange(...)` construction. Public exchange
configuration types, including `ExchangeField` and `RegridderFactory`, are
owned by `vercor.exchange` and imported by setup helper modules rather than
duplicated beside recipes.

Core helper ownership follows the same boundary. Calendar constants,
model-calendar datetime values, leap-year logic, and month/day conversion live
in `vercor.calendar`. Daily forcing-index policy, including noleap and 360-day
calendar mapping to forcing-file day indexes, lives in `vercor.forcing_index`;
`vercor.calendar` preserves thin compatibility delegates for the historic import
path. The canonical exchange field vocabulary lives in `vercor.field_names`.
Rectilinear grid construction, center-to-edge geometry, and grid identity checks
live in `vercor.grid_geometry`; mask math lives in `vercor.grid_masks`, while
default component-topology name validation and lookup are private to
`vercor.runtime.component_topology`. Generic hybrid/sigma-coordinate pressure
and altitude helpers live in `vercor.fluxes.vertical_coordinates`, and generic
PyTree transforms live in `vercor.pytree_utils`.
Adapter-specific runtime and file-output policy lives beside adapters in focused
helpers instead of in factory/bootstrap modules. Exported adapter helpers use
plain public package-internal names in their owner modules; underscored helpers
remain local implementation details and are not listed in external adapter
`__all__` exports. JAXGCM runtime payload, prefill, validation, stepping, and
host recording live in
`vercor.setups.external.jax_gcm_runtime`, which consumes the setup object through
a private protocol rather than an unbounded state object.
`vercor.setups.external.jax_gcm_state` owns JAXGCM setup-time model resources,
spinup policy, lifecycle callbacks, and the public `JCMState` bundle reexported
by the factory module. `vercor.setups.external.jax_gcm` remains a thin public
factory that constructs setup state and binds named lifecycle callbacks without
owning runtime payload or setup-state internals. JAXGCM output cadence and
NetCDF writing live in `vercor.setups.external.jax_gcm_output`, while
surface-temperature cleanup and output-field mapping live in
`vercor.setups.external.jax_gcm_fields`. Veros host-runtime flux application and
substep orchestration live in
`vercor.setups.external.veros_runtime` behind a private runtime-state protocol.
`vercor.setups.external.veros_gcm_state` owns Veros setup-time model resources,
spinup policy, grid derivation, and lifecycle callbacks, while
`vercor.setups.external.veros_gcm` remains the thin public factory.
Veros host-state mutation helpers and the named tuple-compatible
`VerosForcingFields` container live in `vercor.setups.external.veros_state`.
Veros backend settings are imported only inside the explicit configuration
function so setup modules preserve lazy optional-dependency boundaries. CAMulator
prediction-block and runtime step orchestration live in
`vercor.setups.external.camulator_runtime` behind a private runtime-state
protocol, with tensor staging in `vercor.setups.external.camulator_tensors` and
field mapping in `vercor.setups.external.camulator_fields`. CAMulator CREDIT
output writing lives in `vercor.setups.external.camulator_output`.

`vercor.assets` owns generic cache, download, and checksum validation only.
Concrete forcing product registries and `get_forcing_data(...)` defaults live
with setup data adapters in `vercor.setups.data.assets`. Diagnostics are split
into `vercor.diagnostics.fields`, `vercor.diagnostics.tables`, and
`vercor.diagnostics.plotting`, with field lookup delegated to
`vercor.runtime.views` and `vercor.diagnostics` preserving the public reexport
surface.

CAMulator runtime field contracts, optional-dependency loading, forcing cursors,
tensor accessors, runtime stepping, output, wind filtering, land forcing, and
initialization are split across
`vercor.setups.external.camulator_contracts`,
`vercor.setups.external.camulator_imports`,
`vercor.setups.external.camulator_forcing`,
`vercor.setups.external.camulator_tensors`,
`vercor.setups.external.camulator_stepper`,
`vercor.setups.external.camulator_runtime`,
`vercor.setups.external.camulator_output`,
`vercor.setups.external.camulator_wind_filter`,
`vercor.setups.external.camulator_land`, and
`vercor.setups.external.camulator_init`. New code should import directly from
these focused modules; the old one-hop CAMulator state and wind-filter facades
have been removed. CAMulator tensor channel metadata is stored internally as
typed `TensorVariableIndex` values in `camulator_tensors`, while
`StateVariableAccessor.get_var_index(...)` is the canonical metadata lookup for
callers that inspect tensor channels. CAMulator wind-filter configuration
validates with explicit exceptions and avoids mutable function defaults so tests
and callers see stable failure modes independent of Python optimization
settings.

### Settings container

VerCOR uses one metadata-backed `VercorSettings` class for both coupler-level
and component-level settings. `vercor.settings.DEFAULT_SETTINGS` stores the
defaults as `Settings(value, description, units)` namedtuple records; unitless
settings use `"-"` for units. Each `Coupler` and each `Component` receives an
independent `VercorSettings()` instance populated from those defaults at
construction time, so setup-time changes on one owner do not leak into another.

For backward-compatible call sites, `settings.enable_x64` and similar attribute
reads resolve setting values dynamically through `__getattr__`, and assigning an
existing attribute updates only that value through `__setattr__`. Known default
settings are declared as class-level annotations so static type checkers retain
useful types without per-setting runtime property descriptors. New custom
settings must be introduced explicitly with `add_setting()` or passed as keyword
arguments to `VercorSettings(...)`; existing settings should be updated with
`set_value()` where production code is making an intentional configuration
change. `dir(settings)` includes default and custom setting names for
introspection. The obsolete `ComponentSettings` alias has been removed; use
`VercorSettings` directly.

### Precision and dtype policy

VerCOR-owned array dtypes are centralized in `vercor.dtypes`. Real-valued JAX
and NumPy arrays use the `VercorSettings.enable_x64` precision switch whenever a
settings object is available: `False` maps to 32-bit real arrays and `True` maps
to 64-bit real arrays. `Coupler.initialize()` treats the coupler setting as the
run-level precision policy, synchronizes component settings to that policy, and
recasts component-owned grid/data arrays before runtime state creation. Helpers
that create arrays without a settings object follow the active JAX global
`jax_enable_x64` configuration; conversion helpers preserve an already-typed
real array when no settings object is supplied.
Integer/index arrays use the canonical 32-bit index dtype in both
real-precision modes to keep sparse metadata and interpolation indices compact.

Production kernels and adapters should use the dtype helpers rather than
hard-coded `jnp.float64`, `jnp.float32`, `jnp.float_`, `jnp.int64`, or
`jnp.int32` annotations. NumPy remains restricted to explicit host and dtype
boundaries.

### Logging across JAX runtime transforms

The coupler logger is callback-backed through `jax.debug.callback`, so runtime
hooks can emit diagnostics inside `jax.lax.scan`, `jax.jit`, and automatic
differentiation transforms. Coupler logging levels are configured at
instantiation with `Coupler(..., log_level=...)`; disabled levels are filtered
before callbacks enter the traced graph. Runtime hooks should pass traced values
as logger arguments, for example `logger.info("Mean SST: {}", jnp.mean(sst))`,
instead of converting tracers with `float(...)` or `int(...)`.
`vercor.jax_logging` is the public compatibility facade for logging contracts,
constants, setup helpers, host emission, and the callback-backed logger. The
implementation is split across private `vercor._logging` owner modules:
`config` owns canonical Python logger configuration, `protocols` owns
logger-like contracts and level checks, `host` owns host-side formatting and
emission, and `callback` owns traced-value partitioning plus
`JaxCallbackLogger`. Production code outside the facade should import from
`vercor.jax_logging`, not from `vercor._logging`.
Initialization, runtime, and finalization helpers that are reached outside a
coupler context use the default `VerCOR` Python logger from
`vercor.jax_logging.get_default_logger()`. Helpers reached from
`Coupler.initialize()`, `Coupler.run()`, or component runtime contexts receive
the coupler logger explicitly instead of writing directly to stdout.
The host and scanned coupler runtime paths share progress formatting and traced
callback helpers in `vercor.runtime.progress`. The scanned path precomputes
datetime and timestep labels on the host, then selects the per-step label inside
ordered callbacks so progress logging remains traceable without putting Python
datetime objects in the scan carry.

### Runtime interruption across host and scanned integrations

`Coupler.run()` provides an internal runtime interrupt controller to
`vercor.runtime.runner`, which owns host and scanned runtime cancellation
checkpoints. During a run, `SIGINT`, `SIGTERM`, and `SIGTSTP` request graceful
runtime cancellation and are restored to their previous handlers when the run
exits. The host runtime checks the controller at step and component boundaries.
The controller also installs a temporary nonblocking wakeup fd so signals
delivered while the main thread is inside a compiled XLA call are recorded
before Python signal handlers run. The JIT-scanned runtime inserts explicit
ordered `jax.debug.callback` checkpoints at the same boundaries; those callbacks
drain the wakeup fd and observe terminal shortcut commands independently of
logging level. Interrupt callback failures are translated back to a
`KeyboardInterrupt` subclass, while unrelated JAX runtime failures are
preserved.

---

## 3. Module Specifications

### 3.1 Constants and Parameters

**File**: `constants.py`

For physical constants, such as gravitational acceleration, gas constant, etc.

**File**: `parameters.py`

For runtime parameters, such as coupled run identifier, precision, time interpolation type,
type of year (leap, noleap, 360day), etc.

Two parameter containers:

```python
@dataclass(frozen=True)
class PhysicsParameters:
    """All fields are JAX-traceable floats."""
    # Scalars
    gravity: float
    rhoAir: float
    rgas: float
    latvap: float
    zref: float
    mwdair: float
    ...


@dataclass(frozen=True)
class ControlParameters:
    """Control parameters. NOT traced by JAX (static)."""
    get_field_time_slice: bool
    apply_time_interpolation: bool
    enable_x64: bool
    identifier: str
    ...
```

The split between `PhysicsParameters` (traced) and `ControlParameters` (static) is critical:
JAX traces through `PhysicsParameters` for AD, while `ControlParameters` controls array
shapes and solver settings that must be compile-time constants.

## 4. Validation and Testing

### Test design philosophy

The test harness is the most important part of this project. Without high-quality
tests, autonomous agents will solve the wrong problem.

1. **Tests must be nearly perfect.** Agents will optimize for whatever the tests
   measure. If a test is wrong or has loose tolerances, agents will produce code
   that passes the bad test but gives wrong physics. Invest more time in the test
   harness than in the code it tests.

2. **Tests must give concise, actionable feedback.** Print the max relative error
   and where it occurs, not full arrays. Pre-compute aggregate statistics.
   Log details to files, not stdout, to avoid context window pollution.

3. **Tests must be fast by default.** Every test file supports a `--fast` mode
   (~10% subsample) for rapid iteration. Full validation runs before commits.

4. **Tests must decompose monolithic tasks.** Test sub-components independently:
   - Regridding with mock meshes and fields
   - Different clock functionalities and options
   - Coupler stepping with mock models and fields
   - Fluxes computations with mock meshes and fields
   - Exchanges of fields between models with mock models, meshes and fields
   - Input / Output
   - etc.
   This lets different agents work on different subsystems.

5. **Tests must enable bisection.** When solution disagrees, we need to find the
   first module in the pipeline that diverges from original code. The test suite
   should make this easy by testing every intermediate quantity, not just
   the final output. This is the "oracle bisection" pattern.

### Test hierarchy

We use a layered testing approach, from unit tests to full pipeline validation.

**level 1**: Unit tests (fast)

**level 2**: Module tests.
For each module, pre-generate reference data and check agreement.

**level 3**: Gradient tests

For each module, verify that AD gradients match finite-difference gradients.

**level 5**: End-to-end integration tests (from runnable scripts in `examples/`)

---

## 5. Performance Strategy

### JIT compilation

The entire `run_simulation()` function should be JIT-compiled:

Since `ControlParameters` is static (controls array shapes), it should be passed via
`static_argnums` or as a `static_field` in an Equinox module.

First call will be slow (~30-60s for XLA compilation). Subsequent calls with the
same shapes will be fast.
