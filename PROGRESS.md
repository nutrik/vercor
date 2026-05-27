# VerCOR Progress

This file is the short orientation log for active development. The detailed
historical execution transcript was archived verbatim in
`docs/progress-archive-2026-04-23-to-2026-05-15.md`.

Use this file to decide what to do next. Use the archive when you need exact
historical commands, failure messages, or detailed validation notes.

## Current Status

- Latest archived full validation status: passing as of 2026-05-15.
- Latest archived fast validation status: `pytest tests/ -q --fast --tb=short`
  passed as of 2026-05-15.
- Latest archived static checks: Black, flake8, and mypy passed as of
  2026-05-15.
- Latest local organization-refactor validation: Black, flake8, mypy, and fast
  pytest passed as of 2026-05-26.
- Latest local compatibility-facade cleanup validation: Black, flake8, mypy,
  focused fast pytest, full fast pytest, and full pytest passed as of
  2026-05-27.
- Latest local boundary-cohesion validation: Black, flake8, mypy, focused fast
  pytest, full fast pytest, and full pytest passed as of 2026-05-27.
- Latest local boundary-import validation: Black, flake8, mypy, focused fast
  pytest, full fast pytest, and full pytest passed as of 2026-05-27.
- Latest local cohesion-boundary implementation validation: Black, flake8,
  mypy, focused fast pytest, full fast pytest, and full pytest passed as of
  2026-05-27.
- Latest local runtime-dispatch-boundary validation: Black, flake8, mypy,
  focused fast pytest, full fast pytest, and full pytest passed as of
  2026-05-27.
- Latest local runtime-run-boundary validation: Black, flake8, mypy, focused
  fast pytest, full fast pytest, and full pytest passed as of 2026-05-27.
- Latest local runtime-view/component-boundary validation: Black, flake8, mypy,
  focused fast pytest, full fast pytest, and full pytest passed as of
  2026-05-27.
- Latest local diagnostics-runtime-view validation: Black, flake8, mypy,
  focused fast pytest, full fast pytest, and full pytest passed as of
  2026-05-27.
- No active `IN PROGRESS` task is recorded in the archived log.
- No current blocker is recorded in the archived log.
- Recurring known warning: Black may emit the existing Python 3.13 versus
  target-3.14 safety-check warning while still completing successfully.
- Recurring known warning: the JAXGCM runtime gradient test may emit the
  existing JAX dtype promotion `FutureWarning` while the suite still passes.

## Next Session Checklist

1. Read `DESIGN.md` for architecture and public/runtime boundary context.
2. Read `DEPENDENCIES.md` for module ordering before changing code.
3. Run:

   ```bash
   conda run -n scipy pytest tests/ -q --fast --tb=short
   ```

4. If the fast suite passes, pick the next unchecked item from this file or the
   next failing focused test.
5. If the fast suite fails, work from the first failing test and record the
   root cause and fix here.
6. Before stopping, update this file with a compact summary, not a full command
   transcript.

## Recent Work

### 2026-05-27: Diagnostics Runtime View Boundary Refactor

- Added runtime-owned `runtime_field_candidates(...)` and `runtime_field(...)`
  helpers in `vercor.runtime.views`, and routed `RuntimeComponentView` read
  helpers through them.
- Updated diagnostics to use the runtime-view field lookup boundary instead of
  reaching into runtime stores with `.data.get(...)` or `getattr(...)`, while
  preserving `component_vector_speed(...)` compatibility with runtime states.
- Cleaned `examples/run_data_driver.py` diagnostics wiring and kept its
  component typing on the public top-level facade. A first full-suite run caught
  the direct `vercor.components` example import boundary regression; the example
  now imports `Component` from `vercor`.
- Updated boundary tests, `DESIGN.md`, and `DEPENDENCIES.md` for the shared
  runtime field-resolution ownership.
- Validation run for this change:
  focused diagnostics/runtime-view fast pytest,
  `conda run -n scipy pytest tests/ -q --fast --tb=short`,
  `conda run -n scipy black vercor examples tests`,
  `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`,
  `conda run -n scipy mypy vercor examples tests`, and
  `conda run -n scipy pytest tests/ -q --tb=short`. The existing Black
  Python 3.13/target-3.14 warning and JAX dtype-promotion warning remain.

### 2026-05-27: Runtime View and Component Boundary Refactor

- Moved component setup validation and component host/scanned execution policy
  into explicit component-owned bridge modules:
  `vercor.components.setup_validation` and
  `vercor.components.runtime_execution`, removing runtime imports of the old
  private component helper modules.
- Added read helpers to `RuntimeComponentView` and routed diagnostics/output
  field access through that view abstraction instead of iterating runtime store
  internals directly.
- Added `Coupler.runtime_component_views(...)` and updated multi-view examples
  to reuse that public facade.
- Updated boundary tests, `DESIGN.md`, and `DEPENDENCIES.md` for the new
  ownership map.
- Validation run for this change:
  focused boundary/view fast pytest,
  `conda run -n scipy pytest tests/ -q --fast --tb=short`,
  `conda run -n scipy black vercor examples tests`,
  `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`,
  `conda run -n scipy mypy vercor examples tests`, and
  `conda run -n scipy pytest tests/ -q --tb=short`. The existing Black
  Python 3.13/target-3.14 warning and JAX dtype-promotion warning remain.

### 2026-05-27: Runtime Run Boundary Refactor

- Moved `RuntimeRunContext` and the compiled-runtime type alias into
  `vercor.runtime.run_context`, leaving static topology on
  `RuntimeDispatchContext` instead of duplicating it in the run context.
- Moved compiled-runtime cache-key and JIT wrapping policy into
  `vercor.runtime.cache`, and moved host/scanned progress formatting plus JAX
  callback logging helpers into `vercor.runtime.progress`.
- Slimmed `vercor.runtime.runner` to run-mode selection, host/scanned loops,
  donation checks, and interrupt translation while preserving `Coupler.run()`
  behavior and runtime PyTree shapes.
- Updated boundary tests, `DESIGN.md`, and `DEPENDENCIES.md` for the new
  ownership map.
- Validation run for this change:
  focused runtime fast pytest,
  `conda run -n scipy pytest tests/ -q --fast --tb=short`,
  `conda run -n scipy black vercor examples tests`,
  `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`,
  `conda run -n scipy mypy vercor examples tests`, and
  `conda run -n scipy pytest tests/ -q --tb=short`. The existing Black
  Python 3.13/target-3.14 warning and JAX dtype-promotion warning remain.

### 2026-05-27: Runtime Dispatch Boundary Refactor

- Moved static runtime dispatch context construction into
  `vercor.runtime.dispatch_context`, leaving `vercor.runtime.coupler_state`
  focused on runtime state assembly, contract refresh, validation, and output
  masks.
- Added private `vercor.components._runtime_execution` for host-component
  detection and host/scanned component step selection, so `vercor.runtime.driver`
  no longer owns `HostRuntimeComponent` classification.
- Updated `Coupler`, runtime runner/driver imports, boundary tests, and the
  architecture ownership docs for the new dispatch and component-execution
  boundaries.
- Validation run for this change:
  focused runtime/component fast pytest,
  `conda run -n scipy pytest tests/ -q --fast --tb=short`,
  `conda run -n scipy black vercor examples tests`,
  `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`,
  `conda run -n scipy mypy vercor examples tests`, and
  `conda run -n scipy pytest tests/ -q --tb=short`. The existing Black
  Python 3.13/target-3.14 warning and JAX dtype-promotion warning remain.

### 2026-05-27: Cohesion and Boundary Refactor Implementation

- Made component contract output merging pure, stored factory lifecycle hooks in
  a single private `ComponentLifecycleHooks` container, and moved
  component-facing runtime required-field validation into
  `vercor.components._runtime_validation`.
- Added runtime-owned contract refresh, bundled runner execution inputs in
  `RuntimeRunContext`, delegated final-output iteration to `vercor.output`, and
  kept `vercor.coupler.setup_logger` private to the facade implementation.
- Added `ExchangeSpec`, `build_exchanges()`, and `add_exchange_specs()` for
  setup recipes, then migrated examples and the profiling harness away from
  repeated raw `Exchange(...)` wiring.
- Lazied the paired JCM setup helper's optional JCM imports, moved JCM land
  type-only optional imports behind `TYPE_CHECKING`, and extracted focused
  JAXGCM host-recording and CAMulator prediction-block helpers.
- Updated `DESIGN.md`, `DEPENDENCIES.md`, and boundary tests for the new
  ownership map.
- Validation run for this change:
  focused component/runtime/API fast pytest,
  focused setup/CAMulator/external fast pytest,
  focused runtime-cache fast pytest,
  `conda run -n scipy pytest tests/ -q --fast --tb=short`,
  `conda run -n scipy black vercor examples tests`,
  `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`,
  `conda run -n scipy mypy vercor examples tests`, and
  `conda run -n scipy pytest tests/ -q --tb=short`. The existing Black
  Python 3.13/target-3.14 warning and JAX dtype-promotion warning remain.

### 2026-05-27: Runtime Component Boundary Import Refactor

- Moved component-facing required runtime field validation into
  `vercor.components._runtime_fields`, removing its hidden dependency on
  `vercor.runtime.validation` while preserving the existing `CouplerError`
  messages for missing and non-canonical fields.
- Converted annotation-only `Component` imports in the public coupler facade and
  setup/runtime helper modules to `TYPE_CHECKING` imports, keeping runtime
  imports focused on behavior dependencies.
- Added boundary tests that pin `_runtime_fields` away from runtime validation
  internals and guard the planned type-only import shape for coupler/runtime
  modules.
- Validation run for this change:
  `conda run -n scipy pytest tests/test_api_boundaries.py tests/test_component_base_coverage.py tests/test_runtime_state.py -q --fast --tb=short`,
  `conda run -n scipy pytest tests/ -q --fast --tb=short`,
  `conda run -n scipy black vercor examples tests`,
  `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`,
  `conda run -n scipy mypy vercor examples tests`, and
  `conda run -n scipy pytest tests/ -q --tb=short`. The existing Black
  Python 3.13/target-3.14 warning and JAX dtype-promotion warning remain.

### 2026-05-27: Runtime Facade Cohesion Refactor

- Added `vercor.runtime.initialization` as the setup-time boundary for run
  precision synchronization, component initialization contexts, component setup
  validation, runtime contract validation, and exchange-topology handoff.
- Added explicit `ExchangeTopologyState` and `build_exchange_topology(...)` so
  exchange regridders and masks are assembled through a returned state object
  instead of only mutating caller-owned dictionaries.
- Slimmed `Coupler.initialize()` to delegate initialization wiring while
  preserving existing private runtime-state helpers and topology map
  compatibility attributes.
- Validation run for this change:
  `conda run -n scipy pytest tests/test_coupler_coverage.py tests/test_runtime_state.py -q --fast --tb=short`,
  `conda run -n scipy pytest tests/ -q --fast --tb=short`,
  `conda run -n scipy black vercor examples tests`,
  `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`,
  `conda run -n scipy mypy vercor examples tests`, and
  `conda run -n scipy pytest tests/ -q --tb=short`. The existing Black
  Python 3.13/target-3.14 warning and JAX dtype-promotion warning remain.

### 2026-05-27: Boundary Cohesion Refactor

- Moved component lifecycle hook type aliases and hook installation into
  `vercor.components._lifecycle`, leaving `vercor.components.factories` focused
  on public helper construction and breaking the top-level
  `vercor.components` import cycle.
- Switched base component runtime-field adapters to a direct private-module
  import instead of importing through the package namespace.
- Added shared setup-data helpers for positive binary masks and 2D/time-last
  surface-field canonicalization, then routed ERA5 ocean, ERA-Interim ocean,
  and JCM land preparation through those helpers.
- Updated boundary and setup-data tests to cover lifecycle ownership, component
  package import cycles, and shared field-helper behavior.
- Validation run for this change:
  `conda run -n scipy black vercor examples tests`,
  `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`,
  `conda run -n scipy mypy vercor examples tests`, focused API-boundary and
  data-component kernel tests,
  `conda run -n scipy pytest tests/ -q --fast --tb=short`, and
  `conda run -n scipy pytest tests/ -q --tb=short`. The existing Black
  Python 3.13/target-3.14 warning and JAX dtype-promotion warning remain.

### 2026-05-27: Deprecated Compatibility Import Facade Removal

- Removed obsolete one-hop compatibility modules:
  `vercor.runtime.components`, `vercor.setups.data.camulator_land`,
  `vercor.setups.data.forcing`, `vercor.setups.external.camulator_state`,
  `vercor.setups.external.windpp`, and `vercor.setups.jax_array_helpers`.
- Routed remaining imports to canonical owners: runtime component-state,
  field-transfer, and validation helpers; `vercor.forcing_data.read_forcing`;
  calendar datetime classes; vertical-coordinate helpers; grid identity; and
  exchange field names.
- Removed compatibility reexports from `vercor.clock`, `vercor.exchange`,
  `vercor.grid_masks`, and `vercor.fluxes.utilities` while preserving stable
  package aggregators, `ComponentSettings`, settings attribute access, and
  `ComponentForcingData`.
- Updated boundary tests, `DESIGN.md`, and `DEPENDENCIES.md` for canonical
  ownership. During full validation, corrected a stale Veros runtime-settings
  boundary assertion to point at `vercor.setups.external.veros_setup`.
- Validation run for this change:
  `conda run -n scipy black vercor examples tests`,
  `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`,
  `conda run -n scipy mypy vercor examples tests`,
  focused cleanup tests,
  `conda run -n scipy pytest tests/ -q --fast --tb=short`, and
  `conda run -n scipy pytest tests/ -q --tb=short`. The existing Black
  Python 3.13/target-3.14 warning and JAX dtype-promotion warning remain.

### 2026-05-26: Code Organization Audit Implementation

- Split physical defaults into `vercor.physical_constants` and composed them
  into `VercorSettings`, keeping static runtime controls in `settings.py`.
- Replaced concrete interpolator imports in `vercor.regridders.base` with a
  small scalar/vector interpolation protocol.
- Moved default topology component-name validation into
  `vercor.runtime.topology` so `Coupler` delegates topology policy.
- Split broad external adapters:
  `jax_gcm_fields.py` owns JCM field mapping and surface-temperature helpers,
  `camulator_fields.py`/`camulator_tensors.py`/`camulator_init.py`/
  `camulator_runtime_settings.py` own CAMulator field, tensor, init-noise, and
  environment setup helpers, and `veros_setup.py`/`veros_fluxes.py`/
  `veros_state.py` own Veros setup, flux, and host-state helpers.
- Kept adapter factory/state compatibility while removing moved helper symbols
  from old external adapter facades; narrowed
  `vercor.setups.data.camulator_land` to the public land factory only.
- Centralized common example exchange field lists in
  `vercor.setups.exchange_recipes`, added slab land/ocean recipe separation,
  and widened `Exchange.field_names` to accept immutable recipe sequences.
- Updated `DEPENDENCIES.md` and ownership boundary tests for the new module map.
- Validation run for this change:
  `conda run -n scipy black vercor examples tests`,
  `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`,
  `conda run -n scipy mypy vercor examples tests`,
  `conda run -n scipy pytest tests/test_api_boundaries.py -q --fast --tb=short`,
  focused external adapter tests, and
  `conda run -n scipy pytest tests/ -q --fast --tb=short`. The existing Black
  Python 3.13/target-3.14 warning remains. Full pytest was not rerun for this
  refactor.

### 2026-05-26: Ownership Boundary Refactor Follow-Up

- Moved model-calendar datetime values into `vercor.calendar` while preserving
  `vercor.clock` compatibility reexports.
- Split canonical exchange-field vocabulary into `vercor.field_names`, unified
  grid identity in `vercor.grid_geometry`, and removed the runtime daily-index
  wrapper in favor of `vercor.calendar.daily_forcing_index`.
- Consolidated hybrid/sigma pressure and altitude helpers in
  `vercor.fluxes.vertical_coordinates`, leaving old flux utility import paths as
  compatibility aliases.
- Moved setup helper ownership to `vercor.host_arrays` and
  `vercor.diagnostics.fields`; moved CAMulator land, CAMulator output, CAMulator
  wind filtering, and JAXGCM output helpers under `vercor.setups.external`.
- Updated `DESIGN.md`, `DEPENDENCIES.md`, examples, and boundary tests for the
  new ownership map.
- Required validation passed:
  `conda run -n scipy black vercor examples tests`,
  `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`,
  `conda run -n scipy mypy vercor examples tests`,
  `conda run -n scipy pytest tests/ -q --fast --tb=short`, and
  `conda run -n scipy pytest tests/ -q --tb=short`. The existing Black
  Python 3.13/target-3.14 warning and JAX dtype-promotion warning remain.

### 2026-05-26: Refactoring Campaign Ownership Split

- Moved reusable setup adapters under the canonical `vercor.setups` package and
  runnable setup scripts under `examples`, removing in-repo reliance on a
  top-level `setups` package.
- Split public component factory helpers into `vercor.components.factories` and
  kept `vercor.components.base` focused on base authoring contracts.
- Routed setup adapter validation and runtime-boundary imports through private
  validation internals or public component context aliases instead of runtime
  stores/contexts.
- Added focused ownership modules for calendar logic, rectilinear grid geometry,
  generic sigma-coordinate helpers, generic PyTree transforms, setup data asset
  registries, and diagnostics fields/tables/plotting.
- Moved mask math into `vercor.grid_masks` and component topology lookup into
  `vercor.runtime.topology`.
- Split CAMulator optional imports, forcing cursors, tensor accessors, stepping,
  and initialization into focused modules while keeping `camulator_state.py` as
  a thin compatibility facade.
- Focused checks passed for API boundaries, component factories, setup imports,
  runtime-boundary imports, shared helper ownership, assets/diagnostics
  separation, and CAMulator decomposition.
- Required validation passed:
  `conda run -n scipy black vercor examples tests`,
  `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`,
  `conda run -n scipy mypy vercor examples tests`,
  `conda run -n scipy pytest tests/ -q --fast --tb=short`, and
  `conda run -n scipy pytest tests/ -q --tb=short`. The existing Black
  Python 3.13/target-3.14 warning and JAX dtype-promotion warning remain.

### 2026-05-15: Conservative Helper and Compatibility Cleanup

- Removed unused private helpers and wrappers that had become one-hop
  compatibility layers: bilinear `_geo_to_cart(...)`, component author
  `_author_field_spec(...)`, contract refresh indirection, runtime wrapper
  helpers, and Coupler topology delegates.
- Collapsed the setup forcing reader to the canonical
  `vercor.forcing_data.read_forcing` boundary while preserving the setup import
  path.
- Inlined the single-use private NetCDF output helper into the public output
  writer.
- Removed obsolete TODO/commented print blocks from conservative rectilinear
  regridder edge derivation.
- Preserved intentional compatibility surfaces, including settings attribute
  compatibility, `ComponentSettings`, component author facades/context aliases,
  `vercor.runtime.components` reexports, setup lazy exports, setup forcing
  reexports, `ComponentForcingData._read_forcing()`,
  `Coupler._run_scanned_runtime()`, `_runtime_state_from_components()`,
  regridder class/factory APIs, and `Exchange.create()`.

### 2026-05-15: Runtime Helper Consolidation

- Added private `Coupler._prepare_runtime_state(...)` so `run()` and
  `_run_scanned_runtime()` share runtime-state creation/reuse and optional
  validation transition.
- Split runtime exchange dispatch into scalar and vector primitives while
  preserving existing behavior: scalar exchanges apply fractional masks, vector
  exchanges do not.
- Added `CamulatorRuntimeCursor` so CAMulator atmosphere and land adapters share
  forcing-index initialization, counter reset, index lookup, and counter
  advancement.
- Moved generic component step-result application into
  `vercor.components._runtime_fields.apply_step_result(...)` so callable
  wrappers and base components use the same primitive.
- Removed the test-only JCM coordinate wrapper and pointed tests at the real
  `_jcm_coordinates_in_degrees(...)` helper.

### 2026-05-15: Maintainability Follow-Up

- Added `setups.jcm_setup_helpers.build_jcm_land_atmosphere_components(...)`
  for repeated JCM setup construction, land-mask patching, and JAXGCM option
  forwarding.
- Refactored CAMulator `StateVariableAccessor` index-map construction through
  shared private primitives.
- Routed remaining multi-exchange runnable setup scripts through
  `setups.coupler_helpers.add_exchanges(...)`.
- Corrected `align_model_timestep(...)` non-divisible error text so it states
  that the model timestep must evenly divide the coupling timestep.
- Updated `DEPENDENCIES.md` for the JCM setup helper.
- Deferred intentionally high-risk audit findings: JAXGCM mirrored
  runtime/setup state, host/scanned runner unification, callable-wrapper
  architecture changes, and component inheritance changes.

## Milestone Timeline

### 2026-05-14: Setup and Adapter Maintainability

- Removed JAXGCM test-only compatibility attachments from factory-created
  components and moved private setup internals into explicit test fixtures.
- Consolidated NetCDF forcing reads behind `vercor.forcing_data.read_forcing`.
- Added explicit `Component.setup_metadata` for setup-only metadata.
- Extended setup lifecycle helpers for timestep assignment, spinup logging,
  forcing-index calculation, and default-field seeding.
- Added runnable setup helpers while keeping exchange recipes explicit.
- Consolidated common setup adapter paths for JAXGCM, Veros, CAMulator, ERA5,
  ERA-Interim, and JCM where behavior is shared.
- Added lazy optional setup imports so missing optional packages fail only when
  those adapters are used.
- Converted concrete setup components toward factory-based construction and
  reduced duplicate host-array and masked-field helpers.

### 2026-05-13: Logging

- Standardized VerCOR logging format and replaced root-logger capture
  expectations with the canonical logging boundary.

### 2026-05-12: Precision, Performance, and API Simplification

- Audited and propagated the centralized dtype policy.
- Optimized runtime profiling/core dispatch paths.
- Corrected hypsometric altitude calculations.
- Forwarded configured regridder factory options consistently.
- Removed redundant component APIs such as `required_fields` and callable field
  seeding.
- Added the shared PyTree mixin used by immutable JAX containers.

### 2026-05-08: Runtime Ownership and Component Boilerplate

- Moved runtime responsibilities into focused runtime modules.
- Extracted Coupler runtime adapter logic and component runtime-field adapters.
- Fixed time-dependent data field runtime validation.
- Tightened component constructor/runtime boilerplate and removed redundant
  authoring delegates.

### 2026-05-07: Component Authoring API

- Split component internals from the public authoring facade.
- Added and refined helper-first component authoring APIs.
- Polished component field declarations, context aliases, callable wrappers,
  default field seeding, and component author introspection.

### 2026-05-06: Settings and Lifecycle Logging

- Added Coupler lifecycle logging.
- Reworked settings into the unified metadata-backed `VercorSettings`
  container with dynamic attribute access.

### 2026-05-05: Runtime Interrupt Handling

- Added compiled runtime wakeup-fd interrupt handling.
- Unified host and scanned runtime interrupt translation.
- Added scanned runtime progress logging.
- Stabilized JAXGCM forcing payload scan shapes.

### 2026-05-04 to 2026-05-01: Data Layout and Data Components

- Centralized VerCOR dtype policy.
- Canonicalized component data dimension order.
- Added ERA5 atmosphere pure data component support.
- Made component author contracts explicit.

### 2026-04-30 to 2026-04-28: Runtime Package and Boundary Refactors

- Added callback-backed JAX runtime logging.
- Split the runtime package into explicit state, contract, context, driver,
  validation, and transfer boundaries.
- Clarified public/runtime API responsibilities.
- Removed residual compatibility markers and simplified runtime bridge
  ownership.
- Added compile-cache and safe buffer donation runtime support.
- Simplified runtime API validation around component-owned grid shapes.

### 2026-04-27 to 2026-04-23: JAX Translation and Unified Runtime Foundation

- Expanded Coupler, Veros, clock, and flux-kernel coverage.
- Translated flux, grid, bilinear, conservative remapping, slab, Veros, JAXGCM,
  CAMulator, ERA5, ERA-Interim, and data-forcing boundaries toward JAX-first
  runtime paths.
- Added differentiable public runtime and hardened mixed-grid/data-forcing
  runtime execution.
- Unified runtime component paths and removed legacy differentiable/wrapper-era
  APIs.
- Fixed wrapper runtime startup prefill and audited runtime tests against the
  canonical API.

## Known Failed Approaches / Corrections

- Do not fix numerical discrepancies with fudge factors. Earlier successful
  fixes traced missing terms, sign/index errors, dtype/layout issues, or
  boundary ownership mistakes instead.
- CAMulator cursor advancement must occur once after each non-empty forcing
  block, not once per model substep.
- Masked surface fields should use `jnp.where(mask, field, jnp.nan)` rather
  than multiply-by-NaN masking, which produced NaN gradients on valid cells.
- Runtime field stores copy leaves on insertion so donated runtime states do
  not reuse duplicate JAX buffers and trip XLA donation errors.
- Source-boundary assertions should be precise. Several earlier failures came
  from over-broad substring checks that matched intentional helper names.
- Tests that patch host-backed external components should call
  `step_runtime_state()` with explicit `RuntimeComponentState` objects when
  they are exercising runtime behavior.
- Setup-only metadata belongs in `Component.setup_metadata`, not ad-hoc
  attributes that enter public component/runtime contracts.
- Detailed failed attempts and command outputs are in
  `docs/progress-archive-2026-04-23-to-2026-05-15.md`.

## Validation Policy

- Keep this file compact. Record outcomes, current state, durable lessons, and
  next actions.
- Do not paste repeated per-task validation boilerplate into this file.
- For a normal development unit, record only:
  - the focused tests that mattered,
  - whether fast/full validation passed,
  - any new warnings, regressions, or failed approaches worth preserving.
- If a detailed transcript is needed, create or update a dated archive under
  `docs/` and link it from this file.
- Default development validation remains:

  ```bash
  conda run -n scipy pytest tests/ -q --fast --tb=short
  ```

- Before a commit or handoff, run the relevant focused tests plus the project
  static checks and full test suite when the change affects code behavior.
