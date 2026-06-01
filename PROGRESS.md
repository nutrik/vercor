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
- Latest local component-author API split validation: Black, flake8, mypy,
  focused fast pytest, full fast pytest, and full pytest passed as of
  2026-05-27.
- Latest local core-boundary mixin extraction validation: Black, flake8, mypy,
  focused fast pytest, full fast pytest, and full pytest passed as of
  2026-05-27.
- Latest local external-adapter runtime-boundary validation: Black, flake8,
  mypy, focused fast pytest, full fast pytest, and full pytest passed as of
  2026-05-27.
- Latest local callable-component boundary validation: Black, flake8, mypy,
  focused fast pytest, full fast pytest, and full pytest passed as of
  2026-05-27.
- Latest local external-adapter state-boundary validation: Black, flake8, mypy,
  focused fast pytest, full fast pytest, and full pytest passed as of
  2026-05-27.
- Latest local runtime-facade/CAMulator-index validation: Black, flake8, mypy,
  focused fast pytest, full fast pytest, and full pytest passed as of
  2026-05-27.
- Latest local runtime-resource-holder validation: Black, flake8, mypy,
  focused fast pytest, full fast pytest, and full pytest passed as of
  2026-05-28.
- Latest local obsolete compatibility API cleanup validation: Black, flake8,
  mypy, focused compatibility pytest, full fast pytest, and full pytest passed
  as of 2026-05-28.
- Latest local obsolete compatibility API active-doc audit validation:
  API-boundary fast pytest, full fast pytest, Black, flake8, mypy, and full
  pytest passed as of 2026-05-28.
- Latest local component protocol/resource boundary validation: focused fast
  pytest, Black, flake8, mypy, full fast pytest, and full pytest passed as of
  2026-05-28.
- Latest local whole-codebase boundary refactor validation: focused fast pytest,
  full fast pytest, Black, flake8, mypy, and full pytest passed as of
  2026-05-28.
- Latest local runtime-resource boundary refinement validation: focused fast
  pytest, Black, flake8, mypy, full fast pytest, and full pytest passed as of
  2026-05-28.
- Latest local runtime-preparation boundary validation: focused boundary
  pytest, Black, flake8, mypy, full fast pytest, and full pytest passed as of
  2026-05-28.
- Latest local component/runtime boundary alias validation: focused fast
  pytest, Black, flake8, mypy, full fast pytest, and full pytest passed as of
  2026-05-28.
- Latest local runtime-output boundary validation: focused red/green pytest,
  Black, flake8, mypy, full fast pytest, and full pytest passed as of
  2026-06-01.
- Latest local runtime-state validation boundary validation: focused
  red/green pytest, Black, flake8, mypy, full fast pytest, and full pytest
  passed as of 2026-06-01.
- Latest local runtime resource/topology boundary validation: focused
  red/green pytest, Black, flake8, mypy, full fast pytest, and full pytest
  passed as of 2026-06-01.
- Latest local compiled-runtime cache boundary validation: focused red/green
  pytest, Black, flake8, mypy, full fast pytest, and full pytest passed as of
  2026-06-01.
- Latest local runtime compilation cache boundary validation: focused
  red/green pytest, Black, focused fast pytest, flake8, mypy, full fast pytest,
  and full pytest passed as of 2026-06-01.
- Latest local runtime topology policy boundary validation: focused red/green
  pytest, Black, focused boundary pytest, flake8, mypy, full fast pytest, and
  full pytest passed as of 2026-06-01.
- Latest local component-context boundary validation: focused red/green pytest,
  Black, focused fast pytest, flake8, mypy, full fast pytest, and full pytest
  passed as of 2026-06-01.
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

### 2026-06-01: Component Context Boundary Refactor

- Added `vercor.components.contexts` as the canonical owner for
  `ComponentSetupContext` and `ComponentStepContext`.
- Removed the internal `vercor.runtime.contexts` module and replaced production
  and test imports of `ComponentInitContext` / `RuntimeStepContext` with the
  public component-author context names.
- Updated component contracts and package facades so context dataclasses are
  reexported from `vercor.components` and `vercor` through the component-owned
  boundary, while hook type aliases remain in `vercor.components.contracts`.
- Strengthened architecture coverage so context dataclasses cannot drift back
  into runtime ownership and setup adapters remain free of runtime context/store
  imports.
- Updated `DESIGN.md` and `DEPENDENCIES.md` for component-context ownership.
- Validation run for this change:
  baseline `conda run -n scipy pytest tests/ -q --fast --tb=short`, focused red
  pytest for the missing `vercor.components.contexts` owner, focused green
  boundary pytest after implementation,
  `conda run -n scipy black vercor examples tests`,
  `conda run -n scipy pytest tests/test_api_boundaries.py tests/test_runtime_state.py tests/test_component_base_coverage.py -q --fast --tb=short`,
  `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`,
  `conda run -n scipy mypy vercor examples tests`,
  `conda run -n scipy pytest tests/ -q --fast --tb=short`, and
  `conda run -n scipy pytest tests/ -q --tb=short`. The existing Black Python
  3.13/target-3.14 warning and JAX dtype-promotion warning remain.
- Failed approach recorded: the first focused green run exposed a runtime
  import of `ComponentStepContext` in `vercor.components.base`, which widened
  the base module surface. The import is now `TYPE_CHECKING`-only so the base
  module stays narrow while annotations remain type-checkable.

### 2026-06-01: Runtime Topology Policy Boundary Refactor

- Added `vercor.runtime.topology_state` as the neutral owner for
  `RuntimeRegridder`, grouped `RuntimeTopologyMaps`, `SurfaceExchangeMasks`,
  and `ExchangeTopologyState`.
- Split generic exchange regridder/identity-mask map construction into
  `vercor.runtime.exchange_topology`, and moved ATM/OCN/LND surface-mask
  creation, validation, and bilinear mask patching into
  `vercor.runtime.surface_masks`.
- Reduced `vercor.runtime.topology` to orchestration: it composes generic
  exchange topology maps with surface masks and returns one explicit topology
  state. Runtime resources and initialization now import topology state
  contracts from the neutral state module, and `Coupler.initialize()` reads
  public mask attributes through `topology.surface_masks`.
- Strengthened boundary coverage so topology state, generic exchange maps, and
  surface-mask policy cannot drift back into one mixed-responsibility topology
  module.
- Updated `DESIGN.md` and `DEPENDENCIES.md` for the topology-state,
  exchange-topology, surface-mask, and topology-orchestration split.
- Validation run for this change:
  baseline `conda run -n scipy pytest tests/ -q --fast --tb=short`, focused
  red pytest for the missing `topology_state`/`surface_masks` split, focused
  green topology/boundary pytest after implementation,
  `conda run -n scipy black vercor examples tests`,
  `conda run -n scipy pytest tests/test_runtime_facade_boundaries.py tests/test_runtime_state.py tests/test_api_boundaries.py tests/test_coupler_coverage.py -q --fast --tb=short`,
  `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`,
  `conda run -n scipy mypy vercor examples tests`,
  `conda run -n scipy pytest tests/ -q --fast --tb=short`, and
  `conda run -n scipy pytest tests/ -q --tb=short`. The existing Black Python
  3.13/target-3.14 warning and JAX dtype-promotion warning remain.
- Failed approaches recorded: the first focused green run exposed a stale
  boundary assertion that still expected `vercor.runtime.topology` to import
  component-topology lookup helpers; it now checks the new surface-mask owner.
  The first flake8 pass reported one stale unused test local left by the import
  move; it was removed before rerunning flake8.

### 2026-06-01: Runtime Compilation Cache Boundary Refactor

- Added `vercor.runtime.compilation` as the neutral owner for
  `CompiledRuntime` and `RuntimeCompilationKey`.
- Moved context-derived compiled-runtime cache-key construction onto frozen
  `RuntimeRunContext`, leaving `CompiledRuntimeCache` focused on compiled
  callable storage, JIT wrapping, clearing, count, and value inspection.
- Updated the scanned runner to pass
  `context.compiled_runtime_cache_key(...)` into `get_or_compile(...)`, and
  updated runtime resources to type compiled cache values through the neutral
  compilation module instead of importing the alias from `run_context`.
- Strengthened boundary coverage so `vercor.runtime.cache` cannot drift back to
  importing or mentioning `RuntimeRunContext`, context-aware cache helpers, or
  context-derived key construction.
- Updated `DESIGN.md` and `DEPENDENCIES.md` for the compilation alias,
  run-context key, and cache storage/JIT ownership split.
- Validation run for this change:
  baseline `conda run -n scipy pytest tests/ -q --fast --tb=short`, focused
  red pytest for the missing neutral compilation boundary and old runner cache
  helper call, focused green pytest after implementation,
  `conda run -n scipy black vercor examples tests`,
  `conda run -n scipy pytest tests/test_runtime_facade_boundaries.py tests/test_runtime_state.py tests/test_api_boundaries.py tests/test_runtime_run_cache.py tests/test_runtime_interrupts.py -q --fast --tb=short`,
  `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`,
  `conda run -n scipy mypy vercor examples tests`,
  `conda run -n scipy pytest tests/ -q --fast --tb=short`, and
  `conda run -n scipy pytest tests/ -q --tb=short`. The existing Black Python
  3.13/target-3.14 warning and JAX dtype-promotion warning remain.
- Failed approach recorded: the first focused green run exposed a stale runner
  boundary assertion that forbade `compiled_runtime_cache_key(` anywhere in
  `runner.py`; it now forbids runner-owned key definitions while requiring the
  intended `context.compiled_runtime_cache_key(...)` call in the compiled
  scanned helper.

### 2026-06-01: Compiled Runtime Cache Boundary Refactor

- Added `vercor.runtime.cache.CompiledRuntimeCache` as the owner for compiled
  scanned-runtime cache storage, JIT wrapping, context-derived cache-key lookup,
  clearing, count, and value inspection.
- Changed `RuntimeRunContext` to carry the cache owner instead of a mutable
  mapping, and changed `CouplerRuntimeResources` to store a private
  `_runtime_cache` while delegating its public cache facade methods to that
  owner.
- Removed the raw `runtime_cache_mapping()` accessor and updated the scanned
  runner to ask the cache owner for compiled runtime reuse rather than importing
  cache-key and cache-mutation helpers directly.
- Strengthened boundary coverage so resources no longer expose the raw cache
  mapping, run-context annotations cannot drift back to `MutableMapping`, and
  cache compile/reuse/inspection behavior remains owned by
  `CompiledRuntimeCache`.
- Updated `DESIGN.md` and `DEPENDENCIES.md` for the cache-owner boundary.
- Validation run for this change:
  baseline `conda run -n scipy pytest tests/ -q --fast --tb=short`,
  focused red pytest for the missing cache owner and raw mapping leak, focused
  green pytest after implementation,
  `conda run -n scipy black vercor examples tests`,
  `conda run -n scipy pytest tests/test_runtime_state.py tests/test_runtime_facade_boundaries.py tests/test_runtime_run_cache.py tests/test_runtime_interrupts.py -q --fast --tb=short`,
  `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`,
  `conda run -n scipy mypy vercor examples tests`,
  `conda run -n scipy pytest tests/ -q --fast --tb=short`, and
  `conda run -n scipy pytest tests/ -q --tb=short`. The existing Black Python
  3.13/target-3.14 warning and JAX dtype-promotion warning remain.
- Failed approach recorded: the first runner boundary assertion searched for
  `compiled_scanned_runtime(` across the entire runner module and also matched
  the intended private `_run_compiled_scanned_runtime(...)` helper; it now checks
  only the removed cache-helper import/call shape.

### 2026-06-01: Runtime Resource and Topology Boundary Refactor

- Added `vercor.runtime.component_topology` as the owner for default
  topology component-name validation and component lookup, leaving
  `vercor.runtime.topology` focused on exchange regridder/mask setup.
- Added grouped `RuntimeTopologyMaps` and changed `ExchangeTopologyState` to
  carry topology maps as one boundary object instead of exposing three parallel
  map fields.
- Refactored `CouplerRuntimeResources` into a slotted private-field holder with
  explicit topology, contract, runtime-cache, and interrupt accessors; runtime
  facade/preparation code no longer reaches through to raw resource
  dictionaries.
- Made `RuntimeRunContext` frozen to document that run-context identity is a
  static execution input bundle.
- Updated focused boundary coverage plus runtime/topology/output tests so the
  new ownership cannot drift back into `vercor.runtime.topology` or raw
  resource attributes.
- Updated `DESIGN.md` and `DEPENDENCIES.md` for component-topology ownership,
  grouped topology maps, private runtime resources, and the frozen run context.
- Validation run for this change:
  focused red pytest for the missing ownership/resource boundaries,
  focused green pytest after implementation,
  `conda run -n scipy black vercor examples tests`,
  `conda run -n scipy pytest tests/test_runtime_facade_boundaries.py tests/test_runtime_state.py tests/test_coupler_coverage.py -q --fast --tb=short`,
  `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`,
  `conda run -n scipy mypy vercor examples tests`,
  `conda run -n scipy pytest tests/ -q --fast --tb=short`, and
  `conda run -n scipy pytest tests/ -q --tb=short`. The existing Black Python
  3.13/target-3.14 warning and JAX dtype-promotion warning remain.
- Failed approaches recorded: the first new boundary test imported
  `RuntimeTopologyMaps` at module import time and failed during collection
  instead of as an assertion; it now imports dynamically inside the test. The
  first full fast suite exposed one stale API-boundary assertion that still
  expected topology-name validation in `vercor.runtime.topology`; it now checks
  `vercor.runtime.component_topology`.

### 2026-06-01: Runtime State Validation Boundary Refactor

- Moved configured runtime-state/topology validation from
  `vercor.runtime.coupler_state` into `vercor.runtime.state_validation`, leaving
  coupler-state ownership focused on immutable runtime state assembly and
  runtime-contract refresh.
- Updated `vercor.runtime.preparation` to call the new validation owner while
  preserving its preparation-facing validation wrapper and public runtime
  behavior.
- Strengthened boundary coverage so validation ownership cannot drift back into
  `vercor.runtime.coupler_state` or the public `Coupler`/runtime-facade
  boundary.
- Updated `DESIGN.md` and `DEPENDENCIES.md` for the new validation ownership.
- Validation run for this change:
  focused red pytest for the missing validation owner,
  focused green pytest after the move,
  `conda run -n scipy black vercor examples tests`,
  `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`,
  `conda run -n scipy mypy vercor examples tests`,
  `conda run -n scipy pytest tests/ -q --fast --tb=short`, and
  `conda run -n scipy pytest tests/ -q --tb=short`. The existing Black Python
  3.13/target-3.14 warning and JAX dtype-promotion warning remain.
- No failed implementation approaches.

### 2026-06-01: Runtime Output Boundary Refactor

- Moved output mask selection and naming from `vercor.runtime.coupler_state`
  into `vercor.output`, leaving runtime coupler-state ownership focused on
  immutable state assembly, contract refresh, and validation.
- Removed `vercor.output`'s dependency on `vercor.runtime.coupler_state`; final
  output iteration now owns its view construction, file naming, and mask lookup
  in one output boundary.
- Strengthened boundary coverage so `output_masks_for_component(...)` stays in
  `vercor.output` and cannot drift back into runtime-state assembly.
- Updated `DESIGN.md` and `DEPENDENCIES.md` for the new output-mask ownership.
- Validation run for this change:
  focused red pytest for the ownership move,
  focused green pytest after the move,
  `conda run -n scipy black vercor examples tests`,
  `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`,
  `conda run -n scipy mypy vercor examples tests`,
  `conda run -n scipy pytest tests/ -q --fast --tb=short`, and
  `conda run -n scipy pytest tests/ -q --tb=short`. The existing Black Python
  3.13/target-3.14 warning and JAX dtype-promotion warning remain.
- No failed implementation approaches.

### 2026-05-28: Component and Runtime Boundary Alias Refactor

- Moved public lifecycle hook type aliases to `vercor.components.contracts` and
  reexported them from `vercor.components` and `vercor`, leaving
  `vercor.components._lifecycle` focused on private hook storage and
  installation.
- Added shared callable component construction metadata in
  `vercor.components._callable_wrappers` so differentiable and host
  `from_model()` paths share field-spec, payload, settings, and lifecycle-hook
  normalization.
- Split `vercor.runtime.runner.run_coupler_runtime()` into smaller helpers for
  compiled scanned execution and host-runtime donation rejection while
  preserving public runtime behavior.
- Strengthened boundary and lifecycle coverage for public hook ownership,
  callable construction ownership, direct `from_model()` lifecycle hooks, and
  runner path-selection helpers.
- Updated `DESIGN.md` and `DEPENDENCIES.md` for the new component and runtime
  boundary ownership.
- Validation run for this change:
  focused component/runtime-cache fast pytest,
  `conda run -n scipy black vercor examples tests`,
  `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`,
  `conda run -n scipy mypy vercor examples tests`,
  `conda run -n scipy pytest tests/ -q --fast --tb=short`, and
  `conda run -n scipy pytest tests/ -q --tb=short`. The existing Black Python
  3.13/target-3.14 warning and JAX dtype-promotion warning remain.
- Failed approach recorded: the first runner boundary test split source through
  the next public function and accidentally included the new private helper
  body; it now extracts only `run_coupler_runtime()`.

### 2026-05-28: Runtime Preparation Boundary Refactor

- Added `vercor.runtime.preparation` as the focused owner for prepared runtime
  state construction, contract refresh for prepared states, runtime-state
  validation, and initial outgoing-store priming.
- Kept `vercor.runtime.facade` as the coupler-facing orchestration boundary by
  reexporting preparation helpers while leaving dispatch/run context
  construction, execution delegation, runtime views, and final output delegation
  in the facade.
- Centralized `CompiledRuntime` in `vercor.runtime.run_context` and exchange
  field/factory aliases in `vercor.exchange`, removing duplicate alias
  ownership from runtime resources and setup helper modules.
- Strengthened boundary tests for runtime preparation ownership, facade
  reexports, runtime import-cycle absence, shared compiled-runtime typing, and
  exchange alias ownership.
- Updated `DESIGN.md` and `DEPENDENCIES.md` for the new preparation and alias
  ownership boundaries.
- Validation run for this change:
  focused runtime/API/state boundary fast pytest,
  `conda run -n scipy black vercor examples tests`,
  `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`,
  `conda run -n scipy mypy vercor examples tests`,
  `conda run -n scipy pytest tests/ -q --fast --tb=short`, and
  `conda run -n scipy pytest tests/ -q --tb=short`. The existing Black Python
  3.13/target-3.14 warning and JAX dtype-promotion warning remain.
- Failed approaches recorded: the first `RuntimePreparationInputs` protocol used
  mutable attributes, which mypy rejected for frozen `RuntimeFacadeInputs`; the
  protocol now uses read-only properties. The first full pytest run exposed a
  stale architecture assertion that still expected preparation logic in
  `runtime.facade`; the assertion now checks `runtime.preparation` as the owner.

### 2026-05-28: Runtime Resource Boundary Refinement

- Added public `Coupler.clear_runtime_cache()` and
  `Coupler.runtime_cache_entry_count()` as the small profiling-facing runtime
  cache API.
- Added grouped runtime-resource helpers for topology-map replacement and
  compiled-cache clear/count/value inspection, keeping cache dictionaries and
  synthetic topology setup behind the runtime resource holder.
- Updated the runtime profiling example and focused runtime tests to use the
  public cache facade or named test helpers instead of raw runtime cache and
  topology assignments.
- Updated `DESIGN.md` and `DEPENDENCIES.md` for the refined resource boundary.
- Validation run for this change:
  focused runtime-resource/API fast pytest,
  `conda run -n scipy black vercor examples tests`,
  `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`,
  `conda run -n scipy mypy vercor examples tests`,
  `conda run -n scipy pytest tests/ -q --fast --tb=short`, and
  `conda run -n scipy pytest tests/ -q --tb=short`. The existing Black Python
  3.13/target-3.14 warning and JAX dtype-promotion warning remain.
- No failed implementation approaches.

### 2026-05-28: Whole-Codebase Boundary Refactor

- Added shared canonical grid-field shape helpers in `vercor.field_layout`, then
  routed component-facing and runtime-facing required-field validation through
  the shared message/shape policy while preserving existing exception types.
- Narrowed private component helper protocols so runtime helpers no longer
  require setup data storage, and added focused boundary tests for protocol
  ownership.
- Added `vercor.runtime.facade.RuntimeFacadeInputs` so `Coupler` passes one
  grouped internal runtime input bundle into facade helpers instead of repeated
  component/exchange/resource parameter clumps.
- Added lightweight CAMulator runtime field contract ownership in
  `vercor.setups.external.camulator_contracts`, leaving tensor/field mapping
  code focused on runtime arrays.
- Split reusable architecture-test helpers out of `tests/test_api_boundaries.py`
  and added focused tests for field layout, component boundaries, runtime facade
  boundaries, and CAMulator contracts.
- Validation run for this change:
  focused new boundary pytest,
  focused API/runtime/component/CAMulator fast pytest,
  `conda run -n scipy pytest tests/ -q --fast --tb=short`,
  `conda run -n scipy black vercor examples tests`,
  `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`,
  `conda run -n scipy mypy vercor examples tests`, and
  `conda run -n scipy pytest tests/ -q --tb=short`. The existing Black Python
  3.13/target-3.14 warning and JAX dtype-promotion warning remain.
- Failed approach recorded: the initial shared field-layout error wording broke
  an existing component data-field regex contract; the helper now preserves the
  component-facing message shape while still centralizing shape validation.

### 2026-05-28: Component Protocol and Runtime Resource Boundary Refactor

- Added private component helper protocols in `vercor.components._protocols` so
  runtime-field, validation, lifecycle, and callable-wrapper helpers depend on
  structural component contracts instead of type-only imports from the public
  `Component` base class.
- Added grouped `CouplerRuntimeResources.replace_contracts(...)` and
  `replace_topology(...)` methods, then routed runtime-facade contract/topology
  refreshes through the resource holder instead of assigning individual maps in
  facade code.
- Strengthened boundary tests for component helper protocol ownership, runtime
  resource replacement, and runtime-facade assignment cleanup; updated
  `DESIGN.md` and `DEPENDENCIES.md` for the new ownership map.
- Validation run for this change:
  `conda run -n scipy pytest tests/test_api_boundaries.py tests/test_runtime_state.py tests/test_component_base_coverage.py -q --fast --tb=short`,
  `conda run -n scipy black vercor examples tests`,
  `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`,
  `conda run -n scipy mypy vercor examples tests`,
  `conda run -n scipy pytest tests/ -q --fast --tb=short`, and
  `conda run -n scipy pytest tests/ -q --tb=short`. The existing Black Python
  3.13/target-3.14 warning and JAX dtype-promotion warning remain.
- Failed approach recorded: the first full pytest run exposed an overly exact
  existing source-boundary assertion for the `RuntimeRegridder` topology import;
  the assertion now checks topology-owner imports without depending on one-line
  import formatting.

### 2026-05-28: Obsolete Compatibility Active-Doc Audit

- Audited live source, tests, examples, `README.md`, `DESIGN.md`,
  `DEPENDENCIES.md`, and active `PROGRESS.md` for obsolete compatibility import
  paths and shim modules while leaving the historical archive untouched.
- Confirmed the removed facade modules and shim paths remain absent from the
  live tree; remaining runtime-payload references use the canonical
  `vercor.setups.external.jax_gcm_runtime` owner or boundary tests that assert
  removed reexports stay removed.
- Tightened API-boundary coverage so active `PROGRESS.md` no longer advertises
  removed compatibility surfaces as current preserved API, then refreshed stale
  progress entries to point at later canonical ownership.
- Validation run for this change:
  `conda run -n scipy pytest tests/test_api_boundaries.py -q --fast --tb=short`,
  `conda run -n scipy pytest tests/ -q --fast --tb=short`,
  `conda run -n scipy black vercor examples tests`,
  `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`,
  `conda run -n scipy mypy vercor examples tests`, and
  `conda run -n scipy pytest tests/ -q --tb=short`. The existing Black Python
  3.13/target-3.14 warning and JAX dtype-promotion warning remain.
- No failed implementation approaches. The audit found no remaining live
  obsolete shim modules requiring production-code deletion.

### 2026-05-28: Obsolete Compatibility API Cleanup

- Removed compatibility-only runtime reexports from `vercor.runtime`; code,
  examples, and tests now import runtime contracts, state containers, stores,
  step metadata, and exchange dispatch from their focused owner modules.
- Removed obsolete compatibility aliases and methods:
  `vercor.setups.external.jax_gcm.JAXGCMRuntimePayload`,
  the external setup lazy payload export, `ComponentSettings`,
  `ComponentForcingData._read_forcing()`, CAMulator dictionary metadata
  accessors, and private `Coupler` runtime resource/scanned-run shims.
- Added test-only runtime helpers for focused scanned-runtime and
  state-from-components coverage without restoring production compatibility
  methods.
- Updated boundary tests, runtime/coupler/cache/interrupt/forcing/CAMulator
  tests, `DESIGN.md`, and `DEPENDENCIES.md` for the canonical API paths.
- Validation run for this change:
  focused obsolete-API/CAMulator/forcing pytest,
  `conda run -n scipy black vercor examples tests`,
  `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`,
  `conda run -n scipy mypy vercor examples tests`,
  `conda run -n scipy pytest tests/ -q --fast --tb=short`, and
  `conda run -n scipy pytest tests/ -q --tb=short`. The existing Black
  Python 3.13/target-3.14 warning and JAX dtype-promotion warning remain.

### 2026-05-28: Runtime Resource Holder Boundary Refactor

- Added `vercor.runtime.resources.CouplerRuntimeResources` as the owner for
  per-coupler runtime topology maps, refreshed runtime contracts, compiled
  runtime cache, and interrupt controller.
- Updated `Coupler` to store one runtime resource holder while keeping
  then-current private runtime test/profiling aliases; a later cleanup removed
  those aliases.
- Routed runtime facade initialization, state creation, validation, dispatch/run
  context construction, execution, and finalization through the resource holder
  instead of repeated map/cache/interrupt arguments.
- Updated boundary tests, `DESIGN.md`, and `DEPENDENCIES.md` for the new runtime
  resource ownership map.
- Validation run for this change:
  focused runtime/Coupler/cache/interrupt fast pytest,
  `conda run -n scipy black vercor examples tests`,
  `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`,
  `conda run -n scipy mypy vercor examples tests`,
  `conda run -n scipy pytest tests/ -q --fast --tb=short`, and
  `conda run -n scipy pytest tests/ -q --tb=short`. The existing Black
  Python 3.13/target-3.14 warning and JAX dtype-promotion warning remain.

### 2026-05-27: Runtime Facade and CAMulator Tensor Index Refactor

- Added `vercor.runtime.facade` as the high-level orchestration boundary used by
  `Coupler` for runtime-state creation, validation, dispatch/run context
  construction, execution, runtime views, and final output delegation.
- Slimmed `vercor.coupler` so it delegates runtime internals through the facade
  while retaining then-current test delegates; later cleanup removed the
  private compatibility methods.
- Added typed `TensorVariableIndex` metadata for CAMulator tensor access; the
  temporary dictionary metadata accessor was later removed in favor of
  `StateVariableAccessor.get_var_index(...)`.
- Updated boundary/CAMulator tests, `DESIGN.md`, and `DEPENDENCIES.md`.
- Validation run for this change:
  focused runtime/API/Coupler/CAMulator fast pytest,
  `conda run -n scipy black vercor examples tests`,
  `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`,
  `conda run -n scipy mypy vercor examples tests`,
  `conda run -n scipy pytest tests/ -q --fast --tb=short`, and
  `conda run -n scipy pytest tests/ -q --tb=short`. The existing Black
  Python 3.13/target-3.14 warning and JAX dtype-promotion warning remain.

### 2026-05-27: External Adapter State Boundary Refactor

- Added private runtime-state protocols for JAXGCM, Veros, and CAMulator
  runtime helpers so adapter runtime modules no longer accept unbounded setup
  state objects in their public helper signatures.
- Replaced JAXGCM factory lambda lifecycle wiring with named callbacks bound by
  `functools.partial`, and replaced the Veros host step lambda with a named
  private step adapter.
- Kept Veros optional runtime settings lazy by importing `runtime_settings`
  inside `configure_veros_runtime()`, and made CAMulator wind-filter
  configuration fail with explicit `ValueError`s while removing mutable function
  defaults.
- Updated boundary/focused tests, `DESIGN.md`, and `DEPENDENCIES.md` for the
  tightened external-adapter state ownership map.
- Validation run for this change:
  focused external/API/CAMulator fast pytest,
  `conda run -n scipy black vercor examples tests`,
  `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`,
  `conda run -n scipy mypy vercor examples tests`,
  `conda run -n scipy pytest tests/ -q --fast --tb=short`, and
  `conda run -n scipy pytest tests/ -q --tb=short`. The existing Black
  Python 3.13/target-3.14 warning and JAX dtype-promotion warning remain.

### 2026-05-27: Callable Component Boundary Refactor

- Moved concrete callable-backed component classes next to their owning public
  runtime-kind bases: `vercor.components.base` now owns the differentiable
  callable wrapper and `vercor.components.host` now owns the host-runtime
  callable wrapper.
- Slimmed `vercor.components._callable_wrappers` to shared callable signature
  normalization and runtime step-result application, and kept
  `vercor.components.factories` as public helper delegates instead of an
  internal callable construction owner.
- Strengthened boundary tests to reject the old factory import path, callable
  wrapper imports of concrete component bases, and the removed
  `_create_callable_component` / `CallableComponentRequest` construction path.
- Updated `DESIGN.md` and `DEPENDENCIES.md` for the callable wrapper ownership
  map.
- Validation run for this change:
  focused API/callable fast pytest,
  `conda run -n scipy black vercor examples tests`,
  `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`,
  `conda run -n scipy mypy vercor examples tests`,
  `conda run -n scipy pytest tests/ -q --fast --tb=short`, and
  `conda run -n scipy pytest tests/ -q --tb=short`. The existing Black
  Python 3.13/target-3.14 warning and JAX dtype-promotion warning remain.

### 2026-05-27: External Adapter Runtime Boundary Refactor

- Added focused runtime owners for external adapter behavior:
  `jax_gcm_runtime.py` owns JAXGCM runtime payload, defaults, prefill,
  validation, stepping, and host recording; `camulator_runtime.py` owns
  CAMulator datetime coercion, prediction-block execution, and runtime step
  mapping; `veros_runtime.py` owns Veros flux application, host substeps, and
  SST refresh.
- Slimmed `jax_gcm.py`, `camulator.py`, and `veros_gcm.py` back toward
  optional-dependency loading, model construction, setup initialization, spinup,
  and factory wiring while preserving existing public factories. The JAXGCM
  runtime payload is now owned by `jax_gcm_runtime.py` rather than reexported by
  `jax_gcm.py`.
- Updated boundary/focused tests, `DESIGN.md`, and `DEPENDENCIES.md` for the
  new external-adapter ownership map.
- Validation run for this change:
  focused external/runtime fast pytest,
  `conda run -n scipy black vercor examples tests`,
  `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`,
  `conda run -n scipy mypy vercor examples tests`,
  `conda run -n scipy pytest tests/ -q --fast --tb=short`, and
  `conda run -n scipy pytest tests/ -q --tb=short`. The existing Black
  Python 3.13/target-3.14 warning and JAX dtype-promotion warning remain.

### 2026-05-27: Core Boundary Mixin Extraction

- Split `Component` helper behavior into focused private modules:
  `_field_names`, `_field_authoring`, `_runtime_access`, and `_lifecycle_api`,
  leaving `vercor.components.base` focused on the abstract component contract
  and callable factory entrypoint.
- Made `Coupler.run_sequence` an explicit empty `RunSequence` by default and
  removed dynamic `hasattr`/`getattr` branches from runtime preparation.
- Routed the slab example's ICE diagnostic read through `RuntimeComponentView`
  instead of direct runtime-state store access.
- Updated boundary tests, `DESIGN.md`, and `DEPENDENCIES.md` for the new helper
  ownership map.
- Validation run for this change:
  focused boundary/component/coupler fast pytest,
  `conda run -n scipy pytest tests/ -q --fast --tb=short`,
  `conda run -n scipy black vercor examples tests`,
  `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`,
  `conda run -n scipy mypy vercor examples tests`, and
  `conda run -n scipy pytest tests/ -q --tb=short`. The existing Black
  Python 3.13/target-3.14 warning and JAX dtype-promotion warning remain.

### 2026-05-27: Component Author API Split

- Added canonical public component-author modules:
  `vercor.components.contracts` for field contracts/context aliases,
  `vercor.components.data` for `DataComponent`, and
  `vercor.components.host` for `HostRuntimeComponent`.
- Slimmed `vercor.components.base` to the abstract `Component` contract and
  moved concrete component-kind imports in factories, callable wrappers,
  runtime execution policy, tests, and package facades to the new module owners.
- Updated boundary tests, `DESIGN.md`, and `DEPENDENCIES.md` for the new
  component-author ownership map.
- Validation run for this change:
  focused component/API fast pytest,
  `conda run -n scipy pytest tests/ -q --fast --tb=short`,
  `conda run -n scipy black vercor examples tests`,
  `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`,
  `conda run -n scipy mypy vercor examples tests`, and
  `conda run -n scipy pytest tests/ -q --tb=short`. The existing Black
  Python 3.13/target-3.14 warning and JAX dtype-promotion warning remain.

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
  preserving existing private runtime-state helpers and topology map aliases.
  Later cleanup removed those private compatibility attributes.
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
  `vercor.grid_masks`, and `vercor.fluxes.utilities` while keeping stable
  package aggregators and settings attribute access. The remaining
  settings/forcing aliases were removed by the later obsolete compatibility API
  cleanup.
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

- Moved model-calendar datetime values into `vercor.calendar`; later cleanup
  removed the `vercor.clock` reexports.
- Split canonical exchange-field vocabulary into `vercor.field_names`, unified
  grid identity in `vercor.grid_geometry`, and removed the runtime daily-index
  wrapper in favor of `vercor.calendar.daily_forcing_index`.
- Consolidated hybrid/sigma pressure and altitude helpers in
  `vercor.fluxes.vertical_coordinates`; later cleanup removed the old flux
  utility import aliases.
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
  and initialization into focused modules before later cleanup removed the
  `camulator_state.py` facade.
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
- Deferred removal of then-intentional compatibility surfaces such as settings
  aliases, component author facades/context aliases, runtime reexports, setup
  lazy exports, setup forcing reexports, private runtime delegates, regridder
  class/factory APIs, and `Exchange.create()`. Later refactors removed the
  obsolete aliases and shims while preserving documented public facades.

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
