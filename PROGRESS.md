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
- Latest local component execution protocol boundary validation: focused
  red/green pytest, Black, focused fast pytest, flake8, mypy, full fast pytest,
  and full pytest passed as of 2026-06-01.
- Latest local component lifecycle boundary validation: focused red/green
  pytest, Black, focused fast pytest, flake8, mypy, full fast pytest, and full
  pytest passed as of 2026-06-01.
- Latest local external-adapter helper boundary validation: focused red/green
  pytest, Black, focused fast pytest, flake8, mypy, full fast pytest, and full
  pytest passed as of 2026-06-01.
- Latest local external-adapter setup-state boundary validation: focused
  red/green pytest, Black, focused fast pytest, flake8, mypy, full fast pytest,
  and full pytest passed as of 2026-06-01.
- Latest local logging facade/private-owner boundary validation: baseline fast
  pytest, focused red/green pytest, Black, flake8, mypy, focused logging pytest,
  full fast pytest, and full pytest passed as of 2026-06-02.
- Latest local bilinear interpolator boundary validation: baseline fast pytest,
  focused red/green pytest, Black, flake8, mypy, full fast pytest, and full
  pytest passed as of 2026-06-02.
- Latest local calendar forcing-index boundary validation: baseline fast pytest,
  focused red/green pytest, focused runtime pytest, Black, flake8, mypy, full
  fast pytest, and full pytest passed as of 2026-06-02.
- Latest local asset/forcing-data boundary validation: baseline fast pytest,
  focused red/green pytest, Black, focused post-format pytest, flake8, mypy,
  full fast pytest, and full pytest passed as of 2026-06-02.
- Latest local external adapter factory/setup-state boundary validation:
  focused red/green pytest, Black, flake8, mypy, full fast pytest, and full
  pytest passed as of 2026-06-02.
- Latest local CAMulator wind-filter boundary validation: focused red/green
  pytest, Black, focused post-format pytest, flake8, mypy, full fast pytest,
  and full pytest passed as of 2026-06-02.
- Latest local unit-test speedup validation: focused red/green pytest, Black,
  focused post-format pytest, flake8, mypy, full fast pytest with durations,
  full pytest with durations, and coverage pytest passed as of 2026-06-02.
- Latest local JAXGCM h5netcdf average-output validation: focused red/green
  pytest, Black, focused writer/API pytest, flake8, mypy, full fast pytest,
  full pytest, and coverage pytest passed as of 2026-06-03.
- Latest local Veros h5netcdf period-output validation: baseline fast pytest,
  focused red/green pytest, Black, focused post-format pytest, flake8, mypy,
  full fast pytest, full pytest, and coverage pytest passed as of 2026-06-03.
- Latest local streaming period-average output validation: focused red/green
  pytest, Black, focused boundary pytest, flake8, mypy, full fast pytest, full
  pytest, and coverage pytest passed as of 2026-06-04.
- Latest local Veros average dimension-order validation: focused red/green
  pytest, Black, flake8, mypy, full fast pytest, full pytest, and coverage
  pytest passed as of 2026-06-04.
- Latest local Veros spinup period-average validation: focused red/green
  pytest, Black, flake8, mypy, full fast pytest, full pytest, and coverage
  pytest passed as of 2026-06-08.
- Latest local JAX-backed output-array boundary validation: baseline fast
  pytest, focused red/green pytest, focused mypy, Black, flake8, full mypy,
  focused post-format pytest, full fast pytest, full pytest, and coverage
  pytest passed as of 2026-06-09.
- Latest local internal naming consistency validation: focused pytest, Black,
  flake8, mypy, full fast pytest, full pytest, and coverage pytest passed as of
  2026-06-10.
- Latest local trivial internal wrapper cleanup validation: focused cleanup
  scan, focused pytest, Black, flake8, mypy, full fast pytest, and full pytest
  passed as of 2026-06-10.
- Latest local unified GCM output package validation: focused red/green pytest,
  Black, flake8, mypy, full fast pytest, full pytest, and coverage pytest
  passed as of 2026-06-10.
- Latest local shared h5netcdf output helper validation: focused red/green
  pytest, Black, git diff whitespace check, flake8, mypy, full fast pytest,
  full pytest, and coverage pytest passed as of 2026-06-10.
- Latest local shared JCM/Veros output dataset helper validation: focused
  red/green pytest, Black, git diff whitespace check, flake8, mypy, full fast
  pytest, full pytest, and coverage pytest passed as of 2026-06-11.
- Latest local shared period-average output writer validation: focused
  red/green pytest, focused output/API pytest, Black, git diff whitespace
  check, flake8, mypy, full fast pytest, full pytest, and coverage pytest
  passed as of 2026-06-11.
- Latest local over-engineering quick-win cleanup validation: focused red/green
  pytest, focused affected pytest, Black, git diff whitespace check, flake8,
  mypy, full fast pytest, full pytest, and coverage pytest passed as of
  2026-06-11.
- Latest local over-engineering helper-layer cleanup validation: focused
  red/green pytest, focused affected pytest, Black, git diff whitespace check,
  flake8, mypy, full fast pytest, and full pytest passed as of 2026-06-11.
- Latest local runtime/component over-engineering sweep validation: focused
  red/green pytest, Black, git diff whitespace check, flake8, mypy, full fast
  pytest, full pytest, and coverage pytest passed as of 2026-06-11.
- Latest local over-engineering audit quick-win cleanup validation: focused
  red/green pytest, focused external/diagnostics pytest, Black, git diff
  whitespace check, flake8, mypy, full fast pytest, and full pytest passed as
  of 2026-06-11.
- Latest local no-break over-engineering cleanup campaign validation: focused
  red/green pytest, Black, git diff whitespace check, flake8, mypy, full fast
  pytest, full pytest, and coverage pytest passed as of 2026-06-12.
- Latest local external output ownership validation: focused red/green pytest,
  focused output pytest, Black, git diff whitespace check, flake8, mypy, full
  fast pytest, and full pytest passed as of 2026-06-12.
- Latest local CAMulator direct h5netcdf output validation: focused red/green
  pytest, focused output pytest, Black, git diff whitespace check, flake8,
  mypy, full fast pytest, full pytest, and coverage pytest passed as of
  2026-06-16.
- Latest local CAMulator period-average output-frequency validation: focused
  red/green pytest, focused CAMulator/shared-output pytest, Black, flake8,
  mypy, full fast pytest, full pytest, coverage pytest, and git diff
  whitespace check passed as of 2026-06-17.
- Latest local centralized NetCDF filename-logging validation: focused
  red/green pytest, focused output pytest, Black, flake8, mypy, full fast
  pytest, full pytest, and `conda run -n scipy` fast pytest passed as of
  2026-06-19.
- Latest local internal output/runtime helper simplification validation:
  baseline fast pytest, focused red/green pytest, focused runtime/API pytest,
  Black, flake8, mypy, full fast pytest, full pytest, coverage pytest, and git
  diff whitespace check passed as of 2026-06-19.
- Latest local component output adapter refactor validation: baseline fast
  pytest via direct `scipy` env Python, focused adapter/external/API pytest,
  Black, flake8, mypy, full fast pytest, full pytest, and git diff whitespace
  check passed as of 2026-06-29.
- Latest local unused helper API cleanup validation: focused red/green cleanup
  pytest, focused affected pytest, Black, flake8, mypy, full fast pytest,
  full pytest, coverage pytest, and git diff whitespace check passed as of
  2026-06-29.
- Latest local remaining helper-surface over-engineering cleanup validation:
  focused red/green pytest, Black, flake8, mypy, focused affected pytest, full
  fast pytest, full pytest, and coverage pytest passed as of 2026-06-29.
- Latest local centralized output adapter record-logic validation: focused
  red/green pytest, Black, flake8, mypy, focused post-format pytest, full fast
  pytest, full pytest, and git diff whitespace check passed as of 2026-06-30
  using the direct `scipy` environment executable.
- Latest local simplification-plan quick-win validation: baseline fast pytest,
  focused red/green pytest, focused affected pytest, Black, flake8, mypy, full
  fast pytest, full pytest, and git diff whitespace check passed as of
  2026-06-30 using the direct `scipy` environment executable.
- Latest local conservative scalar-only regridder validation: baseline fast
  pytest, focused red/green pytest, focused affected pytest, Black, flake8,
  mypy, full fast pytest, full pytest, and git diff whitespace check passed as
  of 2026-06-30 using the direct `scipy` environment executable.
- Latest local internal helper type-surface simplification validation: focused
  red/green pytest, Black, flake8, mypy, full fast pytest, full pytest, and
  git diff whitespace check passed as of 2026-06-30 using the direct `scipy`
  environment executable.
- Latest local external setup-step/remapper derived-state simplification
  validation: baseline fast pytest, focused red/green pytest, focused affected
  pytest, Black, flake8, mypy, full fast pytest, full pytest, and git diff
  whitespace check passed as of 2026-06-30 using `conda run -n scipy`.
- Latest local RunSequence deprecation and tuple run-order validation: focused
  red/green API tests, focused affected fast runtime/external/helper/component
  tests, Black, flake8, mypy, full fast pytest, full pytest, coverage pytest,
  and git diff whitespace check passed as of 2026-06-30 using the `scipy`
  environment through `conda run`.
- Latest local legacy component seed helper removal validation: baseline fast
  pytest, focused red/green API/component tests, Black, flake8, mypy, full fast
  pytest, full pytest, git diff whitespace check, and `conda run -n scipy`
  fast pytest passed as of 2026-06-30. The earlier planning-time Conda/Rattler
  panic was not reproduced during final validation.
- Latest local Exchange create-wrapper removal validation: baseline fast
  pytest, focused red/green Exchange tests, focused affected pytest, Black,
  flake8, mypy, full fast pytest, full pytest, and `conda run -n scipy`
  smoke pytest passed as of 2026-06-30. The earlier planning-time
  Conda/Rattler panic was not reproduced during final validation.
- Latest local concrete regridder call-ownership validation: baseline fast
  pytest, focused red/green boundary tests, focused affected pytest, Black,
  flake8, mypy, full fast pytest, full pytest, and git diff whitespace check
  passed as of 2026-06-30 using `conda run -n scipy`.
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

## Follow-Up Candidates

- Keep remaining public simplification candidates review-only unless a
  compatibility decision is made: `Grid`, component authoring mixins, calendar
  compatibility delegates, and setup helper APIs are still public or
  boundary-tested surfaces. `RunSequence` now has compatibility-safe plain
  sequence support, but the wrapper remains public API.

## Recent Work

### 2026-06-30: Concrete Regridder Call Ownership

- Moved regridder call dispatch out of the shared `Regridder` base and into
  the concrete bilinear and conservative classes. The base now owns only shared
  grid/interpolator/display state.
- Preserved bilinear scalar/vector behavior, conservative scalar-only errors,
  and identical-grid passthrough while removing the conservative `_ensure_ready`
  override pattern.
- Added boundary coverage for the new ownership split and updated dependency
  wording plus stale test comments.
- Validation run for this change: baseline fast pytest passed; the focused
  red boundary test first failed on the existing base `__call__`; after
  implementation, focused affected pytest, Black, flake8, mypy, full fast
  pytest, full pytest, and `git diff --check` passed using
  `conda run -n scipy`. Black emitted the existing Python 3.13/target-3.14
  warning, and full pytest emitted the existing JAX dtype-promotion warning.

### 2026-06-30: Exchange Create Wrapper Removal

- Removed the public-looking one-line `Exchange.create()` wrapper so exchange
  declarations remain static configuration and runtime topology construction
  calls `exchange.regridder_factory(...)` directly.
- Updated helper and coupler coverage tests to assert the removed wrapper stays
  absent, preserve factory-name formatting behavior, and patch
  `regridder_factory` for topology recording tests while keeping existing
  interpolation keys.
- Validation run for this change: baseline fast pytest passed; focused red
  tests first failed on `Exchange.create` still existing, then passed after
  removal. The first full pytest run exposed one stale test monkeypatching the
  removed wrapper; after updating that test double, focused affected pytest,
  Black, flake8, mypy, full fast pytest, full pytest, and
  `conda run -n scipy python -m pytest tests/test_helpers_coverage.py -q --fast --tb=short`
  passed. Black emitted the existing Python 3.13/target-3.14 safety-check
  warning, and full pytest emitted the existing JAX dtype-promotion warning.

### 2026-06-30: Legacy Component Seed Helper Removal

- Removed the public `Component.seed_zero_field()`, `seed_zero_fields()`, and
  `seed_constant_field()` helpers so component authors use the canonical
  scalar-expanding `seed_field()` and `seed_fields()` path.
- Updated API-boundary and component coverage tests to assert the collapsed
  authoring surface and to validate zero/constant seeding through
  `seed_field(s)`. Updated `DESIGN.md` component-authoring guidance to match.
- Validation run for this change: baseline direct fast pytest passed; focused
  red pytest failed on the existing `seed_zero_field` API, then focused green
  pytest passed after implementation. Black, flake8, mypy, direct full fast
  pytest, direct full pytest, `git diff --check`, and
  `conda run -n scipy pytest tests/ -q --fast --tb=short` passed. Black emitted
  the existing Python 3.13/target-3.14 safety-check warning, and full pytest
  emitted the existing JAX dtype-promotion warning.

### 2026-06-30: External Setup-Step and Remapper Derived-State Simplification

- Removed one-line Veros and CAMulator GCM setup-state `step()` delegates.
  Their factories now pass `partial(...step_*_runtime, state)` directly to
  the host component boundary, matching the existing JAXGCM factory pattern.
- Removed conservative remapper cached derived fields
  `_normalize_fracarea` and `_n_dst_cells`; `apply_scalar()` now derives those
  values locally from declared metadata, so PyTree unflattening no longer
  needs a class-specific post-unflatten hook.
- Updated boundary/PyTree tests to guard the simplified behavior. The first
  full pytest run exposed a stale runtime-state source assertion that assumed
  Veros and CAMulator GCM setup-state files must still define `def step(`;
  that assertion now only inspects the remaining CAMulator land inline step
  and explicitly verifies the removed setup-state delegates stay absent.
- Validation run for this change: baseline fast pytest passed; focused red
  tests first failed on the existing setup-state delegates and remapper cached
  derived attributes. After implementation, focused affected pytest, Black,
  flake8, mypy, full fast pytest, full pytest, and `git diff --check` passed
  using `conda run -n scipy`. Black emitted the existing Python
  3.13/target-3.14 safety-check warning, and full pytest emitted the existing
  JAX dtype-promotion warning.

### 2026-06-30: Internal Helper Type-Surface Simplification

- Removed the unused `FieldDefaults` type alias from component contracts and
  the private component-contract reexport layer. Boundary tests now assert that
  the alias stays absent from both surfaces.
- Removed the `RuntimeRegridder` concrete union from
  `vercor.runtime.topology_state`; grouped runtime topology maps now avoid
  importing bilinear/conservative regridder implementations and type the
  regridder map as an internal object container.
- Updated `DEPENDENCIES.md` so runtime topology state no longer lists direct
  bilinear/conservative regridder dependencies.
- Validation run for this change: focused red tests first failed on the
  existing `FieldDefaults` export and `RuntimeRegridder` topology alias. After
  implementation, focused boundary pytest, Black, flake8, mypy, full fast
  pytest, full pytest, and `git diff --check` passed using direct `scipy`
  environment executables. Black emitted the existing Python 3.13/target-3.14
  safety-check warning, and full pytest emitted the existing JAX
  dtype-promotion warning.

### 2026-06-30: Conservative Scalar-Only Regridder Cleanup

- Removed the unsupported `ConservativeRectilinearRemapper.apply_vector()`
  stub so conservative remapping exposes only the scalar operation it
  implements.
- Added a conservative regridder argument guard that rejects vector calls before
  the shared identical-grid fast path, so both identical and non-identical
  conservative vector calls fail with the same scalar-only `TypeError`.
- Validation run for this change: baseline fast pytest passed; focused red
  tests first failed on the existing remapper vector stub, old remapper-origin
  `RuntimeError`, and identical-grid vector passthrough. After implementation,
  focused red/green pytest, affected conservative pytest, Black, flake8, mypy,
  full fast pytest, full pytest, and `git diff --check` passed using direct
  `scipy` environment executables. Black emitted the existing Python
  3.13/target-3.14 safety-check warning, and full pytest emitted the existing
  JAX dtype-promotion warning.

### 2026-06-30: Simplification Plan Quick Wins

- Relaxed private architecture-locking tests toward behavior and public-boundary
  assertions where cleanup was implemented.
- Removed unused public NumPy dtype helper functions; tests now derive NumPy
  dtypes with `np.dtype(jax_real_dtype(...))` and
  `np.dtype(jax_index_dtype(...))`.
- Added compatibility-safe plain component-name sequence support for `Coupler`,
  setup helpers, and runtime preparation/facade paths while preserving
  `RunSequence` normalization internally. Updated examples and `DESIGN.md`
  accordingly.
- Simplified regridder internals by removing the unused interpolation protocol
  and subclass post-initialization mutation; concrete regridders now pass their
  interpolator and identical-grid flag into the shared base.
- Removed the callable component request dataclass/init hop; callable component
  constructors now build field specs and lifecycle hooks directly.
- Validation run for this change: the focused red tests first failed on the old
  dtype helpers, mandatory `RunSequence` storage, regridder protocol, and
  callable request dataclass. After implementation, focused affected pytest,
  `/Users/romannuterman/miniforge3/envs/scipy/bin/black vercor examples tests`,
  `/Users/romannuterman/miniforge3/envs/scipy/bin/flake8 . --count --exit-zero --max-line-length=120 --statistics`,
  `/Users/romannuterman/miniforge3/envs/scipy/bin/mypy vercor examples tests`,
  `/Users/romannuterman/miniforge3/envs/scipy/bin/python -m pytest tests/ -q --fast --tb=short`,
  `/Users/romannuterman/miniforge3/envs/scipy/bin/python -m pytest tests/ -q`,
  and `git diff --check` passed. Black emitted the existing Python
  3.13/target-3.14 safety-check warning, and full pytest emitted the existing
  JAX dtype-promotion warning. The earlier baseline `conda run -n scipy ...`
  path still hit the local Conda/Rattler panic before pytest, so validation used
  the direct `scipy` environment executable.

### 2026-06-30: Centralized Output Adapter Record Logic

- Added `ComponentOutputAdapter.record_period_average_if_due()` as the shared
  output path for "accumulate sample, check cadence, write if due" behavior.
- Added JAXGCM, Veros, and CAMulator package-internal output adapter factories
  plus record helpers so model-specific extraction, coordinates, metadata, and
  CAMulator forecast-increment output remain beside each external adapter while
  period-average orchestration goes through the shared adapter boundary.
- Rewired JAXGCM, Veros, and CAMulator setup states and runtime output paths to
  use those helpers, and updated API-boundary and focused output tests to guard
  against local write-closure duplication returning to runtime modules.
- Validation run for this change: focused red tests first failed on the missing
  generic adapter method and missing model-specific factory/record helpers.
  After implementation,
  `/Users/romannuterman/miniforge3/envs/scipy/bin/python -m pytest tests/test_output_adapters.py tests/test_external_components_coverage.py tests/test_camulator_component_kernels.py tests/test_api_boundaries.py -q --tb=short`,
  `/Users/romannuterman/miniforge3/envs/scipy/bin/black vercor examples tests`,
  `/Users/romannuterman/miniforge3/envs/scipy/bin/flake8 . --count --exit-zero --max-line-length=120 --statistics`,
  `/Users/romannuterman/miniforge3/envs/scipy/bin/mypy vercor examples tests`,
  `/Users/romannuterman/miniforge3/envs/scipy/bin/python -m pytest tests/ -q --fast --tb=short`,
  `/Users/romannuterman/miniforge3/envs/scipy/bin/python -m pytest tests/ -q --tb=short`,
  and `git diff --check` passed. Black emitted the existing Python
  3.13/target-3.14 safety-check warning, and the full suite emitted the
  existing JAX dtype-promotion warning. Earlier in this session, both
  `conda run -n scipy ...` and `conda --no-plugins run -n scipy ...` hit the
  existing Conda/Rattler `PanicException` before pytest, so validation used the
  direct `scipy` environment executable.

### 2026-06-29: Remaining Helper-Surface Over-Engineering Cleanup

- Added `docs/over-engineering-audit-2026-06-29.md` with the requested
  executive summary, findings table, and conservative refactor plan.
- Removed the hidden `Clock._iter_impl` dispatch attribute; `Clock.iter()` now
  branches directly between Gregorian and model-calendar iterators while
  preserving existing calendar behavior.
- Removed unused CAMulator wind-filter convenience exports
  `wind_filter()` and `simple_wind_artifact_filter()` while keeping the
  config-driven runtime post-processing facade and private tensor mechanics.
- Removed the unused `jax_real_array_copy()` dtype helper and kept the active
  dtype policy helpers used by production code.
- Validation run for this change: focused red tests first failed on the old
  clock dispatch attribute, CAMulator wind-filter wrappers, and dtype copy
  helper, then the same focused suite passed after implementation. Black,
  flake8, mypy, focused affected pytest, full fast pytest, full pytest, and
  coverage pytest at 90% total coverage passed with `conda run -n scipy`. The
  earlier orientation smoke check still recorded the existing Conda Rattler
  plugin crash before pytest on one `conda run` invocation; the direct scipy
  environment fallback
  `/Users/romannuterman/miniforge3/envs/scipy/bin/python -m pytest tests/ -q --fast --tb=short`
  passed. The existing Black Python 3.13/target-3.14 warning and JAX
  dtype-promotion warning remain.

### 2026-06-29: Unused Helper API Cleanup

- Removed unused CAMulator stepper convenience methods and accessor attributes
  so the runtime path owns model input assembly, postprocessing, and state
  shifting directly.
- Removed the one-line period-average accumulation wrapper; the shared output
  adapter now calls its owned accumulator directly.
- Removed the unused generic PyTree concatenation helper and its concat-only
  test coverage while keeping the active PyTree helpers.
- Validation run for this change: focused red cleanup tests failed before the
  production edits for the expected remaining helper surfaces. Focused affected
  pytest, Black, flake8, mypy, full fast pytest, full pytest, coverage pytest
  at 90% total coverage, and `git diff --check` passed using the direct
  `scipy` env Python path. The existing Black Python 3.13/target-3.14 warning
  and JAX dtype-promotion warning remain.

### 2026-06-29: Component Output Adapter Refactor

- Added `vercor.output.ComponentOutputAdapter` as the shared owner for
  external component period-average accumulation, mean conversion, cadence, and
  NetCDF write lifecycle.
- Replaced model-specific period-output wrapper routines in JAXGCM, Veros, and
  CAMulator with small extraction, coordinate, path, and metadata helpers that
  runtimes compose through the adapter. CAMulator immediate forecast-increment
  output remains model-specific.
- Updated architecture/dependency docs and boundary tests so shared
  period-output helper calls live in `vercor.output.adapters`, not in each
  external component output module.
- Validation run for this change: `conda run -n scipy pytest tests/ -v
  --fast` failed before pytest due to the existing Conda rattler plugin
  `PanicException`; `/Users/romannuterman/miniforge3/envs/scipy/bin/python -m
  pytest tests/ -q --fast --tb=short` passed. Focused
  adapter/external/API pytest, Black, flake8, mypy, full fast pytest, full
  pytest, and `git diff --check` passed. The existing Black Python
  3.13/target-3.14 warning and JAX dtype-promotion warning remain.

### 2026-06-19: Internal Output/Runtime Helper Simplification

- Removed the redundant `PeriodAverageSample` alias plus
  `samples_from_output_variables()` / `mean_samples_or_raise()` from
  period-average output. Accumulators now accept `OutputVariable` samples
  directly and `period_mean_output_variables()` owns adapter-specific empty
  accumulator errors.
- Removed the unused `runtime.facade.run_scanned()` shortcut and the thin
  `refresh_runtime_contracts()` wrapper. Focused test helpers now call the
  scanned runtime owner directly, and runtime preparation calls
  `build_runtime_contracts()` at the point of use.
- Made `RuntimeTopologyMaps` a mutable slotted dataclass, matching its actual
  setup-time mutation model while leaving the surrounding topology-state
  containers frozen.
- Validation run for this change: baseline fast pytest, focused red/green
  pytest, focused runtime/API pytest, Black, flake8, mypy, full fast pytest,
  full pytest, coverage pytest, and git diff whitespace check passed. The
  existing Black Python 3.13/target-3.14 warning and JAX dtype-promotion warning
  remain.

### 2026-06-19: Centralized NetCDF Filename Logging

- Moved NetCDF filename log emission into the shared `write_netcdf_dataset`
  boundary and routed period-average and CAMulator forecast-increment writers
  through that single logging path.
- Added regression coverage for shared-writer logger injection, exact-once
  period-file logging, CAMulator forecast/average filename logging, and scalar
  data-variable writes. The scalar test covers the h5py rule that scalar
  datasets cannot use gzip filter options.
- Validation run for this change: focused red/green pytest, focused output
  pytest, Black, flake8, mypy, full fast pytest, full pytest, and
  `conda run -n scipy pytest tests/ -q --fast --tb=short` passed. The existing
  Black Python 3.13/target-3.14 warning and JAX dtype-promotion warning remain.

### 2026-06-17: CAMulator Period-Average Output Frequency

- Added `output_frequency` to the CAMulator GCM public factory and setup state.
  `None` preserves per-forecast-increment output; configured `day`, `month`, or
  `year` streams CAMulator prediction tensors into the shared period-average
  accumulator and writes average files under the configured forecast output
  folder.
- Added CAMulator output helpers for period accumulation and average-file
  writing through shared `vercor.output` primitives while keeping CAMulator
  tensor metadata, `predict.save_vars` filtering, and forecast-increment output
  in `camulator_output.py`.
- Updated the CAMulator/Veros example and architecture/dependency docs for the
  unified external output interface.
- Validation run for this change: focused red/green pytest, focused
  CAMulator/shared-output pytest, Black, flake8, mypy, full fast pytest, full
  pytest, coverage pytest, and `git diff --check` passed. The existing Black
  Python 3.13/target-3.14 warning and JAX dtype-promotion warning remain.

### 2026-06-16: CAMulator Direct h5netcdf Output

- Replaced CAMulator forecast-increment output delegation to CREDIT
  `make_xarray`/NetCDF helpers with VerCOR-owned tensor shaping and direct
  `h5netcdf` writing through the shared output variable boundary.
- Moved CAMulator output metadata loading into `camulator_output.py`, kept
  model/parser/transform CREDIT imports in `camulator_imports.py`, and wired
  runtime output to use the setup state's existing tensor transformer for
  `predict.climate_rescale_output`.
- Added unsupported-option validation for xarray-only CREDIT output features
  such as pressure interpolation, ptype, and CREDIT-specific encoding dicts.
- Red/green notes: new output tests first failed because direct helper APIs and
  the `state_transformer` writer argument were missing; after implementation,
  the fast suite exposed the new NumPy import as an explicit host-output
  boundary, so `tests/test_production_numpy_boundaries.py` now lists
  `camulator_output.py`.
- Validation run for this change: focused CAMulator/output pytest, Black,
  `git diff --check`, flake8, mypy, full fast pytest, full pytest, and coverage
  pytest passed. The existing Black Python 3.13/target-3.14 warning and JAX
  dtype-promotion warning remain.

### 2026-06-12: External Output Adapter Ownership Boundary

- Moved JAXGCM and Veros-specific period-output adapters from `vercor.output`
  to `vercor.setups.external`, leaving `vercor.output` as the shared
  setup-agnostic output primitive package.
- Updated JAXGCM/Veros runtime and setup-state imports, boundary tests,
  functional output tests, `DESIGN.md`, and `DEPENDENCIES.md` for the clean
  break from `vercor.output.jax_gcm` and `vercor.output.veros`.
- Red/green notes: focused external/API tests first failed with missing
  `vercor.setups.external.jax_gcm_output` before the move, then passed after
  moving modules and rewiring imports. A pre-existing period-file log-message
  test drift on this branch was aligned with the current
  `Writing output file:  ...` message.
- Validation run for this change: `conda run -n scipy pytest tests/test_api_boundaries.py tests/test_external_components_coverage.py -q --tb=short`,
  `conda run -n scipy pytest tests/test_output_datasets.py tests/test_output_netcdf.py tests/test_period_averages.py tests/test_period_files.py -q --tb=short`,
  `conda run -n scipy black vercor examples tests`, `git diff --check`,
  `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`,
  `conda run -n scipy mypy vercor examples tests`,
  `conda run -n scipy pytest tests/ -q --fast --tb=short`, and
  `conda run -n scipy pytest tests/ -q --tb=short` passed. The existing Black
  Python 3.13/target-3.14 warning and JAX dtype-promotion warning remain.

### 2026-06-12: No-Break Over-Engineering Cleanup Campaign

- Added `docs/over-engineering-audit-2026-06-12.md` with the requested
  executive summary, findings table, and recommended refactor plan.
- Removed three no-break internal simplification targets from the audit:
  `RuntimeTopologyMaps.from_mappings()` is gone and copy semantics now live at
  the exchange-topology boundary, `vercor.output` directly reexports its three
  runtime-output helpers, and one-use period-file builder aliases are inlined
  into `write_period_average_netcdf()`.
- Left public or compatibility-bound simplification candidates as follow-up
  only: `Grid`, `RunSequence`, component authoring/lifecycle layers, calendar
  compatibility delegates, and optional-dependency setup facades.
- Red/green notes: focused runtime tests first failed on the old
  `from_mappings()` helper, and focused API/period-file tests first failed on
  the old period-file alias layer. The same focused suites passed after the
  cleanup.
- Validation run for this change: baseline fast pytest passed before edits.
  After implementation, `conda run -n scipy pytest tests/test_runtime_state.py tests/test_runtime_facade_boundaries.py -q --tb=short`,
  `conda run -n scipy pytest tests/test_api_boundaries.py tests/test_period_files.py -q --tb=short`,
  `conda run -n scipy black vercor examples tests`, `git diff --check`,
  `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`,
  `conda run -n scipy mypy vercor examples tests`,
  `conda run -n scipy pytest tests/ -q --fast --tb=short`,
  `conda run -n scipy pytest tests/ -q --tb=short`, and
  `conda run -n scipy pytest --cov=vercor tests/ -q` passed. Coverage remained
  at 90% total. The existing Black Python 3.13/target-3.14 warning and JAX
  dtype-promotion warning remain.

### 2026-06-11: Over-Engineering Audit Quick-Win Cleanup

- Completed a whole-codebase over-engineering sweep focused on unnecessary
  internal wrappers, single-use helpers, speculative extension points, and
  public surfaces whose complexity is either justified or compatibility-bound.
- Removed three low-risk internal helper layers:
  private diagnostics `view_field*` delegates now call the runtime view lookup
  owner directly, Veros setup binds `_veros_state.pure` directly instead of a
  one-line `advance_veros_model_step()` wrapper, and `VercorSettings` copies
  immutable default records with `dict(DEFAULT_SETTINGS)` instead of rebuilding
  each `Settings` tuple.
- Added boundary tests preventing those helper shapes from returning while
  preserving existing behavior coverage for settings isolation, diagnostics,
  and external adapter boundaries.
- Audit findings deferred as not quick wins: `RuntimeTopologyMaps.from_mappings`
  is internal but has explicit boundary tests and can wait; component
  authoring/lifecycle mixins and callable wrappers protect documented public
  extension APIs; calendar forcing-index delegates remain active compatibility
  imports.
- Red/green notes: focused boundary/settings tests first failed on the old
  diagnostics wrappers, Veros wrapper, and settings copy helper, then passed
  after the cleanup. Focused external component/diagnostics tests also passed.
- Validation run for this change:
  `conda run -n scipy pytest tests/test_api_boundaries.py tests/test_settings.py -q --tb=short`
  failed first on the expected old helper shapes, then passed after the
  cleanup. `conda run -n scipy pytest tests/test_external_components_coverage.py tests/test_tools_components_and_plotting.py -q --tb=short`,
  `conda run -n scipy black vercor examples tests`, `git diff --check`,
  `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`,
  `conda run -n scipy mypy vercor examples tests`,
  `conda run -n scipy pytest tests/ -q --fast --tb=short`, and
  `conda run -n scipy pytest tests/ -q --tb=short` passed. The existing Black
  Python 3.13/target 3.14 warning and JAX dtype-promotion warning remain.

### 2026-06-11: Runtime and Component Over-Engineering Sweep

- Inlined compiled-runtime aliases into `vercor.runtime.cache` and removed the
  alias-only `vercor.runtime.compilation` module.
- Simplified `CouplerRuntimeResources` to public dataclass fields and moved
  cache clearing/counting call sites to the cache owner.
- Removed the private runtime-preparation input protocol, the time-selection
  lookup protocols, and adapter-local runtime-state protocols in favor of the
  existing facade/setup-state types.
- Moved component runtime-field convenience methods onto `Component`, deleted
  `_runtime_access.py`, and trimmed `_protocols.py` to the runtime-checkable
  host execution protocol.
- Reused `OutputVariable` for period-average samples through the
  `PeriodAverageSample` compatibility alias.
- Red/green notes: focused boundary tests first failed on the old alias module,
  private resource fields/wrappers, annotation-only protocols, runtime-access
  mixin, and duplicate sample dataclass; after implementation the same focused
  suite passed.
- Validation run for this change: focused red/green pytest, Black, `git diff
  --check`, flake8, mypy, full fast pytest, full pytest, and coverage pytest
  passed. Coverage reported 90% total. The existing Black Python 3.13/target
  3.14 warning and JAX dtype-promotion warning remain.

### 2026-06-11: Over-Engineering Helper-Layer Cleanup

- Removed the private `_IdentityInterpolator` helper and let the base regridder
  handle identical-grid scalar/vector passthrough directly before requiring a
  concrete interpolator.
- Narrowed `RuntimeTopologyMaps.from_mappings()` to the only used behavior:
  return an empty bundle or copy an existing topology-map bundle. Removed the
  unused keyword-construction branches.
- Removed the one-line `CouplerRuntimeResources.replace_topology(...)` wrapper;
  the runtime facade now assigns the grouped topology maps directly on runtime
  resources.
- Removed the unused `vercor.output.veros.VerosOutputVariable` alias so Veros
  output uses the shared `OutputVariable` container directly.
- Red/green notes: the focused cleanup tests first failed on the old identity
  helper, resource topology wrapper, broad topology-map constructor, and Veros
  alias; after implementation the same focused suite passed.
- Deferred broader simplifications for public/boundary-tested surfaces:
  `RunSequence`, `Grid`, component authoring mixins, calendar compatibility
  delegates, and setup helper APIs.
- Validation run for this change: focused affected pytest, Black, `git diff
  --check`, flake8, mypy, full fast pytest, and full pytest passed. The existing
  Black Python 3.13/target-3.14 warning and JAX dtype-promotion warning remain.

### 2026-06-11: Over-Engineering Quick-Win Cleanup

- Removed unused PyTree payload from the bilinear interpolator and conservative
  remapper by dropping cached source meshgrids, unused `fracarea_norm`, and the
  unused source-cell count while preserving interpolation/remapping behavior.
- Removed the dead core runtime data-field validator now superseded by the
  component-owned canonical runtime-field validation path.
- Simplified JAXGCM factory wiring so setup state binds runtime-owned lifecycle
  hooks directly, and removed the one-line callback delegate layer from
  `jax_gcm_state`.
- Removed the unused CAMulator `accessor_state` setup attribute while keeping
  the runtime-used input/output accessors.
- Updated `DESIGN.md`, `DEPENDENCIES.md`, and architecture/PyTree tests to
  document and enforce the simplified boundaries.
- Red/green notes: the focused cleanup tests first failed on the old cached
  fields, dead validator, callback delegates, and CAMulator accessor; after the
  cleanup the same focused suite passed.
- Validation run for this change: the focused red cleanup tests failed for the
  expected old symbols, then the same focused suite passed after
  implementation. Focused affected pytest, Black, `git diff --check`, flake8,
  mypy, full fast pytest, full pytest, and coverage pytest also passed.
  Coverage remained source-focused at 90% total. The existing Black Python
  3.13/target-3.14 warning and JAX dtype-promotion warning remain.

### 2026-06-11: Shared Period-Average Output Writer

- Added `vercor.output.period_files.write_period_average_netcdf()` as the
  shared log/build/write/clear lifecycle for period-average NetCDF files,
  keeping direct `h5netcdf` access in `vercor.output.netcdf`.
- Refactored JAXGCM and Veros average writers to provide model-specific mean,
  coordinate, and metadata builders to the shared writer while preserving JCM
  shape-derived coordinates/units and Veros native metadata/axis policy.
- Added focused tests for successful writes, data-variable metadata transforms,
  and preserving accumulated samples when a write fails.
- Updated `DESIGN.md`, `DEPENDENCIES.md`, and architecture tests to record and
  enforce the shared period-file lifecycle boundary.
- Red/green notes: `tests/test_period_files.py` first failed with missing
  `vercor.output.period_files`; after adding the helper, focused period-file
  tests passed. Mypy then caught overly narrow test callback annotations, which
  were corrected to `Mapping[str, OutputVariable]`.
- Validation run for this change:
  `conda run -n scipy pytest tests/test_period_files.py -q --tb=short`,
  `conda run -n scipy pytest tests/test_period_files.py tests/test_period_averages.py tests/test_output_datasets.py tests/test_output_netcdf.py tests/test_external_components_coverage.py -q --tb=short`,
  `conda run -n scipy pytest tests/test_api_boundaries.py -q --tb=short`,
  `conda run -n scipy pytest tests/test_period_files.py tests/test_period_averages.py tests/test_output_datasets.py tests/test_output_netcdf.py tests/test_external_components_coverage.py tests/test_api_boundaries.py -q --tb=short`,
  `conda run -n scipy black vercor examples tests`, `git diff --check`,
  `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`,
  `conda run -n scipy mypy vercor examples tests`,
  `conda run -n scipy pytest tests/ -q --fast --tb=short`,
  `conda run -n scipy pytest tests/ -q --tb=short`, and
  `conda run -n scipy pytest --cov=vercor tests/ -q --tb=short` passed.
  Coverage remained source-focused at 90% total. The existing Black Python
  3.13/target-3.14 warning and JAX dtype-promotion warning remain.

### 2026-06-11: Shared JCM/Veros Output Dataset Helpers

- Added `vercor.output.datasets` for shared one-step time-coordinate variables
  and first-use dimension discovery across output-variable maps.
- Extended `vercor.output.period_averages` with shared helpers for accumulating
  `OutputVariable` mappings and converting accumulated period means into
  one-time-step output variables, keeping JCM/Veros-specific extraction,
  metadata, and dimension policy local to their adapters.
- Refactored JAXGCM and Veros period output to use the shared dataset and
  period-output helpers, preserving direct `h5netcdf` output through
  `vercor.output.netcdf` with no xarray conversion in `vercor.output`.
- Hardened the h5netcdf writer to reject reused dimension names with conflicting
  sizes before creating invalid datasets.
- Updated `DESIGN.md`, `DEPENDENCIES.md`, and architecture tests to record and
  enforce the shared helper ownership.
- Red/green notes: focused helper tests first failed on missing
  `accumulate_output_variables` and missing `vercor.output.datasets`. After
  adding the helpers and refactoring adapters, the focused output/API suite
  passed.
- Validation run for this change:
  `conda run -n scipy pytest tests/test_period_averages.py tests/test_output_datasets.py tests/test_output_netcdf.py tests/test_external_components_coverage.py tests/test_api_boundaries.py -q --tb=short`,
  `conda run -n scipy black vercor examples tests`, `git diff --check`,
  `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`,
  `conda run -n scipy mypy vercor examples tests`,
  `conda run -n scipy pytest tests/ -q --fast --tb=short`,
  `conda run -n scipy pytest tests/ -q --tb=short`, and
  `conda run -n scipy pytest --cov=vercor tests/ -q --tb=short` passed.
  Coverage remained source-focused at 90% total. The existing Black Python
  3.13/target-3.14 warning and JAX dtype-promotion warning remain.

### 2026-06-10: Shared h5netcdf Output Helpers

- Added shared period-output conversion helpers in `vercor.output.period_averages`
  for mapping `OutputVariable` values to accumulator samples, applying
  adapter-specific empty-accumulator errors, and reshaping period means into
  one-time-step output variables with explicit dimension ordering.
- Refactored JAXGCM and Veros average writers to use the shared conversion
  helpers while keeping model-specific coordinate extraction, metadata, and
  dimension policy local to their adapters.
- Replaced the final runtime-view `xarray` writer with `OutputVariable` maps and
  the existing `vercor.output.netcdf.write_netcdf_dataset` h5netcdf boundary.
  Tests now read runtime output back with `h5netcdf`, and architecture tests
  assert `vercor.output.runtime` does not import xarray or call `.to_netcdf()`.
- Updated `DESIGN.md` and `DEPENDENCIES.md` to record shared output helper
  ownership and the direct h5netcdf runtime-output path.
- Red/green notes: new period-helper tests first failed on missing
  `mean_samples_or_raise`; the output ownership test then failed on
  `import xarray as xr` in `vercor.output.runtime`. After adding shared helpers,
  refactoring adapters, and delegating runtime writes to `write_netcdf_dataset`,
  the focused regressions passed.
- Validation run for this change:
  `conda run -n scipy pytest tests/test_period_averages.py tests/test_component_base_coverage.py::test_read_forcing_and_runtime_write_round_trip tests/test_api_boundaries.py -q --tb=short`,
  `conda run -n scipy black vercor examples tests`, `git diff --check`,
  `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`,
  `conda run -n scipy mypy vercor examples tests`,
  `conda run -n scipy pytest tests/ -q --fast --tb=short`,
  `conda run -n scipy pytest tests/ -q --tb=short`, and
  `conda run -n scipy pytest --cov=vercor tests/ -q --tb=short` passed.
  Coverage remained source-focused at 90% total. The existing Black Python
  3.13/target-3.14 warning and JAX dtype-promotion warning remain.

### 2026-06-10: Unified GCM Output Package

- Replaced the parallel JAXGCM/Veros setup-external output modules with the
  canonical `vercor.output` package. Runtime-view final output now lives in
  `vercor.output.runtime` behind lazy top-level reexports, shared period
  accumulation/time/variable/NetCDF helpers live in focused `vercor.output`
  modules, and model-specific period-output adaptation lives in
  `vercor.output.jax_gcm` and `vercor.output.veros`.
- Removed `vercor.setups.external.jax_gcm_output`,
  `vercor.setups.external.period_averages`, and
  `vercor.setups.external.veros_output` without compatibility wrappers. Updated
  JAXGCM/Veros setup and runtime imports plus architecture tests to enforce the
  hard move and centralized `h5netcdf` ownership in `vercor.output.netcdf`.
- Updated `DESIGN.md` and `DEPENDENCIES.md` to record the new output ownership
  and dependency split. `vercor.output.__init__` keeps runtime-output reexports
  lazy so period-output imports do not pull runtime-state internals into setup
  adapters.
- Red/green notes: the focused output/API suite first failed on
  `vercor.output` still being a module instead of a package. After the package
  migration it passed. The first full suite exposed one stale test monkeypatch
  targeting the lazy top-level facade instead of `vercor.output.runtime`; the
  test was updated to patch the new owner module and the isolated regression
  passed.
- Validation run for this change:
  `conda run -n scipy pytest tests/test_period_averages.py tests/test_jax_gcm_output_frequency.py tests/test_external_components_coverage.py tests/test_api_boundaries.py tests/test_production_numpy_boundaries.py -q --tb=short`,
  `conda run -n scipy black vercor examples tests`,
  `git diff --check`,
  `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`,
  `conda run -n scipy mypy vercor examples tests`,
  `conda run -n scipy pytest tests/ -q --fast --tb=short`,
  `conda run -n scipy pytest tests/ -q --tb=short`, and
  `conda run -n scipy pytest --cov=vercor tests/ -q --tb=short` passed.
  Coverage remained source-focused at 90% total. The existing Black Python
  3.13/target-3.14 warning and JAX dtype-promotion warning remain.

### 2026-06-10: Trivial Internal Wrapper Cleanup

- Removed private pass-through helpers that only forwarded to canonical
  implementations: flux `_as_jax_array()` aliases, JAXGCM/Veros output
  `_jax_array()` aliases, the bilinear interpolator source-mask delegate,
  forcing-data legacy transpose/flip delegates, and the CAMulator land
  temperature-normalization delegate.
- Replaced call sites with direct `as_jax_real_array(...)`,
  `_extrapolation.valid_scalar_source_mask(...)`, and `jnp.flip(...)` calls.
  Public compatibility aliases and documented facade/accessor boundaries were
  left unchanged.
- Removed the stale CAMulator helper-only test while keeping component-level
  JAX-array storage coverage.
- The precise cleanup scan reported no removed-helper definitions or call sites:
  `rg -n "def (_as_jax_array|_jax_array|_prepare_camulator_land_surface_temperature|_legacy_transpose_to_time_last_order|_flip_legacy_latitude_axis)\b|_ensure_src_mask\b|\b(_as_jax_array|_jax_array|_prepare_camulator_land_surface_temperature|_legacy_transpose_to_time_last_order|_flip_legacy_latitude_axis)\(" vercor tests examples`,
- Validation run for this change:
  `conda run -n scipy pytest tests/test_fluxes_utilities.py tests/test_bilinear_rectilinear_interpolator.py tests/test_forcing_data.py tests/test_jax_gcm_output_frequency.py tests/test_data_component_kernels.py tests/test_camulator_component_kernels.py tests/test_external_components_coverage.py -q --tb=short`,
  `conda run -n scipy black vercor examples tests`,
  `git diff --check`,
  `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`,
  `conda run -n scipy mypy vercor examples tests`,
  `conda run -n scipy pytest tests/ -q --fast --tb=short`, and
  `conda run -n scipy pytest tests/ -q --tb=short` passed. The broader
  substring cleanup scan still matches preserved public/test names such as
  `torch_tensor_from_jax_array`.

### 2026-06-10: Internal Naming Consistency Pass

- Swept the Python codebase inventory for semantically similar definitions:
  206 Python files, 1,528 functions/methods, and 180 classes were parsed
  successfully. The pass focused on internal/private names only; public and
  ambiguous similarities were left unchanged for API stability.
- Renamed local internal helpers without changing behavior:
  `vercor.fluxes.bulk_formula_cesm._asarray()` is now `_as_jax_array()`,
  the 3-argument callable adapter in
  `vercor.components._callable_wrappers.normalize_component_step_callable()`
  is now `step_fields_context_and_payload()`, and
  `vercor.setups.external.veros_output._coordinate_variable()` is now
  `_extract_coordinate_variable()`.
- Left intentionally parallel names unchanged: JAX/NumPy dtype helpers,
  runtime-cache owner versus resource-facade methods,
  `step_runtime_state()` versus `step_host_runtime_state()`, component factory
  helpers, scalar/vector and inverse helper pairs, and the public historical
  `shr_flux_atmIce()` physics API.
- Updated `DESIGN.md` and `DEPENDENCIES.md` to record the internal naming
  boundary rationale for callable adapters, flux JAX-array normalization, and
  Veros output variable/coordinate extraction. No module dependency order
  changed.
- Validation run for this change:
  `conda run -n scipy pytest tests/test_fluxes_utilities.py tests/test_slab_kernels.py tests/test_component_base_coverage.py tests/test_external_components_coverage.py tests/test_production_numpy_boundaries.py -q --tb=short`,
  `conda run -n scipy black vercor examples tests`,
  `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`,
  `conda run -n scipy mypy vercor examples tests`,
  `conda run -n scipy pytest tests/ -q --fast --tb=short`,
  `conda run -n scipy pytest tests/ -q --tb=short`, and
  `conda run -n scipy pytest --cov=vercor tests/ -q --tb=short` passed.
  Coverage remained source-focused at 90% total. The existing Black Python
  3.13/target-3.14 warning and JAX dtype-promotion warning remain.

### 2026-06-09: JAX-Backed JCM/Veros Output Arrays

- Converted the shared period-average accumulator to store JAX-backed running
  sums, finite-value counts, and mean samples while preserving current
  `nanmean` behavior. Counts now use VerCOR's canonical index dtype.
- Updated JAXGCM and Veros period-output extraction/mean-shaping to keep
  VerCOR-owned output values as JAX arrays and to use `vercor.dtypes` helpers.
  Direct NumPy imports were removed from `jax_gcm_output.py`,
  `period_averages.py`, and `veros_output.py`.
- Added `vercor.host_arrays` helpers for explicit final host transfer and the
  deliberate host `int64` NetCDF time-coordinate exception. The output writers
  now convert to host arrays only at the `h5netcdf` boundary.
- Tightened production NumPy-boundary and output tests so accumulator internals,
  JCM accumulation, and Veros snapshots/accumulation are JAX-backed before file
  writing. Updated `DESIGN.md` and `DEPENDENCIES.md` for the new boundary.
- Validation run for this change:
  baseline
  `conda run -n scipy pytest tests/ -q --fast --tb=short` passed after sandbox
  approval for `conda run -n scipy`. Focused red
  `conda run -n scipy pytest tests/test_period_averages.py tests/test_external_components_coverage.py::test_jax_gcm_write_output_persists_mean_dataset tests/test_external_components_coverage.py::test_veros_output_snapshot_uses_variable_metadata_and_current_timestep tests/test_external_components_coverage.py::test_veros_write_output_persists_period_mean_and_coordinates tests/test_production_numpy_boundaries.py -q --tb=short`
  failed as expected because the accumulator/snapshots were still NumPy-backed
  and the output modules still imported NumPy. After implementation,
  `conda run -n scipy pytest tests/test_period_averages.py tests/test_external_components_coverage.py::test_jax_gcm_write_output_persists_mean_dataset tests/test_external_components_coverage.py::test_veros_output_snapshot_uses_variable_metadata_and_current_timestep tests/test_external_components_coverage.py::test_veros_write_output_persists_period_mean_and_coordinates tests/test_production_numpy_boundaries.py -q --tb=short`,
  `conda run -n scipy mypy vercor/host_arrays.py vercor/setups/external/period_averages.py vercor/setups/external/jax_gcm_output.py vercor/setups/external/veros_output.py tests/test_period_averages.py tests/test_external_components_coverage.py tests/test_production_numpy_boundaries.py`,
  `conda run -n scipy black vercor examples tests`,
  `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`,
  `conda run -n scipy mypy vercor examples tests`,
  `conda run -n scipy pytest tests/test_api_boundaries.py::test_setup_helper_and_external_output_ownership_boundaries -q --tb=short`,
  `conda run -n scipy pytest tests/ -q --fast --tb=short`,
  `conda run -n scipy pytest tests/ -q --tb=short`, and
  `conda run -n scipy pytest --cov=vercor tests/ -q --tb=short` passed.
  Coverage remained source-focused at 90% total. The existing Black Python
  3.13/target-3.14 warning and JAX dtype-promotion warning remain.
- Failed approaches recorded: the first post-implementation fast suite exposed
  a stale API-boundary assertion that still required NumPy in
  `period_averages.py` and `veros_output.py`; the assertion was updated to make
  `host_arrays.py` the explicit NumPy owner for output host conversion.

### 2026-06-08: Veros Spinup Period-Average Accumulation

- Fixed Veros setup spinup to accumulate selected `output_variables` into the
  same `PeriodAverageAccumulator` used by runtime output, matching the existing
  JAXGCM behavior where spinup samples seed the first averaged output period.
  Spinup still does not write NetCDF files; runtime period gates remain the
  only write boundary.
- Added a shared package-internal Veros output helper that extracts and
  accumulates one Veros state. Runtime output recording and setup spinup now
  use that helper, so extraction/accumulation behavior stays in one owner.
- Added regression coverage proving two Veros spinup steps accumulate two
  selected-output samples without breaking setup-time SST seeding.
- Validation run for this change:
  focused red
  `conda run -n scipy pytest tests/test_external_components_coverage.py::test_veros_initialize_spinup_accumulates_selected_outputs -q --tb=short`
  failed as expected because no spinup states were accumulated. After
  implementation,
  `conda run -n scipy pytest tests/test_external_components_coverage.py::test_veros_initialize_spinup_accumulates_selected_outputs -q --tb=short`,
  `conda run -n scipy pytest tests/test_external_components_coverage.py::test_veros_initialize_can_spin_up_and_extract_surface_temperature tests/test_external_components_coverage.py::test_veros_initialize_spinup_accumulates_selected_outputs tests/test_external_components_coverage.py::test_veros_step_records_selected_outputs_and_writes_on_gate tests/test_external_components_coverage.py::test_veros_step_skips_output_when_no_variables_selected tests/test_external_components_coverage.py::test_veros_write_output_persists_period_mean_and_coordinates -q --tb=short`,
  `conda run -n scipy black vercor examples tests`,
  `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`,
  `conda run -n scipy mypy vercor examples tests`,
  `conda run -n scipy pytest tests/ -q --fast --tb=short`,
  `conda run -n scipy pytest tests/ -q --tb=short`, and
  `conda run -n scipy pytest --cov=vercor tests/ -q --tb=short` passed.
  Coverage remained source-focused at 90% total. The existing Black Python
  3.13/target-3.14 warning and JAX dtype-promotion warning remain.
- Failed approaches recorded: the first post-implementation focused run exposed
  a test-fixture issue where fake spinup state temperatures drifted by 1 K;
  the fixture was narrowed to vary only `step_id` so the test remains focused
  on period accumulation.

### 2026-06-04: Veros Average Output Dimension Order

- Fixed Veros h5netcdf average output to keep snapshot extraction and
  accumulation in Veros internal array order, then transpose spatial axes once
  at write time. Files now keep VerCOR's lowercase `time` dimension while
  matching native Veros spatial NetCDF order such as
  `temp(time, zt, yt, xt)` and `psi(time, yu, xu)`.
- Expanded Veros average-output coverage for `temp`, `salt`, `u`,
  `surface_taux`, and `psi`. The writer test now asserts each persisted value
  is the elementwise mean of two runtime snapshots after spatial transposition,
  proving period averaging does not reduce a horizontal or vertical axis.
- Black also normalized the existing Veros/JCM example `output_variables` tuple
  formatting in `examples/run_jcm_with_veros.py` while running the requested
  formatter command across `examples`.
- Validation run for this change:
  focused red
  `conda run -n scipy pytest tests/test_external_components_coverage.py::test_veros_write_output_persists_period_mean_and_coordinates -q --tb=short`
  failed as expected on the old `("time", "xt", "yt", "zt")` dimension order.
  After implementation,
  `conda run -n scipy pytest tests/test_external_components_coverage.py::test_veros_output_snapshot_uses_variable_metadata_and_current_timestep tests/test_external_components_coverage.py::test_veros_write_output_persists_period_mean_and_coordinates tests/test_period_averages.py -q --tb=short`,
  `conda run -n scipy black vercor examples tests`,
  `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`,
  `conda run -n scipy mypy vercor examples tests`,
  `conda run -n scipy pytest tests/ -q --fast --tb=short`,
  `conda run -n scipy pytest tests/ -q --tb=short`, and
  `conda run -n scipy pytest --cov=vercor tests/ -q --tb=short` passed.
  Coverage remained source-focused at 90% total. The existing Black Python
  3.13/target-3.14 warning and JAX dtype-promotion warning remain.
- Failed approaches recorded: no failed implementation approach; the focused
  red test caught the intended pre-fix dimension-order regression.

### 2026-06-04: Streaming Period-Average Output Accumulation

- Replaced the JAXGCM and Veros period-output sample buffers with a shared
  host-side `PeriodAverageAccumulator` that stores one running sum and one
  per-element finite-value count array per variable. This preserves current
  `np.nanmean` behavior, including sparse/all-NaN cells, without retaining each
  timestep/snapshot until output.
- Updated JAXGCM output recording to accumulate prediction variables over their
  prediction `time` dimension at record time, then add the NetCDF `time`
  dimension and canonical output ordering only at write time. JAXGCM spinup
  predictions continue seeding the first output period through the accumulator.
- Updated Veros output recording to accumulate selected extracted snapshots and
  write the same `veros.averages.YYYY-MM-DD.nc` mean files with native
  coordinate/metadata preservation. The output writers clear accumulators only
  after successful file writes.
- Updated `DESIGN.md`, `DEPENDENCIES.md`, explicit NumPy/API boundary coverage,
  and accumulator/runtime/writer tests for the new package-internal host output
  boundary.
- Validation run for this change:
  focused red
  `conda run -n scipy pytest tests/test_period_averages.py -q --tb=short`
  failed as expected on missing `vercor.setups.external.period_averages`.
  After implementation,
  `conda run -n scipy pytest tests/test_period_averages.py -q --tb=short`,
  `conda run -n scipy pytest tests/test_period_averages.py tests/test_external_components_coverage.py::test_jax_gcm_initialize_uses_provided_forcing_and_can_spin_up tests/test_external_components_coverage.py::test_jax_gcm_step_maps_outputs_and_respects_output_gate tests/test_external_components_coverage.py::test_jax_gcm_write_output_persists_mean_dataset tests/test_external_components_coverage.py::test_jax_gcm_write_output_preserves_model_calendar_attrs tests/test_external_components_coverage.py::test_veros_write_output_persists_period_mean_and_coordinates tests/test_external_components_coverage.py::test_veros_step_records_selected_outputs_and_writes_on_gate tests/test_external_components_coverage.py::test_veros_step_skips_output_when_no_variables_selected -q --tb=short`,
  `conda run -n scipy pytest tests/test_api_boundaries.py::test_setup_helper_and_external_output_ownership_boundaries tests/test_production_numpy_boundaries.py::test_numpy_imports_match_explicit_host_boundaries tests/test_external_components_coverage.py::test_jax_gcm_write_output_persists_mean_dataset tests/test_external_components_coverage.py::test_veros_write_output_persists_period_mean_and_coordinates -q --tb=short`,
  `conda run -n scipy pytest tests/test_period_averages.py tests/test_external_components_coverage.py tests/test_api_boundaries.py tests/test_production_numpy_boundaries.py tests/test_jax_gcm_output_frequency.py -q --fast --tb=short`,
  `conda run -n scipy black vercor examples tests`,
  `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`,
  `conda run -n scipy mypy vercor examples tests`,
  `conda run -n scipy pytest tests/ -q --fast --tb=short`,
  `conda run -n scipy pytest tests/ -q --tb=short`, and
  `conda run -n scipy pytest --cov=vercor tests/ -q --tb=short` passed.
  Coverage remained source-focused at 90% total. The existing Black Python
  3.13/target-3.14 warning and JAX dtype-promotion warning remain.
- Failed approaches recorded: an initial focused boundary command used the
  stale test name `test_external_adapter_helpers_stay_in_owner_modules`; rerun
  with `test_setup_helper_and_external_output_ownership_boundaries` covered the
  intended boundary assertions.

### 2026-06-03: Veros h5netcdf Period Output

- Disabled Veros native output machinery in the explicit runtime settings
  boundary by setting `diskless_mode=True` alongside the NumPy backend and
  force-overwrite settings.
- Added opt-in Veros period-output support through `make_veros_gcm` /
  `VerosGCMSetupState` `output_variables` and `output_frequency` arguments.
  Selected Veros variables are extracted with native Veros metadata, current
  timestep selection, ghost-cell removal, and native dimension order, then
  written as period means to `veros.averages.YYYY-MM-DD.nc` via `h5netcdf`.
- Kept output file I/O in the new `vercor.setups.external.veros_output` host
  boundary. Veros runtime now records selected snapshots and flushes through the
  existing JAXGCM day/month/year cadence helper, leaving the SST exchange output
  unchanged when no output variables are selected.
- Validation run for this change:
  baseline
  `conda run -n scipy pytest tests/ -v --fast 2>&1 | tail -20` passed.
  Focused red
  `conda run -n scipy pytest tests/test_external_components_coverage.py::test_configure_veros_runtime_sets_diskless_mode tests/test_external_components_coverage.py::test_veros_output_snapshot_uses_variable_metadata_and_current_timestep tests/test_external_components_coverage.py::test_veros_write_output_persists_period_mean_and_coordinates tests/test_external_components_coverage.py::test_veros_output_variables_rejects_bare_string tests/test_external_components_coverage.py::test_veros_constructor_builds_jax_backed_grid tests/test_external_components_coverage.py::test_veros_step_records_selected_outputs_and_writes_on_gate tests/test_external_components_coverage.py::test_veros_step_skips_output_when_no_variables_selected tests/test_api_boundaries.py::test_setup_helper_and_external_output_ownership_boundaries -q --tb=short`
  failed as expected on missing diskless mode, missing `veros_output`, missing
  Veros output API args, and missing runtime output hooks. After implementation,
  focused feature/API/NumPy-boundary checks passed. Then
  `conda run -n scipy pytest tests/test_external_components_coverage.py tests/test_api_boundaries.py tests/test_production_numpy_boundaries.py tests/test_jax_gcm_output_frequency.py -q --fast --tb=short`,
  `conda run -n scipy black vercor examples tests`,
  `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`,
  `conda run -n scipy mypy vercor examples tests`,
  `conda run -n scipy pytest tests/ -q --fast --tb=short`,
  `conda run -n scipy pytest tests/ -q --tb=short`, and
  `conda run -n scipy pytest --cov=vercor tests/ -q --tb=short` passed.
  Coverage remained source-focused at 90% total. The existing Black Python
  3.13/target-3.14 warning and JAX dtype-promotion warning remain.
- Failed approaches recorded: the first no-op runtime test omitted the
  `_step_function` fake even though `step_veros_runtime` reads it before the
  zero-substep loop; adding an identity fake fixed the fixture. The first writer
  test used fake Veros settings without `coord_degree`, which real Veros
  coordinate metadata requires; adding that setting fixed the fixture. The first
  lint/type pass exposed a Black/E203 slice-spacing conflict and an overly
  precise mutable-sequence type annotation; adding a local `noqa` and matching
  the existing JAXGCM writer's `MutableSequence[Any]` pattern fixed them.

### 2026-06-03: JAXGCM h5netcdf Average Output Writer

- Replaced the JAXGCM averages writer's xarray merge/mean/to_netcdf path with a
  direct `h5netcdf` writer that consumes prediction dynamics/physics/times
  directly, writes runtime-calendar time metadata, preserves JCM unit-table
  metadata, and clears the prediction buffer only after a successful write.
- Updated the host runtime output gate to pass model coordinates, runtime
  timestamp, and the model physics module into the writer. Added coverage that
  `to_xarray()` is not called, h5netcdf dimensions/metadata are persisted, and
  DateTime360/DateTime365 calendar attrs are written.
- Updated `DESIGN.md`, `DEPENDENCIES.md`, and the NumPy-boundary allowlist for
  the new output-file host boundary.
- Validation run for this change:
  focused red
  `conda run -n scipy pytest tests/test_external_components_coverage.py::test_jax_gcm_write_output_persists_mean_dataset tests/test_api_boundaries.py::test_jax_gcm_average_writer_bypasses_xarray_adapter -q --tb=short`
  failed as expected on the old writer missing the `coords` keyword and still
  importing xarray. After implementation, the focused writer/calendar/runtime
  gate/API checks passed. Then
  `conda run -n scipy pytest tests/test_external_components_coverage.py::test_jax_gcm_write_output_persists_mean_dataset -q --tb=short`,
  `conda run -n scipy pytest tests/test_jax_gcm_output_frequency.py tests/test_external_components_coverage.py tests/test_api_boundaries.py -q --fast --tb=short`,
  `conda run -n scipy black vercor examples tests`,
  `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`,
  `conda run -n scipy mypy vercor examples tests`,
  `conda run -n scipy pytest tests/ -q --fast --tb=short`,
  `conda run -n scipy pytest tests/ -q --tb=short`, and
  `conda run -n scipy pytest --cov=vercor tests/ -q --tb=short` passed.
  Coverage remained source-focused at 90% total. The existing Black Python
  3.13/target-3.14 warning and JAX dtype-promotion warning remain.
- Failed approaches recorded: the first h5netcdf green run used a fake coordinate
  fixture with `layers=3` but only two level centers, causing an incompatible
  NetCDF dimension size; aligning the level coordinate length fixed it. The
  first model-calendar write used a boolean NetCDF attr for
  `fixed_30_day_months`, which h5netcdf rejects in valid NetCDF mode; storing
  the flag as `0`/`1` fixed the file metadata.

### 2026-06-02: Unit-Test Speedup Pass

- Added test-cache defaults in `tests/conftest.py` so Matplotlib and
  fontconfig use writable temp cache paths during pytest while preserving any
  caller-provided environment values.
- Added an internal identity interpolator path for identical-grid bilinear and
  conservative regridders. Identical grids now avoid eager interpolator/remapper
  construction while preserving the existing unchanged-field call behavior.
- Consolidated optional setup-import boundary checks from two subprocesses per
  import case to one isolated subprocess per case, keeping both output-marker
  and heavy-optional-module assertions. Reduced the runtime profile smoke grid
  from 4x3 to 2x2 while keeping the parser/build/run/cache coverage.
- Validation run for this change:
  focused red
  `conda run -n scipy pytest tests/test_bilinear_rectilinear_regridder.py::test_regridder_identical_grid_skips_interpolator_construction tests/test_conservative_rectilinear_regridder.py::test_regridder_identical_grid_skips_remapper_construction -q --tb=short`
  failed as expected on the constructors still building the expensive
  interpolator/remapper. Focused cache red
  `conda run -n scipy pytest tests/test_tools_components_and_plotting.py::test_test_environment_uses_writable_plotting_cache_defaults -q --tb=short`
  failed as expected on missing `MPLBACKEND`.
  After implementation, the same focused checks passed. Then
  `conda run -n scipy pytest tests/test_bilinear_rectilinear_regridder.py tests/test_conservative_rectilinear_regridder.py -q --tb=short`,
  `conda run -n scipy pytest tests/test_api_boundaries.py::test_unrelated_setup_imports_do_not_initialize_optional_adapters -q --tb=short --durations=10`,
  `conda run -n scipy pytest tests/test_tools_components_and_plotting.py::test_plot_component_scalar_vector_comparison_aligns_axes_and_shapes -q --tb=short --durations=5`,
  `conda run -n scipy pytest tests/test_runtime_run_cache.py -q --tb=short --durations=10`,
  `conda run -n scipy black vercor examples tests`,
  `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`,
  `conda run -n scipy mypy vercor examples tests`,
  `conda run -n scipy pytest tests/ -q --fast --tb=short --durations=25`,
  `conda run -n scipy pytest tests/ -q --tb=short --durations=25`, and
  `conda run -n scipy pytest --cov=vercor tests/ -q --tb=short` passed.
- Final duration checks: the fast-suite plotting test dropped from the prior
  ~15s hotspot to 0.22s in the fast duration table after cache warm-up; the
  optional setup-import probes dropped from ~2.4-2.8s each to about 1.02-1.59s
  in the final fast duration table and 0.95-1.16s in the final full duration
  table; runtime profile smoke dropped from ~4.8s focused to 0.69s focused.
  Coverage remained source-focused at 90% total. The existing Black Python
  3.13/target warning and JAX dtype-promotion warning remain.

### 2026-06-02: CAMulator Wind-Filter Boundary Refactor

- Split CAMulator wind artifact tensor mechanics into private
  `vercor.setups.external._camulator_wind_filtering`, which now owns PyTorch
  mask/kernel artifact construction, field filtering, and selected in-place
  tensor updates.
- Kept `vercor.setups.external.camulator_wind_filter` as the public facade for
  configuration loading/validation, compatibility functions, and the existing
  log-and-skip failure policy used during optional post-processing.
- Added focused behavior and architecture coverage for shape-stable wind-filter
  artifacts, target-only tensor mutation, log-and-skip failure handling, and the
  public-to-private wind-filter boundary. Updated `DESIGN.md` and
  `DEPENDENCIES.md` for the new owner split.
- Validation run for this change:
  focused red
  `conda run -n scipy pytest tests/test_camulator_component_kernels.py tests/test_api_boundaries.py -q --fast --tb=short`
  failed as expected on missing
  `vercor.setups.external._camulator_wind_filtering` and the missing private
  boundary file assertion. Focused green with the same command passed after
  implementation. Then `conda run -n scipy black vercor examples tests`,
  focused post-format pytest with the same command,
  `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`,
  `conda run -n scipy mypy vercor examples tests`,
  `conda run -n scipy pytest tests/ -q --fast --tb=short`, and
  `conda run -n scipy pytest tests/ -q --tb=short` passed. The existing Black
  Python 3.13/target-3.14 warning and JAX dtype-promotion warning remain.
- Failed approaches recorded: the first flake8 pass found one Black-formatted
  computed slice with E203 in `_camulator_wind_filtering.py`; rewriting the
  split helper to use explicit `start`/`end` variables fixed the style issue.

### 2026-06-02: External Adapter Factory/Setup-State Boundary Refactor

- Removed the public `JCMState` compatibility aliases from
  `vercor.setups.external` and `vercor.setups.external.jax_gcm`; the canonical
  state bundle owner is now only `vercor.setups.external.jax_gcm_state`.
- Split CAMulator atmosphere setup-state ownership into
  `vercor.setups.external.camulator_gcm_state.CAMulatorGCMSetupState`, leaving
  `vercor.setups.external.camulator` as a thin `make_camulator_gcm(...)`
  factory that binds the host-component lifecycle methods.
- Updated boundary/runtime tests, `DESIGN.md`, and `DEPENDENCIES.md` for the
  stricter JAXGCM public surface and CAMulator setup-state owner.
- Validation run for this change:
  initial focused red
  `conda run -n scipy pytest tests/test_api_boundaries.py tests/test_camulator_component_kernels.py tests/test_coupler_runtime.py -q --fast --tb=short`
  first exposed an over-eager test import of the not-yet-created CAMulator
  setup-state module; after moving that assertion back to the boundary test,
  the same focused red command failed as expected on the remaining `JCMState`
  alias and missing `camulator_gcm_state.py`. Focused green with the same
  command passed after implementation. Then
  `conda run -n scipy black vercor examples tests`,
  focused post-format pytest with the same boundary/runtime command,
  `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`,
  `conda run -n scipy mypy vercor examples tests`, and
  `conda run -n scipy pytest tests/ -q --fast --tb=short` passed. The first
  full `conda run -n scipy pytest tests/ -q --tb=short` exposed one stale
  full-only boundary assertion in `tests/test_runtime_state.py` that still
  looked for `def step(...)` in `camulator.py`; the exact failing test then
  passed after retargeting it to `camulator_gcm_state.py`. Final Black,
  focused boundary/runtime pytest, flake8, mypy, full fast pytest, and full
  pytest all passed. The existing Black Python 3.13/target-3.14 warning and
  JAX dtype-promotion warning remain.
- Failed approaches recorded: the initial top-level import of
  `camulator_gcm_state` in the CAMulator kernel tests caused collection to fail
  before the intended red boundary assertion, and the first full suite run found
  a stale boundary test that still treated the CAMulator factory as the
  setup-state owner. Both were corrected in the test harness.

### 2026-06-02: Asset and Forcing-Data Boundary Refactor

- Refactored `vercor.assets` so the generic asset cache/download/checksum layer
  uses private normalized asset helpers and no longer embeds forcing-specific
  error wording. Concrete forcing registries remain in
  `vercor.setups.data.assets`.
- Split `vercor.forcing_data.read_forcing()` into private path-resolution,
  NetCDF variable lookup, legacy transpose, and latitude-flip helpers while
  preserving successful array behavior. Missing mapping keys and missing
  NetCDF variables now raise distinct `KeyError` messages.
- Added explicit `year_type` validation in `vercor.forcing_index`, with
  `vercor.calendar` compatibility delegates preserving the same behavior.
- Added focused asset and forcing-data test files, updated the existing
  read-forcing coverage expectation, and kept the small focused tests included
  in `--fast` runs.
- Updated `DESIGN.md` and `DEPENDENCIES.md` for the refined asset and
  forcing-data boundaries, including removal of the obsolete
  data-component-reader-class wording.
- The broader audit findings for public `JCMState` reexport cleanup and
  CAMulator setup-state splitting were intentionally left out of this
  asset/forcing-data pass and completed in the later 2026-06-02 external
  adapter factory/setup-state boundary refactor.
- Validation run for this change:
  baseline `conda run -n scipy pytest tests/ -q --fast --tb=short`, focused red
  `conda run -n scipy pytest tests/test_assets.py tests/test_forcing_data.py tests/test_tools_time_and_forcing.py::test_forcing_index_rejects_unknown_year_type tests/test_component_base_coverage.py::test_read_forcing_and_runtime_write_round_trip -q --fast --tb=short`,
  focused green after implementation with the same focused command,
  focused boundary/API
  `conda run -n scipy pytest tests/test_assets.py tests/test_forcing_data.py tests/test_tools_time_and_forcing.py tests/test_component_base_coverage.py::test_read_forcing_and_runtime_write_round_trip tests/test_api_boundaries.py -q --fast --tb=short`,
  `conda run -n scipy black vercor examples tests`, focused post-format with
  the same boundary/API command,
  `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`,
  `conda run -n scipy mypy vercor examples tests`,
  `conda run -n scipy pytest tests/ -q --fast --tb=short`, and
  `conda run -n scipy pytest tests/ -q --tb=short`. The existing Black Python
  3.13/target-3.14 warning and JAX dtype-promotion warning remain.
- Failed approach recorded: no failed implementation approach was encountered;
  the only failures were the intentional red tests for generic asset wording,
  missing-variable error reporting, and invalid forcing `year_type` validation.

### 2026-06-02: Calendar Forcing-Index Boundary Refactor

- Added `vercor.forcing_index` as the focused owner for daily forcing lookup
  policy, including Gregorian month lengths, noleap mapping, 360-day to
  Gregorian day mapping, one-based forcing day selection, and zero-based daily
  forcing indexes.
- Kept `vercor.calendar` focused on calendar constants, model-calendar datetime
  values, leap-year logic, and month/day conversion while preserving the
  historic forcing-index imports through thin compatibility delegates.
- Updated `vercor.time_selection` and `vercor.runtime.time` to import forcing
  policy from `vercor.forcing_index`, removing the local 360-day wrapper from
  time selection.
- Added boundary and behavior coverage for forcing-index ownership, runtime
  import direction, absence of a `vercor.forcing_index` top-level import cycle,
  and compatibility-delegate parity across Gregorian, noleap, leap-day, and
  360-day daily forcing cases.
- Updated `DESIGN.md` and `DEPENDENCIES.md` for the new forcing-index owner.
- Validation run for this change:
  baseline `conda run -n scipy pytest tests/ -q --fast --tb=short`, focused red
  `conda run -n scipy pytest tests/test_api_boundaries.py tests/test_tools_time_and_forcing.py -q --fast --tb=short`,
  focused green after implementation
  `conda run -n scipy pytest tests/test_api_boundaries.py tests/test_tools_time_and_forcing.py -q --fast --tb=short`,
  focused runtime
  `conda run -n scipy pytest tests/test_coupler_runtime.py -q --fast --tb=short`,
  `conda run -n scipy black vercor examples tests`, focused post-format
  `conda run -n scipy pytest tests/test_api_boundaries.py tests/test_tools_time_and_forcing.py -q --fast --tb=short`,
  `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`,
  `conda run -n scipy mypy vercor examples tests`,
  `conda run -n scipy pytest tests/ -q --fast --tb=short`, and
  `conda run -n scipy pytest tests/ -q --tb=short`. The existing Black Python
  3.13/target-3.14 warning and JAX dtype-promotion warning remain.
- Failed approach recorded: no failed implementation approach was encountered;
  the only failures were the intentional red boundary/behavior tests before
  adding `vercor.forcing_index` and moving runtime imports to the new owner.

### 2026-06-02: Bilinear Interpolator Private-Owner Boundary Refactor

- Split bilinear interpolation internals into private owner modules under
  `vercor.interpolators`: `_bilinear_geometry` for spherical geometry and
  orientation checks, `_bilinear_weights` for target-to-source cell lookup and
  bilinear weights, and `_bilinear_extrapolation` for nearest/IDW fill policy
  plus valid-source mask normalization.
- Kept `BilinearRectilinearInterpolator` as the public PyTree facade with the
  same constructor options, public precomputed weight attributes, scalar/vector
  methods, JIT behavior, and regridder integration.
- Added architecture-boundary coverage for private helper ownership, package
  import-cycle absence, private helper import direction, periodic dateline
  weight construction, and empty-valid-source extrapolation fill behavior.
- Updated `DESIGN.md` and `DEPENDENCIES.md` for the bilinear private-owner
  boundary and recorded the `calendar.py` forcing-index split as a follow-up.
- Validation run for this change:
  baseline `conda run -n scipy pytest tests/ -q --fast --tb=short`, focused red
  `conda run -n scipy pytest tests/test_bilinear_interpolator_boundaries.py -q --tb=short`,
  focused green after implementation
  `conda run -n scipy pytest tests/test_bilinear_interpolator_boundaries.py tests/test_bilinear_rectilinear_interpolator.py tests/test_bilinear_rectilinear_regridder.py -q --tb=short`,
  `conda run -n scipy black vercor examples tests`,
  focused post-format
  `conda run -n scipy pytest tests/test_bilinear_interpolator_boundaries.py tests/test_bilinear_rectilinear_interpolator.py tests/test_bilinear_rectilinear_regridder.py -q --tb=short`,
  `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`,
  `conda run -n scipy mypy vercor examples tests`,
  `conda run -n scipy pytest tests/ -q --fast --tb=short`, and
  `conda run -n scipy pytest tests/ -q --tb=short`. The existing Black Python
  3.13/target-3.14 warning and JAX dtype-promotion warning remain.
- Failed approach recorded: no failed implementation approach was encountered;
  the only failures were the intentional red boundary tests before adding the
  private helper owner modules and public facade delegation.

### 2026-06-02: Logging Facade Private-Owner Boundary Refactor

- Split the former monolithic `vercor.jax_logging` implementation into private
  owner modules under `vercor._logging`: `config` for canonical logger setup,
  `protocols` for logger-like contracts and level checks, `host` for host-side
  message formatting/emission, and `callback` for traced-value partitioning plus
  `JaxCallbackLogger`.
- Kept `vercor.jax_logging` as the only production-facing public facade with an
  explicit `__all__`, preserving existing public imports and callback logging
  behavior while preventing production modules from importing private logging
  internals directly.
- Added architecture-boundary coverage for the thin facade, private logging
  package ownership, private package cycle absence, public API preservation, and
  production import direction.
- Updated `DESIGN.md` and `DEPENDENCIES.md` for the logging facade/private-owner
  boundary.
- Validation run for this change:
  baseline `conda run -n scipy pytest tests/ -q --fast --tb=short`, focused red
  `conda run -n scipy pytest tests/test_logging_boundaries.py -q --tb=short`,
  focused green after implementation
  `conda run -n scipy pytest tests/test_logging_boundaries.py tests/test_coupler_coverage.py -q --tb=short`,
  `conda run -n scipy black vercor examples tests`,
  `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`,
  `conda run -n scipy mypy vercor examples tests`,
  focused post-format
  `conda run -n scipy pytest tests/test_logging_boundaries.py tests/test_coupler_coverage.py -q --tb=short`,
  `conda run -n scipy pytest tests/ -q --fast --tb=short`, and
  `conda run -n scipy pytest tests/ -q --tb=short`. The existing Black Python
  3.13/target-3.14 warning and JAX dtype-promotion warning remain.
- Failed approach recorded: the first focused green run exposed a migrated
  canonical log-format mismatch (`VerCOR ─ ...` instead of the tested
  `VerCOR: ...` format). The private config constant was corrected before
  continuing validation.

### 2026-06-01: External Adapter Setup-State Boundary Refactor

- Added `vercor.setups.external.jax_gcm_state` as the owner for JAXGCM setup
  state, model construction, spinup, and lifecycle callback wiring; the public
  `jax_gcm.py` module now stays focused on the `make_jax_gcm(...)` factory and
  then-existing `JCMState` reexport, which was later removed in the 2026-06-02
  external adapter factory/setup-state boundary refactor.
- Added `vercor.setups.external.veros_gcm_state` as the owner for Veros setup
  state, grid derivation, spinup, and host step delegation; the public
  `veros_gcm.py` module now stays focused on `make_veros_gcm(...)`.
- Added the named tuple-compatible `VerosForcingFields` container so Veros
  forcing fields have explicit names while preserving existing tuple unpacking.
- Strengthened architecture coverage for external setup-state owners, factory
  thinness, external package import cycles, and named Veros forcing fields.
- Updated `DESIGN.md` and `DEPENDENCIES.md` for the new external adapter
  setup-state boundary.
- Validation run for this change:
  focused red
  `conda run -n scipy pytest tests/test_api_boundaries.py::test_setup_helper_and_external_output_ownership_boundaries tests/test_api_boundaries.py::test_jax_gcm_factory_uses_named_runtime_callbacks tests/test_api_boundaries.py::test_external_package_has_no_top_level_import_cycles tests/test_external_components_coverage.py::test_veros_prepare_surface_forcing_fields_shapes_nan_cleanup_and_qnec_gate -q --fast --tb=short`,
  focused green after implementation,
  `conda run -n scipy pytest tests/test_api_boundaries.py tests/test_external_components_coverage.py tests/test_coupler_runtime.py -q --fast --tb=short`,
  `conda run -n scipy black vercor examples tests`,
  `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`,
  `conda run -n scipy mypy vercor examples tests`,
  `conda run -n scipy pytest tests/ -q --fast --tb=short`, and
  `conda run -n scipy pytest tests/ -q --tb=short`. The existing Black Python
  3.13/target-3.14 warning and JAX dtype-promotion warning remain.
- Failed approach recorded: no failed implementation approach was encountered;
  the only failures were the intentional focused red tests before adding the
  setup-state owner modules and named Veros forcing container.

### 2026-06-01: External Adapter Helper Boundary Refactor

- Renamed exported external-adapter helper functions to public
  package-internal names in JAXGCM, CAMulator, and Veros setup modules, while
  leaving underscored helpers as local implementation details.
- Removed private underscored helpers and setup-state classes from literal
  external adapter `__all__` exports.
- Updated external runtime call sites so adapter modules no longer reach through
  another module's private helper namespace for JAXGCM field mapping,
  CAMulator tensor/field staging, CAMulator optional-dependency loading, or
  Veros host-state mutation helpers.
- Strengthened architecture coverage so external adapter `__all__` exports
  cannot drift back to private names and runtime helpers use the named
  helper boundary.
- Updated `DESIGN.md` and `DEPENDENCIES.md` for the explicit external-adapter
  helper boundary.
- Validation run for this change:
  baseline `conda run -n scipy pytest tests/ -q --fast --tb=short`, focused red
  pytest for the missing public helper names/private `__all__` exports, focused
  green after implementation,
  `conda run -n scipy black vercor examples tests`,
  `conda run -n scipy pytest tests/test_api_boundaries.py tests/test_external_components_coverage.py tests/test_camulator_component_kernels.py -q --fast --tb=short`,
  `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`,
  `conda run -n scipy mypy vercor examples tests`,
  `conda run -n scipy pytest tests/ -q --fast --tb=short`, and
  `conda run -n scipy pytest tests/ -q --tb=short`. The existing Black Python
  3.13/target-3.14 warning and JAX dtype-promotion warning remain.
- Failed approach recorded: the first focused red run imported renamed
  CAMulator helpers directly from the module and failed at collection time.
  The test was adjusted to access helpers through the module object so the red
  phase exercised the intended missing boundary attributes.

### 2026-06-01: Component Lifecycle Boundary Refactor

- Added a typed private lifecycle-owner boundary in
  `vercor.components._lifecycle` and narrowed component authoring protocols so
  lifecycle storage is no longer exposed as `Any`.
- Grouped callable factory lifecycle callbacks into one
  `ComponentLifecycleHooks` container inside callable construction metadata.
- Centralized runtime-payload hook precedence in
  `ComponentLifecycleMixin`; callable wrappers now provide only the default
  callable payload fallback when no custom payload hook is installed.
- Strengthened boundary and behavior coverage for typed lifecycle ownership,
  callable-wrapper hook dispatch, callable payload preservation, and custom
  payload-hook override behavior.
- Updated `DESIGN.md` and `DEPENDENCIES.md` for the lifecycle-owner/callable
  lifecycle boundary.
- Validation run for this change:
  focused red
  `conda run -n scipy pytest tests/test_component_boundary_contracts.py tests/test_component_base_coverage.py -q --fast --tb=short`,
  focused green after implementation,
  `conda run -n scipy black vercor examples tests`,
  `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`,
  `conda run -n scipy mypy vercor examples tests`,
  `conda run -n scipy pytest tests/ -q --fast --tb=short`, and
  `conda run -n scipy pytest tests/ -q --tb=short`. The existing Black Python
  3.13/target-3.14 warning and JAX dtype-promotion warning remain.
- Failed approach recorded: no failed implementation approach was encountered;
  the only failure was the intentional focused red test run before adding the
  typed lifecycle-owner boundary and grouped callable hook metadata.

### 2026-06-01: Component Execution Protocol Boundary Refactor

- Added private structural execution protocols in `vercor.components._protocols`
  so component host/scanned execution policy no longer imports concrete
  `Component` or `HostRuntimeComponent` classes.
- Updated `vercor.components.runtime_execution` to detect host-backed runtime
  components through `HostRuntimeExecutionProtocol` while preserving the
  existing public helpers and host-runtime error behavior.
- Narrowed runtime-only context imports in component modules to type-checking
  imports where they are only annotation support.
- Strengthened architecture coverage so runtime execution must use private
  protocols and cannot drift back to concrete component-class imports.
- Updated `DESIGN.md` and `DEPENDENCIES.md` for protocol-backed execution
  policy ownership.
- Validation run for this change:
  baseline `conda run -n scipy pytest tests/ -q --fast --tb=short`, focused red
  pytest for the missing execution protocols and concrete-class runtime
  execution import, focused green boundary pytest after implementation,
  `conda run -n scipy black vercor examples tests`,
  `conda run -n scipy pytest tests/test_component_boundary_contracts.py tests/test_component_base_coverage.py tests/test_api_boundaries.py tests/test_runtime_state.py -q --fast --tb=short`,
  `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`,
  `conda run -n scipy mypy vercor examples tests`,
  `conda run -n scipy pytest tests/ -q --fast --tb=short`, and
  `conda run -n scipy pytest tests/ -q --tb=short`. The existing Black Python
  3.13/target-3.14 warning and JAX dtype-promotion warning remain.
- Failed approach recorded: no failed implementation approach was encountered;
  the only failure was the intentional focused red test run before adding the
  private execution protocols.

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

- Added `vercor.runtime.topology_state` as the neutral owner for grouped
  `RuntimeTopologyMaps`, `SurfaceExchangeMasks`, and `ExchangeTopologyState`.
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
  existing source-boundary assertion for the former topology regridder import;
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
