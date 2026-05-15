# 2026-05-15

## Conservative Compatibility Cleanup

- Collapsed `setups.data.forcing.read_forcing` to a direct re-export of the
  canonical `vercor.forcing_data.read_forcing` while preserving the setup import
  path.
- Inlined the single-use private NetCDF output helper into the public
  `write_runtime_component_view_to_netcdf(...)` function.
- Removed single-use Coupler topology delegate methods by calling the
  runtime-owned topology helpers directly from `Coupler.initialize()`.
- Removed obsolete TODO/commented print blocks from the conservative rectilinear
  regridder edge-derivation path.
- Preserved documented compatibility surfaces: settings attribute access,
  `ComponentSettings`, component author facades/context aliases,
  `vercor.runtime.components`, setup lazy exports, `ComponentForcingData`,
  `Coupler._run_scanned_runtime()`, and `_runtime_state_from_components()`.
- Updated source-boundary tests so the removed private output helper, collapsed
  forcing wrapper, and removed Coupler topology delegates stay removed.

## Validation (Conservative Compatibility Cleanup, 2026-05-15)

- `conda run -n scipy pytest tests/test_api_boundaries.py::test_setup_forcing_reader_reexports_canonical_read_boundary tests/test_runtime_state.py::test_runtime_module_does_not_own_component_specific_steps -q --tb=short`
  - failed before implementation on the remaining setup forcing wrapper and
    private Coupler topology delegates
  - passed after implementation
- `conda run -n scipy pytest tests/test_component_base_coverage.py tests/test_runtime_state.py tests/test_coupler_coverage.py -q --tb=short`
  - first failed after removing the Coupler delegates because stale coverage
    still patched/called the removed private methods
  - passed after moving that coverage to the runtime topology boundary and the
    inlined `Coupler.initialize()` call site
- `conda run -n scipy black vercor setups tests`
  - first reformatted `tests/test_coupler_coverage.py`
  - final run left all 131 files unchanged
  - note: Black emitted the existing Python 3.13 vs target-3.14 safety-check
    warning
- `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`
  - passed (`0`)
- `conda run -n scipy mypy vercor setups tests`
  - first reported the test logger double did not satisfy `LoggerLike`
  - passed after using the real configured logger in topology tests
- `conda run -n scipy pytest tests/ -q --fast --tb=short`
  - passed
- `conda run -n scipy pytest tests/ -q --tb=short`
  - passed
  - note: JAX emitted the existing dtype promotion `FutureWarning` in the
    JAXGCM runtime gradient test

## Private Runtime Helper Consolidation

- Added private `Coupler._prepare_runtime_state(...)` so `run()` and
  `_run_scanned_runtime()` share the same runtime-state creation/reuse and
  optional validation transition.
- Split runtime exchange dispatch into private scalar and vector field
  primitives while preserving the existing behavior: scalar exchanges apply
  fractional masks, vector exchanges do not.
- Added boundary coverage that locks the private helper structure without
  changing public APIs or exported names.
- Deferred high-risk audit findings intentionally: `Component` /
  `HostRuntimeComponent` inheritance changes, host/scanned runner unification,
  and callable-wrapper architecture changes.
- Implementation commit: `15fc761` (`Consolidate private runtime maintenance helpers`).

## Validation (Private Runtime Helper Consolidation, 2026-05-15)

- `conda run -n scipy pytest tests/test_runtime_exchange.py::test_exchange_dispatch_uses_scalar_and_vector_primitives tests/test_runtime_state.py::test_runtime_module_does_not_own_component_specific_steps -q --tb=short`
  - failed before implementation on missing `_dispatch_scalar_exchange_field`,
    `_dispatch_vector_exchange_field`, and `_prepare_runtime_state`
  - passed after implementation
- `conda run -n scipy pytest tests/test_runtime_exchange.py tests/test_runtime_state.py tests/test_api_boundaries.py -q --tb=short`
  - passed
- `conda run -n scipy black vercor setups tests`
  - first reformatted `vercor/runtime/exchange_dispatch.py`
  - final run left all 131 files unchanged
  - note: Black emitted the existing Python 3.13 vs target-3.14 safety-check
    warning
- `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`
  - passed (`0`)
- `conda run -n scipy mypy vercor setups tests`
  - first reported the extracted exchange helper update mapping needed an
    explicit type annotation
  - passed after annotating it (`131 source files`)
- `conda run -n scipy pytest tests/ -q --fast --tb=short`
  - passed
- `conda run -n scipy pytest tests/ -q --tb=short`
  - passed
  - note: JAX emitted the existing dtype promotion `FutureWarning` in the
    JAXGCM runtime gradient test

## Runtime/Setup Helper Boundary Cleanup

- Added `CamulatorRuntimeCursor` so CAMulator atmosphere and land adapters share
  forcing-index initialization, counter reset, index lookup, and counter
  advancement through one helper instead of duplicating state transitions.
- Moved generic component step-result application into
  `vercor.components._runtime_fields.apply_step_result(...)`; callable wrappers
  now use the same runtime-field primitive as `Component.apply_step_result(...)`.
- Removed the test-only JCM coordinate wrapper and pointed tests at the real
  `_jcm_coordinates_in_degrees(...)` helper.

## Validation (Runtime/Setup Helper Boundary Cleanup, 2026-05-15)

- `conda run -n scipy pytest tests/test_setup_lifecycle_helpers.py::test_camulator_runtime_cursor_initializes_indexes_and_advances tests/test_api_boundaries.py::test_component_base_internals_are_private_modules tests/test_api_boundaries.py::test_camulator_adapters_share_runtime_cursor_state_transition_helper tests/test_api_boundaries.py::test_jcm_land_uses_single_coordinate_conversion_helper tests/test_data_component_kernels.py::test_jcm_land_coordinate_helper_supports_jit -q --tb=short`
  - failed before implementation on missing `CamulatorRuntimeCursor`, missing
    runtime-field `apply_step_result`, remaining direct CAMulator cursor state,
    and the leftover JCM coordinate wrapper
- `conda run -n scipy pytest tests/test_camulator_component_kernels.py::test_camulator_step_uses_jax_prepared_forcing_boundaries -q --tb=short`
  - failed before the cursor semantic fix because the CAMulator atmosphere
    cursor advanced once per model substep instead of once per coupling step
    (`2 == 1`)
  - passed after advancing the cursor once after each non-empty forcing block
- `conda run -n scipy pytest tests/test_setup_lifecycle_helpers.py::test_camulator_runtime_cursor_initializes_indexes_and_advances tests/test_api_boundaries.py::test_component_base_internals_are_private_modules tests/test_api_boundaries.py::test_camulator_adapters_share_runtime_cursor_state_transition_helper tests/test_api_boundaries.py::test_jcm_land_uses_single_coordinate_conversion_helper tests/test_data_component_kernels.py::test_jcm_land_coordinate_helper_supports_jit tests/test_component_base_coverage.py::test_apply_step_result_updates_fields_and_payload tests/test_camulator_component_kernels.py::test_camulator_step_uses_jax_prepared_forcing_boundaries -q --tb=short`
  - passed after implementation
- `conda run -n scipy pytest tests/test_setup_lifecycle_helpers.py tests/test_api_boundaries.py tests/test_data_component_kernels.py tests/test_component_base_coverage.py tests/test_camulator_component_kernels.py -q --tb=short`
  - passed
- `conda run -n scipy black vercor setups tests`
  - first reformatted `setups/data/camulator_land.py`
  - final run left all 131 files unchanged
  - note: Black emitted the existing Python 3.13 vs target-3.14 safety-check
    warning
- `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`
  - passed (`0`)
- `conda run -n scipy mypy vercor setups tests`
  - passed (`131 source files`)
- `conda run -n scipy pytest tests/ -q --fast --tb=short`
  - passed
- `conda run -n scipy pytest tests/ -q --tb=short`
  - passed
  - note: JAX emitted the existing dtype promotion `FutureWarning` in the
    JAXGCM runtime gradient test

## Maintainability Audit Follow-Up Consolidation

- Added `setups.jcm_setup_helpers.build_jcm_land_atmosphere_components(...)`
  for the repeated JCM setup path that generates JCM inputs, builds JCM land,
  patches `terrain.fmask` from the land mask, and constructs JAXGCM with the
  caller's explicit options.
- Refactored CAMulator `StateVariableAccessor` index-map construction through
  shared private primitives for available indexed variables and unavailable
  configured variables, preserving state/input/output tensor contracts.
- Routed remaining multi-exchange runnable setup scripts through
  `setups.coupler_helpers.add_exchanges(...)` while keeping exchange recipes
  explicit in each script.
- Corrected the `align_model_timestep(...)` non-divisible error text so it
  states that the model timestep must evenly divide the coupling timestep.
- Updated `DEPENDENCIES.md` for the new JCM setup helper.
- Deferred intentionally high-risk items from the audit: JAXGCM mirrored
  runtime/setup state, host/scanned runner unification, and component
  inheritance changes.

## Validation (Maintainability Audit Follow-Up Consolidation, 2026-05-15)

- `conda run -n scipy pytest tests/ -v --fast 2>&1 | tail -20`
  - passed baseline before implementation (`123 passed, 286 deselected`)
- `conda run -n scipy pytest tests/test_setup_lifecycle_helpers.py::test_align_model_timestep_rejects_non_divisible_model_step tests/test_setup_lifecycle_helpers.py::test_build_jcm_land_atmosphere_components_patches_mask_and_options -q --tb=short`
  - failed before implementation on the old timestep wording and missing
    `setups.jcm_setup_helpers`
  - passed after implementing the helper and correcting the error text
- `conda run -n scipy pytest tests/test_camulator_component_kernels.py::test_state_variable_accessor_builds_exact_index_maps tests/test_camulator_component_kernels.py::test_state_variable_accessor_uses_shared_index_map_builders -q --tb=short`
  - failed before implementation on the missing shared index-map helpers
  - passed after refactoring `StateVariableAccessor`
- `conda run -n scipy pytest tests/test_api_boundaries.py::test_multi_exchange_setup_scripts_use_shared_add_exchanges_helper -q --tb=short`
  - failed before implementation on direct `cpl.add_exchange(...)` usage in
    runnable setup scripts
  - passed after routing multi-exchange scripts through `add_exchanges(...)`
- `conda run -n scipy pytest tests/test_setup_lifecycle_helpers.py -q --tb=short`
  - passed after implementation
- `conda run -n scipy pytest tests/test_camulator_component_kernels.py -q --tb=short`
  - passed after implementation
- `conda run -n scipy pytest tests/test_api_boundaries.py -q --tb=short`
  - passed after implementation
- `conda run -n scipy black vercor setups tests`
  - first reformatted `tests/test_setup_lifecycle_helpers.py`
  - final run left all 131 files unchanged
  - note: Black emitted the existing Python 3.13 vs target-3.14 safety-check
    warning
- `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`
  - passed (`0`)
- `conda run -n scipy mypy vercor setups tests`
  - first reported the JCM helper's optional binary-mask type and one test
    double annotation issue
  - passed after adding an explicit mask guard and widening the test double
    annotation (`131 source files`)
- `conda run -n scipy pytest tests/ -q --fast --tb=short`
  - passed
- `conda run -n scipy pytest tests/ -q --tb=short`
  - passed
  - note: JAX emitted the existing dtype promotion `FutureWarning` in the
    JAXGCM runtime gradient test

# 2026-05-14

## JAXGCM Test-Only Compatibility Surface Removal

- Removed the factory-time compatibility attachments from `make_jax_gcm`;
  public JAXGCM components no longer receive ad-hoc `model`, `sigma_levels`,
  or `_setup_state` attributes for tests.
- Reworked JAXGCM runtime tests to keep private setup internals in an explicit
  local fixture instead of attaching them to the public component object.
- Added an API boundary regression that forbids the removed factory block and
  test `_setup_state` access from returning.
- Removed the constructor coverage assertion that required `sigma_levels` on
  the public component object.

## Validation (JAXGCM Test-Only Compatibility Surface Removal, 2026-05-14)

- `conda run -n scipy pytest tests/test_api_boundaries.py::test_jax_gcm_factory_does_not_attach_test_only_setup_state -q --tb=short`
  - failed before implementation on the existing `component_any` compatibility
    block
  - passed after removing the block and updating runtime tests
- `conda run -n scipy pytest tests/test_coupler_runtime.py::test_jax_gcm_runs_inside_runtime_under_jit_and_grad tests/test_coupler_runtime.py::test_jax_gcm_runtime_keeps_time_dependent_forcing_payload_shape_stable tests/test_coupler_runtime.py::test_jax_gcm_runtime_requires_initialized_payload -q --tb=short`
  - passed after moving setup internals into the test fixture
  - note: JAX emitted the existing dtype promotion `FutureWarning` in the
    JAXGCM runtime gradient test
- `conda run -n scipy pytest tests/test_external_components_coverage.py::test_jax_gcm_constructor_builds_jax_backed_grid -q --tb=short`
  - passed after removing the public `sigma_levels` assertion
- `conda run -n scipy black vercor setups tests`
  - reformatted `tests/test_coupler_runtime.py`
  - note: Black emitted the existing Python 3.13 vs target-3.14 safety-check
    warning
- `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`
  - passed (`0`)
- `conda run -n scipy mypy vercor setups tests`
  - passed (`130 source files`)
- `conda run -n scipy pytest tests/ -q --fast --tb=short`
  - passed
- `conda run -n scipy pytest tests/ -q --tb=short`
  - passed
  - note: JAX emitted the existing dtype promotion `FutureWarning` in the
    JAXGCM runtime gradient test

## Maintainability Audit Refactor Implementation

- Consolidated NetCDF forcing reads behind `vercor.forcing_data.read_forcing`;
  `setups.data.forcing.read_forcing` and `ComponentForcingData._read_forcing`
  now share the same implementation while preserving compatibility error text.
- Added explicit `Component.setup_metadata` for setup-only metadata. Time
  interpolated data components store `DATA_FILES` there, and ERA5 atmosphere
  stores hybrid coefficients there instead of attaching ad-hoc attributes to
  factory-created components.
- Extended setup lifecycle helpers with common timestep assignment, spinup
  logging, forcing-index calculation, and default-field seeding primitives.
  JAXGCM, Veros, CAMulator atmosphere, and CAMulator land use the helpers where
  behavior is identical.
- Added `setups.coupler_helpers` and routed runnable setup component
  registration/run-sequence setup through `build_coupler(...)`; exchange
  recipes remain explicit in each script.

## Validation (Maintainability Audit Refactor Implementation, 2026-05-14)

- `conda run -n scipy pytest tests/test_component_base_coverage.py::test_component_forcing_data_read_and_runtime_write_round_trip tests/test_component_models_coverage.py::test_era5_land_constructor_uses_masked_grid_and_enables_interpolation tests/test_component_models_coverage.py::test_era5_ocean_constructor_applies_land_mask_and_reverses_latitude tests/test_component_models_coverage.py::test_era5_atmosphere_constructor_initialize_and_step tests/test_setup_lifecycle_helpers.py tests/test_api_boundaries.py::test_setup_components_use_explicit_metadata_mapping tests/test_api_boundaries.py::test_setup_coupler_helpers_register_components_and_add_exchanges -q --tb=short`
  - failed before implementation on missing lifecycle helper imports
  - passed after implementing shared readers, metadata, lifecycle helpers, and
    coupler setup helpers
- `conda run -n scipy pytest tests/test_component_base_coverage.py tests/test_data_component_kernels.py tests/test_setup_lifecycle_helpers.py tests/test_api_boundaries.py -q --tb=short`
  - passed after implementation
- `conda run -n scipy pytest tests/ -q --fast --tb=short`
  - passed after implementation
- `conda run -n scipy black vercor setups tests`
  - first reformatted `setups/external/jax_gcm.py`,
    `setups/external/veros_gcm.py`, and
    `tests/test_component_models_coverage.py`
  - final run left all 130 files unchanged
  - note: Black emitted the existing Python 3.13 vs target-3.14 safety-check
    warning
- `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`
  - passed (`0`)
- `conda run -n scipy mypy vercor setups tests`
  - first reported missing setup-state attribute annotations plus two
    test-only typing issues after introducing shared lifecycle helpers
  - passed after adding explicit annotations and tightening tests
    (`130 source files`)
- `conda run -n scipy pytest tests/ -q --tb=short`
  - passed after implementation
  - note: JAX emitted the existing dtype promotion `FutureWarning` in the
    JAXGCM runtime gradient test

## Setup Lifecycle Helper Consolidation

- Added `setups._time_helpers.align_model_timestep(...)` as the shared
  setup-time validation path for coupling/model timestep divisibility in
  JAXGCM, Veros, CAMulator atmosphere, and CAMulator land adapters.
- Added `setups.external.camulator_state.initialize_camulator_forcing_cursor(...)`
  so CAMulator atmosphere and CAMulator land share forcing index lookup,
  initialization timestamp formatting, and start-time mismatch warnings without
  importing the full CAMulator adapter from the land forcing module.
- Removed private setup-state borrowing of `Component` methods from
  `_JAXGCMState` and `_VerosGCMState`; tests now pass real component objects
  into lifecycle hooks to match the production factory path.
- Added `setups.exchange_recipes` constants for repeated runnable setup exchange
  field recipes while leaving each script's orchestration explicit.
- Added `setups.data._component_helpers.time_interpolated_data_component(...)`
  for ERA5/ERA-Interim time-interpolated data adapters, preserving
  adapter-specific field preparation in each module.
- Updated `DEPENDENCIES.md` for the new helper modules.
- Failed approaches / corrections:
  - The first lifecycle-helper test run failed at import because the planned
    helper module did not exist yet; after adding the helper, the boundary test
    correctly failed on remaining private `Component` method borrowing.
  - The first full-suite run exposed an over-broad source-boundary assertion:
    `initialize_camulator_forcing_cursor` intentionally contains the old
    substring. The assertion now forbids the heavy `initialize_camulator(` call
    shape instead.

## Validation (Setup Lifecycle Helper Consolidation, 2026-05-14)

- `conda run -n scipy pytest tests/test_setup_lifecycle_helpers.py tests/test_api_boundaries.py::test_private_setup_state_objects_do_not_borrow_component_methods -q --tb=short`
  - failed as expected before implementation on missing helper/boundary paths
  - passed after adding helpers and removing private method borrowing
- `conda run -n scipy pytest tests/test_setup_lifecycle_helpers.py tests/test_api_boundaries.py tests/test_data_component_kernels.py tests/test_external_components_coverage.py::test_jax_gcm_initialize_validates_timestep_multiple tests/test_external_components_coverage.py::test_jax_gcm_initialize_uses_provided_forcing_and_can_spin_up tests/test_external_components_coverage.py::test_jax_gcm_initialize_builds_default_forcing_when_missing tests/test_external_components_coverage.py::test_jax_gcm_step_maps_outputs_and_respects_output_gate tests/test_external_components_coverage.py::test_veros_initialize_validates_timestep_multiple tests/test_external_components_coverage.py::test_veros_initialize_can_spin_up_and_extract_surface_temperature tests/test_external_components_coverage.py::test_veros_step_sets_forcing_fields_and_refreshes_sst tests/test_external_components_coverage.py::test_veros_step_nan_cleans_forcing_fields_before_set_variable tests/test_camulator_component_kernels.py::test_camulator_land_stores_jax_runtime_arrays tests/test_camulator_component_kernels.py::test_camulator_step_uses_jax_prepared_forcing_boundaries -q --tb=short`
  - passed after implementation
- `conda run -n scipy black vercor setups tests`
  - passed
  - note: Black emitted the existing Python 3.13 vs target-3.14 safety-check
    warning; final run left all 129 files unchanged
- `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`
  - first reported one unused import in the new helper test
  - passed after removing it (`0`)
- `conda run -n scipy mypy vercor setups tests`
  - first reported the CAMulator cursor helper's start-time parameter was typed
    too narrowly for VerCOR model-date clocks
  - passed after widening that parameter (`129 source files`)
- `conda run -n scipy pytest tests/ -q --fast --tb=short`
  - passed after implementation
- `conda run -n scipy pytest tests/ -q --tb=short`
  - first failed on the over-broad CAMulator land boundary assertion noted
    above
  - passed after narrowing the assertion
  - note: JAX emitted the existing dtype promotion `FutureWarning` in the
    JAXGCM runtime gradient test

## Conservative Architectural Redundancy Cleanup

- Added `setups.data._field_helpers.mask_time_last_surface_field(...)` as the
  single canonical data-adapter helper for masking time-last surface fields.
  ERA5 and ERA-Interim ocean setup adapters now share this path instead of
  carrying duplicate SST masking helpers.
- Routed setup host-array transfer through the canonical `vercor.host_arrays`
  boundary and removed the duplicate `setups.jax_array_helpers.to_host_array`
  helper while keeping setup-specific transpose/vector-speed helpers.
- Removed the no-op `fill_value` parameter from the conservative rectilinear
  regridder class and factory; bilinear interpolation keeps its real fill-value
  API.
- Removed component-like compatibility methods from private external setup
  state objects: `_JAXGCMState.step_runtime_state`,
  `_VerosGCMState.step_host_runtime_state`, and
  `_CAMulatorGCMState.step_host_runtime_state`.
- Narrowed private JAXGCM/Veros setup hooks to the production factory hook
  signatures rather than supporting direct test-only calls.
- Preserved intentional compatibility surfaces: `vercor.runtime.components`
  reexports, public component author facades, context aliases, and setup lazy
  imports.
- Failed approaches / corrections:
  - The first shared masked-field helper reused the old multiply-by-NaN masking
    pattern; the new gradient regression exposed NaN gradients on masked cells.
    Switching to `jnp.where(mask, field, jnp.nan)` preserved masked values while
    keeping valid-cell gradients finite.
  - Direct private-state runtime-state creation in a JAXGCM unit test depended
    on the removed dual-signature hook; the test now calls the production hook
    signature and constructs the runtime state explicitly.

## Validation (Conservative Architectural Redundancy Cleanup, 2026-05-14)

- `conda run -n scipy pytest tests/ -q --fast --tb=short`
  - passed baseline before implementation
- `conda run -n scipy pytest tests/test_data_component_kernels.py tests/test_example_jax_helpers.py tests/test_conservative_rectilinear_regridder.py -q --tb=short`
  - failed as expected before implementation on missing shared field helper,
    remaining duplicate host-transfer helper, and no-op conservative
    `fill_value` API
  - first implementation run found NaN gradients from the old masking pattern;
    passed after switching the shared helper to `jnp.where`
- `conda run -n scipy pytest tests/test_runtime_state.py::test_runtime_module_does_not_own_component_specific_steps tests/test_external_components_coverage.py::test_jax_gcm_initialize_validates_timestep_multiple tests/test_external_components_coverage.py::test_jax_gcm_initialize_uses_provided_forcing_and_can_spin_up tests/test_external_components_coverage.py::test_jax_gcm_initialize_builds_default_forcing_when_missing tests/test_external_components_coverage.py::test_jax_gcm_step_maps_outputs_and_respects_output_gate tests/test_external_components_coverage.py::test_veros_initialize_validates_timestep_multiple tests/test_external_components_coverage.py::test_veros_initialize_can_spin_up_and_extract_surface_temperature tests/test_external_components_coverage.py::test_veros_step_sets_forcing_fields_and_refreshes_sst tests/test_external_components_coverage.py::test_veros_step_nan_cleans_forcing_fields_before_set_variable tests/test_camulator_component_kernels.py::test_camulator_step_uses_jax_prepared_forcing_boundaries -q --tb=short`
  - failed as expected before removing the private state-level step wrappers
  - passed after moving tests to production hook/callable paths and removing the
    wrappers
- `conda run -n scipy pytest tests/test_data_component_kernels.py tests/test_example_jax_helpers.py tests/test_conservative_rectilinear_regridder.py tests/test_api_boundaries.py tests/test_runtime_state.py::test_runtime_module_does_not_own_component_specific_steps tests/test_external_components_coverage.py::test_jax_gcm_step_maps_outputs_and_respects_output_gate tests/test_external_components_coverage.py::test_veros_step_sets_forcing_fields_and_refreshes_sst tests/test_external_components_coverage.py::test_veros_step_nan_cleans_forcing_fields_before_set_variable tests/test_camulator_component_kernels.py::test_camulator_step_uses_jax_prepared_forcing_boundaries -q --tb=short`
  - passed after implementation
- `conda run -n scipy black vercor setups tests`
  - passed
  - note: Black emitted the existing Python 3.13 vs target-3.14 safety-check
    warning and left all 125 files unchanged
- `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`
  - first reported one stale unused import in
    `tests/test_external_components_coverage.py`
  - passed after removing it (`0`)
- `conda run -n scipy mypy vercor setups tests`
  - first reported missing annotations in the updated helper test plus
    test-only type errors after narrowing private setup hook signatures
  - passed after adding test annotations and explicit test-only casts
    (`125 source files`)
- `conda run -n scipy pytest tests/ -q --fast --tb=short`
  - passed after implementation
- `conda run -n scipy pytest tests/ -q --tb=short`
  - passed after implementation
  - note: JAX emitted the existing dtype promotion `FutureWarning` in the
    JAXGCM runtime gradient test

## Lazy Optional Setup Adapter Imports

- Added a shared `setups._lazy_imports` helper and moved `setups.data` /
  `setups.external` package exports behind PEP 562 `__getattr__` lazy loading,
  so unrelated setup imports do not initialize optional CAMulator or Veros
  adapters.
- Deferred CREDIT output/core/postblock imports to CAMulator execution
  boundaries. Missing CREDIT now raises a clear `ImportError` only when
  CAMulator forcing/model/output functionality is actually used.
- Preserved package-level setup exports and module aliases such as
  `from setups.external import jax_gcm as jax_gcm_module`.
- Found an import-order issue in the local environment where importing Torch
  before xarray aborts on duplicate OpenMP runtime initialization; reordered
  `camulator_state` imports to keep the module importable without unsafe
  OpenMP environment overrides.
- Failed approaches / corrections:
  - The first regression run confirmed all unrelated setup imports emitted the
    reported CREDIT warnings and Veros startup messages.
  - After lazy package exports and deferred CREDIT imports, the direct
    `camulator_state` import exposed the Torch/xarray OpenMP abort; reordering
    those imports fixed the remaining subprocess failure.
  - The first flake8 pass exposed one unused import in the new lazy-import
    helper; removing it restored a clean lint run.

## Validation (Lazy Optional Setup Adapter Imports, 2026-05-14)

- `conda run -n scipy pytest tests/ -v --fast 2>&1 | tail -20`
  - passed baseline before implementation (`111 passed, 277 deselected`)
- `conda run -n scipy pytest tests/test_api_boundaries.py::test_unrelated_setup_imports_do_not_initialize_optional_adapters -q --tb=short`
  - failed as expected before implementation on CREDIT warning output from all
    covered imports
  - passed after lazy setup exports, deferred CREDIT imports, and the
    Torch/xarray import-order correction
- `conda run -n scipy pytest tests/test_api_boundaries.py tests/test_camulator_component_kernels.py -q --tb=short`
  - passed after implementation
- `conda run -n scipy black vercor setups tests`
  - passed
  - note: Black emitted the existing Python 3.13 vs target-3.14 safety-check
    warning and reformatted `setups/external/camulator.py` on the first run;
    the final run left all 124 files unchanged
- `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`
  - first reported one unused import in `setups/_lazy_imports.py`
  - passed after removing it (`0`)
- `conda run -n scipy mypy vercor setups tests`
  - passed (`124 source files`)
- `conda run -n scipy pytest tests/ -q --fast --tb=short`
  - passed
- `conda run -n scipy pytest tests/ -q --tb=short`
  - passed
  - note: JAX emitted the existing dtype promotion `FutureWarning` in the
    JAXGCM runtime gradient test

## Factory-Based Setup Components Refactor

- Added a top-level `setups` package and moved runnable setup scripts plus
  slab, data, and external concrete component definitions out of
  `vercor.components`.
- Converted slab, forcing-data, JAXGCM, Veros, CAMulator, and CAMulator-land
  definitions to snake_case factory functions backed by `data_component(...)`,
  `differentiable_component(...)`, and `host_component(...)`.
- Extended core component factories with optional lifecycle hooks for
  initialization, runtime payload creation, runtime-state prefill, and runtime
  validation, so concrete setup modules no longer subclass `Component`,
  `DataComponent`, or `HostRuntimeComponent`.
- Removed the old concrete `vercor/components/slab`, `vercor/components/data`,
  and `vercor/components/external` packages as primary/public implementation
  locations.
- Updated tests, runnable setup imports, coverage source, and dependency notes
  to use `setups`.
- Failed approaches / corrections:
  - Initial moved external adapters still relied on class-style state fixtures;
    private state objects now expose compatibility helpers only for targeted
    unit tests while public construction uses factories.
  - The first full-suite run exposed stale tests expecting old class aliases and
    `examples.*` imports; those tests now exercise factory components and
    `setups.*`.

## Validation (Factory-Based Setup Components Refactor, 2026-05-14)

- `conda run -n scipy pytest tests/test_api_boundaries.py::test_setup_factories_are_primary_concrete_component_api tests/test_api_boundaries.py::test_old_concrete_component_packages_are_removed tests/test_api_boundaries.py::test_setup_modules_do_not_subclass_component_contracts tests/test_api_boundaries.py::test_data_and_host_factories_return_core_contract_instances -q --tb=short`
  - failed as expected before implementation on missing `setups` package and
    old concrete component directories still existing
  - passed after moving setup modules and converting concrete components to
    factories
- `conda run -n scipy black vercor setups tests`
  - passed
  - note: Black emitted the existing Python 3.13 vs target-3.14 safety-check
    warning and left all 123 files unchanged on the final run
- `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`
  - passed (`0`)
- `conda run -n scipy mypy vercor setups tests`
  - passed (`123 source files`)
- `conda run -n scipy pytest tests/ -q --fast --tb=short`
  - passed
- `conda run -n scipy pytest tests/ -q --tb=short`
  - passed
  - note: JAX emitted the existing dtype promotion `FutureWarning` in the
    JAXGCM runtime gradient test

# 2026-05-13

## Canonical VerCOR Logging Format

- Centralized VerCOR logging format policy in `vercor.jax_logging` with a
  canonical owned handler that emits
  `VerCOR: YYYY-MM-DD HH:MM:SS [LEVEL]: message`.
- Replaced the previous `logging.basicConfig(...)` root-logger dependency with
  idempotent VerCOR logger configuration and `propagate=False` on the default
  `VerCOR` logger, preventing root logging setup from reformatting VerCOR
  records.
- Routed `VerCOR.*` child loggers through the canonical parent handler and made
  injected Python loggers passed to `Coupler` enter the same configuration
  boundary before JAX callback wrapping.
- Replaced direct root warnings in the CAMulator state import boundary with the
  default VerCOR logger.
- Updated logging tests to capture at the VerCOR logger boundary instead of
  relying on root `caplog` propagation.
- Failed approaches / corrections:
  - The red tests failed as expected because the default logger still
    propagated to root and had no canonical owned handler.
  - The first fast-suite rerun exposed remaining `caplog` assertions that no
    longer observe VerCOR records once `propagate=False` is intentional; those
    tests now use the shared logging capture helper.
  - The first shared capture helper used a normal temporary handler, which
    `get_default_logger()` correctly removed as noncanonical; marking the test
    handler as VerCOR-canonical keeps capture local without weakening the
    production handler cleanup.

## Validation (Canonical VerCOR Logging Format, 2026-05-13)

- `conda run -n scipy pytest tests/test_coupler_coverage.py::test_setup_logger_installs_canonical_owned_handler_format tests/test_coupler_coverage.py::test_setup_logger_routes_child_loggers_through_parent_canonical_handler tests/test_coupler_coverage.py::test_coupler_configures_injected_python_logger_with_canonical_boundary -q --tb=short`
  - failed as expected before implementation on root propagation and missing
    canonical owned handler
- `conda run -n scipy pytest tests/test_coupler_coverage.py::test_setup_logger_installs_canonical_owned_handler_format tests/test_coupler_coverage.py::test_setup_logger_routes_child_loggers_through_parent_canonical_handler tests/test_coupler_coverage.py::test_coupler_configures_injected_python_logger_with_canonical_boundary tests/test_coupler_coverage.py::test_coupler_wraps_injected_python_logger_for_scanned_runtime tests/test_coupler_coverage.py::test_setup_logger_formats_traced_values_under_scan tests/test_coupler_coverage.py::test_scanned_runtime_passes_callback_logger_to_components tests/test_coupler_coverage.py::test_scanned_runtime_logs_host_equivalent_progress_messages tests/test_coupler_coverage.py::test_scanned_runtime_suppresses_info_below_log_level -q --tb=short`
  - passed after implementation and test-capture updates
- `conda run -n scipy pytest tests/test_camulator_component_kernels.py::test_camulator_constructor_logs_save_forecast_path tests/test_tools_assets_and_regridding.py::test_check_remap_conservation_handles_skip_and_mismatch -q --tb=short`
  - passed after replacing root `caplog` expectations
- `conda run -n scipy pytest tests/test_external_components_coverage.py::test_jax_gcm_write_output_persists_mean_dataset -q --tb=short`
  - passed after replacing the final root `caplog` expectation
- `conda run -n scipy black vercor examples tests`
  - passed
  - note: Black emitted the existing Python 3.13 vs target-3.14 safety-check
    warning and left all 121 files unchanged on the final run
- `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`
  - passed (`0`)
- `conda run -n scipy mypy vercor examples tests`
  - passed (`121 source files`)
- `conda run -n scipy pytest tests/ -q --fast --tb=short`
  - passed
- `conda run -n scipy pytest tests/ -q --tb=short`
  - passed

# 2026-05-12

## Precision Policy Consistency Audit

- Added regression coverage for component field seeding, runtime import/export
  prefill, and coupler initialization when `enable_x64=False` while the test
  process has global JAX x64 enabled.
- Extended `vercor.dtypes` with copy-preserving real-array and real-valued
  `arange` helpers, keeping index arrays on canonical int32.
- Made `Coupler.initialize()` cascade the coupler precision policy to registered
  components and recast component-owned grid/data/default arrays before runtime
  state creation.
- Routed component author-field normalization and runtime contract prefill
  through settings-backed dtype helpers so missing/imported/exported fields do
  not silently inherit the process-global JAX dtype.
- Audited representative production array-creation sites in grids, forcing
  adapters, grid-mask helpers, JAXGCM setup/prefill, runtime scan indices, and
  runnable examples to use the central dtype helpers or same-dtype construction.
- Failed approaches / corrections:
  - The first runtime-prefill red test missed the `create_runtime_component_state`
    import and failed during test setup; adding the import exposed the intended
    float64-vs-float32 failure.
  - Full-field ERA-Interim assembly keeps explicit `dtype=core_field_array.dtype`
    because it must match the already-normalized source array, not a separate
    global/default policy.

## Validation (Precision Policy Consistency Audit, 2026-05-12)

- `conda run -n scipy pytest tests/ -v --fast 2>&1 | tail -20`
  - passed baseline before implementation (`105 passed, 273 deselected`)
- `conda run -n scipy pytest tests/test_component_base_coverage.py::test_seeded_component_arrays_follow_float32_policy_with_global_x64_enabled tests/test_runtime_state.py::test_runtime_contract_prefill_uses_component_float32_policy tests/test_coupler_runtime.py::test_coupler_initialize_cascades_float32_precision_to_component_arrays -q --tb=short`
  - failed as expected before implementation on component seed, runtime prefill,
    and grid arrays remaining `float64` under an explicit float32 policy
  - passed after policy propagation fixes
- `conda run -n scipy pytest tests/test_dtypes.py tests/test_component_base_coverage.py::test_seeded_component_arrays_follow_float32_policy_with_global_x64_enabled tests/test_runtime_state.py::test_runtime_contract_prefill_uses_component_float32_policy tests/test_coupler_runtime.py::test_coupler_initialize_cascades_float32_precision_to_component_arrays tests/test_component_models_coverage.py tests/test_external_components_coverage.py -q --fast --tb=short`
  - passed after the first production audit edits
- `conda run -n scipy pytest tests/test_dtypes.py tests/test_runtime_run_cache.py tests/test_runtime_state.py::test_runtime_contract_prefill_uses_component_float32_policy tests/test_coupler_runtime.py::test_coupler_initialize_cascades_float32_precision_to_component_arrays tests/test_component_models_coverage.py tests/test_external_components_coverage.py -q --fast --tb=short`
  - passed after scan-index and data-adapter follow-up edits
- `conda run -n scipy pytest tests/ -q --tb=short`
  - first full-suite run found a `JAXGCM.__new__` unit-test fixture without
    `settings`; `_generate_step_function()` now falls back to the global dtype
    policy for that lightweight fixture while using component settings in normal
    construction
- `conda run -n scipy pytest tests/test_external_components_coverage.py::test_generate_step_function_non_jitted_averages_predictions -q --tb=short`
  - passed after adding the JAXGCM settings fallback
- `conda run -n scipy black vercor examples tests`
  - passed
  - note: Black emitted the existing Python 3.13 vs target-3.14 safety-check
    warning and left all 121 files unchanged on the final run
- `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`
  - first reported one unused import after changing runtime scan indices
  - passed after removing it (`0`)
- `conda run -n scipy mypy vercor examples tests`
  - passed (`121 source files`)
- `conda run -n scipy pytest tests/ -q --fast --tb=short`
  - passed
- `conda run -n scipy pytest tests/ -q --tb=short`
  - passed

## Runtime Profiling and Core Dispatch Optimization

- Added `examples/profile_runtime.py`, a compact synthetic slab-coupler profiling
  harness for the pure JAX scanned runtime with configurable step count, grid
  size, log level, and optional runtime-state donation timing.
- Added derived name-to-index lookup caches to `RuntimeFieldStore` and
  `RuntimeCouplerState`, restoring those caches after PyTree unflattening so
  traced runtime lookups avoid repeated tuple scans while preserving the public
  immutable runtime API.
- Added `RuntimeFieldStore.set_many(...)` and routed receive, send, and exchange
  dispatch updates through one bulk store rebuild per phase instead of repeated
  single-field tuple reconstruction.
- Added destination-grouped exchange metadata to `RuntimeDispatchContext` so
  scanned component steps only iterate exchanges targeting the active
  destination component.
- Profiling note: `log_level="WARNING"` avoids INFO-level per-step JAX host
  callbacks for performance-sensitive runs while preserving the default
  `Coupler(log_level="INFO")` behavior.
- Failed approaches / corrections:
  - The first profile harness topology left `LND` without any exported exchange,
    and setup validation correctly rejected the run; adding `LND -> ATM`
    `soil_moisture` and `ICE -> OCN` `ice_fraction` exchanges made the
    synthetic topology match the component contracts.

## Validation (Runtime Profiling and Core Dispatch Optimization, 2026-05-12)

- `conda run -n scipy pytest tests/ -v --fast 2>&1 | tail -20`
  - passed baseline before implementation (`105 passed, 268 deselected`)
- `conda run -n scipy pytest tests/test_runtime_state.py::test_runtime_field_store_uses_index_cache_for_bulk_set_and_pytree_restore tests/test_runtime_state.py::test_runtime_coupler_state_restores_component_index_cache_after_pytree_roundtrip tests/test_runtime_exchange.py::test_runtime_dispatch_context_groups_exchanges_by_destination tests/test_runtime_run_cache.py::test_runtime_profile_harness_exposes_cli_entrypoint -q --tb=short`
  - failed as expected before implementation on missing lookup caches, grouped
    dispatch metadata, and profiling harness
  - passed after implementation
- `conda run -n scipy pytest tests/test_runtime_run_cache.py::test_runtime_profile_harness_runs_small_slab_profile -q --tb=short`
  - passed after correcting the profile harness exchange topology
- `conda run -n scipy python examples/profile_runtime.py --steps 10 --grid-nx 8 --grid-ny 4 --log-level WARNING --donate-state`
  - passed with `first_non_donating_s=0.091091`,
    `cached_non_donating_s=0.004868`, `first_donating_s=0.082407`,
    `compiled_cache_entries=2`, and `final_state_leaves=46`
- `conda run -n scipy pytest tests/test_runtime_state.py tests/test_runtime_exchange.py tests/test_runtime_run_cache.py tests/test_coupler_runtime.py -q --fast --tb=short`
  - passed
- `conda run -n scipy black vercor examples tests`
  - passed
  - note: Black emitted the existing Python 3.13 vs target-3.14 safety-check
    warning and reformatted `examples/profile_runtime.py`
- `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`
  - passed (`0`)
- `conda run -n scipy mypy vercor examples tests`
  - passed (`121 source files`)
- `conda run -n scipy pytest tests/ -q --fast --tb=short`
  - passed
- `conda run -n scipy pytest tests/ -q --tb=short`
  - passed

## Hypsometric Altitude Corrections

- Corrected sigma-level virtual temperature for specific humidity by removing
  the mixing-ratio-style denominator from the hypsometric thickness formula.
- Added shared JAX-native helpers for specific-humidity virtual temperature and
  hybrid-sigma full-level geometric altitude calculations.
- Corrected ECMWF/ERA-style hybrid-sigma altitude calculations when the top
  half-level pressure is zero by applying the documented top-layer
  `dlog_p = log(p_next / 0.1)` and `alpha = log(2)` treatment.
- Routed the CAMulator model-level-height diagnostic through the shared
  hybrid-sigma altitude helper to remove the duplicated formula path.
- Failed approaches / corrections:
  - The first red-test command used the wrong Camulator test node and failed
    during test collection; rerunning with the exact test name reached the
    intended sigma and hybrid failures.

## Validation (Hypsometric Altitude Corrections, 2026-05-12)

- `conda run -n scipy pytest tests/test_hypsometric.py::test_specific_humidity_uses_exact_virtual_temperature_without_mixing_ratio_denominator tests/test_fluxes_utilities.py::test_get_altitudes_hybrid_sigma_levels_handles_zero_top_half_level tests/test_camulator_component_kernels.py::test_map_camulator_prediction_arrays_supports_jit_and_preserves_conventions -q --tb=short`
  - failed as expected before implementation: moist sigma thickness was
    955.5581 m instead of 936.4468 m, and zero-top hybrid altitude returned
    `nan` at the top returned level.
- `conda run -n scipy pytest tests/test_hypsometric.py tests/test_fluxes_utilities.py::test_get_altitudes_hybrid_sigma_levels_returns_finite_increasing_profile tests/test_fluxes_utilities.py::test_get_altitudes_hybrid_sigma_levels_handles_zero_top_half_level tests/test_camulator_component_kernels.py::test_map_camulator_prediction_arrays_supports_jit_and_preserves_conventions -q --tb=short`
  - passed after implementation.
- `conda run -n scipy pytest tests/test_hypsometric.py tests/test_fluxes_utilities.py tests/test_camulator_component_kernels.py -q --tb=short`
  - passed.
- `conda run -n scipy pytest tests/ -q --fast --tb=short`
  - passed.
- `conda run -n scipy black vercor examples tests`
  - passed.
  - note: Black emitted the existing Python 3.13 vs target-3.14 safety-check
    warning and reformatted `tests/test_hypsometric.py`.
- `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`
  - passed (`0`).
- `conda run -n scipy mypy vercor examples tests`
  - passed (`120 source files`).
- `conda run -n scipy pytest tests/ -q --tb=short`
  - passed.

## Configured Regridder Factory Forwarding

- Extended the public `bilinear()` and `conservative()` factory helpers so
  they mirror and forward the existing rectilinear regridder constructor
  options through keyword-only parameters.
- Kept `Exchange.create(source_grid, destination_grid)` unchanged and added a
  private factory-name helper so plain functions, `functools.partial(...)`
  factories, and callable objects produce stable exchange names and runtime
  interpolation keys.
- Added regression coverage for bilinear option forwarding, conservative
  remapper option forwarding including source-mask behavior, and
  `Exchange(..., regridder_factory=partial(bilinear, ...))`.
- Failed approaches / corrections:
  - The red tests failed as expected before implementation because the public
    factories rejected forwarded keyword arguments and `Exchange` accessed
    `__name__` directly on a `functools.partial`.

## Validation (Configured Regridder Factory Forwarding, 2026-05-12)

- `conda run -n scipy pytest tests/test_bilinear_rectilinear_regridder.py::test_bilinear_factory_forwards_interpolator_options tests/test_conservative_rectilinear_regridder.py::test_conservative_factory_forwards_remapper_options tests/test_helpers_coverage.py::test_exchange_uses_wrapped_factory_name_and_create_keeps_partial_options -q --tb=short`
  - failed as expected before implementation on missing keyword forwarding and
    partial factory naming
  - passed after implementation
- `conda run -n scipy pytest tests/test_bilinear_rectilinear_regridder.py tests/test_conservative_rectilinear_regridder.py tests/test_helpers_coverage.py -q --fast --tb=short`
  - passed
- `conda run -n scipy pytest tests/ -q --fast --tb=short`
  - passed
- `conda run -n scipy black vercor examples tests`
  - passed
  - note: Black emitted the existing Python 3.13 vs target-3.14 safety-check
    warning and left all 120 files unchanged
- `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`
  - passed (`0`)
- `conda run -n scipy mypy vercor examples tests`
  - passed (`120 source files`)
- `conda run -n scipy pytest tests/ -q --tb=short`
  - passed

## Redundant `required_fields` Component API Removal

- Audited the component contract, runtime-field validation, concrete
  components, examples, and tests; `required_fields` had no unique behavior
  beyond `inputs`, which already means "must exist at runtime and is not
  component-prefilled."
- Removed the `required_fields` declaration parameter/attribute from
  `ComponentFieldSpec`, `Component.from_model()`,
  `HostRuntimeComponent.from_model()`, `differentiable_component()`,
  `host_component()`, `declare_fields()`, and private forwarding helpers.
- Kept `require_runtime_fields()` as the imperative validation helper, and
  renamed the private declaration union helper to
  `declared_runtime_field_names(...)` so declared `inputs`, `outputs`, and
  `default_fields` remain validated.
- Updated API-boundary and component coverage tests to assert stale callers now
  fail with normal unexpected-keyword `TypeError`, while missing `inputs`,
  output prefill, and default-field prefill behavior remain covered.
- Updated `DESIGN.md` so the declared runtime contract vocabulary is only
  `inputs`, `outputs`, and `default_fields`.
- Failed approaches / corrections:
  - The red tests failed as expected before implementation because public
    signatures still accepted the removed declaration keyword and
    `ComponentFieldSpec` still exposed the attribute.
  - The first type-check run after implementation found the new mixed callable
    test table needed an explicit type annotation; adding it kept the removed
    keyword test type-clean.

## Validation (Redundant `required_fields` Component API Removal, 2026-05-12)

- `conda run -n scipy pytest tests/test_component_base_coverage.py::test_required_fields_declaration_api_is_removed tests/test_component_base_coverage.py::test_seed_helpers_accept_scalar_author_values_and_expose_field_spec tests/test_api_boundaries.py::test_callable_author_api_does_not_expose_legacy_field_seed_keyword -q --tb=short`
  - failed as expected before implementation on the still-present
    `required_fields` API
  - passed after removing the API
- `conda run -n scipy pytest tests/test_component_base_coverage.py tests/test_api_boundaries.py -q --fast --tb=short`
  - passed
- `conda run -n scipy pytest tests/ -q --fast --tb=short`
  - passed
- `conda run -n scipy black vercor examples tests`
  - passed
  - note: Black emitted the existing Python 3.13 vs target-3.14 safety-check
    warning and left all 120 files unchanged
- `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`
  - passed (`0`)
- `conda run -n scipy mypy vercor examples tests`
  - first reported one missing test-variable annotation
  - passed after annotating the removed-keyword call table (`120 source files`)
- `conda run -n scipy pytest tests/ -q --tb=short`
  - passed

## Callable Field Seeding API Removal

- Removed the duplicate callable field-seeding keyword from the public
  `Component.from_model()`, `HostRuntimeComponent.from_model()`,
  `differentiable_component()`, and `host_component()` APIs.
- Kept `default_fields` as the single callable author-facing mechanism for
  concrete runtime field defaults; payload and settings are now keyword-only on
  callable factories so legacy positional calls fail clearly.
- Removed the private callable-wrapper setup-time seeding branch and routed
  callable runtime field creation through declared defaults, outputs, exchanges,
  or explicit subclass/manual seeding helpers.
- Updated the custom component wrapping example, component API tests, design
  docs, and historical progress wording to avoid the removed keyword.
- Failed approaches / corrections:
  - The red API tests failed as expected before implementation because callable
    factories still exposed the removed keyword and accepted it at runtime.
  - The first type-check run caught the intentionally invalid dynamic keyword
    test as a static argument-type issue; the test now casts that one call site
    so mypy does not treat it as a supported production call.

## Validation (Callable Field Seeding API Removal, 2026-05-12)

- `conda run -n scipy pytest tests/test_component_base_coverage.py tests/test_api_boundaries.py -q --fast --tb=short`
  - failed as expected before implementation on the still-present callable
    field-seeding keyword
  - passed after removing the API and updating tests
- `conda run -n scipy pytest tests/test_runtime_state.py tests/test_coupler_runtime.py -q --fast --tb=short`
  - passed
- `rg -n "initial"_"fields" vercor examples tests DESIGN.md DEPENDENCIES.md README.md PROGRESS.md`
  - passed with no matches
- `conda run -n scipy black vercor examples tests`
  - passed
  - note: Black emitted the existing Python 3.13 vs target-3.14 safety-check
    warning and left all 120 files unchanged
- `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`
  - passed (`0`)
- `conda run -n scipy mypy vercor examples tests`
  - first reported one static type issue in the dynamic removed-keyword test
  - passed after casting that intentionally invalid call site (`120 source files`)
- `conda run -n scipy pytest tests/ -q --fast --tb=short`
  - passed
- `conda run -n scipy pytest tests/ -q --tb=short`
  - passed

## Shared PyTree Mixin Refactor

- Added `vercor.pytree.PyTreeNodeMixin` as the shared declarative PyTree base
  for immutable JAX-registered containers.
- Replaced repeated `tree_flatten()` / `tree_unflatten()` implementations with
  explicit `pytree_children` and `pytree_aux_data` declarations in runtime
  state, runtime stores, runtime time metadata, `RectilinearGrid`, both
  rectilinear interpolators/remappers, and the JAXGCM runtime payload.
- Preserved constructor behavior by reconstructing through `object.__new__` and
  `object.__setattr__`; derived static remapper state and runtime coupler-state
  invariants are restored through `_pytree_post_unflatten()`.
- Added focused PyTree tests for inherited shared methods, array-only round
  trips, static metadata preservation, and derived remapper state restoration.
- Updated `DESIGN.md` and `DEPENDENCIES.md` to document the shared PyTree mixin
  convention and module dependency order.
- Failed approaches / corrections:
  - The focused red test failed as expected before implementation because
    `vercor.pytree` did not exist.
  - A first one-line Perl attempt to mechanically update dependency reference
    numbers had a syntax error; the reference rewrite was completed with a
    small mechanical script and then hand-adjusted for the new PyTree module.

## Validation (Shared PyTree Mixin Refactor, 2026-05-12)

- `conda run -n scipy pytest tests/test_pytree.py -q --tb=short`
  - failed as expected before implementation on missing `vercor.pytree`
  - passed after adding the mixin and refactoring registered classes
- `conda run -n scipy pytest tests/test_pytree.py tests/test_runtime_state.py tests/test_helpers_coverage.py tests/test_bilinear_rectilinear_interpolator.py tests/test_conservative_rectilinear_remapper.py tests/test_runtime_run_cache.py -q --tb=short`
  - passed
- `conda run -n scipy pytest tests/ -q --fast --tb=short`
  - passed
- `conda run -n scipy black vercor examples tests`
  - passed
  - note: Black emitted the existing Python 3.13 vs target-3.14 safety-check
    warning and reformatted `tests/test_pytree.py`
- `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`
  - passed (`0`)
- `conda run -n scipy mypy vercor examples tests`
  - passed (`120 source files`)
- `conda run -n scipy pytest tests/ -q --tb=short`
  - passed

# 2026-05-08

## Runtime Module Ownership Refactor

- Split broad runtime internals into focused modules:
  `contracts`, `stores`, `exchange_dispatch`, `component_state`,
  `field_transfer`, `validation`, and `topology`.
- Moved `RuntimeStepInfo` into `vercor.runtime.time` with the runtime time
  metadata builders.
- Kept the internal `vercor.runtime` re-export surface stable and kept
  `vercor.runtime.components` as a compatibility re-export shim while production
  imports use the focused modules.
- Moved exchange topology mask/regridder setup out of `Coupler`; the existing
  private `Coupler` methods now delegate to `vercor.runtime.topology`.
- Updated `DESIGN.md` and `DEPENDENCIES.md` to document the focused runtime
  module ownership.
- Failed approaches / corrections:
  - The focused red run failed as expected before implementation because
    `vercor/runtime/contracts.py` and the other focused runtime modules did not
    exist.

## Validation (Runtime Module Ownership Refactor, 2026-05-08)

- `conda run -n scipy pytest tests/ -v --fast 2>&1 | tail -20`
  - passed baseline before implementation (`101 passed, 259 deselected`)
- `conda run -n scipy pytest tests/test_runtime_state.py::test_runtime_module_does_not_own_component_specific_steps tests/test_runtime_state.py::test_runtime_focused_modules_keep_compatibility_reexports -q --tb=short`
  - failed as expected before implementation on missing focused runtime modules
  - passed after extracting focused modules and compatibility reexports
- `conda run -n scipy pytest tests/test_runtime_state.py tests/test_runtime_exchange.py tests/test_component_base_coverage.py tests/test_coupler_coverage.py -q --tb=short`
  - passed
- `conda run -n scipy black vercor examples tests`
  - passed
  - note: Black emitted the existing Python 3.13 vs target-3.14 safety-check
    warning and reformatted `tests/test_runtime_state.py` on the first run
- `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`
  - first reported one unused import in `vercor/runtime/topology.py`
  - passed after removing it (`0`)
- `conda run -n scipy mypy vercor examples tests`
  - first reported `get_component(...)` type mismatches in
    `vercor/runtime/topology.py`
  - passed after tightening the topology helper annotations (`118 source files`)
- `conda run -n scipy pytest tests/ -q --fast --tb=short`
  - passed
- `conda run -n scipy pytest tests/ -q --tb=short`
  - passed

## Coupler Runtime Adapter Refactor

- Moved runtime coupler-state construction, contract refresh, topology
  validation, dispatch-context creation, outgoing priming, and output-mask lookup
  into internal `vercor.runtime.coupler_state`.
- Moved host/scanned runtime execution loops, progress-message formatting,
  compiled-runtime cache-key creation, JIT wrapping, donation checks, and
  interrupt translation into internal `vercor.runtime.runner`.
- Kept the public `Coupler` facade stable while reducing private runtime
  methods to thin delegates for compatibility with existing internal tests.
- Updated source-boundary and monkeypatch-based tests so runtime adapter
  mechanics are asserted under `vercor.runtime` instead of `vercor.coupler`.
- Updated `DESIGN.md` and `DEPENDENCIES.md` to document the new runtime module
  ownership.
- Failed approaches / corrections:
  - The focused red run failed as expected before implementation because
    `vercor/runtime/coupler_state.py` and `vercor/runtime/runner.py` did not
    exist.

## Validation (Coupler Runtime Adapter Refactor, 2026-05-08)

- `conda run -n scipy pytest tests/test_runtime_state.py::test_runtime_module_does_not_own_component_specific_steps tests/test_coupler_coverage.py::test_host_and_scanned_run_use_runtime_component_helper -q --tb=short`
  - failed as expected before implementation on missing runtime modules
  - passed after extracting runtime state and runner helpers
- `conda run -n scipy pytest tests/test_runtime_state.py tests/test_coupler_coverage.py tests/test_coupler_runtime.py tests/test_runtime_run_cache.py tests/test_runtime_interrupts.py -q --tb=short`
  - passed
- `conda run -n scipy black vercor examples tests`
  - passed
  - note: Black emitted the existing Python 3.13 vs target-3.14 safety-check
    warning and reformatted `tests/test_coupler_coverage.py`
- `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`
  - passed (`0`)
- `conda run -n scipy mypy vercor examples tests`
  - passed (`111 source files`)
- `conda run -n scipy pytest tests/ -q --fast --tb=short`
  - passed
- `conda run -n scipy pytest tests/ -q --tb=short`
  - passed

## Component Runtime Field Adapter Extraction

- Moved the remaining component-facing runtime-field adapter bodies from
  `vercor.components.base` into private `vercor.components._runtime_fields`.
- Kept the public component helper methods and signatures stable while making
  `Component` delegate runtime-field mapping, reads, optional fallbacks,
  existing-field replacement, prefill, and declared-field validation through the
  private adapter module.
- Updated source-boundary tests to assert the adapter mechanics live outside
  `base.py` and remain unexported from `vercor.components`.
- Updated `DESIGN.md` and `DEPENDENCIES.md` to document the new private
  component runtime-field adapter module.
- Failed approaches / corrections:
  - The focused red run failed as expected because the updated tests looked for
    `vercor/components/_runtime_fields.py` before the module existed.

## Validation (Component Runtime Field Adapter Extraction, 2026-05-08)

- `conda run -n scipy pytest tests/test_api_boundaries.py::test_component_base_internals_are_private_modules tests/test_runtime_state.py::test_runtime_module_does_not_own_component_specific_steps tests/test_component_base_coverage.py::test_component_helpers_seed_and_update_runtime_fields -q --tb=short`
  - failed as expected before implementation on missing
    `vercor/components/_runtime_fields.py`
  - passed after extracting the adapter helpers and delegating from `Component`
- `conda run -n scipy pytest tests/test_component_base_coverage.py tests/test_api_boundaries.py tests/test_runtime_state.py -q --tb=short`
  - passed
- `conda run -n scipy black vercor examples tests`
  - passed
  - note: Black emitted the existing Python 3.13 vs target-3.14 safety-check
    warning and reformatted `vercor/components/_runtime_fields.py`
- `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`
  - passed (`0`)
- `conda run -n scipy mypy vercor examples tests`
  - passed (`109 source files`)
- `conda run -n scipy pytest tests/ -q --fast --tb=short`
  - passed
- `conda run -n scipy pytest tests/ -q --tb=short`
  - passed

## Time-Dependent Data Field Runtime Validation Fix

- Fixed the `examples/run_data_driver.py` runtime-state creation failure where
  `ERAInterimOcean` stored monthly `sea_surface_temperature` as
  `(nTime, nLat, nLon)` but `Component.validate_runtime_state()` required
  declared component data fields to be exactly `(nLat, nLon)`.
- Added runtime-owned canonical data-field validation for required component
  data while preserving strict grid-shape validation for incoming/outgoing
  exchange stores and component-specific grid-field checks.
- Updated `Component.require_runtime_fields()` to validate declared data through
  the canonical component-data layout contract, so time-dependent forcing cubes
  remain in component data and runtime exchange still sends selected 2-D slices.
- Added regression coverage for the public `data_component(...)` authoring path
  with monthly SST forcing and for direct helper validation of canonical
  time-dependent required fields.
- Failed approaches / corrections:
  - The first red helper test had an invalid `RuntimeComponentState` fixture
    missing incoming/outgoing stores; after fixing the fixture, both focused
    tests failed on the intended strict shape validator.

## Validation (Time-Dependent Data Field Runtime Validation Fix, 2026-05-08)

- `conda run -n scipy pytest tests/ -v --fast 2>&1 | tail -20`
  - passed baseline before implementation (`100 passed, 258 deselected`)
- `conda run -n scipy pytest tests/test_component_base_coverage.py::test_required_field_validator_accepts_time_dependent_canonical_data tests/test_coupler_runtime.py::test_public_data_component_monthly_output_validates_and_sends_runtime_slice -q --tb=short`
  - first exposed a test fixture construction error in the helper test
  - failed as expected after fixture correction on the strict `(nLat, nLon)`
    required-data validator
  - passed after routing required component data through canonical layout
    validation
- `conda run -n scipy pytest tests/test_component_base_coverage.py tests/test_coupler_runtime.py -q --tb=short`
  - passed
- `conda run -n scipy pytest tests/ -q --fast --tb=short`
  - passed
- `conda run -n scipy black vercor examples tests`
  - passed
  - note: Black emitted the existing Python 3.13 vs target-3.14 safety-check
    warning and reformatted the touched test/runtime files
- `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`
  - passed (`0`)
- `conda run -n scipy mypy vercor examples tests`
  - passed (`108 source files`)
- `conda run -n scipy pytest tests/ -q --tb=short`
  - passed

## Component Constructor Boilerplate Tightening

- Audited the post-refactor component authoring surface and preserved the
  documented public API while tightening the remaining constructor internals.
- Added a private `_author_field_spec(...)` helper so callable-backed
  differentiable and host constructors build field declarations through one
  path.
- Added a private `_callable_component_from_model(...)` helper so
  `Component.from_model()` and `HostRuntimeComponent.from_model()` delegate to
  the same callable-wrapper construction path.
- Simplified `DataComponent.from_fields()` so author field normalization happens
  once through `seed_fields(...)`, while seeded fields still become declared
  outputs.
- Updated component/API boundary tests to guard the shared constructor path and
  single-normalization behavior.
- Failed approaches / corrections:
  - The focused red run failed as expected on missing private constructor
    helpers and on `DataComponent.from_fields()` normalizing author fields
    twice.
  - The source-boundary guard was adjusted to count both constructor call sites
    through `_callable_component_from_model(...)` while allowing the host path
    to keep its return-type cast.

## Validation (Component Constructor Boilerplate Tightening, 2026-05-08)

- `conda run -n scipy pytest tests/ -v --fast 2>&1 | tail -20`
  - passed baseline before implementation (`99 passed, 258 deselected`)
- `conda run -n scipy pytest tests/test_component_base_coverage.py::test_data_component_from_fields_normalizes_author_fields_once tests/test_api_boundaries.py::test_component_base_internals_are_private_modules -q --tb=short`
  - failed as expected before implementation on missing helper functions and
    double normalization
  - passed after implementation
- `conda run -n scipy pytest tests/test_component_base_coverage.py tests/test_api_boundaries.py -q --tb=short`
  - passed
- `conda run -n scipy black vercor examples tests`
  - passed
  - note: Black emitted the existing Python 3.13 vs target-3.14 safety-check
    warning and left all 108 files unchanged
- `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`
  - passed (`0`)
- `conda run -n scipy mypy vercor examples tests`
  - passed (`108 source files`)
- `conda run -n scipy pytest tests/ -v --fast`
  - passed (`100 passed, 258 deselected`)
- `conda run -n scipy pytest tests/ -q --tb=short`
  - passed

## Component Runtime Boilerplate Refactor

- Moved reusable immutable runtime-field mechanics onto `RuntimeFieldStore`:
  membership checks, mapping roundtrip, fallback reads, and existing-field
  replacement helpers.
- Kept component-author helper names stable while making
  `runtime_fields()`, optional runtime-field reads, `with_runtime_fields()`,
  and `require_runtime_fields()` delegate to runtime-owned store/validation
  logic.
- Collapsed callable-backed differentiable and host wrapper setup through one
  private `_create_callable_component(...)` path, while preserving
  `data_component()`, `differentiable_component()`, `host_component()`,
  `DataComponent.from_fields()`, `Component.from_model()`, and
  `HostRuntimeComponent.from_model()`.
- Updated API-boundary, runtime-state, and component-base tests to assert
  runtime ownership of the helper mechanics and absence of public-looking
  callable `make_*` internals.
- Updated `DESIGN.md` and `DEPENDENCIES.md` to describe runtime ownership of
  field-store mechanics and thin component-facing adapters.
- Failed approaches / corrections:
  - The red focused runtime/helper run failed as expected on missing
    `RuntimeFieldStore.to_mapping()`, `replace()`, `replace_many()`, component
    delegation, and callable factory cleanup.
  - The first focused file run exposed an outdated source-regression guard that
    still forbade `RuntimeFieldStore.to_mapping()`; the guard now asserts that
    runtime owns it and `Component.runtime_fields()` delegates to it.
  - The first flake8 run exposed unused re-export imports after deleting the
    private `_PUBLIC_REEXPORTS` anchor; `vercor.components.base.__all__` now
    records the base-module public surface explicitly.

## Validation (Component Runtime Boilerplate Refactor, 2026-05-08)

- `conda run -n scipy pytest tests/ -v --fast 2>&1 | tail -20`
  - passed baseline before implementation (`98 passed, 256 deselected`)
- `conda run -n scipy pytest tests/test_runtime_state.py::test_runtime_field_store_exposes_mapping_membership_and_fallback_helpers tests/test_runtime_state.py::test_runtime_field_store_replace_helpers_preserve_dtype_and_reject_missing tests/test_runtime_state.py::test_runtime_field_store_new_helpers_are_jit_compatible tests/test_component_base_coverage.py::test_component_helpers_seed_and_update_runtime_fields tests/test_api_boundaries.py::test_component_base_internals_are_private_modules -q --tb=short`
  - failed as expected before implementation on missing runtime-store helpers,
    component delegation, and callable factory cleanup
  - passed after implementation
- `conda run -n scipy pytest tests/test_runtime_state.py tests/test_component_base_coverage.py tests/test_api_boundaries.py -q --tb=short`
  - first failed on the stale `to_mapping()` source guard
  - passed after updating the architecture guard
- `conda run -n scipy pytest tests/ -q --fast --tb=short`
  - passed
- `conda run -n scipy black vercor examples tests`
  - passed
  - note: Black emitted the existing Python 3.13 vs target-3.14 safety-check
    warning; it reformatted the touched Python files, then left all 108 files
    unchanged on the final run
- `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`
  - first reported two unused re-export imports in `vercor/components/base.py`
  - passed after adding explicit base-module `__all__` (`0`)
- `conda run -n scipy mypy vercor examples tests`
  - passed (`108 source files`)
- `conda run -n scipy pytest tests/ -q --tb=short`
  - passed

## Component Author API Cleanup

- Removed the legacy public component wrapper/factory surface:
  `Component.wrap()`, `DataComponent.wrap()`, `HostRuntimeComponent.wrap()`,
  `make_data_component()`, `make_differentiable_component()`, and
  `make_host_component()`.
- Kept the user-facing component author API centered on
  `data_component()`, `differentiable_component()`, `host_component()`,
  `DataComponent.from_fields()`, `Component.from_model()`,
  `HostRuntimeComponent.from_model()`, subclass helpers, and
  `ComponentFieldSpec`.
- Collapsed callable-backed component internals so private callable wrappers
  accept one `ComponentFieldSpec`, call `declare_fields(...)`, and use
  `seed_fields(...)` rather than carrying legacy `_required_fields`,
  `_prefill_fields`, or `_field_defaults` metadata.
- Updated API-boundary and component-base tests to assert the legacy entrypoints
  stay absent and callable wrappers use the shared declaration path.
- Updated `DESIGN.md` and `DEPENDENCIES.md` to describe the reduced
  component-author surface.
- Failed approaches / corrections:
  - The red focused API cleanup run failed as expected on remaining `wrap()`,
    `make_*()`, and callable legacy metadata.
  - The focused component/API run passed after removing the legacy delegates and
    routing callable wrappers through `ComponentFieldSpec`.

## Validation (Component Author API Cleanup, 2026-05-08)

- `conda run -n scipy pytest tests/ -v --fast 2>&1 | tail -20`
  - passed baseline before implementation (`98 passed, 256 deselected`)
- `conda run -n scipy pytest tests/test_component_base_coverage.py::test_legacy_wrapper_entrypoints_are_removed tests/test_component_base_coverage.py::test_callable_component_prefills_and_validates_declared_fields tests/test_api_boundaries.py::test_top_level_exports_public_orchestration_and_component_author_api tests/test_api_boundaries.py::test_components_package_exports_only_component_author_contracts tests/test_api_boundaries.py::test_component_base_internals_are_private_modules -q --tb=short`
  - failed as expected before implementation on legacy public entrypoints,
    legacy callable metadata, and stale exports
  - passed after implementation
- `conda run -n scipy pytest tests/test_component_base_coverage.py tests/test_api_boundaries.py -q --tb=short`
  - passed
- `conda run -n scipy pytest tests/ -q --fast --tb=short`
  - passed
- `conda run -n scipy black vercor examples tests`
  - passed
  - note: Black emitted the existing Python 3.13 vs target-3.14 safety-check
    warning and left all 108 files unchanged
- `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`
  - passed (`0`)
- `conda run -n scipy mypy vercor examples tests`
  - passed (`108 source files`)
- `conda run -n scipy pytest tests/ -q --tb=short`
  - passed

# 2026-05-07

## Component API Internal Split

- Split `vercor.components.base` internals into private modules while keeping
  the public component API non-breaking:
  - `_contracts.py` owns `ComponentFieldSpec`, `ComponentStepResult`, author
    field normalization, field-name de-duplication, and field-spec helpers.
  - `_callable_wrappers.py` owns callable signature normalization,
    callable-backed differentiable/host component implementations, and
    field-plus-payload step-result application.
  - `_validation.py` owns component setup validation.
- Kept `Component`, `DataComponent`, `HostRuntimeComponent`, top-level helper
  functions, `wrap()` classmethods, `make_*()` factories, and public context
  aliases available from the same import locations.
- Reduced callable factory chaining so classmethods, facade constructors, and
  compatibility factories share one private constructor per runtime kind.
- Clarified the Veros default-field declaration by passing the output field name
  explicitly to `grid_field_defaults(...)`.
- Failed approaches / corrections:
  - The red API-boundary test failed as expected before private modules existed.
  - The first focused component run exposed recursive helper wiring after the
    factory split; the helpers now import the private callable-wrapper factories.

## Validation (Component API Internal Split, 2026-05-07)

- `conda run -n scipy pytest tests/ -v --fast 2>&1 | tail -20`
  - passed baseline before implementation (`97 passed, 256 deselected`)
- `conda run -n scipy pytest tests/test_api_boundaries.py::test_component_base_internals_are_private_modules -q --tb=short`
  - failed as expected before implementation on missing private modules
  - passed after the split
- `conda run -n scipy pytest tests/test_component_base_coverage.py tests/test_api_boundaries.py -q --tb=short`
  - first failed on recursive helper wiring
  - passed after wiring private callable-wrapper factory imports
- `conda run -n scipy pytest tests/test_component_models_coverage.py tests/test_external_components_coverage.py tests/test_camulator_component_kernels.py -q --tb=short`
  - passed
- `conda run -n scipy black vercor examples tests`
  - passed
  - note: Black emitted the existing Python 3.13 vs target-3.14 safety-check
    warning; it reformatted `vercor/components/base.py` during the public
    re-export cleanup and left all 108 files unchanged on the final run
- `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`
  - first reported intentional public re-export imports plus one unused private
    import
  - passed after making re-exports explicit and removing the unused import (`0`)
- `conda run -n scipy mypy vercor examples tests`
  - passed (`108 source files`)
- `conda run -n scipy pytest tests/ -q --fast --tb=short`
  - passed
- `conda run -n scipy pytest tests/ -q --tb=short`
  - passed

## Additive Component Authoring API Polish

- Added small public `Component` helpers for common subclass-author boilerplate:
  `update_settings(...)`, `grid_field_defaults(...)`, and
  `apply_step_result(...)`.
- Made the base `Component.initialize()` seed declared defaults automatically,
  allowing slab components to rely on their module-level field specs without
  duplicate initialize hooks.
- Refactored bundled data adapters to use `update_settings(...)`; refactored
  CAMulator, JAXGCM, and Veros defaults onto `grid_field_defaults(...)` where
  it keeps field contracts clearer; and used `apply_step_result(...)` for
  JAXGCM field-plus-payload updates.
- Updated the custom component wrapping example and design/dependency docs to
  describe the new helper surface.
- Failed approaches / corrections:
  - The red helper tests failed as expected on missing helper methods and base
    declared-default seeding.
  - The first focused run exposed a test setup error with
    `ComponentInitContext.logger`; the test now supplies the required argument.
  - The first mypy run exposed a tuple/list mismatch for `RunSequence.order`;
    the test now uses the documented list type.

## Validation (Additive Component Authoring API Polish, 2026-05-07)

- `conda run -n scipy pytest tests/test_component_base_coverage.py::test_base_initialize_seeds_declared_defaults tests/test_component_base_coverage.py::test_update_settings_is_chainable tests/test_component_base_coverage.py::test_grid_field_defaults_expands_default_value_and_overrides tests/test_component_base_coverage.py::test_apply_step_result_updates_fields_and_payload -q --tb=short`
  - failed as expected before implementation on missing helper behavior
  - passed after base helper implementation
- `conda run -n scipy pytest tests/test_component_base_coverage.py tests/test_component_models_coverage.py tests/test_external_components_coverage.py tests/test_camulator_component_kernels.py -q --tb=short`
  - passed
- `conda run -n scipy black vercor examples tests`
  - passed
  - note: Black emitted the existing Python 3.13 vs target-3.14 safety-check
    warning and left all 105 files unchanged
- `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`
  - passed (`0`)
- `conda run -n scipy mypy vercor examples tests`
  - first failed on the `RunSequence.order` tuple/list mismatch in the new test
  - passed after the test fix (`105 source files`)
- `conda run -n scipy pytest tests/ -q --fast --tb=short`
  - passed
- `conda run -n scipy pytest tests/ -q --tb=short`
  - passed

## Component Authoring API Polish and Adapter Rewrite

- Added flexible callable wrapper signatures: author step callbacks can now use
  `step(fields)`, `step(fields, context)`, or
  `step(fields, context, payload)` while the runtime still receives one
  normalized internal callable shape.
- Added `field_names` setup introspection and `seed_declared_defaults()` for
  components whose initialization mirrors their declared `ComponentFieldSpec`
  defaults.
- Made `DataComponent` seeding automatically expose seeded fields as declared
  outputs, so data-only adapters remain introspectable even when fields are
  added through helper seeding.
- Refactored slab components to use module-level `ComponentFieldSpec`
  declarations with complete defaults, and refactored data/external adapters to
  expose explicit field contracts while preserving existing constructor call
  sites and runtime behavior.
- Updated the custom wrapping example to show the shorter
  `(fields, context)` differentiable callback signature.
- Failed approaches / corrections:
  - The red focused run failed as expected on shorter callbacks, unsupported
    callback signature validation, missing default-seeding/field-name helpers,
    empty data-component field specs, incomplete slab defaults, and missing
    bundled adapter field contracts.

## Validation (Component Authoring API Polish and Adapter Rewrite, 2026-05-07)

- `conda run -n scipy pytest tests/test_component_base_coverage.py tests/test_component_models_coverage.py tests/test_external_components_coverage.py tests/test_camulator_component_kernels.py -q --tb=short`
  - failed as expected before implementation on the missing polished API and
    bundled adapter declarations
  - passed after base and adapter refactors
- `conda run -n scipy pytest tests/test_component_base_coverage.py tests/test_api_boundaries.py -q --tb=short`
  - passed
- `conda run -n scipy pytest tests/test_component_models_coverage.py tests/test_external_components_coverage.py tests/test_slab_kernels.py tests/test_camulator_component_kernels.py -q --tb=short`
  - passed
- `conda run -n scipy black vercor examples tests`
  - passed
  - note: Black emitted the existing Python 3.13 vs target-3.14 safety-check
    warning and reformatted six touched Python files
- `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`
  - passed (`0`)
- `conda run -n scipy mypy vercor examples tests`
  - passed (`105 source files`)
- `conda run -n scipy pytest tests/ -q --fast --tb=short`
  - passed
- `conda run -n scipy pytest tests/ -q --tb=short`
  - passed

## Component Authoring API Refinement

- Added concise public authoring helpers: `data_component()`,
  `differentiable_component()`, and `host_component()` at both the top-level
  `vercor` package and `vercor.components`.
- Added public `ComponentSetupContext` and `ComponentStepContext` aliases so
  examples and user callbacks can type setup/step contexts without importing
  internal runtime modules.
- Broadened `seed_field()` and `seed_fields()` to accept scalar and array-like
  author values, expanding scalars to grid-shaped fields under the selected
  precision policy.
- Added read-only `field_spec` introspection for declared
  `ComponentFieldSpec` contracts and preserved `inputs` / `outputs`
  declarations through the `from_model()` facade.
- Refactored the custom wrapping example to use the new public helpers and
  step-context alias.
- Refactored slab components to declare their field contracts once, rely on
  base declaration prefill/validation, and use scalar-friendly setup seeding.
- Failed approaches / corrections:
  - The red API run failed as expected because the helper functions, context
    aliases, and `field_spec` property did not exist yet.
  - The first green run exposed that declared slab outputs now prefill
    uninitialized runtime state; slab declarations now provide meaningful
    scalar defaults for prognostic state fields, and coverage expectations were
    updated to reflect the declared contract.

## Validation (Component Authoring API Refinement, 2026-05-07)

- `conda run -n scipy pytest tests/test_component_base_coverage.py tests/test_api_boundaries.py tests/test_component_models_coverage.py -q --tb=short`
  - failed as expected before implementation on missing `data_component()`,
    `field_spec`, public exports, and slab field declarations
  - passed after implementation and slab default refinements
- `conda run -n scipy pytest tests/test_component_base_coverage.py tests/test_api_boundaries.py -q --tb=short`
  - passed
- `conda run -n scipy pytest tests/ -q --fast --tb=short`
  - passed
- `conda run -n scipy black vercor examples tests`
  - passed
  - note: Black emitted the existing Python 3.13 vs target-3.14 safety-check
    warning and left all 105 files unchanged
- `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`
  - passed (`0`)
- `conda run -n scipy mypy vercor examples tests`
  - passed (`105 source files`)
- `conda run -n scipy pytest tests/ -q --tb=short`
  - passed

## Component Authoring Facade Refinement

- Added the public `ComponentFieldSpec` declaration type and author-friendly
  constructors: `DataComponent.from_fields()`, `Component.from_model()`, and
  `HostRuntimeComponent.from_model()`.
- Mapped `inputs`, `outputs`, `default_fields`, and `required_fields` onto the
  existing runtime prefill/validation machinery without changing runtime state
  containers or the backward-compatible `wrap()` / `make_*()` APIs.
- Added scalar-to-grid expansion for callable `default_fields` and data
  `from_fields()` values.
- Added subclass helper methods `declare_fields()`, `has_runtime_field()`,
  `runtime_field_or()`, `runtime_field_or_zeros_like()`, and
  `prefill_runtime_fields()`.
- Refactored the custom wrapping example and slab optional-field reads to use
  the latest helper-first API; kept complex external components as subclasses
  and used the new prefill helper where it clarified JAXGCM runtime setup.
- Updated `DESIGN.md` and `DEPENDENCIES.md` to document the new recommended
  component-author facade.
- Failed approaches / corrections:
  - The red facade tests failed as expected because `ComponentFieldSpec`,
    `from_fields()`, `from_model()`, and optional runtime-field helpers did not
    exist yet.
  - The first fast-suite run exposed lightweight fixtures that construct
    bundled data components with `object.__new__()` and only provide documented
    base setup attributes; base runtime hooks now treat a missing private field
    declaration as an empty `ComponentFieldSpec`.

## Validation (Component Authoring Facade Refinement, 2026-05-07)

- `conda run -n scipy pytest tests/test_component_base_coverage.py tests/test_api_boundaries.py -q --tb=short`
  - failed as expected before implementation because the public facade was
    missing
  - passed after implementation (`34 passed`)
- `conda run -n scipy pytest tests/test_component_base_coverage.py tests/test_component_models_coverage.py tests/test_slab_kernels.py tests/test_external_components_coverage.py tests/test_camulator_component_kernels.py tests/test_api_boundaries.py -q --tb=short`
  - passed after slab/example/JAXGCM refactors
- `conda run -n scipy pytest tests/test_coupler_runtime.py::test_data_forcing_components_run_inside_runtime tests/test_coupler_runtime.py::test_erainterim_ocean_monthly_forcing_replays_to_slab_atmosphere_with_real_regridder tests/test_coupler_runtime.py::test_jcm_land_daily_forcing_replays_to_data_atmosphere_under_jit_and_grad -q --tb=short`
  - passed after the lightweight-fixture field-spec fallback
- `conda run -n scipy black vercor examples tests`
  - passed
  - note: Black emitted the existing Python 3.13 vs target-3.14 safety-check
    warning; it reformatted `vercor/components/base.py` and
    `tests/test_component_base_coverage.py` on the first run and left all files
    unchanged on the final run
- `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`
  - passed (`0`)
- `conda run -n scipy mypy vercor examples tests`
  - passed (`105 source files`)
- `conda run -n scipy pytest tests/ -q --fast --tb=short`
  - first failed on lightweight fixture components missing `_field_spec`
  - passed after the base-hook fallback
- `conda run -n scipy pytest tests/ -q --tb=short`
  - passed

## Helper-First Component Wrapping API

- Added `Component.wrap()`, `DataComponent.wrap()`, and
  `HostRuntimeComponent.wrap()` classmethod constructors while keeping the
  existing `make_data_component()`, `make_differentiable_component()`, and
  `make_host_component()` functions as backward-compatible delegates.
- Extended callable wrappers with declarative `required_fields`,
  `prefill_fields`, and `field_defaults` metadata so user callables can reserve
  scan-stable runtime field slots without touching `RuntimeComponentState`.
- Added grid-shaped seed helpers (`seed_zero_field()`, `seed_zero_fields()`,
  `seed_constant_field()`) and `require_runtime_fields()` for subclass
  validators.
- Refactored bundled data, slab, JAXGCM, CAMulator, and Veros adapters to use
  helper methods for setup field seeding and runtime field updates where their
  lifecycle contracts allow it.
- Added `examples/custom_component_wrapping.py` with data-only, differentiable
  callable, and host-runtime wrapper examples.
- Updated `DESIGN.md` and `DEPENDENCIES.md` to document the helper-first
  component-author API.
- Failed approaches / corrections:
  - The red wrapper tests failed as expected because `wrap()` classmethods and
    seed/default helpers did not exist yet.
  - Focused external tests exposed lightweight `__new__()` fixtures that skipped
    base initialization; those fixtures now provide the same minimal base
    attributes and runtime field slots real initialized components provide.
  - The first mypy pass rejected the intentionally narrower
    `DataComponent.wrap()` signature and untyped mixin attributes; the final
    implementation uses a targeted override ignore plus explicit casts in the
    shared callable mixin.
  - The first full-suite run exposed the same base-initialization fixture issue
    in CAMulator step coverage; the fixture now uses the component runtime-field
    initializer before host stepping.

## Validation (Helper-First Component Wrapping API, 2026-05-07)

- `conda run -n scipy pytest tests/test_component_base_coverage.py::test_wrap_classmethods_create_data_differentiable_and_host_components tests/test_component_base_coverage.py::test_wrapped_callable_component_prefills_and_validates_required_fields tests/test_component_base_coverage.py::test_wrapped_callable_component_reports_missing_required_fields tests/test_component_base_coverage.py::test_component_seed_default_helpers_and_required_field_validator -q --tb=short`
  - failed as expected before implementation because the helper API was missing
  - passed after implementation
- `conda run -n scipy pytest tests/test_component_base_coverage.py tests/test_component_models_coverage.py tests/test_external_components_coverage.py -q --tb=short`
  - passed
- `conda run -n scipy black vercor examples tests`
  - passed
  - note: Black emitted the existing Python 3.13 vs target-3.14 safety-check
    warning and reformatted `tests/test_camulator_component_kernels.py`
- `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`
  - passed (`0`)
- `conda run -n scipy mypy vercor examples tests`
  - passed (`105 source files`)
- `conda run -n scipy pytest tests/ -q --fast --tb=short`
  - passed
- `conda run -n scipy pytest tests/ -q --tb=short`
  - passed

## User-Friendly Component Wrapping API

- Added top-level and `vercor.components` factory helpers for common custom
  adapter cases: data-only components, differentiable callable components, and
  host-runtime callable components.
- Added `ComponentStepResult` so callable wrappers can update runtime fields
  while replacing optional runtime payloads for shape-stable model state.
- Added `Component.seed_field()`, `seed_fields()`, `runtime_fields()`,
  `runtime_field()`, and `with_runtime_fields()` helpers to keep subclass
  authors away from direct `RuntimeComponentState` store manipulation in common
  cases.
- Kept the existing subclass contracts intact for components that need custom
  initialization, validation, prefill, or full lifecycle hooks.
- Updated `DESIGN.md` and `DEPENDENCIES.md` to document the helper layer.
- Failed approaches / corrections:
  - The focused red test run failed as expected because the factories, helper
    methods, `ComponentStepResult`, and public exports did not exist yet.

## Validation (User-Friendly Component Wrapping API, 2026-05-07)

- `conda run -n scipy pytest tests/test_component_base_coverage.py tests/test_api_boundaries.py -q --tb=short`
  - failed as expected before implementation because the helper API was missing
  - passed after implementation
- `conda run -n scipy black vercor examples tests`
  - passed
  - note: Black emitted the existing Python 3.13 vs target-3.14 safety-check
    warning and reformatted `tests/test_component_base_coverage.py`
- `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`
  - passed (`0`)
- `conda run -n scipy mypy vercor examples tests`
  - first failed on payload tests that treated `Any | None` as a mapping
  - passed after adding explicit mapping assertions (`104 source files`)
- `conda run -n scipy pytest tests/ -q --fast --tb=short`
  - passed
- `conda run -n scipy pytest tests/ -q --tb=short`
  - passed

# 2026-05-06

## Coupler Lifecycle Logging

- Added `DEFAULT_LOGGER_NAME` and `get_default_logger()` as the shared default
  VerCOR host logger boundary.
- Replaced active `print(...)` calls reached by coupler initialization, runtime,
  and final-output paths with logging through either the coupler logger or the
  default `VerCOR` Python logger.
- Threaded the coupler logger into mask conservation checks during
  `Coupler.initialize()` and added optional logger injection for CAMulator
  initialization/noise helpers and JAXGCM output writes.
- Kept standalone diagnostics and examples unchanged because their stdout
  behavior is user-facing and outside the coupler lifecycle scope.
- Updated `DESIGN.md` to document the default logger fallback and explicit
  coupler logger injection boundary.
- Failed approaches / corrections:
  - The first red test run failed during collection because
    `DEFAULT_LOGGER_NAME` and `get_default_logger()` did not exist yet.
  - The first green test run exposed that optional CREDIT imports are absent in
    the local environment; the CAMulator logging test now monkeypatches those
    optional symbols explicitly.
  - The first mypy run exposed that the test recording logger had narrower
    method signatures than `LoggerLike`; the helper now matches the protocol.

## Validation (Coupler Lifecycle Logging, 2026-05-06)

- `conda run -n scipy pytest tests/test_coupler_coverage.py::test_default_logger_uses_vercor_logger_name tests/test_tools_assets_and_regridding.py::test_check_remap_conservation_handles_skip_and_mismatch tests/test_camulator_component_kernels.py::test_camulator_constructor_logs_save_forecast_path tests/test_camulator_component_kernels.py::test_add_init_noise_logs_through_injected_logger tests/test_camulator_component_kernels.py::test_initialize_camulator_logs_lifecycle_through_injected_logger tests/test_external_components_coverage.py::test_jax_gcm_write_output_persists_mean_dataset -q --tb=short`
  - failed as expected before implementation because the default logger
    interface was missing
  - passed after implementation (`6 passed`)
- `conda run -n scipy black vercor examples tests`
  - passed
  - note: Black emitted the existing Python 3.13 vs target-3.14 safety-check
    warning and left 104 files unchanged on the final run
- `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`
  - passed (`0`)
- `conda run -n scipy mypy vercor examples tests`
  - passed (`104 source files`)
- `conda run -n scipy pytest tests/ -q --fast --tb=short`
  - passed
- `conda run -n scipy pytest tests/ -q --tb=short`
  - passed

## Dynamic Settings Attribute Refactor

- Removed per-setting `@property` descriptors from `VercorSettings`; settings
  now read and write through the existing `__getattr__` / `__setattr__`
  metadata path.
- Added class-level annotations for default settings so `mypy` still sees known
  precision, timing, and physical-constant attributes with concrete types.
- Kept `dtype_policy` as the only computed settings property because it is not a
  stored setting value.
- Added `__dir__` support so default and custom settings appear in
  introspection/autocomplete.
- Added regression coverage for dynamic settings descriptors, metadata
  preservation through dot assignment, `dir()` output, and precision-protocol
  compatibility.
- Updated `DESIGN.md` and `DEPENDENCIES.md` to document the dynamic settings
  boundary.
- Failed approaches / corrections:
  - The first red test run failed as expected because default setting names were
    not class-level annotations and custom settings were absent from `dir()`.

## Validation (Dynamic Settings Attribute Refactor, 2026-05-06)

- `conda run -n scipy pytest tests/test_settings.py -q --tb=short`
  - failed as expected before implementation because default settings were not
    class-level annotations and custom settings were absent from `dir()`
  - passed after implementation
- `conda run -n scipy pytest tests/test_settings.py tests/test_dtypes.py tests/test_fluxes_utilities.py tests/test_camulator_component_kernels.py -q --fast --tb=short`
  - passed
- `conda run -n scipy black vercor examples tests`
  - passed
  - note: Black emitted the existing Python 3.13 vs target-3.14 safety-check
    warning and left 104 files unchanged
- `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`
  - passed (`0`)
- `conda run -n scipy mypy vercor examples tests`
  - passed (`104 source files`)
- `conda run -n scipy pytest tests/ -q --fast --tb=short`
  - passed
- `conda run -n scipy pytest tests/ -q --tb=short`
  - passed

## Unified Metadata-Backed Settings Container

- Replaced the separate `VercorSettings` and `ComponentSettings` dataclasses
  with one metadata-backed `VercorSettings` container populated from
  `DEFAULT_SETTINGS`.
- Added `Settings(value, description, units)` metadata records for coupler,
  component, precision, timing, physical-constant, and bulk-formula settings.
- Preserved attribute-style reads and existing-setting assignment for
  compatibility while adding explicit `add_setting()`, `set_value()`,
  `get_value()`, `get_metadata()`, and `as_values()` APIs.
- Updated components and the coupler so each instance owns an independent
  `VercorSettings` container, with `ComponentSettings` retained only as a
  compatibility alias.
- Updated `DESIGN.md` and `DEPENDENCIES.md` to document the unified settings
  boundary.
- Failed approaches / corrections:
  - The first red test run failed during collection because `DEFAULT_SETTINGS`
    and `Settings` did not exist yet.
  - The first green test run exposed a test fixture bug using `Clock(steps=1)`;
    the fixture now supplies an explicit start date and timestep.
  - The first mypy run exposed that the dynamic settings container leaked `Any`
    through typed flux utilities and no longer satisfied `SupportsEnableX64` as
    a settable protocol attribute; typed known-setting properties, setters, and
    a read-only precision protocol fixed the static contract.

## Validation (Unified Metadata-Backed Settings Container, 2026-05-06)

- `conda run -n scipy pytest tests/test_settings.py tests/test_dtypes.py::test_dtype_policy_reads_updated_settings_value -q --tb=short`
  - failed as expected before implementation because the metadata settings API
    did not exist
  - passed after implementation
- `conda run -n scipy pytest tests/test_settings.py tests/test_dtypes.py tests/test_component_base_coverage.py tests/test_runtime_state.py tests/test_coupler_runtime.py -q --fast --tb=short`
  - passed
- `conda run -n scipy black vercor examples tests`
  - passed
  - note: Black emitted the existing Python 3.13 vs target-3.14 safety-check
    warning and reformatted `vercor/settings.py` and `tests/test_settings.py`
- `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`
  - passed (`0`)
- `conda run -n scipy mypy vercor examples tests`
  - passed (`104 source files`)
- `conda run -n scipy pytest tests/ -q --fast --tb=short`
  - passed
- `conda run -n scipy pytest tests/ -q --tb=short`
  - passed
- `conda run -n scipy pytest tests/test_settings.py tests/test_dtypes.py tests/test_component_base_coverage.py tests/test_runtime_state.py tests/test_coupler_runtime.py -q --fast --tb=short`
  - passed after the final static typing fixes

# 2026-05-05

## Compiled Runtime Wakeup-Fd Interrupt Handling

- Added wakeup-fd polling to the runtime interrupt controller so terminal
  signals delivered while the main thread is inside compiled XLA work can be
  observed by existing scanned-runtime callback checkpoints.
- `RuntimeInterruptController.signal_scope()` now installs and restores a
  temporary nonblocking wakeup pipe alongside the existing terminal signal
  handlers.
- Runtime checkpoints now drain pending wakeup-fd signal bytes before checking
  the shared pending interrupt request, so host and scanned runtimes keep one
  cancellation path.
- Updated `DESIGN.md` and `DEPENDENCIES.md` to document wakeup-fd polling as
  the compiled-runtime signal bridge.
- Failed approaches / corrections:
  - The first regression tests failed as expected because the controller had no
    `_wakeup` bridge and compiled callbacks surfaced the missing path as a
    `JaxRuntimeError`.

## Validation (Compiled Runtime Wakeup-Fd Interrupt Handling, 2026-05-05)

- `conda run -n scipy pytest tests/ -v --fast --tb=short`
  - passed before implementation (`68 passed`)
- `conda run -n scipy pytest tests/test_runtime_interrupts.py::test_checkpoint_observes_wakeup_fd_signal_without_python_handler tests/test_runtime_interrupts.py::test_compiled_scanned_runtime_observes_wakeup_fd_interrupt -q --tb=short`
  - failed as expected before implementation because
    `RuntimeInterruptController` had no `_wakeup` bridge
  - passed after implementation (`2 passed`)
- `conda run -n scipy pytest tests/test_runtime_interrupts.py -q --tb=short`
  - passed (`10 passed`)
- `conda run -n scipy black vercor examples tests`
  - passed
  - note: Black emitted the existing Python 3.13 vs target-3.14 safety-check
    warning and left 103 files unchanged
- `conda run -n scipy pytest tests/test_runtime_interrupts.py -q --tb=short`
  - passed after formatting (`10 passed`)
- `conda run -n scipy pytest tests/ -q --fast --tb=short`
  - passed
- `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`
  - passed (`0`)
- `conda run -n scipy mypy vercor examples tests`
  - passed (`103 source files`)
- `conda run -n scipy pytest tests/ -q --tb=short`
  - passed

## Unified Runtime Interrupt Handling

- Added an internal runtime interrupt controller for terminal shortcut
  cancellation across both host and JIT-scanned coupler integrations.
- `Coupler.run()` now installs temporary handlers for `SIGINT`, `SIGTERM`, and
  `SIGTSTP`, restoring previous handlers when the run exits.
- Host runtime steps and components now share explicit interruption
  checkpoints with the scanned runtime.
- The scanned runtime now inserts ordered `jax.debug.callback` checkpoints that
  are independent of progress logging, so compiled scans can observe pending
  terminal interrupts even when info logging is disabled.
- Interrupt callback failures from JAX are translated back to a
  `KeyboardInterrupt` subclass while unrelated JAX runtime errors are preserved.
- Updated `DESIGN.md` and `DEPENDENCIES.md` to document the interrupt boundary.
- Failed approaches / corrections:
  - The initial red tests failed at collection because
    `vercor.runtime.interrupts` did not exist.
  - After adding the controller but before wiring `Coupler.run()`, the host
    integration test still surfaced a raw `KeyboardInterrupt`; wrapping `run()`
    in the controller signal scope and adding host checkpoints fixed the path.

## Validation (Unified Runtime Interrupt Handling, 2026-05-05)

- `conda run -n scipy pytest tests/test_runtime_interrupts.py -q --tb=short`
  - failed as expected before implementation because the interrupt module was
    missing
  - passed after implementation (`8 passed`)
- `conda run -n scipy pytest tests/test_coupler_coverage.py tests/test_runtime_run_cache.py tests/test_runtime_state.py -q --tb=short`
  - passed (`37 passed`)
- `conda run -n scipy pytest tests/test_runtime_interrupts.py tests/test_coupler_coverage.py tests/test_runtime_run_cache.py -q --tb=short`
  - passed (`35 passed`)
- `conda run -n scipy black vercor examples tests`
  - passed
  - note: Black emitted the existing Python 3.13 vs target-3.14 safety-check
    warning and reformatted `vercor/runtime/interrupts.py` and
    `tests/test_runtime_interrupts.py`
- `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`
  - passed (`0`)
- `conda run -n scipy mypy vercor examples tests`
  - passed (`103 source files`)
- `conda run -n scipy pytest tests/ -q --fast --tb=short`
  - passed (`68 passed`)
- `conda run -n scipy pytest tests/ -q --tb=short`
  - passed

## Scanned Runtime Progress Logging

- Added host-equivalent step and component progress logging to the pure
  `jax.lax.scan` runtime path.
- Centralized runtime progress message formatting so the host and scanned loops
  share the same step-header and component-run strings.
- Added host-side log emission for ordered JAX callbacks, allowing scanned
  progress callbacks to write through callback-backed, standard Python, or
  lightweight test loggers without nesting another callback.
- Updated `DESIGN.md` to document that scanned runtime progress labels are
  precomputed on the host and selected inside ordered callbacks.
- Failed approaches / corrections:
  - The new regression test failed before implementation because scanned
    runtime logs only contained component-internal callback messages and lacked
    the outer step/component progress lines.

## Validation (Scanned Runtime Progress Logging, 2026-05-05)

- `conda run -n scipy pytest tests/test_coupler_coverage.py::test_scanned_runtime_logs_host_equivalent_progress_messages -q --tb=short`
  - failed as expected before implementation because step headers were missing
  - passed after implementation
- `conda run -n scipy pytest tests/test_coupler_coverage.py -q --fast --tb=short`
  - passed
- `conda run -n scipy pytest tests/test_coupler_coverage.py::test_scanned_runtime_suppresses_info_below_log_level -q --tb=short`
  - passed
- `conda run -n scipy black vercor examples tests`
  - passed
  - note: Black emitted the existing Python 3.13 vs target-3.14 safety-check
    warning and left 101 files unchanged
- `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`
  - passed (`0`)
- `conda run -n scipy mypy vercor examples tests`
  - passed (`101 source files`)
- `conda run -n scipy pytest tests/ -q --fast --tb=short`
  - passed
- `conda run -n scipy pytest tests/ -q --tb=short`
  - passed

## JAXGCM Forcing Payload Scan Shape Stability

- Added a focused regression test for scanned JAXGCM runtime payloads whose
  forcing template carries time-dependent fields.
- Kept `JAXGCMRuntimePayload.forcing` shape-stable across `jax.lax.scan` by
  using the copied exchange-forcing object only for the local model step and
  preserving the original forcing template in the returned payload.
- Updated `DESIGN.md` to document that runtime payload pytrees carried through
  `jax.lax.scan` must preserve leaf shapes and dtypes.
- Failed approaches / corrections:
  - The regression test failed as expected before implementation with a
    `scan body function carry input and carry output must have equal types`
    error because forcing leaves changed from `(nLat, nLon, nTime)` to
    `(nLat, nLon)`.
  - Avoided changing public component APIs or storing per-step forcing slices in
    the payload, since that would keep the scan carry shape unstable for
    time-dependent forcing templates.

## Validation (JAXGCM Forcing Payload Scan Shape Stability, 2026-05-05)

- `conda run -n scipy pytest tests/test_coupler_runtime.py::test_jax_gcm_runtime_keeps_time_dependent_forcing_payload_shape_stable -q --tb=short`
  - failed as expected before implementation with the scan carry shape mismatch
  - passed after preserving the forcing template in the runtime payload
- `conda run -n scipy pytest tests/test_coupler_runtime.py::test_jax_gcm_runtime_keeps_time_dependent_forcing_payload_shape_stable tests/test_coupler_runtime.py::test_jax_gcm_runs_inside_runtime_under_jit_and_grad tests/test_coupler_runtime.py::test_data_forcing_replays_into_jax_gcm_runtime_under_jit_grad_and_jvp tests/test_external_components_coverage.py::test_jax_gcm_step_maps_outputs_and_respects_output_gate -q --tb=short`
  - passed
- `conda run -n scipy black vercor examples tests`
  - passed
  - note: Black emitted the existing Python 3.13 vs target-3.14 safety-check
    warning and reformatted `tests/test_coupler_runtime.py`
- `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`
  - passed (`0`)
- `conda run -n scipy mypy vercor examples tests`
  - passed (`101 source files`)
- `conda run -n scipy pytest tests/ -q --fast --tb=short`
  - passed
- `conda run -n scipy pytest tests/ -q --tb=short`
  - passed

# 2026-05-04

## Centralized VerCOR Dtype Policy

- Added `vercor.dtypes` as the single dtype policy boundary for VerCOR-owned
  JAX/NumPy real and index arrays.
- Added `VercorSettings.dtype_policy`; settings-bound helpers use
  `enable_x64` for real arrays while canonical index arrays remain `int32`.
- Replaced production hard-coded JAX dtype spellings with dtype helpers across
  flux kernels, slab components, external adapters, interpolators, regridders,
  runtime metadata, and coupler-owned masks.
- Kept no-settings conversion helpers from upcasting already-typed arrays; array
  creation helpers without settings still follow the active JAX precision.
- Made `RuntimeFieldStore.set()` preserve the dtype of existing fields during
  replacement so `jax.lax.scan` carry dtypes stay stable.
- Updated exact dtype tests, the explicit NumPy-boundary test, `DESIGN.md`, and
  `DEPENDENCIES.md`.
- Failed approaches / corrections:
  - The first new dtype test failed as expected because `vercor.dtypes` did not
    exist.
  - A no-settings conversion initially upcast initialized `float32` fields to
    the test harness's global `float64`; a regression test now preserves existing
    array dtypes when no settings are supplied.
  - Runtime scan tests exposed regridder/kernel outputs replacing `float32`
    carry fields with `float64`; replacement now casts to the existing field
    dtype.
  - Symmetrically disabling global `jax_enable_x64` during
    `Coupler.initialize()` caused process-wide test-harness side effects; the
    implementation keeps the previous "enable when requested" global behavior
    and uses explicit settings-bound dtype helpers where settings are available.

## Validation (Centralized VerCOR Dtype Policy, 2026-05-04)

- `conda run -n scipy pytest tests/test_dtypes.py tests/test_conservative_rectilinear_remapper.py::test_remapper_accepts_jax_backed_constructor_inputs tests/test_bilinear_rectilinear_interpolator.py::test_scalar_periodic_longitude_wrap_uses_dateline_cell -q --tb=short`
  - failed as expected before implementation because `vercor.dtypes` did not
    exist
  - passed after implementation
- `conda run -n scipy pytest tests/test_dtypes.py::test_unconfigured_real_conversion_preserves_existing_array_dtype tests/test_component_models_coverage.py::test_slab_component_initialize_and_step_behaviors -q --tb=short`
  - failed before the no-settings conversion fix
  - passed after preserving existing array dtypes
- `conda run -n scipy pytest tests/test_runtime_state.py::test_runtime_field_store_replacement_preserves_existing_dtype tests/test_tools_components_and_plotting.py::test_grids_identical_detects_equal_and_unequal_grids tests/test_coupler_runtime.py::test_initialized_slab_coupler_creates_jittable_runtime_state tests/test_coupler_runtime.py::test_initialized_slab_coupler_run_prefills_missing_imports tests/test_coupler_runtime.py::test_scanned_runtime_state_uses_runtime_field_stores tests/test_coupler_runtime.py::test_mixed_grid_slab_coupler_runs_with_real_regridders_under_jit_grad_and_jvp -q --tb=short`
  - failed before `RuntimeFieldStore.set()` preserved replacement dtypes
  - passed after the runtime store fix
- `conda run -n scipy black vercor examples tests`
  - passed
  - note: Black emitted the existing Python 3.13 vs target-3.14 safety-check
    warning; final run left 101 files unchanged
- `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`
  - passed (`0`)
- `conda run -n scipy mypy vercor examples tests`
  - passed (`101 source files`)
- `conda run -n scipy pytest tests/ -q --fast --tb=short`
  - passed
- `conda run -n scipy pytest tests/ -q --tb=short`
  - passed

## Canonical Component Data Dimension Order

- Added a shared `vercor.field_layout` module for canonical component
  `data` field validation and time-last forcing normalization helpers.
- Enforced `Component.data` as a grid-field store with trailing
  `(nLat, nLon)` dimensions and accepted layouts `(nLat, nLon)`,
  `(nTime, nLat, nLon)`, `(nLev, nLat, nLon)`, and
  `(nTime, nLev, nLat, nLon)`.
- Runtime state creation validates component seed data after generic/component
  prefill, so invalid setup data fails before JAX runtime execution.
- Moved ERA5 hybrid-coordinate coefficients (`hyai`, `hybi`, `hyam`, `hybm`)
  from `ERA5Atmosphere.data` to component attributes.
- Converted ERA5, ERA-Interim, and JCM data adapters and runtime monthly
  forcing selection to the canonical leading-time-axis convention.
- Updated `DESIGN.md` and `DEPENDENCIES.md` for the canonical data-field
  layout boundary.
- Failed approaches / corrections:
  - The initial red tests confirmed that legacy `(nLon, nLat, nTime)` data and
    1D metadata were not rejected before this slice.
  - The runtime monthly-send red test confirmed that interpolation still read
    time from the trailing axis and transposed the selected field.

## Validation (Canonical Component Data Dimension Order, 2026-05-04)

- `conda run -n scipy pytest tests/test_component_base_coverage.py::test_component_data_layout_validation_rejects_non_grid_data_fields -q --tb=short`
  - failed as expected before implementation because invalid data layout was not
    rejected
  - passed after implementation
- `conda run -n scipy pytest tests/test_runtime_state.py::test_runtime_send_applies_monthly_interpolation_under_jit_and_grad -q --tb=short`
  - failed as expected before implementation because monthly send still used
    the trailing time axis
  - passed after implementation
- `conda run -n scipy pytest tests/test_component_base_coverage.py tests/test_runtime_state.py tests/test_component_models_coverage.py tests/test_data_component_kernels.py tests/test_coupler_runtime.py tests/test_tools_time_and_forcing.py -q --fast --tb=short`
  - passed
- `conda run -n scipy pytest tests/ -q --fast --tb=short`
  - passed
- `conda run -n scipy black vercor examples tests`
  - passed
  - note: Black emitted the existing Python 3.13 vs target-3.14 safety-check
    warning and reformatted four touched files
- `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`
  - passed (`0`)
- `conda run -n scipy mypy vercor examples tests`
  - passed (`99 source files`)
- `conda run -n scipy pytest tests/ -q --fast --tb=short`
  - passed after formatting
- `conda run -n scipy pytest tests/ -q --tb=short`
  - passed

# 2026-05-01

## ERA5 Atmosphere Pure Data Component

- Converted `ERA5Atmosphere` to inherit `DataComponent` and removed its custom
  runtime prefill, validation, and step hooks.
- Kept `DataComponent` as the shared no-op runtime-step contract and marked the
  data/host scanned-runtime boundary implementations as final.
- Moved plotting-only combined land/sea surface temperature into diagnostics:
  `combine_surface_temperatures()` and `total_surface_temperature(view)`.
- Updated `examples/run_data_driver.py` so ATM total surface temperature is a
  callable plotting diagnostic rather than a runtime-state field.
- Updated runtime tests to assert ERA5 atmosphere receives land/sea fields while
  `total_surface_temperature` remains absent from ATM runtime data.
- Updated `DESIGN.md` and `DEPENDENCIES.md` to document derived plotting
  diagnostics outside component runtime state.
- Failed approaches / corrections:
  - The red tests failed as expected because `ERA5Atmosphere` still inherited
    `Component`, diagnostics helpers were missing, and the runtime still carried
    `total_surface_temperature`.
  - The first `mypy` run exposed two active-runtime test doubles that inherited
    `DataComponent` while overriding its now-final no-op step; those helpers now
    inherit `Component`.

## Validation (ERA5 Atmosphere Pure Data Component, 2026-05-01)

- `conda run -n scipy pytest tests/test_component_base_coverage.py::test_era5_atmosphere_uses_data_component_runtime_contract tests/test_data_component_kernels.py::test_era5_atmosphere_helpers_support_jit_and_gradients tests/test_data_component_kernels.py::test_total_surface_temperature_diagnostic_uses_runtime_view_fields tests/test_tools_components_and_plotting.py::test_plot_component_scalar_vector_comparison_accepts_callable_scalar tests/test_coupler_runtime.py::test_data_forcing_components_run_inside_runtime -q --tb=short`
  - failed as expected before implementation
  - passed after implementation
- `conda run -n scipy pytest tests/test_component_base_coverage.py tests/test_component_models_coverage.py tests/test_data_component_kernels.py tests/test_tools_components_and_plotting.py tests/test_coupler_runtime.py -q --fast --tb=short`
  - passed
- `conda run -n scipy black vercor examples tests`
  - passed
  - note: Black emitted the existing Python 3.13 vs target-3.14 safety-check
    warning and left 98 files unchanged
- `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`
  - passed (`0`)
- `conda run -n scipy mypy vercor examples tests`
  - first failed on two test doubles overriding `DataComponent.step_runtime_state()`;
    passed after switching those helpers to `Component`
- `conda run -n scipy pytest tests/ -q --fast --tb=short`
  - passed
- `conda run -n scipy pytest tests/ -q --tb=short`
  - passed

## Explicit Component Author Contracts

- Added `DataComponent` as the explicit base class for data-only/forcing
  adapters with an intentional no-op runtime step.
- Made `Component` the active differentiable-model base class with an abstract
  `step_runtime_state()` hook.
- Made `HostRuntimeComponent` require `step_host_runtime_state()` and raise a
  clear `ComponentError` if a host-backed adapter is accidentally sent through
  the differentiable scanned runtime.
- Added setup validation for required component-author attributes (`name`,
  `grid`, `data`, and `settings`) so adapters that skip
  `super().__init__(name, grid=...)` fail with actionable diagnostics before
  initialization, runtime execution, or finalization.
- Updated bundled data components, public exports, architecture docs, and
  component/API-boundary tests for the three public author-facing roles:
  `Component`, `DataComponent`, and `HostRuntimeComponent`.
- Failed approaches / corrections:
  - The first API-boundary red test imported `DataComponent` directly before the
    symbol existed, causing collection to stop; it was changed to assert the
    missing public symbol during test execution.
  - `mypy` correctly flagged tests that intentionally instantiate abstract
    classes; those lines now use targeted `# type: ignore[abstract]` comments.
  - The first full-suite run exposed a test-local `JAXGCM.__new__()` fixture
    that skipped the base `settings` attribute; the fixture now supplies
    `ComponentSettings()` rather than weakening production validation.

## Validation (Explicit Component Author Contracts, 2026-05-01)

- `conda run -n scipy pytest tests/test_component_base_coverage.py tests/test_api_boundaries.py -q --fast --tb=short`
  - failed as expected before implementation because `DataComponent` was not
    exported and `HostRuntimeComponent` did not require an explicit host step
  - passed after implementation
- `conda run -n scipy pytest tests/test_component_base_coverage.py tests/test_api_boundaries.py tests/test_runtime_state.py tests/test_coupler_runtime.py tests/test_coupler_coverage.py -q --fast --tb=short`
  - passed
- `conda run -n scipy black vercor examples tests`
  - passed
  - note: Black emitted the existing Python 3.13 vs target-3.14 safety-check
    warning; final run left 98 files unchanged
- `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`
  - passed (`0`)
- `conda run -n scipy mypy vercor examples tests`
  - passed (`98 source files`)
- `conda run -n scipy pytest tests/ -q --fast --tb=short`
  - passed
- `conda run -n scipy pytest tests/ -q --tb=short`
  - first failed because a test-local `JAXGCM.__new__()` fixture skipped
    `settings`; passed after adding the missing fixture attribute

# 2026-04-30

## JAX Callback Runtime Logging

- Added `vercor/jax_logging.py` with a callback-backed `JaxCallbackLogger`,
  `LoggerLike` protocol, `setup_logger(level=..., name=...)`, and logging-level
  normalization helpers.
- Added `Coupler(..., log_level=...)` so the logging threshold is configured at
  instantiation, and included logger identity/effective level in the scanned
  runtime compile-cache key. Plain `logging.Logger` instances passed at
  construction are wrapped with the same callback-backed logger.
- Passed the coupler logger into scanned runtime component contexts so pure
  components can log from inside `jax.lax.scan`, `jax.jit`, `jax.grad`, and
  `jax.jvp` paths through `jax.debug.callback`.
- Updated JAXGCM traced diagnostic logs to pass JAX values as logger arguments
  instead of converting tracers with `float(...)` or `int(...)`.
- Documented the callback logging boundary in `DESIGN.md` and added
  `vercor/jax_logging.py` to `DEPENDENCIES.md`.
- Failed approaches / corrections:
  - The first implementation imported NumPy for callback value formatting; the
    production NumPy-boundary guard caught this, so callback host values now use
    `jax.device_get` and scalar `.item()` instead.
  - The first full-suite run exposed test-local recording loggers that accepted
    only a single message argument; those test doubles now match the runtime
    logger protocol.

## Validation (JAX Callback Runtime Logging, 2026-04-30)

- `conda run -n scipy pytest tests/test_coupler_coverage.py::test_coupler_accepts_log_level_at_instantiation tests/test_coupler_coverage.py::test_setup_logger_formats_traced_values_under_scan tests/test_coupler_coverage.py::test_scanned_runtime_passes_callback_logger_to_components tests/test_coupler_coverage.py::test_scanned_runtime_suppresses_info_below_log_level -q --tb=short`
  - failed as expected before implementation because `Coupler.__init__()` had no
    `log_level` argument and `setup_logger()` accepted no `level`/`name`
    arguments
  - passed after implementation
- `conda run -n scipy pytest tests/test_coupler_coverage.py tests/test_coupler_runtime.py tests/test_runtime_state.py -q --fast --tb=short`
  - passed
- `conda run -n scipy pytest tests/ -q --fast --tb=short`
  - first failed because `vercor/jax_logging.py` imported NumPy outside the
    explicit host boundaries
  - passed after removing the NumPy dependency
- `conda run -n scipy pytest tests/test_external_components_coverage.py::test_jax_gcm_step_maps_outputs_and_respects_output_gate -q --tb=short`
  - passed after updating recording logger test doubles
- `conda run -n scipy black vercor examples tests`
  - passed
  - note: Black emitted the existing Python 3.13 vs target-3.14 safety-check
    warning and left 98 files unchanged
- `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`
  - passed (`0`)
- `conda run -n scipy mypy vercor examples tests`
  - passed (`98 source files`)
- `conda run -n scipy pytest tests/ -q --fast --tb=short`
  - passed
- `conda run -n scipy pytest tests/ -q --tb=short`
  - passed

## Runtime Package Refactor

- Moved all root-level runtime modules into the `vercor.runtime` package:
  - `vercor/runtime/state.py` owns immutable runtime state, contracts, field
    stores, and exchange dispatch
  - `vercor/runtime/components.py` owns component runtime-state creation,
    prefill/validation, receive/send, and time-selection helpers
  - `vercor/runtime/contexts.py`, `driver.py`, `time.py`, and `views.py` own the
    focused runtime context, dispatch, time metadata, and view helpers
- Removed the old root sibling runtime module paths instead of keeping
  compatibility shims.
- Kept `from vercor.runtime import RuntimeComponentState, ...` working through
  the internal runtime package re-export surface while keeping runtime internals
  out of top-level `vercor.__all__`.
- Updated source, tests, monkeypatch paths, `DESIGN.md`, and `DEPENDENCIES.md`
  for the package layout.
- No failed implementation approaches. The TDD red check failed for the expected
  reason before the package move: `vercor.runtime` was still a file module and
  could not expose `vercor.runtime.contexts`.

## Validation (Runtime Package Refactor, 2026-04-30)

- `conda run -n scipy pytest tests/test_runtime_state.py::test_runtime_module_does_not_own_component_specific_steps tests/test_api_boundaries.py -q --fast --tb=short`
  - failed as expected before implementation because `vercor.runtime` was not a
    package
- `conda run -n scipy pytest tests/test_runtime_state.py tests/test_api_boundaries.py tests/test_component_base_coverage.py tests/test_coupler_coverage.py tests/test_runtime_run_cache.py tests/test_runtime_exchange.py -q --fast --tb=short`
  - passed
- `conda run -n scipy black vercor examples tests`
  - passed
  - note: Black emitted the existing Python 3.13 vs target-3.14 safety-check
    warning and reformatted `tests/test_runtime_state.py`
- `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`
  - passed (`0`)
- `conda run -n scipy mypy vercor examples tests`
  - passed (`97 source files`)
- `conda run -n scipy pytest tests/ -q --fast --tb=short`
  - passed
- `conda run -n scipy pytest tests/ -q --tb=short`
  - passed

## Public/Runtime API Boundary Clarification

- Preserved the current differentiable runtime architecture and kept runtime
  state containers, runtime contexts, and component runtime hook signatures
  unchanged.
- Clarified the public package surface:
  - `vercor.__all__` now exports `Component`, `HostRuntimeComponent`, and
    `RunSequence` alongside the existing orchestration API
  - runtime internals remain out of the package top-level
  - `vercor.components` remains limited to `Component` and
    `HostRuntimeComponent`
- Updated examples to import `RunSequence` from the top-level `vercor` public
  API instead of `vercor.coupler`.
- Converted misplaced class documentation in `Component`, `Coupler`,
  `Exchange`, and `RunSequence` into actual class docstrings explaining the
  setup-time public API vs immutable runtime-state boundary.
- Added API-boundary regression coverage in `tests/test_api_boundaries.py`.
- Added a `DESIGN.md` section documenting public orchestration APIs,
  component-author hooks, and internal runtime state APIs.
- No `DEPENDENCIES.md` update was needed because module ownership and
  dependency order did not change.
- No failed architectural approaches. The TDD red check failed as expected
  before implementation; one broad multi-file example import patch had stale
  context and was split into smaller patches without changing the design.

## Validation (Public/Runtime API Boundary Clarification, 2026-04-30)

- `conda run -n scipy pytest tests/test_api_boundaries.py -q --fast --tb=short`
  - failed as expected before implementation because top-level public exports
    were missing and examples still imported `RunSequence` from `vercor.coupler`
- `conda run -n scipy pytest tests/test_api_boundaries.py -q --fast --tb=short`
  - passed after implementation
- `conda run -n scipy pytest tests/test_api_boundaries.py tests/test_runtime_state.py tests/test_component_base_coverage.py tests/test_coupler_coverage.py -q --fast --tb=short`
  - passed
- `conda run -n scipy black vercor examples tests`
  - passed
  - note: Black emitted the existing Python 3.13 vs target-3.14 safety-check
    warning and reformatted `tests/test_api_boundaries.py`
- `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`
  - passed (`0`)
- `conda run -n scipy mypy vercor examples tests`
  - passed (`96 source files`)
- `conda run -n scipy pytest tests/ -q --fast --tb=short`
  - passed
- `conda run -n scipy pytest tests/ -q --tb=short`
  - passed

## Runtime Context Boundary Cleanup

- Moved `ComponentInitContext` and `RuntimeStepContext` from
  `vercor.components.base` into the focused `vercor.runtime_contexts` module.
- Kept the runtime state architecture unchanged:
  - `RuntimeComponentState`, `RuntimeCouplerState`, and `RuntimeFieldStore`
    remain the integration-state containers
  - component runtime hook method names and signatures remain unchanged
  - `Coupler.run()` remains the single public runtime entrypoint
- Slimmed `vercor.components` to export only `Component` and
  `HostRuntimeComponent`; context types are now imported from
  `vercor.runtime_contexts`.
- Updated component, runtime-driver, coupler, and test imports to use the new
  context owner.
- Updated architecture coverage to guard the new boundary and removed one dead
  test-local assignment exposed by the lint pass.
- Updated `DEPENDENCIES.md` to document `vercor.runtime_contexts`.
- No failed implementation approaches. The TDD red check failed as expected
  before the new module existed; the first lint pass exposed a dead local in a
  touched test file, which was removed without changing runtime behavior.

## Validation (Runtime Context Boundary Cleanup, 2026-04-30)

- `conda run -n scipy pytest tests/test_runtime_state.py::test_runtime_module_does_not_own_component_specific_steps -q --fast --tb=short`
  - failed as expected before implementation because `vercor/runtime_contexts.py`
    did not exist
- `conda run -n scipy pytest tests/test_runtime_state.py tests/test_component_base_coverage.py tests/test_coupler_coverage.py -q --fast --tb=short`
  - passed
- `conda run -n scipy pytest tests/ -q --fast --tb=short`
  - passed
- `conda run -n scipy black vercor examples tests`
  - passed
  - note: Black emitted the existing Python 3.13 vs target-3.14 safety-check warning and reformatted `tests/test_external_components_coverage.py`
- `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`
  - passed (`0`)
- `conda run -n scipy mypy vercor examples tests`
  - passed
- `conda run -n scipy pytest tests/ -q --tb=short`
  - passed

# 2026-04-29

## Unified Coupler Runtime Entrypoint

- Removed the public `Coupler.compile_runtime()` API.
- Made `Coupler.run(initial_state=None, *, donate_state=False)` the single public
  runtime entrypoint:
  - pure differentiable component sets use a cached JIT-scanned runtime
  - host-backed component sets use the existing Python host bridge
  - `donate_state=True` is rejected for host-backed runs with a clear
    `CouplerError`
- Renamed the compile-cache coverage to runtime-run cache coverage and updated
  architecture tests to guard the removed public compile method.
- No `DEPENDENCIES.md` update was needed because module dependency order and
  runtime ownership boundaries did not change.
- No failed implementation approaches. The change was limited to API selection,
  private JIT caching, and tests.

## Validation (Unified Coupler Runtime Entrypoint, 2026-04-29)

- `conda run -n scipy pytest tests/test_runtime_run_cache.py tests/test_coupler_coverage.py tests/test_runtime_state.py -q --fast --tb=short`
  - passed
- `conda run -n scipy pytest tests/test_runtime_run_cache.py tests/test_coupler_coverage.py tests/test_runtime_state.py -q --tb=short`
  - passed before and after formatting
- `conda run -n scipy pytest tests/ -q --fast --tb=short`
  - passed
- `conda run -n scipy black vercor examples tests`
  - passed
  - note: Black emitted the existing Python 3.13 vs target-3.14 safety-check warning and reformatted `tests/test_coupler_coverage.py` and `tests/test_runtime_state.py`
- `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`
  - passed (`0`)
- `conda run -n scipy mypy vercor examples tests`
  - passed
- `conda run -n scipy pytest tests/ -q --tb=short`
  - passed

## Residual Compatibility Marker Cleanup

- Audited remaining `legacy`, `old_`, `compat`, `wrapper`, `deprecated`, and
  old runtime alias markers across `vercor`, `examples`, and `tests`.
- Removed the last test-local old component `step()` stub from runtime-state
  coverage; runtime send tests now use only the canonical runtime send helper.
- Renamed the removed-component-API regression test so the active suite no
  longer carries legacy/wrapper terminology in test names.
- Cleaned stale CAMulator bridge commentary while preserving the required
  host-backed `HostRuntimeComponent.step_host_runtime_state()` boundary.
- No `DEPENDENCIES.md` update was needed because module ownership and runtime
  dependency order did not change.
- No failed implementation approaches. Remaining marker hits are negative
  regression guards, current JAX/Python compatibility wording, or external
  CAMulator package names.

## Validation (Residual Compatibility Marker Cleanup, 2026-04-29)

- `conda run -n scipy pytest tests/ -q --fast --tb=short`
  - passed before edits
- `conda run -n scipy pytest tests/test_runtime_state.py tests/test_component_base_coverage.py tests/test_coupler_coverage.py tests/test_camulator_component_kernels.py tests/test_external_components_coverage.py -q --fast --tb=short`
  - passed
- `conda run -n scipy black vercor examples tests`
  - passed
  - note: Black emitted the existing Python 3.13 vs target-3.14 safety-check warning and left 94 files unchanged
- `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`
  - passed (`0`)
- `conda run -n scipy mypy vercor examples tests`
  - passed
- `conda run -n scipy pytest tests/ -q --fast --tb=short`
  - passed
- `conda run -n scipy pytest tests/ -q --tb=short`
  - passed

## Source Simplification Audit

- Consolidated runtime stepping into one explicit helper:
  - replaced the separate pure and host-enabled runtime step wrappers with
    `step_runtime_component(..., allow_host_runtime=...)`
  - kept CAMulator and Veros behind the required `HostRuntimeComponent` bridge
  - cached the runtime dispatch context once per `run()` / scanned-runtime call
    instead of rebuilding it for every component step
- Simplified `Regridder.__call__()` by removing scalar/vector wrapper methods
  and the handler dictionary; scalar and vector calls now dispatch directly.
- Updated architecture coverage to guard the single runtime helper, dispatch-context
  reuse, and the removed regridder wrapper methods.
- Updated `DEPENDENCIES.md` to document the single runtime step helper boundary.
- No failed implementation approaches. The cleanup removed internal seams without
  changing physics kernels, exchange behavior, or the required host adapters.

## Validation (Source Simplification Audit, 2026-04-29)

- `conda run -n scipy pytest tests/ -q --fast`
  - passed before edits
- `conda run -n scipy pytest tests/test_runtime_state.py tests/test_coupler_coverage.py tests/test_bilinear_rectilinear_regridder.py tests/test_conservative_rectilinear_regridder.py -q --fast --tb=short`
  - passed
- `conda run -n scipy black vercor examples tests`
  - passed
  - note: Black emitted the existing Python 3.13 vs target-3.14 safety-check warning and reformatted `vercor/runtime_driver.py`
- `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`
  - passed (`0`)
- `conda run -n scipy mypy vercor examples tests`
  - passed
- `conda run -n scipy pytest tests/ -q --fast --tb=short`
  - passed
- `conda run -n scipy pytest tests/ -q --tb=short`
  - passed

## Source Boundary Simplification

- Slimmed `vercor.components` to export only the base component contracts.
  Concrete examples/tests now import slab, data, and external model components
  from their owning modules, so optional Veros/CAMulator adapters are not loaded
  by importing `vercor.components`.
- Removed the thin `vercor.runtime_contracts` module:
  - moved exchange-field flattening, unique field appending, and runtime contract
    construction next to `RuntimeComponentContract` in `vercor.runtime`
  - updated architecture coverage to guard the deleted module and direct example imports
- Kept CAMulator and Veros as explicit host-runtime bridges while simplifying internals:
  - Veros host stepping now delegates forcing writes, substeps, and SST refresh to
    focused helpers
  - CAMulator host stepping now delegates output writes and prediction-to-runtime
    field mapping to focused helpers
  - CAMulator land now loads only CAMulator config/raw forcing and no longer
    imports the full CAMulator atmosphere adapter for land-surface temperature forcing
- Updated `DEPENDENCIES.md` to document runtime contract ownership in `vercor.runtime`
  and the forcing-only CAMulator land boundary.
- No failed implementation approaches. The cleanup stayed at ownership/import
  boundaries and host-adapter helper extraction; physics kernels and runtime behavior
  were not intentionally changed.

## Validation (Source Boundary Simplification, 2026-04-29)

- `conda run -n scipy pytest tests/test_runtime_state.py tests/test_component_base_coverage.py tests/test_tools_components_and_plotting.py tests/test_camulator_component_kernels.py tests/test_external_components_coverage.py -q --fast --tb=short`
  - passed
- `conda run -n scipy pytest tests/ -q --fast --tb=short`
  - passed
- `conda run -n scipy black vercor examples tests`
  - passed
  - note: Black emitted the existing Python 3.13 vs target-3.14 safety-check warning and left 94 files unchanged on the final run
- `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`
  - passed (`0`)
- `conda run -n scipy mypy vercor examples tests`
  - passed
- `conda run -n scipy pytest tests/ -q --tb=short`
  - passed

## Internal Runtime Compatibility Seam Cleanup

- Removed remaining implicit runtime-contract fallbacks:
  - runtime dispatch and coupler validation now use direct coupler-owned contract lookups
  - architecture coverage now guards against reintroducing `.get(..., RuntimeComponentContract())` fallbacks
- Trimmed the runtime field-store API:
  - deleted unused `RuntimeFieldStore.merge()`
  - kept `empty()`, `from_mapping()`, `get()`, and `set()` as the simple PyTree field-store surface
- Made Veros runtime configuration explicit:
  - replaced the wildcard runtime-settings side-effect import with `configure_veros_runtime()`
  - kept the configuration call before Veros setup imports while moving ordinary imports above it
- Simplified CAMulator bridge internals:
  - removed stale old-source header/commentary and unused `num_ts` / chunk-size state
  - added all host-written CAMulator data fields to the runtime-field initializer
- Updated `DEPENDENCIES.md` to document the explicit Veros runtime-settings boundary.
- The first flake8 pass exposed E402 warnings from the intentional Veros configure-before-import ordering; the final version keeps only the delayed Veros imports after configuration and marks those imports with targeted `noqa: E402`.

## Validation (Internal Runtime Compatibility Seam Cleanup, 2026-04-29)

- `conda run -n scipy pytest tests/test_runtime_state.py tests/test_camulator_component_kernels.py -q --fast --tb=short`
  - passed
- `conda run -n scipy black vercor examples tests`
  - passed
  - note: Black emitted the existing Python 3.13 vs target-3.14 safety-check warning; reformatted `tests/test_runtime_state.py`, then `vercor/components/external/veros_gcm.py` after the lint fix
- `conda run -n scipy pytest tests/ -q --fast --tb=short`
  - passed
- `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`
  - passed (`0`)
- `conda run -n scipy mypy vercor examples tests`
  - passed
- `conda run -n scipy pytest tests/ -q --tb=short`
  - passed

## Compatibility Boundary Simplification

- Moved forcing-file I/O out of `vercor.components.base`:
  - added `vercor.forcing_data.ComponentForcingData` as the focused NetCDF forcing read boundary
  - updated ERA5 / ERA-Interim data components and tests to import the forcing reader directly
  - removed the forcing-reader re-export from `vercor.components`
- Removed old atmosphere-ocean flux compatibility naming:
  - deleted `old_flux_atmOcn()`
  - renamed the canonical JAX implementation to `compute_ocean_surface_fluxes()`
  - updated Veros flux coupling and tests to use the canonical name
  - replaced old/new comparison coverage with fixed reference-value regression coverage
- Simplified component typing:
  - `Coupler` and grid-mask helpers now type against the base `Component` contract instead of concrete component unions
  - `vercor.types` now keeps only the shared runtime-array alias
- Removed the `RuntimeComponentContract.empty()` convenience constructor; call sites now use `RuntimeComponentContract()` directly.
- Added architecture regression coverage for the forcing-data boundary, removed legacy flux names, removed empty-contract helper, and preserved host-backed CAMulator/Veros bridge ownership.
- Updated `DEPENDENCIES.md` with `vercor.forcing_data` and the simplified coupler component contract.
- No failed implementation approaches. The cleanup stayed limited to compatibility surfaces and kept example-level `vercor.components` model imports stable.

## Validation (Compatibility Boundary Simplification, 2026-04-29)

- `conda run -n scipy pytest tests/test_component_base_coverage.py tests/test_component_models_coverage.py tests/test_fluxes_utilities.py tests/test_external_components_coverage.py tests/test_runtime_state.py -q --fast --tb=short`
  - passed
- `conda run -n scipy black vercor examples tests`
  - passed
  - note: Black emitted the existing Python 3.13 vs target-3.14 safety-check warning and left 95 files unchanged on the final run
- `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`
  - passed (`0`)
- `conda run -n scipy mypy vercor examples tests`
  - passed
- `conda run -n scipy pytest tests/ -q --fast --tb=short`
  - passed
- `conda run -n scipy pytest tests/ -q --tb=short`
  - passed

# 2026-04-28

## Runtime Utility Ownership Simplification

- Split the mixed `vercor.tools` module into focused ownership boundaries:
  - `vercor.assets` for forcing asset resolution/download/checksum validation
  - `vercor.diagnostics` for runtime-view tables and plotting
  - `vercor.host_arrays` for explicit JAX-to-host array transfer
  - `vercor.time_selection` for calendar/time-slice interpolation helpers
  - `vercor.grid_masks` for grid lookup, land/ocean mask creation, and remap checks
- Deleted `vercor/tools.py` instead of keeping a compatibility re-export layer.
- Made the host runtime bridge public:
  - `HostRuntimeComponent.step_host_runtime_state()` is now the explicit CAMulator/Veros bridge
  - runtime dispatch still selects it only after `isinstance(component, HostRuntimeComponent)`
- Removed implicit empty-contract compatibility in generic runtime helpers:
  - runtime helper functions now require a concrete `RuntimeComponentContract`
  - tests use `RuntimeComponentContract.empty()` where an empty contract is intentional
- Updated architecture regression coverage for the deleted tools module, public host bridge, and explicit runtime contracts.
- Updated `DEPENDENCIES.md` with focused utility modules and the public host bridge boundary.
- No failed implementation approaches. The cleanup stayed mechanical after the initial module split and preserved the canonical runtime state interfaces.

## Validation (Runtime Utility Ownership Simplification, 2026-04-28)

- `conda run -n scipy pytest tests/ -q --fast --tb=short`
  - passed
- `conda run -n scipy black vercor examples tests`
  - passed
  - note: Black emitted the existing Python 3.13 vs target-3.14 safety-check warning and left 94 files unchanged
- `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`
  - passed (`0`)
- `conda run -n scipy mypy vercor examples tests`
  - passed
- `conda run -n scipy pytest tests/ -q --tb=short`
  - passed

## Runtime Bridge Boundary Simplification

- Removed redundant runtime contract construction over component objects:
  - `Coupler` now builds contracts directly from component-name order
  - deleted the `build_runtime_contracts_for_components()` wrapper
- Reduced repeated runtime driver plumbing:
  - added immutable `RuntimeDispatchContext` for static dispatch inputs
  - pure, host-enabled, and outgoing-prime runtime helpers now receive one dispatch context
- Removed stale compatibility surfaces:
  - deleted unused `RuntimeFieldStore.subset()` and `RuntimeFieldStore.to_mapping()`
  - deleted unused deprecated CAMulator wind post-processing wrapper
- Consolidated JAXGCM runtime field ownership:
  - added one set of runtime field constants/helpers for initialization, prefill, and validation
- Added architecture regression coverage so redundant contract wrappers, mapping-style field-store helpers, and deprecated wind wrapper are not reintroduced.
- Updated `DEPENDENCIES.md` to document the name-sequence contract builder and dispatch context.
- Note: mypy initially flagged the JAXGCM default-field helper dict inference; fixed with an explicit `dict[str, RuntimeArray]` annotation.

## Validation (Runtime Bridge Boundary Simplification, 2026-04-28)

- `conda run -n scipy pytest tests/test_runtime_state.py tests/test_coupler_coverage.py tests/test_coupler_runtime.py tests/test_external_tools_coverage.py tests/test_external_components_coverage.py -q --fast --tb=short`
  - passed
- `conda run -n scipy pytest tests/ -q --fast --tb=short`
  - passed
- `conda run -n scipy black vercor examples tests`
  - passed
  - note: Black emitted the existing Python 3.13 vs target-3.14 safety-check warning and left 90 files unchanged on the final run
- `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`
  - passed (`0`)
- `conda run -n scipy mypy vercor examples tests`
  - passed
- `conda run -n scipy pytest tests/ -q --tb=short`
  - passed

## Runtime Boundary Helper Refactor

- Extracted remaining runtime-only orchestration out of `Coupler`:
  - added `vercor.runtime_contracts` for exchange-field flattening and runtime contract construction
  - added `vercor.runtime_time` for daily/monthly runtime step metadata
  - added `vercor.runtime_driver` for outgoing priming, per-component dispatch/receive/step/send, and host-adapter detection
- Kept public orchestration stable:
  - `Coupler` remains the facade for registration, initialization, runtime-state creation, `run()`, `compile_runtime()`, runtime views, and final output
  - scanned runtime still rejects CAMulator/Veros host-backed adapters before JIT execution
  - CAMulator and Veros still use the explicit `HostRuntimeComponent` bridge
- Tightened diagnostics around runtime views:
  - plotting/table helpers now consume `RuntimeComponentView` instead of probing arbitrary component/runtime objects
  - examples now pass `cpl.runtime_component_view(...)` into diagnostics helpers
- Added architecture regression coverage for extracted runtime boundaries, explicit host dispatch ownership, and runtime-view-only diagnostics.
- Updated `DEPENDENCIES.md` with the new runtime helper modules.

## Validation (Runtime Boundary Helper Refactor, 2026-04-28)

- `conda run -n scipy pytest tests/test_runtime_state.py tests/test_tools_components_and_plotting.py tests/test_component_base_coverage.py tests/test_coupler_coverage.py tests/test_coupler_runtime.py -q --fast --tb=short`
  - passed
- `conda run -n scipy pytest tests/ -q --fast --tb=short`
  - passed
- `conda run -n scipy black vercor examples tests`
  - passed
  - note: Black emitted the existing Python 3.13 vs target-3.14 safety-check warning and reformatted 4 files
- `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`
  - passed (`0`)
- `conda run -n scipy mypy vercor examples tests`
  - passed
- `conda run -n scipy pytest tests/ -q --tb=short`
  - passed

## Internal Runtime Responsibility Cleanup

- Moved generic runtime/component operations out of `Component`:
  - added `vercor.runtime_components` for runtime state creation, contract prefill/validation, receive/send, and monthly/daily send selection
  - `Component` now keeps only seed data plus component-specific init, prefill, validation, and step hooks
  - concrete components now use runtime helper validation for component-specific required fields instead of calling generic base validation
- Kept the public model import surface stable:
  - `vercor.components` still exports component classes and base context/bridge types
  - `Coupler` imports base contracts directly from `vercor.components.base`
- Preserved the explicit non-differentiable bridge:
  - CAMulator atmosphere, CAMulator land, and Veros remain `HostRuntimeComponent` adapters
  - scanned/compiled runtime still rejects host-backed adapters
- Added/updated architecture regression coverage so generic runtime helpers stay out of `Component` and compatibility APIs remain absent.
- Updated `DEPENDENCIES.md` to document the new runtime-components helper layer.

## Validation (Internal Runtime Responsibility Cleanup, 2026-04-28)

- `conda run -n scipy pytest tests/test_component_base_coverage.py tests/test_runtime_state.py tests/test_coupler_coverage.py tests/test_coupler_runtime.py tests/test_runtime_compile_cache.py -q --fast --tb=short`
  - passed
- `conda run -n scipy pytest tests/test_external_components_coverage.py tests/test_camulator_component_kernels.py tests/test_component_models_coverage.py -q --fast --tb=short`
  - passed
- `conda run -n scipy pytest tests/ -q --fast --tb=short`
  - passed
- `conda run -n scipy black vercor examples tests`
  - passed
  - note: Black emitted the existing Python 3.13 vs target-3.14 safety-check warning and left 87 files unchanged
- `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`
  - passed (`0`)
- `conda run -n scipy mypy vercor examples tests`
  - passed
- `conda run -n scipy pytest tests/ -q --tb=short`
  - passed

## Targeted Runtime Boundary Cleanup

- Replaced remaining broad component runtime signatures with explicit context objects:
  - added immutable `ComponentInitContext` and `RuntimeStepContext`
  - `Coupler.initialize()` now passes only start time, timestep, run sequence, settings, and logger to components
  - runtime stepping now passes one context object to pure and host-backed component step boundaries
- Kept non-differentiable host bridges explicit:
  - CAMulator atmosphere, CAMulator land, and Veros still inherit `HostRuntimeComponent`
  - host stepping remains restricted to `Coupler.run()`; scanned/compiled runtime still rejects host-backed components
- Removed residual duplicated metadata:
  - runtime NetCDF output private writer now reads component name/grid from `RuntimeComponentView`
  - data component constructors no longer assign `self.grid` before base initialization when a local grid is sufficient
  - component modules prefer direct `vercor.components.base` imports for base contracts
- Added architecture regression coverage for context-based signatures and removed duplicated writer metadata.
- Updated `DEPENDENCIES.md` to document the context boundary.

## Validation (Targeted Runtime Boundary Cleanup, 2026-04-28)

- `conda run -n scipy pytest tests/test_component_base_coverage.py tests/test_component_models_coverage.py tests/test_coupler_coverage.py tests/test_runtime_state.py tests/test_external_components_coverage.py tests/test_camulator_component_kernels.py -q --fast --tb=short`
  - passed
- `conda run -n scipy pytest tests/ -q --fast --tb=short`
  - passed
- `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`
  - passed (`0`)
- `conda run -n scipy black vercor examples tests`
  - passed
  - note: Black emitted the existing Python 3.13 vs target-3.14 safety-check warning and left 86 files unchanged on the final run
- `conda run -n scipy mypy vercor examples tests`
  - passed
- `conda run -n scipy pytest tests/ -q --tb=short`
  - passed

## Runtime Component Contract Cleanup

- Moved runtime import/export field ownership out of component instances:
  - added immutable `RuntimeComponentContract` metadata in `vercor.runtime`
  - `Coupler` now derives per-component contracts from exchanges
  - `Component` no longer stores `_fields2import` / `_fields2export`
- Made `Component.data` a seed-state surface:
  - runtime state creation copies seed fields into `RuntimeComponentState`
  - runtime receive/send/validation now use explicit coupler-owned contracts
  - tests assert production code does not reintroduce component-owned private field lists
- Kept non-differentiable host bridges explicit:
  - CAMulator and Veros remain `HostRuntimeComponent` adapters
  - host stepping still receives runtime state, settings, time, and logger only
- Updated `DEPENDENCIES.md` to document contract ownership and seed-state responsibilities.

## Validation (Runtime Component Contract Cleanup, 2026-04-28)

- `conda run -n scipy pytest tests/ -q --fast --tb=short`
  - passed
- `conda run -n scipy black vercor examples tests`
  - passed
  - note: Black emitted the existing Python 3.13 vs target-3.14 safety-check warning and left 86 files unchanged on the final run
- `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`
  - passed (`0`)
- `conda run -n scipy mypy vercor examples tests`
  - passed
- `conda run -n scipy pytest tests/ -q --tb=short`
  - passed

## Runtime Bridge Compatibility Surface Simplification

- Removed remaining compatibility aliases:
  - runtime output writers are no longer re-exported from `vercor.components` or `vercor.components.base`
  - `RuntimeComponentView` is no longer re-exported from `vercor.tools`
  - tests now import runtime output and view helpers from `vercor.output` and `vercor.runtime_views`
- Simplified runtime diagnostics/output:
  - `RuntimeComponentView` now stores typed `RuntimeFieldStore` objects
  - view construction now takes explicit `name`, `grid`, and `RuntimeComponentState`
  - runtime NetCDF output iterates field stores directly instead of converting stores back to dictionaries
- Simplified component stepping context:
  - `Component.step_runtime_state()` and `HostRuntimeComponent._step_host_runtime_state()` no longer receive the full coupler object
  - runtime stepping passes settings, optional time, and optional logger only
  - CAMulator, CAMulator land, Veros, JAXGCM, slab components, and tests were updated to the slimmer signature
- Tightened runtime-state authority in tests by removing a stale helper that copied stepped runtime fields back into `component.data`.
- Added architecture regressions for removed aliases and host bridge signatures.

## Validation (Runtime Bridge Compatibility Surface Simplification, 2026-04-28)

- `conda run -n scipy pytest tests/test_runtime_state.py tests/test_component_base_coverage.py tests/test_coupler_coverage.py tests/test_coupler_runtime.py tests/test_external_components_coverage.py tests/test_tools_components_and_plotting.py -q --fast --tb=short`
  - passed
- `conda run -n scipy pytest tests/ -q --fast --tb=short`
  - passed
- `conda run -n scipy black vercor examples tests`
  - passed
  - note: Black emitted the existing Python 3.13 vs target-3.14 safety-check warning and reformatted 2 test files
- `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`
  - passed (`0`)
- `conda run -n scipy mypy vercor examples tests`
  - passed
- `conda run -n scipy pytest tests/ -q --tb=short`
  - passed

## Residual Runtime Bridge Cleanup

- Split remaining mixed runtime responsibilities into focused modules:
  - added `RuntimeComponentView` in `vercor/runtime_views.py`
  - moved runtime NetCDF writers to `vercor/output.py`
  - kept compatibility imports from `vercor.tools`, `vercor.components`, and `vercor.components.base`
  - `Component` no longer owns runtime output writer implementations
- Tightened host-backed bridge data flow:
  - CAMulator atmosphere and land update `RuntimeFieldStore` directly instead of round-tripping through dictionaries
  - Veros flux coupling now consumes `RuntimeFieldStore` directly
  - host adapters still isolate non-differentiable CAMulator/Veros mutation behind `HostRuntimeComponent`
- Updated examples to request runtime views through `Coupler.runtime_component_view()`.
- Added architecture regressions for focused runtime-view/output ownership, host adapter store access, and example view construction.

## Validation (Residual Runtime Bridge Cleanup, 2026-04-28)

- `conda run -n scipy pytest tests/test_runtime_state.py tests/test_component_base_coverage.py tests/test_coupler_coverage.py tests/test_external_components_coverage.py tests/test_camulator_component_kernels.py tests/test_tools_components_and_plotting.py -q --fast --tb=short`
  - passed
- `conda run -n scipy pytest tests/ -q --fast --tb=short`
  - passed
- `conda run -n scipy black vercor examples tests`
  - passed
  - note: Black emitted the existing Python 3.13 vs target-3.14 safety-check warning and left 86 files unchanged on the final run
- `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`
  - passed (`0`)
- `conda run -n scipy mypy vercor examples tests`
  - passed
- `conda run -n scipy pytest tests/ -q --tb=short`
  - passed

## Incremental Runtime Bridge Simplification

- Made the non-differentiable host boundary explicit:
  - added `HostRuntimeComponent` for adapters that require Python/host-side mutation
  - Veros, CAMulator atmosphere, and CAMulator land now inherit that contract
  - `Coupler` dispatches host steps with `isinstance(..., HostRuntimeComponent)` instead of introspecting for private methods
  - `compile_runtime()` now raises a clear `CouplerError` when host-backed adapters are registered
- Removed the base runtime-to-component sync compatibility hook:
  - deleted `Component._sync_data_from_runtime_state()`
  - host adapters read from `RuntimeComponentState.data` mappings directly
  - CAMulator and CAMulator land return updated runtime states without copying runtime fields back into `self.data`
  - Veros flux coupling now receives explicit Veros state and runtime fields instead of the whole component instance
- Consolidated runtime diagnostic/output views:
  - replaced plotting-side `ComponentFieldView` usage with `RuntimeComponentView`
  - added `Coupler.runtime_component_view()` and runtime-view NetCDF output
  - examples now request views from the coupler instead of manually pairing component objects with runtime states
- Updated `DEPENDENCIES.md` to document the host-runtime contract and runtime-view coupler responsibilities.

## Validation (Incremental Runtime Bridge Simplification, 2026-04-28)

- `conda run -n scipy pytest tests/test_component_base_coverage.py tests/test_coupler_coverage.py tests/test_runtime_state.py tests/test_external_tools_coverage.py tests/test_external_components_coverage.py tests/test_camulator_component_kernels.py tests/test_tools_components_and_plotting.py -q --fast --tb=short`
  - passed
- `conda run -n scipy pytest tests/test_external_components_coverage.py -q --fast --tb=short`
  - passed
- `conda run -n scipy pytest tests/ -q --fast --tb=short`
  - passed
- `conda run -n scipy black vercor examples tests`
  - passed
  - note: Black emitted the existing Python 3.13 vs target-3.14 safety-check warning
- `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`
  - passed (`0`)
- `conda run -n scipy mypy vercor examples tests`
  - passed
- `conda run -n scipy pytest tests/ -q --tb=short`
  - passed

## Notes / Failed Approaches

- Removing the Veros runtime-state refresh from `component.data` exposed stale tests that still asserted on adapter-owned mutable storage. The tests now assert on the returned runtime state, matching the new runtime-state authority.

## Runtime Metadata De-Compatibility Cleanup

- Removed duplicated component metadata from `RuntimeComponentState`:
  - component names now live once in `RuntimeCouplerState.component_names`
  - import/export field lists now stay on registered component instances
  - runtime state helpers now replace component states by explicit name
- Kept non-differentiable host bridges explicit and private:
  - Veros, CAMulator atmosphere, and CAMulator land now expose `_step_host_runtime_state()`
  - scanned runtime continues through pure `step_runtime_state()` and does not execute those host mutations
- Replaced loose plotting `(component, runtime_state)` tuples with `ComponentFieldView`.
- Updated final runtime NetCDF output metadata so `write_runtime_component_to_netcdf()` receives the component name explicitly.
- Updated `DEPENDENCIES.md` to describe the new runtime/base/coupler responsibility split.

## Validation (Runtime Metadata De-Compatibility Cleanup, 2026-04-28)

- `conda run -n scipy pytest tests/test_runtime_state.py tests/test_component_base_coverage.py tests/test_runtime_exchange.py tests/test_coupler_runtime.py -q --fast --tb=short`
  - passed
- `conda run -n scipy pytest tests/test_external_components_coverage.py tests/test_camulator_component_kernels.py tests/test_tools_components_and_plotting.py -q --fast --tb=short`
  - passed
- `conda run -n scipy pytest tests/ -q --fast --tb=short`
  - passed
- `conda run -n scipy black vercor examples tests`
  - passed
  - note: Black emitted the existing Python 3.13 vs target-3.14 safety-check warning and left 84 files unchanged on the final run
- `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`
  - passed (`0`)
- `conda run -n scipy mypy vercor examples tests`
  - passed
- `conda run -n scipy pytest tests/ -q --tb=short`
  - passed

## Notes / Failed Approaches

- The first full-suite run after moving import/export metadata off `RuntimeComponentState` exposed manual slab runtime fixtures whose registered components did not own equivalent field metadata. The fix updated those fixtures to declare component-owned import/export lists instead of reintroducing runtime-state duplication.

## Data Driver Runtime-State Plotting Fix

- Fixed `examples/run_data_driver.py` plotting after the runtime API de-compatibility refactor:
  - plot rows use explicit component/runtime-state views rather than stale component object fields
  - plotting now reads `total_surface_temperature` from the returned runtime state instead of the stale component object
- Hardened component plotting helpers for runtime states with multiple stores:
  - plotting still preserves normal field lookup order for scalar/table helpers
  - plot-specific lookup now selects a 2D field when one is available, which handles ERA5 fields where monthly 3D forcing remains in `data` and the runtime-selected 2D field is in `outgoing`
- Added regression coverage for explicit runtime-state plotting views with 3D data-store winds and 2D outgoing winds.
- Commit: `daf9382edf7eaf20d1f3e7156e08e0ed43575013`

## Validation (Data Driver Runtime-State Plotting Fix, 2026-04-28)

- `conda run -n scipy pytest tests/test_tools_components_and_plotting.py -q --fast --tb=short`
  - passed
- `conda run -n scipy pytest tests/ -q --fast --tb=short`
  - passed
- `conda run -n scipy black vercor examples tests`
  - passed
  - note: Black emitted the existing Python 3.13 vs target-3.14 safety-check warning and left 84 files unchanged
- `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`
  - passed (`0`)
- `conda run -n scipy mypy vercor examples tests`
  - passed
- `conda run -n scipy pytest tests/ -q --tb=short`
  - passed
- `PYTHONPATH=/Users/romannuterman/Documents/Science/scodes/Python/VerCOR MPLBACKEND=Agg MPLCONFIGDIR=/private/tmp/vercor-mpl XDG_CACHE_HOME=/private/tmp/vercor-cache conda run -n scipy python /Users/romannuterman/Documents/Science/scodes/Python/VerCOR/examples/run_data_driver.py`
  - passed from `/private/tmp/vercor-run-data-driver-smoke`

## Notes / Failed Approaches

- The first smoke test after switching `run_data_driver.py` to runtime-state plotting passed the original missing-field point but failed in Matplotlib quiver because ERA5 atmosphere `u_velocity` / `v_velocity` in runtime `data` are monthly 3D arrays. The final fix keeps the example on runtime states and teaches plotting to prefer plottable 2D runtime fields when available.

## Runtime API De-Compatibility Refactor

- Removed the legacy component wrapper compatibility surface from production code:
  - deleted public `Shared`, `TimedNamedArray`, `write_shared_to_netcdf`, `Component.step()`, wrapper field attributes, wrapper commits, and wrapper merge/get helpers
  - removed `Shared` / `TimedNamedArray` from `vercor.components` exports
  - made `RuntimeFieldStore`, `RuntimeComponentState`, and `RuntimeCouplerState` the only integration-state containers
- Simplified runtime execution:
  - `Coupler.run()` now always returns a runtime state and no longer accepts `commit_wrappers`
  - `Coupler.compile_runtime()` calls the scanned runtime directly
  - wrapper initialization checks and runtime commit side effects were removed from the coupler loop
- Replaced wrapper output with runtime-state output:
  - added `write_runtime_component_to_netcdf()` for final incoming/outgoing runtime fields and masks
  - changed `Coupler.finalize(final_state, ...)` to write outputs from the returned runtime state
- Kept host-backed adapter synchronization explicit and private:
  - added `Component._sync_data_from_runtime_state()`
  - updated Veros, CAMulator, and CAMulatorLand to synchronize only their adapter-owned mutable `data` before host-side stepping
- Updated examples, plotting/table helpers, and tests to read from runtime states instead of wrapper fields.
- `DEPENDENCIES.md` did not require an update because the output writer stayed in the existing component base module and no dependency order changed.

## Tests Added / Updated

- Replaced wrapper coverage with runtime-state regression coverage in component/coupler tests.
- Added assertions that `Shared`, `TimedNamedArray`, and `write_shared_to_netcdf` are no longer public APIs.
- Added runtime NetCDF writer coverage for JAX-backed incoming/outgoing fields and mask fields.
- Updated adapter tests to call `step_runtime_state()` with explicit runtime states.
- Updated example helper tests to use `RuntimeComponentState`.

## Validation (Runtime API De-Compatibility Refactor, 2026-04-28)

- `conda run -n scipy pytest tests/test_component_base_coverage.py tests/test_coupler_coverage.py tests/test_coupler_runtime.py tests/test_external_components_coverage.py tests/test_camulator_component_kernels.py -q --tb=short`
  - passed
- `conda run -n scipy pytest tests/test_component_models_coverage.py tests/test_tools_components_and_plotting.py tests/test_component_base_coverage.py tests/test_coupler_coverage.py -q --tb=short`
  - passed
- `conda run -n scipy pytest tests/ -q --fast --tb=short`
  - passed
- `conda run -n scipy black vercor examples tests`
  - passed
  - note: Black emitted the existing Python 3.13 vs target-3.14 safety-check warning and left 84 files unchanged on the final run
- `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`
  - passed (`0`)
- `conda run -n scipy mypy vercor examples tests`
  - passed
- `conda run -n scipy pytest tests/ -q --tb=short`
  - passed

## Notes / Failed Approaches

- The first focused test run exposed stale test fixtures that still constructed wrapper fields by hand; those were replaced with explicit runtime states.
- The first full-suite run exposed one example helper test still passing a fake wrapper-style component; the helper test now uses `RuntimeComponentState`.

# 2026-04-23

## Coupler / Veros / Clock Coverage Expansion

- Extended `tests/test_coupler_coverage.py` to cover the remaining in-scope coupler control flow:
  - `initialize()` happy path with `ATM`, `OCN`, `LND`, and `ICE`
  - duplicate regridder creation warning path
  - `enable_x64_computations` override with patched `jax.config.update()`
  - `_create_exchange_masks()` failure branches for mismatched land/atmosphere grids and missing ocean masks
  - `finalize()`, `__str__`, `__repr__`, and `run()` happy path ordering
- Extended `tests/test_external_components_coverage.py` to cover more unit-testable `vercor/components/external/veros_gcm.py` helpers:
  - `compute_fluxes()` `qnec` zeroing branch for sentinel `dqfldt`
  - `CustomGlobalFourDegree.set_diagnostics()` via the undecorated Veros routine function
  - `copy_state()` jitted deep-copy path

## First JAX Translation Slice: Flux Kernels

- Completed the first incremental NumPy-to-JAX translation slice without changing the public `Coupler` / `Component` API.
- Translated `vercor/fluxes/utilities.py` to JAX-native array math:
  - inputs are coerced with `jnp.asarray`
  - NumPy-only ops were replaced with `jax.numpy`
  - the ECMWF hybrid-level altitude helper now uses JAX-safe padding instead of `np.insert`
- Rewrote `vercor/fluxes/bulk_formula_cesm.py` as JAX-native kernels:
  - `old_flux_atmOcn()`
  - `new_flux_atmOcn()`
  - `shr_flux_atmIce()`
  - shared stability / exchange-coefficient logic was factored into internal helpers to keep the two ocean schemes numerically aligned
- Made `new_flux_atmOcn()` compatible with `jax.jit` and reverse-mode AD by replacing the dynamic `lax.while_loop` attempt with a fixed two-step `lax.fori_loop` using masked carry updates.
- Tightened direct boundary adapters so JAX kernels run internally and NumPy conversion happens only where the external runtimes need it:
  - `vercor/components/external/veros_gcm.py`
  - `vercor/components/external/camulator.py`
  - `vercor/components/external/jax_gcm.py`

## Tests Added / Updated

- Extended `tests/test_fluxes_utilities.py` with:
  - `jax.jit` coverage for the translated utility kernels
  - `jax.jit` coverage for `new_flux_atmOcn()` and `shr_flux_atmIce()`
  - a finite-difference gradient smoke test for `new_flux_atmOcn()` sensible heat with respect to sea-surface temperature
- Extended `tests/test_external_tools_coverage.py` so `vercor/components/external/jax_gcm_tools.compute_pressure_levels()` is exercised under `jax.jit`.

## Validation (Flux Translation Slice, 2026-04-23)

- `conda run -n scipy black vercor examples tests`
  - passed
- `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`
  - passed (`0`)
- `conda run -n scipy mypy vercor examples tests`
  - passed
- `conda run -n scipy pytest tests/test_component_models_coverage.py tests/test_external_components_coverage.py tests/test_external_tools_coverage.py tests/test_fluxes_utilities.py -q`
  - passed
- `conda run -n scipy pytest tests/ -q --fast`
  - passed
- `conda run -n scipy pytest tests/ -q`
  - passed

## Failed Approaches / Notes

- An initial `lax.while_loop` implementation for `new_flux_atmOcn()` was `jax.jit`-compatible but failed reverse-mode AD with:
  - `ValueError: Reverse-mode differentiation does not work for lax.while_loop ...`
- The final implementation uses a static two-iteration `lax.fori_loop`, which matches the current convergence limit and keeps gradients available.

## Next Translation Targets

- Second slice still pending:
  - `vercor/grid.py`
  - `vercor/regridders/helpers.py`
  - bilinear / conservative interpolation math
- Third slice still pending:
  - slab component pure kernels in `vercor/components/slab/`
  - `pure()` copy-before-mutate behavior
  - `set_variable()` interior update path
- Extended `tests/test_clock.py` for uncovered calendar helpers:
  - `isoformat()` and `timetuple()`
  - missing `day_of_year` validation in `timetuple()`
  - invalid `day_of_year` handling in `_month_day_from_day_of_year()`
  - negative ordinal overflow in `_from_ordinal_microseconds()`
  - `Clock.days_per_year` and `Clock.fixed_30_day_months` properties
- No production code changes were required; the work stayed in tests only.

## Coverage Outcome

- Overall `vercor` coverage increased from `73%` to `76%` via `conda run -n scipy pytest tests/ --cov=vercor --cov-report=term-missing -q`.
- Module-level gains from the coverage run:
  - `vercor/coupler.py`: `67%` -> `95%`
  - `vercor/components/external/veros_gcm.py`: `57%` -> `73%`
  - `vercor/clock.py`: `81%` -> `86%`
  - `vercor/components/base.py`: `85%` -> `86%`
- The main remaining misses in `vercor/components/external/veros_gcm.py` are the heavy Veros kernel/setup regions (`set_forcing_kernel()` and `__init__()`), which are intentionally not exercised as real integrations in these unit tests.

## Validation (2026-04-23)

- `conda run -n scipy pytest tests/test_external_components_coverage.py -q`
  - passed
- `conda run -n scipy pytest tests/test_coupler_coverage.py -q`
  - passed
- `conda run -n scipy pytest tests/test_clock.py -q`
  - passed
- `conda run -n scipy black vercor examples tests`
  - passed
- `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`
  - passed (`0`)
- `conda run -n scipy mypy vercor examples tests`
  - passed
- `conda run -n scipy pytest tests/ -q --fast`
  - passed
- `conda run -n scipy pytest tests/ -q`
  - passed
- `conda run -n scipy pytest tests/ --cov=vercor --cov-report=term-missing -q`
  - passed

## Notes / Failed Approaches

- `CustomGlobalFourDegree.set_diagnostics()` is wrapped by Veros runtime validation, so calling it directly in a unit test raises a `TypeError` unless the argument is a real `VerosState`. The test now calls the underlying routine function instead of going through the runtime wrapper.
- Per task instructions, no coverage work was added for:
  - `vercor/components/data/camulator_land.py`
  - `vercor/components/external/camulator.py`
  - `vercor/components/external/camulator_state.py`
  - `vercor/components/external/windpp.py`

## Second JAX Translation Slice 2A: Grid and Bilinear Regridding

- Completed the bilinear-first second translation slice without changing the public construction patterns used by components, examples, or tests.
- Translated `vercor/grid.py` to JAX-friendly grid holders:
  - `RectilinearGrid` now stores JAX arrays internally
  - eager validation for mask dimensionality and strict coordinate monotonicity is preserved
  - the legacy compact `__repr__` / `__str__` behavior is preserved
  - `RectilinearGrid` is now registered as a JAX PyTree
- Translated `vercor/regridders/helpers.py` to JAX-native helper kernels:
  - `make_rectilinear_grid()`
  - `centers_to_edges()`
  - `compute_land_mask()`
  - longitude clamping vs periodic-overhang behavior is preserved under `jax.jit`
- Rewrote `vercor/interpolators/bilinear_rectilinear.py` as a JAX-native interpolator:
  - all geometry helpers now use `jax.numpy`
  - scalar and vector apply paths are `jax.jit`-safe
  - extrapolation now uses JAX array operations instead of Python loops
  - periodic longitude, descending latitude, NaN renormalization, nearest / IDW fallback, and vector rotation behavior remain covered by the existing tests
  - the interpolator is registered as a JAX PyTree
- Kept the public regridder API stable:
  - `vercor/regridders/base.py` still dispatches scalar vs vector calls the same way
  - `vercor/regridders/bilinear.py` required no behavioral change
  - the conservative SciPy-backed remapper remains pending for the next slice

## Tests Added / Updated (Slice 2A)

- Extended `tests/test_helpers_coverage.py` with:
  - `RectilinearGrid` PyTree round-trip coverage
  - `jax.jit` coverage for `centers_to_edges()` and `compute_land_mask()`
- Extended `tests/test_bilinear_rectilinear_interpolator.py` with:
  - interpolator PyTree round-trip coverage
  - `jax.jit` coverage for scalar and vector interpolation
  - a gradient smoke test for scalar interpolation with respect to source field values
- Extended `tests/test_bilinear_rectilinear_regridder.py` so the bilinear regridder is exercised with JAX array input

## Validation (Slice 2A, 2026-04-23)

- `conda run -n scipy black vercor examples tests`
  - passed
- `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`
  - passed (`0`)
- `conda run -n scipy mypy vercor examples tests`
  - passed
- `conda run -n scipy pytest tests/test_helpers_coverage.py tests/test_bilinear_rectilinear_interpolator.py tests/test_bilinear_rectilinear_regridder.py -q`
  - passed
- `conda run -n scipy pytest tests/ -q --fast`
  - passed
- `conda run -n scipy pytest tests/ -q`
  - passed

## Failed Approaches / Notes (Slice 2A)

- Narrowing public signatures all the way to `jax.Array` broke `mypy` across the existing mixed NumPy/JAX call sites. The final version keeps the public interfaces NumPy-compatible while normalizing to JAX arrays internally.
- The conservative remapper rewrite remains intentionally deferred because its SciPy sparse representation needs a separate JAX-native design pass.

## Second JAX Translation Slice 2B: Conservative Remapping

- Replaced the conservative remapper runtime path in `vercor/interpolators/conservative_remap_rectilinear.py`:
  - removed the SciPy sparse dependency from runtime application
  - precompute now builds eager overlap triplets `(dst_index, src_index, weight)` in Python
  - scalar application now uses only `jax.numpy` gathers and indexed reductions
  - `src_lon_b`, `src_lat_b`, `dst_lon_b`, `dst_lat_b`, `dst_areas`, normalization semantics, periodic longitude handling, descending-latitude handling, source masking, and NaN behavior were preserved
  - `ConservativeRectilinearRemapper` is now registered as a JAX PyTree
- Kept the public conservative wrapper API stable in `vercor/regridders/conservative.py`; no constructor or call signatures changed.
- Extended conservative tests:
  - `tests/test_conservative_rectilinear_remapper.py`
    - PyTree round-trip coverage
    - `jax.jit` execution coverage for `apply_scalar()`
    - linearity + reverse-mode gradient smoke test with respect to the source field
  - `tests/test_conservative_rectilinear_regridder.py`
    - JAX-array input coverage through the public regridder call path

## Validation (Slice 2B Conservative Remapping, 2026-04-23)

- `conda run -n scipy black vercor examples tests`
  - passed
  - note: Black emitted the existing Python 3.13 vs target-3.14 safety-check warning but completed successfully
- `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`
  - passed (`0`)
- `conda run -n scipy mypy vercor examples tests`
  - passed
- `conda run -n scipy pytest tests/test_conservative_rectilinear_remapper.py -q`
  - passed
- `conda run -n scipy pytest tests/test_conservative_rectilinear_regridder.py -q`
  - passed
- `conda run -n scipy pytest tests/ -q --fast`
  - passed
- `conda run -n scipy pytest tests/ -q`
  - passed

## Third JAX Translation Slice 3A: Slab Pure Kernels

- Translated the slab component compute paths to JAX while keeping the current component wrapper API unchanged:
  - `vercor/components/slab/atmosphere.py`
    - extracted pure JAX helpers for default SST, bulk-flux update, and 10 m wind construction
    - `initialize()` now seeds fields with `jnp.full` / `jnp.zeros`
    - `step()` now computes through JAX kernels and writes the results back to `self.data`
  - `vercor/components/slab/ocean.py`
    - extracted a pure JAX SST update kernel from sensible + latent heat fluxes, restoring, and `dt_seconds`
    - `initialize()` now seeds SST with `jnp.full`
  - `vercor/components/slab/land.py`
    - extracted a pure JAX soil-moisture update kernel using `jnp.clip`
    - `initialize()` now seeds soil moisture and land temperature with `jnp.full`
  - `vercor/components/slab/seaice.py`
    - extracted a pure JAX logistic ice-fraction diagnostic using `jnp.exp`
    - `initialize()` now seeds ice fraction with `jnp.zeros`
- Added dedicated slab-kernel tests in `tests/test_slab_kernels.py`:
  - `jax.jit` coverage for every new pure kernel
  - gradient smoke tests for atmosphere, ocean, land, and sea-ice kernels
  - edge cases for default SST, clipping, and cold-versus-warm sea-ice response
- Trimmed the slab portion of `tests/test_component_models_coverage.py` so it remains focused on wrapper-level initialization and dispatch behavior rather than duplicating all kernel math checks.

## Validation (Slice 3A Slab Kernels, 2026-04-23)

- `conda run -n scipy black vercor examples tests`
  - passed
  - note: Black emitted the existing Python 3.13 vs target-3.14 safety-check warning but completed successfully
- `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`
  - passed (`0`)
- `conda run -n scipy mypy vercor examples tests`
  - passed
- `conda run -n scipy pytest tests/test_slab_kernels.py -q`
  - passed
- `conda run -n scipy pytest tests/test_component_models_coverage.py -q`
  - passed
- `conda run -n scipy pytest tests/test_slab_kernels.py tests/test_component_models_coverage.py -q`
  - passed
- `conda run -n scipy pytest tests/ -q --fast`
  - passed
- `conda run -n scipy pytest tests/ -q`
  - passed

## Third JAX Translation Slice 3B: Veros Helper Boundary

- Refactored the remaining Veros boundary helper math in `vercor/components/external/veros_gcm.py` without changing public component APIs:
  - added `_update_veros_interior()` as a private JAX helper for fixed `2:-2, 2:-2, ...` halo-preserving interior replacement
  - added `_prepare_surface_forcing_fields()` as a private JAX helper for transpose, singleton-axis expansion, `NaN` cleanup, and `qnec` gating by `restore_to_climatology`
  - kept `pure()` as the copy-before-mutate boundary helper for Veros runtime objects and clarified that scope in the docstring
  - narrowed `set_variable()` into a thin state adapter that copies the state, calls the JAX interior-update helper, and writes NumPy arrays back to the Veros state object
- Audited `compute_fluxes()` so the boundary math now stays JAX-native through masking, velocity interpolation, temperature assembly, and `qnet` / `qnec` construction until the final NumPy conversion required by the Veros adapter boundary.
- Extended `tests/test_external_components_coverage.py` with direct helper coverage:
  - `jax.jit` coverage and a gradient smoke test for `_update_veros_interior()`
  - helper coverage for `_prepare_surface_forcing_fields()` shape/orientation, `NaN` cleanup, and `restore_to_climatology=False` `qnec` zeroing
  - wrapper-level `VerosGCM.step()` coverage that confirms cleaned forcing payloads are what reach `set_variable()`
- `DEPENDENCIES.md` did not require changes for this slice because no new module-level dependency edge was introduced.

## Validation (Slice 3B Veros Helper Boundary, 2026-04-23)

- `conda run -n scipy pytest tests/test_external_components_coverage.py -q`
  - passed
- `conda run -n scipy black vercor examples tests`
  - passed
  - note: Black emitted the existing Python 3.13 vs target-3.14 safety-check warning but completed successfully
- `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`
  - passed (`0`)
- `conda run -n scipy mypy vercor examples tests`
  - passed
- `conda run -n scipy pytest tests/ -q --fast`
  - passed
- `conda run -n scipy pytest tests/ -q`
  - passed

## Notes / Failed Approaches (Slice 3B)

- The numeric helper layer is now explicitly `jax.jit`-safe, but the full Veros runtime object model is still intentionally kept outside `jax.jit`; forcing the entire `pure()` / model-step boundary into JIT would require a broader redesign of how Veros state objects are represented.

## Fourth JAX Translation Slice 4A: JAXGCM Adapter Boundary

- Refactored `vercor/components/external/jax_gcm.py` so the JCM adapter now keeps its internal preprocessing and output mapping in JAX-native helpers while preserving the public wrapper API:
  - added `_cleanup_surface_temperature_fields()` as a private `jax.jit` helper for `NaN` cleanup, total surface temperature assembly, and cold-cell diagnostics
  - added `_prepare_surface_temperature_forcing()` as a private `jax.jit` helper for land/ocean masking and `288.15 K` zero-cell fallback before the JCM forcing boundary
  - added `_map_jcm_output_fields()` as a private `jax.jit` helper for transpose conventions, humidity conversion, flux sign handling, pressure assembly, density / potential-temperature diagnostics, and sigma-level height mapping
  - kept NumPy conversion only at the external forcing boundary passed into `self.forcing.copy(...)`
- Updated `JAXGCM.initialize()` to seed translated runtime fields with `jnp.zeros` / `jnp.full` instead of NumPy arrays.
- Kept `JAXGCM.step()` thin:
  - incoming land / sea surface temperatures are normalized once through the new helper
  - forcing fields are prepared through the new helper and converted to NumPy only when handed back to JCM
  - mapped JCM outputs are written back to `self.data` directly from the jitted helper output
- Extended `tests/test_external_components_coverage.py` with:
  - direct `jax.jit` coverage for all three new private helpers
  - a gradient smoke test for `_cleanup_surface_temperature_fields()`
  - wrapper-level regression assertions for total surface temperature assembly, forcing masking / fallback, transpose conventions, flux signs, humidity scaling, pressure/density/potential-temperature wiring, and output gating
- `DEPENDENCIES.md` did not require changes for this slice because no new module-level dependency edge was introduced.

## Validation (Slice 4A JAXGCM Adapter Boundary, 2026-04-23)

- `conda run -n scipy pytest tests/test_external_components_coverage.py tests/test_jax_gcm_output_frequency.py -q`
  - passed
- `conda run -n scipy pytest tests/ -q --fast`
  - passed
- `conda run -n scipy black vercor examples tests`
  - passed
  - note: Black emitted the existing Python 3.13 vs target-3.14 safety-check warning but completed successfully
- `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`
  - passed (`0`)
- `conda run -n scipy mypy vercor examples tests`
  - passed
- `conda run -n scipy pytest tests/ -q`
  - passed

## Notes / Failed Approaches (Slice 4A)

- `Component.data` is still typed as `dict[str, NDArray]`, so directly assigning `jax.Array` values in `JAXGCM.initialize()` triggered `mypy` assignment errors. The final implementation keeps the JAX runtime values and uses narrow `cast(Any, ...)` annotations at initialization to satisfy the current shared component typing without widening that interface in this slice.

## Fifth JAX Translation Slice 5A: Shared Core Array Boundary

- Refactored the shared component storage layer so JAX arrays can move through the runtime core without being eagerly coerced to NumPy:
  - added `RuntimeArray` in `vercor/types.py` for mixed NumPy/JAX in-memory field storage
  - widened `TimedNamedArray.data`, `Shared.fields()`, `Component.data`, and `Component.get()` in `vercor/components/base.py`
  - removed the eager `np.asarray(...)` coercion in `Shared._assign_field()` so JAX-backed fields stay JAX-backed in runtime storage
  - kept explicit NumPy conversion at file/output boundaries in `TimedNamedArray.__array__()`, `ComponentForcingData._read_forcing()`, and `write_shared_to_netcdf()`
- Refactored the coupler dispatch boundary in `vercor/coupler.py`:
  - widened in-memory mask annotations to `RuntimeArray`
  - removed the unconditional `np.asarray(...)` cast around scalar regridder outputs in `interpolate_and_dispatch_fields()`
  - kept land/ocean mask creation and validation NumPy-backed where the existing helper logic depends on NumPy comparison semantics
- Cleaned up translated JAX component slices that no longer needed shared-storage casts:
  - removed now-unnecessary `cast(Any, ...)` assignments in `vercor/components/slab/atmosphere.py`, `vercor/components/slab/ocean.py`, `vercor/components/slab/land.py`, `vercor/components/slab/seaice.py`, and the JAX-backed field seeding path in `vercor/components/external/jax_gcm.py`
- Widened the time-slice helper signatures in `vercor/tools.py` to accept mixed runtime arrays so `Component.send_fields()` remains type-clean after the shared-core change.
- Extended coverage to lock the new runtime guarantees:
  - `tests/test_component_base_coverage.py` now verifies `Shared`, `receive_fields()`, `send_fields()`, and `get()` preserve JAX-backed arrays end to end while the netCDF writer still succeeds at the NumPy/xarray boundary
  - `tests/test_coupler_coverage.py` now verifies scalar exchange dispatch preserves JAX regridder outputs after masking and accepts mixed NumPy/JAX field flow
  - `tests/_coverage_support.py` now accepts mixed runtime arrays in the recording regridder scaffolding
- `DEPENDENCIES.md` did not require changes for this slice because the new runtime-array alias did not introduce a new module-level dependency edge.

## Validation (Slice 5A Shared Core Array Boundary, 2026-04-23)

- `conda run -n scipy pytest tests/test_component_base_coverage.py tests/test_coupler_coverage.py tests/test_external_components_coverage.py -q`
  - passed
- `conda run -n scipy black vercor examples tests`
  - passed
  - note: Black emitted the existing Python 3.13 vs target-3.14 safety-check warning but completed successfully
- `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`
  - passed (`0`)
- `conda run -n scipy mypy vercor examples tests`
  - passed
- `conda run -n scipy pytest tests/ -q --fast`
  - passed
- `conda run -n scipy pytest tests/ -q`
  - passed

## Notes / Failed Approaches (Slice 5A)

- No production redesign of mask-generation helpers was needed in this slice. Keeping those helpers NumPy-backed avoided broad churn in conservation/mask validation code while still removing the unnecessary runtime coercions from shared storage and scalar dispatch.

## Sixth JAX Translation Slice 6A: JAX-First Rectilinear Regridding

- Refactored the reusable rectilinear regridding core to remove the remaining NumPy-only validation and mask-plumbing paths while preserving public APIs and numerical behavior:
  - `vercor/grid.py`
    - replaced eager monotonicity validation with JAX-backed checks while keeping strict ascending-coordinate requirements and existing error text
  - `vercor/regridders/base.py`
    - replaced NumPy identical-grid detection with JAX-backed coordinate equality collapsed to a Python `bool`
  - `vercor/interpolators/bilinear_rectilinear.py`
    - removed NumPy from constructor monotonicity/orientation checks
    - switched the default `fill_value` to a Python `NaN` literal instead of `np.nan`
    - kept the existing JAX runtime interpolation/extrapolation path unchanged
  - `vercor/interpolators/conservative_remap_rectilinear.py`
    - cleaned up `apply_scalar()` shape validation and the source/destination mass helpers to use JAX-backed arrays end to end
    - intentionally left the eager overlap/precompute assembly in `__init__` host-side for now
  - `vercor/tools.py`
    - moved ocean/land mask construction, land/ocean mask-sum checks, conservation checks, and land-mask creation to JAX-first internal array handling
    - widened those helper signatures to accept `RuntimeArray` so NumPy and JAX callers remain type-clean
- Extended tests around the translated regridding core:
  - `tests/test_helpers_coverage.py`
    - JAX-backed `RectilinearGrid` construction and mask preservation
  - `tests/test_bilinear_rectilinear_interpolator.py`
    - JAX-array constructor inputs and longitude/latitude orientation flags
  - `tests/test_bilinear_rectilinear_regridder.py`
    - identical-grid scalar short-circuit with JAX-backed coordinates and JAX field input
  - `tests/test_conservative_rectilinear_regridder.py`
    - identical-grid scalar short-circuit with JAX-backed coordinates and JAX field input
  - `tests/test_conservative_rectilinear_remapper.py`
    - mass-helper coverage with JAX-array inputs
  - `tests/test_tools_assets_and_regridding.py`
    - JAX-backed inputs/outputs for mask clipping, mask-sum validation, and `create_lnd_mask_from_ocn()`
- `DEPENDENCIES.md` did not require changes for this slice because no new module-level dependency edge was introduced.

## Validation (Slice 6A JAX-First Rectilinear Regridding, 2026-04-23)

- `conda run -n scipy black vercor examples tests`
  - passed
  - note: Black emitted the existing Python 3.13 vs target-3.14 safety-check warning but completed successfully
- `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`
  - passed (`0`)
- `conda run -n scipy mypy vercor examples tests`
  - passed
- `conda run -n scipy pytest tests/test_helpers_coverage.py tests/test_bilinear_rectilinear_interpolator.py tests/test_bilinear_rectilinear_regridder.py tests/test_conservative_rectilinear_remapper.py tests/test_conservative_rectilinear_regridder.py tests/test_tools_assets_and_regridding.py -q`
  - passed
- `conda run -n scipy pytest tests/ -q --fast`
  - passed
- `conda run -n scipy pytest tests/ -q`
  - passed

## Notes / Failed Approaches (Slice 6A)

- A first pass kept NumPy-only type annotations on the regridding mask helpers after switching their internal logic to JAX arrays; `mypy` rejected the new JAX-backed call sites. Widening those helper signatures to `RuntimeArray` resolved the issue without changing runtime behavior.
- Full conservative remapper overlap assembly is still intentionally deferred. The current slice only translated validation/runtime application and mask plumbing because the overlap preprocessing is static setup code rather than the differentiable hot path.

## Sixth JAX Translation Slice 6B: Conservative Overlap Assembly

- Finished the remaining conservative-core translation work in `vercor/interpolators/conservative_remap_rectilinear.py` while keeping the public remapper and regridder call patterns unchanged.
- Replaced the NumPy-based overlap/precompute assembly in `ConservativeRectilinearRemapper.__init__()` with JAX-first eager helpers:
  - latitude standardization now uses JAX arrays and preserves the existing flipped-latitude behavior
  - interval-overlap assembly now builds dense destination-by-source overlap matrices with broadcasted `jax.numpy` min/max arithmetic
  - longitude periodic overlap now sums three shifted dense overlap matrices (`0`, `+360`, `-360`) before flattening, so duplicate shift contributions are merged without `np.unique` / `np.add.at`
  - source-mask filtering now drops invalid triplets through JAX boolean/index operations before storing `dst_indices`, `src_indices`, and `overlap_weights`
  - destination-area preparation is now JAX-native and still preserves the `np.inf`-equivalent zero-area sentinel behavior via `jnp.inf`
- Kept the constructor eager and host-side on purpose; the slice removes NumPy from the precompute math without trying to JIT the constructor itself.
- Cleaned the supporting wrapper in `vercor/regridders/conservative.py`:
  - removed the unused runtime NumPy import
  - switched the `fill_value` default literal from `np.nan` to `float("nan")`
  - widened `source_mask` typing to `RuntimeArray` so mixed NumPy/JAX callers remain type-clean
- Extended conservative tests to lock the new guarantees:
  - `tests/test_conservative_rectilinear_remapper.py`
    - JAX-backed constructor inputs for bounds and masks
    - periodic duplicate-shift overlap merging
    - eager masked-triplet dropping
  - `tests/test_conservative_rectilinear_regridder.py`
    - mixed NumPy/JAX edge arrays and JAX-backed `source_mask` through the public wrapper
- `DEPENDENCIES.md` did not require changes for this slice because no new module-level dependency edge was introduced.

## Validation (Slice 6B Conservative Overlap Assembly, 2026-04-23)

- `conda run -n scipy pytest tests/test_conservative_rectilinear_remapper.py tests/test_conservative_rectilinear_regridder.py -q`
  - passed
- `conda run -n scipy black vercor examples tests`
  - passed
  - note: Black emitted the existing Python 3.13 vs target-3.14 safety-check warning but completed successfully
- `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`
  - passed (`0`)
- `conda run -n scipy mypy vercor examples tests`
  - passed
- `conda run -n scipy pytest tests/ -q --fast`
  - passed
- `conda run -n scipy pytest tests/ -q`
  - passed

## Notes / Failed Approaches (Slice 6B)

- A first implementation used Python `sum(...)` to accumulate the three longitude-shift overlap matrices. `mypy` inferred that expression as `Array | Literal[0]`, so the final version uses explicit staged JAX-array accumulation instead.
- The first test pass also kept NumPy-only `NDArray` annotations on the conservative remapper/regridder constructor surfaces. `mypy` rejected the new JAX-backed test inputs until those annotations were widened to the shared `RuntimeArray` alias.

## Next Remaining Migration Targets

- Remaining NumPy-heavy production paths are now mostly outside the conservative-core hot path:
  - data adapters and forcing preparation in `vercor/components/data/`
  - explicit runtime-boundary cleanup still present in `vercor/coupler.py`
  - non-core utility and plotting helpers in `vercor/tools.py`
  - file/output boundaries in `vercor/components/base.py`, which should stay NumPy/xarray-backed unless there is a concrete reason to redesign them

## Seventh JAX Translation Slice 7A: JAX-First Data Adapters

- Translated the remaining in-scope data adapters to keep runtime arrays JAX-backed while preserving the existing public component APIs and NumPy/xarray file boundaries:
  - `vercor/components/data/era5_atmosphere.py`
    - forcing reads are normalized to `jnp.asarray(...)` at the component boundary
    - added private pure helpers for surface-pressure decoding, one-month diagnostic assembly, and total surface-temperature combination
    - `initialize()` now stacks per-month diagnostic outputs from the JAX helper back into runtime storage
  - `vercor/components/data/era5_ocean.py`
    - added private JAX helpers for ocean-mask derivation from land fraction and masked SST application
    - longitude/latitude, binary mask, and stored SST now stay JAX-backed in memory
  - `vercor/components/data/erainterim_ocean.py`
    - added private JAX helpers for global latitude assembly, full-grid field staging, binary-mask derivation, and masked SST application
    - the existing 1 degree vs 4 degree padding, longitude shift, and Celsius-to-Kelvin behavior were preserved
  - `vercor/components/data/jcm_land.py`
    - added a private JAX coordinate-conversion helper using `jnp.rad2deg`
    - stored land temperature and soil moisture now remain JAX-backed in memory
- Added a dedicated helper test module:
  - `tests/test_data_component_kernels.py`
    - `jax.jit` coverage for ERA5 atmosphere pressure/diagnostic helpers and JCM coordinate conversion
    - reverse-mode gradient smoke test for the ERA5 atmosphere diagnostic helper
    - JAX-array input/output checks for ERA5 ocean and ERA-Interim ocean helper paths
- Updated `tests/test_component_models_coverage.py` only at the wrapper level:
  - constructors, masks, and shapes remain unchanged
  - translated components now explicitly preserve JAX-backed runtime arrays
- Updated `DEPENDENCIES.md` to include the translated data-adapter layer.

## Validation (Slice 7A JAX-First Data Adapters, 2026-04-23)

- `conda run -n scipy pytest tests/test_component_models_coverage.py tests/test_data_component_kernels.py -q`
  - passed
- `conda run -n scipy black vercor examples tests`
  - passed
  - note: Black emitted the existing Python 3.13 vs target-3.14 safety-check warning but completed successfully
- `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`
  - passed (`0`)
- `conda run -n scipy mypy vercor examples tests`
  - passed
- `conda run -n scipy pytest tests/ -q --fast`
  - passed
- `conda run -n scipy pytest tests/ -q`
  - passed

## Notes / Failed Approaches (Slice 7A)

- A first pass vectorized the ERA5 atmosphere monthly diagnostics with `jax.vmap`. That helper was numerically fine, but it broke the existing wrapper tests because they monkeypatch host-side physics helpers that call Python `float(...)` internally. The final implementation keeps the extracted one-month helper JAX-pure and `jax.jit`/`grad`-safe, while `initialize()` loops over months on the host and stacks the results.

## Seventh JAX Translation Slice 7B: Core Runtime NumPy Cleanup

- Cleaned the remaining in-scope runtime NumPy usage in the shared core while preserving the existing public APIs and explicit plotting / file-I/O NumPy boundaries:
  - `vercor/tools.py`
    - `safe_component_nanmean()` and `_safe_component_metric_mean()` now normalize through `jnp.asarray(...)` and compute NaN-aware reductions with `jax.numpy`
    - `grids_identical()` now uses JAX-backed coordinate comparisons collapsed to Python `bool`
    - `get_periodic_interval()` now computes host integer indices with pure scalar arithmetic instead of `np.array(..., dtype="int")`
    - plotting helpers intentionally remain NumPy / Matplotlib boundaries
  - `vercor/coupler.py`
    - removed the top-level NumPy dependency
    - default `_binary_masks` and `_fractional_masks` are now created as JAX arrays
    - `_create_exchange_masks()` now passes runtime arrays directly into `check_remap_conservation()`
    - `_validate_land_mask_consistency()` now compares masks with JAX-backed equality and mismatch counting
  - `vercor/regridders/bilinear.py`
    - removed the NumPy import used only for `np.nan`
    - switched the default `fill_value` to `float("nan")`
- Updated targeted tests:
  - `tests/test_tools_components_and_plotting.py`
    - JAX-backed coordinates for `grids_identical()`
    - JAX-backed component field input for `safe_component_nanmean()`
    - plotting path now consumes JAX-backed runtime fields while preserving the NumPy conversion boundary
  - `tests/test_tools_time_and_forcing.py`
    - added JAX-backed forcing-cube coverage for `get_field_at_specific_time()`
    - locked `get_periodic_interval()` indices to host `int` values
  - `tests/test_coupler_coverage.py`
    - added assertions that untouched default mask pools remain JAX-backed after `initialize()`
  - `tests/_tools_support.py`
    - widened `DummyGridComponent` test storage from NumPy-only arrays to the shared `RuntimeArray` alias so JAX-backed test inputs stay type-clean
- `DEPENDENCIES.md` did not require changes for this slice because no new module-level dependency edge was introduced.

## Validation (Slice 7B Core Runtime NumPy Cleanup, 2026-04-23)

- `conda run -n scipy pytest tests/test_tools_components_and_plotting.py tests/test_tools_time_and_forcing.py tests/test_coupler_coverage.py -q`
  - passed
- `conda run -n scipy black vercor examples tests`
  - passed
  - note: Black emitted the existing Python 3.13 vs target-3.14 safety-check warning but completed successfully
- `conda run -n scipy pytest tests/test_tools_components_and_plotting.py tests/test_tools_time_and_forcing.py tests/test_coupler_coverage.py -q`
  - passed
  - rerun after `black` reformatted `vercor/tools.py` and `tests/test_tools_components_and_plotting.py`
- `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`
  - passed (`0`)
- `conda run -n scipy mypy vercor examples tests`
  - passed
- `conda run -n scipy pytest tests/ -q --fast`
  - passed
- `conda run -n scipy pytest tests/ -q`
  - passed

## Notes / Failed Approaches (Slice 7B)

- The first test-only pass fed JAX arrays into `DummyGridComponent`, but that helper was still typed as `dict[str, np.ndarray]`, so `mypy` rejected the updated coverage. The final version widens that test helper to the existing `RuntimeArray` alias instead of changing any production interface.

## Eighth JAX Translation Slice 8A: CAMulator Boundary

- Translated the in-scope CAMulator adapter boundary while preserving the explicit Torch, xarray, CREDIT, and file-output boundaries:
  - `vercor/components/external/camulator.py`
    - added a JAX-backed runtime-field initializer for exchange storage
    - added `_prepare_camulator_surface_forcing()` for NaN cleanup, land-mask fallback, and rescaling through `jax.numpy`
    - added `_map_camulator_prediction_arrays()` to map host-transferred CAMulator tensor outputs into JAX-backed VerCOR runtime fields
    - kept all Torch tensor creation, xarray output, and NetCDF writes host-side
  - `vercor/components/data/camulator_land.py`
    - initialized and stepped land surface temperature storage with JAX arrays
- Added `tests/test_camulator_component_kernels.py`:
  - `jax.jit` coverage for the CAMulator surface-forcing and prediction-mapping helpers
  - reverse-mode gradient smoke test for surface-forcing preparation
  - flux sign, pressure/height, shape, and JAX runtime-storage checks
  - lightweight patched CAMulatorLand coverage without real CAMulator model files
- Updated `DEPENDENCIES.md` to describe the CAMulator adapter and land forcing layer.

## Validation (Slice 8A CAMulator Boundary, 2026-04-24)

- `conda run -n scipy pytest tests/test_camulator_component_kernels.py -q`
  - passed
- `conda run -n scipy black vercor examples tests`
  - passed
  - note: Black emitted the existing Python 3.13 vs target-3.14 safety-check warning but completed successfully
- `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`
  - passed (`0`)
- `conda run -n scipy mypy vercor examples tests`
  - passed
- `conda run -n scipy pytest tests/ -q --fast`
  - passed
- `conda run -n scipy pytest tests/ -q`
  - passed

## Notes / Failed Approaches (Slice 8A)

- The first flake8 pass surfaced an unrelated stale `numpy` import in `tests/_tools_support.py`; removing it restored the project lint count to `0`.

## Eighth JAX Translation Slice 8B: JAX-First Example Drivers

- Translated the remaining NumPy-heavy example drivers to use JAX-first runtime array handling while keeping plotting and external runtime boundaries explicit:
  - `examples/run_data_driver.py`
    - replaced the NumPy speed metric with a shared JAX helper
    - removed stale `NDArray` typing from the metric path
  - `examples/run_slab_driver.py`
    - replaced example mask construction and ice-fraction diagnostics with `jax.numpy`
  - `examples/run_jcm_with_slab.py`
    - replaced mask construction, coordinate conversion, and mask summaries with `jax.numpy`
  - `examples/run_jcm_with_era5data.py`, `examples/run_jcm_with_veros.py`, and `examples/run_jcm_with_verosdata.py`
    - replaced direct `np.array(...).T` terrain-mask mutation with an explicit JAX-to-host transfer helper
- Added `examples/jax_array_helpers.py` for example-local JAX diagnostics and explicit host transfer at third-party model boundaries.
- Added `examples/__init__.py` so `mypy` resolves the helper under a single module name.
- Added `tests/test_example_jax_helpers.py` to cover:
  - host transfer from JAX runtime arrays
  - transposed host transfer for mutable third-party masks
  - JAX-backed component vector-speed diagnostics
- No core coupler, component, or regridder APIs changed.
- `DEPENDENCIES.md` did not require changes because this slice only updated example-driver code and test coverage.

## Validation (Slice 8B Example Drivers, 2026-04-24)

- `conda run -n scipy pytest tests/test_example_jax_helpers.py -q`
  - passed
- `conda run -n scipy black vercor examples tests`
  - passed
  - note: Black emitted the existing Python 3.13 vs target-3.14 safety-check warning but completed successfully
- `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`
  - passed (`0`)
- `conda run -n scipy mypy vercor examples tests`
  - passed
- `conda run -n scipy pytest tests/ -q --fast`
  - passed
- `conda run -n scipy pytest tests/ -q`
  - passed

## Notes / Failed Approaches (Slice 8B)

- The first `mypy` pass after adding the helper reported `examples/jax_array_helpers.py` twice, as both `jax_array_helpers` and `examples.jax_array_helpers`. Adding `examples/__init__.py` made `examples` an explicit package and resolved the duplicate-module error.

## Ninth JAX Translation Slice 9A: External Adapter Boundary Cleanup

- Tightened the remaining in-scope external adapter construction and initialization boundaries while preserving explicit third-party runtime transfers:
  - `vercor/components/external/jax_gcm.py`
    - JCM grid longitude/latitude and interpolation mask construction now use `jax.numpy`
    - `sigma_levels` storage now uses the shared mixed `RuntimeArray` alias instead of a NumPy-only annotation
    - NumPy conversion remains only at the JCM forcing/output boundary
  - `vercor/components/external/veros_gcm.py`
    - Veros grid mask derivation now uses JAX-backed array logic
    - initialized and refreshed sea-surface temperature storage is now explicitly JAX-backed
    - NumPy conversion remains at the Veros state mutation boundary
  - `vercor/components/external/camulator.py`
    - CAMulator static component mask construction now enters VerCOR as a JAX array
    - Torch, xarray, and NetCDF host boundaries remain unchanged
  - `vercor/components/external/jax_gcm_tools.py`
    - public helper annotations were widened from NumPy-only arrays to the shared `RuntimeArray` alias
- Extended targeted coverage:
  - `tests/test_external_components_coverage.py`
    - lightweight JAXGCM constructor coverage for JAX-backed grid and sigma-level storage
    - Veros constructor and runtime SST storage coverage for JAX-backed arrays
  - `tests/test_camulator_component_kernels.py`
    - lightweight CAMulatorGCM constructor coverage for JAX-backed static masks
- `DEPENDENCIES.md` did not require changes because no new module-level dependency edge was introduced.

## Validation (Slice 9A External Adapter Boundary Cleanup, 2026-04-24)

- `conda run -n scipy pytest tests/test_external_components_coverage.py tests/test_camulator_component_kernels.py tests/test_external_tools_coverage.py -q`
  - passed
- `conda run -n scipy mypy vercor/components/external/jax_gcm.py vercor/components/external/veros_gcm.py vercor/components/external/camulator.py vercor/components/external/jax_gcm_tools.py tests/test_external_components_coverage.py tests/test_camulator_component_kernels.py`
  - passed
- `conda run -n scipy black vercor examples tests`
  - passed
  - note: Black emitted the existing Python 3.13 vs target-3.14 safety-check warning but completed successfully
- `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`
  - passed (`0`)
- `conda run -n scipy mypy vercor examples tests`
  - passed
- `conda run -n scipy pytest tests/ -q --fast`
  - passed
- `conda run -n scipy pytest tests/ -q`
  - passed

## Ninth JAX Translation Slice 9B: Veros Boundary Cleanup

- Tightened the Veros adapter boundary in `vercor/components/external/veros_gcm.py` without changing the public component API:
  - `compute_fluxes()` now returns JAX arrays and keeps VerCOR-side flux math JAX-backed until the Veros state mutation boundary
  - added `_extract_surface_temperature()` as a private jitted helper for the repeated Veros SST readout and Celsius-to-Kelvin conversion
  - `VerosGCM.initialize()` and `VerosGCM.step()` now use the shared helper for JAX-backed SST storage
  - NumPy conversion remains explicit at `set_variable()` / Veros mutable-state handoff
- Extended `tests/test_external_components_coverage.py` with:
  - assertions that `compute_fluxes()` returns JAX-backed arrays while preserving existing sign and `qnec` masking behavior
  - `jax.jit` and reverse-mode gradient coverage for `_extract_surface_temperature()`
- `DEPENDENCIES.md` did not require changes because no new module-level dependency edge was introduced.

## Validation (Slice 9B Veros Boundary Cleanup, 2026-04-24)

- `conda run -n scipy pytest tests/test_external_components_coverage.py -q`
  - passed
- `conda run -n scipy black vercor examples tests`
  - passed
  - note: Black emitted the existing Python 3.13 vs target-3.14 safety-check warning but completed successfully
- `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`
  - passed (`0`)
- `conda run -n scipy mypy vercor examples tests`
  - passed
- `conda run -n scipy pytest tests/ -q --fast`
  - passed
- `conda run -n scipy pytest tests/ -q`
  - passed

## Notes / Failed Approaches (Slice 9B)

- No failed implementation approaches. The slice kept Veros object mutation host-side and only moved the VerCOR-side flux/SST handoff later in the boundary.

## Ninth JAX Translation Slice 9C: CAMulator Forcing Boundary Cleanup

- Tightened the remaining CAMulator forcing-input boundary while preserving Torch, xarray, CREDIT, and NetCDF as explicit external runtime boundaries:
  - added JAX helpers in `vercor/components/external/camulator.py` for dynamic forcing layout conversion and CAMulator SST input expansion
  - added a single explicit JAX-to-host-to-Torch transfer helper for CAMulator step inputs
  - replaced inline dynamic forcing `np.stack(...)` and inline SST `torch.tensor(np.asarray(...))` staging in `CAMulatorGCM.step()`
  - replaced static forcing `np.stack(...)` in `vercor/components/external/camulator_state.py` with xarray `to_array(...)` staging before the Torch boundary
- Extended `tests/test_camulator_component_kernels.py` with:
  - `jax.jit` coverage for dynamic forcing layout conversion and SST input expansion
  - static forcing order/shape coverage through the xarray/Torch helper
  - lightweight patched `CAMulatorGCM.step()` coverage confirming dynamic forcing shape, SST tensor shape, and JAX-backed total surface temperature storage
- `DEPENDENCIES.md` did not require changes because no new module-level dependency edge was introduced.

## Validation (Slice 9C CAMulator Forcing Boundary Cleanup, 2026-04-24)

- `conda run -n scipy pytest tests/test_camulator_component_kernels.py -q`
  - passed
- `conda run -n scipy black vercor examples tests`
  - passed
  - note: Black emitted the existing Python 3.13 vs target-3.14 safety-check warning but completed successfully
- `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`
  - passed (`0`)
- `conda run -n scipy mypy vercor examples tests`
  - passed
- `conda run -n scipy pytest tests/ -q --fast`
  - passed
- `conda run -n scipy pytest tests/ -q`
  - passed

## Notes / Failed Approaches (Slice 9C)

- The first helper implementation used `torch.as_tensor()` directly on a JAX host transfer, which produced a read-only NumPy-array warning from Torch. The final helper copies the host array before constructing the Torch tensor.
- The first `mypy` pass rejected fake test accessors assigned to `CAMulatorGCM` attributes; the final test casts the manually constructed component to `Any` because this is intentionally patched wrapper coverage rather than normal construction.

## Tenth JAX Translation Slice 10A: Shared Runtime Array Boundaries

- Tightened the shared component/tooling runtime-array boundaries without changing public component, coupler, or exchange APIs:
  - added `vercor.tools._runtime_array_to_host()` as the explicit JAX device-to-host transfer helper for NumPy-only consumers
  - normalized `get_field_time_slice()` and `get_field_at_specific_time()` through `jax.numpy` so sliced/interpolated fields return JAX-backed arrays for both NumPy and JAX input data
  - moved plotting data extraction in `vercor/tools.py` to use the explicit host-transfer helper at the Matplotlib boundary
  - moved `TimedNamedArray.__array__()` and `write_shared_to_netcdf()` in `vercor/components/base.py` to the same explicit host-transfer boundary
- Extended focused coverage:
  - `tests/test_tools_time_and_forcing.py`
    - asserts time slicing and monthly interpolation return `jax.Array` from NumPy-backed data
    - adds direct JAX-backed time-slice coverage
  - `tests/test_component_base_coverage.py`
    - asserts `TimedNamedArray.__array__()` works for JAX-backed data
    - writes JAX-backed shared fields and JAX-backed grid coordinates through NetCDF output
  - `tests/test_tools_components_and_plotting.py`
    - keeps mixed NumPy/JAX component plotting coverage and now uses JAX-backed grid coordinates in one plotted component
- `DEPENDENCIES.md` did not require changes because no new module-level dependency edge was introduced.

## Validation (Slice 10A Shared Runtime Array Boundaries, 2026-04-24)

- `conda run -n scipy pytest tests/test_component_base_coverage.py tests/test_tools_time_and_forcing.py tests/test_tools_components_and_plotting.py -q`
  - passed
- `conda run -n scipy black vercor examples tests`
  - passed
  - note: Black emitted the existing Python 3.13 vs target-3.14 safety-check warning but completed successfully
- `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`
  - passed (`0`)
- `conda run -n scipy mypy vercor examples tests`
  - passed
- `conda run -n scipy pytest tests/ -q --fast`
  - passed
- `conda run -n scipy pytest tests/ -q`
  - passed

## Notes / Failed Approaches (Slice 10A)

- No failed implementation approaches. The slice only moves NumPy conversion to explicit host-only boundaries and keeps VerCOR runtime data JAX-backed.

## Tenth JAX Translation Slice 10B: External Host Transfer Centralization

- Centralized the remaining external-adapter host transfers on `vercor.tools._runtime_array_to_host()` while preserving explicit third-party runtime boundaries:
  - `vercor/components/external/jax_gcm.py`
    - replaced direct `np.asarray(...).transpose()` forcing handoffs with shared host transfers and `.T`
    - removed the now-unused NumPy import from the adapter
  - `vercor/components/external/veros_gcm.py`
    - widened `set_variable()` to accept mixed `RuntimeArray` inputs
    - moved the JAX-to-host conversion inside the Veros mutable-state boundary
    - stopped converting prepared forcing fields to NumPy before calling `set_variable()`
  - `vercor/components/external/camulator.py`
    - removed the adapter-local JAX-to-host helper
    - reused the shared host-transfer helper for Torch tensor staging and CAMulator output mapping inputs
    - removed the now-unused NumPy import from the adapter
- Extended focused boundary tests:
  - `tests/test_external_components_coverage.py`
    - asserts JAXGCM forcing copy receives host NumPy arrays with the existing transpose convention
    - asserts Veros `set_variable()` accepts JAX-backed inputs and stores host arrays at the mutation boundary
    - asserts `VerosGCM.step()` passes JAX-backed prepared forcing fields into `set_variable()`
  - `tests/test_camulator_component_kernels.py`
    - asserts CAMulator Torch staging copies host data so mutating the tensor does not mutate the JAX source
- `DEPENDENCIES.md` did not require changes because no new module-level dependency edge was introduced.

## Validation (Slice 10B External Host Transfer Centralization, 2026-04-24)

- `conda run -n scipy pytest tests/test_external_components_coverage.py tests/test_camulator_component_kernels.py -q`
  - passed
- `conda run -n scipy black vercor examples tests`
  - passed
  - note: Black emitted the existing Python 3.13 vs target-3.14 safety-check warning but completed successfully
- `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`
  - passed (`0`)
- `conda run -n scipy mypy vercor examples tests`
  - passed
- `conda run -n scipy pytest tests/test_external_components_coverage.py tests/test_camulator_component_kernels.py -q`
  - passed after Black reformatted `vercor/components/external/camulator.py`
- `conda run -n scipy pytest tests/ -q --fast`
  - passed
- `conda run -n scipy pytest tests/ -q`
  - passed

## Notes / Failed Approaches (Slice 10B)

- No failed implementation approaches. The slice keeps NumPy, xarray, Torch, Matplotlib, and Veros object mutation as explicit host-only boundaries.

## Eleventh JAX Translation Slice 11A: ERA5 Land Adapter

- Translated the ERA5 land forcing adapter runtime path while preserving the public component API and explicit h5netcdf/NumPy file-read boundary:
  - `vercor/components/data/era5_land.py`
    - added `_prepare_era5_land_runtime_fields()` for JAX-backed longitude, latitude, transposed land mask, and land surface temperature storage
    - `ERA5Land.__init__()` now normalizes forcing arrays through that helper before constructing the grid and storing `land_surface_temperature`
    - `initialize()` and `step()` remain no-op dataset-adapter hooks
- Extended focused coverage:
  - `tests/test_data_component_kernels.py`
    - `jax.jit` coverage for the ERA5 land runtime-field helper
    - reverse-mode gradient smoke coverage for land surface temperature passthrough
  - `tests/test_component_models_coverage.py`
    - constructor coverage now asserts ERA5 land grid coordinates, binary mask, and runtime temperature storage are JAX-backed
- Updated `DEPENDENCIES.md` to include the ERA5 land forcing adapter layer.

## Validation (Slice 11A ERA5 Land Adapter, 2026-04-24)

- `conda run -n scipy pytest tests/test_data_component_kernels.py tests/test_component_models_coverage.py -q`
  - passed
- `conda run -n scipy black vercor examples tests`
  - passed
  - note: Black emitted the existing Python 3.13 vs target-3.14 safety-check warning but completed successfully
- `conda run -n scipy pytest tests/test_data_component_kernels.py tests/test_component_models_coverage.py -q`
  - passed after Black reformatted `tests/test_data_component_kernels.py`
- `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`
  - passed (`0`)
- `conda run -n scipy mypy vercor examples tests`
  - passed
- `conda run -n scipy pytest tests/ -q --fast`
  - passed
- `conda run -n scipy pytest tests/ -q`
  - passed

## Notes / Failed Approaches (Slice 11A)

- No failed implementation approaches. The slice keeps forcing file reads host-side and only normalizes the in-memory VerCOR runtime fields to JAX arrays.

## Eleventh JAX Translation Slice 11B: JAX-Backed Forcing Read Boundary

- Moved the shared forcing-read boundary to JAX-backed runtime storage while preserving the explicit h5netcdf/NumPy file-read boundary:
  - `vercor/components/base.py`
    - `_read_forcing()` now returns `RuntimeArray`
    - file loading remains host-side through h5netcdf and NumPy
    - transposed forcing arrays are normalized with `jnp.asarray(...)`
    - `flip_y=True` now uses `jnp.flip(..., axis=1)`
  - `vercor/components/data/era5_atmosphere.py`, `vercor/components/data/era5_ocean.py`, and `vercor/components/data/erainterim_ocean.py`
    - removed redundant `jnp.asarray(self._read_forcing(...))` wrappers now that `_read_forcing()` is the normalization point
- Extended focused coverage:
  - `tests/test_component_base_coverage.py`
    - asserts normal and flipped `_read_forcing()` calls return `jax.Array`
  - `tests/test_component_models_coverage.py`
    - widened patched `_read_forcing()` helper annotations to `RuntimeArray`
- No public component, coupler, exchange, or regridder APIs changed.

## Validation (Slice 11B JAX-Backed Forcing Read Boundary, 2026-04-24)

- `conda run -n scipy pytest tests/test_component_base_coverage.py tests/test_component_models_coverage.py tests/test_data_component_kernels.py -q`
  - passed
- `conda run -n scipy mypy vercor/components/base.py vercor/components/data/era5_atmosphere.py vercor/components/data/era5_ocean.py vercor/components/data/erainterim_ocean.py tests/test_component_base_coverage.py tests/test_component_models_coverage.py`
  - passed
- `conda run -n scipy black vercor examples tests`
  - passed
  - note: Black emitted the existing Python 3.13 vs target-3.14 safety-check warning but completed successfully
- `conda run -n scipy pytest tests/test_component_base_coverage.py tests/test_component_models_coverage.py tests/test_data_component_kernels.py -q`
  - passed after Black reformatted the ERA5 atmosphere and ERA5 ocean adapters
- `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`
  - passed (`0`)
- `conda run -n scipy mypy vercor examples tests`
  - passed
- `conda run -n scipy pytest tests/ -q --fast`
  - passed
- `conda run -n scipy pytest tests/ -q`
  - passed

## Notes / Failed Approaches (Slice 11B)

- No failed implementation approaches. The slice keeps h5netcdf/NumPy as the file-read boundary and makes the returned in-memory forcing arrays JAX-backed.

## Twelfth JAX Translation Slice 12A: Land Adapter JAX Boundary Cleanup

- Tightened the remaining land-adapter runtime boundary helpers without changing public component, coupler, exchange, or regridder APIs:
  - `vercor/components/data/jcm_land.py`
    - added `_prepare_jcm_land_runtime_fields()` for JAX-backed coordinate conversion plus transposed land temperature and soil-moisture storage
    - kept `_coordinates_in_degrees()` as the existing public test helper and routed it through the shared coordinate logic
    - `JCMLand.__init__()` now stores both land-surface temperature and soil moisture from the helper output
  - `vercor/components/data/camulator_land.py`
    - added `_prepare_camulator_land_surface_temperature()` for JAX-backed CAMulator land-temperature storage
    - `CAMulatorLand.step()` now uses the helper at the xarray-to-runtime boundary
- Extended focused coverage:
  - `tests/test_data_component_kernels.py`
    - `jax.jit` coverage for the new JCM land runtime helper
    - reverse-mode gradient smoke coverage for JCM land temperature and soil-moisture passthrough
    - `jax.jit` coverage for the CAMulator land temperature helper
  - `tests/test_component_models_coverage.py`
    - asserts JCM land `soil_moisture` is JAX-backed, matching the existing temperature assertion
  - `tests/test_production_numpy_boundaries.py`
    - adds an AST-based production audit that limits NumPy imports to explicit host/file/plotting/type-boundary modules
- `DEPENDENCIES.md` did not require changes because no new module-level dependency edge was introduced.

## Validation (Slice 12A Land Adapter JAX Boundary Cleanup, 2026-04-24)

- `conda run -n scipy pytest tests/test_data_component_kernels.py tests/test_component_models_coverage.py tests/test_camulator_component_kernels.py tests/test_production_numpy_boundaries.py -q`
  - passed
- `conda run -n scipy black vercor examples tests`
  - passed
  - note: Black emitted the existing Python 3.13 vs target-3.14 safety-check warning but completed successfully
- `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`
  - passed (`0`)
- `conda run -n scipy mypy vercor examples tests`
  - passed
- `conda run -n scipy pytest tests/ -q --fast`
  - passed
- `conda run -n scipy pytest tests/ -q`
  - passed

## Notes / Failed Approaches (Slice 12A)

- No failed implementation approaches. The slice only adds named JAX helper boundaries around existing land-adapter behavior and leaves host-only NumPy boundaries explicit.

## Twelfth JAX Translation Slice 12B: Migration Completion Audit

- Tightened the production NumPy-boundary audit now that the NumPy-to-JAX migration phase is reduced to explicit host-only boundaries:
  - `tests/test_production_numpy_boundaries.py`
    - removed the stale CAMulator-state allowance from the direct NumPy boundary set
    - changed the assertion from subset matching to exact matching so new direct production NumPy imports fail immediately
    - preserved `veros.core.operators.numpy as npx` as a Veros backend boundary rather than a direct NumPy dependency
- Confirmed the remaining direct NumPy imports are intentionally limited to:
  - `vercor/components/base.py`
  - `vercor/tools.py`
  - `vercor/types.py`
- No public component, coupler, exchange, regridder, or runtime-array APIs changed.
- `DEPENDENCIES.md` did not require changes because this slice only tightens migration audit coverage.

## Validation (Slice 12B Migration Completion Audit, 2026-04-24)

- `conda run -n scipy pytest tests/test_production_numpy_boundaries.py -q`
  - passed
- `conda run -n scipy black vercor examples tests`
  - passed
  - note: Black emitted the existing Python 3.13 vs target-3.14 safety-check warning but completed successfully
- `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`
  - passed (`0`)
- `conda run -n scipy mypy vercor examples tests`
  - passed
- `conda run -n scipy pytest tests/ -q --fast`
  - passed
- `conda run -n scipy pytest tests/ -q`
  - passed

## Notes / Failed Approaches (Slice 12B)

- No failed implementation approaches. The slice intentionally keeps NumPy, xarray, Matplotlib, file output, and Veros backend integration as explicit host-only boundaries.

## Thirteenth JAX Translation Slice 13A: Differentiable Public Runtime

- Added the first pure differentiable runtime path while keeping the existing public component, coupler, exchange, regridder, and runtime-array APIs compatible:
  - `vercor/runtime.py`
    - added immutable PyTree containers for runtime field stores, component state, and coupler state
    - added pure exchange dispatch for scalar and vector exchanges
    - added receive/send helpers that update runtime field stores without mutating component objects
    - added pure slab-component stepping over the existing JAX kernels
  - `vercor/coupler.py`
    - routed `interpolate_and_dispatch_fields()` through the pure exchange dispatcher while preserving `Shared` / `TimedNamedArray` wrapper behavior
    - added `run_differentiable()` using `jax.lax.scan` over static run-sequence and exchange metadata
- The differentiable runtime currently supports VerCOR-owned slab components end to end. File I/O, plotting, Veros mutable state, Torch/CAMulator, xarray, and NetCDF remain explicit host-only boundaries.
- Updated `DEPENDENCIES.md` with the new runtime layer.

## Tests Added (Slice 13A)

- Added `tests/test_runtime_state.py` for PyTree round trips, immutable store updates, mapping conversion, and `jax.jit` coverage.
- Added `tests/test_runtime_exchange.py` for scalar mask dispatch, vector exchange dispatch, `jax.jit`, and gradients with respect to source fields and fractional masks.
- Added `tests/test_differentiable_coupler_runtime.py` for one-step and multi-step slab coupler runs under `jax.jit`, `jax.grad`, and `jax.jvp`.

## Validation (Slice 13A, 2026-04-24)

- `conda run -n scipy black vercor examples tests`
  - passed
  - note: Black emitted the existing Python 3.13 vs target-3.14 safety-check warning but completed successfully
- `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`
  - passed (`0`)
- `conda run -n scipy mypy vercor examples tests`
  - passed
- `conda run -n scipy pytest tests/test_runtime_state.py tests/test_runtime_exchange.py tests/test_differentiable_coupler_runtime.py -q`
  - passed
- `conda run -n scipy pytest tests/test_coupler_coverage.py tests/test_component_base_coverage.py tests/test_slab_kernels.py -q`
  - passed
- `conda run -n scipy pytest tests/test_runtime_state.py tests/test_runtime_exchange.py tests/test_differentiable_coupler_runtime.py tests/test_coupler_coverage.py tests/test_component_base_coverage.py tests/test_slab_kernels.py tests/test_production_numpy_boundaries.py -q`
  - passed
- `conda run -n scipy pytest tests/ -q --fast`
  - passed
- `conda run -n scipy pytest tests/ -q`
  - passed

## Notes / Failed Approaches (Slice 13A)

- The first scalar dispatch test expected `13.0` for a masked scaled field, but the correct sum is `12.0`; the test was corrected before implementation validation continued.
- The first slab-ocean closed-form test omitted the existing restoring term in `_advance_sea_surface_temperature()`; the test now includes that term.

## Thirteenth JAX Translation Slice 13B: Harden Differentiable Slab Runtime

- Hardened the public differentiable slab runtime path without changing the existing `run_differentiable(initial_state=None)` signature:
  - added `Coupler.create_differentiable_state(prefill_missing=True)` as the public immutable runtime-state builder
  - added preflight validation for configured run sequence, slab-only components, runtime-state component coverage, initialized regridders, and initialized fractional masks
  - `run_differentiable()` now validates both internally created states and caller-provided initial states before entering `jax.lax.scan`
- Made real VerCOR regridders safe to use inside the traced differentiable runtime by caching identical-grid status at regridder construction time instead of recomputing a Python `bool(...)` from JAX arrays inside the scan body.
- Extended `tests/test_differentiable_coupler_runtime.py`:
  - normal four-slab `Coupler` construction through `register()`, `add_exchange()`, `set_components_run_sequence()`, and `initialize()`
  - real bilinear regridder coverage under `jax.jit`
  - gradient coverage for final ocean SST with respect to initialized runtime-state SST
  - clear validation errors for missing run sequence, unsupported non-slab components, missing regridders, and missing fractional masks
- `DEPENDENCIES.md` did not require changes because no new module-level dependency edge was introduced.

## Validation (Slice 13B, 2026-04-24)

- `conda run -n scipy pytest tests/test_differentiable_coupler_runtime.py -q`
  - passed
- `conda run -n scipy pytest tests/test_runtime_state.py tests/test_runtime_exchange.py tests/test_differentiable_coupler_runtime.py tests/test_bilinear_rectilinear_regridder.py tests/test_conservative_rectilinear_regridder.py tests/test_coupler_coverage.py -q`
  - passed
- `conda run -n scipy black vercor examples tests`
  - passed
  - note: Black emitted the existing Python 3.13 vs target-3.14 safety-check warning but completed successfully
- `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`
  - passed (`0`)
- `conda run -n scipy mypy vercor examples tests`
  - passed
- `conda run -n scipy pytest tests/ -q --fast`
  - passed
- `conda run -n scipy pytest tests/ -q`
  - passed

## Notes / Failed Approaches (Slice 13B)

- The first initialized-coupler JIT test failed because real regridders recomputed `has_identical_grids` inside `jax.lax.scan`, triggering `TracerBoolConversionError` from `bool(jnp.all(...))`. Caching the identical-grid result when the regridder is constructed keeps that branch static and preserves the existing identical-grid short-circuit behavior.

## Thirteenth JAX Translation Slice 13C: Mixed-Grid Differentiable Runtime Hardening

- Hardened the pure slab differentiable runtime for non-identical component grids without changing the public `run_differentiable(initial_state=None)` or `create_differentiable_state(prefill_missing=True)` APIs:
  - added mixed-grid four-slab coverage with ATM/LND on a 2x2 grid and OCN/ICE on a 3x3 grid
  - exercised real conservative OCN -> ATM remapping and real bilinear ATM -> OCN / OCN -> ICE remapping inside `jax.lax.scan`
  - kept external adapters, file I/O, plotting, Torch/CAMulator, xarray, NetCDF, and Veros object mutation outside the differentiable runtime path
- Strengthened differentiable-runtime preflight validation in `vercor/coupler.py`:
  - exported source fields must exist before entering the scan
  - source/data/incoming runtime fields must match their owning component grid shape
  - fractional masks must exist and match destination-grid shape
  - invalid caller-provided runtime states now fail with `CouplerError` before traced execution
- Extended `tests/test_differentiable_coupler_runtime.py` with:
  - mixed-grid `jax.jit`, `jax.grad`, and `jax.jvp` coverage
  - destination-shape assertions for conservative and bilinear exchange results
  - explicit validation coverage for missing source fields and fractional-mask shape mismatches
- `DEPENDENCIES.md` did not require changes because no new module-level dependency edge was introduced.

## Validation (Slice 13C, 2026-04-24)

- `conda run -n scipy pytest tests/test_differentiable_coupler_runtime.py -q`
  - passed
- `conda run -n scipy pytest tests/test_runtime_state.py tests/test_runtime_exchange.py tests/test_bilinear_rectilinear_regridder.py tests/test_conservative_rectilinear_regridder.py -q`
  - passed
- `conda run -n scipy black vercor examples tests`
  - passed
  - note: Black emitted the existing Python 3.13 vs target-3.14 safety-check warning but completed successfully
- `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`
  - passed (`0`)
- `conda run -n scipy mypy vercor examples tests`
  - passed
- `conda run -n scipy pytest tests/ -q --fast`
  - passed
- `conda run -n scipy pytest tests/ -q`
  - passed

## Notes / Failed Approaches (Slice 13C)

- The first mixed-grid test used ocean and atmosphere center coordinates whose inferred latitude bounds did not match, so initialization correctly failed the land-mask consistency check. The final test uses explicit matching cell bounds with different grid resolutions.

## Fourteenth JAX Translation Slice 14A: Data-Forcing Differentiable Runtime

- Extended the pure differentiable runtime beyond slab-only couplers while preserving the existing `run_differentiable(initial_state=None)` and `create_differentiable_state(prefill_missing=True)` APIs:
  - added `RuntimeStepInfo` as a JAX PyTree containing precomputed monthly interpolation indices/weights and daily time-slice indices
  - `Coupler.run_differentiable()` now scans over precomputed step metadata instead of deriving forcing times inside the traced body
  - runtime field sending now supports direct 2D fields, monthly interpolated forcing cubes, and daily time-sliced forcing arrays
  - differentiable component validation now accepts VerCOR slab components plus pure data-forcing adapters (`ERA5Atmosphere`, `ERA5Ocean`, `ERA5Land`, `ERAInterimOcean`, and `JCMLand`)
  - external runtime boundaries, including CAMulator-backed land forcing, were rejected at this stage; Slice 16A replaced this with unified runtime-state acceptance
- Added a pure data-component runtime step for `ERA5Atmosphere`:
  - combines imported land and sea surface temperatures into `total_surface_temperature`
  - keeps other supported data-forcing components as no-op steps whose runtime behavior is forcing replay through `send_component_fields()`
- Relaxed differentiable-runtime data-store validation so time cubes and auxiliary arrays can live in component data, while incoming/outgoing exchange fields are still required to match their component grid shape.
- Updated `DEPENDENCIES.md` to record that `vercor/runtime.py` now also depends on the pure data-forcing adapter layer.

## Tests Added / Updated (Slice 14A)

- Extended `tests/test_runtime_state.py` with:
  - `jax.jit` and reverse-mode gradient coverage for monthly runtime forcing interpolation
  - `jax.jit` and reverse-mode gradient coverage for daily runtime time slicing
- Extended `tests/test_differentiable_coupler_runtime.py` with:
  - a lightweight real data-component coupler using manually constructed `ERA5Ocean`, `ERA5Land`, and `ERA5Atmosphere` instances without asset downloads
  - gradient coverage through data-forcing replay into the atmosphere diagnostic
  - a data-to-slab runtime path using a real bilinear regridder
  - unsupported CAMulator land-boundary validation coverage

## Validation (Slice 14A, 2026-04-24)

- `conda run -n scipy pytest tests/test_runtime_state.py tests/test_runtime_exchange.py tests/test_differentiable_coupler_runtime.py -q`
  - passed
- `conda run -n scipy black vercor examples tests`
  - passed
  - note: Black emitted the existing Python 3.13 vs target-3.14 safety-check warning but completed successfully
- `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`
  - passed (`0`)
- `conda run -n scipy mypy vercor examples tests`
  - passed
- `conda run -n scipy pytest tests/ -q --fast`
  - passed
- `conda run -n scipy pytest tests/ -q`
  - passed

## Notes / Failed Approaches (Slice 14A)

- The first step-metadata implementation reused `get_field_time_slice()` with a JAX marker array during `run_differentiable()`. When called inside a jitted closure, converting that JAX scalar to `int` triggered `ConcretizationTypeError`; the final implementation computes daily indices with host scalar calendar logic before `jax.lax.scan`.
- The first `ERA5Atmosphere` data-runtime step added `total_surface_temperature` inside the scan body, changing the carry PyTree structure. The final implementation pre-seeds that diagnostic field in the runtime state and validates caller-provided states before traced execution.

## Fourteenth JAX Translation Slice 14B: Broaden Data-Forcing Runtime Coverage

- Hardened the pure differentiable data-forcing runtime with coverage for the remaining supported adapters without changing public component, coupler, exchange, regridder, or runtime-state APIs:
  - `ERAInterimOcean` monthly sea-surface-temperature forcing now has `run_differentiable()` coverage through real bilinear regridding into a slab atmosphere.
  - `JCMLand` daily land-surface-temperature forcing now has `run_differentiable()` coverage with `get_field_time_slice=True` into an ERA5-style data atmosphere.
  - Both paths are covered under `jax.jit` and reverse-mode gradients through selected forcing records.
- No production runtime changes were required; the existing `RuntimeStepInfo`, send-field selection, and supported-data-component dispatch paths already handled both adapters.
- `DEPENDENCIES.md` did not require changes because no new module-level dependency edge was introduced.

## Validation (Slice 14B, 2026-04-24)

- `conda run -n scipy pytest tests/test_differentiable_coupler_runtime.py tests/test_runtime_state.py tests/test_runtime_exchange.py -q`
  - passed
- `conda run -n scipy black vercor examples tests`
  - passed
  - note: Black emitted the existing Python 3.13 vs target-3.14 safety-check warning but completed successfully
- `conda run -n scipy pytest tests/test_differentiable_coupler_runtime.py tests/test_runtime_state.py tests/test_runtime_exchange.py -q`
  - passed after Black reformatted `tests/test_differentiable_coupler_runtime.py`
- `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`
  - passed (`0`)
- `conda run -n scipy mypy vercor examples tests`
  - passed
- `conda run -n scipy pytest tests/ -q --fast`
  - passed
- `conda run -n scipy pytest tests/ -q`
  - passed

## Notes / Failed Approaches (Slice 14B)

- No failed implementation approaches. This slice was test-first runtime hardening, and the existing differentiable data-forcing path passed without production changes.

## Fourteenth JAX Translation Slice 14C: Calendar-Aware Differentiable Forcing Runtime

- Hardened the pure differentiable data-forcing runtime calendar coverage without changing public component, coupler, exchange, regridder, or runtime-state APIs:
  - daily `get_field_time_slice=True` forcing now has `run_differentiable()` coverage under a no-leap model calendar that skips Gregorian February 29.
  - daily 360-day forcing now verifies the runtime step metadata selects the same no-leap Gregorian day index as the host `get_field_time_slice()` helper.
  - monthly `apply_time_interpolation=True` forcing now has year-boundary wrap coverage under `jax.jit` and reverse-mode gradients.
- No production runtime changes were required; the existing host-precomputed `RuntimeStepInfo` path already matched the host forcing calendar helpers.
- `DEPENDENCIES.md` did not require changes because no new module-level dependency edge was introduced.

## Validation (Slice 14C, 2026-04-24)

- `conda run -n scipy pytest tests/test_differentiable_coupler_runtime.py tests/test_runtime_state.py tests/test_runtime_exchange.py -q`
  - passed
- `conda run -n scipy black vercor examples tests`
  - passed
  - note: Black emitted the existing Python 3.13 vs target-3.14 safety-check warning but completed successfully
- `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`
  - passed (`0`)
- `conda run -n scipy mypy vercor examples tests`
  - passed
- `conda run -n scipy pytest tests/ -q --fast`
  - passed
- `conda run -n scipy pytest tests/ -q`
  - passed

## Notes / Failed Approaches (Slice 14C)

- No failed implementation approaches. This slice was test-first calendar hardening, and the existing differentiable forcing metadata path passed unchanged.

## Fifteenth JAX Translation Slice 15A: JAXGCM Pure Runtime Integration

- Extended the immutable differentiable runtime so initialized JAXGCM components can participate in `run_differentiable()` without mutating the public wrapper object:
  - added optional runtime payload support to `RuntimeComponentState`
  - added `JAXGCMRuntimePayload` for immutable JCM state and forcing carry data
  - added JAXGCM support detection and preflight payload validation
  - added a pure JAXGCM runtime step that prepares surface-temperature forcing, calls the existing JCM step function, maps JCM outputs back into runtime fields, and skips prediction history / file output
- Preserved explicit host/runtime boundaries at this stage; this was superseded by Slice 16A:
  - CAMulator and Veros were still rejected by `run_differentiable()`
  - JAXGCM imperative `step()` still owns host transfers and output writing outside the pure runtime path
- Pre-seeded JAXGCM runtime output fields, including 3D pressure from sigma-level count, so `jax.lax.scan` carries a stable PyTree structure.
- Updated `DEPENDENCIES.md` with the JAXGCM runtime payload dependency edge.

## Tests Added / Updated (Slice 15A)

- Extended `tests/test_runtime_state.py` with optional payload PyTree and `jax.jit` coverage.
- Extended `tests/test_differentiable_coupler_runtime.py` with lightweight patched JAXGCM runtime coverage:
  - `jax.jit` execution through `run_differentiable()`
  - reverse-mode gradient flow from sea-surface temperature into JAXGCM output fields
  - wrapper state/forcing immutability assertions
  - missing-initialization and missing-payload validation
  - explicit Veros boundary rejection coverage
- Hardened `tests/test_external_components_coverage.py` by clearing the `_map_jcm_output_fields` JIT cache after monkeypatching its helper globals.

## Validation (Slice 15A, 2026-04-24)

- `conda run -n scipy pytest tests/test_runtime_state.py tests/test_differentiable_coupler_runtime.py -q`
  - passed
- `conda run -n scipy pytest tests/test_differentiable_coupler_runtime.py tests/test_runtime_state.py tests/test_runtime_exchange.py tests/test_external_components_coverage.py -q`
  - passed
- `conda run -n scipy black vercor examples tests`
  - passed
  - note: Black emitted the existing Python 3.13 vs target-3.14 safety-check warning but completed successfully
- `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`
  - passed (`0`)
- `conda run -n scipy mypy vercor examples tests`
  - passed
- `conda run -n scipy pytest tests/ -q --fast`
  - passed
- `conda run -n scipy pytest tests/ -q`
  - passed

## Notes / Failed Approaches (Slice 15A)

- The first JAXGCM runtime test failed because `component.settings` is the component time-selection settings, not the coupler physical constants used by JAXGCM output mapping. The runtime step now receives `Coupler.settings` explicitly.
- The first JAXGCM scan attempt added output fields inside the scan body, changing the carry PyTree structure. The coupler now pre-seeds all JAXGCM runtime output fields before scanning.
- Running JAXGCM runtime tests before external coverage exposed a cached `jax.jit` monkeypatch hazard in `_map_jcm_output_fields`; the affected test now clears the JIT cache after monkeypatching.

## Fifteenth JAX Translation Slice 15B: Differentiable Integration Hardening

- Hardened the pure differentiable runtime preflight checks without changing public component, coupler, exchange, regridder, or runtime-state APIs:
  - caller-provided runtime states now fail before `jax.lax.scan` if slab components are missing required data fields needed to preserve a stable carry PyTree
  - imported fields must be present in both incoming and data stores before traced receive/update logic can run
  - exported fields must be present in component data before traced send logic can run
  - ERA5Atmosphere data-runtime diagnostics now require land, sea, and total surface temperature fields up front
  - JAXGCM runtime states now validate all pre-seeded 2D output fields plus the sigma-level pressure field before traced execution
- Clarified unsupported external boundary errors at this stage; this was superseded by Slice 16A:
  - CAMulator components were reported as explicit host/runtime boundaries
  - VerosGCM was reported as an explicit host/runtime boundary
- Extended differentiable integration coverage:
  - data-forcing ERA5Ocean now replays into a JAXGCM runtime component under `jax.jit`, `jax.grad`, and `jax.jvp`
  - missing slab required data, missing import/export data, and missing JAXGCM preseeded pressure now raise `CouplerError` before traced execution
  - CAMulatorLand and VerosGCM rejection tests also asserted their VerCOR boundary data remained JAX-backed; Slice 16A replaced these with runtime-acceptance tests
- `DEPENDENCIES.md` did not require changes because no new module-level dependency edge was introduced.

## Validation (Slice 15B, 2026-04-27)

- `conda run -n scipy pytest tests/test_differentiable_coupler_runtime.py tests/test_runtime_state.py tests/test_runtime_exchange.py -q`
  - passed
- `conda run -n scipy black vercor examples tests`
  - passed
  - note: Black emitted the existing Python 3.13 vs target-3.14 safety-check warning but completed successfully
- `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`
  - passed (`0`)
- `conda run -n scipy mypy vercor examples tests`
  - passed
- `conda run -n scipy pytest tests/ -q --fast`
  - passed
- `conda run -n scipy pytest tests/ -q`
  - passed

## Notes / Failed Approaches (Slice 15B)

- No failed implementation approaches. The slice tightened preflight validation and expanded integration coverage while preserving the existing differentiable runtime APIs.

## Sixteenth JAX Translation Slice 16A: Unified Runtime Component Path

- Replaced the divergent imperative/differentiable coupler execution split with a shared runtime-state sequence:
  - `Coupler.run()` now builds `RuntimeCouplerState` and advances components through exchange dispatch, runtime receive, runtime step, runtime send, and wrapper commit.
  - `Coupler.run_differentiable()` remains as a compatibility entrypoint but delegates each scanned component step to the same runtime component helper used by `run()`.
  - legacy `Component.receive_fields()` and `Component.send_fields()` now delegate to runtime receive/send helpers.
- Added a component-owned runtime interface on `Component`:
  - `create_runtime_payload()`
  - `prefill_runtime_state_fields()`
  - `validate_runtime_state()`
  - `step_runtime_state()`
  - `commit_runtime_state()`
- Moved component-specific runtime stepping out of `vercor/runtime.py`:
  - slab atmosphere, ocean, land, and sea-ice runtime steps now live in their component files
  - ERA5 atmosphere surface-temperature diagnostics now live in `vercor/components/data/era5_atmosphere.py`
  - JAXGCM runtime payload, validation, and immutable step logic now live in `vercor/components/external/jax_gcm.py`
  - CAMulatorGCM and VerosGCM expose host-backed runtime-step overrides in their own component files
- Removed CAMulator and Veros runtime-validation rejection paths. They now create and validate runtime state through the same component interface, with host internals remaining explicit component-owned boundaries.
- Updated runtime/coupler tests so `Coupler.run()` is verified against runtime dispatch/receive/step/send order, and CAMulatorLand / VerosGCM are accepted by runtime state creation instead of rejected.

## Validation (Slice 16A, 2026-04-27)

- `conda run -n scipy pytest tests/test_runtime_state.py tests/test_runtime_exchange.py tests/test_differentiable_coupler_runtime.py tests/test_coupler_coverage.py -q`
  - passed
- `conda run -n scipy pytest tests/test_component_base_coverage.py tests/test_component_models_coverage.py tests/test_external_components_coverage.py tests/test_camulator_component_kernels.py -q`
  - passed
- `conda run -n scipy pytest tests/test_runtime_state.py tests/test_runtime_exchange.py tests/test_differentiable_coupler_runtime.py tests/test_coupler_coverage.py tests/test_component_base_coverage.py -q`
  - passed
- `conda run -n scipy pytest tests/ -q --fast`
  - passed
- `conda run -n scipy black vercor examples tests`
  - passed
  - note: Black emitted the existing Python 3.13 vs target-3.14 safety-check warning but completed successfully
- `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`
  - passed (`0`)
- `conda run -n scipy mypy vercor examples tests`
  - passed

## Notes / Failed Approaches (Slice 16A)

- The first runtime-interface type pass broadened `Component.step()` to `CustomDateTime`, which made external component overrides appear too narrow to `mypy`. The final interface uses `datetime | ModelDateTime`, while component methods that accept `CustomDateTime` remain valid broader overrides.

## Sixteenth JAX Translation Slice 16B: Runtime Compatibility API Cleanup

- Completed the follow-up unified runtime cleanup:
  - added canonical `Coupler.create_runtime_state(prefill_missing=True)`
  - kept `create_differentiable_state()` and `run_differentiable()` as compatibility aliases over the unified runtime path
  - made `Coupler.run()` return the final `RuntimeCouplerState` while preserving wrapper commits on the host path
  - replaced `interpolate_and_dispatch_fields()` internals with runtime exchange dispatch plus a wrapper-field commit
- Removed remaining duplicated compatibility helpers:
  - deleted stale slab/data runtime validators from `vercor/coupler.py`
  - deleted `is_supported_differentiable_component()` and `step_slab_component_state()` from `vercor/runtime.py`
  - removed no-op CAMulatorGCM and VerosGCM `step_runtime_state()` overrides so both use the shared base host-boundary implementation
  - removed the old non-runtime `Component.send_fields()` interpolation/time-slice fallback
- Updated tests so compatibility methods are checked as runtime delegates, while `create_runtime_state()` is covered as the canonical state factory.
- `DEPENDENCIES.md` did not require changes because no new module-level dependency edge was introduced.

## Validation (Slice 16B, 2026-04-27)

- `conda run -n scipy pytest tests/test_runtime_state.py tests/test_runtime_exchange.py tests/test_differentiable_coupler_runtime.py tests/test_coupler_coverage.py -q`
  - passed
- `conda run -n scipy pytest tests/ -q --fast`
  - passed
- `conda run -n scipy black vercor examples tests`
  - passed
  - note: Black emitted the existing Python 3.13 vs target-3.14 safety-check warning but completed successfully
- `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`
  - passed (`0`)
- `conda run -n scipy mypy vercor examples tests`
  - passed
- `conda run -n scipy pytest tests/ -q`
  - passed

## Notes / Failed Approaches (Slice 16B)

- An intermediate lint run caught a stale unused `Any` import in `vercor/components/external/camulator.py` after removing its no-op runtime override. The import was removed and flake8 then reported `0`.

## Sixteenth JAX Translation Slice 16C: Unified Runtime Cleanup Completion

- Completed the unified-runtime cleanup requested after Slice 16B:
  - canonicalized runtime naming in private coupler helpers and runtime-state docstrings
  - kept `create_differentiable_state()` and `run_differentiable()` only as compatibility delegates
  - moved generic import/export/incoming runtime validation into `Component.validate_runtime_state()`
  - updated slab, ERA5 atmosphere, and JAXGCM runtime validators to layer component-specific checks on top of the shared base validation
  - removed remaining component-category validation branching from `Coupler`
  - made `Coupler` create component runtime state through `Component.to_runtime_component_state(prefill_missing=...)`
  - thinned `Component.receive_fields()` to the runtime receive delegate
- Added regression coverage that:
  - `run()` and `run_differentiable()` both use `_step_runtime_component()`
  - `vercor/runtime.py` does not own component-specific step helpers or external-component payload classes
- Updated `DEPENDENCIES.md` to describe `run()` / `create_runtime_state()` as the canonical runtime path with compatibility aliases.

## Validation (Slice 16C, 2026-04-27)

- `conda run -n scipy pytest tests/test_runtime_state.py tests/test_runtime_exchange.py tests/test_coupler_coverage.py tests/test_component_base_coverage.py tests/test_differentiable_coupler_runtime.py -q`
  - passed
- `conda run -n scipy black vercor examples tests`
  - passed
  - note: Black emitted the existing Python 3.13 vs target-3.14 safety-check warning but completed successfully
- `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`
  - passed (`0`)
- `conda run -n scipy mypy vercor examples tests`
  - passed
- `conda run -n scipy pytest tests/ -q --fast`
  - passed
- `conda run -n scipy pytest tests/ -q`
  - passed

## Notes / Failed Approaches (Slice 16C)

- No failed implementation approaches. The slice kept Veros, CAMulator, Torch, xarray, NetCDF, and file output as explicit component-owned host boundaries while unifying the VerCOR runtime interface around immutable runtime state.

## Sixteenth JAX Translation Slice 16D: Remove Legacy Differentiable API

- Completed the final unified-runtime API cleanup:
  - removed `Coupler.create_differentiable_state()`, `Coupler.run_differentiable()`, and `Coupler.interpolate_and_dispatch_fields()`
  - made `Coupler.run(initial_state=None, commit_wrappers=True)` the only execution entrypoint, with `commit_wrappers=False` using the scanned JAX runtime path
  - moved receive/send runtime field handling from `vercor/runtime.py` into `Component.receive_runtime_fields()` and `Component.send_runtime_fields()`
  - removed `receive_component_fields()`, `send_component_fields()`, and `step_component_state()` from `vercor/runtime.py`
  - thinned pure slab, ERA5 atmosphere, and JAXGCM `step()` wrappers so component math goes through `step_runtime_state()`
- Renamed the large runtime integration test module from `tests/test_differentiable_coupler_runtime.py` to `tests/test_coupler_runtime.py`.
- Updated regression coverage so removed legacy API names are absent from `Coupler` and generic runtime.py no longer owns component receive/send/step dispatch.
- Updated `DEPENDENCIES.md` to describe `vercor/runtime.py` as immutable state plus generic exchange dispatch only.

## Validation (Slice 16D, 2026-04-27)

- `conda run -n scipy black vercor examples tests`
  - passed
  - note: Black emitted the existing Python 3.13 vs target-3.14 safety-check warning but completed successfully
- `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`
  - passed (`0`)
- `conda run -n scipy mypy vercor examples tests`
  - passed
- `conda run -n scipy pytest tests/ -q --fast`
  - passed
- `conda run -n scipy pytest tests/ -q`
  - passed

## Notes / Failed Approaches (Slice 16D)

- The first thinned slab-ocean wrapper broke a direct unit test that calls `Ocean.step()` before initialization; the runtime step now preserves the old no-op behavior when SST is absent, while validated coupler runtime states still reject missing required SST before scans.
- The first JAXGCM external coverage update still monkeypatched the old `do_jcm_steps()` host path. The test now seeds the runtime payload and `_step_function` directly, matching the canonical component runtime path.

## Sixteenth JAX Translation Slice 16E: Unified Runtime Audit Completion

- Completed the unified-runtime audit / hardening pass requested after Slice 16D:
  - confirmed the legacy divergent public APIs remain absent (`run_differentiable`, `create_differentiable_state`, and `interpolate_and_dispatch_fields`)
  - confirmed generic runtime.py remains limited to immutable runtime state plus exchange dispatch, with component-specific runtime stepping kept in component modules
  - added direct `Coupler.run(..., commit_wrappers=False)` coverage for `CAMulatorGCM` so CAMulator atmosphere, CAMulator land, Veros, slab, data, and JAXGCM adapters are all covered through the canonical runtime-state path
- No production code changes were required; the slice only added targeted regression coverage.
- `DEPENDENCIES.md` did not require changes because no new module dependency edge was introduced.

## Validation (Slice 16E, 2026-04-27)

- `conda run -n scipy pytest tests/test_coupler_runtime.py::test_run_accepts_camulator_gcm_runtime_boundary -q`
  - passed
- `conda run -n scipy pytest tests/ -q --fast`
  - passed
- `conda run -n scipy black vercor examples tests`
  - passed
  - note: Black emitted the existing Python 3.13 vs target-3.14 safety-check warning but completed successfully
- `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`
  - passed (`0`)
- `conda run -n scipy mypy vercor examples tests`
  - passed
- `conda run -n scipy pytest tests/ -q`
  - passed

## Notes / Failed Approaches (Slice 16E)

- No failed implementation approaches. The audit found only the missing direct CAMulatorGCM runtime-acceptance coverage, which is now locked by a lightweight test that avoids model-file, Torch, and xarray execution boundaries.

## Sixteenth JAX Translation Slice 16F: Runtime-First Component API Cleanup

- Completed the remaining wrapper-era component cleanup:
  - removed `Component.export_fields()` and `Component.import_fields()`
  - made `Component.initialize()` a concrete no-op default
  - made `Component.step()` a thin compatibility adapter over `step_runtime_state()`
  - made the default `Component.step_runtime_state()` a no-op immutable runtime transition
- Replaced the last production `component.import_fields(...)` call in `Coupler._commit_runtime_incoming_fields()` with direct assignment of runtime-built `Shared` incoming fields.
- Removed redundant no-op `initialize()` / `step()` implementations from data-forcing components and redundant `step()` wrappers from slab components and `ERA5Atmosphere`.
- Moved host-backed adapter stepping into component-owned runtime overrides:
  - `CAMulatorLand.step_runtime_state()`
  - `CAMulatorGCM.step_runtime_state()`
  - `VerosGCM.step_runtime_state()`
- Updated `JAXGCM.step_runtime_state()` so host bookkeeping, prediction storage, logging, and optional output happen when `time` and `coupler` are supplied, while scanned runtime execution remains side-effect free.
- Extended regression coverage so the removed component import/export API is absent, base `step()` delegates through runtime state, `Coupler` no longer calls `import_fields`, and external runtime stepping remains in component files.
- `DEPENDENCIES.md` did not require changes because the module dependency order and ownership descriptions stayed valid.

## Validation (Slice 16F, 2026-04-27)

- `conda run -n scipy pytest tests/test_component_base_coverage.py tests/test_runtime_state.py tests/test_coupler_runtime.py tests/test_external_components_coverage.py tests/test_camulator_component_kernels.py -q`
  - passed
- `conda run -n scipy black vercor examples tests`
  - passed
  - note: Black emitted the existing Python 3.13 vs target-3.14 safety-check warning but completed successfully
- `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`
  - passed (`0`)
- `conda run -n scipy mypy vercor examples tests`
  - passed
- `conda run -n scipy pytest tests/ -q --fast`
  - passed
- `conda run -n scipy pytest tests/ -q`
  - passed

## Notes / Failed Approaches (Slice 16F)

- The first focused test run exposed old tests that constructed host-backed external components through `__new__()` and called `step()` directly, bypassing the base runtime state expected by the new compatibility wrapper. The tests now call `step_runtime_state()` with explicit `RuntimeComponentState` objects for those patched adapter-boundary cases.
- The first `mypy` pass rejected `CAMulatorLand.step_runtime_state()` because it accepted the broader `CustomDateTime` alias while `commit_runtime_state()` expects `ModelDateTime`; the annotation was narrowed to the base runtime contract.

## Wrapper Runtime Startup Prefill Fix

- Fixed the default host/wrapper `Coupler.run()` startup path so it creates the same prefilled and primed runtime state as the scanned runtime path when no explicit initial state is supplied.
- Preserved strict validation for caller-supplied `initial_state` objects; only the internally-created default state is prefilled.
- Added regression coverage for an initialized slab coupler whose wrapper incoming fields start empty but whose default `run()` still succeeds and commits imported runtime fields.
- Updated the wrapper-run coverage expectation to include startup outgoing-field priming before the first step.

## Validation (Wrapper Runtime Startup Prefill Fix, 2026-04-27)

- `conda run -n scipy pytest tests/test_coupler_runtime.py::test_initialized_slab_coupler_wrapper_run_prefills_missing_imports -q`
  - passed
- `conda run -n scipy black vercor examples tests`
  - passed
  - note: Black emitted the existing Python 3.13 vs target-3.14 safety-check warning but completed successfully
- `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`
  - passed (`0`)
- `conda run -n scipy mypy vercor examples tests`
  - passed
- `conda run -n scipy pytest tests/test_coupler_runtime.py -q`
  - passed
- `conda run -n scipy pytest tests/ -q --fast`
  - passed
- `conda run -n scipy pytest tests/ -q`
  - passed
- `MPLCONFIGDIR=/tmp/vercor-mplconfig MPLBACKEND=Agg conda run -n scipy python examples/run_slab_driver.py`
  - passed
- `MPLCONFIGDIR=/tmp/vercor-mplconfig MPLBACKEND=Agg conda run -n scipy python examples/run_data_driver.py`
  - passed

## Notes / Failed Approaches (Wrapper Runtime Startup Prefill Fix)

- The first new regression test asserted the full example-driver wind imports against the smaller existing slab test helper, which only imports heat fluxes into `OCN`; the final assertion uses the helper's actual imported fields while preserving the same missing-prefill failure mode.
- The first full-suite run exposed that the wrapper-run coverage test was still expecting no startup priming. The test now records the intentional initial `send_runtime_fields()` priming events before step dispatch.

## Unified Runtime Test Audit and Cleanup

- Audited `tests/` against the current canonical runtime API:
  - `Coupler.run(...)`
  - `Coupler.create_runtime_state(...)`
  - `Component.step_runtime_state(...)`
  - `Component.receive_runtime_fields(...)`
  - `Component.send_runtime_fields(...)`
- Confirmed removed wrapper-era APIs are referenced only by negative regression guards:
  - `run_differentiable`
  - `create_differentiable_state`
  - `interpolate_and_dispatch_fields`
  - `import_fields`
  - `export_fields`
  - `receive_component_fields`
  - `send_component_fields`
  - `step_component_state`
  - `step_slab_component_state`
  - `is_supported_differentiable_component`
- Rechecked patched external-component tests that construct components through `__new__()`:
  - host-backed Veros and CAMulator boundary tests now call `step_runtime_state()` with explicit `RuntimeComponentState` objects where they exercise runtime behavior
  - remaining direct `step()` tests cover the current compatibility wrapper, not removed APIs
- No stale behavior tests were found, so no unit tests were removed.
- Confirmed the production NumPy-boundary audit still limits direct NumPy imports to explicit host/type/output boundary modules:
  - `vercor/components/base.py`
  - `vercor/tools.py`
  - `vercor/types.py`

## Validation (Unified Runtime Test Audit and Cleanup, 2026-04-27)

- `conda run -n scipy pytest tests/ -q --fast`
  - passed
- `conda run -n scipy pytest tests/ --collect-only -q`
  - passed
- `conda run -n scipy black vercor examples tests`
  - passed
  - note: Black emitted the existing Python 3.13 vs target-3.14 safety-check warning but completed successfully and left 83 files unchanged
- `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`
  - passed (`0`)
- `conda run -n scipy mypy vercor examples tests`
  - passed
- `conda run -n scipy pytest tests/ -q --fast`
  - passed after lint/type checks
- `conda run -n scipy pytest tests/ -q`
  - passed

## Notes / Failed Approaches (Unified Runtime Test Audit and Cleanup)

- No failed implementation approaches. The audit found only intentional absence guards for removed APIs, so deleting tests would have weakened regression coverage rather than removing stale behavior coverage.

## Test-Only Coupler Runtime Wrapper Removal

- Removed the private `Coupler._dispatch_runtime_fields()` and `Coupler._commit_runtime_incoming_fields()` compatibility wrappers from production code.
- Moved their remaining coverage-only behavior into local helpers in `tests/test_coupler_coverage.py`:
  - exchange dispatch now calls the canonical `dispatch_component_exchanges()` runtime function directly
  - wrapper incoming-field commit logic is now test-local for compatibility assertions only
- Extended removed-API regression coverage so `Coupler` is asserted not to expose either private test-only wrapper.
- No `DEPENDENCIES.md` update was required because this removed dead compatibility methods without changing module dependency order.

## Validation (Test-Only Coupler Runtime Wrapper Removal, 2026-04-27)

- `conda run -n scipy black vercor examples tests`
  - passed
  - note: Black emitted the existing Python 3.13 vs target-3.14 safety-check warning and left 83 files unchanged
- `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`
  - passed (`0`)
- `conda run -n scipy mypy vercor examples tests`
  - passed
- `conda run -n scipy pytest tests/test_coupler_coverage.py tests/test_coupler_runtime.py tests/test_runtime_exchange.py -q`
  - passed
- `conda run -n scipy pytest tests/ -q --fast`
  - passed
- `conda run -n scipy pytest tests/ -q`
  - passed

## Notes / Failed Approaches (Test-Only Coupler Runtime Wrapper Removal)

- No failed implementation approaches. The cleanup stayed limited to private test-only compatibility wrappers and kept public/protocol/lifecycle APIs intact.

## Compile Cache and Safe Donation Runtime Audit

- Added `Coupler.compile_runtime(donate_state=True)` as an explicit reusable compiled scanned-runtime helper:
  - keeps `Coupler.run()` and `create_runtime_state()` behavior unchanged
  - wraps `run(state, commit_wrappers=False)` in `jax.jit`
  - donates the outer `RuntimeCouplerState` only when requested
  - documents that donated input states are consumed and must not be read after invocation
- Hardened runtime field stores for donation safety:
  - `RuntimeFieldStore.from_mapping()` and `RuntimeFieldStore.set()` now materialize stored leaves with `jnp.array(..., copy=True)`
  - this prevents repeated field references from producing duplicate donated buffers
- Added `tests/test_runtime_compile_cache.py`:
  - verifies repeated compiled runtime calls with the same treedef/shapes reuse the JIT cache
  - verifies changed runtime array values do not trigger a new compile
  - verifies donated runtime execution succeeds with fresh consumed states
  - verifies non-donating compiled runs preserve runtime-state treedef and static field names

## Validation (Compile Cache and Safe Donation Runtime Audit, 2026-04-28)

- `conda run -n scipy pytest tests/test_runtime_compile_cache.py -q`
  - passed
- `conda run -n scipy pytest tests/ -q --fast`
  - passed
- `conda run -n scipy black vercor examples tests`
  - passed
  - note: Black emitted the existing Python 3.13 vs target-3.14 safety-check warning and left 84 files unchanged on the final run
- `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`
  - passed (`0`)
- `conda run -n scipy mypy vercor examples tests`
  - passed
- `conda run -n scipy pytest tests/ -q`
  - passed

## Notes / Failed Approaches (Compile Cache and Safe Donation Runtime Audit)

- The first donation test failed with XLA's duplicate donated-buffer error because several runtime fields reused the same JAX buffer. The runtime field store now copies leaves on insertion so canonical runtime states can be donated safely.
- The first `mypy` run rejected direct equality checks on JAX `PyTreeDef` objects in the new test. The final assertion compares the stable treedef representation instead.

## Runtime API Simplification Audit

- Completed the conservative runtime API cleanup:
  - confirmed `Shared` and `TimedNamedArray` are host wrapper / output metadata containers, not scanned-runtime integration state
  - documented that wrapper boundary in `vercor/components/base.py` while keeping both classes exported for compatibility
  - made `Component.validate_runtime_state()` and its helper methods derive expected 2D field shapes from `self.grid.shape`
  - removed the redundant `expected_shape` argument from slab, ERA5 atmosphere, and JAXGCM validation overrides
  - updated `Coupler._validate_runtime_state()` to call component validation without passing shape metadata already owned by the component
- Added regression coverage for:
  - direct component runtime validation without caller-provided shape metadata
  - shape errors using the component's own grid shape
  - scanned runtime states continuing to use `RuntimeFieldStore` for data, incoming, and outgoing fields
- `DEPENDENCIES.md` did not require changes because no module dependency edge changed.

## Validation (Runtime API Simplification Audit, 2026-04-28)

- `conda run -n scipy pytest tests/test_component_base_coverage.py tests/test_coupler_runtime.py tests/test_runtime_state.py -q`
  - passed
- `conda run -n scipy pytest tests/ -q --fast`
  - passed
- `conda run -n scipy black vercor examples tests`
  - passed
  - note: Black emitted the existing Python 3.13 vs target-3.14 safety-check warning and left 84 files unchanged
- `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`
  - passed (`0`)
- `conda run -n scipy mypy vercor examples tests`
  - passed
- `conda run -n scipy pytest tests/ -q`
  - passed

## Notes / Failed Approaches (Runtime API Simplification Audit)

- No failed implementation approaches. The cleanup stayed conservative and did not remove the public `Shared` / `TimedNamedArray` compatibility surface.
