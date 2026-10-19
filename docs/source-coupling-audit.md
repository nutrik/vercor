# Source and coupling audit

This is the durable audit of the approved 2026-10-19 source and coupling
cleanup on `feat/veris-seaice`, from baseline `e1d417a` through Tasks 1–8.
Task 8 changes tests and documentation only. It records completed changes,
retained owners, exact assertion migrations, failures and validation limits.
The final combined coverage, fresh artifacts and native gallery acceptance
are a separate Task 9 gate; their results must be recorded here on completion.
Historical archives are unchanged.

The source inventory remains 179 Python modules. There are 104 `test_*.py`
modules after the new CAMulator initialization and component substitution
suites. The inventory below covers every current source module, including
CLI/gallery entry points and package initializers. The complete dependency
layers and public/private manifests remain in `DEPENDENCIES.md`
and `docs/api-architecture-review.md`.

## Approved decisions and costs

The following four recorded rulings remain authoritative; they are copied
verbatim so the decision and cost survive removal of the temporary workspace.

Ruling: Work in the existing requested feat/veris-seaice checkout — the user selected this branch and IDE workspace, the only tracked edits are this task's approved documentation, and the baseline is verified; creating another branch or force-sharing the checked-out branch would add avoidable coordination risk — cost if wrong: approved edits remain visible in the user's current checkout rather than a separate directory.

Ruling: Treat artifact builds, strict Sphinx, combined coverage, and installed-package/gallery acceptance as the final Task 9 gate; retain focused, static, fast/full stock, and applicable AD gates before each implementation commit — the approved plan explicitly assigns comprehensive artifact acceptance to Task 9, while the specification verification paragraph describes it collectively before delivery; repeating artifact builds per small commit adds no distinct acceptance guarantee — cost if wrong: an intermediate local commit may reveal an artifact/documentation issue only in Task 9, and must be fixed before final completion.

Ruling: Require an explicit gallery vapor-channel declaration validated against loaded native input and prediction layouts — the published available configuration declares Qtot only and no vapor-trained checkpoint/channel exists locally, so automatically selecting a vapor name would invent a physical capability — cost if wrong: gallery callers must supply a compatible checkpoint and its actual vapor channel rather than rely on an automatic selection.

Ruling: Retain existing inference-only torch.jit.trace after all new bootstrap validation, while initial-field mapping makes no model/stepper calls — tracing invokes the native model on zeros but no prognostic state or cursor assignment is demonstrated; removing an existing inference optimization without native/performance evidence exceeds the proven correction — cost if wrong: a native model with hidden forward side effects may still need tracing changes; native CREDIT/performance validation is unavailable and this limit remains explicit.

## Final removal decisions

| Retired mechanism | Actual former caller | Current owner and decision |
| --- | --- | --- |
| `Coupler._initialize_runtime` | Runtime/setup test fixtures | Public `initial_state`/`run`; private preparation tests use `_ensure_prepared`. No alternate initializer remains. |
| Optional contract rebuilding and unused exchanges in `coupler_state` | Preparation | `PreparedCoupling.contracts` is mandatory; no rebuilding fallback. |
| `build_exchange_contracts(validate_endpoints=...)` | Initialization and contract tests | Always validate both endpoint names before contracts, fan-in and setup. |
| `check_not_empty_import_export_lists` | Guarded runtime validation | Unreachable no-op removed; declarations and required fields are still validated. |
| Preseeded `topology_maps` | Initialization/topology and test fixture overrides | Declared factories build once; capability checks precede topology policy patches. |
| `prepare_runtime_state(validate_state=...)` | Test-only bypass | Supplied states always validate before current masks refresh. |
| Raw-component topology fallback | Topology context construction | Prepared bindings expose original author components directly. |
| Private structural step discovery/arity adaptation | Component preparation | Bind the public three-argument `step`; convenience callable normalization has its own retained owner. |
| JCM snapshot setup `_state` fallback | Snapshot writer | Require final runtime payload; missing payload raises before writing. |
| Guaranteed JCM/Veros policy-attribute defaults | Native setup resources and incomplete fixtures | Constructor-owned attributes read directly; fixtures now complete. Optional native capabilities remain explicit. |
| Private bilinear/conservative factories | Public facade and tests | Public factories directly construct concrete private regridders; public `radius_km` preserved. |
| `flatten_exchange_fields` | Runtime contracts and tests | Existing `fields._flatten_field_items` owns flatten order. |
| `tree_as_real_dtype`, `tree_unwrap_leading_dims`, `tree_stack` | Tests only | Production `tree_as_runtime_dtype` and `tree_mean`; category/complex precision and mean semantics preserved. |
| `canonicalize_surface_field` | Tests only | Explicit time-last layout functions; static native land transpose at its owner. |
| `initialize_camulator_runtime_fields` | Tests only | Component preparation owns stores; actual native decoded startup fields come from `SetupResult`. |
| `prepare_veris_forcing` | Tests only | `prepare_veris_forcing_pack(...).native_forcing`; same forcing, conservation and AD assertions. |
| JCM default-parameter convenience | Tests only | Native parameter attributes and production `change_jcm_parameter_values`. |
| JCM prediction-output convenience | Tests only | Runtime provider/snapshot extraction from payload. The private extraction implementation remains necessary. |
| `forcing_index.ForcingYearType` | Public inventory | Canonical `calendar.YearType`; no compatibility alias. |
| Seven unused recipe tuples and their gallery routes | Prescribed/slab consumers with no matching physical read | Removed in Task 7; genuine heat feedback and explicit diagnostic routes retained. Exact names below. |

The seven retired recipes are `ATMOSPHERE_TO_DATA_OCEAN_FIELDS`,
`ATMOSPHERE_TO_JCM_LAND_FLUX_FIELDS`, `ATMOSPHERE_TO_LAND_BASIC_FIELDS`,
`JCM_ATMOSPHERE_TO_SLAB_OCEAN_FIELDS`, `SLAB_ATMOSPHERE_TO_OCEAN_FIELDS`,
`LAND_TO_ATMOSPHERE_SOIL_FIELDS`, and `SEAICE_TO_OCEAN_FIELDS`.

## Necessary adapters and remaining candidates

These were reviewed against current callers, including short forwarding
functions. None is removed solely because it is short, reflective or called an
adapter. There is no unresolved removal promised by this audit.

| Retained mechanism | Current caller | Purpose and retention decision |
| --- | --- | --- |
| Preparation `runtime_state_from_components` / `validate_runtime_state` | `create_runtime_state`, supplied-state preparation, facade/backend boundary | Unpack the one prepared binding into lower-level state/schema owners; keep initial, supplied and backend validation consistent. |
| `_ComponentBinding`, structural checks, `_copy_owned_pytree` | Runtime initialization, lifecycle and step execution | Author identity, frozen declarations, payload ownership and native host-copy boundary. Reflection validates untrusted structural authors and native copy capability. |
| `_callable_wrappers` | `CallableComponent` | Public convenience callable normalization; structural component steps do not use this path. |
| `runtime_fields` and `declared_runtime_field_names` | `components.runtime_execution`; `_runtime.validation` | Convert immutable stores to callback mappings; centralize declared field union. Not initialization alternatives. |
| `receive_runtime_fields` / `send_runtime_fields` | `_runtime.driver` | Distinct receive ownership, time-selected exports and finite-support checks. |
| Veros `copy_state`, `pure`, solver cache capability detection | Veros setup/runtime stepping | Stock host copy versus fork native copy, component solver detachment and exception restoration. Version strings cannot replace capability checks. |
| `extract_veros_runtime_sst` | `extract_veros_runtime_surface_fields` | Native current-timestep surface extraction with Celsius/K and layout contract; retain alongside current/SSH extraction. |
| `load_veros_data_snapshot`, `make_veros_data_ocean` | `_veris.make_veris_verosdata` | One native setup snapshot and prescribed monthly surface factory on the exact shared wet grid; no prognostic feedback. |
| JCM optional vertical-layer/float0/struct accessors | Native provider/snapshot coordinate and variable extraction | Native coordinate alternatives and differentiation sentinel conversion; no mutable setup-state fallback. |
| JCM epoch presence guard | `JAXGCMSetupState.setup` | Detect clock mismatch across setup reuse; not a default for guaranteed policy attributes. |
| `_runtime_veris_constants` | Veris initialize/advance and parameter replacement | Reapply native dependent constants and coupler-owned freshwater density. |
| CAMulator input/prediction converters and shared meteorology mapper | Bootstrap and runtime prediction | Distinct native inverse transforms and channel layouts, one traced pressure/height/density/thermodynamics owner. |
| Native output-provider factories | JCM/Veros/Veris/CAMulator setup factories | Bind setup resources to providers that sample immutable runtime payloads; native snapshots remain separate from periodic means. |
| Output session initialization/schema helpers | Execution coordinator and provider sampling | Opt-in output lifetime, shape/dtype consistency and host transfer; no backend-owned I/O. |
| `_host_arrays`, dtype allocation functions | Diagnostics/output and physical setup/kernels | Explicit host-transfer versus traced arrays, float32/float64 and int32 policy. |
| Public regridding factories and private concrete classes | Exchange factories / runtime preparation | Public stable signatures; separate bilinear interpolation/vector and conservative integral behavior. |
| `resolve_lazy_export`, `lazy_export_names`, `setups.__getattr__` | Sole `vercor.setups` export table | Optional dependencies remain lazy with deterministic export inventory. |
| `_setup_runner._load_setup`, CLI discovery/copy | Installed `vercor run` and CLI commands | Child import isolation and explicit setup contract, distinct from in-process gallery construction. |
| Field/layout/vector/vertical-coordinate helpers | Contracts, grids, setup conversions and CESM closure | Domain validation, orientation and physical equations; compact implementations remain useful owners. |

## Physical and native boundaries retained

For source coverage `s`, target binary support `b`, target fraction `t` and
remapper `R`, the dispatcher alone forms `N = R(where(s > 0, x, 0) * s)` and
`d = R(s)`. Intensive receive is `N/d`, source flux is `N*b`, and target flux
is `(N/d)*t`, with finite zero values/derivatives at empty overlap. Vector
products are formed before remapping/rotation. Preweighted Veris feedback
uses binary source support. Static support, current fractions, current route
diagnostics and last-dispatch received support remain separate.

Temperatures are K (Veros Celsius converted once); vapor humidity is kg/kg;
pressure Pa; reference height/SSH m; winds/currents m/s; rain/snow kg/m²/s
positive downward; heat W/m² positive into surface. Salinity retains native
g/kg. Actual salt mass flux and virtual salinity forcing remain distinct:
`(-S_ref*M + 1000*J)/rho_fw` is the latter. Active NaNs and all infinities
fail; inactive NaNs are neutralized before nonlinear arithmetic. Endpoint
coverage reconciliation remains limited to machine precision.

Slab atmosphere sensible/latent heat now follows downward sign; ocean
restoring damps SST anomalies; land moisture responds with the correct
latent-flux sign before clipping. Coefficients were not retuned. CAMulator
radiation divides accumulated energy by `3600 * lead_time_periods` once and
adds surface emission once to signed net longwave. Startup decodes the latest
native input frame, requires exact-time TS/FSNS/FLNS diagnostics and an
explicit vapor channel in input and output layouts. `Qtot` is total water and
is rejected as vapor. Existing inference tracing is retained after validation.

Prescribed ERA5/ERA-Interim oceans, ERA5 atmosphere/land and CAMulator land
export without fictitious feedback. Slab atmosphere reads SST, slab ocean
reads two turbulent heat fluxes, and slab land reads latent heat. The data
gallery explicitly declares and displays diagnostic imports. The substitution
matrix tests identical physical forcing through data, callable JAX, structural
JAX and host producers, including differing scalar grids, rotated vectors,
fractional coasts, empty/reactivated support and stored received masks.
Host producers are rejected by forced JAX. Matching field names alone does not
certify interchangeable native geometry, species or execution capabilities.


## Test consolidation and surviving assertions

Every row was recorded before deletion. Test filenames are relative to `tests/`; unqualified surviving names are in the same file. Source-only checks have no backend, precision or installed-artifact lane.

| Deleted test or assertion | Exact assertion | Surviving owner | Preserved distinctions |
| --- | --- | --- | --- |
| `test_api_boundaries.py::test_root_api_is_core_only_after_boundary_redesign` | Exact six root exports; advanced names absent from root exports; actual lookup rejects bilinear, conservative, Regridder and RegridderFactory. | `test_v0_4_public_api.py::test_root_exports_exactly_the_six_primary_symbols` (exact manifest and both protocol/factory lookup checks); `test_manifested_symbols_are_absent_from_every_non_owner_public_module` (bound names); `test_api_boundaries.py::test_public_facades_hide_private_implementation_modules` (bilinear/conservative lookup) | No precision/backend/artifact parameter. Actual `hasattr` checks reject lazy aliases; `vars(module)` alone does not. |
| `test_api_boundaries.py::test_top_level_exports_public_orchestration_and_component_author_api` | Exact root/component manifests and identity of Component, ComponentSpec, SetupContext, StepContext, StepResult, SetupResult, TransferPolicy, DataComponent and CallableComponent. | `test_v0_4_public_api.py::test_supported_public_module_has_exact_manifest; test_root_exports_exactly_the_six_primary_symbols; test_api_boundaries.py::test_components_package_exports_only_component_author_contracts` | Private implementation identities remain in the latter test; no protocol, lazy-import or artifact check is removed. |
| `test_coupler_coverage.py::test_run_api_does_not_expose_state_donation_option` | Coupler.run signature excludes donate_state. | `test_api_architecture_review.py::test_static_public_signature_contract_is_complete_and_matches_source` | Exact complete signature; profiling CLI argument absence is retained separately. |
| `test_runtime_run.py::test_run_api_does_not_expose_state_donation` | Coupler.run signature excludes donate_state. | `test_api_architecture_review.py::test_static_public_signature_contract_is_complete_and_matches_source` | Exact complete signature; real scanned rollout and treedef assertions retained. |
| `test_public_api_contracts.py::test_run_state_remains_a_jax_pytree` | tree_leaves(state) is nonempty; scalar temperature seed expands to grid shape. | `test_v0_4_routes_and_state.py::test_opaque_run_state_remains_a_jax_pytree; test_runtime_state.py::test_runtime_coupler_state_restores_component_index_cache_after_pytree_roundtrip; test_runtime_component_and_coupler_state_are_pytrees; test_public_api_contracts.py::test_run_state_exposes_component_state_view_not_runtime_state` | Retained round-trip and JIT state update cover registration; identical scalar seed and shape assertion already exist in the retained public-state test. |
| `test_dtypes.py::test_numpy_and_jax_helpers_agree_on_dtype_policy` | np.dtype(jax_real_dtype(True)) == np.float64 and np.dtype(jax_index_dtype(True)) == np.int32. | `test_dtype_policy_enable_x64_maps_real_arrays_to_float64; test_index_dtype_is_int32_for_both_real_precision_modes` | Exact expressions already present; float32 allocation, JIT index, and normalization AD stay distinct. |
| `test_bilinear_rectilinear_regridder.py::test_regridder_has_identical_grids_true_for_equal_coords` | Equal copied NumPy longitude/latitude arrays set has_identical_grids is True. | `test_regridder_identical_grid_skips_interpolator_construction` | Same coordinates; stronger constructor-not-called assertion, concrete class and missing interpolator checks retained. |
| `test_bilinear_rectilinear_regridder.py::test_regridder_has_identical_grids_false_for_different_coords` | Different coordinates set has_identical_grids is False. | `test_regridder_non_identical_grid_constructs_interpolator` | Same coordinates and stronger exactly-one construction assertion. Bilinear spacing and conservative shape distinctions retained independently. |
| `test_bilinear_rectilinear_regridder.py::test_regridder_identical_grid_passthrough_does_not_use_identity_helper` | Private source lacks the obsolete _IdentityInterpolator spelling. | `test_regridder_identical_grid_scalar_short_circuit_returns_input_object; test_regridder_identical_grid_scalar_short_circuit_with_jax_backed_coords` | Replaced obsolete spelling-only guarantee with existing exact-object NumPy/JAX passthrough; bilinear vector passthrough and conservative scalar-only remain distinct. |
| `test_conservative_rectilinear_regridder.py::test_regridder_has_identical_grids_true_for_equal_coords` | Equal copied NumPy longitude/latitude arrays set has_identical_grids is True. | `test_regridder_identical_grid_skips_remapper_construction` | Same coordinates; stronger constructor-not-called assertion, concrete class and missing interpolator checks retained. |
| `test_conservative_rectilinear_regridder.py::test_regridder_has_identical_grids_false_for_different_coords` | Different coordinates set has_identical_grids is False. | `test_regridder_non_identical_grid_constructs_remapper` | Same coordinates and stronger exactly-one construction assertion. Bilinear spacing and conservative shape distinctions retained independently. |
| `test_conservative_rectilinear_regridder.py::test_regridder_identical_grid_passthrough_does_not_use_identity_helper` | Private source lacks the obsolete _IdentityInterpolator spelling. | `test_regridder_identical_grid_scalar_short_circuit_returns_input_object; test_regridder_identical_grid_scalar_short_circuit_with_jax_backed_coords` | Replaced obsolete spelling-only guarantee with existing exact-object NumPy/JAX passthrough; bilinear vector passthrough and conservative scalar-only remain distinct. |
| `test_v0_4_public_api.py::test_constructor_rejects_invalid_nonempty_run_order` | Duplicate MODEL and unknown UNKNOWN error patterns. | `test_constructor_rejects_invalid_run_order_kinds` | Both original parameters join None, string, set, generator, mixed object and empty name; all eight original error cases survive. |
| `test_v0_4_routes_and_state.py::test_supplied_state_enforces_full_private_alignment_schema` | All eleven private-alignment corruption errors. | `test_supplied_foreign_state_is_strictly_validated` | Concatenate the original parameter tuples, preserving all fifteen IDs/messages; driver/backend/JIT and compatible-foreign cases remain separate. |
| `test_runtime_state.py::test_runtime_modules_use_current_domain_owners` (partial); `test_runtime_facade_boundaries.py::test_runtime_facade_builds_and_executes_the_validated_plan_directly` (partial) | Three private-source donate_state absence assertions. | `test_api_architecture_review.py::test_static_public_signature_contract_is_complete_and_matches_source` | Only retired option spelling is removed; other coordinator ownership, import-cycle and execution behavior checks stay. |

## Earlier assertion migrations: Task 2

Two tests were renamed/migrated; no behavioral test was simply deleted:

| Previous test or fixture | Surviving behavior |
| --- | --- |
| test_build_exchange_topology_does_not_mutate_existing_mappings | test_prepared_topology_keeps_declared_route_maps_immutable checks only declared route IDs, identity target support, and rejection of map writes. Seeded unrelated-route preservation is retired with map injection. Caller-map copy ownership is checked by test_runtime_topology_maps_copy_and_freeze_input_mappings. Surface policy land restriction values remain in test_coupler_initialize_happy_path_builds_unique_regridders_and_supports_x64 and custom topology policy tests. |
| test_runtime_topology_maps_copying_stays_at_exchange_topology_boundary | Renamed to test_runtime_topology_maps_copy_and_freeze_input_mappings; mutating/clearing original dictionaries cannot affect frozen topology maps, original leaf identity survives, and writes fail. Source-spelling assertions about a now-retired construction branch are removed. |
| test_runtime_modules_use_current_domain_owners | Retains all other domain-owner assertions; removes only presence assertions for retired flatten_exchange_fields and empty-contract validation helper. Canonical flatten behavior remains in test_flatten_fields_and_append_unique; declaration checks and standalone empty-contract components remain covered by focused/full runtime tests. |
| test_flatten_fields_and_append_unique | Same literal scalar/vector flatten order and unique-append assertions, now calling canonical vercor.fields._flatten_field_items directly. |
| test_coupler_initialize_happy_path_builds_unique_regridders_and_supports_x64 | Same six-factory uniqueness, precision, contracts, and land restriction assertions; setup uses _ensure_prepared. |
| test_build_exchange_topology_returns_runtime_topology_maps | Same map type and route membership, through real preparation. |
| test_build_exchange_topology_defaults_to_target_active_domain | Same literal target wet mask and float32 dtype, through real preparation. |
| test_build_exchange_topology_rejects_duplicate_topology_keys | Same duplicate route rejection, now at constructor before preparation; message pattern follows the canonical owner. |
| test_surface_mask_policy_uses_one_build_protocol | Same single policy.build call, through prepared bindings and original author objects. |
| test_runtime_field_dispatch_handles_scalar_and_vector_paths | Same scalar 5, vector +9/-9, and JAX array assertions. Recording regridders are returned by each declared factory with actual source/target grids. |
| test_runtime_field_dispatch_accepts_mixed_numpy_and_jax_arrays | Same temperature 5 and JAX result assertions, using the declared recording factory. |
| test_runtime_field_dispatch_rejects_missing_scalar_and_vector_fields | Same missing scalar and missing vector error assertions, using declared recording factories. |
| tests/test_runtime_run.py::_make_coupler and tests/test_coupler_runtime.py::_make_coupler | Removed identity regridder overwrites identical to declared identity factories. Every existing rollout, numerical, shape, JIT, JVP/VJP, and finite-difference assertion is retained. |
| _make_initialized_slab_coupler; _make_initialized_mixed_grid_slab_coupler; test_jcm_slab_ocean_exchange_recipe_prefills_required_flux_imports | Replaced private initialization alias with canonical _ensure_prepared; all original assertions retained. |
| test_data_forcing_components_run_inside_runtime; test_public_data_component_monthly_output_validates_and_sends_runtime_slice | Removed redundant identity overwrites. Monthly center slices, shapes, outgoing fields and JVP/VJP assertions retained. |
| test_daily_data_forcing_sends_time_slice_to_slab_component_with_real_regridder; test_erainterim_ocean_monthly_forcing_replays_to_slab_atmosphere_with_real_regridder | Removed redundant same bilinear factory results. Real remapping and forcing/flux/gradient assertions retained. |
| test_jcm_land_daily_forcing_replays_to_data_atmosphere_under_jit_and_grad; test_noleap_daily_forcing_replays_calendar_slice_under_jit_and_grad; test_360_day_daily_forcing_matches_host_calendar_mapping_under_jit_and_grad; test_monthly_forcing_wraps_year_boundary_under_jit_and_grad | Removed identity/bilinear overwrites identical to declared factories. Every calendar mapping, shape, literal slice/weight, and adjoint assertion retained. |
| test_data_forcing_replays_into_jax_gcm_runtime_under_jit_grad_and_jvp | Removed redundant identity overwrite; JCM fields, finite evolution, replay, JVP/VJP checks retained. |
| test_run_validates_regridders_and_fractional_masks | Missing regridder is still tested via locally corrupted immutable prepared topology; all missing/wrong fractional masks remain tested. |
| test_runtime_store_finiteness_uses_route_mask_only_for_inbound_stores | Removed no-op topology replacement; stored received support, inactive NaN and outgoing finiteness assertions retained. |
| test_component_models_coverage.py, test_camulator_component_kernels.py, test_dynamic_mask_exchanges.py | Only removed mandatory endpoint flag at real contract-builder call sites; all contract, mask and numerical assertions retained. |
| tests/_runtime_helpers.py::run_scanned_coupler | Removed unused validate_state bypass; same canonical preparation plus selected JAX backend remains exercised. |

The retained route/public API suites protect unknown endpoints, fan-in conflicts, per-route scalar/vector capabilities, policy masks, received versus current support, supplied-state schemas, lifecycle identity, and original author component identity in lifecycle callbacks. Topology contexts bind original author objects directly.

## Earlier assertion migrations: Task 3

| Retired surface | Callers before removal | Canonical owner and surviving assertions |
| --- | --- | --- |
| Private bilinear/conservative factories | Regridding facade and regridder/helper/coupler tests | Public regridding factories construct concrete private classes; existing factory options, identical-grid no-construction/pass-through, NumPy/JAX scalar, vector rotation, source-mask/radius and conservation assertions remain in their test files. |
| `forcing_index.ForcingYearType` | Public inventory only | `calendar.YearType`; forcing lookup/calendar tests retain year/calendar semantics. |
| `tree_as_real_dtype` | One external-component helper test and API callable assertion | `tree_as_runtime_dtype`; migrate value/shape checks and strengthen floating, integer, bool, complex dtype preservation for both precision policies. The retired all-real coercion is intentionally replaced by runtime dtype semantics. |
| `tree_unwrap_leading_dims`, `tree_stack` | One combined external-tools helper test and API callable assertions | Native prediction reduction through `tree_mean` and runtime dtype normalization; retain mean value checks and test category/complex behavior. Removed reshape/stack assertions concern unused wrappers around JAX primitives, not a supported VerCOR behavior. |
| `canonicalize_surface_field` | Data helper/kernel tests only | Explicit `canonicalize_time_last_surface_field` for 3D and `as_jax_real_array(...).T` for static native land layouts; preserve values, JIT, dtype and both input gradients. |
| `get_default_parameter_values` | Parameter-change helper test only | Native parameter attributes; retain original/default and updated-value assertions around production `change_jcm_parameter_values`. |
| `jax_gcm_prediction_output_variables` | Mean/calendar output tests only | Native runtime output provider consumes payload JCM state; preserve accumulated means/counts, units/dimensions, cached coordinates, calendars and file output assertions. |
| `prepare_veris_forcing` | Veris forcing/parameter/rollout tests only | `prepare_veris_forcing_pack(...).native_forcing`; all existing unit/sign/axes, dry/wet finite-AD, mass conversion, density, native rollout, JVP/VJP/FD assertions survive. |
| Setup-owned attribute defaults | Incomplete JCM/Veros native test resources | Constructors own resource attributes; fixtures supply complete explicit resources and retain their original behavior assertions. Native optional leaf/coordinate capability detection remains where it is meaningful. |
| JCM snapshot mutable `_state` fallback | Snapshot writer | Only `SnapshotContext.payload` supplies final native state; retain final-payload values and missing-payload component-scoped error/no-output regression. |

## Earlier assertion migrations: Task 4

No tests were deleted and no unique assertion was removed.

| Existing node | Updated expectation | Preserved assertions |
| --- | --- | --- |
| tests/test_slab_kernels.py::test_atmosphere_kernels_support_jit_and_gradients | SHF coefficient -10 to +10, LHF ratio -0.5 to +0.5, dSHF/dSST +10 to -10 | Default SST values, whole 2x2 flux arrays, air relaxation, both wind shapes, explicit jitted calls, reverse gradient and finite JVP/VJP |
| tests/test_slab_kernels.py::test_land_kernel_supports_jit_and_clipping | Moisture update subtract to add latent term | Full 2x2 result including both clipped endpoints, JIT, unit interior moisture gradient, finite JVP/VJP |
| tests/test_component_models_coverage.py::test_slab_component_initialize_and_step_behaviors | Warm-air sensible flux becomes positive; positive latent flux increases soil moisture | All declarations/defaults, zero flux, flux/wind/soil/SST shapes, positive latent flux, nonzero winds, air cooling, zero-flux SST identity, positive-flux ocean warming and sea-ice assertions |
| tests/test_coupler_runtime.py::test_run_matches_one_step_closed_form_for_slab_ocean | SHF -10 to +10, latent ratio -0.5 to +0.5, restoring anomaly plus to minus | The full one-step coupled ocean SST closed-form comparison |

The additional runtime expectation was discovered by the focused gate and approved by the controller as necessary migration. Its assertion was retained. The first focused run had 123 passed and exactly this one expected old-algebra failure; XML: /private/tmp/vercor-task4-focused.xml.

## Earlier assertion migrations: Task 5

No tests or unique assertions were deleted.

| Existing test | Migration | Surviving assertions |
| --- | --- | --- |
| test_map_camulator_prediction_arrays_supports_jit_and_preserves_conventions | Parameterized at exactly 3600,10800,21600 seconds; SW input/expectation changes from 8 W/m² to the specified 100 W/m²; LW stays -3 W/m² net | Jitted actual mapper, bottom-level U/V/T/humidity arrays, both radiation expectations, height/density/potential-temperature shapes and finiteness, independent hybrid-height comparison, original finite temperature JVP/VJP |
| same mapper test | Local mapped_arrays helper permits independent perturbations without a second production path | Added analytic JVP/VJP and centered finite differences for SW energy, LW energy, TS and duration; expected gradients [4/t,4/t,4*sigma*sum(TS**3),-4*97/t] follow the literal four-cell radiation law |
| test_camulator_runtime_step_is_reproducible_from_one_payload | Stand-in output accessor now supplies physical channel shapes/energies; mapper wrapper delegates to the actual mapper rather than returning a constant temperature; native3h/interval6h fixture is consistent | Both StepResults, cursor advance and unchanged original cursor, unchanged original native tensor, finite fields, reproducible prediction, dynamic-forcing shape/values, SST shape, JAX total TS, total TS and temperature9 arrays, captured prediction2, two output samples and [1,2] sample sequence |
| same runtime test | Hybrid coefficients now have the actual interface/full-level counts needed by the real mapper | Added independent SW100 and downward LW sigma*280**4-3 assertions; no assertion was removed |

## Earlier assertion migrations: Task 6

- Deleted test_camulator_runtime_field_initializer_returns_jax_arrays: nonempty mapping, all-JAX leaves, total_surface_temperature/temperature_3d/specific_humidity_3d presence, native horizontal shape and exact initial temperature survive in test_native_bootstrap_maps_latest_input_without_advancing. The obsolete expected zero temperature is replaced with independent physical 280 K, not dropped. Generic runtime zero prefill remains owned by component/spec preparation and test_camulator_runtime_field_names_have_lightweight_contract_owner.
- Deleted test_camulator_gcm_setup_returns_initial_runtime_payload: every original payload assertion (native model_state identity, forecast_hour=1, no output predictions/samples, start_ix=0, model_substeps=1, timestep_counter=0) survives in test_native_bootstrap_maps_latest_input_without_advancing, alongside physical export/time/inverse-method/trace/native-call assertions.
- The remaining runtime reproducibility fixture now builds explicit test-owned zero storage instead of invoking the deleted production initializer; all its model-state, cursor, output-provider, radiation-duration and substep assertions remain. Its declared fake vapor channel is explicit and typed by actual TensorVariableIndex metadata.
- Existing prediction sign/native-duration/IFS-height/gradient and all output/forcing/tensor-layout tests remain. No skips, timeouts or weakenings were introduced.

## Earlier declaration migrations: Task 7

No test function was deleted. Data-driver/public-data-ocean declarations retain
exchange assertions through explicit diagnostic specs. The CAMulator-land
radiation case now rejects the unsupported route early; its cursor, type,
temperature and repeated-step assertions survive. Removed slab fixture inputs
were physically unread; precision, shapes, real regridding, JIT/JVP/VJP,
closed-form SST and required heat assertions remain. Wind rotation and store
coverage remain in dynamic-mask, bulk substitution and native forcing suites.
The independent shared coastal data contains literal quantity values 8/4/2 and
analytic coverage/field gradients; it does not calculate its expectations with
the implementation. Native face geometry preparation rejects each of four
malformed west/south mask/area shapes before any native advance.

Task 1 retained the single exact stock-extra assertion in
`test_distribution_boundaries.py::test_runtime_metadata_separates_test_and_development_dependencies`.
The duplicate stock assertion was removed from the Veris dependency test;
`test_optional_dependency_pins_veris_release` retains its distinct release pin.
Installed wheel/sdist metadata and exact-fork provenance are separate guarantees.

## Evidence retained from completed commits

The table reports prior task receipts, not newly repeated measurements.
All listed final gates have zero failures, errors and skips. Native stock uses
Python 3.12 at `/private/tmp/vercor-time-ci/envs/scipy/bin/python` with
`VEROS_ASSET_DIR=/private/tmp/vercor-cleanup-assets`. The stock runtime is
Veros 1.7.0; only its requirement changed to `veros>=1.7.0`. The exact fork
remains `7a8c964cf00b5aa0713c995edd760643a431b3c9`. The reviewed JCM 3.0.0 /
JAX 0.11.2 upper-bound installer exception is unchanged.

| Task / commit | RED evidence | Final focused / fast / full stock | Additional evidence |
| --- | --- | --- | --- |
| 1 / `82b1c68` | 3 metadata/cache-diagnostic failures | 103 / 1284 / 2209 | Fast 208.226 s; full 246.316 s; native copy/cache restoration retained. Black 4, mypy 297, lint 0. |
| 2 / `755214c` | 5 structural public-step failures | Retained 323; focus 512 + migrated 138 / 1289 / 2214 | Fast 310.735 s; full 325.890 s; exact-fork AD 15 (578.773 s); Black/mypy 297, lint 0. |
| 3 / `c017670` | Missing final JCM payload did not raise (1) | Retained 170; focus 381 + final assertions 6 / 1289 / 2215 | Fast 218.551 s; full 277.837 s; native forcing rollout AD in focus; exact-fork 11 retained plus corrected precipitation case 1; Black/mypy 297, lint 0. |
| 4 / `b38ce0b` | 6 independent slab physical-direction failures | Unit 29 + gallery 2; focus 124 / 1295 / 2222 | Fast 287.281 s; full 303.573 s; analytic restoring JVP/VJP/FD; Black/mypy 297, lint 0. |
| 5 / `de0b582` | 3 duration/substep failures + 8 invalid durations | 83 / 1296 / 2232 | Fast 212.237 s; full 244.246 s; 1/3/6 h radiation, energy/TS/duration JVP/VJP/FD; Black/mypy 297, lint 0. |
| 6 / `5b85691` | Startup 14, host errors 5, singleton 3, native timestamp 2 | 222 / 1328 / 2263 | Final fast 202.978 s; full 238.996 s; installed artifact/public boundaries; Black/mypy 298, lint 0. |
| 7 / `dbd858f` | 4 declaration contracts + CAMulator land | 135 (includes 41 new cases) / 1369 / 2304 | Fast 337.860 s; full 356.055 s; exact-fork AD 15 in 676.436 s; Black/mypy 300, lint 0. |

Exact stock gate command convention for these receipts:

```bash
VEROS_ASSET_DIR=/private/tmp/vercor-cleanup-assets /private/tmp/vercor-time-ci/envs/scipy/bin/python -m pytest tests/ -q --fast -m 'not veros_autodiff' --tb=short --basetemp=/private/tmp/vercor-cleanup-fast
VEROS_ASSET_DIR=/private/tmp/vercor-cleanup-assets /private/tmp/vercor-time-ci/envs/scipy/bin/python -m pytest tests/ -q -m 'not veros_autodiff' --tb=short --basetemp=/private/tmp/vercor-cleanup-full
```

Runs with overlapping processes used distinct task-specific basetemps. XML/log
receipts use `/private/tmp/vercor-cleanup-task1-{fast,full}`,
`/private/tmp/vercor-cleanup-task2-fast-final`,
`/private/tmp/vercor-cleanup-task2-full`, and
`/private/tmp/vercor-task{3,4,5,6,7}-*`; the exact final filenames, counts and
times are preserved from the completed reports. The final XML basenames are:

| Task | Fast XML | Full XML | Basetemp distinction |
| --- | --- | --- | --- |
| 1 | `vercor-cleanup-task1-fast.xml` | `vercor-cleanup-task1-full.xml` | XML stem under `/private/tmp` |
| 2 | `vercor-cleanup-task2-fast-final.xml` | `vercor-cleanup-task2-full.xml` | XML stem under `/private/tmp` |
| 3 | `vercor-task3-fast-final.xml` | `vercor-task3-full.xml` | XML stem under `/private/tmp` |
| 4 | `vercor-task4-fast.xml` | `vercor-task4-full.xml` | `/private/tmp/vercor-cleanup-fast` and `-full` |
| 5 | `vercor-task5-fast.xml` | `vercor-task5-full.xml` | `/private/tmp/vercor-cleanup-fast` and `-full` |
| 6 | `vercor-task6-fast-final.xml` | `vercor-task6-full-final.xml` | XML stem under `/private/tmp` |
| 7 | `vercor-task7-fast-final.xml` | `vercor-task7-full-final.xml` | XML stem under `/private/tmp` |

All XML/log files reside under `/private/tmp`; log suffix is `.log`.

Separate fork runs use
`PYTHONPATH=/private/tmp/vercor-cleanup-ad.BvXQyJ JAX_ENABLE_X64=true` and the
same interpreter, serial `-n0 -m veros_autodiff` selection of
`test_veros_autodiff_rollouts.py`, `test_veris_prognostic_autodiff.py` and
`test_veris_veros_jcm_autodiff.py`. PEP 610 confirms the fork repository and
requested/actual revision. Stock and fork never share a process.

Final static commands are Black `--check vercor tests`, strict flake8 over
maintained source/tests (also tools/benchmarks where recorded), mypy
`vercor tests`, compileall and `git diff --check`. No warning suppression was
added. Existing JCM/xarray FutureWarnings remain; the fork also reports its
known JAX scatter dtype FutureWarning. Repository-wide lint includes ignored
historical `test_logs` snippets and is reported separately from maintained code.

## Failed approaches and corrections

- Task 1 first selected a nonexistent cache-test node; corrected selection
  produced the expected RED. Sandboxed dependency resolution failed; authorized
  cached-wheel installation succeeded without changing the fork environment.
- Task 2 initially named nonexistent `test_dynamic_masks.py`; the three actual
  modules ran before removal. A stale duplicate-route diagnostic and removed
  topology-source assertion were migrated to constructor diagnostics and
  behavioral immutable-map checks. A fast gate found annotation-only imports,
  a transient 181-line progress log and denied loopback. Imports/log were fixed;
  authorized loopback gates passed without dropping any tests.
- Task 3 incomplete `seaice_coupling` fixtures and a doubled forcing-pack rename
  failed the first focus; both were corrected. Mypy required concrete narrowing
  for implementation option checks. Exact-fork precipitation fixtures gained
  the two explicit policy attributes; analytic salt/adjoint assertions stayed.
- Task 4 the existing coupled SST expectation used the old sign algebra and
  was updated alongside independent physical-direction tests. One test wrapper
  needed an explicit JAX return cast for mypy; no numerical adjustment was made.
- Task 5 invalid-duration tests first used undocumented JIT unwrap access;
  they now use `jax.disable_jit`. Invalid duration still fails before arithmetic.
- Task 6 retained inference tracing after clarifying that bootstrap must not
  advance native time or resample noise. Host nonfinite checks, singleton-axis
  preservation and native diagnostic timestamp checks were added from separate
  REDs. Initial fast failures exposed gallery and NumPy-boundary inventory
  migrations; final fixtures/inventory preserve those guarantees.
- Task 7 stale slab fixtures, recipe manifests and route counts were corrected.
  An overbroad route edit removed a real diagnostic route; its unchanged
  assertion failed and the route was restored. A native-face fixture had an
  inconsistent timestep and was corrected before the four geometry checks.
  Loopback escalation and a public Exchange annotation fixed infrastructure and
  typing failures. Final full stock includes every numerical integration case.

## Scientific and execution limits

The slab latent/moisture closures remain toy models, not a certified water
inventory budget. Native CREDIT, compatible vapor-channel checkpoints and
startup diagnostic data were unavailable; CAMulator evidence exercises actual
adapter/conversion code with stand-ins and temporary NetCDF files, not native
climate skill or inference performance. Supported native geometry, explicit
species, forcing units and time alignment remain prerequisites. Host tracing,
binary-transition derivatives, higher derivatives and long-term climate skill
are not certified. Continuous-field JIT/JVP/VJP/finite differences, native
versus wrapper rollouts, inactive NaN versus infinity, payload/solver ownership,
conservation versus interpolation and wheel/sdist/extension/CLI isolation stay
separate evidence categories. No timeouts, skips, weakened gates, compatibility
aliases, field registry or alternate runtime were introduced by consolidation.

## Complete retained source inventory

Every Python module is listed once by its package directory below. Files without
an ordinary import caller are package roots, lazy exports or CLI-discovered
gallery entry points, not evidence of dead code. The 164 non-gallery modules
are explicitly listed in dependency layers; the 15 gallery modules form the
application layer. Numerical/runtime mechanisms are covered by the removal
and retention tables above. No additional production module is removed in Task 8.

| Directory | Retained files | Responsibility / callers |
| --- | --- | --- |
| `vercor` (30) | `__init__.py`, `_field_names.py`, `_host_arrays.py`, `_numerical_safety.py`, `_pytree.py`, `_run_order.py`, `_setup_runner.py`, `assets.py`, `calendar.py`, `cli.py`, `clock.py`, `coupler.py`, `dtypes.py`, `exceptions.py`, `exchanges.py`, `field_layout.py`, `fields.py`, `forcing_data.py`, `forcing_index.py`, `grid_geometry.py`, `grid_masks.py`, `grids.py`, `jax_logging.py`, `physics.py`, `recipes.py`, `regridding.py`, `state.py`, `time_selection.py`, `topology.py`, `types.py` | Public owner facades, physical/static policies, canonical field/time/grid utilities and private transfer/CLI foundations; downstream runtime, setups and external users. |
| `vercor/_interpolators` (6) | `__init__.py`, `_bilinear_extrapolation.py`, `_bilinear_geometry.py`, `_bilinear_weights.py`, `bilinear_rectilinear.py`, `conservative_remap_rectilinear.py` | Bilinear geometry/weights/extrapolation and conservative overlap integrals; concrete regridders. |
| `vercor/_logging` (5) | `__init__.py`, `callback.py`, `config.py`, `host.py`, `protocols.py` | Host/callback logging and configuration; CLI/Coupler/runtime. |
| `vercor/_regridders` (4) | `__init__.py`, `base.py`, `bilinear.py`, `conservative.py` | Concrete scalar/vector capabilities and shared dispatch; public regridding factories. |
| `vercor/_runtime` (29) | `__init__.py`, `backends.py`, `component_masks.py`, `component_state.py`, `contracts.py`, `coupler_state.py`, `dispatch_context.py`, `driver.py`, `exchange_dispatch.py`, `exchange_masks.py`, `exchange_topology.py`, `execution.py`, `facade.py`, `field_transfer.py`, `initialization.py`, `interrupts.py`, `preparation.py`, `prepared.py`, `progress.py`, `run_context.py`, `state.py`, `state_validation.py`, `stores.py`, `surface_masks.py`, `time.py`, `topology.py`, `topology_policy.py`, `topology_state.py`, `validation.py` | Prepared binding, immutable state/schema, topology/masks, receive-step-send, workflow/backend/interrupt/output coordination; Coupler and internal driver. |
| `vercor/components` (12) | `__init__.py`, `_adapter.py`, `_callable_wrappers.py`, `_contracts.py`, `_protocol.py`, `_runtime_fields.py`, `base.py`, `contexts.py`, `contracts.py`, `data.py`, `runtime_execution.py`, `setup_validation.py` | Public structural author contracts and sole private lifecycle/payload/step adapter; preparation and runtime execution. |
| `vercor/diagnostics` (4) | `__init__.py`, `fields.py`, `plotting.py`, `tables.py` | Tables, fields and plots; explicit gallery diagnostic consumers. |
| `vercor/fluxes` (4) | `__init__.py`, `bulk_formula_cesm.py`, `utilities.py`, `vertical_coordinates.py` | CESM bulk fluxes, vertical coordinates and numerically safe utilities; surface forcing and native conversion. |
| `vercor/output` (6) | `__init__.py`, `_dataset.py`, `_netcdf.py`, `_period.py`, `_runtime.py`, `_session.py` | Public opt-in output contracts and private session/schema/accumulation/NetCDF owners; execution coordinator. |
| `vercor/runtime` (1) | `__init__.py` | Public backend/workflow/context contracts; Coupler and external extensions. |
| `vercor/setups` (8) | `__init__.py`, `_jcm.py`, `_lazy_imports.py`, `_output.py`, `_surface_fluxes.py`, `_time_helpers.py`, `_veris.py`, `config.py` | Lazy public factory table, frozen configuration, paired JCM/Veris construction and shared physical/output/time helpers; external users and galleries. |
| `vercor/setups/_data` (10) | `__init__.py`, `_component_helpers.py`, `_field_helpers.py`, `assets.py`, `era5_atmosphere.py`, `era5_land.py`, `era5_ocean.py`, `erainterim_ocean.py`, `jcm_land.py`, `veros_ocean.py` | Prescribed forcing and explicit layouts/assets; public lazy factories and paired setups. |
| `vercor/setups/_external` (40) | `__init__.py`, `_camulator_wind_filtering.py`, `_jax_gcm_pytree.py`, `camulator.py`, `camulator_contracts.py`, `camulator_fields.py`, `camulator_forcing.py`, `camulator_gcm_state.py`, `camulator_imports.py`, `camulator_init.py`, `camulator_land.py`, `camulator_output.py`, `camulator_runtime.py`, `camulator_runtime_settings.py`, `camulator_stepper.py`, `camulator_tensors.py`, `camulator_wind_filter.py`, `jax_gcm.py`, `jax_gcm_fields.py`, `jax_gcm_output.py`, `jax_gcm_runtime.py`, `jax_gcm_state.py`, `jax_gcm_surface.py`, `jax_gcm_tools.py`, `veris.py`, `veris_fluxes.py`, `veris_forcing.py`, `veris_output.py`, `veris_parameters.py`, `veris_runtime.py`, `veros_data.py`, `veros_fluxes.py`, `veros_gcm.py`, `veros_gcm_state.py`, `veros_output.py`, `veros_runtime.py`, `veros_runtime_settings.py`, `veros_setup.py`, `veros_setup_jax.py`, `veros_state.py` | Native model configuration, payload/forcing/step/output adapters; public lazy factories, paired setups and shared surface coupling. |
| `vercor/setups/_slab` (5) | `__init__.py`, `atmosphere.py`, `land.py`, `ocean.py`, `seaice.py` | Traced pedagogical kernels and structural factories; public lazy factories and galleries. |
| `vercor/setups/gallery` (15) | `__init__.py`, `custom_component_wrapping.py`, `profile_runtime.py`, `run_camulator_with_veros.py`, `run_data_driver.py`, `run_jcm_with_era5data.py`, `run_jcm_with_slab.py`, `run_jcm_with_veros.py`, `run_jcm_with_verosdata.py`, `run_slab_driver.py`, `run_veris_calibration.py`, `run_veris_dataveros_era5.py`, `run_veris_veros_era5.py`, `run_veris_veros_jcm.py`, `run_veros_with_era5data.py` | CLI-discovered application layer: explicit physical/diagnostic routes, public factory examples and profiling/calibration entry points. |

## Task 8 verification

This task changes no production code or numerical expectation. TDD RED is
not applicable to duplicate-test removal: retained before/after evidence is
the required check. Before edits, the nine affected suites passed 349/349
in 15.050 s, and the three private/architecture owner suites passed 92/92
in 3.864 s. The combined final focus passed 429/429 in 16.201 s, with no
failures, errors, skips or warnings. Exactly 12 redundant collected cases
were removed; combining two bodies preserved eight run-order cases and
fifteen supplied-state corruption cases. No artifact/backend/precision case
was deleted.

Focused commands use the interpreter above and `-q -n0 --tb=short`, with
`--basetemp=/private/tmp/vercor-task8-{before,owners-before,after}` and matching
`--junitxml` paths. The nine files are `test_api_boundaries.py`,
`test_public_api_contracts.py`, `test_v0_4_public_api.py`,
`test_v0_4_routes_and_state.py`, `test_coupler_coverage.py`,
`test_runtime_run.py`, `test_dtypes.py`, `test_bilinear_rectilinear_regridder.py`
and `test_conservative_rectilinear_regridder.py`. The three owner files are
`test_runtime_state.py`, `test_runtime_facade_boundaries.py` and
`test_api_architecture_review.py`. The after command selects all twelve.

Static checks passed: Black 300 unchanged, strict flake8 over
`vercor tests tools benchmarks` zero, mypy 300 files, compileall and whitespace.
Initial strict Sphinx found two links to repository-only documents excluded
from its source set; changed them to plain file references. The durable audit
is linked through `project-resources.rst`; the corrected strict build and final stock receipts are recorded below. The subsequent navigation-manifest migration adds the durable audit page to
`PROJECT_RESOURCE_PAGES` in `test_docs_build.py`; every exact page-order and
archive/private-document exclusion assertion is retained.

Initial stock gates caught the required navigation manifest update; fast also
collected the two Sphinx link failures before their correction. These are
documentation integration failures, with no numerical or runtime failure.
The exact navigation expectation now includes the new durable page; no check
is deleted or skipped. The corrected documentation/architecture focus passes 53/53 in 4.941 s,
including strict Sphinx, with zero failures/errors/skips.

### Final Task 8 gate receipts

Final fast stock passed 1365/1365 in 274.138 s; final full
stock passed 2292/2292 in 293.041 s, both with zero failures,
errors and skips. Full selection decreased from 2304 to 2292 by exactly the
twelve recorded duplicate cases. The deterministic fast selector uses test
identifiers, so combining parameter bodies changes its sample membership; its
final count is 1365. No selection policy, timeout or skip marker changed.

Exact final commands (authorized loopback for the installed GitHub CLI test):

```bash
VEROS_ASSET_DIR=/private/tmp/vercor-cleanup-assets /private/tmp/vercor-time-ci/envs/scipy/bin/python -m pytest tests/ -q --fast -m 'not veros_autodiff' --tb=short --basetemp=/private/tmp/vercor-task8-fast-final --junitxml=/private/tmp/vercor-task8-fast-final.xml
VEROS_ASSET_DIR=/private/tmp/vercor-cleanup-assets /private/tmp/vercor-time-ci/envs/scipy/bin/python -m pytest tests/ -q -m 'not veros_autodiff' --tb=short --basetemp=/private/tmp/vercor-task8-full-final --junitxml=/private/tmp/vercor-task8-full-final.xml
```

Corresponding `.log` files preserve diagnostics. Initial fast was 1363 passes
and two documentation failures of 1365 (279.157 s); initial full was 2291
passes and one navigation-manifest failure of 2292 (297.379 s). Both were
rerun completely after correction. Full stock retains two known JCM/xarray
FutureWarning call sites; warnings are not suppressed. Fresh final static
checks again passed Black 300, mypy 300, maintained flake8 zero, compileall
and whitespace. Repository lint remains 200 findings confined to ignored
historical test_logs snippets. No additional native-fork run is required for
this test/documentation-only consolidation; the recorded Task 7 fifteen-case
exact-fork evidence remains unchanged and final unified acceptance is Task 9.

### Task 8 review correction I1

The review demonstrated that the original surviving-owner map missed lazy
root aliases: `vars(module)` does not invoke `__getattr__`. Before the fix,
separate temporary aliases for `Regridder` and `RegridderFactory` passed both
mapped canonical tests while `hasattr` returned true. Two direct lookup
assertions are restored inside `test_root_exports_exactly_the_six_primary_symbols`;
no extra test function or production code is added. The table above now names
the actual lookup owners separately from the bound-name manifest owner.
The temporary mutation is restored in `finally`; each alias is independently
rejected after the change, while the unmodified root passes.

Review-fix final fast passed 1365/1365 in 274.018 s; full stock with branch coverage passed 2292/2292 in 325.962 s. Both have zero failures/errors/skips. Focus207/207 and Black300/mypy300/maintained lint0/compileall/whitespace pass. Native JCM/xarray warnings remain explicitly tracked (M1), with no warning suppression.

Review-fix logs/XML are `/private/tmp/vercor-task8-fix1-{focus,fast,full}`.
The full command adds `COVERAGE_FILE=/private/tmp/vercor-cleanup-final.coverage.stock`
and `--cov=vercor --cov-branch --cov-report= --cov-fail-under=0` to the unchanged
stock selection, with `--basetemp=/private/tmp/vercor-task8-fix1-full`. This raw
stock branch dataset is for Task 9 combination with isolated exact-fork data;
the final combined 90% gate is unchanged. Code/test state is `1f679cf` plus
exactly the two restored root lookup assertions; no source/test changes
followed collection.
