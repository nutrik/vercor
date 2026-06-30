1. `vercor/dtypes.py` - canonical JAX/NumPy dtype policy and array-construction helpers
2. `vercor/physical_constants.py` - physical and bulk-formula default settings with AD-owned semantics
3. `vercor/pytree.py` - shared declarative PyTree mixin for immutable JAX-registered containers
4. `vercor/settings.py` - unified metadata-backed `VercorSettings` container and static runtime controls built on (2)
5. `vercor/field_names.py` - canonical exchange-field vocabulary
6. `vercor/calendar.py` and `vercor/forcing_index.py` - calendar constants, model-calendar datetime values, leap-year logic, month/day conversion, compatibility delegates, and daily forcing-index policy
7. `vercor/fluxes/vertical_coordinates.py` - hybrid/sigma-coordinate pressure and altitude helpers built on (1, 4)
8. `vercor/fluxes/utilities.py` - JAX-native scalar/array thermodynamic helpers built on (1, 4)
9. `vercor/fluxes/bulk_formula_cesm.py` - JAX-native atmosphere-ocean / atmosphere-ice bulk flux kernels with local JAX-array normalization built on (1, 4, 8)
10. `vercor/host_arrays.py` - explicit JAX/NumPy host-transfer boundary for non-differentiable adapters, output, and host `int64` NetCDF time coordinates
11. `vercor/pytree_utils.py` - generic leafwise PyTree transforms and casting helpers used by setup adapters
12. `vercor/grid.py` - JAX-friendly `RectilinearGrid` holder with eager validation and PyTree registration built on (3)
13. `vercor/grid_geometry.py` - rectilinear grid construction, center-to-edge geometry, and grid identity built on (1, 12)
14. `vercor/field_layout.py` - shared canonical grid-field shape validation, component data-field layout validation, and time-last forcing normalization helpers built on (12)
15. `vercor/interpolators/_bilinear_geometry.py`, `_bilinear_weights.py`, `_bilinear_extrapolation.py`, and `bilinear_rectilinear.py` - private spherical geometry, cell-weight, and extrapolation helper owners behind the public JAX-native bilinear scalar/vector interpolation facade built on (1, 3)
16. `vercor/interpolators/conservative_remap_rectilinear.py` - JAX-native conservative scalar remapping runtime with eager overlap preprocessing built on (1, 3, 12, 13)
17. `vercor/regridders/base.py` - shared regridder grid/interpolator/display state built on (12, 13)
18. `vercor/regridders/bilinear.py` - bilinear regridder wrapper over (15, 17)
19. `vercor/regridders/conservative.py` - conservative regridder wrapper over (13, 16, 17)
20. `vercor/grid_masks.py` - land/ocean mask construction and remap-conservation checks built on (1, 12, 13, 19)
21. `vercor/assets.py` - generic asset cache/download/checksum boundary
22. `vercor/forcing_data.py` - canonical NetCDF forcing-file read boundary
23. `vercor/time_selection.py` - day-slice and periodic interpolation index helpers built on (6)
24. `vercor/run_sequence.py` and `vercor/clock.py` - deprecated run-sequence compatibility adapter plus immutable tuple normalization (no deps), and coupler clock/timestep iteration helpers built on (6)
25. `vercor/_logging/*.py` and `vercor/jax_logging.py` - private logger configuration, protocol, host-emission, and JAX-callback owners behind the public logging facade for Python and traced JAX runtime diagnostics
26. `vercor/setups/_time_helpers.py` - shared setup-time timestep validation, lifecycle assignment, spinup logging, forcing-index, and default-field seeding helpers built on (23)
27. `vercor/setups/_lazy_imports.py` - shared lazy package export helper for setup modules with optional dependencies
28. `vercor/exchange.py`, `vercor/setups/exchange_recipes.py`, and `vercor/setups/coupler_helpers.py` - public exchange declarations and exchange-owned field/factory aliases, shared exchange field recipes, compact `ExchangeSpec` construction, and orchestration helpers built on (12, 17, 18, 19)
29. `vercor/setups/external/jax_gcm_tools.py` - JCM-specific parameter and input-data helpers built on (1, 7, 8, 11)
30. `vercor/setups/external/jax_gcm_fields.py` - public package-internal JCM output-field mapping and surface-temperature forcing helpers built on (1, 7)
31. `vercor/output/variables.py`, `period_averages.py`, `period_files.py`, `adapters.py`, `datasets.py`, `time.py`, and `netcdf.py` - shared period-output variable containers, JAX-backed sum/count period-average accumulation using the shared `OutputVariable` sample shape, shared mean-output conversion helpers, lightweight component output adapter ownership, centralized record/write orchestration, period-average file write lifecycle, dataset coordinate discovery, VerCOR-calendar time encoding, cadence policy, and direct h5netcdf averaged NetCDF writing with host conversion delegated to (10) built on (1, 6, 10, 25)
32. `vercor/setups/external/jax_gcm_output.py`, `jax_gcm_runtime.py`, `jax_gcm_state.py`, and `jax_gcm.py` - JAXGCM prediction-to-output adaptation, metadata helpers, configured output-adapter construction, and period-output record delegation through `ComponentOutputAdapter`, payload/hooks/stepping with concrete setup-state annotations, setup-state initialization and canonical `JCMState` ownership, direct runtime hook binding, and thin JCM adapter factory boundary built on (1, 3, 11, 26, 29, 30, 31, 62)
33. `vercor/setups/external/veros_runtime_settings.py` - explicit lazy Veros backend/runtime configuration side-effect boundary
34. `vercor/setups/external/veros_setup.py` - concrete Veros setup subclass and setup policy built on (33)
35. `vercor/setups/external/veros_fluxes.py` - Veros-to-VerCOR flux conversion built on (1, 4, 9, 33)
36. `vercor/setups/external/veros_state.py` - public package-internal Veros host-state copy, mutation, named forcing-field container, forcing, and stepping helpers built on (1, 10, 33)
37. `vercor/setups/external/veros_output.py`, `veros_runtime.py`, `veros_gcm_state.py`, and `veros_gcm.py` - Veros opt-in JAX-backed period-output variable/coordinate extraction, native dimension-order helpers, configured output-adapter construction, and period-output record delegation through `ComponentOutputAdapter`, host-runtime stepping helper with concrete setup-state annotations, setup-state/lifecycle ownership, named host step adapter, and thin adapter factory boundary built on (1, 6, 10, 25, 31, 33, 34, 35, 36)
38. `vercor/setups/external/camulator_imports.py` - public package-internal lazy CREDIT/postblock/CAMulator wind-filter optional-dependency loading
39. `vercor/setups/external/camulator_forcing.py` - CAMulator config loading, forcing context, and runtime forcing cursors built on (26, 38)
40. `vercor/setups/external/camulator_tensors.py` - typed CAMulator tensor-variable indexing, public package-internal static forcing tensor staging, and JAX-to-Torch transfer helpers
41. `vercor/setups/external/camulator_contracts.py` and `camulator_fields.py` - lightweight CAMulator runtime field contract ownership, public package-internal runtime field initialization, forcing prep, and prediction-field mapping built on (1, 7, 10, 40)
42. `vercor/setups/external/_camulator_wind_filtering.py` and `camulator_wind_filter.py` - private CAMulator wind-mask/kernel/tensor filtering mechanics plus the public configuration and log-and-skip post-processing facade built on (25)
43. `vercor/setups/external/camulator_stepper.py` - CAMulator state shifting, forcing concatenation, stepping, and optional post-processing built on (38, 40, 42)
44. `vercor/setups/external/camulator_output.py` - CAMulator tensor-to-output-variable shaping, average-output path/coordinate/metadata helpers, configured output-adapter construction, period-output record delegation through `ComponentOutputAdapter`, metadata loading, unsupported CREDIT-output option validation, and direct h5netcdf forecast-increment writing through shared VerCOR output helpers built on (25, 31)
45. `vercor/setups/external/camulator_init.py` - CAMulator model, transform, forcing, output metadata, init-noise, and stepper initialization built on (38, 40, 43, 44)
46. `vercor/setups/external/camulator_runtime_settings.py` - CAMulator TensorFlow/OpenMP/MKL import-time environment settings boundary
47. `vercor/setups/external/camulator_land.py` - CAMulator land host-runtime adapter with shared timestep/cursor setup built on (1, 12, 20, 26, 39)
48. `vercor/setups/external/camulator_runtime.py`, `camulator_gcm_state.py`, and `camulator.py` - CAMulator public package-internal host-runtime prediction-block/step helpers with concrete setup-state annotations, optional period output delegated through the CAMulator output helper and shared `ComponentOutputAdapter`, atmosphere setup-state/lifecycle ownership, and thin adapter factory boundary built on (7, 10, 26, 39, 40, 41, 44, 45, 46)
49. `vercor/setups/data/assets.py` - concrete ERA/ECMWF setup forcing asset registry built on (21)
50. `vercor/setups/data/_field_helpers.py` - shared JAX-backed data-adapter positive masks, surface canonicalization, and field normalization helpers built on (1, 14)
51. `vercor/setups/data/_component_helpers.py` - shared data-component construction helper for time-interpolated forcing adapters built on (12, 23, 26)
52. `vercor/setups/data/era5_atmosphere.py` - ERA5 atmospheric data component built on (7, 8, 12, 14, 22, 49, 51)
53. `vercor/setups/data/era5_ocean.py` - ERA5 ocean forcing adapter with canonical layout and masked SST handling built on (12, 14, 22, 49, 50, 51)
54. `vercor/setups/data/era5_land.py` - ERA5 land forcing adapter with canonical layout and runtime temperature storage built on (12, 14, 22, 49, 51)
55. `vercor/setups/data/erainterim_ocean.py` - ERA-Interim ocean forcing adapter built on (12, 14, 22, 49, 50, 51)
56. `vercor/setups/data/jcm_land.py` - JCM land forcing adapter with coordinate conversion and runtime storage built on (1, 12, 20, 50)
57. `vercor/setups/jcm_setup_helpers.py` - paired JCM atmosphere/land setup construction with lazy optional JCM adapter imports built on (10, 32, 56)
58. `vercor/setups/slab/atmosphere.py`, `ocean.py`, `land.py`, and `seaice.py` - slab model adapters built on (1, 9, 12)
59. `vercor/components/contexts.py` - immutable component setup and step context payloads built on calendar, tuple run-order sequences, settings helpers, and (25)
60. `vercor/runtime/views.py` - explicit runtime component metadata and shared read-only field resolution for diagnostics/output views and compatible runtime states built on (12, 64)
61. `vercor/diagnostics/fields.py`, `tables.py`, `plotting.py`, and `__init__.py` - derived diagnostic fields, console tables, optional plotting, and public reexports with runtime field lookup delegated to (60) built on (10, 60)
62. `vercor/components/contracts.py`, `base.py`, `data.py`, `host.py`, `_field_names.py`, `_protocols.py`, `_field_authoring.py`, `_contracts.py`, `_lifecycle.py`, `_lifecycle_api.py`, `_callable_wrappers.py`, `_runtime_fields.py`, `_runtime_validation.py`, `runtime_execution.py`, `setup_validation.py`, and `factories.py` - public component author contracts and lifecycle hook aliases, abstract/component-kind owners, concrete callable-backed wrappers beside their runtime-kind bases, shared field-name de-duplication, one private host-runtime execution protocol, field-authoring helpers, internal field normalization, lifecycle hook storage/dispatch/default-payload fallback, shared callable construction/signature/runtime helpers with explicit author-argument adapter names, runtime-field adapters plus direct `Component` accessors, component-facing runtime validation through shared field-layout checks, protocol-backed component-owned runtime execution policy, setup validation, and public factory helpers built on (1, 12, 14, 59)
63. `vercor/runtime/contracts.py` - runtime import/export contracts, exchange-field flattening, stable exchange mask key names, and contract construction
64. `vercor/runtime/stores.py`, `state.py`, and `__init__.py` - immutable runtime field stores, component/coupler runtime state containers, and an empty runtime package initializer with no compatibility reexports built on (1, 3, 63)
65. `vercor/runtime/time.py` - `RuntimeStepInfo` plus host-precomputed daily/monthly runtime step metadata built on (3, 6, 23, 64)
66. `vercor/runtime/component_state.py`, `field_transfer.py`, `validation.py`, and `state_validation.py` - component runtime state creation, contract prefill, receive/send, time-sliced export selection, component/store runtime validation, and configured runtime-state/topology validation built on (1, 5, 14, 28, 62, 63, 64, 65)
67. `vercor/runtime/dispatch_context.py`, `exchange_dispatch.py`, and `driver.py` - static dispatch context construction, exchange dispatch, component runtime stepping orchestration, and outgoing priming built on (6, 59, 62, 63, 64, 65, 66)
68. `vercor/runtime/component_topology.py` - default topology component-name validation and component lookup built on (62)
69. `vercor/runtime/coupler_state.py` and `topology_state.py` - runtime coupler-state assembly, runtime-contract refresh, grouped exchange topology maps, surface-mask state, and topology-state contracts built on (1, 28, 63, 64, 66)
70. `vercor/runtime/exchange_topology.py`, `surface_masks.py`, and `topology.py` - generic exchange regridder/identity-mask map construction, ATM/OCN/LND surface-mask policy, and topology orchestration built on (1, 13, 19, 20, 25, 28, 68, 69)
71. `vercor/runtime/initialization.py` - setup-time precision synchronization, component initialization, setup validation, runtime contract validation, and exchange-topology handoff built on (24, 59, 62, 63, 66, 68, 70)
72. `vercor/runtime/run_context.py`, `cache.py`, `progress.py`, and `interrupts.py` - frozen runtime execution context with context-derived cache-key construction, compiled-runtime callable/key aliases owned by `CompiledRuntimeCache`, cache storage/JIT policy, progress callbacks, and cancellation controller built on (24, 25, 64, 67)
73. `vercor/runtime/resources.py` - public-field per-coupler runtime resource holder for topology maps, refreshed contracts, compiled-runtime cache owner, and interrupts built on (63, 64, 69, 72)
74. `vercor/runtime/preparation.py` - runtime state preparation, prepared-state bundle ownership, refreshed contract validation, and initial outgoing-store priming built on (24, 63, 64, 65, 67, 69, 73)
75. `vercor/runtime/runner.py` - host/scanned runtime loops, run-mode selection, cache-owner delegated compiled scanned dispatch, donation checks, and interrupt translation built on (65, 67, 72)
76. `vercor/output/runtime.py` and `vercor/output/__init__.py` - runtime-view output mask selection/naming, coupler-final-output direct h5netcdf boundary through (31), and direct top-level runtime-output reexports built on (10, 12, 31, 60, 64)
77. `vercor/runtime/facade.py` - high-level runtime orchestration boundary and internal repeated-input bundle for the public coupler facade and runtime resource holder built on (24, 25, 59, 60, 63, 64, 67, 69, 71, 72, 73, 74, 75, 76)
78. `vercor/coupler.py` - public setup/finalization facade for `run()`, `create_runtime_state()`, runtime views, and minimal runtime-cache profiling helpers built on (25, 59, 62, 64, 73, 77)
79. `examples/` - runnable setup scripts that assemble packaged adapters from `vercor.setups`
