1. `vercor/dtypes.py` - canonical JAX/NumPy dtype policy and array-construction helpers
2. `vercor/pytree.py` - shared declarative PyTree mixin for immutable JAX-registered containers
3. `vercor/settings.py` - unified metadata-backed `VercorSettings` container, defaults, physical constants, and settings-bound dtype policy consumed by kernels
4. `vercor/field_names.py` - canonical exchange-field vocabulary
5. `vercor/calendar.py` - calendar constants, model-calendar datetime values, leap-year logic, noleap/360-day mapping, and daily forcing-index helpers
6. `vercor/fluxes/vertical_coordinates.py` - hybrid/sigma-coordinate pressure and altitude helpers built on (1, 3)
7. `vercor/fluxes/utilities.py` - JAX-native scalar/array thermodynamic helpers plus compatibility aliases built on (1, 3, 6)
8. `vercor/fluxes/bulk_formula_cesm.py` - JAX-native atmosphere-ocean / atmosphere-ice bulk flux kernels built on (1, 3, 7)
9. `vercor/host_arrays.py` - explicit JAX/NumPy host-transfer boundary for non-differentiable adapters and output
10. `vercor/pytree_utils.py` - generic leafwise PyTree transforms used by setup adapters
11. `vercor/grid.py` - JAX-friendly `RectilinearGrid` holder with eager validation and PyTree registration built on (2)
12. `vercor/grid_geometry.py` - rectilinear grid construction, center-to-edge geometry, and grid identity built on (1, 11)
13. `vercor/field_layout.py` - canonical component data-field layout validation and time-last forcing normalization helpers built on (11)
14. `vercor/interpolators/bilinear_rectilinear.py` - JAX-native bilinear scalar/vector interpolation and extrapolation built on (1, 2, 11, 12)
15. `vercor/interpolators/conservative_remap_rectilinear.py` - JAX-native conservative scalar remapping runtime with eager overlap preprocessing built on (1, 2, 11, 12)
16. `vercor/regridders/conservative.py` - conservative regridder wrapper over (15)
17. `vercor/grid_masks.py` - land/ocean mask construction and remap-conservation checks built on (1, 11, 12, 16)
18. `vercor/assets.py` - generic asset cache/download/checksum boundary
19. `vercor/forcing_data.py` - canonical NetCDF forcing-file read boundary and compatibility reader class for data components
20. `vercor/time_selection.py` - day-slice and periodic interpolation index helpers built on (5)
21. `vercor/clock.py` - coupler clock and timestep iteration helpers built on (5)
22. `vercor/setups/_time_helpers.py` - shared setup-time timestep validation, lifecycle assignment, spinup logging, forcing-index, and default-field seeding helpers
23. `vercor/setups/_lazy_imports.py` - shared lazy package export helper for setup modules with optional dependencies
24. `vercor/setups/exchange_recipes.py`, `vercor/setups/coupler_helpers.py`, and `vercor/setups/jcm_setup_helpers.py` - shared exchange recipes, orchestration helpers, and JCM setup helpers
25. `vercor/setups/external/jax_gcm_tools.py` - JCM-specific parameter and input-data helpers built on (1, 6, 7, 10)
26. `vercor/setups/external/jax_gcm_output.py` - JAXGCM output cadence and NetCDF writing helpers built on (5, 48)
27. `vercor/setups/external/jax_gcm.py` - JCM adapter boundary that stores translated kernel outputs built on (1, 2, 6, 10, 22, 25, 26)
28. `vercor/setups/external/veros_runtime_settings.py` and `vercor/setups/external/veros_gcm.py` - Veros host-runtime configuration and adapter boundary built on (1, 8, 9, 22)
29. `vercor/setups/external/camulator_imports.py` - lazy CREDIT/postblock/CAMulator wind-filter optional-dependency loading
30. `vercor/setups/external/camulator_forcing.py` - CAMulator config loading, forcing context, and runtime forcing cursors built on (22, 29)
31. `vercor/setups/external/camulator_tensors.py` - CAMulator tensor indexing and static forcing tensor staging
32. `vercor/setups/external/camulator_wind_filter.py` - CAMulator-only wind artifact post-processing
33. `vercor/setups/external/camulator_stepper.py` - CAMulator state shifting, forcing concatenation, stepping, and optional post-processing built on (29, 31, 32)
34. `vercor/setups/external/camulator_init.py` - CAMulator model, transform, forcing, metadata, and stepper initialization built on (29, 31, 33)
35. `vercor/setups/external/camulator_output.py` - CAMulator CREDIT output writing helpers
36. `vercor/setups/external/camulator_land.py` - CAMulator land host-runtime adapter with shared timestep/cursor setup built on (1, 11, 17, 22, 30)
37. `vercor/setups/external/camulator_state.py` and `vercor/setups/external/camulator.py` - CAMulator compatibility facade plus host-runtime atmosphere adapter built on (6, 9, 22, 30, 31, 34, 35)
38. `vercor/setups/data/assets.py` - concrete ERA/ECMWF setup forcing asset registry built on (18)
39. `vercor/setups/data/_field_helpers.py` - shared JAX-backed data-adapter field normalization helpers built on (1, 13)
40. `vercor/setups/data/_component_helpers.py` - shared data-component construction helper for time-interpolated forcing adapters built on (11, 20, 22)
41. `vercor/setups/data/era5_atmosphere.py` - ERA5 atmospheric data component built on (6, 7, 11, 13, 19, 38, 40)
42. `vercor/setups/data/era5_ocean.py` - ERA5 ocean forcing adapter with canonical layout and masked SST handling built on (11, 13, 19, 38, 39, 40)
43. `vercor/setups/data/era5_land.py` - ERA5 land forcing adapter with canonical layout and runtime temperature storage built on (11, 13, 19, 38, 40)
44. `vercor/setups/data/erainterim_ocean.py` - ERA-Interim ocean forcing adapter built on (11, 13, 19, 38, 39, 40)
45. `vercor/setups/data/jcm_land.py` - JCM land forcing adapter with coordinate conversion and runtime storage built on (11, 13, 17)
46. `vercor/setups/data/camulator_land.py` - compatibility facade for (36)
47. `vercor/setups/slab/atmosphere.py`, `ocean.py`, `land.py`, and `seaice.py` - slab model adapters built on (1, 8, 11)
48. `vercor/jax_logging.py` - callback-backed logger protocol and setup helper for Python and traced JAX runtime diagnostics
49. `vercor/runtime/contexts.py` - immutable setup and runtime step context payloads built on calendar, run-sequence, settings helpers, and (48)
50. `vercor/runtime/views.py` - explicit runtime component metadata/field view used by diagnostics and output built on (11)
51. `vercor/diagnostics/fields.py`, `tables.py`, `plotting.py`, and `__init__.py` - derived diagnostic fields, console tables, optional plotting, and public reexports built on (9, 50)
52. `vercor/components/_contracts.py`, `_callable_wrappers.py`, `_runtime_fields.py`, `_validation.py`, `base.py`, and `factories.py` - component contracts, callable-wrapper internals, runtime-field adapters, setup validation, base author classes, and public factory helpers built on (1, 11, 13, 49)
53. `vercor/runtime/contracts.py` - runtime import/export contracts, exchange-field flattening, stable exchange mask key names, and contract construction
54. `vercor/runtime/stores.py`, `state.py`, and `__init__.py` - immutable runtime field stores, component/coupler runtime state containers, and internal runtime reexports built on (1, 2, 53)
55. `vercor/runtime/time.py` - `RuntimeStepInfo` plus host-precomputed daily/monthly runtime step metadata built on (2, 5, 20, 54)
56. `vercor/runtime/component_state.py`, `field_transfer.py`, `validation.py`, and `components.py` - component runtime state creation, contract prefill, receive/send, time-sliced export selection, runtime validation, and compatibility reexports built on (1, 4, 13, 52, 53, 54, 55)
57. `vercor/runtime/exchange_dispatch.py` and `driver.py` - exchange dispatch, component runtime stepping, outgoing priming, and host-adapter detection built on (5, 49, 52, 53, 54, 55, 56)
58. `vercor/runtime/coupler_state.py` and `topology.py` - runtime coupler-state assembly, dispatch-context construction, output-mask lookup, component lookup, and exchange topology setup built on (11, 16, 17, 49, 52, 53, 54, 56, 57)
59. `vercor/runtime/runner.py` - host/scanned runtime loops, progress logging, compile-cache keys, JIT wrapping, donation checks, and interrupt translation built on (48, 55, 57)
60. `vercor/output.py` - runtime-view NetCDF output boundary built on (9, 11, 50, 54)
61. `vercor/coupler.py` - public runtime facade for `run()` and `create_runtime_state()` built on (48, 49, 52, 53, 54, 55, 56, 57, 58, 59, 60)
62. `examples/` - runnable setup scripts that assemble packaged adapters from `vercor.setups`
