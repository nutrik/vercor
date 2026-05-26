1. `vercor/dtypes.py` - canonical JAX/NumPy dtype policy and array-construction helpers
2. `vercor/pytree.py` - shared declarative PyTree mixin for immutable JAX-registered containers
3. `vercor/settings.py` - unified metadata-backed `VercorSettings` container, defaults, physical constants, and settings-bound dtype policy consumed by kernels
4. `vercor/calendar.py` - calendar constants, leap-year logic, noleap/360-day mapping, and daily forcing-index helpers
5. `vercor/fluxes/utilities.py` - JAX-native scalar/array thermodynamic helpers, virtual-temperature conversion, and hybrid-sigma altitude kernels built on (1, 3)
6. `vercor/fluxes/vertical_coordinates.py` - generic sigma-coordinate pressure and altitude helpers built on (1, 5)
7. `vercor/fluxes/bulk_formula_cesm.py` - JAX-native atmosphere-ocean / atmosphere-ice bulk flux kernels built on (1, 3, 5)
8. `vercor/host_arrays.py` - explicit JAX/NumPy host-transfer boundary for non-differentiable adapters and output
9. `vercor/pytree_utils.py` - generic leafwise PyTree transforms used by setup adapters
10. `vercor/grid.py` - JAX-friendly `RectilinearGrid` holder with eager validation and PyTree registration built on (2)
11. `vercor/grid_geometry.py` - rectilinear grid construction and center-to-edge geometry built on (1, 10)
12. `vercor/field_layout.py` - canonical component data-field layout validation and time-last forcing normalization helpers built on (10)
13. `vercor/interpolators/bilinear_rectilinear.py` - JAX-native bilinear scalar/vector interpolation and extrapolation built on (1, 2, 10, 11)
14. `vercor/interpolators/conservative_remap_rectilinear.py` - JAX-native conservative scalar remapping runtime with eager overlap preprocessing built on (1, 2, 10, 11)
15. `vercor/regridders/conservative.py` - conservative regridder wrapper over (14)
16. `vercor/grid_masks.py` - grid identity, land/ocean mask construction, and remap-conservation checks built on (1, 10, 15)
17. `vercor/assets.py` - generic asset cache/download/checksum boundary
18. `vercor/forcing_data.py` - canonical NetCDF forcing-file read boundary and compatibility reader class for data components
19. `vercor/time_selection.py` - day-slice and periodic interpolation index helpers built on (4)
20. `vercor/clock.py` - model/coupler calendar and timestep helpers built on (4)
21. `vercor/setups/_time_helpers.py` - shared setup-time timestep validation, lifecycle assignment, spinup logging, forcing-index, and default-field seeding helpers
22. `vercor/setups/_lazy_imports.py` - shared lazy package export helper for setup modules with optional dependencies
23. `vercor/setups/exchange_recipes.py`, `vercor/setups/coupler_helpers.py`, and `vercor/setups/jcm_setup_helpers.py` - shared exchange recipes, orchestration helpers, and JCM setup helpers
24. `vercor/setups/external/jax_gcm_tools.py` - JCM-specific file/output helpers built on (1, 5, 6, 9)
25. `vercor/setups/external/jax_gcm.py` - JCM adapter boundary that stores translated kernel outputs built on (1, 2, 6, 9, 21, 24)
26. `vercor/setups/external/veros_runtime_settings.py` and `vercor/setups/external/veros_gcm.py` - Veros host-runtime configuration and adapter boundary built on (1, 7, 8, 21)
27. `vercor/setups/external/camulator_imports.py` - lazy CREDIT/postblock/windpp optional-dependency loading
28. `vercor/setups/external/camulator_forcing.py` - CAMulator config loading, forcing context, and runtime forcing cursors built on (21, 27)
29. `vercor/setups/external/camulator_tensors.py` - CAMulator tensor indexing and static forcing tensor staging
30. `vercor/setups/external/camulator_stepper.py` - CAMulator state shifting, forcing concatenation, stepping, and optional post-processing built on (27, 29)
31. `vercor/setups/external/camulator_init.py` - CAMulator model, transform, forcing, metadata, and stepper initialization built on (27, 29, 30)
32. `vercor/setups/external/camulator_state.py` and `vercor/setups/external/camulator.py` - CAMulator compatibility facade plus host-runtime atmosphere adapter built on (6, 8, 21, 28, 29, 31)
33. `vercor/setups/data/assets.py` - concrete ERA/ECMWF setup forcing asset registry built on (17)
34. `vercor/setups/data/_field_helpers.py` - shared JAX-backed data-adapter field normalization helpers built on (1, 12)
35. `vercor/setups/data/_component_helpers.py` - shared data-component construction helper for time-interpolated forcing adapters built on (10, 19, 21)
36. `vercor/setups/data/era5_atmosphere.py` - ERA5 atmospheric data component built on (5, 10, 12, 18, 33, 35)
37. `vercor/setups/data/era5_ocean.py` - ERA5 ocean forcing adapter with canonical layout and masked SST handling built on (10, 12, 18, 33, 34, 35)
38. `vercor/setups/data/era5_land.py` - ERA5 land forcing adapter with canonical layout and runtime temperature storage built on (10, 12, 18, 33, 35)
39. `vercor/setups/data/erainterim_ocean.py` - ERA-Interim ocean forcing adapter built on (10, 12, 18, 33, 34, 35)
40. `vercor/setups/data/jcm_land.py` - JCM land forcing adapter with coordinate conversion and runtime storage built on (10, 12, 16)
41. `vercor/setups/data/camulator_land.py` - CAMulator land forcing adapter with shared timestep/cursor setup built on (1, 10, 16, 21, 28)
42. `vercor/setups/slab/atmosphere.py`, `ocean.py`, `land.py`, and `seaice.py` - slab model adapters built on (1, 7, 10)
43. `vercor/jax_logging.py` - callback-backed logger protocol and setup helper for Python and traced JAX runtime diagnostics
44. `vercor/runtime/contexts.py` - immutable setup and runtime step context payloads built on clock, run-sequence, settings helpers, and (43)
45. `vercor/runtime/views.py` - explicit runtime component metadata/field view used by diagnostics and output built on (10)
46. `vercor/diagnostics/fields.py`, `tables.py`, `plotting.py`, and `__init__.py` - derived diagnostic fields, console tables, optional plotting, and public reexports built on (8, 45)
47. `vercor/components/_contracts.py`, `_callable_wrappers.py`, `_runtime_fields.py`, `_validation.py`, `base.py`, and `factories.py` - component contracts, callable-wrapper internals, runtime-field adapters, setup validation, base author classes, and public factory helpers built on (1, 10, 12, 44)
48. `vercor/runtime/contracts.py` - runtime import/export contracts, exchange-field flattening, stable exchange mask key names, and contract construction
49. `vercor/runtime/stores.py`, `state.py`, and `__init__.py` - immutable runtime field stores, component/coupler runtime state containers, and internal runtime reexports built on (1, 2, 48)
50. `vercor/runtime/time.py` - `RuntimeStepInfo` plus host-precomputed daily/monthly runtime step metadata built on (2, 4, 19, 49)
51. `vercor/runtime/component_state.py`, `field_transfer.py`, `validation.py`, and `components.py` - component runtime state creation, contract prefill, receive/send, time-sliced export selection, runtime validation, and compatibility reexports built on (1, 12, 47, 48, 49, 50)
52. `vercor/runtime/exchange_dispatch.py` and `driver.py` - exchange dispatch, component runtime stepping, outgoing priming, and host-adapter detection built on (44, 47, 48, 49, 50, 51)
53. `vercor/runtime/coupler_state.py` and `topology.py` - runtime coupler-state assembly, dispatch-context construction, output-mask lookup, component lookup, and exchange topology setup built on (10, 15, 16, 44, 47, 48, 49, 51, 52)
54. `vercor/runtime/runner.py` - host/scanned runtime loops, progress logging, compile-cache keys, JIT wrapping, donation checks, and interrupt translation built on (43, 50, 52)
55. `vercor/output.py` - runtime-view NetCDF output boundary built on (8, 10, 45, 49)
56. `vercor/coupler.py` - public runtime facade for `run()` and `create_runtime_state()` built on (43, 44, 47, 48, 49, 50, 51, 52, 53, 54, 55)
57. `examples/` - runnable setup scripts that assemble packaged adapters from `vercor.setups`
