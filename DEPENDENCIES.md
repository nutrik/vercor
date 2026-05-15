1. `vercor/dtypes.py` - canonical JAX/NumPy dtype policy and array-construction helpers
2. `vercor/pytree.py` - shared declarative PyTree mixin for immutable JAX-registered containers
3. `vercor/settings.py` - unified metadata-backed `VercorSettings` container with dynamic attribute access, typed known-setting annotations, default settings records, physical constants, runtime/component settings, and settings-bound dtype policy consumed by translated kernels
4. `vercor/fluxes/utilities.py` - JAX-native scalar/array thermodynamic helpers, shared virtual-temperature conversion, and hybrid-sigma altitude kernels built on (1, 3)
5. `vercor/fluxes/bulk_formula_cesm.py` - JAX-native atmosphere-ocean / atmosphere-ice bulk flux kernels built on (1, 3, 4)
6. `vercor/host_arrays.py` - explicit JAX/NumPy host-transfer boundary for non-differentiable adapters and output
7. `setups/_time_helpers.py` - shared setup-time coupling/model timestep validation, lifecycle assignment, spinup logging, forcing-index, and default-field seeding helpers for setup adapters
8. `setups/_lazy_imports.py` - shared lazy package export helper for setup modules with optional dependencies
9. `setups/exchange_recipes.py`, `setups/coupler_helpers.py`, and `setups/jcm_setup_helpers.py` - shared field-recipe constants, runnable setup orchestration helpers, and JCM land/atmosphere setup helpers for repeated runnable scripts
10. `setups/external/jax_gcm_tools.py` - existing JAX helper layer used by the JCM adapter; validated for `jax.jit` and built on (1, 4)
11. `setups/external/jax_gcm.py` - JCM adapter boundary that stores translated kernel outputs built on (1, 2, 7, 10)
12. `setups/external/veros_runtime_settings.py` and `setups/external/veros_gcm.py` - explicit Veros host-runtime configuration plus the Veros adapter boundary that converts translated flux outputs back to NumPy for Veros state updates built on (1, 5, 6, 7)
13. `setups/external/camulator_state.py` and `setups/external/camulator.py` - CAMulator state loading, shared forcing cursor setup, host-runtime adapter boundary, JAX-backed runtime-field helpers, hybrid-sigma altitude diagnostics, and explicit Torch / xarray output boundaries built on (1, 4, 6, 7)
14. `vercor/grid.py` - JAX-friendly `RectilinearGrid` holder with eager validation and PyTree registration built on (2)
15. `vercor/field_layout.py` - canonical component data-field layout validation and time-last forcing normalization helpers built on (14)
16. `vercor/regridders/helpers.py` - JAX-native rectilinear helper kernels built on (1, 14)
17. `vercor/interpolators/bilinear_rectilinear.py` - JAX-native bilinear scalar/vector interpolation and extrapolation built on (1, 2, 14, 16)
18. `vercor/interpolators/conservative_remap_rectilinear.py` - JAX-native conservative scalar remapping runtime with eager overlap preprocessing built on (1, 2, 14, 16)
19. `vercor/regridders/conservative.py` - conservative regridder wrapper over (18)
20. `vercor/grid_masks.py` - grid identity, component lookup, land/ocean mask construction, and remap-conservation checks built on (1, 14, 19)
21. `vercor/assets.py` - forcing asset cache/download/checksum boundary for data components
22. `vercor/forcing_data.py` - canonical NetCDF forcing-file read boundary and compatibility reader class for data components
23. `vercor/time_selection.py` - calendar, day-slice, and periodic interpolation index helpers
24. `setups/slab/atmosphere.py` - slab atmosphere wrapper over pure JAX bulk-flux, default-SST, and wind kernels built on (1, 14)
25. `setups/slab/ocean.py` - slab ocean wrapper over pure JAX SST tendency kernel built on (1, 14)
26. `setups/slab/land.py` - slab land wrapper over pure JAX soil-moisture update kernel built on (1, 14)
27. `setups/slab/seaice.py` - slab sea-ice wrapper over pure JAX ice-fraction diagnostic kernel built on (1, 14)
28. `setups/data/_field_helpers.py` - shared JAX-backed data-adapter field normalization helpers, including masked time-last surface fields built on (1, 15)
29. `setups/data/_component_helpers.py` - shared setup data-component construction helper for time-interpolated forcing adapters built on (14, 40)
30. `setups/data/era5_atmosphere.py` - pure ERA5 atmospheric data component with canonical data-field layout and JAX-backed pressure/model-level diagnostic initialization built on (4, 14, 15, 21, 22, 29)
31. `setups/data/era5_ocean.py` - ERA5 ocean forcing adapter with canonical data-field layout, JAX-backed mask, and shared SST masking built on (14, 15, 21, 22, 28, 29)
32. `setups/data/era5_land.py` - ERA5 land forcing adapter with canonical data-field layout, JAX-backed mask preparation, and runtime temperature storage built on (14, 15, 21, 22, 29)
33. `setups/data/erainterim_ocean.py` - ERA-Interim ocean forcing adapter with canonical data-field layout, shared SST masking, and JAX-backed global field assembly built on (14, 15, 21, 22, 28, 29)
34. `setups/data/jcm_land.py` - JCM land forcing adapter with canonical data-field layout, JAX-backed coordinate conversion, and runtime storage built on (14, 15, 20)
35. `setups/data/camulator_land.py` - CAMulator land forcing adapter with shared timestep/cursor setup, JAX-backed runtime temperature storage, and forcing-only CAMulator config loading built on (1, 7, 13, 14, 20)
36. `vercor/jax_logging.py` - callback-backed logger protocol and setup helper for Python and traced JAX runtime diagnostics
37. `vercor/runtime/interrupts.py` - internal terminal-signal runtime cancellation controller with host and JAX callback checkpoints, plus wakeup-fd polling for compiled runtime signals, built on JAX callback errors and Python signal handling
38. `vercor/runtime/contexts.py` - immutable component initialization and runtime step context payloads built on clock, run-sequence, settings helpers, and (36)
39. `vercor/runtime/views.py` - explicit runtime component metadata/field view used by diagnostics and output built on (14)
40. `vercor/diagnostics.py` - runtime-view means tables, plotting helpers, and plotting-only derived field helpers built on (6, 39)
41. `vercor/components/_contracts.py`, `vercor/components/_callable_wrappers.py`, `vercor/components/_runtime_fields.py`, `vercor/components/_validation.py`, and `vercor/components/base.py` - private component contract normalization, callable-wrapper runtime internals, component-facing runtime-field adapters, setup validation, and the slim public component-author surface for active differentiable components, data-only components, host-runtime adapters, top-level authoring helpers, `from_fields()` / `from_model()` authoring facade, flexible one/two/three-argument callable steps, public setup/step context aliases, `ComponentFieldSpec` declarations and introspection, setup field-name introspection, explicit setup metadata, chainable component setting updates, declared grid-field default builders, base declared-default initialization, automatic data-component output declarations, scalar-to-grid field defaults/seeds, explicit runtime contexts, canonical data-field validation, step-result application, and finalized runtime boundary hooks built on (1, 14, 15, 38)
42. `vercor/runtime/contracts.py` - runtime import/export contracts, exchange-field flattening, stable exchange mask key names, and contract construction built on exchanges and coupler errors
43. `vercor/runtime/stores.py`, `vercor/runtime/state.py`, and `vercor/runtime/__init__.py` - immutable runtime field stores, component/coupler runtime state containers, and the internal `vercor.runtime` re-export surface built on (1, 2, 42)
44. `vercor/runtime/time.py` - `RuntimeStepInfo` plus host-precomputed daily/monthly runtime step metadata built on clock/settings helpers and (2, 23, 43)
45. `vercor/runtime/component_state.py`, `vercor/runtime/field_transfer.py`, `vercor/runtime/validation.py`, and `vercor/runtime/components.py` - component runtime state creation, contract prefill, receive/send, time-sliced export selection, runtime validation, and compatibility reexports built on (1, 15, 41, 42, 43, 44)
46. `vercor/runtime/exchange_dispatch.py` and `vercor/runtime/driver.py` - pure exchange dispatch, bundled runtime dispatch context, single per-component runtime step helper with explicit host-runtime allowance, outgoing priming, and host-adapter detection built on (38, 41, 42, 43, 44, 45)
47. `vercor/runtime/coupler_state.py` and `vercor/runtime/topology.py` - runtime coupler-state assembly, contract refresh, dispatch-context construction, outgoing priming adapter, output-mask lookup, and exchange topology mask/regridder setup built on (14, 19, 20, 38, 41, 42, 43, 45, 46)
48. `vercor/runtime/runner.py` - host/scanned runtime loops, callback-safe progress logging, compiled-runtime cache keys, JIT wrapping, donation checks, and interrupt translation built on (36, 37, 44, 46)
49. `vercor/output.py` - runtime-view NetCDF output boundary built on (6, 14, 39, 43)
50. `vercor/coupler.py` unified runtime facade - canonical `run()` / `create_runtime_state()` path, component registration through the base component contract, runtime component views, final output, callback-backed logging setup, and thin delegation to runtime-owned state/topology/runner adapters built on (1, 36, 37, 38, 39, 42, 43, 44, 45, 46, 47, 48, 49)
