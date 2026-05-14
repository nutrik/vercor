1. `vercor/dtypes.py` - canonical JAX/NumPy dtype policy and array-construction helpers
2. `vercor/pytree.py` - shared declarative PyTree mixin for immutable JAX-registered containers
3. `vercor/settings.py` - unified metadata-backed `VercorSettings` container with dynamic attribute access, typed known-setting annotations, default settings records, physical constants, runtime/component settings, and settings-bound dtype policy consumed by translated kernels
4. `vercor/fluxes/utilities.py` - JAX-native scalar/array thermodynamic helpers, shared virtual-temperature conversion, and hybrid-sigma altitude kernels built on (1, 3)
5. `vercor/fluxes/bulk_formula_cesm.py` - JAX-native atmosphere-ocean / atmosphere-ice bulk flux kernels built on (1, 3, 4)
6. `vercor/host_arrays.py` - explicit JAX/NumPy host-transfer boundary for non-differentiable adapters and output
7. `setups/_lazy_imports.py` - shared lazy package export helper for setup modules with optional dependencies
8. `setups/external/jax_gcm_tools.py` - existing JAX helper layer used by the JCM adapter; validated for `jax.jit` and built on (1, 4)
9. `setups/external/jax_gcm.py` - JCM adapter boundary that stores translated kernel outputs built on (1, 2, 8)
10. `setups/external/veros_runtime_settings.py` and `setups/external/veros_gcm.py` - explicit Veros host-runtime configuration plus the Veros adapter boundary that converts translated flux outputs back to NumPy for Veros state updates built on (1, 5, 6)
11. `setups/external/camulator.py` - CAMulator adapter boundary with JAX-backed runtime-field helpers, shared hybrid-sigma altitude diagnostics, and explicit Torch / xarray output boundaries built on (1, 4, 6)
12. `vercor/grid.py` - JAX-friendly `RectilinearGrid` holder with eager validation and PyTree registration built on (2)
13. `vercor/field_layout.py` - canonical component data-field layout validation and time-last forcing normalization helpers built on (12)
14. `vercor/regridders/helpers.py` - JAX-native rectilinear helper kernels built on (1, 12)
15. `vercor/interpolators/bilinear_rectilinear.py` - JAX-native bilinear scalar/vector interpolation and extrapolation built on (1, 2, 12, 14)
16. `vercor/interpolators/conservative_remap_rectilinear.py` - JAX-native conservative scalar remapping runtime with eager overlap preprocessing built on (1, 2, 12, 14)
17. `vercor/regridders/conservative.py` - conservative regridder wrapper over (15)
18. `vercor/grid_masks.py` - grid identity, component lookup, land/ocean mask construction, and remap-conservation checks built on (1, 12, 17)
19. `vercor/assets.py` - forcing asset cache/download/checksum boundary for data components
20. `vercor/forcing_data.py` - NetCDF forcing-file read boundary for data components
21. `vercor/time_selection.py` - calendar, day-slice, and periodic interpolation index helpers
22. `setups/slab/atmosphere.py` - slab atmosphere wrapper over pure JAX bulk-flux, default-SST, and wind kernels built on (1, 12)
23. `setups/slab/ocean.py` - slab ocean wrapper over pure JAX SST tendency kernel built on (1, 12)
24. `setups/slab/land.py` - slab land wrapper over pure JAX soil-moisture update kernel built on (1, 12)
25. `setups/slab/seaice.py` - slab sea-ice wrapper over pure JAX ice-fraction diagnostic kernel built on (1, 12)
26. `setups/data/_field_helpers.py` - shared JAX-backed data-adapter field normalization helpers, including masked time-last surface fields built on (1, 13)
27. `setups/data/era5_atmosphere.py` - pure ERA5 atmospheric data component with canonical data-field layout and JAX-backed pressure/model-level diagnostic initialization built on (4, 12, 13, 19, 20)
28. `setups/data/era5_ocean.py` - ERA5 ocean forcing adapter with canonical data-field layout, JAX-backed mask, and shared SST masking built on (12, 13, 19, 20, 26)
29. `setups/data/era5_land.py` - ERA5 land forcing adapter with canonical data-field layout, JAX-backed mask preparation, and runtime temperature storage built on (12, 13, 19, 20)
30. `setups/data/erainterim_ocean.py` - ERA-Interim ocean forcing adapter with canonical data-field layout, shared SST masking, and JAX-backed global field assembly built on (12, 13, 19, 20, 26)
31. `setups/data/jcm_land.py` - JCM land forcing adapter with canonical data-field layout, JAX-backed coordinate conversion, and runtime storage built on (12, 13, 18)
32. `setups/data/camulator_land.py` - CAMulator land forcing adapter with JAX-backed runtime temperature storage and forcing-only CAMulator config loading built on (1, 11, 12, 18)
33. `vercor/jax_logging.py` - callback-backed logger protocol and setup helper for Python and traced JAX runtime diagnostics
34. `vercor/runtime/interrupts.py` - internal terminal-signal runtime cancellation controller with host and JAX callback checkpoints, plus wakeup-fd polling for compiled runtime signals, built on JAX callback errors and Python signal handling
35. `vercor/runtime/contexts.py` - immutable component initialization and runtime step context payloads built on clock, run-sequence, settings helpers, and (33)
36. `vercor/runtime/views.py` - explicit runtime component metadata/field view used by diagnostics and output built on (12)
37. `vercor/diagnostics.py` - runtime-view means tables, plotting helpers, and plotting-only derived field helpers built on (6, 36)
38. `vercor/components/_contracts.py`, `vercor/components/_callable_wrappers.py`, `vercor/components/_runtime_fields.py`, `vercor/components/_validation.py`, and `vercor/components/base.py` - private component contract normalization, callable-wrapper runtime internals, component-facing runtime-field adapters, setup validation, and the slim public component-author surface for active differentiable components, data-only components, host-runtime adapters, top-level authoring helpers, `from_fields()` / `from_model()` authoring facade, flexible one/two/three-argument callable steps, public setup/step context aliases, `ComponentFieldSpec` declarations and introspection, setup field-name introspection, chainable component setting updates, declared grid-field default builders, base declared-default initialization, automatic data-component output declarations, scalar-to-grid field defaults/seeds, explicit runtime contexts, canonical data-field validation, step-result application, and finalized runtime boundary hooks built on (1, 12, 13, 35)
39. `vercor/runtime/contracts.py` - runtime import/export contracts, exchange-field flattening, stable exchange mask key names, and contract construction built on exchanges and coupler errors
40. `vercor/runtime/stores.py`, `vercor/runtime/state.py`, and `vercor/runtime/__init__.py` - immutable runtime field stores, component/coupler runtime state containers, and the internal `vercor.runtime` re-export surface built on (1, 2, 39)
41. `vercor/runtime/time.py` - `RuntimeStepInfo` plus host-precomputed daily/monthly runtime step metadata built on clock/settings helpers and (2, 21, 40)
42. `vercor/runtime/component_state.py`, `vercor/runtime/field_transfer.py`, `vercor/runtime/validation.py`, and `vercor/runtime/components.py` - component runtime state creation, contract prefill, receive/send, time-sliced export selection, runtime validation, and compatibility reexports built on (1, 13, 38, 39, 40, 41)
43. `vercor/runtime/exchange_dispatch.py` and `vercor/runtime/driver.py` - pure exchange dispatch, bundled runtime dispatch context, single per-component runtime step helper with explicit host-runtime allowance, outgoing priming, and host-adapter detection built on (35, 38, 39, 40, 41, 42)
44. `vercor/runtime/coupler_state.py` and `vercor/runtime/topology.py` - runtime coupler-state assembly, contract refresh, dispatch-context construction, outgoing priming adapter, output-mask lookup, and exchange topology mask/regridder setup built on (12, 17, 18, 35, 38, 39, 40, 42, 43)
45. `vercor/runtime/runner.py` - host/scanned runtime loops, callback-safe progress logging, compiled-runtime cache keys, JIT wrapping, donation checks, and interrupt translation built on (33, 34, 41, 43)
46. `vercor/output.py` - runtime-view NetCDF output boundary built on (6, 12, 36, 40)
47. `setups/external/jax_gcm.py` runtime payload path - immutable JCM state/forcing runtime integration built on (2, 9, 35, 38, 40, 42)
48. `vercor/coupler.py` unified runtime facade - canonical `run()` / `create_runtime_state()` path, component registration through the base component contract, runtime component views, final output, callback-backed logging setup, and thin delegation to runtime-owned state/topology/runner adapters built on (1, 33, 34, 35, 36, 39, 40, 41, 42, 43, 44, 45, 46, 47)
