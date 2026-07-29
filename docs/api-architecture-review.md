# VerCOR 0.4.0 API architecture review

This review records the stable 0.4.0 architecture. Public compatibility is
defined by canonical owner modules, their `__all__` manifests, live signatures,
and behavior tests—not by the layout of private implementation modules.

## 1. Executive summary

VerCOR 0.4 is a deliberate protocol-first break from 0.3. A configured
`Coupler` normalizes structural components once, validates stable exchange
routes and topology, asks a workflow for an exact clock-step plan, executes
core-defined chunks through a backend and validated driver, and coordinates all
requested output at the run boundary.

The package root contains six conveniences only: `Clock`, `Coupler`, `Exchange`,
`RectilinearGrid`, `RunState`, and `RuntimeOptions`. Advanced contracts each
have one canonical owner. `ComponentSpec` is the sole component declaration;
`PhysicalConstants` is the traced physics PyTree; `RuntimeOptions` owns static
execution policy; setup configuration belongs to the setup or plugin.

The stable extension tier is the six-symbol root plus `vercor.components`,
`vercor.coupler`, `vercor.exchanges`, `vercor.grids`, `vercor.output`,
`vercor.physics`, `vercor.regridding`, `vercor.runtime`, `vercor.state`,
`vercor.topology`, and `vercor.types`. The complete public manifest and
signature JSON retain the rest of the current inventory without making every
helper an independent workflow-stability promise.

The differentiable default is `Coupler.run(output=None)`. That path performs no
I/O and retains JVP and reverse-mode behavior. An `OutputTarget` opts into the
single coordinator for provider sampling, selection, immutable accumulation,
host transfer, period files, final fields, and snapshots. Optional JCM, Veros,
and CAMulator dependencies remain lazy. CAMulator is source-tested but is not
installed or pinned for this release.

Component identity is immutable through every lifecycle callback: the declared
name, grid, and specification remain author-owned configuration. External model
evolution lives only in setup/runtime payloads and is replaced functionally by
each step, never retained as hidden adapter state.

## 2. Duplication map

| Earlier overlap | VerCOR 0.4 owner | Resolution |
| --- | --- | --- |
| Inherited, callable, host, and data authoring hierarchies | `vercor.components` | One structural `Component` protocol plus `CallableComponent` and `DataComponent`. |
| Defaults, initialization, payload creation, transfer, and output properties | `ComponentSpec` | One immutable declaration and one `LifecycleHooks.setup` result. |
| Coupler recipes, mutators, and facade reexports | `vercor.coupler.Coupler` | Constructor-only assembly with immutable configuration views. |
| Callable-derived exchange keys and topology tuple keys | `Exchange.route_id` | Stable route identity used by topology and validation. |
| Backend-specific scheduling | `Workflow` and the private execution coordinator | One validated plan and core-owned chunks for every backend. |
| Backend/native output sessions | `vercor.output._session` | One immutable accumulator and one run-level output lifecycle. |
| Public and internal state mutation helpers | `RunState.replace_fields` | One immutable public replacement operation; alignment remains private. |
| Setup import registries | `vercor.setups` | One lazy public facade; implementation imports occur at factory invocation. |

These consolidations are intentionally narrow. Scalar and vector regridding
remain separate capabilities. Provider sampling and file writing remain
separate boundaries. Physics values and static precision policy remain
separate because JAX traces the former and uses the latter to control arrays.

## 3. Bad design decisions

Each decision below records the observed problem, its consequence, the concrete
0.4 correction, and whether it is required for a sound public contract or is
deferred internal cleanup.

| Problem | Consequence | Concrete fix | Priority |
| --- | --- | --- | --- |
| Broad root reexports and aliases had multiple apparent owners. | Imports drifted and users could not tell which module defined compatibility. | Keep exactly six root conveniences and one canonical public owner for every advanced symbol. | **must change** |
| Mutable coupler assembly and lifecycle replacement of `name`, `grid`, or `spec` could change identity after registration. | Public component maps, prepared bindings, routes, and runtime state could disagree. | Use constructor-only assembly, preserve the original identity references, and revalidate after setup, prefill, and validation callbacks. | **must change** |
| Reflection over arbitrary author attributes tried to infer configuration identity. | Harmless caches looked like configuration while hidden mutable state escaped validation. | Validate only the explicit structural contract and the three static identity attributes. | **must change** |
| Veros and CAMulator evolution lived on adapter closures. | Reusing or supplying a `RunState` was not reproducible, and output could sample a different run. | Seed native state in `SetupResult.payload`, return replacements in `StepResult`, and sample output contexts only. | **must change** |
| CAMulator silently reindexed mismatched starts and accepted model-calendar clocks it cannot represent. | A run could use the wrong forcing index or fail later during host execution. | Require explicit `strict` or `forcing_start` policy and reject non-standard-library datetime clocks during lifecycle setup. | **must change** |
| Static typing and runtime introspection used different regridder-factory definitions. | Built-ins and plugins could satisfy one checker while violating the other. | Define one runtime-checkable `RegridderFactory` protocol requiring the source and target grids. | **must change** |
| During alpha hardening, plugin metadata excluded `0.4.0a1`, the alpha it was intended to test, while installation bypassed dependency resolution. | The fixture had not proved a normally installable third-party extension. | The stable fixture now requires `vercor>=0.4.0,<0.5` and installs with ordinary resolution against built artifacts. | **must change** |
| Component-specific output markers and native accumulators duplicated cadence. | Compiled execution could skip output, and native adapters disagreed about paths and means. | Keep provider sampling at components but give one private session cadence, accumulation, host transfer, and file ownership. | **must change** |
| Custom backends could invoke arbitrary component steps. | Schedule order, cancellation, state checks, and output boundaries were unverifiable. | Give backends core-authored chunks and require exact consumption through `RuntimeDriver.run_step`. | **must change** |
| Callable-derived exchange identity collided for repeated endpoints. | Topology patches and diagnostics could not address routes deterministically. | Give every exchange a stable explicit or deterministic `route_id` and reject collisions and ambiguous fan-in. | **must change** |
| Shape-only foreign-state checks admitted changed coordinates, dtypes, payload schemas, and masks. | Structurally incompatible state could reach physics kernels or external backends. | Validate exact names, grids, shapes, dtypes, payload structure where required, and finite mask constraints at every boundary. | **must change** |
| Eager optional-model imports made core import depend on unused frameworks. | Installing or importing core VerCOR could fail because of an unselected model stack. | Keep `vercor.setups` as a lazy facade and load optional frameworks only when their factory is called. | **must change** |
| Runtime schema information is still represented by several private views. | Internal changes require coordinated edits even though public behavior is correct. | Consider one private schema owner in a later internal-only refactor; do not expose it. | **nice to improve** |
| Output finalization remains split across a few private coordinator helpers. | Internal ownership takes more navigation than necessary but cadence and writes are already singular. | Consider consolidating finalization behind the existing private session without changing provider or target contracts. | **nice to improve** |

A fan-in reducer, public prepared graph, registry, entry-point discovery,
Pydantic hierarchy, and fractional subcycling remain out of scope: no evidenced
0.4 workflow requires them.

## 4. Public API redesign

The machine-readable inventory below is ordered by canonical owner. CI executes
it against each live module and independently checks central constructor
signatures from installed artifacts.

<!-- public-api-manifest:start -->
```json
{
  "vercor": ["Clock", "Coupler", "Exchange", "RectilinearGrid", "RunState", "RuntimeOptions"],
  "vercor.assets": ["VERCOR_ASSETS_BASE_URL", "ensure_registered_asset"],
  "vercor.calendar": ["CalendarDate", "DAYS_PER_MONTH_360", "DAYS_PER_MONTH_GREGORIAN_LEAP", "DAYS_PER_MONTH_GREGORIAN_NO_LEAP", "DateTime360", "DateTime365", "ModelDateTime", "YearType", "day_of_year_from_month_day", "is_leap_year", "model_year_seconds", "month_day_from_day_of_year", "year_type_for_calendar"],
  "vercor.cli": ["cli"],
  "vercor.clock": ["Clock"],
  "vercor.components": ["CallableComponent", "Component", "ComponentSpec", "DataComponent", "LifecycleHooks", "PrefillContext", "PrefillResult", "SetupContext", "SetupResult", "StepContext", "StepResult", "TransferPolicy", "ValidationContext"],
  "vercor.coupler": ["Coupler"],
  "vercor.diagnostics": ["ComponentMetric", "combine_surface_temperatures", "component_vector_speed", "plot_component_scalar_vector_comparison", "print_component_field_means_table", "safe_component_nanmean", "total_surface_temperature"],
  "vercor.dtypes": ["DTypePolicy", "PrecisionPolicy", "ShapeLike", "as_jax_index_array", "as_jax_real_array", "dtype_policy", "jax_arange", "jax_full", "jax_index_dtype", "jax_linspace", "jax_ones", "jax_real_dtype", "jax_zeros"],
  "vercor.exceptions": ["AssetError", "ComponentError", "CouplerError", "ExchangeError", "GridError", "RegridderError"],
  "vercor.exchanges": ["Exchange"],
  "vercor.field_layout": ["CANONICAL_DATA_LAYOUTS", "canonical_data_layout_description", "canonical_grid_field_shape", "canonical_grid_field_shape_error", "canonicalize_time_last_level_field", "canonicalize_time_last_surface_field", "is_canonical_grid_field_shape", "validate_canonical_grid_field_shape", "validate_component_data_layout"],
  "vercor.fields": ["COMMON_FIELD_NAMES", "ExchangeField", "VectorField", "vector"],
  "vercor.fluxes": ["cdn", "compute_air_density", "compute_hybrid_pressure_levels", "compute_hybrid_sigma_full_level_altitudes", "compute_ocean_surface_fluxes", "compute_potential_temperature", "compute_sigma_pressure_levels", "get_altitudes_hybrid_sigma_levels", "get_altitudes_sigma_levels", "psimhu", "psixhu", "qsat", "qsat_august_eqn", "shr_flux_atmIce"],
  "vercor.forcing_data": ["read_forcing"],
  "vercor.forcing_index": ["ForcingYearType", "daily_forcing_day_of_year", "daily_forcing_index", "day_of_year_360_to_gregorian", "gregorian_month_lengths", "noleap_day_of_year"],
  "vercor.grid_geometry": ["centers_to_edges", "grids_identical"],
  "vercor.grid_masks": ["check_remap_conservation", "check_total_lnd_ocn_mask_sum", "compute_land_mask", "compute_ocn_lnd_masks_on_atm_grid", "create_lnd_mask_from_ocn"],
  "vercor.grids": ["RectilinearGrid"],
  "vercor.jax_logging": ["CANONICAL_LOG_DATE_FORMAT", "CANONICAL_LOG_FORMAT", "DEFAULT_LOGGER_NAME", "JaxCallbackLogger", "LoggerLike", "configure_python_logger", "effective_log_level", "emit_host_log", "get_default_logger", "logger_enabled_for", "normalize_log_level", "setup_logger"],
  "vercor.output": ["OutputContext", "OutputFrame", "OutputProvider", "OutputSpec", "OutputTarget", "OutputVariable", "PeriodOutput", "SnapshotContext", "SnapshotWriter"],
  "vercor.physics": ["PhysicalConstants"],
  "vercor.recipes": ["ATMOSPHERE_TO_DATA_OCEAN_FIELDS", "ATMOSPHERE_TO_JCM_LAND_FLUX_FIELDS", "ATMOSPHERE_TO_LAND_BASIC_FIELDS", "ATMOSPHERE_TO_LAND_RADIATION_FIELDS", "ATMOSPHERE_TO_LAND_STATE_FIELDS", "ATMOSPHERE_TO_OCEAN_RADIATION_FIELDS", "ATMOSPHERE_TO_OCEAN_STATE_FIELDS", "ATMOSPHERE_TO_VEROS_FORCING_FIELDS", "JCM_ATMOSPHERE_TO_SLAB_OCEAN_FIELDS", "JCM_LAND_TO_ATMOSPHERE_FIELDS", "LAND_TO_ATMOSPHERE_SOIL_FIELDS", "LAND_TO_ATMOSPHERE_SURFACE_FIELDS", "OCEAN_TO_ATMOSPHERE_SURFACE_FIELDS", "OCEAN_TO_SEAICE_SURFACE_FIELDS", "SEAICE_TO_OCEAN_FIELDS", "SLAB_ATMOSPHERE_TO_LAND_FLUX_FIELDS", "SLAB_ATMOSPHERE_TO_OCEAN_FIELDS", "SLAB_ATMOSPHERE_TO_OCEAN_FLUX_FIELDS"],
  "vercor.regridding": ["Regridder", "RegridderFactory", "VectorRegridder", "bilinear", "conservative"],
  "vercor.runtime": ["ExecutionBackend", "ExecutionChunk", "ExecutionContext", "ExecutionPlan", "RuntimeDriver", "RuntimeOptions", "SequentialWorkflow", "StepPlan", "Workflow", "WorkflowContext"],
  "vercor.setups": ["CAMulatorConfig", "JAXGCMConfig", "JCMLandAtmosphereConfig", "JCMLandAtmosphereSetup", "JCMInputs", "Spinup", "VerosConfig", "load_jcm_inputs", "make_slab_atmosphere", "make_slab_land", "make_slab_ocean", "make_slab_seaice", "make_jcm_land_atmosphere", "make_camulator_gcm", "make_camulator_land", "make_era5_atmosphere", "make_era5_land", "make_era5_ocean", "make_erainterim_ocean", "make_jax_gcm", "make_jcm_land", "make_veros_gcm"],
  "vercor.state": ["ComponentState", "FieldLookupScope", "FieldScope", "RunState"],
  "vercor.time_selection": ["datetime_to_seconds_in_year", "get_periodic_interval"],
  "vercor.topology": ["ExchangeTopologyPatch", "SurfaceMaskPolicy", "TopologyContext", "TopologyPolicy"],
  "vercor.types": ["RuntimeArray"]
}
```
<!-- public-api-manifest:end -->

The central signatures are constructor-only and keyword-delimited where policy
could otherwise be confused with data:

```text
Coupler(clock, *, components=(), exchanges=(), run_order=(), runtime=None,
        constants=None, logger=None, log_level="INFO")
Exchange(source, target, fields, *, route_id=None, regridder_factory=bilinear)
ComponentSpec(inputs=(), outputs=(), initial_fields=None, *, execution="jax",
              lifecycle=None, transfer=None, output=None)
RuntimeOptions(dtype=DTypePolicy(), backend="auto", workflow=SequentialWorkflow(),
               topology=None)
OutputTarget(directory, *, write_period=True, write_final_fields=True,
             write_snapshots=True)
```

The five signatures above are a readable representative sample. The complete,
static executable inventory is
[`tests/contracts/vercor-0.4.0-public-signatures.json`](../tests/contracts/vercor-0.4.0-public-signatures.json):
it covers all 150 concrete callable exports in canonical non-root owner
manifests and 55 public class/protocol-call methods. Every
normalized value includes parameter order and kind, defaults, resolved public
annotations, and the return annotation. Source and isolated installed-artifact
tests require exact key-set equality and execute the same frozen contracts.

Model-year policy is calendar-owned rather than runtime-owned:
`Clock.calendar` selects the calendar, and `vercor.calendar` resolves each
timestamp's canonical year type and duration for runtime forcing metadata.

`RunState` is opaque. Its public operations are `component`, `components`, and
`replace_fields`; callers never receive runtime stores, topology maps, or a
prepared graph. Public annotations resolve without importing private symbols.

## 5. Private API redesign

Private layout is free to change and is not a compatibility contract. The
following inventory is complete for `0.4.0` and documents responsibility,
not import permission.

```text
vercor._field_names
vercor._host_arrays
vercor._interpolators
vercor._interpolators._bilinear_extrapolation
vercor._interpolators._bilinear_geometry
vercor._interpolators._bilinear_weights
vercor._interpolators.bilinear_rectilinear
vercor._interpolators.conservative_remap_rectilinear
vercor._logging
vercor._logging.callback
vercor._logging.config
vercor._logging.host
vercor._logging.protocols
vercor._pytree
vercor._regridders
vercor._regridders.base
vercor._regridders.bilinear
vercor._regridders.conservative
vercor._run_order
vercor._runtime
vercor._runtime.backends
vercor._runtime.component_state
vercor._runtime.contracts
vercor._runtime.coupler_state
vercor._runtime.dispatch_context
vercor._runtime.driver
vercor._runtime.exchange_dispatch
vercor._runtime.exchange_topology
vercor._runtime.execution
vercor._runtime.facade
vercor._runtime.field_transfer
vercor._runtime.initialization
vercor._runtime.interrupts
vercor._runtime.preparation
vercor._runtime.prepared
vercor._runtime.progress
vercor._runtime.run_context
vercor._runtime.state
vercor._runtime.state_validation
vercor._runtime.stores
vercor._runtime.surface_masks
vercor._runtime.time
vercor._runtime.topology
vercor._runtime.topology_policy
vercor._runtime.topology_state
vercor._runtime.validation
vercor.components._adapter
vercor.components._callable_wrappers
vercor.components._contracts
vercor.components._protocol
vercor.components._runtime_fields
vercor.components.base
vercor.components.contexts
vercor.components.contracts
vercor.components.data
vercor.components.runtime_execution
vercor.components.setup_validation
vercor.diagnostics.fields
vercor.diagnostics.plotting
vercor.diagnostics.tables
vercor.fluxes.bulk_formula_cesm
vercor.fluxes.utilities
vercor.fluxes.vertical_coordinates
vercor.output._dataset
vercor.output._netcdf
vercor.output._period
vercor.output._runtime
vercor.output._session
vercor.setups._data
vercor.setups._data._component_helpers
vercor.setups._data._field_helpers
vercor.setups._data.assets
vercor.setups._data.era5_atmosphere
vercor.setups._data.era5_land
vercor.setups._data.era5_ocean
vercor.setups._data.erainterim_ocean
vercor.setups._data.jcm_land
vercor.setups._external
vercor.setups._external._camulator_wind_filtering
vercor.setups._external._jax_gcm_pytree
vercor.setups._external.camulator
vercor.setups._external.camulator_contracts
vercor.setups._external.camulator_fields
vercor.setups._external.camulator_forcing
vercor.setups._external.camulator_gcm_state
vercor.setups._external.camulator_imports
vercor.setups._external.camulator_init
vercor.setups._external.camulator_land
vercor.setups._external.camulator_output
vercor.setups._external.camulator_runtime
vercor.setups._external.camulator_runtime_settings
vercor.setups._external.camulator_stepper
vercor.setups._external.camulator_tensors
vercor.setups._external.camulator_wind_filter
vercor.setups._external.jax_gcm
vercor.setups._external.jax_gcm_fields
vercor.setups._external.jax_gcm_output
vercor.setups._external.jax_gcm_runtime
vercor.setups._external.jax_gcm_state
vercor.setups._external.jax_gcm_tools
vercor.setups._external.veros_fluxes
vercor.setups._external.veros_gcm
vercor.setups._external.veros_gcm_state
vercor.setups._external.veros_output
vercor.setups._external.veros_runtime
vercor.setups._external.veros_runtime_settings
vercor.setups._external.veros_setup
vercor.setups._external.veros_state
vercor.setups.gallery
vercor.setups.gallery.custom_component_wrapping
vercor.setups.gallery.profile_runtime
vercor.setups.gallery.run_camulator_with_veros
vercor.setups.gallery.run_data_driver
vercor.setups.gallery.run_jcm_with_era5data
vercor.setups.gallery.run_jcm_with_slab
vercor.setups.gallery.run_jcm_with_veros
vercor.setups.gallery.run_jcm_with_verosdata
vercor.setups.gallery.run_slab_driver
vercor.setups.gallery.run_veros_with_era5data
vercor.setups._jcm
vercor.setups._lazy_imports
vercor.setups._output
vercor.setups._slab
vercor.setups._slab.atmosphere
vercor.setups._slab.land
vercor.setups._slab.ocean
vercor.setups._slab.seaice
vercor.setups._time_helpers
vercor.setups.config
```

### Foundations and numerical implementations

- Field/runtime helpers: `vercor._field_names`, `vercor._host_arrays`,
  `vercor._pytree`, and `vercor._run_order`.
- Interpolation: `vercor._interpolators`,
  `vercor._interpolators._bilinear_extrapolation`,
  `vercor._interpolators._bilinear_geometry`,
  `vercor._interpolators._bilinear_weights`,
  `vercor._interpolators.bilinear_rectilinear`, and
  `vercor._interpolators.conservative_remap_rectilinear`.
- Logging: `vercor._logging`, `vercor._logging.callback`,
  `vercor._logging.config`, `vercor._logging.host`, and
  `vercor._logging.protocols`.
- Regridding: `vercor._regridders`, `vercor._regridders.base`,
  `vercor._regridders.bilinear`, and `vercor._regridders.conservative`.

### Runtime coordinator

The private runtime package consists of `vercor._runtime` and these focused
owners: `backends`, `component_state`, `contracts`, `coupler_state`,
`dispatch_context`, `driver`, `exchange_dispatch`,
`exchange_topology`, `execution`, `facade`, `field_transfer`, `initialization`,
`interrupts`, `preparation`, `prepared`, `progress`, `run_context`,
`state`, `state_validation`, `stores`, `surface_masks`, `time`, `topology`,
`topology_policy`, `topology_state`, and `validation`, each beneath
`vercor._runtime`.

`prepared` owns the single immutable post-setup binding. It contains normalized
components, routes, contracts, topology, clock, constants, and static runtime
policy; it contains neither reflective author snapshots nor a public prepared
graph. `execution` validates the workflow and owns chunk boundaries.
`backends` adapts JAX, host, and custom executors. `driver` is the only component
dispatch route exposed through the public driver wrapper. State validation
covers exact components, fields, payload structure, route maps, coordinates,
shapes, dtypes, and finite mask constraints before and after external backend
calls.

### Component, diagnostics, flux, and output implementations

- Components: `vercor.components._adapter`, `_callable_wrappers`, `_contracts`,
  `_protocol`, `_runtime_fields`, `base`, `contexts`, `contracts`, `data`,
  `runtime_execution`, and `setup_validation`.
- Diagnostics: `vercor.diagnostics.fields`, `plotting`, and `tables`.
- Fluxes: `vercor.fluxes.bulk_formula_cesm`, `utilities`, and
  `vertical_coordinates`.
- Output: `vercor.output._dataset`, `_netcdf`, `_period`, `_runtime`, and
  `_session`.

The component adapter is the only declaration-to-runtime normalization
boundary. The output session is the only cadence and mean-accumulation owner;
`_netcdf` is the only file-writing primitive. There is no hidden output marker,
second component adapter, duplicate accumulator, or native period-file path.

### Bundled setup implementations

- Data package: `vercor.setups._data`, `_component_helpers`, `_field_helpers`,
  `assets`, `era5_atmosphere`, `era5_land`, `era5_ocean`,
  `erainterim_ocean`, and `jcm_land`.
- External package: `vercor.setups._external`, `_camulator_wind_filtering`,
  `_jax_gcm_pytree`, `camulator`, `camulator_contracts`, `camulator_fields`,
  `camulator_forcing`, `camulator_gcm_state`, `camulator_imports`,
  `camulator_init`, `camulator_land`, `camulator_output`, `camulator_runtime`,
  `camulator_runtime_settings`, `camulator_stepper`, `camulator_tensors`,
  `camulator_wind_filter`, `jax_gcm`, `jax_gcm_fields`, `jax_gcm_output`,
  `jax_gcm_runtime`, `jax_gcm_state`, `jax_gcm_tools`, `veros_fluxes`,
  `veros_gcm`, `veros_gcm_state`, `veros_output`, `veros_runtime`,
  `veros_runtime_settings`, `veros_setup`, and `veros_state`.
- Runnable gallery: `vercor.setups.gallery`, `custom_component_wrapping`,
  `profile_runtime`, `run_camulator_with_veros`, `run_data_driver`,
  `run_jcm_with_era5data`, `run_jcm_with_slab`, `run_jcm_with_veros`,
  `run_jcm_with_verosdata`, `run_slab_driver`, and
  `run_veros_with_era5data`.
- Remaining setup owners: `vercor.setups._jcm`, `vercor.setups._lazy_imports`,
  `vercor.setups._slab`, `vercor.setups._slab.atmosphere`,
  `vercor.setups._slab.land`, `vercor.setups._slab.ocean`,
  `vercor.setups._slab.seaice`, `vercor.setups._time_helpers`, and
  `vercor.setups.config`.

Public setup access is always through `vercor.setups`. Private factories defer
JCM/Dinosaur, Veros, CREDIT, Torch, and TensorFlow imports until invocation.
JAXGCM, Veros, and CAMulator expose ordinary output providers and snapshot
writers; they do not own cadence, paths, period accumulation, or writes.

## 6. Setup-agnostic plugin architecture

Plugins are ordinary Python packages that inject objects explicitly. There is
no registry or entry-point discovery. The temporarily built fixture under
`tests/fixtures/external_extension_test_fixture` proves the complete boundary
using only public imports:

1. a plugin-owned frozen configuration and assembly factory;
2. structural JAX and host components plus setup payload replacement;
3. a structural scalar regridder and injected factory on an explicit route ID;
4. a non-empty route-keyed topology patch;
5. a plugin workflow and chunk-oriented custom backend;
6. immutable `RunState.replace_fields` before driver execution; and
7. per-step provider output and a final snapshot.

Its wheel is built in a temporary directory for each contract job, then
installed with the VerCOR wheel in a clean target; it is not installed beside or
uploaded with the two VerCOR release artifacts. Its smoke runs outside the
checkout, and its package plus external use site pass strict mypy. CI repeats
the external-extension contract lanes on Python 3.12 and 3.13.
The executable [plugin authoring guide](plugin-authoring.md) documents the same
public-only contracts. The installed fixture protects the documented stable
extension tier rather than promising every retained public manifest as an
independent plugin workflow.

Bundled slab, data, JCM, and Veros factories return ordinary structural
components and use the same constructor and output contracts. Slab and data
factories accept a complete keyword-only `OutputSpec`; omission selects
`OutputSpec()` with no period policy, matching external configuration defaults.
Paired JCM configuration owns land and atmosphere output independently. CI has
installed base, JCM, and Veros lanes. CAMulator remains lazy and source-tested
because a compatible external release is not yet pinned.

## 7. Compatibility plan

VerCOR 0.4 is intentionally source-breaking and version 0.4.0 does not ship a
0.3 adapter namespace. Task 9 was explicitly skipped. Applications migrate
imports and construction directly using `docs/migration-0.3-to-0.4.md`;
primary 0.4 modules remain alias-free.

No legacy adapter namespace or executable VerCOR 0.3 evidence ships. No earlier
API is restored.

Compatibility within the 0.4.x line is defined by canonical public owner
manifests, signatures, public-only plugin behavior, output-free gradients, and
installed wheel/sdist tests. The root and stable extension tier are the
workflow-facing compatibility commitments; the remaining manifest is a current
inventory retained without export removal. Public PyTree hooks are JAX
integration details, not independent workflow promises. Private module names
in section 5 are descriptive and may change without a deprecation cycle.

## 8. Final rewritten API

### 8.1 Complete public API

The complete public module/class/function/protocol/configuration inventory is
the executable manifest in section 4. It is the actual stable `0.4.0` surface,
not a future sketch. Exact signatures for every callable export and public
behavioral method are in
[`vercor-0.4.0-public-signatures.json`](../tests/contracts/vercor-0.4.0-public-signatures.json);
section 4 also gives the five central readable signatures.

The stable workflow-facing subset is:

| Public owner | Stable contracts and purpose |
| --- | --- |
| `vercor` | `Clock`, `Coupler`, `Exchange`, `RectilinearGrid`, `RunState`, and `RuntimeOptions` for assembly and execution. |
| `vercor.components` | Structural `Component`; `ComponentSpec`; lifecycle contexts/results; `CallableComponent` and `DataComponent`. |
| `vercor.regridding`, `vercor.exchanges`, `vercor.topology` | `Regridder`, `VectorRegridder`, `RegridderFactory`, `Exchange`, and route-keyed topology policies. |
| `vercor.runtime` | `Workflow`, plans, chunks, `ExecutionBackend`, contexts, and the validated `RuntimeDriver`. |
| `vercor.output` | Providers, frames, variables, cadence/specification, targets, and snapshot contexts. |
| `vercor.state`, `vercor.physics`, `vercor.grids`, `vercor.types` | Opaque immutable state, traced physical constants, canonical grids, and runtime array typing. |
| `vercor.setups` | Frozen setup configurations and lazy built-in factories; retained public inventory rather than an additional plugin tier. |
| `vercor.cli` | Click command group for copying a packaged setup or running a local setup file. |

The built-in path remains a small constructor workflow:

```python
from datetime import datetime

from vercor import Clock, Coupler, RectilinearGrid
from vercor.setups import make_slab_ocean

grid = RectilinearGrid.uniform(
    "ocean", nlon=4, nlat=3,
    longitude=(0.0, 360.0), latitude=(-90.0, 90.0),
)
ocean = make_slab_ocean(grid)
coupler = Coupler(
    Clock(datetime(2000, 1, 1), dt_seconds=3600.0, steps=1),
    components=(ocean,),
    run_order=(ocean.name,),
)
final_state = coupler.run(output=None)
```

The complete custom configuration/component/regridder/topology/workflow/backend/
output example is executable in
[`plugin-authoring.md`](plugin-authoring.md). The temporary fixture in
`tests/fixtures/external_extension_test_fixture` verifies the same imports,
strict typing, normal dependency resolution, non-default configuration,
immutable state replacement, period output, and snapshot output outside the
checkout.

### 8.2 Complete private API

The complete private module inventory is the code block in section 5;
the responsibility groups immediately below it define every internal owner's
role. These names document `0.4.0` implementation relationships but are not
import or compatibility promises. Useful internal boundaries include:

```text
normalize_component(component) -> _ComponentDeclaration
prepare_component(declaration, context, dtype) -> _ComponentBinding
initialize_coupler_runtime(...) -> RuntimeInitializationState
create_runtime_state(*, prepared, prefill_missing) -> RunState
validate_runtime_state(runtime_state, *, prepared) -> None
build_validated_execution_plan(context) -> ExecutionPlan
execute_plan(state, *, plan, context) -> RunState
```

Public-to-private relationships are intentionally one-way:

| Public contract | Private support and reason it remains private |
| --- | --- |
| `Component` / `ComponentSpec` | `components._adapter`, `_contracts`, `_runtime_fields`: normalization, identity checks, payload copying, and dispatch policy may evolve. |
| `Coupler` / `RunState` | `_runtime.prepared`, `preparation`, `state_validation`, `facade`: prepared graphs, stores, schemas, and orchestration are implementation details. |
| `Exchange` / topology policies | `_regridders` and `_runtime.exchange_*`, `topology_*`, `field_transfer`: compiled routes, masks, and dispatch maps are backend-owned internals. |
| Workflows / execution backends | `_runtime.execution`, `backends`, `driver`, `interrupts`: chunk grouping, JIT scans, cancellation, and call accounting must remain core-controlled. |
| Output providers / targets | `output._session`, `_period`, `_dataset`, `_netcdf`, `_runtime`: schema locking, accumulation, filenames, host transfer, and writes remain replaceable internals. |
| `vercor.setups` factories | `setups._lazy_imports`, `_data`, `_external`, and `_slab`: framework adapters and model-specific validation can change independently. |

The final public/private data flow is:

```text
components + exchanges + clock + RuntimeOptions
                    |
                    v
       private immutable preparation
                    |
                    v
 Workflow -> ExecutionPlan -> validated chunks
                    |
                    v
       backend -> RuntimeDriver.run_step
                    |
                    v
              immutable RunState
                    |
          OutputTarget supplied?
             /             \
           no               yes
      no host I/O      one output coordinator
```

Everything below the public contracts in that flow remains private so it can be
optimized or reorganized without forcing plugin migrations. Wheel and
installed-sdist probes verify the public surface and PEP 561 marker outside the
checkout.

Deferred features stay deferred: no registry, entry-point discovery, Pydantic,
fan-in reducer, public prepared graph, fractional subcycling, or CAMulator
dependency pin. No tag, push, publication, or release upload is part of
preparing the stable release.
