# Dynamic Masks and Veris Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add evolving component masks, then a functional Veris callable component with Veros data, ERA5 forcing, native output, and a runnable coupled gallery.

**Architecture:** Reference ordinary runtime fields for dynamic masks while retaining immutable grid geometry and fixed remappers. Explicit exchange quantities distinguish intensive values from source/target area flux contributions. Wrap native Veris `initialize_experiment`/`advance`, share one setup-only Veros geometry snapshot with its data component, and reuse CESM surface calculations and VerCOR output ownership.

**Tech Stack:** Python >=3.12, JAX >=0.11.2 for Veris, NumPy, Veros, h5netcdf, pytest, Black, flake8, mypy, Git/GitHub CLI.

**Spec:** [Approved design](../specs/2026-10-16-veris-integration-design.md), approved 2026-10-16 including the user-requested CESM `tref` source for `ATemp`.

## Global Constraints

- Work on `feat/veris-seaice`; preserve the original `PROGRESS.md` edits and the subsequent `ATemp` review revision.
- Complete Tasks 1-3, including their full checks, before Stage 2; no CICE feedback routes to data components.
- Runtime geometry and `RectilinearGrid.binary_mask` remain static; native `Area` becomes traced `ice_fraction` on the Veros wet domain.
- Native Veris revision: `a9d6c0ade52fbae4c0d1da694878943d6dbf7e6e`; existing Veros/Veros-AD contracts stay intact.
- Use `vercor/setups/_external/`, lazy public setup exports, frozen PyTrees, explicit payload replacement, and existing native physics.
- `ATemp = CESM tref`, `aqh = CESM qref`, at the configured air-temperature reference height, 2 m by default. Retain colocated model-level inputs to that calculation.
- `Qnet` and `Qsw` are upward-positive native heat inputs; precipitation/snow are separate water-equivalent m/s; reapply prescribed forcing each native substep.
- Runtime dtype has one owner: `RuntimeOptions.dtype`; native configuration/constants are static, while state/forcing and fractional masks support AD.
- The gallery order is exactly `LND -> CICE -> OCN -> ATM`; its filename and `run_setup(*, loglevel, float_type)` contract are fixed.
- No numerical fudge factors, new model runner, hidden evolving native owner, unsolicited release, or unrelated refactor.
- Read `AGENTS.md`, `DESIGN.md`, `PROGRESS.md`; update progress after each unit, keeping it <=180 lines. Write tests before implementation.
- Use concise development `--fast` checks and full tests before each commit, as `AGENTS.md` requires. Detailed logs go to `test_logs/`.

## Review Focus

1. Zero ice concentration must still receive forcing on wet cells, allowing new ice to form: Task 3's ice-free receiver test and Task 6's native rollout.
2. Inactive NaNs must not poison derivatives; reactivated fields must become finite: Tasks 1-3's validation, remapping, and AD tests.
3. A supplied state with replaced mask fields must not retain stale route/output masks: Task 3's replacement and final-output tests.
4. Prescribed `Qnet`/`Qsw` must not recursively reuse native residuals across substeps: Tasks 5-6's sign and forcing-reset tests.
5. Staggered ocean velocity, humidity reference height, and spherical geometry must not be silently relabeled or converted twice: Tasks 4-5 and Task 7's geometry/output tests.

---

## File ownership and verification conventions

| Owner | Files and responsibilities |
| --- | --- |
| Public declarations | `components/contracts.py`, `components/__init__.py`, `exchanges.py`: mask references and quantity mode |
| Mask arithmetic | New `_runtime/component_masks.py`: field resolution/value validation; new `_runtime/exchange_masks.py`: route coverage and refresh |
| Runtime integration | Preparation, driver, dispatch, transfer, state validation and output handoff use current masks |
| Ocean resources | New `setups/_external/veros_data.py`: one immutable global-four-degree snapshot; new `setups/_data/veros_ocean.py`: monthly data component |
| Shared surface calculations | New `setups/_surface_fluxes.py`: array-only CESM/radiation results; existing `veros_fluxes.py` delegates without changing outputs |
| Native sea ice | New `_external/veris.py`, `veris_runtime.py`, `veris_forcing.py`, `veris_output.py`: factory, lifecycle/step, forcing, output |
| Paired construction | New `setups/_veris.py`: paired sea-ice/data-ocean result and factory; `config.py` owns frozen `VerisConfig` |
| Example | New `gallery/run_veris_dataveros_era5.py`; recipe declarations remain in `recipes.py` |
| Tests/docs | New focused test files below, existing public contracts, `DESIGN.md`, `DEPENDENCIES.md`, API/developer/setup documentation and progress |

All paths above are relative to `vercor/` unless prefixed otherwise. Update
`docs/api-architecture-review.md` public manifests/private inventory and
`tests/contracts/vercor-0.4.0-public-signatures.json` in the same task that
changes a public declaration or adds a private module; keep the root six exports.

During implementation activate the existing Python 3.12 environment named
`scipy`: `conda activate /private/tmp/vercor-time-ci/envs/scipy`. The default
Python 3.14 `scipy` aborts importing Torch with duplicate OpenMP initialization.
Network/cache/localhost permission needs are environmental; use approved tool
escalation rather than skip or alter affected tests.

Use these commands for the named gates, logging full output:

```bash
# Focused tests: replace the file/node with the task's exact selection.
python -m pytest tests/test_dynamic_component_masks.py -q --fast --tb=short
# Stock pre-commit fast and full gates, in separate invocations.
python -m pytest tests/ -q --fast -m 'not veros_autodiff' --tb=short
python -m pytest tests/ -q -m 'not veros_autodiff' --tb=short
# Formatting, strict lint, typing, compile and whitespace gates.
python -m black vercor tests
python -m flake8 . --count --max-line-length=120 --statistics
python -m mypy vercor tests
python -m compileall -q vercor tests
git diff --check
```

Before every task commit, run the affected focus, fast/full stock gates, and
relevant static checks; stage only that task's files plus progress/contracts.
Never claim the separate Veros AD lane from stock results. Read the final log
summary and exit code before recording a pass. Each test file participates in
the existing deterministic `--fast` selection; critical mask/sign/ownership
acceptance tests use `fast_always`.

## Stage 1: dynamically updated component masks

### Task 1: Declare and validate component mask fields

**Files:** Modify `vercor/components/contracts.py`, `components/__init__.py`,
`components/_adapter.py`, `components/setup_validation.py`, `_numerical_safety.py`.
Create `vercor/_runtime/component_masks.py`, `tests/test_dynamic_component_masks.py`.
Update `docs/api/components.rst`, dependency/API contracts and progress.

**Interfaces:**
- Produce `ComponentMaskSpec(binary_field: str | None = None, fractional_field: str | None = None)`, frozen in `components/contracts.py`.
- Add keyword-only `ComponentSpec(..., masks: ComponentMaskSpec | None = None)`; preserve all existing defaults.
- Produce private `resolve_component_masks(component: Component, fields: Mapping[str, RuntimeArray], dtype: DTypePolicy) -> tuple[RuntimeArray, RuntimeArray]`, returning effective binary support and fraction.
- If needed, add transform-safe `require_mask_values(values: object, *, owner: str, binary: bool) -> None` alongside existing numerical assertions; do not introduce host conversion in traced paths.

- [x] **1. Write failing declaration/resolution tests.** `test_mask_spec_references_declared_current_fields` rejects undeclared/empty/non-string names and masks on non-`current` transfer policy. `test_component_masks_respect_static_support` uses static wet `[1,1,0]`, binary `[1,0,0]`, fraction `[0.25,0,0]` and asserts those exact resolved arrays. `test_mask_values_reject_invalid_inputs` covers NaN/inf, fraction -0.1/1.1, nonbinary 0.5, containment, wrong shape/dtype, and the same failures through JIT. Mask fields themselves must be finite everywhere.
- [x] **2. Run RED.** Run the new file with `-n0 --fast`; verify missing declarations/resolution fail for the intended reason, not test import mistakes.
- [x] **3. Implement the interfaces and binding validation.** Mask fields are ordinary declared fields, normalized by existing setup machinery. Resolve defaults from immutable wet support, and validate fractions are zero outside dynamic binary support. Preserve the existing behavior for `masks=None`; avoid any evolving mask attributes on components.
- [x] **4. Run GREEN and public contracts.** New mask tests, `test_v0_4_component_contracts.py`, `test_component_boundary_contracts.py`, and API manifest/signature tests pass. Confirm an existing component's `spec` identity remains stable.
- [x] **5. Update docs/progress and run commit gates.** Explain current-time/grid-shape requirements, default support, and fractional-versus-binary AD semantics. Add the new low-level helper to the correct dependency layer without a runtime import of the high-level component facade.
- [x] **6. Commit.** `feat: declare runtime component mask fields`.

### Task 2: Add explicit intensive and area-flux exchange modes

**Files:** Modify `vercor/exchanges.py`, `_runtime/exchange_topology.py`,
`_runtime/topology_policy.py`, `_runtime/exchange_dispatch.py`, `vercor/state.py`.
Create `vercor/_runtime/exchange_masks.py`, `tests/test_dynamic_mask_exchanges.py`.
Update existing runtime-exchange tests, `docs/api/grids-exchanges.rst`, contracts and progress.

**Interfaces:**
- Add keyword-only `Exchange(..., quantity: Literal['legacy', 'intensive', 'source_flux', 'target_flux'] = 'legacy')`.
- Consume Task 1's mask resolver. Add private `exchange_coverage(state: RunState, exchange: Exchange, *, context: RuntimeDispatchContext) -> RuntimeArray` and `refresh_runtime_masks(state: RunState, *, context: RuntimeDispatchContext) -> RunState` in `exchange_masks.py` for Task 3 to integrate.
- Add a private immutable `RunState._with_fractional_masks(masks: FieldStore) -> RunState`; no public mutable stores.
- Preserve the existing `dispatch_component_exchanges` positional contract; add optional keyword `context: RuntimeDispatchContext | None = None` for explicit modes. Existing direct legacy tests remain valid.

- [x] **1. Write failing quantity tests.** For an equal-area remap `R(x)=mean(x)`, source fraction `[0.25,0.75]` and field `[2,10]` give `R(a*f)=4`, `R(a)=0.5`: assert intensive `8`, source flux `4`, and target flux `2` when target fraction is `0.25`. This also exposes the incorrect product-of-remapped-fields value `3`. Add vector-pair versions, empty-source coverage returning finite zero, inactive NaNs with finite JVP/VJP, and rejection of infinity even outside support. A spherical conservative case must preserve area-weighted source-flux integrals.
- [x] **2. Run RED.** Run `test_dynamic_mask_exchanges.py -n0 --fast` and verify the new mode/weighting assertions fail before implementation.
- [x] **3. Implement quantity validation and arithmetic.** Require an explicit quantity for routes touching dynamic declarations. Multiply source fields by source fraction before remapping; intensive/target-flux modes use existing `safe_masked_divide` with a strictly positive overlap condition. Gate intensive/source flux by target binary support, not target concentration. Use scalar coverage remapping and the existing vector transformation; never rebuild geometry from masks.
- [x] **4. Validate topology composition.** Explicit modes intersect prepared binary restrictions. Reject fractional patches that differ from the prepared binary gate for an explicit route, naming the route and conflict. Legacy routes retain existing scalar weighting and unchanged vector behavior. Cover both binary-patched and incompatible fractional-patched routes in tests.
- [x] **5. Run GREEN, update contracts/docs/progress and commit gates.** Existing vector-preservation tests stay unchanged and passing. Document that conservative remapping, rather than bilinear interpolation, provides integral conservation.
- [x] **6. Commit.** `feat: add explicit masked exchange quantities`.

### Task 3: Refresh masks across lifecycle, validation, and output

**Files:** Modify `vercor/_runtime/dispatch_context.py`, `prepared.py`,
`preparation.py`, `driver.py`, `field_transfer.py`, `state_validation.py`,
`facade.py`, `vercor/components/runtime_execution.py`, `vercor/output/_runtime.py`.
Extend the new mask tests and `tests/_runtime_helpers.py`; add
`tests/test_dynamic_mask_rollouts.py`. Update `DESIGN.md`, dependency/API docs and progress.

**Interfaces:**
- Add immutable `topology_binary_masks` and `topology_fractional_masks` mappings to `RuntimeDispatchContext` alongside its existing regridders, with identically named builder keyword arguments; update its builder and all direct test constructions.
- Consume `refresh_runtime_masks` at initial priming, after every component step before send, and for supplied states before execution validation.
- Produce `runtime_exchange_masks(state: RunState, *, context: RuntimeDispatchContext) -> tuple[dict[str, RuntimeArray], dict[str, RuntimeArray]]` for final output, returning current binary and fractional route maps. Fractional diagnostics record overlap/coverage, not intensive field attenuation.

- [x] **1. Write failing lifecycle tests.** `test_mask_refresh_respects_sequential_order` puts callable recipients before/after a source whose fraction changes `0.25 -> 0.75`; assert prior/new values respectively. `test_ice_free_receiver_keeps_intensive_forcing` asserts 270 K still arrives where target fraction is zero but wet support is one. Cover omitted components, initial prefill, scalar initial masks, and retained immutable input state.
- [x] **2. Write failing validation/output/AD tests.** `test_replaced_mask_fields_refresh_supplied_state` replaces an initialized fraction with `0.5` and checks fresh exchange/output coverage. `test_binary_reactivation_requires_finite_exports` permits an inactive NaN but rejects it when the binary support becomes one. `test_final_output_uses_final_masks` inspects written NetCDF masks after two changes. `test_fractional_mask_rollout_gradients` compares host/JIT values and JVP/VJP/central finite differences at interior fractions, using the analytic weighted two-cell case from Task 2.
- [x] **3. Run RED.** Run the three new mask files with `-n0 --fast`; verify timing, stale-state/output, and derivative failures are reproduced.
- [x] **4. Integrate mask resolution.** Use one resolver at component output/send/state validation boundaries. Validate mask values before using them to exempt other fields from finiteness. Refresh authoritative field-derived caches after each step and on supplied states; reject changed shape/dtype/schema instead of silently repairing invalid inputs. Pass final runtime masks to output, preserving fixed PyTree structure and the one-scan execution path.
- [x] **5. Run GREEN and Stage 1 acceptance.** All mask tests plus runtime, numerical-safety, workflow, output, and external-plugin focuses pass in full. Exercise the new public mask API in the existing external extension fixture. Run static checks, stock fast/full, and isolated Veros AD regression tests; record results and an independent review before moving to Stage 2.
- [x] **6. Commit.** `feat: propagate evolving masks through coupling lifecycle`.

## Stage 2: native Veris, prescribed ocean, and ERA5

### Task 4: Snapshot authentic Veros geometry and create its data ocean

**Files:** Create `vercor/setups/_external/veros_data.py`,
`vercor/setups/_data/veros_ocean.py`, `tests/test_veros_data_ocean.py`.
Update dependency/private-inventory documentation and progress.

**Interfaces:**
- Private frozen `VerosDataSnapshot(grid: RectilinearGrid, geometry: Mapping[str, RuntimeArray], fields: Mapping[str, RuntimeArray])` owns independent arrays, not a live native model.
- Private `load_veros_data_snapshot() -> VerosDataSnapshot` initializes the existing global-four-degree setup once, without spinup, stepping, or an extra retained solver/native owner.
- `make_veros_data_ocean(snapshot: VerosDataSnapshot, *, name: str = 'OCN', output: OutputSpec | None = None) -> DataComponent` reuses `time_interpolated_data_component`.
- Ocean outputs: `sea_surface_temperature`, `sea_surface_salinity`, `ocean_u_velocity`, `ocean_v_velocity`, `sea_surface_height`, all in canonical time-first layout for monthly selection.

- [x] **1. Write failing snapshot/data tests.** A non-square native-state fixture with distinct halo/interior values verifies every geometry slice, axis, and staggering. Assert native spacings/areas are unchanged, `ht` changes sign once, surface masks match the exact shared grid, SST 0 deg C becomes 273.15 K, salinity remains g/kg, and default currents/SSH are explicitly zero. Mutating the source native fixture must not affect the snapshot.
- [x] **2. Run RED.** Run the new file with `-n0 --fast`; verify absent snapshot/data APIs fail as expected.
- [x] **3. Implement extraction and the data component.** Use `maskT/U/V`, `ht`, `coriolis_t`, `dxt/dxu/dyt/dyu`, `area_t/u/v`, `sst_clim`, and `sss_clim` after native setup. Preserve `xt/yt`; retain corresponding `xu/yu` edges from the native haloed coordinates for later staggered output. Snapshot physical interiors; let public Veris initialization add its halos. Broadcast prescribed zero current/SSH fields to the same 12-record monthly layout. The data ocean declares no fictitious atmospheric inputs.
- [x] **4. Run GREEN with a real setup snapshot.** Assert only one native setup call, no native step call, no retained mutable owner, identical snapshot/ocean grid reference, and actual monthly interpolation at a record midpoint. Real Veros initialization runs in its own host-compatible process; do not change an already locked JAX Veros backend.
- [x] **5. Update progress/docs, run gates, and commit.** `feat: provide prescribed Veros surface data and geometry`.

### Task 5: Reuse CESM diagnostics and supply all native forcing

**Files:** Create `vercor/setups/_surface_fluxes.py`,
`setups/_external/veris_forcing.py`, `tests/test_veris_forcing.py`,
`tests/test_shared_surface_fluxes.py`. Modify `veros_fluxes.py`,
`setups/_data/era5_atmosphere.py`, `recipes.py`, and affected ERA5 fixtures/tests.
Update API/dependency documentation and progress.

**Interfaces:**
- Private `SurfaceFluxes` named tuple: `taux`, `tauy`, `qnet`, `qnec`, `evap`, `tref`, `qref`.
- `compute_surface_fluxes(runtime_fields: Mapping[str, RuntimeArray], *, sea_surface_temperature: RuntimeArray, u_ocean: RuntimeArray, v_ocean: RuntimeArray, ocean_mask: RuntimeArray, constants: PhysicalConstants, dtype: DTypePolicy) -> SurfaceFluxes`; currents here are tracer-centered.
- `prepare_veris_forcing(fields: Mapping[str, RuntimeArray], *, ocean_mask: RuntimeArray, constants: PhysicalConstants, dtype: DTypePolicy) -> dict[str, RuntimeArray]` returns native physical `(nx, ny)` forcing, with exact accepted keys from the approved table and no `wSpeed` override.
- Add ERA5 outputs `downward_shortwave_radiation_flux`, `u_velocity_10m`, `v_velocity_10m` from `msdwswrf/u10/v10`; do not add `t2m` as Veris forcing.
- Add recipes `ATMOSPHERE_TO_VERIS_FORCING_FIELDS` and `OCEAN_TO_VERIS_FORCING_FIELDS`; retain model-level and 10 m wind vectors as separate pairs.

- [ ] **1. Write failing forcing/reference-height tests.** With independently fixed bulk diagnostics `qnet=45`, `tref=268.5`, `qref=0.001`, `evap=-2e-5`, net shortwave 120, rain `3e-5`, snow `1e-5`, density 1000, assert native `Qnet=-45`, `Qsw=-120`, `ATemp=268.5`, `aqh=0.001`, `evap=-2e-8`, `precip=3e-8`, `snowfall=1e-8`. Assert both radiation fields remain distinct and wind uses the actual 10 m components. Use non-square tagged arrays to verify the one transpose.
- [ ] **2. Write failing reuse/geometry tests.** Compare real shared CESM outputs against the existing Veros outputs before extraction. Vary model height with fixed model-level fields and assert both `tref` and `qref` follow the native bulk diagnostic, not input temperature/humidity. Verify staggered-to-tracer current centering with unequal neighbor values, periodic longitude, and closed latitude; zero-current data are a separate explicit case. Inactive NaNs remain derivative-safe.
- [ ] **3. Run RED.** Run the two new files and ERA5/Veros forcing focuses; confirm failures originate at missing/new mappings or the intended conversion assertions.
- [ ] **4. Extract the common array calculation and implement packing.** Leave `compute_ocean_surface_fluxes` and physical formulas unchanged. Preserve Veros's existing native current centering and five-result contract while delegating only shared CESM/radiative arithmetic. Use those same `tref/qref` results for native Veris; document the open-ocean similarity assumption under ice. Convert mass fluxes once with traced freshwater density, retain signs, mask inactive operands before nonlinear arithmetic, and supply explicit zero SSH/runoff defaults. Before native equations, replace dry-cell temperature/pressure operands with safe 273.15 K/reference-pressure values and zero inactive fluxes; test that inactive zero/NaN input values cannot poison native derivatives.
- [ ] **5. Run GREEN, update ERA5 fixture schemas/recipes/docs/progress and gates.** Test both runtime dtypes, finite forcing JVP/VJP, no precipitation double counting, and unchanged existing Veros heat/salt/stress regressions.
- [ ] **6. Commit.** `feat: prepare Veris forcing from ERA5 and CESM diagnostics`.

### Task 6: Implement the native Veris component and paired factory

**Files:** Create `vercor/setups/_external/veris.py`, `veris_runtime.py`,
`vercor/setups/_veris.py`, `tests/test_veris_component.py`,
`tests/test_veris_rollouts.py`. Modify `setups/config.py`, `setups/__init__.py`,
`pyproject.toml`, setup-boundary/distribution contracts and progress.

**Interfaces:**
- Public frozen `VerisConfig(name: str = 'CICE', model_timestep: timedelta = timedelta(hours=1), custom_parameters: Mapping[str, object] | None = None, initial_conditions: Mapping[str, object] | None = None, output: OutputSpec = OutputSpec())` with owned mappings and a positive timestep.
- Public `make_veris(grid: RectilinearGrid, geometry: Any, *, config: VerisConfig | None = None) -> Component`; `geometry` is native `OceanGeometry`, checked after lazy import.
- Public frozen `VerisVerosDataSetup(seaice: Component, ocean: DataComponent)` and `make_veris_verosdata(*, config: VerisConfig | None = None, ocean_name: str = 'OCN', ocean_output: OutputSpec | None = None) -> VerisVerosDataSetup`.
- Private frozen registered `VerisPayload(state: Any, diagnostics: Any)`; both children are actual native registered PyTrees. Native resources are setup-owned immutable configuration/constants/substep policy.
- Private `step_veris_runtime(resources: VerisRuntimeResources, fields: Mapping[str, RuntimeArray], context: StepContext, payload: VerisPayload) -> StepResult`; `VerisRuntimeResources` is a frozen static owner defined in the same module for native configuration, native constants, and integer substeps.
- A private `VerisSetupState` in `veris_runtime.py` provides `setup(component: Component, context: SetupContext) -> SetupResult` and `step(fields: Mapping[str, RuntimeArray], context: StepContext, payload: VerisPayload) -> StepResult`. Setup assigns the frozen resources once; step delegates to `step_veris_runtime` and never changes the owner. All evolving native arrays stay in the payload.
- Native `Area` maps to output `ice_fraction`, declared through `ComponentMaskSpec(fractional_field='ice_fraction')`. Its binary support is the shared ocean wet mask.

- [ ] **1. Write failing lazy/configuration/ownership tests.** Core and config imports work without Veris/Veros; factory errors name the missing standalone capability/install extra. Check independent initial-condition arrays, duplicate component names, geometry/grid mismatch, reserved dtype/grid/timestep override conflicts, equal native dynamic/thermodynamic timesteps, and nondivisible/zero coupling intervals. Assert pair construction initializes Veros once and reuses the exact ocean grid.
- [ ] **2. Write failing native transition tests.** Initialize a small custom ocean with explicit `Area=0.5`, `hIceMean=0.25` and valid forcing; compare two wrapper steps with independent direct `veris.advance` calls. Use `dataclasses.replace`/native initialization, not a fake integrator, for this acceptance test. Assert unchanged input payload, stable schema/dtype, bounded wet concentration, zero dry-cell ice, and forcing present on initially ice-free wet cells.
- [ ] **3. Write a forcing-reset and AD regression, then run RED.** For two native substeps compare with two direct calls that reapply the same prescribed `Qnet/Qsw` each time; expose the different result from omitting the second forcing refresh. On a smooth nonzero-ice small-grid case compare JIT output, JVP, VJP and central finite differences of a forcing perturbation at two step lengths. Require finite nontrivial derivatives; do not differentiate static native constants or binary thresholds.
- [ ] **4. Implement factory/setup/step boundaries.** Lazily load native Veris; initialize `Experiment(Grid(...), geometry=..., settings=..., initial_conditions=...)` only in lifecycle setup after applying runtime dtype. Set equal `deltatDyn/deltatTherm`, closed latitude, and serial native execution. Initialize all diagnostic leaves to native-shaped zeros so the first step does not change the carry schema. Bind static native resources and loop with `lax.fori_loop` over aligned integer substeps, passing freshly prepared prescribed forcing every time. Return new `ice_fraction` and payload; keep I/O outside stepping.
- [ ] **5. Add optional dependency and run GREEN/gates.** The `veris` extra uses `veris @ git+https://github.com/team-ocean/veris.git@a9d6c0ade52fbae4c0d1da694878943d6dbf7e6e`; do not add incompatible Veros metadata to it or alter the existing Veros extras. Install into the working validation environment only after inspecting requirements, then verify PEP 610 provenance and native exports. Actual Veros data construction additionally requires its existing optional dependency. All new native acceptance tests run with Veris installed; do not count skips as acceptance.
- [ ] **6. Update public contracts/docs/progress and commit.** `feat: wrap standalone Veris as a callable component`.

### Task 7: Adapt native output to VerCOR providers and snapshots

**Files:** Create `vercor/setups/_external/veris_output.py`,
`tests/test_veris_output.py`; modify `veris.py`, output/API docs, inventory and progress.

**Interfaces:**
- `veris_output_provider() -> OutputProvider`; its `sample(context: OutputContext) -> OutputFrame` consumes only `context.payload` and component grid.
- `write_veris_snapshot_output(context: SnapshotContext) -> None` uses the shared frame/snapshot writer outside model execution.
- Native variable names remain `Area`, `hIceMean`, `hSnowMean`, `TSurf`, `uIce`, `vIce`, and supported diagnostics; the runtime field alias remains `ice_fraction`.

- [ ] **1. Write failing provider tests.** A tagged non-square payload verifies halo removal, dimension order, native registry units and distinct center/face coordinates. Snapshot/period tests check `Area` agrees with current runtime `ice_fraction`, mean thickness is volume-equivalent, and `forc_salt_surface` honestly retains unknown units. Test empty/subset/unknown `PeriodOutput.variables` through the shared coordinator.
- [ ] **2. Run RED.** Execute `test_veris_output.py -n0 --fast`; verify absent provider/writer and metadata/coordinate assertions fail for the expected reasons.
- [ ] **3. Implement sampling and snapshot conversion.** Reuse `VARIABLES`/`DIAGNOSTICS` metadata and existing shared output writers. Strip two halo cells, transpose arrays/dimensions consistently, use grid tracer coordinates and matching native face edges, and include only dimensions actually sampled. Do not invent a verified salt-flux conversion or ocean-feedback export.
- [ ] **4. Run GREEN and no-I/O acceptance.** Output-disabled runs perform no provider call, host conversion, directory creation or writes, and still work under outer JIT/AD. Inspect actual NetCDF variables, dimensions, end-of-step times and final masks after a short native run. Respect a supplied custom provider/snapshot writer.
- [ ] **5. Update docs/progress, run gates, and commit.** `feat: expose Veris output through shared providers`.

### Task 8: Add the coupled gallery and regression CI coverage

**Files:** Create `vercor/setups/gallery/run_veris_dataveros_era5.py`,
`tests/test_veris_gallery.py`.
Modify `tests/test_setup_gallery_smoke.py`, `tests/test_cli.py`,
`.github/workflows/python-package.yml`, `docs/setup-gallery.rst`,
`docs/api/setups-physics-types.rst`, `docs/researchers/getting-started.rst`,
API/distribution contracts and progress. Extend the existing setup-gallery
documentation owner rather than add a parallel navigation tree.

**Interfaces:**
- Public gallery contract `run_setup(*, loglevel: str, float_type: str) -> None`.
- Gallery-local `_make_coupler(*, loglevel: str, float_type: str, steps: int = 365, veris_config: VerisConfig | None = None) -> Coupler` allows the identical assembled setup to run a two-step acceptance case with explicit native initial conditions.
- Assembly uses `make_veris_verosdata`, ERA5 atmosphere/land, `Clock(start=datetime(2000,1,1), dt_seconds=86400.0, steps=steps, calendar='noleap')`, default one-hour native steps, explicit runtime dtype and opt-in monthly CICE output.
- Meaningful forcing routes are `ATM -> CICE` and `OCN -> CICE`, both `quantity='intensive'`. Land is prescribed and scheduled; there is no fabricated runoff source or CICE feedback route. Do not introduce surface-topology patches that conflict with these explicit incoming modes.

- [ ] **1. Write failing assembly/CLI tests.** Assert exact order `('LND','CICE','OCN','ATM')`, data types for ocean/atmosphere/land, the shared ocean/CICE grid, forcing recipe completeness, no exchange sourced by CICE, finite clock length, standard CLI copying/discovery, and propagated log level/dtype/output policy. Assert the gallery imports only public VerCOR owners.
- [ ] **2. Write a real two-day global acceptance test, then run RED.** Use the actual packaged ERA5 assets, one real Veros global-four-degree snapshot and native coupled Veris; call `_make_coupler(steps=2, veris_config=...)` with explicit native initial conditions. Check actual ice concentration/thickness evolution, bounded `Area`, finite wet fields, dry-cell zeros, correct native substep counts and produced period/final files. Keep this slow test explicit; it must run in final verification, not merely be collected. Separately verify the gallery's default native initialization runs without overrides.
- [ ] **3. Implement the gallery/docs.** Document installation, exact run/copy commands, zero prescribed currents/SSH/runoff, CESM `tref/qref` and their shared similarity assumption, signs/units, native timestep, output variables, and the absence of feedback. Retain native configurable initial conditions; do not create an additional sea-ice initialization algorithm.
- [ ] **4. Add native CI coverage and run GREEN.** Install the immutable Veris extra in the existing stock quality lane so native tests actually execute; retain the separate Veros AD process and existing action pins/artifact ownership. Verify dependency provenance and marker selection in workflow/distribution contract tests. Hosted results are distinct from local results.
- [ ] **5. Update docs/progress, run full gates and commit.** `feat: add Veris with prescribed Veros and ERA5 gallery`.

### Task 9: Verify the full integration, review, commit corrections, and push

**Files:** Review every changed file plus `PROGRESS.md`, `DESIGN.md`,
`DEPENDENCIES.md`, the approved spec, this plan, API/distribution contracts,
native tests, gallery and CI. Corrections belong to their affected modules/tests.

**Interfaces:** The complete public setup/mask API, native transition/output
and the original user requirements are the acceptance boundary; a design/plan
commit or passing mocks alone do not complete the goal.

- [ ] **1. Run comprehensive static/documentation gates.** Black check, strict whole-tree flake8, mypy, compileall, whitespace, and strict Sphinx all pass. Confirm lazy core/config imports in an environment without optional models and complete public manifest/signature/private-inventory alignment.
- [ ] **2. Run meaningful full tests and coverage.** Stock fast/full plus branch coverage pass with real Veris installed, including the slow two-day global case and native equivalence/AD tests. Run the exact pinned Veros AD tests in a separate process with the approved fork, preserving existing pin/provenance. Do not change its process-wide backend in a running stock process. Keep combined coverage at the repository's >=90% requirement.
- [ ] **3. Verify built artifacts and the example.** Build a wheel/sdist into a task-specific temporary directory, inspect the Veris extra and packaged gallery, and run existing supplied-artifact/plugin smoke tests. Execute the new short gallery from the installed artifact with native dependencies; inspect resulting NetCDF files and mask bounds. Do not substitute an import-only smoke for this run.
- [ ] **4. Request an independent whole-branch review.** Cover mask refresh/source weighting, schema safety, physical units/signs, geometry/staggering, CESM reference height, forcing reset, AD and native output. Under native execution this is the single final fresh reviewer; subagent-driven execution additionally retains its per-task reviews. Fix verified findings with failing tests first, rerun affected/full gates after code changes, and record evidence in progress.
- [ ] **5. Audit the original objective and finish commits.** Account for both sequential stages, all seven numbered integration requirements, exact gallery artifact/order, preserved progress edits, all quality gates, real runtime evidence, and separate branch. Mark task checkboxes only when their evidence exists; mark the specification implemented only after this audit. Commit remaining corrections/docs after fresh pre-commit tests.
- [ ] **6. Push the authorized branch and verify GitHub's SHA.** Use `gh repo view --json nameWithOwner,url` to confirm `nutrik/vercor`, inspect existing CLI authentication without printing secrets, then `git push -u origin feat/veris-seaice`. Verify `gh api repos/nutrik/vercor/git/ref/heads/feat/veris-seaice --jq .object.sha` equals `git rev-parse HEAD`, and verify a clean working tree. No merge/release is part of this task.
- [ ] **7. Report the implementation result.** Include branch/head, actual push verification, concise native/stock/AD/static results and material limitations. Mark the goal complete only after current evidence proves the entire requested integration and remote handoff.

## Plan self-review

- The approved two-stage scope maps to Tasks 1-3 and 4-9; the user-requested CESM `ATemp` revision is pinned in Tasks 5 and 8.
- Mask declaration, arithmetic, refresh/output, source-data geometry, forcing, native lifecycle, output and gallery have separate focused test cycles and explicit interfaces.
- Zero-concentration growth, inactive NaNs/reactivation, replaced-state cache refresh, substep forcing reset, and reference-height/staggering risks each have named tests above.
- The plan preserves the root API, existing default exchanges, native numerical equations and dependency lanes; new public signatures are updated with their contract tests.
- The user approved this plan on 2026-10-16; execution uses the recommended native method with independent final review. Implementation is in progress.
