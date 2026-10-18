# Veris, Veros and JCM Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Deliver the verified, committed and pushed `run_veris_veros_jcm.py` gallery with correct surface ownership, evolving masks and LND → CICE → OCN → ATM execution.

**Architecture:** Reuse both paired setup factories and the existing explicit exchange operations. Veris exports native ice properties, Veros exports its current open-water SST contribution, and a narrow JCM forcing adapter composes the separately remapped surface contributions. Native Veris owns its albedo law; JCM keeps its single effective marine surface channel.

**Tech Stack:** Python, JAX, jaxtyping, standalone Veris, Veros stock and its pinned autodiff fork, JCM 2.0.1, pytest, Conda, Git/GitHub CLI, shared NetCDF output.

**Spec:** `docs/superpowers/specs/2026-10-18-veris-veros-jcm-design.md`, approved 2026-10-18.

## Global Constraints

- VerCOR work stays on `feat/veris-seaice`; preserve existing edits and stage only task-owned paths.
- Native Veris work uses existing `feat/vercor-scientific-validation`, starting from its latest fetched remote tip immediately before changes. Last observed tip: `6c205a356dd1b25c26430f899cb8998523ab3447`; this is evidence, not a substitute for fetching.
- Preserve the user's separate editable Veris checkout. Use `/private/tmp/vercor-jcm-veris` for an isolated clone; inspect an existing directory before reuse. The user's branch instruction supersedes native AGENTS.md's older `jax-only` branch instruction.
- Keep order `LND`, `CICE`, `OCN`, `ATM`; no direct ATM → OCN forcing in this gallery.
- Keep binary ocean support static; form source products before remapping, with no second concentration factor.
- Use `mod_radcon.albsea` for open water and native Veris snow/ice albedo for covered water.
- Keep all evolving model state in existing immutable payloads; numerical mode/configuration stays static, physical parameters traced.
- New configuration modes/flags are keyword-only and validated; existing positional constructor arguments keep their order.
- Preserve the public CESM bulk return tuple and existing default forcing paths except the demonstrated JCM flux-channel summation defect.
- Apply dtype policy before native construction. Use the isolated Python 3.12 Conda environment named `scipy`; never use an OpenMP duplicate-library bypass. Stock and autodiff Veros run in separate processes.
- Use `/private/tmp` for pytest basetemps. Keep success/failure stdout concise; full evidence belongs under `test_logs/veris-veros-jcm/`.
- Full correctness suites must pass before commits. Update PROGRESS.md after each task while retaining its 180-line bound; update native CHANGELOG.md for native work.
- Every task that changes public declarations updates its owned entries in `tests/contracts/vercor-0.4.0-public-signatures.json`, `tests/test_v0_4_public_api.py` and `docs/api-architecture-review.md`; private-module additions also update the existing source inventory and DEPENDENCIES.md. Preserve unrelated entries and unique assertions.
- Push native changes first, then pin VerCOR to the verified reachable SHA. Use GitHub CLI authentication/ref checks and ordinary fast-forward pushes; never force-push, merge, tag or publish a release.

## Review Focus

1. Disabled JCM spinup must still provide physically evaluated initial radiation, without advancing or corrupting native carry (Task 6).
2. Tiny coastal coverage at the existing mask threshold, disappearing ice and inactive NaNs must not introduce negative coverage or non-finite derivatives (Tasks 5 and 7).
3. A moving ocean, zero relative wind and a model level at exactly 10 m must preserve correct reference-wind direction and frame (Task 3).
4. Native noleap epoch days must match VerCOR's date across February and year boundaries; Gregorian date conversion is not interchangeable (Task 6).
5. Time-varying native forcing selection, plain native bootstrap and independent repeated runs must retain the albedo extension without hidden mutable arrays (Tasks 5 and 6).

## File responsibilities and dependency order

| Unit | Files | Responsibility |
| --- | --- | --- |
| Native surface physics | Native `veris/numerics/surface_properties.py`, `surface_temperature.py`, `physics/thermodynamics/growth.py`, `veris/__init__.py` | Shared literal albedo/category geometry and public instantaneous diagnosis |
| Ice adapter | `vercor/setups/_external/veris.py`, `veris_runtime.py`, `veris_output.py`, `config.py` | Seed/update native surface exports; select forcing source |
| Shared bulk | `vercor/fluxes/bulk_formula_cesm.py`, `setups/_surface_fluxes.py`, `_external/veris_forcing.py` | One CESM solve with actual reference-height wind diagnostics |
| Ocean adapter | `_external/veros_gcm.py`, `veros_gcm_state.py`, `veros_runtime.py` | Optional concentration import and post-ocean open-water SST contribution |
| Pure JCM surface assembly | `_external/jax_gcm_fields.py` | Covered contributions → intensive native forcing and inspectable surface diagnostics |
| Native JCM hook | New `_external/jax_gcm_surface.py` | Frozen forcing extension and native composable forcing-term override |
| JCM runtime/initialization | `_external/jax_gcm.py`, `jax_gcm_state.py`, `jax_gcm_runtime.py`, `setups/_jcm.py`, `_data/jcm_land.py` | Coupled contracts, fractional terrain, epoch, first-step diagnostics and payload stepping |
| Gallery/output/contracts | New gallery and tests; `recipes.py`, `fields.py`, native output providers, docs/API artifacts | Public composition, metadata and end-to-end acceptance |

Tasks 2–7 depend on Task 1's public native interface; Task 7 joins all adapter paths. Execute tasks sequentially because configuration, recipes and API artifacts overlap. Update the existing numbered DEPENDENCIES.md layers rather than invent a second graph.

## Execution preparation and commit gates

Before Task 1, read the fetched native AGENTS.md, DESIGN.md and CHANGELOG.md. Clone with GitHub CLI on the requested branch, fetch again, verify clean status and record `HEAD` plus the advertised remote SHA. Review any changes since the last observed tip before capturing numerical baselines.

Activate the existing isolated Conda environment with `source /Users/spj483/miniforge3/etc/profile.d/conda.sh` and `conda activate /private/tmp/vercor-time-ci/envs/scipy`; verify Python 3.12. For native work, create/activate the native checkout's `.venv` from that interpreter with system-site-packages, following native environment rules. Do not initialize a new project or modify the user's normal environments. Install only missing tooling and the exact dependency revisions needed for the approved work.

**VerCOR commit gate G:** focused tests pass; Black, strict flake8, mypy and whitespace checks pass; then `python -m pytest tests/ -q --fast -m "not veros_autodiff" --tb=short` and the full stock command without `--fast` pass. Run affected autodiff tests in the separate exact-fork process when the task changes their path. Use the configured n4/loadscope policy and `/private/tmp` basetemps. Do not commit from a known failing environment; replace its stale Veris installation with the verified revision first.

**Native gate N:** `ruff format --check .`, `ruff check .`, `ty check .`, and the full native correctness suite pass in the native environment. Use native fast selection while iterating, without a redundant fast run immediately adjacent to the full suite. Report hardware-dependent skips accurately.

Each task's commit step requires its applicable gate, named-path staging and a PROGRESS.md/CHANGELOG.md update. A failing gate returns to that task's focused regression; it does not justify skipping or weakening assertions.

---

### Task 1: Extract and expose native Veris albedo without numerical changes

**Files:** native `veris/numerics/surface_properties.py` (new); modify `veris/numerics/surface_temperature.py`, `veris/physics/thermodynamics/growth.py`, `veris/__init__.py`, DESIGN.md, CHANGELOG.md. Tests: new `tests/numerics/test_surface_properties.py`, existing `tests/numerics/test_solve4temp.py`, `tests/physics/thermodynamics/test_growth.py`, `tests/model/test_experiment_exports.py`; add compact baseline data under `tests/reference/surface_properties_baseline.npz`.

**Interfaces:** public native `diagnose_surface_albedo(state: State, configuration: Configuration, physical_constants: PhysicalConstants) -> Float[Array, "x y"]`, halo-inclusive and per ice area. Private `_surface_albedo(temperature: Float[Array, "..."], snow_thickness: Float[Array, "..."], coriolis: Float[Array, "..."], physical_constants: PhysicalConstants) -> Float[Array, "..."]`, `_prepare_thickness_geometry(state: State, physical_constants: PhysicalConstants) -> _ThicknessGeometry` and `_category_factors(configuration: Configuration) -> Float[Array, "category"]` share the exact native operations. `_ThicknessGeometry` is a private NamedTuple of native arrays: ice_mean, snow_mean, area, reciprocal_area, reciprocal_ice_thickness, ice_thickness and snow_thickness. Growth consumes these results for its later budgets as well as category setup; do not duplicate its regularization outside the helper. Only the diagnosis is exported from the root.

- [x] **Write RED tests and capture the unchanged solver baseline.** Record all five solve4temp outputs and representative JVPs from the fetched starting SHA, with exact inputs/provenance. Assert north/south dry/wet constants, equality at the wet threshold, snow thickness 0/0.05/0.3 m, `nITC=1/3/5`, zero ice and inactive NaNs. Construct actual category snow thickness 0.1 m with `nITC=3`; default dry albedo must be `(0.77 + 0.81 + 0.84)/3`, not 0.81. Include Area_reg=0 and ice-free wet-cell growth.
  Name the key regressions `test_category_albedo_averages_native_law` with `np.testing.assert_allclose(albedo, (0.77 + 0.81 + 0.84)/3)`, and `test_surface_extraction_preserves_solver_outputs_and_jvp` with `np.testing.assert_allclose(actual, recorded, rtol=0, atol=0)` for unchanged float64 operation order.
- [x] **Run RED:** `python -m pytest tests/numerics/test_surface_properties.py -q --tb=short`. Expect missing diagnosis/helper assertions, not fixture or dependency errors. Existing baseline cases must pass before extraction.
- [x] **Implement the shared native helpers and diagnosis.** Preserve the original operation order, regularized concentration, wet-cell minimum-thickness clamp, category factors, hemispheric/wet selection and solver call location before Newton. Diagnose current stored TSurf with current category snow thickness, average categories, and return zero where ice/wet support is absent. Do not change pressure physics, constants, Newton iterations or growth accounting.
- [x] **Verify GREEN:** run the new tests plus solve4temp, growth, thermodynamic AD and precision tests; compare baseline outputs/JVPs with the pre-extraction values. Require finite JIT/JVP/VJP and finite-difference agreement away from thresholds in both supported precisions.
- [x] **Run N, commit and push native work.** Commit only the named native paths as `refactor: share native sea-ice surface albedo diagnosis`. Re-fetch the branch before ordinary push; integrate remote advances and repeat affected checks. Verify `gh api repos/team-ocean/veris/git/ref/heads/feat/vercor-scientific-validation --jq .object.sha` equals the local commit. Record that exact SHA for Task 2.

### Task 2: Pin the native revision and export current Veris surface properties

**Files:** modify `pyproject.toml`, `.github/workflows/python-package.yml`, `vercor/setups/_external/veris.py`, `veris_runtime.py`, `veris_output.py`, `vercor/recipes.py`, `vercor/fields.py`; tests `tests/test_veris_component.py`, `test_veris_rollouts.py`, `test_veris_output.py`, `test_public_api_contracts.py`, `test_v0_4_public_api.py`; update affected API inventories and dependency documentation.

**Interfaces:** CICE gains `ice_surface_temperature` (K) and `ice_surface_albedo` (dimensionless per ice area). Private `veris_surface_state_fields(payload: VerisPayload, resources: VerisRuntimeResources) -> dict[str, RuntimeArray]` provides the same setup/step extraction. Public recipes `VERIS_TO_JCM_SURFACE_FIELDS=("ice_surface_temperature", "ice_surface_albedo")` and `VERIS_TO_JCM_COVERAGE_FIELDS=("ice_fraction",)` carry separate quantities.

- [x] **Write RED tests.** Compare both setup and post-step exports with physical-interior native TSurf/albedo; assert `[2:-2,2:-2].T`, non-square shape, runtime dtype, native parameter sensitivity, zero support and immutable payload ownership. An import-blocking test must keep root/setup imports lazy. Assert all live native dependency/CI pins agree with Task 1's verified SHA.
  In `test_veris_surface_exports_follow_payload`, use `np.testing.assert_allclose(fields["ice_surface_temperature"], payload.state.TSurf[2:-2,2:-2].T)` and compare the albedo field to the native public diagnosis, for both setup and one real transition.
- [x] **Run RED:** `python -m pytest tests/test_veris_component.py tests/test_veris_rollouts.py tests/test_veris_output.py -q --fast --tb=short`; expect absent export/capability/pin assertions.
- [x] **Implement extraction and pin atomically.** Capability-check the public native diagnosis, seed and update both exports through the shared helper, preserve interval-mean ocean-flux fields, and add provider variables/metadata. Replace production pins and corresponding explicit assertions with the exact verified SHA. Preserve historical scientific reports and their old provenance; do not rewrite them as new validation.
- [x] **Install the reachable exact revision in the isolated scipy environment and verify GREEN.** Confirm native VCS direct_url metadata, run the focused files without fast selection, lazy/public contracts and existing native/direct-advance equivalence.
- [x] **Run G and commit** as `feat: export native Veris surface temperature and albedo` with the atomic pin, metadata, tests and contract updates.

### Task 3: Supply model-level atmospheric forcing through one CESM solve

**Files:** modify `vercor/fluxes/bulk_formula_cesm.py`, `vercor/setups/_surface_fluxes.py`, `_external/veris_forcing.py`, `_external/veris.py`, `_external/veris_runtime.py`, `setups/config.py`, `recipes.py`. Tests: new `tests/test_reference_winds.py`; extend `tests/test_shared_surface_fluxes.py`, `test_veris_forcing.py`, `test_veris_component.py`; affected signature contracts.

**Interfaces:** `VerisConfig.atmosphere_forcing: Literal["observed_10m", "model_level"]="observed_10m"`. Private `_OceanBulkResult` contains existing `fluxes: tuple[Array,...]` plus actual `u_velocity_10m`/`v_velocity_10m`; `_compute_ocean_surface_fluxes_with_winds` accepts the same arguments as `compute_ocean_surface_fluxes`. The public function still returns its existing tuple. Shared `SurfaceFluxes` gains those two wind diagnostics. `prepare_veris_forcing_pack` and `prepare_veris_forcing` accept the static mode keyword, default observed.

Private `_reference_height_wind(constants: PhysicalConstants, ubot: Array, vbot: Array, us: Array, vs: Array, zbot: Array, zref: Array, rd: Array, hol: Array, psimh: Array, stable: Array, mask: Array) -> tuple[Array, Array]` consumes the retained final profile; `rd` is the square root of the actual drag coefficient. It is called inside the same bulk solve, with zref=10 m.

`JCM_ATMOSPHERE_TO_VERIS_FORCING_FIELDS` contains the model wind vector, specific_humidity, model_level_height, density, potential_temperature, temperature, surface_pressure, downward_shortwave_radiation_flux, downward_longwave_radiation_flux, rain_rate, snow_rate and open_water_albedo. It excludes observed 10 m winds and mixed absorbed shortwave.

- [x] **Write RED tests.** At model height 10 m, diagnosed vectors equal input air wind. Adding `(3,-2)` m/s to both air and water adds exactly that vector to diagnosed air wind. Check independent analytic profiles: for zbot=40 m, zref=10 m, rd=.04, kappa=.4, hol=.2 and stable=1, the relative-wind factor is `1-.1*(log(4)+.75)`. Include neutral and unstable analytic cases, non-square masks, zero relative wind, inactive NaNs and JVP/VJP/FD. At SW_down=200 and alpha_water=0.1, native SWdown=200 and open-water absorbed SW=180, with native Qsw=-180. Model-level mode must not require observed u/v10m; observed mode must still use its deliberately different input vectors.
  Name `test_reference_wind_identity_and_frame` with `np.testing.assert_allclose(reference_at_10m, air_wind)` and `np.testing.assert_allclose(shifted-reference, [3,-2])`; name the independent profile test `test_reference_wind_matches_analytic_stability_profiles`.
- [x] **Run RED:** `python -m pytest tests/test_reference_winds.py tests/test_veris_forcing.py -q --fast --tb=short`; expect missing mode/wind diagnostics.
- [x] **Implement one retained bulk solution.** Retain the momentum stability correction and scales used in the final existing iteration. Apply the same profile at 10 m, diagnose relative vectors, then add ocean current back. Extend the private result while keeping all original public tuple values unchanged. Pack model-level mode using diagnosed winds and water-specific absorbed SW; validate static mode and albedo range. Thread the mode through frozen resources and factory input declarations.
- [x] **Verify GREEN and compatibility.** Run new/focused files in full, existing CESM independent literals (including qnet=-433.5958440702862 for its reference case), Veros flux tests and affected exact-fork AD. Assert no second bulk iteration/solve and finite derivatives at calm conditions.
- [x] **Run G and commit** as `feat: diagnose reference winds for model-level sea-ice forcing` with configuration/API documentation.

### Task 4: Export the post-ocean open-water SST contribution

**Files:** modify `vercor/setups/config.py`, `_external/veros_gcm.py`, `veros_gcm_state.py`, `veros_runtime.py`, `recipes.py`. Tests: extend `tests/test_veros_ice_feedback.py`, `test_veris_prognostic_gallery.py`, `_veros_surface_support.py`; add `tests/test_ocean_surface_contributions.py`; update signature contracts.

**Interfaces:** `VerosConfig.export_open_water_surface: bool=False`, allowed only for coupled surface forcing. Enabled OCN imports `ice_fraction` and exports `open_water_temperature_contribution` in addition to its existing fields. Private `open_water_temperature_contribution(sst: RuntimeArray, ice_fraction: RuntimeArray, wet_mask: RuntimeArray) -> Array` is pure/JIT-compatible. Recipe `OCEAN_TO_JCM_OPEN_WATER_FIELDS=("open_water_temperature_contribution",)` uses binary source-flux support at assembly.

- [x] **Write RED tests.** A native ocean step changes SST; the contribution must use that updated SST and the latest imported concentration. A two-cell conservative average with A=(0.2,0.8), SST=(280,300) gives 142 K, not the 145 K product-of-means result. Test A=0/1, masked NaNs, invalid active A, host/JAX parity, setup defaults and default-disabled compatibility. Full ocean SST stays available under ice and wet masks stay unchanged.
  In `test_post_ocean_sst_is_weighted_before_remap`, assert `np.testing.assert_allclose(received_contribution, 142.0)` and compare the retained full SST with `(280,300)` after the tagged native step; in `test_open_water_export_does_not_mask_underice_ocean`, assert the original wet mask is unchanged at A=1.
- [x] **Run RED:** `python -m pytest tests/test_ocean_surface_contributions.py tests/test_veros_ice_feedback.py -q --fast --tb=short`; expect missing flag/field assertions.
- [x] **Implement the opt-in contract.** Thread the static flag through the existing ocean owner, setup/default fields and factory; after native surface extraction compute `(1-A)*new_SST` with inactive operands neutralized. Validate active concentration rather than silently clipping it. Native scalar/stress feedback and restoring remain unchanged.
- [x] **Verify GREEN:** run focused full tests and affected prognostic exact-fork AD separately; prove original output contracts with the flag off and no model reinitialization.
- [x] **Run G and commit** as `feat: export current open-water SST contributions` with contract/documentation updates.

### Task 5: Build the pure JCM surface compositor and native albedo hook

**Files:** modify `vercor/setups/_external/jax_gcm_fields.py`; create `vercor/setups/_external/jax_gcm_surface.py`. Tests: new `tests/test_jcm_surface_fields.py`, `tests/test_jcm_surface_forcing.py`; extend native-hook compatibility tests in `tests/test_external_components_coverage.py` only where shared fixtures need it.

**Interfaces:** registered frozen `JCMCoupledSurface` has land_surface_temperature, sea_surface_temperature, soil_moisture, marine_surface_albedo, marine_ice_fraction and total_surface_temperature arrays. `prepare_coupled_surface_forcing(fields: Mapping[str, RuntimeArray], land_fraction: RuntimeArray, open_water_albedo: RuntimeArray, *, dtype: DTypePolicy) -> JCMCoupledSurface` consumes the exact Task 2/4 names; its incoming `ice_fraction` is atmosphere-cell ice area.

Frozen registered `CoupledJCMForcing(base: ForcingData, marine_surface_albedo: Array)` delegates native attributes; `.copy(*, marine_surface_albedo: Array|None=None, **native_updates: Any)` and `.select(date: Any, calendar: str=DEFAULT_CALENDAR)` return the wrapper type. `CoupledSpeedyForcing` subclasses native SpeedyForcing and overrides `__call__`, preserving the native constructor contract.

- [x] **Write RED surface tests.** For f_o=.75, f_i=.3, T_water=290, T_ice=260, alpha_water=.1, alpha_ice=.7, assert T_marine=278 and alpha_marine=.34. With T_land=300, total T=283.5; soil moisture remains the distinct land value. Cover A=0/1, all-land/all-water, .001 coverage threshold endpoints, absent-surface NaNs and JIT/JVP/VJP/FD. Assert no second A weighting and no overwrite of received f_i with ocean-relative A.
  In `test_fractional_coast_composes_independent_surfaces`, assert `np.testing.assert_allclose(surface.sea_surface_temperature, 278.0)`, `np.testing.assert_allclose(surface.marine_surface_albedo, .34)` and `np.testing.assert_allclose(surface.total_surface_temperature, 283.5)`.
- [x] **Write RED hook tests and run:** `python -m pytest tests/test_jcm_surface_fields.py tests/test_jcm_surface_forcing.py -q --fast --tb=short`. Check wrapper copy/select under native time series and calendars, tree dtype normalization and no mutation. Plain native bootstrap must produce unchanged native diagnostics/tendencies. Wrapped forcing must assign the precomposed alpha directly to alb_s and combine with native land alpha through fmask once.
- [x] **Implement the compositor and hook.** Use the approved source-contribution equations, existing mask-threshold convention and safe masked divisions. Fill absent temperatures with existing REFERENCE_SURFACE_TEMPERATURE and soil moisture with zero before nonlinear native use. Keep imported contribution names distinct from produced marine diagnostics. Call native forcing first; replace only alb_s/albsfc functionally. Select plain/native behavior by static wrapper type, never a traced-value branch.
- [x] **Verify GREEN:** run both new files in full, native bootstrap/forcing compatibility, inactive-NaN derivatives and repeat-copy/select tests. Neither model objects nor physics terms may acquire an evolving ice-array attribute.
- [x] **Run G and commit** as `feat: compose masked JCM marine surface forcing` with dependency-layer documentation for the new private module.

### Task 6: Wire JCM coupled fields, fractional terrain and physical initialization

**Files:** modify `vercor/setups/config.py`, `_jcm.py`, `_data/jcm_land.py`, `_external/jax_gcm.py`, `jax_gcm_state.py`, `jax_gcm_runtime.py`, `jax_gcm_fields.py`, `recipes.py`. Tests: new `tests/test_jcm_surface_runtime.py`, `tests/test_jcm_initial_forcing.py`; extend `tests/test_setup_lifecycle_helpers.py`, `test_external_components_coverage.py`, `test_jcm_precipitation.py`, public signatures/inventories.

**Interfaces:** `JAXGCMConfig.seaice_coupling: bool=False`. Coupled ATM inputs are land_surface_temperature, soil_moisture, open_water_temperature_contribution, ice_surface_temperature, ice_surface_albedo and ice_fraction. Coupled output diagnostics include sea_surface_temperature, total_surface_temperature, marine_surface_albedo and marine_ice_fraction. All atmosphere paths gain raw downward_shortwave_radiation_flux, surface_pressure, open_water_albedo, land_sensible_heat_flux and land_latent_heat_flux exports with declared defaults/shapes.

`map_jcm_output_fields` adds required keyword inputs `downward_shortwave_radiation_flux` and `open_water_albedo`; update its named call sites/fixtures while preserving precipitation/height assertions. Private `jax_gcm_noleap_epoch(start: datetime|ModelDateTime, effective_spinup: timedelta) -> Any` binds native start date; `diagnose_jax_gcm_initial_fields(state, payload, constants, dtype) -> tuple[Mapping[str, RuntimeArray], JAXGCMRuntimePayload]` evaluates native initial physics without advancing.

- [x] **Write RED tests.** Assert fractional terrain instead of binary coastal support; distinct land/marine temperatures and land moisture reach applied forcing. For native shf=(10,20,17.5), total downward SHF=-17.5 and land SHF=-10, never their sum. For evaporation=(1,2,1.75) g/m2/s, verify each selected channel's kg conversion and latent heat sign. Assert raw rsds, pressure=1e5*normalized_ps, and configured open-water alpha.
  Name `test_native_flux_channels_are_selected_once` with `np.testing.assert_allclose(mapped["sensible_heat_flux"], -17.5)` and `np.testing.assert_allclose(mapped["land_sensible_heat_flux"], -10.0)`; name `test_coupled_land_moisture_and_fraction_reach_native_forcing` and compare the captured native arrays to the distinct authored land values.
- [x] **Write initialization/ownership RED tests and run:** `python -m pytest tests/test_jcm_surface_runtime.py tests/test_jcm_initial_forcing.py tests/test_setup_lifecycle_helpers.py -q --fast --tb=short`. Disabled spinup must evaluate real diagnostics without changing dycore sim_time or cross-step physics carry. With two-day spinup ending 2000-01-03, native start is noleap day 10950 since 1970; coupling epoch is day 10952. Assert native model_year=2000 and tyear=2/365, February/year-boundary continuity, exact spinup alignment, conflicting-epoch rejection and independent runs from supplied payloads.
  In `test_initial_forcing_does_not_advance_native_carry`, compare every dycore/cross-step carry leaf before/after diagnosis and assert the exported radiation matches a direct native diagnostic evaluation. In `test_native_noleap_epoch_matches_coupler`, use `assert int(model_year) == 2000` and `np.testing.assert_allclose(tyear, 2/365)` at the coupling epoch.
- [x] **Implement coupled lifecycle and stepping.** Install Task 5's term using native `.replace("forcing", ...)` before Model construction. In coupled setup bind calendar 365_day and noleap ordinal epoch, using DateTime365 arithmetic and native epoch+Timedelta rather than Gregorian string parsing. Effective spinup is zero when disabled; reject nonintegral enabled spinup/coupling alignment. Reuse the mask helper for fractional native terrain. Evaluate native diagnostics on actual initialized dynamics/selected forcing/time and export them without advancing carry. Store the evaluated snapshot in a new payload's jcm_state.physics for coherent provider sampling; preserve jcm_state.physics_carry and dycore_state exactly. In coupled steps apply Task 5's fields/wrapper from payload; keep legacy temperature path when disabled.
- [x] **Implement selected channels and named land inputs.** Select native channel two for total flux, zero for land-specific flux; apply signs/units once. Seed/validate every declared field, forward imported soil moisture in coupled mode, and let JCM land accept both new specific and existing legacy flux inputs. Add `JCM_ATMOSPHERE_TO_LAND_FLUX_FIELDS=("land_latent_heat_flux", "land_sensible_heat_flux")` without changing the legacy recipe.
- [x] **Verify GREEN:** full focused tests plus JCM output-frequency, precipitation, runtime ownership and legacy gallery/adapter checks. Update old total-flux expected values explicitly for the demonstrated bug, retaining every independent assertion. Run affected exact-fork JCM/Veros acceptance separately.
- [x] **Run G and commit** as `feat: apply coupled JCM surfaces and initialize physical forcing` with live API/docs updates.

### Task 7: Assemble and exercise the four-component gallery and shared output

**Files:** create `vercor/setups/gallery/run_veris_veros_jcm.py`, `tests/test_veris_veros_jcm_gallery.py`, `tests/test_veris_veros_jcm_rollouts.py`, `tests/test_veris_veros_jcm_autodiff.py`; modify `vercor/setups/_external/jax_gcm_output.py`, existing Veris/Veros output providers where new field sampling is needed, `tests/test_setup_gallery_smoke.py`, `test_distribution_boundaries.py`, `test_veris_output.py`, `.github/workflows/python-package.yml`, `docs/setup-gallery.rst`, `docs/api/setups-physics-types.rst`, DESIGN.md, DEPENDENCIES.md and API inventories.

**Interfaces:** gallery `build_coupler(*, pair: VerisVerosSetup|None=None, jcm_inputs: JCMInputs|None=None, clock: Clock|None=None, veris_config: VerisConfig|None=None, ocean_config: VerosConfig|None=None, jcm_config: JCMLandAtmosphereConfig|None=None, log_level: int|str="INFO", dtype: DTypePolicy|None=None) -> Coupler`; standard `run_setup(*, loglevel: str, float_type: str) -> None`. Default clock is 2000-01-03, dt=86400 s, 365 steps, noleap. Default JCM spinup is two days; enable Veris model_level forcing, ocean open-water export and JCM seaice_coupling. Reject incompatible injected components/configuration/calendar before run.

- [x] **Write gallery RED tests.** Assert one ocean initialization, shared native grid, dtype enabled before construction, lazy import with all optional models blocked, standard CLI discovery/copying, exact order and this route table:

  Name `test_gallery_routes_preserve_surface_ownership` and use `assert tuple(coupler.run_order) == ("LND", "CICE", "OCN", "ATM")`, then compare each route's actual fields, quantity, source_coverage and remapper with the table. Name `test_first_ice_step_uses_initialized_jcm` for the direct-native versus captured forcing comparison.

| Route | Fields | Quantity/support/remapper |
| --- | --- | --- |
| ATM → CICE | JCM_ATMOSPHERE_TO_VERIS_FORCING_FIELDS | intensive/component/bilinear |
| OCN → CICE | OCEAN_TO_VERIS_FORCING_FIELDS | intensive/binary/bilinear identity |
| CICE → OCN scalars | existing total scalar flux recipe | source_flux/binary/conservative |
| CICE → OCN stress | existing total stress vector recipe | source_flux/binary/bilinear identity |
| CICE → OCN coverage | ice_fraction | intensive/binary/bilinear identity |
| OCN → ATM | OCEAN_TO_JCM_OPEN_WATER_FIELDS | source_flux/binary/conservative |
| CICE → ATM surface | VERIS_TO_JCM_SURFACE_FIELDS | source_flux/component/conservative |
| CICE → ATM coverage | VERIS_TO_JCM_COVERAGE_FIELDS | source_flux/binary/conservative |
| LND → ATM | JCM_LAND_TO_ATMOSPHERE_FIELDS | intensive/component/bilinear identity |
| ATM → LND | JCM_ATMOSPHERE_TO_LAND_FLUX_FIELDS | intensive/component/bilinear identity |

- [x] **Write rollout/output RED tests and run:** `python -m pytest tests/test_veris_veros_jcm_gallery.py tests/test_veris_veros_jcm_rollouts.py -q --fast --tb=short`. Use actual small native ocean/ice fixtures and native low-resolution JCM inputs where supported. Tagged unequal surfaces must prove current CICE A and post-OCN SST reach ATM, zero/partial/full ice and threshold coasts are handled, first-CICE air comes from native diagnostics, land fluxes remain per land area and OCN receives one total boundary. Output must include matching evolving marine alpha/concentration/temperature with correct coordinates/units.
- [x] **Implement the public assembly and output fields.** Reuse paired factories; validate enabled modes and exact shared geometry of injected pair. Reject simultaneous injected pair and native pair configuration overrides, and reject explicitly supplied incompatible forcing/export/coupling modes rather than ignore them. Give every repeated component pair a distinct route_id. Use existing SurfaceMaskPolicy for static surface topology; explicit preweighted source-flux routes must not receive another f_o multiplier. Add canonical marine diagnostics to the existing JCM provider with lat/lon dimensions while retaining native outputs and shared coordinator ownership. Add no private gallery imports or new output lifecycle.
- [x] **Verify real integration and gradients.** Run a real global four-component two-day smoke in a separate process with shared output inspected. Run a small smooth case through JIT/JVP/VJP and finite differences, including a nonzero derivative from ice albedo/concentration through JCM and back to a later sea-ice/ocean interval. Mark real differentiable ocean checks veros_autodiff and run only in the exact-fork process. Keep native physics defaults; label reduced test numerics explicitly and do not infer climate skill from the smoke.
- [x] **Verify installed use.** Build wheel/sdist, run the copied setup from an installed wheel outside the checkout, verify optional lazy imports, and run affected distribution/public contracts. Add the new AD file to the existing pinned-fork CI selection. Document effective marine-temperature approximation, per-area meanings, initial lag/spinup and source ownership.
- [x] **Run G and commit** as `feat: add coupled Veris Veros JCM gallery` with its tests, provider, CI and documentation updates.

### Task 8: Complete independent review, final quality gates and verified publication

**Files:** task report `test_logs/veris-veros-jcm/` (detailed local evidence); PROGRESS.md, approved spec/plan and necessary final documentation. Any production finding returns to its owning task with a failing regression first.

**Interfaces:** a requirement/evidence ledger covers specification Sections 1–7 and acceptance items 1–9, exact native/VerCOR SHAs, commands, test counts, output paths and remaining scientific limitations. No requirement is satisfied merely by an indirect manifest/test-name check.

- [x] **Review the whole change.** Use the fresh reviewer required by the selected execution method. Review source ownership, source weighting, temporal order, native equivalence, flux signs/units, actual reference winds, inactive operands, gradients, lazy imports, config ownership and output sampling. Reproduce and fix substantive findings through their owning regression; do not add mirror tests for documentation-only changes.
- [x] **Run final static checks:** `python -m black --check vercor tests`, `python -m flake8 . --count --max-line-length=120 --statistics`, `python -m mypy vercor tests`, `python -m compileall -q vercor tests`, `git diff --check`. Require successful exits; flake8 must report zero.
- [x] **Run final fast/full coverage gates.** Fast stock passes. Run the full stock suite once with `--cov=vercor --cov-branch --cov-report= --cov-fail-under=0`; preserve its data before a separate exact-fork process runs `tests/test_veros_autodiff_rollouts.py`, `tests/test_veris_prognostic_autodiff.py` and the new JCM AD file with coverage. Use Veros fork `7a8c964cf00b5aa0713c995edd760643a431b3c9` and the exact new native pin. Combine both data files and require the existing 90% coverage gate. Do not mix native backends in one process.
- [x] **Verify documentation/artifacts and acceptance ledger.** Build strict Sphinx, wheel and sdist, run Twine/distribution boundary checks and the installed copied-gallery smoke. Check actual output arrays/masks/dates and native equivalence/gradient evidence against every spec acceptance item. Record scientific limitations without weakening the approved functional requirements.
- [x] **Commit verified handoff records.** Update PROGRESS.md and spec/plan task checkboxes to reflect actual evidence, retaining historical reports and unrelated edits. Commit only named final record paths after the full gates, as `docs: record coupled surface validation and handoff`.
- [x] **Push and verify the requested branch.** Recheck clean task state, branch, remote and local HEAD. Use `gh auth status` and GitHub CLI's credential helper for `git push origin HEAD:refs/heads/feat/veris-seaice`, without persisting Git configuration changes. Check `gh api repos/nutrik/vercor/git/ref/heads/feat/veris-seaice --jq .object.sha` against local HEAD and recheck native branch/pin reachability. If either remote advanced, integrate without force and rerun affected gates before push. Mark the goal complete only when code, evidence, commits and both remote refs prove the requested result.

## Plan self-review and execution handoff

Coverage: native extraction → Task 1; native exports/pin → Task 2; air forcing/winds → Task 3; current ocean contribution → Task 4; equations/wrapper/hook → Task 5; JCM masks/fluxes/initialization → Task 6; gallery/output/end-to-end behavior → Task 7; quality/review/publication → Task 8. Each Review Focus item has an owning test step. Exact field names match across producers, consumers and the route table; no exchange renaming is assumed.

Approved by the user on 2026-10-18 with **Native** inline execution and one fresh whole-branch reviewer. Tasks 1–8 are implemented and verified; the final evidence is recorded in `docs/veris-veros-jcm-validation.rst` and `PROGRESS.md`.
