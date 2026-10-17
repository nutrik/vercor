# VerCOR Veris Calibration and Ocean Feedback Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Integrate corrected native Veris with traced calibration parameters, conservative ocean feedback, and measured extended validation on `feat/veris-seaice`.

**Architecture:** Consume the tested native interfaces from the [native-science plan](2026-10-17-veris-native-science.md). Reuse VerCOR's opaque immutable state, component lifecycle, exchange machinery, shared surface bulk calculation and output coordinator; expose only the narrow setup/route additions approved in the design.

**Tech Stack:** Python 3.12 conda `scipy`, JAX, SciPy, stock Veros and the separately installed exact differentiable fork, pytest, Black, flake8, mypy, Sphinx, Git/GitHub CLI.

**Spec:** [Approved scientific design](../specs/2026-10-17-veris-scientific-validation-design.md), sections 3–5 and the VerCOR/native boundary; written-spec planning approved 2026-10-17.

## Global Constraints

- VerCOR work stays on `feat/veris-seaice`; preserve unrelated edits. Native fixes use the separate dedicated branch, with no main/jax-only merge, release, tag or publication.
- Native physical parameters are dynamic; numerical shape/iteration/boundary controls are static. Binary transitions retain hard physics and receive branch-local AD tests.
- `VerisConfig.physical_constants` is separate from numerical `custom_parameters`; `replace_veris_parameters(state, parameters, *, component="CICE") -> RunState` is the only new public state bridge.
- The traced VerCOR `freshwater_density` owns freshwater conversion and native rhoFresh/derived ratios for each transition.
- `Exchange.source_coverage: Literal["component", "binary"] = "component"`; binary override requires an explicit quantity.
- `VerosConfig.surface_forcing: Literal["atmosphere", "coupled"] = "atmosphere"`; reject coupled mode for native-internally-forced ACC.
- Native `ocean_mass_flux` and `ocean_salt_mass_flux` are downward-positive actual bulk/salt mass rates. Veros salt forcing is `(-S_ref*M + 1000*J)/rho_fw`, with fixed `S_ref=34.7` g/kg by default, in m (g/kg)/s.
- The fixed-reference Boussinesq virtual-salt budget differs from physical ocean/ice salt inventory when local salinity differs from the reference; validate/report the two separately.
- Native residual heat is upward-positive W/m2; ocean heat input is its negative. Do not add shortwave, atmospheric precipitation/evaporation, concentration or timestep factors twice.
- Keep final instantaneous diagnostics separate from `sum(dt_native*flux)/dt_coupling`; keep shapes/dtypes/PyTree schemas fixed.
- The prognostic example shares the exact ocean/ice grid and native area quadrature; do not generalize its conservation proof to different staggered/remapping grids.
- `run_veris_calibration.py` and `run_veris_veros_era5.py` use the existing packaged `run_setup(*, loglevel, float_type)` contract. Keep the prescribed-data gallery's purpose.
- Use Python 3.12 conda `scipy` for VerCOR. Keep native dependencies isolated; stock/JAX Veros run in separate processes. Never use `KMP_DUPLICATE_LIB_OK`.
- Every meaningful task updates bounded `PROGRESS.md`; tests precede code, full unit correctness precedes commits, and `pytest tests/ -q --fast` precedes every VerCOR commit.

## Review Focus

1. Malformed/unknown/owned physical overrides, zero/all-dry weights and an observation window insensitive to h0 must fail clearly without mutating an initial state or discovering optional models during import (C1/C5).
2. An ice-free or newly melted wet source can carry nonzero total flux; all send, supplied-state, coverage and dispatch checks must honor binary support (C2/C3).
3. Stress must use the momentum-time face fraction, valid distinct U/V faces, and mean of products; neither final tracer concentration nor product of means is equivalent (C3).
4. Coupled forcing must preserve independent restoring and spinup behavior while removing the cold proxy only after explicit feedback is active; SSH must be real SSH, not streamfunction (C4/C5).
5. Intermediate invalid physics, segmented runs, native/remap area conventions and an installed optional-dependency wheel must be exercised rather than inferred from final finite fields (C3/C6).

## Prerequisite and shared gates

- [ ] Complete native N1–N4; read its exact SHA, artifact/source location, status/diagnostic metadata and results. Install that local wheel/source in a task-specific target for development; verify import provenance, not just distribution version. Do not overwrite the user's environments or use the older editable native checkout.
- [ ] Activate the existing working Python 3.12 `scipy` environment. Use a unique `/private/tmp` test base/cache, never a repo-local basetemp. Native assets and the installed-GitHub-CLI test need cache-lock/localhost permissions, not code changes.
- [ ] Preserve current baseline evidence: stock 1,955/1,955, fast 1,057/1,057, exact-fork AD 10/10, Black/strict flake8/mypy/compileall passing. Run new focused tests RED before each unit.

For each code commit: Black `vercor tests`, strict `flake8 . --count --max-line-length=120 --statistics` (no exit-zero for acceptance), `mypy vercor tests`, compileall, whitespace, full stock `pytest tests/ -q -m 'not veros_autodiff' --tb=short`, then the required fast gate. Run isolated exact-fork AD for units changing native/coupled execution, masks, payloads or fluxes. Native and stock checks stay separate; record exact commands/counts and known third-party warnings. Update active API signature/manifests and private-module inventory in the same unit as an intentional API/file change. Never rewrite historical specifications or audit SHA citations during repinning.

### Task 1: C1 — Spherical snapshot, physical payload and public parameter replacement

**Files:**
- Create: `vercor/setups/_external/veris_parameters.py`, `tests/test_veris_parameters.py`.
- Modify: `vercor/setups/config.py`, `_external/veris.py`, `veris_runtime.py`, `veros_data.py`, `vercor/setups/__init__.py`.
- Modify active pin: `pyproject.toml`, `.github/workflows/python-package.yml`, `tests/test_veris_component.py`, `tests/test_veris_gallery.py` pin expectations; use the actual N4 SHA in all active controls.
- Test: `tests/test_veros_data_ocean.py`, `test_veris_component.py`, `test_veris_rollouts.py`, API/lazy/distribution contracts.
- Document: `DESIGN.md`, `DEPENDENCIES.md`, `docs/api-architecture-review.md`, `PROGRESS.md`.

**Interfaces:**
- Consumes native N1 geometry fields and N2 PhysicalConstants PyTree/methods/status.
- Produces `VerisConfig.physical_constants: Mapping[str, object] | None = None`, defensively owned like existing configuration mappings.
- Produces `VerisPayload(state, diagnostics, physical_constants)`; static `VerisRuntimeResources` retains configuration/substep alignment/dtype only.
- Produces lazy public `replace_veris_parameters(state: RunState, parameters: Mapping[str, object], *, component: str="CICE") -> RunState` in `veris_parameters.py`. It replaces only payload physical values, using native `with_overrides`; the caller's native state, stores and component owner remain unchanged.
- Snapshot geometry adds authoritative native `latitude`, `latitude_edges`, `radius`; native face conversions stay west/south. `load_standalone_veris` capability-checks the corrected parameter/geometry/status interface.

- [x] **Write RED contracts.** Define `test_public_parameter_replacement_differentiates_without_mutation`. Assert immutable mapping/array ownership, lazy import without Veris/Veros installed, correct scalar shapes/dtypes, and a tracer-safe public path:

  ```python
  candidate = replace_veris_parameters(initial, {"h0": jnp.asarray(0.43)})
  final = coupler.run(candidate)
  assert candidate is not initial
  assert np.isfinite(jax.grad(concentration_loss)(jnp.asarray(0.43)))
  ```

  Define `concentration_loss` in this test through only the public replacement helper, public fields and Coupler.run. Check nonzero derivative and unchanged original result after repeated .4/.6 candidates. Reject unknown/derived/numerical/dtype keys, absent/wrong component, nonscalar/complex values and direct rhoFresh overrides (owned by VerCOR). Test conflicting authored rhoFresh versus matching setup values. Verify density 997 and 1030 kg/m3 round-trip atmospheric SI mass into native physics. Check snapshot coordinates/radius against Veros settings, plus tagged face geometry and latitude-dependent native length at 60 degrees.
- [x] **Run RED.** `pytest tests/test_veris_parameters.py tests/test_veros_data_ocean.py -q --tb=short`; expect missing public helper/config/coordinates or the current static-parameter failure. Verify native imports are the N4 source first.
- [x] **Implement the thin bridge.** Forward native constants during setup; retain parameters in payload. Derive native rhoFresh from StepContext constants for each transition. Use `state._with_component_state(...with_payload(...))` internally, preserving the opaque public boundary. Normalize physical scalar leaves to runtime dtype, validate names/shape/domain with native rules, and keep cached numerical resources free of evolving parameters. Capture actual snapshot radius and edges without rescaling stored areas.
- [x] **Verify GREEN and compatibility.** Run named tests, native direct-advance equivalence, public/lazy contracts, both dtypes, parameter JVP/VJP/finite differences and shared gates. Update tests which intentionally expected static resources or a single exported native contract; preserve their ownership/assertions.
- [x] **Commit.** Include actual pin/CI/API changes and progress; commit as `feat: expose traced Veris physics on spherical ocean geometry`.

### Task 2: C2 — Binary source coverage for already integrated fluxes

**Files:**
- Modify: `vercor/exchanges.py`, `_runtime/contracts.py`, `component_masks.py`, `exchange_masks.py`, `exchange_dispatch.py`, `field_transfer.py`, `state_validation.py`.
- Test: `tests/test_dynamic_mask_exchanges.py`, `test_dynamic_mask_rollouts.py`, `test_v0_4_routes_and_state.py`, `test_runtime_state.py`, public signature contracts.
- Document: `DESIGN.md`, `docs/api/grids-exchanges.rst`, architecture/API artifacts, `PROGRESS.md`.

**Interfaces:**
- Produces approved `Exchange(..., source_coverage="component")`; valid choices are component/binary, with binary+legacy rejected.
- Extends private `ExchangeContract` with `binary_sends: tuple[str, ...]=()`, built from outgoing binary routes while preserving existing import/export order and fan-in checks.
- Extends shared `field_active_mask` with keyword-only `binary_mask: RuntimeArray | None=None`, `binary_output_fields: tuple[str, ...]=()`; the default remains unchanged. Send/supplied-state validation passes contract metadata and current binary support for those fields.
- `exchange_mask_operands` chooses source binary or fractional support once; dispatch weighting, overlap, current diagnostics and last-received support consume that same result.

- [ ] **Write RED cases.** Extend the existing test-only `_scenario` builder with a forwarded source_coverage keyword. Define `test_binary_source_flux_survives_zero_ice`:

  ```python
  state, route, context = _scenario("source_flux", source_area=jnp.zeros((2, 2)), source_coverage="binary")
  state = state.replace_fields("SRC", {"flux": jnp.full((2, 2), 12.0)})
  result = _dispatch(state, route, context)
  np.testing.assert_allclose(result.component("DST").field("flux", scope="received"), 12.0)
  ```

  Default component source_flux delivers zero; A=.5 still delivers 12 under binary support. Add `test_binary_coverage_retains_last_dispatch_support` for complete melt, source omission, supplied replacement and host/JAX parity, and `test_binary_send_requires_finite_wet_flux` for NaN with A=0 at send/supplied-state boundaries. Cover invalid choice/type, binary+legacy, topology intersections, empty overlap and inactive NaNs.
- [ ] **Run RED.** `pytest tests/test_dynamic_mask_exchanges.py tests/test_dynamic_mask_rollouts.py -q --fast --tb=short`; first demonstrate missing keyword, then pin the numerical/stored-support failures.
- [ ] **Implement one support policy.** Build binary_sends through the existing contract builder, preserving it when a component is both source and destination. Use the shared mask resolver for weighting, coverage and validation rather than create a parallel dispatch path. Any field sent by a binary route needs finite binary support even if another route uses component coverage. Keep last-dispatch support separate from current coverage.
- [ ] **Verify GREEN.** Assert source weighting before remapping, conservative scalar totals, vector transformation, distinct finite east/north/coast/exterior values, topology restrictions, current versus stored masks, and old default behavior. Use JVP/VJP/finite differences through continuous fields/fractions; binary membership stays discrete. Run shared gates and exact-fork AD.
- [ ] **Commit.** Commit as `feat: support binary source coverage for total-area fluxes` with signatures/docs/progress.

### Task 3: C3 — Interval-mean Veris ocean fluxes and correct face stresses

**Files:**
- Create: `vercor/setups/_external/veris_fluxes.py`, `tests/test_veris_feedback.py`.
- Modify: `_external/veris_forcing.py`, `veris_runtime.py`, `veris.py`, `veris_output.py`; `vercor/recipes.py`.
- Test: `tests/test_veris_forcing.py`, `test_veris_rollouts.py`, `test_veris_output.py`, quantity/recipe/API contracts.
- Document: units/coverage in module docstrings, setup/output docs, dependencies and `PROGRESS.md`.

**Interfaces:**
- Produces frozen PyTree `VerisForcingPack(native_forcing, open_water_u_stress, open_water_v_stress)` and `prepare_veris_forcing_pack(fields, *, ocean_mask, constants, dtype) -> VerisForcingPack`; keep existing `prepare_veris_forcing(...) -> dict` as a delegating compatibility helper. Both use one shared bulk evaluation, not repeated bulk physics.
- Produces frozen PyTree `VerisSurfaceFluxes(sea_surface_heat_flux, sea_surface_mass_flux, sea_surface_salt_mass_flux, ocean_u_stress, ocean_v_stress)` and `veris_surface_fluxes(state, diagnostics, forcing_pack, *, dtype) -> VerisSurfaceFluxes` in the new owner.
- Produces `VERIS_TO_OCEAN_FLUX_FIELDS`: three named scalar fields plus `VectorField("ocean_u_stress", "ocean_v_stress")`, with the approved units/signs.
- Produces frozen PyTree `VerisIntervalBudget(external_mass_flux, external_heat_flux)` and extends payload with `interval_budget`. These two native boundary-input means support independent long-run closure; the five canonical means remain in ordinary fields, avoiding duplicate payload ownership. Parameter replacement uses immutable dataclass replacement so it preserves this added carry.
- CICE exports `ice_fraction` plus those five interval means, with finite zero initial fluxes. Instantaneous native diagnostics remain in payload; means live in ordinary runtime fields.

- [ ] **Write RED averaging/stress tests.** Define `test_two_native_substeps_export_interval_means`, `test_stress_uses_mean_of_momentum_face_products`, and `test_intermediate_native_failure_cannot_disappear`. Own their pure controlled native-transition fixture in `test_veris_feedback.py`, with two equal substeps reporting downward heat 10/30, mass 2/4, and salt .001/.003. Assert interval means 20,3,.002, not the final values. For face concentration .25/.75 and ice traction 2/4 with zero open-water traction, require mean stress 1.75, not 1.5 or the final product. Tag west/south arrays to prove canonical east/north shifts, periodic x, closed north edge and coast zeros. Set first substep physics_status nonzero and final status zero: coupled execution must fail, including under JIT. A separate controlled hidden native leaf becoming nonfinite and later finite must also fail.
- [ ] **Run RED.** `pytest tests/test_veris_feedback.py tests/test_veris_rollouts.py -q --fast --tb=short`; reproduce missing exports/last-substep behavior before implementing means.
- [ ] **Implement the flux pack and scan.** Compute bulk once per forcing pack; reapply native Qnet/Qsw every substep. Use each native result's retained momentum-time AreaW/AreaS and face masks to combine ice traction plus complementary open-water stress. Zero closed/coastal faces before export and shift once to canonical positions. Accumulate dt-weighted complete flux products/external boundary inputs and nonzero physics status through the fixed-shape loop; divide once by coupling duration. Aggregate finiteness of all owned native numeric leaves plus ice/snow/Area bounds before/after each substep into a scalar validity carry; do not rely on the final state or issue one host callback per field/substep. Invalid carry must not be reused for another physics step. Validate the aggregate before accepting results. Use corrected actual mass/salt diagnostics, never reverse-convert legacy salt forcing.
- [ ] **Verify GREEN with real native physics/output.** Test A=0/.5/1, complete melt with nonzero residual flux, freshwater/latent budgets, rho changes and smooth parameter gradients. Keep real direct-advance/forcing-reset controls. Output native instantaneous diagnostics with registry units, plus canonical mean fields; give east/north stress coordinates their actual upper-edge positions, distinct from native west/south velocity coordinates. Check tagged dimensions, period selection, snapshot units and output-free outer JIT. Run shared gates and AD.
- [ ] **Commit.** Commit as `feat: export interval-mean Veris ocean boundary fluxes` with public recipe and metadata updates.

### Task 4: C4 — Explicit total-feedback mode and prognostic ocean surfaces

**Files:**
- Create: `tests/test_veros_ice_feedback.py`, `tests/_veros_surface_support.py` for an actual small native column/grid fixture.
- Modify: `vercor/setups/config.py`, `_external/veros_gcm.py`, `veros_gcm_state.py`, `veros_runtime.py`, `veros_state.py`, `veros_fluxes.py`, `veros_setup.py`; inherited JAX callback as required.
- Test: existing independent restoring/freshwater coverage, `tests/test_veros_autodiff_rollouts.py`, provider/contract tests.
- Document: ocean forcing/SSH/restoring/spinup semantics and `PROGRESS.md`.

**Interfaces:**
- Produces `VerosConfig.surface_forcing` and propagation into the setup owner. Coupled global mode consumes only `VERIS_TO_OCEAN_FLUX_FIELDS`; atmosphere mode retains its input/output behavior. Reject coupled ACC.
- Produces `compute_coupled_fluxes(fields, constants, dtype) -> tuple[Array, Array, Array, Array]` returning canonical taux,tauy,downward qnet,converted virtual salt flux.
- Existing `prepare_surface_forcing_fields(..., qnec, ...)` accepts qnec=None; `apply_veros_forcing_fields` then preserves native climatological qnec. Avoid a second layout/copy pipeline.
- Coupled mode requires pressure/free-surface execution (`enable_streamfunction=False`) so native `ssh` is active; reject conflicting explicit streamfunction overrides. Atmosphere-only behavior is unchanged.
- Produces `extract_veros_runtime_surface_fields(state, *, coupled: bool) -> dict[str, Array]`: coupled mode exports SST in K, SSS in g/kg, canonical native east/north surface u/v in m/s, and actual native ssh in m. Atmosphere mode continues SST-only.

- [ ] **Write RED conversions and real native increments.** Define `test_total_feedback_enters_native_tracers_once`. Use a four-by-four, two-level actual Veros fixture with mixing/advection disabled only to isolate the source increment; retain actual surface kernels/callbacks. With S_ref=34.7, rho_fw=1000, M=1e-4 and J=0:

  ```python
  np.testing.assert_allclose(salt_flux, -3.47e-6, rtol=1e-12)
  np.testing.assert_allclose(delta_salt, dt*salt_flux/dz, rtol=1e-11)
  np.testing.assert_allclose(delta_temp, dt*qnet/(rho_0*cp_0*dz), rtol=1e-11)
  ```

  Add pure-freezing actual-salt, zero-local-salinity/reference-budget, nondefault density, masked and no-double-dt/depth cases. Tag native u/v/sss/ssh and an unrelated large psi: verify real units/positions and no psi-as-height conversion. Parameterize independent temperature/salinity restoring switches. At SST below -1.8 C with cooling, coupled heat and salt remain nonzero while the old atmosphere-only control retains its behavior.
- [ ] **Run RED in separate lanes.** Run the stock test module and the corresponding `veros_autodiff` cases in the exact-fork process; reproduce missing mode/exports and the cold-cutoff loss rather than use cross-backend mocks.
- [ ] **Implement the one-owner mode.** Add native scalar activation metadata initialized inactive for ocean climatological spinup and set active only when total feedback is installed. The callback then gates the proxy cutoff without Python branching on traced values. Preserve restoring coefficients/weights and nonaccumulating salt assignment; skip bulk/rain/snow recomputation in coupled runtime. Keep single host payload copy and functional JAX copying. Extract updated surfaces after native stepping with native time levels; never fabricate zero SSH to satisfy the export.
- [ ] **Verify GREEN and compatibility.** Check spinup before activation, first activation, repeated substeps, all restoring combinations, boundary halos, canonical faces and both dtypes. Test invalid mode/ACC/conflicting streamfunction settings early. Use JVP/VJP/directional finite differences for total heat/mass/salt and native output surfaces. Run old atmosphere-only tests, shared gates and AD.
- [ ] **Commit.** Commit as `feat: couple Veros to total sea-ice surface forcing` with config/signature/docs/progress.

### Task 5: C5 — Shared prognostic setup and reusable synthetic calibration

**Files:**
- Modify: `vercor/setups/_veris.py`, `_external/veros_gcm.py`, `veros_data.py` geometry extraction, `vercor/setups/__init__.py`.
- Create: `vercor/setups/gallery/run_veris_veros_era5.py`, `run_veris_calibration.py`; `tests/test_veris_prognostic_gallery.py`, `tests/test_veris_calibration.py`.
- Modify: gallery catalog/CLI/docs/contracts, `docs/setup-gallery.rst`, `PROGRESS.md`.

**Interfaces:**
- Produces frozen `VerisVerosSetup(seaice: Component, ocean: Component)` and lazy `make_veris_veros(*, config: VerisConfig | None=None, ocean_config: VerosConfig | None=None) -> VerisVerosSetup`. Default ocean_config is coupled global mode; reject an explicitly incompatible mode. Initialize Veros once, reuse its exact grid for both components and extract geometry from that same native state. Refactor private owner/component assembly in `veros_gcm.py` for reuse; do not duplicate setup or retain a second mutable native ocean.
- Produces the two approved gallery modules' `run_setup(*, loglevel: str, float_type: str) -> None`.
- Calibration example owns frozen PyTree `VerisCalibrationResult(h0, loss, gradient, iterations, evaluations, converged)` and reusable `calibrate_lead_closing_thickness(coupler, initial_state, observed_fraction, *, weights=None, initial_h0=0.5, bounds=(0.25,0.8), component="CICE") -> VerisCalibrationResult`.

- [ ] **Write RED setup/calibration tests.** Define `test_prognostic_pair_initializes_one_ocean_on_one_grid` for identical ocean/ice grid objects, exact authoritative edges/areas, actual bidirectional recipes, run order LND->CICE->OCN->ATM and all incoming intensive versus outgoing binary source_flux quantities. Preserve the existing prescribed-data gallery's absence of feedback. Define parameterized `test_synthetic_h0_recovery_and_holdout`: five wet cells, 300 s native steps, nEVPsteps=8, initial A=.5/hIceMean=.25/TSurf=260 K and fixed cold forcing; truths .35/.43/.65 m, starts .3/.7 m, bounds .25–.8 m. Assert `abs(result.h0-truth) < 1e-5`, `result.converged`, and held-out concentration error <1e-7 in float64. Pin wrong-shape/negative/nonfinite/all-zero wet weights, invalid bounds and original-state immutability. Add `test_calibration_rejects_insensitive_observation_window` using stationary full ice/no lead opening: do not accept an arbitrary h0 simply because SciPy sees zero gradient. Require float32 smoke recovery error <2e-3 with the same bounds, explicitly reflecting its quantized concentration target.
- [ ] **Run RED.** `pytest tests/test_veris_prognostic_gallery.py tests/test_veris_calibration.py -q --tb=short`; expect missing factory/templates/functions first. Keep scenario construction local to the packaged example, not imports from tests.
- [ ] **Implement focused factories/workflows.** Separate geometry-only extraction from data-climatology copying where needed. Build the prognostic example with conservative total-flux routes on shared geometry and explicit sequential lag timing. Build the calibration example around the public parameter replacement helper, fixed wet weights including ice-free cells, JAX-jitted value_and_grad and SciPy L-BFGS-B (ftol=1e-14, gtol=1e-10, maxiter=100). Normalize concentration residuals by 1e-3; reject invalid observations/weights/bounds before optimization. Check that the bounded window produces resolvably different weighted concentration (endpoint difference greater than 32*eps times the concentration scale); report an insensitive window rather than successful parameter recovery. Do not add a general optimizer/observation framework or a new dependency. Print concise recovery diagnostics; no I/O in the differentiated objective.
- [ ] **Verify GREEN with native models.** Exercise CLI list/copy/run and optional-import behavior. Run a short actual host and exact-fork JAX prognostic coupling with evolving ice/ocean, finite state and declared mass/virtual-salt/heat increments; test coupled h0 gradients in a smooth regime. Test fitted h0 against a distinct smooth cold forcing, zero source ice coverage, both dtypes and no hidden owner mutation. Run shared gates and AD.
- [ ] **Commit.** Commit setup/gallery/calibration work as `feat: add prognostic Veris coupling and synthetic calibration` with manifests/docs/progress.

### Task 6: C6 — Extended scientific report, installed artifacts and authorized pushes

**Files:**
- Create: `benchmarks/__init__.py`, `benchmarks/validate_veris.py`, `tests/test_veris_validation.py`, `docs/veris-validation.rst`.
- Modify: `tests/test_veris_gallery.py`, `docs/index.rst`, benchmark/gallery documentation, `DESIGN.md`, `DEPENDENCIES.md`, bounded `PROGRESS.md`.
- Evidence: ignored `test_logs/veris-scientific-validation/` with JSON/logs/selected outputs, actual native/VerCOR SHAs and build hashes.

**Interfaces:**
- Produces `python -m benchmarks.validate_veris --days 30 --model-seconds 3600 --output <directory>` and an explicit `--days 365` benchmark using the real prescribed-data gallery with output optional. The verifier owns monitoring/JSON, not an alternative model integrator.
- Report records forcing provenance, geometry/radius/native areas, initial polar ice, native/coupling steps, dtype, commits, all-native-step finite/bounds/status summaries, stored-area inventories, interval flux budgets, segmented equivalence, and timestep sensitivity.

- [ ] **Write verifier tests first.** Define `test_validation_report_tracks_all_native_steps`, `test_validation_rejects_lost_or_duplicated_flux`, and `test_segmented_validation_preserves_parameters_and_flux_carry`. Exercise a small controlled case and negative controls for intermediate nonfinite/status failure, last-substep substitution, duplicated fraction/density/dt factors and mismatched geometry conventions. Assert checked-step count equals the configured native-step count, each bad control fails, and segmented/uninterrupted results agree at equivalent forcing timestamps. Keep annual sweeps as explicit acceptance commands, not repetitions in every ordinary unit gate; log detailed diagnostics.
- [ ] **Run RED, then implement the monitor using existing workflows.** Aggregate finite/bounds/status checks through every native substep without storing an unbounded trajectory. Use initial polar wet ice A=.5/hIceMean=.25 m, TSurf=260 K to exercise ice even when an all-zero default stays ice-free. Preserve shared I/O boundaries and reduce scalar diagnostics on device before host output.
- [ ] **Execute extended acceptance.** Retain N4's >=1,000-step transport proof. Run real 30-day prescribed global cases at 3600/1800/900 s native steps, with the same daily coupling/forcing timestamps and stored native areas. For area-weighted volume/cumulative-flux vectors, require decreasing successive timestep differences outside a reported roundoff floor of `64*eps*number_of_native_steps*quantity_scale`; investigate unresolved refinement rather than claim accuracy. Independently close stored ice/snow mass plus ocean mass input against the external mass diagnostic, ice salt against actual ocean salt, and modeled enthalpy/residual ocean heat against the external heat diagnostic; use normalized float64 budget residual <=1e-10. Run annual 3600/1800 s cases, record their sensitivity and finite/bounded/storage/status checks, and verify segmented equivalence at fixed dtype/timestep. Record measured results without claiming observed climate skill. Complete synthetic recovery and actual prognostic budget/gradient checks from C4/C5.
- [ ] **Review and run final gates.** Use independent scientific, gradient and code-quality review; reproduce findings before fixes. Run native and VerCOR required lint/format/type/docs/fast/full/AD gates on final sources; collect combined branch coverage, build wheel/sdist, and run installed artifacts outside the checkout. Do not interpret a feature-only push as a hosted-CI run.
- [ ] **Commit, publish native dependency, verify remote pin, then push VerCOR.** After all local source gates and reviews pass, commit the report/harness/docs unit. Authenticate/configure Git transport with `gh auth status`/`gh auth setup-git`, push the passing native dedicated branch with Git, and verify its exact SHA with `gh api repos/team-ocean/veris/git/ref/heads/feat/vercor-scientific-validation`. Install the actual GitHub immutable pin in an isolated target, verify PEP 610 URL/commit and all required capabilities, and rerun the installed native/VerCOR smoke. If that exposes a defect, fix and repeat the applicable gates before VerCOR publication. Push `feat/veris-seaice` and compare `git rev-parse HEAD` with `gh api repos/nutrik/vercor/git/ref/heads/feat/veris-seaice`; require clean local status and matching SHAs. No PR, merge, release or tag is requested.

## Self-review and ordering

Native N1–N4 precede C1. C1 supplies payload parameters/geometry; C2 supplies wet-support routing; C3 supplies interval means and the exact flux recipe; C4 consumes that recipe and exports the prognostic ocean surfaces; C5 assembles the shared-grid workflows and proves coupled inverse recovery; C6 owns extended runs, installed/remote evidence and final push. Every approved spec section has a task; each Review Focus condition has named assertions. New public symbols/signatures, native status handling, both density/salinity conventions, active/stored route support, face coordinates, native spinup/restoring, parameter gradients, hard transitions and optional imports are covered explicitly. Implementation must not start until the user reviews both plans and chooses the execution method.
