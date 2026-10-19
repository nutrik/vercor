# Source and coupling cleanup implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Require stock Veros >=1.7.0, remove obsolete source and test duplication, and enforce correct coupling field contracts with executable physical and AD evidence.

**Architecture:** Retain structural components, immutable runtime state, prepared coupling, explicit exchange quantities, native model boundaries, and shared output. Remove alternative owners and unused forwarding paths. Correct field behavior at the first producer or consumer where it diverges from the physical contract.

**Tech Stack:** Python 3.12 conda `scipy`, JAX 0.11.2, released JCM 3.0.0 and Veris 0.2.1, stock Veros >=1.7.0, separate pinned Veros AD, pytest, Black, flake8, mypy, Sphinx, Git, and GitHub CLI.

**Spec:** [Approved specification](../specs/2026-10-19-source-and-coupling-cleanup-design.md).

Status: approved on 2026-10-19 for subagent-driven execution.

## Global constraints

- Work and commit on `feat/veris-seaice`; preserve baseline `e1d417a` and unrelated edits.
- Change only stock Veros to `veros>=1.7.0`; retain fork revision `7a8c964cf00b5aa0713c995edd760643a431b3c9`.
- Keep the reviewed JCM 3.0.0/JAX 0.11.2 dependency exception unchanged.
- Follow `AGENTS.md`, `DESIGN.md`, and the complete approved spec; update `DEPENDENCIES.md` when ownership changes.
- Keep physics traced, precision static, payloads immutable, optional imports lazy, and output opt-in.
- Preserve explicit `intensive`, `source_flux`, `target_flux`, binary preweighted exports, and current versus received coverage.
- No replacement compatibility aliases, alternate runtime paths, generic field registry, numerical fudge factors, new test timeouts, or weakened gates.
- Use `/private/tmp` for basetemps/artifacts and writable Veros assets; run stock and fork in separate processes.
- Write regressions before bug fixes; retain existing behavior tests before pure removals. Record RED/GREEN evidence and every deleted test's surviving assertion.
- Every logical commit requires passing focused, fast, and full stock tests in its actual state. Run applicable AD checks for numerical/runtime changes. Stage intended paths only.
- No push, release, or publication is part of this plan.

## Review focus

1. A missing or stale runtime payload must never fall back to mutable setup state; Task 3 tests this with conflicting setup and runtime values.
2. Non-six-hour native CAMulator steps must preserve fixed radiation rates and signs; Task 5 covers 1, 3, and 6 hours and multiple native substeps.
3. Initial tensors may have history and forcing channels but no radiation diagnostics or vapor species; Task 6 covers the latest frame, exact timestamp, missing channels, and total-water rejection.
4. A component swap at a fractional coast may double-weight a preweighted field or invalidate stored imports; Task 7 covers unequal/evolving/zero coverage, conservation, and AD through the public coupler.
5. An apparently duplicate test may protect a different precision, artifact, backend, or derivative; Task 8 records surviving assertions and retains these distinctions.

## Commit and validation gates

Use the existing isolated environment interpreter
`/private/tmp/vercor-time-ci/envs/scipy/bin/python` until a task explicitly
needs its native dependency changed. The command examples below use `python`
inside that conda environment. Set
`VEROS_ASSET_DIR=/private/tmp/vercor-cleanup-assets` for native stock tests.
Use loopback access for the installed GitHub CLI request-construction test.

Focused commands use `-n0` for clear RED evidence and `--tb=short`. Keep verbose
logs and XML under `test_logs/source-coupling-cleanup/` or `/private/tmp` and
print concise results. Before each logical commit, run:

```bash
python -m pytest tests/ -q --fast -m 'not veros_autodiff' --tb=short --basetemp=/private/tmp/vercor-cleanup-fast
python -m pytest tests/ -q -m 'not veros_autodiff' --tb=short --basetemp=/private/tmp/vercor-cleanup-full
git diff --check
```

Use distinct basetemps when processes overlap. A failure blocks the commit;
fix it without deleting or skipping the relevant contract. Source-independent
documentation commits may reuse the just-completed full suite when the source
and tests have not changed, while retaining the required pre-commit fast gate.

### Task 1: Stock Veros requirement and native compatibility

**Files:** Modify `pyproject.toml`, `DESIGN.md`, `vercor/setups/_external/veros_state.py`, dependency assertions in `tests/test_distribution_boundaries.py` and `tests/test_veris_component.py`, and the cache-interface assertion in `tests/test_external_components_coverage.py`.

**Interfaces:** Keep public factories and the fork extra unchanged. Stock native state copies and component-owned solver caches retain their current behavior.

- [ ] Change the existing metadata assertion to require exactly `['veros>=1.7.0']`, leaving one canonical assertion in distribution tests. Run it and observe failure against the current extra.
- [ ] Set the stock extra to `veros>=1.7.0` and change the solver-interface diagnostic to describe the required mutable cache capability, without a `<1.7` version claim.
- [ ] Install stock Veros 1.7.0 into the isolated `scipy` lane after plan approval. Verify `importlib.metadata.version('veros')` and preserve the separate fork target/process.
- [ ] Run `tests/test_external_components_coverage.py`, `tests/test_veros_data_ocean.py`, `tests/test_veros_ice_feedback.py`, and the dependency metadata cases. Preserve copy completeness, cache detachment, owner isolation, and restoration after native exceptions.
- [ ] Update active dependency/CI documentation; retain archived Veros 1.6.2 evidence unchanged. Run the commit gates and commit this dependency migration.

### Task 2: One prepared runtime path

**Files:** Modify `vercor/coupler.py`, `vercor/_runtime/{contracts,coupler_state,preparation,initialization,exchange_topology,topology,topology_policy,validation}.py`, `tests/_runtime_helpers.py`, and their runtime/topology contract tests.

**Interfaces:** `build_exchange_contracts(component_names, exchanges)` always validates endpoints. `runtime_state_from_components(components, fractional_masks, *, contracts, prefill_missing=False)` requires prepared contracts. `prepare_runtime_state(initial_state, *, prepared)` always validates supplied state. Topology builders accept declared components/exchanges/policy, without preseeded maps.

- [ ] Retain and run current unknown-endpoint, fan-in, route capability, immutable topology, supplied-state schema, lifecycle identity, and received-support tests before changing implementation.
- [ ] Remove `_initialize_runtime`, optional contract rebuilding, `validate_endpoints`, unreachable endpoint skips, the empty-contract no-op check, unused topology-map injection, and `validate_state`. Use canonical field flattening directly.
- [ ] Bind structural `component.step` directly; keep signature normalization inside `CallableComponent`. Reject an invalid structural step signature at preparation, and preserve the callable convenience's current arity/error cases.
- [ ] Replace raw-component topology fallbacks with binding-owned author objects. Migrate test fixtures to the real declared factories and preparation boundary; remove tests that exercise only retired private alternatives.
- [ ] Run `tests/test_runtime_state.py`, `tests/test_runtime_run.py`, `tests/test_runtime_exchange.py`, `tests/test_coupler_runtime.py`, `tests/test_coupler_coverage.py`, `tests/test_plugin_architecture.py`, and all dynamic-mask modules. Run applicable isolated fork runtime/rollout tests, update ownership docs, then run commit gates and commit.

### Task 3: Canonical helpers and runtime-owned output

**Files:** Modify `vercor/regridding.py`, `vercor/_regridders/{bilinear,conservative}.py`, `vercor/forcing_index.py`, `vercor/setups/_data/_field_helpers.py`, `vercor/setups/_external/{_jax_gcm_pytree,jax_gcm_tools,jax_gcm_output,veris_forcing,veros_runtime}.py`, public manifests/signatures, and the related regridder/helper/native-output tests.

**Interfaces:** Public regridding signatures remain unchanged, including `radius_km`. `calendar.YearType` is the sole year-type owner. JCM snapshots consume `SnapshotContext.payload`; native Veris forcing consumers use `prepare_veris_forcing_pack`.

- [ ] Extend the existing JCM final-payload snapshot regression: use conflicting mutable setup `_state`, pass no runtime payload, and require a component-scoped error without creating output. Run the case and observe the stale-state fallback failure.
- [ ] Remove that fallback and constructor-owned attribute defaults in JCM/Veros setup resources. Complete native test fixtures rather than preserving fake legacy resources.
- [ ] Have public factories construct concrete private regridders directly and remove duplicate private factories. Preserve identical-grid no-construction behavior, scalar conservation, vector rotation, and both NumPy/JAX inputs.
- [ ] Remove `ForcingYearType`, test-only dtype/stack/unwrap/surface helpers, the unused JCM parameter helper, standalone prediction-output convenience, and the Veris forcing forwarder. Migrate unique assertions to the current production owners; keep integer/complex/category, conservation, and gradient guarantees.
- [ ] Update imports, public inventories, dependency order, and migration notes. Run focused regridder, helper, native output, Veris forcing/parameter/rollout, public API, and installed extension cases, followed by commit gates and a refactoring commit.

### Task 4: Consistent slab heat and moisture signs

**Files:** Modify `vercor/setups/_slab/{atmosphere,land,ocean}.py`, `tests/test_slab_kernels.py`, `tests/test_component_models_coverage.py`, and slab field documentation.

**Interfaces:** Preserve factory names, supported physical fields, runtime dtype, and existing coefficients. `SHF=10*(T_air-SST)`, `LHF=0.5*SHF`; ocean restoring is `-(T-T_ref)/tau`; downward negative latent heat dries soil.

- [ ] Add parameterized warm/cold zero-flux ocean cases at `T_ref +/- 1 K`, `dt=86400 s`, `tau=30*86400 s`; assert anomaly contraction by `29/30`, and analytic JVP/VJP `29/30`. Observe current anti-restoring failures.
- [ ] Add independent flux direction cases for `(T_air,SST)=(280,290)` and `(300,290) K`, expecting sensible heat `-100/+100 W/m2` and latent heat `-50/+50 W/m2`. Assert matching ocean tendency direction. Observe the current sensible-sign failure.
- [ ] Add soil cases `moisture=.3`, `LHF=-100/+100 W/m2`, and `dt=86400 s`; expect `.29136/.30864`. Preserve clipping endpoints, finite derivatives, and zero-flux identity. Observe current evaporation/condensation direction failures.
- [ ] Correct the three kernels and docstrings. Update prior tests that mirror incorrect algebra while retaining their JIT/AD and shape assertions; introduce no new fitted coefficients.
- [ ] Run slab kernels, component model, JCM/slab gallery, and runtime integration tests, then commit gates and a physics correction commit.

### Task 5: Native CAMulator radiation duration

**Files:** Modify `vercor/setups/_external/{camulator_fields,camulator_runtime,camulator_gcm_state}.py`, `tests/test_camulator_component_kernels.py`, and the CAMulator field audit documentation.

**Interfaces:** Add required keyword `native_step_seconds: float` to `map_camulator_prediction_arrays` and `map_camulator_prediction_to_runtime_fields`, preserving their existing arguments. Derive it from the existing native model timestep; retain the accumulated-energy and downward-positive training contract.

- [ ] Parameterize the existing mapper test at `3600`, `10800`, and `21600 s`, supplying shortwave energy `100*seconds J/m2` and net longwave energy `-3*seconds J/m2`. Expect `100 W/m2` shortwave and `sigma*TS**4-3 W/m2` downward longwave. Observe non-six-hour failures.
- [ ] Add zero/negative duration rejection before array arithmetic and a runtime case with two native substeps per coupling interval; rate conversion must use each native duration, not the coupling duration.
- [ ] Thread the native duration through existing resources and mapping calls; replace both fixed `21600` divisors. Preserve source signs and surface-emission placement exactly once.
- [ ] Run CAMulator mapping/runtime, shared bulk, and hypsometric tests, then commit gates and the duration fix commit. Cite the pinned upstream energy contract in active docs.

### Task 6: Authentic CAMulator startup and vapor capability

**Files:** Modify `vercor/setups/config.py`, `vercor/setups/_external/{camulator,camulator_gcm_state,camulator_fields,camulator_contracts,camulator_runtime}.py`, `vercor/setups/gallery/run_camulator_with_veros.py`, CAMulator tests, public signature contracts, and native setup documentation.

**Interfaces:** Add `CAMulatorConfig.initial_diagnostics: Mapping[str, object] | None = None` and `water_vapor_variable: str | None = None`. The mapping contains `time` and required absent native `TS/FSNS/FLNS`; own its mutable values defensively. Setup uses the existing cursor's effective initial time. Both initial input and prediction accessors must provide the declared vapor channel; `Qtot` is rejected as total water.

The pure shared kernel has signature
`_map_camulator_air_fields(constants: PhysicalConstants, *, reference_pressure: float, hyai: object, hybi: object, hyam: object, hybm: object, u_wind: object, v_wind: object, temperature_3d: object, water_vapor_3d: object, surface_pressure: object) -> dict[str, jax.Array]`.
The host initialization mapper is
`map_camulator_initial_runtime_fields(resources: CAMulatorGCMSetupState, model_state: torch.Tensor, cursor: CamulatorRuntimeCursor, constants: PhysicalConstants) -> dict[str, jax.Array]`;
use annotation-only imports for the resource type to avoid an import cycle.
The resource constructor receives the two new config declarations, and the
prediction wrapper receives the selected vapor variable plus Task 5's duration.

- [ ] Before implementation, add config ownership tests and setup regressions using a history-bearing normalized input tensor with distinct latest-frame meteorology. The transformer exposes separate input/output inverse methods; assert only the input method is called during setup and the stepper/cursor/native state do not advance.
- [ ] Add explicit diagnostics at the selected initial time and require exact mapped temperature, vapor humidity, pressure, height, density, winds, and radiation. Feed those exports to the existing bulk closure on wet cells and assert finite, physically directed nonzero forcing.
- [ ] Add missing/mismatched timestamp, missing diagnostic, wrong shape, nonfinite active value, absent vapor declaration, `Qtot`, missing input channel, and missing prediction channel cases. Confirm they fail before coupling or native advancement. Preserve optional-dependency-free config access and lazy factory import behavior.
- [ ] Factor the existing array mapper's meteorology into one pure `_map_camulator_air_fields` owner accepting traced `PhysicalConstants` plus native arrays. Prediction and initial mapping reuse it; keep host tensor decoding outside the JIT kernel and use the latest history frame explicitly.
- [ ] Build initial fields in the existing setup lifecycle from decoded native state and explicit diagnostics. Remove the test-only zero initializer, keep runtime payload ownership unchanged, and validate the new vapor and diagnostic declarations with component-scoped errors.
- [ ] Update the gallery to load timestamp-matched diagnostics from its native initial-condition data and name a real configured vapor channel. A total-water-only checkpoint must report the missing physical capability; do not substitute a saturation law or reference weather.
- [ ] Run CAMulator config/field/setup/output/runtime and optional-import cases, public signatures, and installed artifact boundaries. Clearly separate boundary tests from unavailable native CREDIT/checkpoint execution. Run commit gates and commit this native boundary correction.

### Task 7: Honest prescribed contracts and interchangeable components

**Files:** Modify `vercor/setups/_data/{era5_atmosphere,era5_land,era5_ocean,erainterim_ocean}.py`, `vercor/setups/_external/camulator_land.py`, `vercor/recipes.py`, affected gallery routes, and their contract tests. Create `tests/test_component_interchangeability.py` before any new production behavior.

**Interfaces:** Prescribed model outputs remain prescribed. General `DataComponent` imports and forwarding remain supported; a data gallery that deliberately displays imports declares that diagnostic role explicitly. Valid swaps use public component/spec/exchange APIs and identical physical field meaning.

- [ ] Map each bundled declared input to a physics read or an explicitly demonstrated diagnostic use. Remove unused slab/prescribed-model imports and their dead routes/recipes; retain genuine diagnostic relays in the data gallery using explicit existing `DataComponent` declarations.
- [ ] Add `test_swapping_producer_preserves_bulk_flux_units_and_signs`, parameterized over data, callable JAX, structural JAX, and host producers. Supply `T=theta=280 K`, `q=.002 kg/kg`, `rho=1.2 kg/m3`, height `10 m`, wind `(5,0) m/s`, SST `290 K`, zero currents, shortwave `100 W/m2`, longwave `300 W/m2`, and rain/snow `1e-3/2e-4 kg/m2/s`. Use the existing CESM bulk receiving closure; require identical fields/fluxes across producers, negative sensible/latent heat, and virtual salt forcing `-34.7*(rain+snow+evap)/1000 < 0`. This checks swapping and independent physical signs without duplicating the bulk solver's implementation.
- [ ] Add `test_swapped_components_preserve_coastal_products_and_received_support` using the existing dynamic-mask suite's independently verified quantities across the same four producer kinds. Cover nonuniform fractions, inactive NaNs, evolving zero/reactivated support, different scalar grids, and rotated vectors. Reuse the existing quantity/conservation expected cases rather than reproducing their numerical test bodies; the new guarantee is independent component substitution and receiving physics.
- [ ] Include JIT, analytic JVP/VJP, and finite differences for continuous fields and coverage. Preserve binary transitions as discrete. Demonstrate a preweighted Veris field receives one area factor, not its square.
- [ ] Exercise existing native ERA5/JCM conversions and prescribed/prognostic Veros-to-Veris surfaces. Assert early errors for missing required fields, unsupported native face geometry, and incompatible physical species.
- [ ] Run the new matrix with dynamic-mask, freshwater, native forcing, setup/gallery, and isolated AD suites. Update field producers/consumers/units/signs in `docs/exchange-coverage.rst`, run commit gates, and commit the contract cleanup and interchangeability evidence.

### Task 8: Redundant tests and complete source audit record

**Files:** Modify `tests/test_{api_boundaries,public_api_contracts,v0_4_public_api,v0_4_routes_and_state,coupler_coverage,runtime_run,dtypes,bilinear_rectilinear_regridder,conservative_rectilinear_regridder}.py` and touched private-layout test modules. Create `docs/source-coupling-audit.md` with final removal/retention and surviving-assertion tables.

**Interfaces:** Canonical manifests/signatures, public behavior, physical invariants, AD, precision, artifact isolation, and native ownership remain the test owners. The report describes actual final changes, not planned deletions.

- [ ] Before deleting each duplicate, record its exact assertion and surviving test, including precision/backend/artifact/input distinctions. Start from the audited duplicate absence, export, identity-grid, dtype, and PyTree smoke cases in the approved spec.
- [ ] Combine duplicate run-order and foreign-state test bodies without losing any parameter/corruption cases. Keep one dependency metadata assertion and remove obsolete private-import/implementation-spelling checks as their source layers disappear.
- [ ] Run the retained tests before and after consolidation. Preserve conservation versus interpolation, native versus wrapper rollout, JVP/VJP versus finite differences, inactive NaN versus infinity, payload/solver ownership, and wheel/sdist/extension/CLI distinctions.
- [ ] Review the entire final source inventory against the spec's removal table. Record necessary adapters and any remaining candidate with its current caller and purpose; remove no additional mechanism solely by name or coverage.
- [ ] Update `DESIGN.md`, `DEPENDENCIES.md`, architecture inventories, migration notes, active examples, and `PROGRESS.md`. Run commit gates and commit the test consolidation/audit record separately from physics changes.

### Task 9: Final validation and Git delivery

**Files:** Update validation evidence in `docs/source-coupling-audit.md`, `docs/exchange-coverage.rst`, this plan, and `PROGRESS.md`; use ignored `test_logs/source-coupling-cleanup/` for detailed logs and artifacts.

- [ ] Run Black, strict flake8 on maintained `vercor tests tools benchmarks`, the requested repository lint report, mypy on `vercor tests`, compileall, and `git diff --check`. Resolve all maintained-source failures.
- [ ] Run full stock tests and fast tests with actual stock Veros >=1.7.0. Run the exact fork's native AD suites in a separate process with verified PEP 610 provenance; combine coverage and enforce the current 90% project gate without changing exclusions.
- [ ] Run strict Sphinx, build wheel/sdist, check metadata with Twine, and run installed wheel/sdist, extension runtime/typing, and CLI-copy probes outside the checkout.
- [ ] Run a short real JCM/Veros/Veris coupled gallery from the final built wheel. Check finite saved fields, exact native step count, route quantities, field signs/units, output ownership, and evolving ice/coastal coverage.
- [ ] Review the complete branch diff against every original numbered requirement. Record native CAMulator limits explicitly; stand-in tests are not native model validation. Fix any discovered discrepancy at its first source and repeat only the affected and final required gates.
- [ ] Complete the removal/test assertion maps and final gate receipt, update progress within 180 lines, stage only intended paths, commit the delivery record, and verify branch, SHA, clean status, and logical commit history with Git/GitHub CLI. Do not publish or push.

## Plan self-review

The stock dependency is Task 1; private compatibility cleanup is Tasks 2 and 3;
physical signs and native field conversion are Tasks 4 through 6; honest model
contracts and interchangeable-component evidence are Task 7; redundant tests
and the complete source inventory are Task 8; all delivery gates are Task 9.
Each review-focus condition has an explicit owning test step. Implementation
begins only after this written plan is reviewed and its execution method is
selected; the spec's written approval does not authorize skipping that stage.
