# Temporal exchange implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans for inline execution, specialized parallel workers for independent owners, and superpowers:requesting-code-review before delivery. Steps use checkbox syntax.

**Goal:** Correct findings1–4 and verified finding5 defects, validate and publish on feat/veris-seaice.

**Architecture:** Endpoint states plus native-interval flux means in existing model adapters; shared dispatch owns record-time refresh. Native model carry and public interfaces remain stable unless a verified contract correction requires explicit documentation.

**Tech Stack:** Python, JAX, JCM3, stock/exact-fork Veros, Veris, Torch/CAMulator adapter stand-ins, pytest, mypy, Black, flake8, Sphinx, gh.

**Spec:** docs/superpowers/specs/2026-10-30-temporal-exchange-design.md

## Global constraints

- Existing branch feat/veris-seaice; preserve pre-existing PROGRESS receipt. No installed-package mutations.
- Frozen functional runtime payloads, traced physical constants and static shape/dtype policy.
- Endpoint state, native-step-integrated flux, framewise nonlinear diagnosis, exact native clocks.
- Public manifests and lazy optional imports remain stable; no field registry or duplicate selector.
- Write and observe regressions before production edits; no weakened tolerances or skipped failures.

## Review focus

- Flux integrals invariant under diagnostic save interval, with exact native endpoint continuation.
- Rain/snow phase covariance and longwave endpoint/interval distinctions.
- Daily/monthly selection for source-after-target, omitted sources, masks and current evolving feedback on host/JAX/mixed/custom runs.
- Native SSH units, pressure time levels, and explicitly selected asynchronous integration policy.
- Gradients, inactive NaNs, precision, payload schema and repeated-run compilation reuse.

### Task 1: JCM interval fluxes and phase

Files: existing jax_gcm_state.py, jax_gcm_runtime.py, _jax_gcm_pytree.py as needed; dedicated temporal tests plus affected native/runtime tests.
- [x] Add failing independently integrated native/synthetic flux and framewise-phase tests, save-interval invariance and endpoint/categorical/carry checks.
- [x] Request native-step snapshots via run_from_state_with_carry; retain endpoint dynamics/physics and exact carry; map each sample before averaging flux fields with dtype-preserving helpers.
- [x] Run focused tests, native trajectory equivalence and JIT/JVP/VJP checks; update deliberately changed saved-array assumptions explicitly.

### Task 2: CAMulator complete-window radiation

Files: camulator_runtime.py and camulator_fields.py if needed; dedicated temporal tests and existing CAMulator kernel tests.
- [x] Add failing changing native-window energy/TS tests with analytic integral and final-meteorology assertions.
- [x] Reuse canonical mapper for each native sample; average radiation rates across the complete block, retain final meteorology and output history ownership.
- [x] Verify native-duration conversion, one/multiple windows, precision and output distinction.

### Task 3: Clock-selected dispatch

Files: _runtime/driver.py, exchange_dispatch.py, field_transfer.py as needed; dedicated temporal exchange tests.
- [x] Reproduce one-step lag for daily/monthly data on host/JAX/mixed/custom plus omitted-source workflows, masks and current feedback controls.
- [x] Refresh non-current source sent fields with existing selector at dispatch for the receiving clock; do not invoke author step or change current feedback semantics.
- [x] Verify runtime state, mask ownership, output/supplied-state contracts and compilation boundaries.

### Task 4: Veros timing investigation and necessary repair

Files: existing Veros adapter owners, setup/config if verified necessary; new actual-kernel temporal tests.
- [x] Characterize pressure slots/units on stock/exact-fork and the native driver rotation, test endpoint SSH coherence.
- [x] Characterize default dt_mom/dt_tracer policy against coupled flux integration; repair only proven coupled-boundary inconsistencies and test explicit opt-in native policy.
- [x] Verify fresh-process host/JAX native paths and independent budget/endpoint expectations.

### Task 5: Review, quality gates and delivery

- [x] Independent implementation/test/gradient/performance review; resolve every verified finding.
- [x] Update coupling docs, audit, DESIGN/DEPENDENCIES and bounded PROGRESS with final behavior, failed approaches and exact validation.
- [x] Run Black, maintained flake8 plus account for whole-repository historical lint, mypy, compileall, fast/full stock, isolated exact-fork AD, appropriate coverage/native/docs/package checks.
- [x] Create small coherent source commits after full suite passes. Authorized current-branch push follows the documentation receipt commit; remote SHA and clean-tree verification are recorded in the final handoff.

## Execution receipts

- Started at6db074a with only pre-existing PROGRESS.md audit note dirty. Inline implementation authorized by the user's explicit findings and current-branch delivery request; routine design choices use the supplied audit scope.
- Baseline from the immediately preceding read-only audit:178 focused pass,1 installed AD-fork/SciPy tol API incompatibility; scipy3.14 workers abort before meaningful validation. Reuse verified temporary stock/AD launchers for final lanes, never patch installed dependencies.
- JCM RED:6 corrected synthetic regressions fail against isolated HEAD source; projected-history RED1 and final-applied-forcing RED1 observed separately. Corrected test-only clock integer to exact JAX datetime and reference latent heat to the actual2501000J/kg constant; physical tolerances unchanged. Broad JCM/native/mapping focus167pass; compact-return JCM/external-owner focus85pass. Existing endpoint/save fixture assumptions updated explicitly.
- CAMulator RED5 plus isolated nonlinear-LW RED1; final133focus pass in both available JAX environments. Real mapper/accessor/output around deterministic CREDIT stand-in; native CREDIT unavailable.
- Dispatch RED26/11controls; focused337pass incl37new acrosshost/JAX/mixed/custom, mask/NaN/gradient/supplied-state and repeated-schedule executable checks. Current feedback unchanged.
- Veros actual stock/exact-fork pressure RED and timestep RED verified;11newnative tests pass in each lane. Three-step native replay leaves prognostic state/pressure/clock bitwise unchanged while SSH follows the current solution. Explicit600s column fixture now pins its intended momentum timestep.
- Independent review: no remaining correctness blockers in all repairs. Performance review required projection before native wrapper JIT return: T31L8/daily/float64 returned history405.003MiB ->13.500MiB; endpoint-inclusive420.960MiB ->29.457MiB (93%reduction). These are shape totals; wrapper jitted=False may materialize full native intermediates. CAMulator maps full per-frame meteorology too; bounded native-window cost remains a nonblocking observation.
- Independent native JCM6h vs3h+3h across Feb29:151endpoint/carry leaves bitwise equal in both precisions, mean heat/radiation difference<=1.14e-13W/m2(float64),6.11e-5W/m2(float32). Fresh-process midpoint restart:158endpoint/mean leaves bitwise equal in both precisions. Native short probe precipitation is zero; nonzero phase and conservation evidence is analytical. Constant/date changes reuse existing helper executables.
- Failed validation approaches: initial full-fast orientation overlapped worker edits and mixed stock/AD native runtimes, so is not a final gate. Sandbox blocked loopback and existing native asset locks; final broad checks run with authorized access in separate stock/AD processes. Removing the now-unused but independently tested private tree_mean helper was rejected by mypy; original helper restored unchanged. No installed dependency edits or weakened tests/tolerances.

- Final settled-source gates: full stock2651, fast1662, exact-forkAD26; zero failures/errors/skips. Combined branch coverage92.96638804908412% exceeds unchanged90% requirement; both raw lane hashes preserved. All316 source/test hashes unchanged through final gates. Black/mypy316/maintainedflake8zero/compileall/strictSphinx/build/Twine/source-wheel-sdist180/diff pass; repository-wide lint retains200 historical test_logs findings.
- Initial broad trial failed on five implicit600s native fixtures and two cached references to the temporarily removed helper; all resolved by explicit intended fixture parameters and unchanged original helper restoration. ExactAD trial23pass/3alignment failures resolved by the same600s fixture declarations; all original derivative assertions retained. Final gates above supersede trials.
- Actual global2day validation passes in independent stock-host and exact-forkJAX processes:96ocean/48ice/96ATM steps each, both clocks+172800s,2319wet cells,68finite runtime fields,13output files perlane. Current SSH/export identity exact; pressure/g error host0/JAX2.22e-16m. Native endpoint output and canonical marine readback match exactly. Host58.924s/JAX251.264s; native48-call JAX loop has a longer initial compile, while the prescribed physical substep count is retained. No climate-skill claim.
- Native global receipts: /private/tmp/vercor-temporal-native-{host,jax}/verification.json; scientific restart/integral probe: /private/tmp/vercor-jcm-interval-review.py; final lane/coverage receipts: /private/tmp/vercor-temporal-final-results.json. Native CREDIT remains unavailable; CAMulator real-mapper/adapter regressions use deterministic inference stand-ins.
- Delivery ruling: create coherent JCM, CAMulator, dispatch, Veros and documentation commits after the identical final source tree passed full/fast/native/AD gates. Source remains unchanged between these commits, so completed gates apply to the delivered tree without redundant reruns. User already authorized current-branch commit and GitHub CLI push; no merge or PR is requested.

- Created source commits after final gates: 6b4060d Align coupled Veros surface state and native timesteps; 0ab3f48 Select prescribed forcing at the receiving clock; 068a222 Average CAMulator radiation over all native windows; e411780 Fix JCM interval flux integration and endpoint state.
