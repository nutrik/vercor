# Veros Float32 and Runtime Performance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Validate real Veros float32 behavior, measure and improve runtime performance, resolve repository lint findings, and consolidate redundant tests.

**Architecture:** Keep scientific behavior and public owners unchanged. Optimize measured validation overhead in existing private helpers; extend the existing profiler and retain independent numerical/physical/AD tests.

**Tech Stack:** Python 3.12 scipy312, JAX 0.11.2, stock Veros 1.7.0, pinned exact-fork Veros, JCM 3, Veris 0.2.1, Perfetto, pytest, Black, flake8, mypy.

**Spec:** `docs/superpowers/specs/2026-11-02-performance-float32-design.md`

## Global Constraints

- Preserve public APIs, physics, native continuation, masks and temporal means.
- Stock and exact-fork Veros execute in separate Python processes.
- Retain distinct numerical, physical, autodiff, backend, boundary and error tests.
- Do not suppress lint codes, weaken physical tolerances or alter native packages.
- Use the existing feature checkout and put detailed artifacts in `/private/tmp`.
- Full and fast suites must pass before local commits; stage intended paths only.

## Review Focus

- Precision selection before native import/lock; fixtures must not promote arrays.
- Successful validation should avoid host work without losing invalid rejection.
- JVP/VJP, inactive NaNs, vmap and late scan failures must retain their behavior.
- Warm timing must exclude compilation and synchronize asynchronous execution.
- Consolidated API tests must preserve independent expected-value oracles.

---

### Task 1: Explicit native Veros float32

**Files:** `tests/_veros_surface_support.py`, focused Veros tests and precision helper if needed; relevant CI lane in `.github/workflows/python-package.yml`.

**Interfaces:** Use existing `surface_model`, `step_veros_runtime`, `StepContext`, and `VEROS_FLOAT_TYPE=float32` before native backend imports. No new product API.

- [x] Add dtype-first tests that fail if float64 is accidentally allocated or exported; run before changing fixture precision.
- [x] Keep fixture depth/forcing arrays in the explicitly selected dtype.
- [x] Validate one/multiple-step mapped means, native continuation/reset, wet/dry cells and actual native tracer source budgets with precision-derived tolerances.
- [x] Validate JIT/JVP/reverse source sensitivity in an isolated exact-fork float32 process; preserve float64 regressions.
- [x] Run focused host/JAX precision lanes and record exact provenance and results.

### Task 2: Profiling and measured optimization

**Files:** `vercor/_numerical_safety.py`, `vercor/_runtime/state_validation.py` if measured, `tests/test_numerical_safety.py`, existing profiler `vercor/setups/gallery/profile_runtime.py`, `tests/test_runtime_run.py`.

**Interfaces:** Keep `_require_zero(count, owner, requirement)` and `_require_true(predicate, message)` behavior. Extend the gallery profiler without changing the shared `run_setup(loglevel, float_type)` CLI contract.

- [x] Capture baseline cold and warm slab/native profiles under the verified dependency stack; save timings and Perfetto traces.
- [x] Add a failing regression proving valid compiled checks avoid a host callback while invalid values still raise with their owner and requirement, including later scan steps and transformed paths.
- [x] Implement conditional failure callbacks in the existing validation owner only if the profile supports it; benchmark identical runs before/after and keep only beneficial changes.
- [x] Add profiler tests for synchronized cold/warm measurements and trace export before extending its implementation.
- [x] Record reproducible slab/native host/JAX timing and trace evidence, compilation/memory limits and numerical equivalence.

### Task 3: Lint boundary and test consolidation

**Files:** `.flake8`, duplicate test files identified by the audit, retained contract tests and the test audit report.

**Interfaces:** Existing repository-wide `flake8 . --max-line-length=120`; unchanged pytest markers/selection and expected contract inventory.

- [x] Reproduce 200 lint findings and establish every affected file is ignored evidence; verify tracked Python is clean.
- [x] Exclude `test_logs` using `extend-exclude`; run repository-wide strict lint without error-code suppression.
- [x] Record removed-to-retained node/assertion maps, preserving independent oracles.
- [x] Remove only proved functional duplicates and run all affected tests.
- [x] Compare full collection and branch coverage with baseline; explain any material coverage change.

### Task 4: Review and final verification

**Files:** this plan, approved spec, bounded `PROGRESS.md`, performance evidence documentation and docs inventory if needed.

**Interfaces:** Stock prefix `PYTHONPATH=/private/tmp/vercor-mask-stock:/private/tmp/vercor-mask-tools`; exact-fork prefix adds `/private/tmp/vercor-temporal-ad`; interpreter `/Users/spj483/miniforge3/envs/scipy312/bin/python`.

- [x] Review complete scientific, performance, test-removal and lint diff independently; fix material findings with focused evidence.
- [x] Run Black, strict repository flake8, mypy, compileall, fast/full stock, exact-fork AD, explicit float32 lanes and combined branch coverage.
- [x] Run relevant docs checks, whitespace validation and source-hash checks over settled final gates.
- [x] Update plan/PROGRESS with verified counts, timings, limitations and failed approaches; make focused local commits only after required suites pass.

## Execution evidence and rulings

- The user chose the focused approach and autonomous implementation/verification on 2026-11-02. Continue without additional design questions.
- Ruling: use the existing clean feature checkout rather than create a new worktree; it is the active feature workspace and prior project work explicitly uses this branch. No change to main/master.
- Ruling: root owns profiling/lint/integration; existing focused audit agents can implement the disjoint precision and test-consolidation tasks. They do not commit; one independent whole-diff review checks the integrated result.
- Preliminary JAX0.10.2 CPU slab traces establish feasibility only: 24 steps on16x32, seven synchronized warm runs, median0.098039s. Final comparisons use the verified JAX0.11.2 overlay and disclose instrumentation separately.
- Baseline lint:200 findings in four ignored evidence snippets;327 tracked Python files pass. Raw audit at `/private/tmp/vercor-lint-baseline.md`.

- Runtime TDD RED:2 expected valid-callback failures; GREEN35 numerical cases and100 runtime/state/compile/output cases. Profiler RED:2 missing-option/signature failures; GREEN4 harness cases, including real Perfetto trace content.
- Final isolated JAX0.11.2 slab comparisons (nine warm samples) are bitwise identical: f64 83.242→6.721ms, f32 90.151→6.722ms; callbacks2506→240. Native two-step T21/4x4 all-JAX (five samples):107.776→19.558ms, callbacks2411→20, all68canonical fields identical. Cold compile rises7.500→9.496s. Host474.388→461.922ms has no material gain established. Actual Perfetto v59.1 SQL confirms callback counts.
- Discarded initial after-host timing overlapped focused runtime tests; used isolated `native-after-host-clean` repeat. Exact CPU model query is sandbox-denied; report ARM64 macOS26.7.2/cpu:0 without inventing hardware.
- Consolidation removes33 collected nodes across five files, retains all independent expected values; affected158 pass and five mutation checks fail as intended. Strict repository lint nowzero via generated-evidence exclusion only.
- Native f32 RED exposed untyped fixture dzt under x64 capability; explicit dtype fixes allocation. Actual upstream mixing correctly invalidated an early unmixed heat/salt oracle; replaced forcing by a physically stable heat/freshwater case, without changing native physics/tolerances. Native AD-fork x64-capability promotion of kappaH requires separate authentic x64-disabled float32 process; allocation remains separately tested with capability enabled.
- Review live probe confirms valid batched vmap may execute both callback branches; safety and gradients are preserved. Documentation scopes host-call avoidance to scalar compiled checks.

- First integrated stock diagnostic:2637 cases,3 failures (334.479s). Two existing checkify tests exposed conditional-XLA error wrapping at the deliberately unjitted bilinear source boundary; one existing CI command literal lacked the newly added precision module. CI literal corrected after isolated RED/GREEN. Existing checkify expectations remain unchanged. A static private callback policy opts out only at the scalar source-entry boundary; compiled extrapolation/default core checks remain conditional. Focused bilinear/numerical/runtime-compilation80 pass; scoped independent review clear. All final timings are being remeasured and complete gates rerun; preliminary gains above are superseded by final receipt.

- Final settled-code remeasurements after checkify repair: f64 slab12.8779x, f32slab13.1574x, nativeall-JAX5.3936x; all39slab leaves and68native canonical fields match raw bytes. Nativehost ratio0.982 indicates no material improvement. Perfetto SQL callbacks slab2506→240, nativeJAX2411→55 (20interruptions+35preserved entryguards), host152→0. Final comparison receipt: `/private/tmp/vercor-performance-final-comparisons.json`. Fullstock2/AD and final static gates running on318-module source snapshot2; source edits are frozen.

- Read-only approved hardware query resolves the initial sandbox limitation: AppleM4Pro,14logicalcores,24GiB RAM; macOS26.7.2. Both final stock/AD processes are active CPU consumers (not stalled); final source318 hashes remain unchanged.

- Final settled-source gates: stock2637/fast1677/exact-forkAD17, failures/errors/skips0; combinedcoverage92.9490022172949% (94.86430965509786% statements and84.20533070088845% branches), unchanged90% floor.318source/test hashes remain unchanged, raw stock/AD hashes preserved. Black318/mypy318/repolintzero/compileall/diff pass. Dedicated f32 stock39/exact+f64temporal16 pass. Final profiler realCLI JSON/trace smoke passed. Final strict docs and focused local commits follow.

- Final strictSphinx HTML build3 passes; final coverage JSON/reporter pass and raw source318 + both lane hashes verified unchanged. All requested implementation, focused/scientific validation, full/fast/AD gates and independent/scoped reviews are complete. Delivery remains local on existing feat/veris-seaice; no remote merge, push, release or PR is requested.
