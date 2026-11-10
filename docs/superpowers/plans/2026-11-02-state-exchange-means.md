# State Exchange Means Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Exchange each bundled component's mapped state as a mean over native substeps, preserving single-step endpoints and exact native continuation.

**Architecture:** Keep sampling, mapping and temporal reduction in the existing native adapters. Canonical fields are diagnosed before averaging; immutable native payloads retain endpoint state. Generic dispatch, public protocols and prescribed-data selection retain their ownership.

**Tech Stack:** Python 3.12, JAX, JCM 3, Veros stock/exact JAX fork, Veris, pytest, Black, flake8, mypy, Sphinx, gh.

**Spec:** `docs/superpowers/specs/2026-11-02-state-exchange-means-design.md`

## Global Constraints

- Work and deliver on existing `feat/veris-seaice`; no new worktree/branch.
- Mean N post-native-step samples; omit the initial instant; N=1 gives endpoint.
- Diagnose nonlinear canonical fields before temporal reduction.
- Preserve native continuation payload, clock, categorical dtype and snapshots.
- No new public API, dependency, native numerical change or weakened tolerance.
- Use approved `scipy312`; stock Veros and exact-fork AD run in separate processes.
- Full and fast suites pass before commits; stage only intended paths, push with gh authentication and verify remote SHA.

## Review Focus

- Initial setup has no completed interval: initialization remains instantaneous.
- Multiple coupling intervals must reset accumulators and preserve continuation.
- Moving ice fraction must refresh component/route support from its mean export.
- Inactive values must retain existing mask/NaN and finite derivative behavior.
- Native diagnostic output must retain endpoint/sample meaning independently of exchange means.

---

### Task 1: JCM atmospheric means

**Files:** `vercor/setups/_external/jax_gcm_state.py`, `jax_gcm_runtime.py`; `tests/test_jcm_temporal_exchange.py`.

**Interfaces:** keep `_generate_step_function(...): Callable[[JCMState, ForcingData], tuple[JCMState, dict[str, jax.Array]]]`. Extend compact raw samples with bottom-level dynamics/normalized pressure. Reuse `map_jcm_output_fields` per sample and reduce mapped exports. Preserve `JCMState` endpoint and eight surface rate histories' phase semantics.

- [x] Update/add RED tests for temperatures 280/300 ->290, wind 1/5 ->3, pressure .9/1 ->95000; density/potential-temperature independent per-sample oracles; float32/64; singleton; gradient, native continuation and compact samples.
- [x] Run JCM fast tests and record expected mean/endpoint failures.
- [x] Implement compact sampling and mapped-state reduction; retain initialization and applied forcing echoes.
- [x] Run focused JCM tests including actual native continuation/save cadence and differentiation.

### Task 3: Veros surface means

**Files:** `vercor/setups/_external/veros_runtime.py`, `veros_state.py`; `tests/test_veros_temporal_surfaces.py` (or focused state-mean companion).

**Interfaces:** accumulate `extract_veros_runtime_surface_fields(state, coupled=...)` after each native step. Preserve the existing native advancement helper if independently used; factor actual duplication only. Return final native state and mean surface mapping without storing trajectories. Per-sample open-water contribution uses received ice fraction.

- [x] Add RED tests for 1/2/3-step analytic SST, SSS, currents, SSH and open-water means, host/JAX precision, immutable endpoint and two-interval reset.
- [x] Implement per-step surface accumulation with no native numerical changes.
- [x] Run GREEN in separate stock and exact-fork processes; check JVP/reverse and actual native pressure/continuation regressions.

### Task 4: Veris ice property means and route evidence

**Files:** `vercor/setups/_external/veris_runtime.py`; `tests/test_veris_feedback.py` and related route/compilation tests.

**Interfaces:** extend `_VerisCarry` with canonical surface state sums, preserving sticky validity and native flux/budget sums. Return mean ice fraction, temperature and per-sample diagnosed albedo with final `VerisPayload`.

- [x] Add RED tests with changing ice fraction/temperature and nonlinear albedo; singleton, reset, mean-mask dispatch, inactive support and gradients.
- [x] Accumulate state fields in the existing pure loop, retaining native validity and flux meanings.
- [x] Run GREEN plus Veris compilation, output and finite gradient checks on JAX/host/mixed/custom lanes.

### Task 5: Contract documentation, review, verification and delivery

**Files:** `DESIGN.md`, `DEPENDENCIES.md`, `docs/coupling-fields.rst`, `docs/interface-wrapper-audit.md`, bounded `PROGRESS.md`, this plan and approved spec.

- [x] Update temporal state/flux/output contract and third-party/prescribed/single-step semantics without changing dependency layers.
- [x] Obtain independent code/test review, address material findings with regressions.
- [x] Run Black, maintained strict flake8, repository lint accounting, mypy, compileall, fast/full stock tests with coverage, isolated exact-fork AD, documentation and short native host/JAX coupled integrations.
- [x] Record actual results and failed approaches below. Git delivery follows the verified gates on the current branch; the final handoff records the committed/pushed SHA independently verified through GitHub CLI.

## Evidence and failed approaches

- Required scipy Python3.14 fast orientation: all four workers aborted before test results; exit5. User approved existing scipy312. No environment modified.
- scipy312 focused pre-change baseline:26 pass. Its installed JAX0.10.2 stubs cause two pre-existing `Config.jax_enable_x64` mypy errors; verified existing temporary stock overlay (Veros1.7.0/JAX0.11.2) passes mypy316. Repository lint baseline200 historical findings; production/test maintained lint tracked separately.
- Existing branch/local and GitHub tip initially both `52afaf2cbc62726ea57feb8f89aa7227cccf5a72`; starting checkout clean. User approved the design and implementation. Root executes tasks; independent reviewers read scientific design and final code/tests.

- Independent final review: one Important output-ownership finding fixed via a RED changing-substep provider/period/snapshot regression. Native output uses setup-bound endpoint diagnosis; exchanged fields remain means. Explicit Veros float32 regression remains a deferred Minor; host and exact-fork JAX temporal tests use float64. No runtime/peak-memory performance claim is made.
- Host and all-JAX native coupled probes each completed two 3600-second coupling intervals, four atmospheric/ocean and 24 ice native steps, and 13 output files. Independently captured native samples match exchange means to at most5.69e-14; all runtime fields finite and native clocks correct. The probes distinguish means from endpoints. Receipts: /private/tmp/vercor-state-mean-native-{host,jax}/verification.json.
- Interim broad failures were traced to stale endpoint assertions, a zero-step-only forcing stand-in, bounded PROGRESS181 lines, and sandbox-denied native asset lock files. Regressions now assert independent native means and separate native-output endpoints; PROGRESS restored to180 lines. Final suites use the authorized native cache-lock boundary without altering installed packages. A probe imported native Veros before runtime configuration; corrected import order before any coupled validation.
- First authorized final full stock run:2664 pass/1 stale gallery assertion (native snapshot Area compared with exchange mean); stock coverage92.37%. Isolated exact-fork marked AD15/15 passed. The gallery assertion now compares snapshot Area with endpoint native payload; full/fast reruns are required before commit. No product or physical tolerance change for this correction.
- Execution ruling: user-approved implementation stays in the named current branch; one atomic state-exchange feature commit after all final gates avoids transient intermediate commits with mismatched endpoint tests. Starting checkout was clean. Performance not profiled; validation makes numerical/ownership claims only.
- Final settled-tree gates: full stock2665/2665, fast1677/1677, isolated exact-fork markedAD15/15, native temporal host14/14 and exact-JAX14/14; zero failures/errors/skips. Combined line/branch coverage92.94524802555684% (9245 statements,2024 branches), unchanged90% floor. Black316 unchanged, mypy316 pass, maintained flake8zero, repositorylint200 historical unchanged, compileall/diff/strictSphinx pass. Final316 source/test hashes verified. Raw stock/AD coverage lanes preserved; the combined JSON report was moved outside the coverage database prefix after the reporter mistook its JSON suffix for parallel data. Clean coverage report passes. Machine receipt: /private/tmp/vercor-state-mean-final-results.json.
- Delivery is one atomic implementation commit on the user-requested branch, with gh-authenticated Git transport and independent GitHub ref lookup. No PR, merge, release or tag.
