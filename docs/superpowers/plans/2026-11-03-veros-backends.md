# Released Veros Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enable and measure Veros1.7.0 NumPy/JAX backends and coupled host/scanned execution, adopting native JAX by default if it measurably improves throughput.

**Architecture:** Separate native numerical backend from component execution in the existing Veros adapter. Keep stock host orchestration and the pinned differentiable-fork scanned path. Add a focused reproducible benchmark harness with explicit provenance and numerical comparisons.

**Tech Stack:** Python3.12 scipy312, JAX/JAXlib0.11.2, stock Veros1.7.0, fork7a8c964, Veris0.2.1, JCM3.0.0, pytest, Black, flake8, mypy, GitHub CLI.

**Spec:** `docs/superpowers/specs/2026-11-03-veros-backends-design.md`

## Global Constraints

- Work in the current `feat/veris-seaice` checkout; preserve installed environments.
- Put isolated dependencies, logs and raw benchmark artifacts in `/private/tmp`.
- Configure native backend before native implementation import, in separate processes.
- Preserve native physics, solver tolerances, dtype policy, temporal means and output.
- No full-stock-scanned claim; stock kernels and fork scans are distinct evidence.
- Benchmark without concurrent tests; synchronize leaves/effects before stopping clocks.
- Default decision follows repeatable steady-state coupled throughput above uncertainty.
- Run fast/full suites before commits; stage intended files and verify pushed SHA.
- User approved design and explicitly instructed proceeding; execute natively in this session with independent final review.

## Review Focus

- Conflicting backend requests and already-locked runtimes fail before native mutation.
- Stock JAX does not accidentally import fork-only helpers; fork calm-stress AD stays finite.
- Host JAX payloads do not share mutable diagnostic/timer ownership across runs.
- Benchmark repetitions start from unchanged native and canonical initial state.
- New defaults preserve explicit NumPy selection, optional imports and output semantics.

---

### Task 1: Separate native backend and component execution

**Files:** `vercor/setups/config.py`, `vercor/setups/_external/veros_runtime_settings.py`, `veros_gcm.py`, `veros_setup.py`, `veros_state.py`, `veros_runtime.py`; `tests/test_veros_backends.py`; existing setup/API tests as required.

**Interfaces:** Add keyword-only `VerosConfig.backend: Literal["numpy", "jax"] | None = None`. Extend `configure_veros_runtime(execution: Literal["host", "jax"] = "host", *, backend: Literal["numpy", "jax"] | None = None) -> None`. None retains current execution-based policy initially; contradictory NumPy/scanned-JAX is invalid. No new public adapter class.

- [x] Rebuild an isolated stock stack and verify imported versions/paths; use separate fork precedence for AD.
- [x] Add failing tests: host+JAX runtime config; invalid backend; NumPy/scanned rejection; factory import order with explicit backend; locked conflict; stock TKE finite forward values; forcing JAX placement; actual native host-JAX progression and state independence.
- [x] Run focused tests and record expected failures before production edits.
- [x] Add backend config/validation and pass through the factory without widening core runtime APIs.
- [x] Select stock-compatible TKE arithmetic while preserving fork-specific safe derivatives.
- [x] Reproduce any native aux-owner sharing; fix only demonstrated adapter-local ownership/array placement issues.
- [x] Run focused `tests/test_veros_backends.py`, setup/runtime/state regressions and isolated native probes; update API signature expectations deliberately.

### Task 2: Measure and decide the default

**Files:** create `benchmarks/benchmark_veros.py`; create `tests/test_veros_benchmark.py`; existing native gallery/config and documentation only where needed.

**Interfaces:** `python -m benchmarks.benchmark_veros` accepts native backend, execution policy, setup (ACC or full coupled gallery), steps, repeats, and artifact directory. Run one lane per process. JSON contains configuration/provenance, setup/cold/warm samples, throughput and native/canonical numerical summaries; NPZ provides full compared arrays. Use existing native gallery injection/configuration APIs.

- [x] Write failing timing/synchronization and argument-validation tests before the harness.
- [x] Implement one lane per process; initialize once and reset from the same owned state for repetitions, with explicit synchronization and initial-state fingerprint validation.
- [x] Run native ACC stock NumPy/JAX and full global coupled stock NumPy/JAX, five or more warm replicates and two fresh processes per backend.
- [x] Compare stock native prognostic/canonical fields, masks, finite state and continuation/output; use independent source-budget tests to protect physics.
- [x] Run exact-fork coupled JAX-native with forced host and scanned JAX using identical setup; compare full state/canonical outputs within that distribution.
- [x] Inspect repeat distributions, compute median/spread and startup break-even; do not mix versions when attributing orchestration benefits.
- [x] If coupled stock JAX benefits are repeatable above uncertainty, update automatic native selection to JAX, retain explicit NumPy, and add default-selection tests. Otherwise document the measured reason for retaining NumPy.
- [x] Save reproducible summarized JSON and interpretation with exact commands, machine/dependency provenance and limitations.

### Task 3: Integrate, verify and deliver

**Files:** appropriate existing docs including `docs/veris-veros-jcm-validation.rst`, `DEPENDENCIES.md`, `PROGRESS.md`, this plan/spec; API and CI contracts if needed.

**Interfaces:** standard Black/flake8/mypy/pytest tools under the verified stock prefix; separate exact-fork `veros_autodiff` process. Existing public output/state APIs remain owners.

- [x] Update relevant docs and configuration signatures; document stock host/JAX versus fork scanned semantics and default decision.
- [x] Obtain independent whole-diff scientific/API/benchmark review, reproduce and fix material findings.
- [x] Run Black, strict repository flake8, mypy, compileall, whitespace, focused regression checks, fast/full stock suites, separate exact-fork AD and output smoke.
- [x] Verify finalized source hashes match benchmark/validation evidence; remeasure if product changes affect the measured path.
- [x] Update PROGRESS and plan with counts, timings, failed approaches and limitations.
- [x] Commit intended paths only after the full suite passes; authenticate Git with GitHub CLI, push `feat/veris-seaice`, and compare local/remote SHA.

## Execution evidence

- Baseline Python3.14 scipy aborted importing an optional native dependency. Existing temporary overlay skeletons do not contain package files. Python3.12 installed stack has older JAX0.10.2 and the fork: baseline fast13failed/32errors from mixed backend locks/metadata; mypy2 JAX-config errors; Black323 and repositoryflake8zero pass. These are baseline/environment evidence, not stock acceptance.
- Design approved by user on2026-11-03; no additional plan-approval stop is needed after their explicit instruction to proceed. Native execution preserves the user's current branch requirement.

- Final validation: fullstock2661/fast1701/exactforkAD17/focused24, zero failures/errors/skips; Black331/mypy331/repositoryflake8zero/compileall/strictSphinx/diff pass. Combined branch coverage 92.993125331 percent; new host-state module100 percent. All181production hashes match seven final benchmark lanes. NineNetCDFs reopen;68canonical fields and4native endpoint snapshots match exactly. Raw stock/AD/host-JAX coverage SHA256 values are preserved in the result JSON.
- Independent whole-diff review found no blocking issue. Deferred minor: comparison CLI does not automatically check every source/device/thread provenance field; current evidence was independently verified to match. GPU/long climate behavior is unmeasured, and stock full scans remain outside the approved focused design.
- Failed approaches: scipy3.14 native import abort and old empty temporary overlays; initial nested-kernel cache check was insufficient; copying native JAX metadata invalidated real root kernels (ACC11.59s), corrected by owned host state/value-based configuration keys; old source-format literals and home asset locks corrected; incomplete native deepcopy test double corrected using its existing native variables property, with assertions retained. No numerical tolerances or model source were changed.

- Delivery verified: source commit `d265173e120b905caecaa32d87649e296e047afe` and benchmark/evidence commit `092a2925a6047576f2786541fd28fca752cc8a64` pushed to `origin/feat/veris-seaice` using the GitHub CLI credential helper. `gh api` returned the exact benchmark commit SHA. This final receipt is a documentation-only follow-up; the final handoff verifies its remote SHA.
