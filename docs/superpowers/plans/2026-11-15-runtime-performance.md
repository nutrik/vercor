# Host and JAX Performance Implementation Plan

> **For agentic workers:** Use `superpowers:executing-plans` task by task. Root implements; specialist agents inspect performance, tests and final correctness.

**Goal:** Reduce measured CPU overhead in both execution lanes without changing scientific or validation contracts.

**Architecture:** Keep current runtime, exchange and numerical-safety owners. Compile only count arithmetic and reuse only invocation-local immutable mask operands. Extend the existing profiler and developer benchmarks.

**Tech Stack:** Python 3.12, JAX, pytest, XProf, Perfetto, TensorBoard.

**Spec:** `docs/superpowers/specs/2026-11-15-runtime-performance-design.md`.

## Global Constraints

- Preserve physics, precision, eager/checkify error behavior, JVP/VJP and lifecycle order.
- Keep scan widths, output boundaries, custom driver identity checks and route remapping unchanged.
- Use isolated scipy312 tooling, stock and exact-fork AD processes; installed environments stay unchanged.
- Write failing tests first. Commit only after the full suites pass.
- Require three paired CPU comparisons, at least 5% target improvement in every pair, and no repeatable control regression above 5%.

## Review Focus

- Exhausted repeated plans must still report repetition before generic exhaustion.
- Count-kernel JIT staging must preserve eager and nested checkify failures.
- Route-specific topology and binary-source policy must follow raw-mask reuse.
- An invalid earlier remap must prevent later route calls.
- Output-enabled profiles use API invocation and fresh output directories.

### Task 1: Profiling interfaces and reproducible baseline

**Files:** existing gallery profiler and runtime-run tests; developer runtime benchmark and its new test file.

**Interfaces:** `build_slab_coupler(..., backend="jax")`; `profile_runtime(..., backend="jax", invocation=None)`; CLI `--backend host|jax`, `--invocation api|outer-jit`. `None` preserves JAX outer JIT and selects API for host. Result metadata records the resolved mode.

- [x] Establish compatible stock/AD baselines and archive baseline source.
- [x] Write and run failing parser, actual host/API, JAX/API, mode-validation and synchronization tests.
- [x] Extend the existing profiler and add small typed benchmark cases without a new runtime dependency.
- [x] Run focused tests and record provenance; keep traces separate from timings.

### Task 2: Constant-time expected-plan validation

**Files:** runtime backend owner and workflow execution tests.

**Interfaces:** preserve `_RuntimeDriverAdapter._expected_plan(plan) -> StepPlan`.

- [x] Add a failing long-chunk regression proving normal dispatch does not inspect consumed prefixes; retain real identity and completeness cases.
- [x] Return the current expected identity before error-path scans, after the existing type check.
- [x] Verify forged, reordered, repeated, invalid-type and exhausted plans, including repeated-last-plan precedence.
- [x] Run workflow tests and compare the isolated plan-validation benchmark.

### Task 3: Compiled numerical-check arithmetic

**Files:** numerical-safety owner and existing numerical-safety tests.

**Interfaces:** retain all existing callers and error-policy arguments. Private pure JIT kernels return scalar invalid counts; binary mask mode is static.

- [x] Add failing fixed-schema compilation/launch-reuse tests and behavioral precision/transform cases.
- [x] Keep shape validation eager; put mask comparison, broadcasting and reduction arithmetic in reusable count kernels, without owner strings.
- [x] Preserve `_require_zero` and callback placement; verify all nested checkify orders and later invalid values in scan/vmap.
- [x] Run numerical, remapping, dynamic-mask and AD regressions and compare count-check and complete-run benchmarks.

### Task 4: Evaluate invocation-local component mask reuse (rejected)

**Files:** exchange-mask owner and dynamic-mask exchange tests.

**Interfaces:** existing scalar coverage/dispatch behavior remains; private refresh plumbing may accept a lazily populated raw-mask mapping.

- [x] Add and watch failing resolver-count regressions; retain independent shared/self-route value tests after rejecting the optimization.
- [x] Trial lazy per-refresh resolution in first-reference order; reject it after isolated compiled-control regression and restore the original owner.
- [x] Verify changing masks, binary source coverage, topology, remap ordering/failure, JIT/JVP/VJP and unchanged received support.
- [x] Run exchange tests and paired refresh/complete-run benchmarks.

### Task 5: Evidence, full verification and delivery

- [x] Run three alternating paired comparisons in each precision, saving raw samples and every output leaf.
- [x] Discard candidates that fail the declared performance or correctness rule; document negative results.
- [x] Capture and analyze XSpace/Perfetto; verify XProf conversion and TensorBoard Profile loading; run short native ocean/coupled integration checks.
- [x] Update runtime profiling documentation, machine-readable results, PROGRESS and DEPENDENCIES.
- [x] Run Black, strict Flake8, mypy, compileall, fast/full stock and isolated exact-fork AD tests, combined branch coverage, strict Sphinx and API/distribution checks.
- [x] Obtain fresh whole-branch review; resolve consequential findings with RED/GREEN regressions.
- [x] Make small tested commits; verify branch, commit SHAs and clean tree; deliver local evidence paths and limitations.

## Final outcome

Retained constant-time plan checks and explicitly inline numerical-count kernels. Rejected raw-mask reuse after its compiled-control regression; no tracing-specific cache policy was introduced. Final stock2669/AD22/fast1783 pass, branch-enabled coverage94.05088%, Black/mypy320/strictlint0/compileall/strictSphinx/API/distribution/diff pass. All320 Python hashes remain unchanged through verification. All36000 saved comparisons are byte-identical. XProf14 captures, Perfetto and TensorBoard Profile endpoints are verified; independent whole-branch review is clear. Local branch/worktree retention is the approved delivery method.
