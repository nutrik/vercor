# Linux CPU and CUDA Runtime Performance Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans task by task. Root implements; specialized agents inspect performance and final correctness.

**Goal:** Freshly profile and improve host/JAX execution on Linux CPUs and directly accessed CUDA 12 GPUs, then verify and push tested changes.

**Architecture:** Reuse existing runtime/regridding ownership and benchmarks. Fuse only measured eager numerical overhead, preserving all validation and AD contracts.

**Tech Stack:** uv `.venv`, Python 3.12, CUDA 12 JAX, pytest, XProf, Perfetto, TensorBoard.

**Spec:** `docs/superpowers/specs/2026-11-16-linux-runtime-performance-design.md`.

## Global Constraints

- Preserve physics, precision, error order, lifecycle boundaries and JVP/VJP.
- Use `perf/host-jax-runtime`; GPUs are outside SLURM.
- Keep warm synchronized measurements separate from setup/compilation/traces.
- Use three paired processes; target gain >=5%, no repeatable control loss >5%.
- Write regression tests before production changes; run full tests before commits.

## Review Focus

- GPU scalar validation can cause device-to-host synchronization despite small kernels.
- Fusion must preserve first error and static shape checks.
- Native optional packages have conflicting JAX dependency ranges.
- GPU and CPU numerical behavior must retain existing dtype and gradient policy.
- Repeated fixed-shape simulation steps must reuse compiled executables.

### Task 1: Environment, baseline and bottleneck evidence

**Files:** `.gitignore`, existing `benchmarks/benchmark_runtime.py`, ignored `test_logs/linux-runtime/`.
**Interfaces:** Current sixteen benchmark cases, explicit `JAX_PLATFORMS=cpu|cuda`.

- [x] Create `.venv` with uv and CUDA 12 JAX; verify direct GPU arithmetic.
- [x] Run baseline fast tests, record optional-model compatibility and isolate AD fork.
- [x] Archive baseline source; profile CPU/GPU cases across execution lanes and numerics.
- [x] Use XProf/Perfetto to select bounded optimization targets before code changes.

### Task 2: Implement and measure selected fusion/vectorization

**Files:** Existing numerical owner(s), corresponding test files, existing benchmark as needed.
**Interfaces:** Preserve public API and numerical safety boundaries.

- [x] Specify selected functions and tests from Task 1 evidence in the ledger.
- [x] Add failing value/error/AD/compile-reuse regressions for the selected mechanisms.
- [x] Implement minimal pure numerical fusion in its existing owner.
- [x] Run focused and fast tests; compare independent CPU/GPU pairs in both precisions.
- [x] Revert any candidate failing the performance/correctness gates; record failed approaches.

### Task 3: Reproducible report, verification and delivery

**Files:** Profiling report/results, `PROGRESS.md`, `DEPENDENCIES.md` if dependencies change.
**Interfaces:** Trace paths, provenance, benchmark values and explicit limitations.

- [x] Verify XProf conversion and TensorBoard Profile HTTP results; analyze Perfetto SQL.
- [x] Document before/after timing distributions, dispatch/synchronization, vectorization and limitations.
- [x] Run full stock and exact-fork AD suites, appropriate GPU tests, Black, strict Flake8, mypy, compileall, strict Sphinx and whitespace checks.
- [x] Obtain independent final review; resolve consequential findings with regression tests.
- [ ] Commit small tested units and push current branch; verify exact remote SHA and clean worktree.
