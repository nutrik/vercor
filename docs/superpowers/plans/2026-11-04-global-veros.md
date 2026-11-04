# Global-only Veros Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans for inline execution
> and obtain an independent whole-change review before delivery.

**Goal:** Remove ACCSetup, migrate its checks to global_4deg, audit, validate,
commit and push on feat/veris-seaice.

**Architecture:** Retain the existing global setup and native state/runtime
owners. Remove the alternative setup configuration and derived forcing flag.
Benchmark ocean-only and coupled workloads both use the global ocean.

**Tech Stack:** Python, JAX, Veros, pytest, Black, flake8, mypy, Sphinx, GitHub.

**Spec:** docs/superpowers/specs/2026-11-04-global-veros-design.md

## Global Constraints

- Work in the current checkout and branch as explicitly requested.
- No native model source patches, physics changes, relaxed tolerances, numerical
  fudge factors, or reduced validation scope.
- No installed packages change.
- Full tests and static checks must pass before commit and push.

## Review Focus

- Former positional setup strings cannot silently select another setting.
- Both host backends receive the same global geometry and advance real steps.
- Ocean-only benchmark fields supply every required total-flux input, and
  its metadata cannot be compared with a coupled workload accidentally.
- Migrated gradients probe wet global cells and have finite, nontrivial
  finite-difference/JVP/VJP agreement across real three/twenty-day rollouts.
- Historical measurements are not relabeled as new global measurements.

### Task 1: Remove the alternative setup and migrate its owners

**Files:** vercor/setups/config.py; _external/veros_gcm_state.py,
veros_gcm.py, veros_setup_jax.py, veros_runtime.py; _veris.py and
gallery/veris_veros_jcm.py; native backend/float32 probes; existing Veros,
prognostic gallery and public signature tests; benchmarks/benchmark_veros.py
and compare_veros.py; current docs, DESIGN.md, DEPENDENCIES.md, CHANGELOG.md,
PROGRESS.md.

**Interfaces:** VerosConfig has no setup field. The class selector consumes
execution only. surface_forcing remains the sole forcing-mode selector.
Benchmark --case selects ocean/coupled and metadata records both case and
setup=global_4deg. Comparison requires matching case and physical metadata.

- [x] Add global-only configuration, forcing, benchmark, and migrated native
  regression tests; observe expected failures before implementation.
- [x] Remove ACC and redundant branches; migrate callers and benchmark.
- [x] Verify focused fast tests and global native/AD rollouts, resolving the
  first source of any failure without relaxing numerical checks.
- [x] Audit and update maintained signatures, docs, dependency graph and log.
- [x] Run Black, strict flake8, mypy, compileall, strict Sphinx, fast/full
  stock tests, exact-fork AD/native tests, branch coverage, and diff checks.
- [x] Obtain independent source/test review; resolve substantive findings.
- [x] Record final validation evidence and prepare the intended-file Git delivery.

**Delivery:** Commit only intended files, push and verify remote SHA and a clean
worktree after the source-bearing commit. The final external receipt records
these mutation results: `/private/tmp/vercor-global4deg-validation.json`.

## Execution Notes

Pre-flight: one bounded refactor; no cross-task interfaces. Explicit branch and
delivery authorization fix checkout and Git workflow. Baseline source tip:
74a817daa3749ba37530080dce32cd6e7b1c5857.

- RED: 10 expected failures covered removal of setup, the internal-forcing
  bypass, and benchmark case selection/attribution. Focused fast tests pass;
  native backend/API focus passes 35/35 after correcting an embedded-script
  f-string and keeping the bounded progress log at 180 lines.
- The unoverlaid scipy312 baseline included the unsupported stock AD lane
  and JAX0.10.2. Required validation uses the existing JAX0.11.2 stock overlay
  and separate exact-fork source, with marker lanes accounted independently.
  Native assets were copied into a writable temporary cache; original asset
  hashes are still verified and installed packages/native sources are unchanged.
- Gradient migration preserves three/twenty simulated days, complete global
  geometry, FD steps/tolerances and JVP/VJP checks. A weak Southern Ocean c_k
  objective failed FD; the active North Pacific column nearest180E30N has
  derivative -0.2800627363K per c_k and relative FD differences0.3075%,
  0.6733%,0.3991% at h=1e-3,1e-4,1e-5. No global smoothness claim is made.
  All16 owned-file AD cases pass, including the coupled five-day adjoint.
- Both stock NumPy/JAX global ocean benchmarks advance two1800-second steps,
  retain reproducible repeat hashes, and write five NetCDF files each.
  Comparisons cover141 canonical/native arrays per phase; backend roundoff
  differences are reported rather than treated as bitwise equivalence.
- Independent source and test review: no Critical, Important or Minor findings;
  no behaviors declined. Reviewer independently verified the saved gradient
  map, real global benchmark geometry/fluxes and historical attribution.
  Full stock/native coverage and final fast verification passed as recorded below.

## Final verification

- Full stock:2776/2776; exact-fork marked native/AD:22/22; fast stock:1807/1807.
  Every2798collected case is accounted for, with zero failures/errors/skips.
  The full marker lane includes the shared-output JAX parameter, velocity/SSH
  derivatives and both authentic float32 probes.
- Combined branch coverage:92.58844796419041%; all335source/test/benchmark
  hashes match the frozen validation snapshot.
- Black335files, mypy331files, repository flake8zero at120columns, compileall,
  strict Sphinx and whitespace checks pass. Independent review is clear.
- Native/API focus35/35 and both global benchmark/output smokes pass.
- Receipt, logs, JUnit and source hashes: `/private/tmp/vercor-global4deg-*`.
  Delivery is explicitly authorized on the current branch; the final receipt
  and user handoff record the committed/pushed SHA after those operations.

- Git delivery approval: automatic review initially rejected the commit twice
  by treating the earlier read-only request as authoritative. No history or
  remote mutation occurred. The user subsequently explicitly confirmed
  "Yes, commit and push"; delivery proceeds under that renewed authorization.
