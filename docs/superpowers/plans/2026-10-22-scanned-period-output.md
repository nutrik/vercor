# Scanned Period Output Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Accumulate numeric period output inside boundary-to-boundary JAX scans.

**Architecture:** Preserve the existing runtime step and output session owners.
An optional numeric provider contract supplies fixed-schema frames to a private
output scan executor; ordinary providers and host/custom backends retain their
current sampling semantics.

**Tech Stack:** Python, JAX, frozen PyTrees, h5netcdf, pytest, mypy, Black, flake8.

**Spec:** `docs/superpowers/specs/2026-10-22-scanned-period-output-design.md`

## Global Constraints

- Work on current `feat/veris-seaice`; preserve the dirty audit PROGRESS entry.
- No physics, timestep, exchange/mask ownership or cancellation-policy changes.
- No new runtime dependency, trajectory storage or alternate output coordinator.
- Full suite passes before commit; push through GitHub CLI after acceptance.

## Review Focus

- First actual post-step sample must remain inside the first full scan.
- Short final windows and mixed cadence resets must retain the correct counts.
- Third-party host providers must never be speculatively traced.
- Native providers must extract current payloads and dynamic Veros time slots.
- Output-free AD, interrupted runs and repeated runs must remain isolated.

### Task 1: Pin boundary execution and provider semantics

**Files:** Create `tests/test_scanned_period_output.py`; modify output API tests
and public signature/manifest fixtures when adding the optional contracts.

**Interfaces:** Tests consume public Coupler/output declarations and observe
actual executor invocations, output files and final RunState values.

- [x] Add [4,4,2] scan-length and independent-period numeric mean regressions.
- [x] Add numeric-provider, ordinary-provider fallback, disabled-output,
  repeated-run, schema and sample-axis cases with explicit expected values.
- [x] Run focused tests in scipy and record expected feature-missing RED.

### Task 2: Compose numeric output with the shared JAX step

**Files:** Modify `vercor/output/__init__.py`, `vercor/output/_session.py`,
`vercor/_runtime/backends.py`, `vercor/_runtime/execution.py`; create
`vercor/output/_scan.py` with its tests already present from Task 1.

**Interfaces:** `JaxOutputProvider.sample_jax(context: JaxOutputContext) ->
OutputFrame`; private reusable JAX step and output scan builders; unchanged
public backend and ordinary provider contracts.

- [x] Add frozen numeric context and optional protocol with pure fixed-schema
  extraction; reuse one frame-to-accumulator validation helper.
- [x] Register the output session as a PyTree and abstractly prepare zero carry.
- [x] Share runtime step construction, compose checkpointed numeric accumulation
  inside scan, and split accelerated chunks at cadence/schedule boundaries.
- [x] Run new tests and existing workflow/output suites to GREEN.

### Task 3: Adapt bundled JAX output providers

**Files:** Modify `vercor/setups/_external/{jax_gcm,veris,veros}_output.py` and
their corresponding tests. Keep native host snapshots and timestamps intact.

**Interfaces:** Implement the optional protocol with common numeric frame
helpers; use JAX indexing for Veros tau.

- [x] Add failing pure numeric extraction/current-time-slot regressions.
- [x] Factor shared extraction and separate host time-coordinate formatting.
- [x] Run provider tests and real installed native JAX extraction checks.

### Task 4: Whole-change verification and delivery

**Files:** Update DESIGN, DEPENDENCIES, API inventory/signatures, output docs
and PROGRESS with exact behavior, fallback, evidence and limits.

- [x] Cover calendars, partial periods, cancellation, outer-transform rejection,
  independent native sample counts, JVP/VJP and repeated run isolation.
- [x] Run Black, flake8, mypy, compileall/diff, fast and full pytest, coverage,
  separate approved-fork native AD and a real coupled output smoke.
- [x] Benchmark warm scans with numeric accumulation and verify equal means.
- [x] Obtain independent review and resolve findings; repeat affected gates.
- Delivery after acceptance: commit intended files and push with gh-configured Git
  authentication and verify GitHub remote branch SHA equals local HEAD.

The user explicitly selected the implementation and delivery objective after
the in-chat design. Execute inline with specialist adapter/review assistance.

## Execution record

- User explicitly requested preserving the dirty PROGRESS.md in final delivery.
- Core feature RED: 7 failures/5 passes before production edits; numeric native
  RED: 6 missing-capability failures. Initial core GREEN12, independent acceptance7,
  native provider focus40, final added acceptance30 pass.
- Review fixed host coordinate precision (mutation reproduces both int64 overflow
  and float64 truncation) and per-step lifecycle hooks (transient-state RED,
  GREEN; hooks called once per actual step, no abstract/end-point duplicates).
- Ruling: arbitrary lifecycle validation remains an ordered concrete callback in
  the numeric scan; batching must not silently drop hooks. This preserves the
  contract and can limit accelerator speedups for payload-heavy custom hooks.
- Ruling: no extra worktree or design approval round: user selected the current
  feature branch and explicitly requested implementing the presented design.
- Python3.14 scipy full collection aborts in Torch/OpenMP; approved existing
  scipy312/native overlay used, environments unchanged. Stock and exact Veros
  fork run separately per CI; preliminary mixed-lane/sandbox errors were not
  physics regressions. Public fixture generated under Python3.12 to preserve
  established annotation formatting.
- Fresh native integrations: Veris+forcing, JCM+surface and Veros ACC JAX each
  execute output scan [2] and write one finite mean file; host sample forbidden.
  Reproduction: /private/tmp/vercor-scan-native-integration.py.
- Formatting/mypy304/maintainedlint0/compileall/diff/strictSphinx/build pass.
  Repository lint retains the known200 historical snippet findings.
- Final stock suite handle74276 and exact-forkAD handle81934 launched on reviewed
  source; raw coverage kept separately in /private/tmp/vercor-scan-{stock,ad-final}.coverage.
  Final JUnit/log evidence has matching stock-final/ad-final names. Combined90%
  gate, fresh fast gate, performance comparison, commit and gh push remain.

## Final acceptance

- Reviewed final source: full stock2401, exact-forkAD15 and fast1451 pass with
  zero failures/errors/skips (JUnit /private/tmp/vercor-scan-{stock,ad,fast}-final.xml).
- Raw lanes retained with copy-only combine; unchanged90% gate passes92.83511699619082%:
  8576/9048 statements and1660/1978 branches. Combined report and JSON:
  /private/tmp/vercor-scan-coverage-final.{log,json}.
- Black/mypy304, maintainedflake8zero, compileall/diff, strictSphinx, wheel/sdist,
  Twine and outside-checkout installed-wheel[2,2] scan/mean smoke pass. Wheel
  byte-identity covers180 production modules. Twine was isolated in a temporary
  target, leaving both Conda environments unchanged.
- Final rereview resolved both findings; no substantive open issue. Arbitrary
  author validation retains concrete ordered callbacks, so accelerator costs
  for payload-heavy hooks remain workload-dependent.
- Warm48step numeric-output benchmark, five-sample medians, CPU float64:
  host/scanned112.013/54.996ms (2x2),150.663/75.000ms (64x64), identical means,
  actual scanned lengths[24,24], filewrites excluded. This certifies only the
  stated synthetic workload. Reproduction and raw samples:
  /private/tmp/vercor_scan_accumulation_benchmark.py and
  /private/tmp/vercor-scan-accumulation-benchmark.{log,json}.
- Commit/push delivery is authorized and follows these gates; final Git history
  and independently verified remote ref are the authoritative delivery record.
