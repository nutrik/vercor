# Veros Sub-step Logging Removal Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans for inline execution
> and obtain an independent whole-change review before delivery.

**Goal:** Remove Veros adapter sub-step logging, adapt tests and documentation,
then validate, commit and push on `feat/veris-seaice`.

**Architecture:** Delete the logger argument and log call from the private
native-step helper and its sole current caller. Retain native step order,
post-step sampling, interval means and endpoint ownership.

**Tech Stack:** Python, JAX, Veros, pytest, Black, flake8, mypy, Sphinx, GitHub.

**Spec:** The user's explicit removal request and repository `AGENTS.md`.

## Global constraints

- Work on the existing `feat/veris-seaice` branch.
- Retain coupler progress reporting and native scientific behavior.
- Full tests and static checks must pass before commit and push.
- Verify the pushed remote SHA.

## Review focus

- Host and JAX adapter execution with a supplied logger.
- One and multiple native steps; sample means and continuation endpoints.
- Remove logger plumbing without introducing a replacement logging policy.
- Preserve zero-substep validation and native exception propagation.
- Current documentation describes the reduced progress output.

### Task 1: Remove logging and deliver

**Files:** `vercor/setups/_external/veros_state.py`, `veros_runtime.py`,
`tests/test_external_components_coverage.py`, `tests/test_veros_temporal_surfaces.py`,
`DESIGN.md`, `CHANGELOG.md`, and `PROGRESS.md`.

**Interfaces:** `advance_veros_substeps` retains its state, step function,
substep count, sample callback and return value; remove its `logger` keyword.

- [x] Adapt existing host/JAX internal-forcing and native temporal tests first.
- [x] Observe five expected host failures from emitted sub-step logs.
- [x] Remove the helper logger parameter, log call and caller keyword.
- [x] Verify focused tests, fast/full suites and separate native JAX tests.
- [x] Run Black, strict repository flake8, mypy, compileall, Sphinx and diff checks.
- [x] Review the change independently; no findings. Final evidence follows.
**Delivery:** Commit intended paths, push the requested branch and verify the
remote SHA; the final handoff and external receipt record the delivered commit.

## Execution notes

- Baseline `scipy` Python 3.14 native workers aborted before tests ran.
  Runtime checks use the existing isolated `scipy312` interpreter with
  `/private/tmp/vercor-veros170-stack` and `/private/tmp/vercor-veros170-assets`.
  Installed environments remain unchanged.
- Initial fake-host cases used `time=None`, which intentionally skips host
  advancement. Corrected the test context to a real timestamp and reran against
  the original logging implementation: all five host cases fail on log output;
  the two JAX adapter cases pass. The test execution annotation uses the native
  `Literal["host", "jax"]` contract to satisfy mypy.
- Focused fast tests: 30/30. Complete stock fast suite: 1,779/1,779.
  Native JAX temporal tests: 3/3, including JVP/VJP and continuation.
- The old `/private/tmp/vercor-temporal-ad` overlay now resolves to stock
  Veros and lacks native copy/safe-sqrt capabilities. Native JAX checks use a
  fresh temporary checkout at the approved fork commit
  `7a8c964cf00b5aa0713c995edd760643a431b3c9` instead; installed packages unchanged.
- Black/mypy check all 331 source/test files; strict repository flake8 reports
  zero findings. Compileall, strict Sphinx and whitespace checks pass. Black
  retains its existing Python 3.14/target-Python-3.15 safety-parse advisory.
- Detailed logs and JUnit reports: `/private/tmp/vercor-substep-*`.
- Full stock suite: 2,748/2,748, no failures/errors/skips. Stock branch coverage
  is 92.00788076066472%; all 331 source/test hashes remain unchanged.
  The separate 18-case native autodiff acceptance suite passes all 18 cases,
  with no failures/errors/skips. Native third-party future warnings remain.
