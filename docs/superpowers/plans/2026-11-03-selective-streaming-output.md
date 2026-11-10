# Selective Streaming Output Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete-by-default component period output with selective extraction and bounded streaming means.

**Architecture:** Existing output contexts carry selection; the shared immutable accumulator handles ordinary samples and reduced sums/counts. Bundled providers unite native and canonical state.

**Tech Stack:** Python, JAX, NumPy, h5netcdf, pytest, Black, flake8, mypy, GitHub CLI; isolated scipy312 native stack.

**Spec:** `docs/superpowers/specs/2026-11-03-selective-streaming-output-design.md`

## Global Constraints

- Work in existing `feat/veris-seaice`; installed environments remain unchanged.
- Output remains opt-in, and empty variables means all; None in step output policy means disabled.
- Preserve endpoint snapshots, native physics, exchange means, backend behavior and finite counts.
- No trajectory; only selected arrays participate in output extraction/reduction.
- Full and fast checks before commits; GitHub CLI authentication and remote-SHA verification.
- User approved design and said "Approve and implement"; proceed in this session without another approval stop.

## Review Focus

- Input fields under climatological export transfer retain received values (Task 1).
- Unselected poisoned arrays/diagnostic callbacks are never accessed (Tasks 1-2).
- Implicit/repeated native dimensions and native/canonical collisions retain correct coordinates and meaning (Task 2).
- Missing native samples merge exact finite sums/counts, including zero counts (Tasks 1 and 3).
- Repeated output on/off runs preserve immutable setup and endpoint continuation (Task 3).

---

### Task 1: Shared selection and reduced frame contracts

**Files:** `vercor/output/__init__.py`, `_dataset.py`, `_session.py`; `vercor/components/contexts.py`; `vercor/_runtime/driver.py`, `backends.py`; `tests/test_selective_period_output.py`; existing signature contracts.

**Interfaces:** Add default-empty `variables: tuple[str, ...]` to `OutputContext`/`JaxOutputContext`, default-None `output_variables: tuple[str, ...] | None` to `StepContext`, and optional `OutputFrame.reduction_counts`. Counted variable values are sums with same-shaped nonnegative integer counts. Reuse `_OutputAccumulator` for merging both frame kinds. Shared private helpers select available names and merge canonical runtime variables without overriding native names.

- [x] Write failing tests for input-only default fields, pre-extraction selection and provider contexts on JAX/host/mixed/custom runs, counted-frame finite means/reset/schema failures, and disabled step output requests.
- [x] Run new tests; require failures from missing behavior before source changes.
- [x] Implement contracts and helper methods; sample default provider from all runtime state, applying export transfer only to outputs; propagate static run output policy through both drivers.
- [x] Run new and existing scanned/backend output tests; preserve constant PyTree leaf sizes and write-only mean evaluation.

### Task 2: Complete selective bundled native providers

**Files:** `vercor/setups/_external/veris_output.py`, `veros_output.py`, `jax_gcm_output.py`; `tests/test_veris_output.py`, `test_external_components_coverage.py`, new selective tests as appropriate.

**Interfaces:** Consume context selection and shared merge helpers from Task 1. Providers return all eligible native plus canonical runtime fields by default. Native variables win collisions. Veris intersects registry with initialized payload arrays. Veros gives repeated axes unique names and implicit axes index coordinates. JCM selects flattened native references before conversion/leading-time expansion.

- [x] Add failing native completeness and poisoned unselected extraction tests; update old intentional allowlist expectations explicitly.
- [x] Run these tests RED, then implement selection and completeness in existing provider owners.
- [x] Run native/provider/scanned output regression tests including dimensions, metadata and endpoint snapshots.

### Task 4: Review, documentation and delivery

**Files:** `DESIGN.md`, `DEPENDENCIES.md`, applicable researcher/API output docs, `PROGRESS.md`, this spec/plan and public signature expectations.

- [x] Update docs and deliberate public signature changes; obtain independent whole-diff numerical/performance review and fix findings.
- [x] Run Black, repository flake8, mypy, compileall and diff checks; run fast/full stock suites plus isolated exact-fork AD tests. Keep logs in `/private/tmp`.
- [x] Record counts, failures, limitations and requirement evidence in this plan/progress; verify final source matches checks.
- [ ] Commit intended paths after full suite passes. Authenticate Git using `gh`, push existing branch, and compare `gh api` remote SHA with local HEAD.

## Evidence

- Design approved 2026-11-03; builder review incorporated native/canonical union, run-specific native output policy and finite-count reduction.
- Initial scipy3.14 baseline aborted all workers. First scipy312 run mixed stock/AD and used unwritable native asset locks. Corrected isolated stock baseline: 1701 passed, 3 warnings, 258.63 s, with `VEROS_ASSET_DIR=/private/tmp/vercor-veros170-assets`, `PYTHONPATH=/private/tmp/vercor-veros170-stack:$PWD`, scipy312 interpreter, `--fast -m 'not veros_autodiff'`, four workers. Installed environments unchanged.

- Final-source regression repair: empty nodal-shape tuple avoids a native JCM typed-struct formatter's None-indexing error while leaving reshapes/channel extraction deferred. Focused native/temporal repair33 passed; integrated focused136 passed. Updated old allowlist/history expectations explicitly; numerical/unit/endpoint assertions retained.
- Independent final review reproduced and resolved two excess-work issues. Temperature-only Veris output bypasses albedo diagnosis. No deferred code findings; GPU throughput remains unmeasured.
- Full final stock2703 and isolated exact-fork AD18 passed, zero failures/errors/skips. Exact collected-node comparison proves2721/2721 tests covered across disjoint stock/AD lanes. All327 source/test SHA256 values match the final gate-start snapshot. Combined branch coverage92.52610684387676%; raw lane files/hashes preserved. Black327, repository flake8zero, mypy327, compileall, strictSphinx and diff checks passed. Fast precommit check1742 passed, zero failures/errors/skips. Machine evidence: /private/tmp/vercor-output-final-results.json; logs/XML/coverage/source hashes share that prefix.
- Failed approaches retained: scipy3.14 worker abort before tests; initial mixed stock/AD stack and home asset locks; None as native nodal-format sentinel; literal source-format factory test and previous native-history assertions. Installed model environments, physical constants and numerical tolerances were not changed.

- Delivery authorization is explicit: commit and push current feat/veris-seaice using GitHub CLI authentication. Remote starting SHA69d3a67581781ebff95e33bd8a34cb4d2fe05541 matches local base. The final handoff supplies the verified pushed SHA; no merge or PR is requested.
