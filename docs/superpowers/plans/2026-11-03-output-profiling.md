# Output Profiling Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans for implementation;
> obtain one independent whole-change review before delivery.

**Goal:** Reduce measured output accumulation overhead and safely consolidate redundant tests.

**Architecture:** Existing host schema preparation feeds tuple arrays into one
pure compiled accumulation kernel; scanned execution reuses that arithmetic.

**Tech Stack:** Python 3.12 scipy312, JAX 0.11.2, XProf, Perfetto,
TensorBoard, pytest, Black, flake8, mypy, GitHub.

**Spec:** `docs/superpowers/specs/2026-11-03-output-profiling-design.md`

## Global constraints

- Keep the user's current `feat/veris-seaice` branch and existing architecture.
- Preserve scientific values, masks, dtype, native endpoints and AD contracts.
- Keep schema/count validation and field-specific errors.
- Commit only after full tests pass; push and verify the remote SHA.

## Review focus

- Mixed field shapes, sample axes, and already-reduced inputs.
- Missing/infinite samples, zero support and inactive gradients.
- Changing metadata, shape, dtype, counts and coordinates.
- Bounded storage and executable reuse across repeated frames.
- Host/mixed speed versus already compiled scanned execution and cold cost.

## Tasks

- [x] Capture baseline cold/warm output timings, arrays and real traces.
- [x] Add numerical/AD regression tests for the batched numeric interface; observe RED.
- [x] Separate metadata preparation from arithmetic, batch accumulation, observe GREEN.
- [x] Repeat timings/traces and compare all output arrays; keep only measured improvements.
- [x] Consolidate only redundant basic cases with a retained-assertion ownership map.
- [x] Inspect traces with XProf, Perfetto and TensorBoard; retain reproducible evidence.
- [x] Obtain independent whole-change review; resolve material findings.
- [x] Run formatting, strict lint, type checking, fast/full tests and separate native AD.
- [x] Update PROGRESS and prepare the validated intended paths for Git delivery.
- Delivery: commit, push and verify the remote SHA; the external final receipt
  and final handoff record the delivered commit without a self-referential SHA.

## Execution ledger

- Baseline `fd137c3`; clean requested branch, GitHub connector confirms remote branch.
- Ruling: work in place because the user explicitly requires the current branch.
- Ruling: this is explicitly authorized maintenance with fixed behavior; choose
  the measured implementation without introducing a design-approval pause.
- Orientation `scipy` Python 3.14 aborts native workers before tests. Reuse
  documented scipy312 with `/private/tmp/vercor-veros170-stack` precedence;
  native asset directory `/private/tmp/vercor-veros170-assets`.
- Installed profiling tools only into `/private/tmp/vercor-profile-tools`.
- Independent tasks: test deduplication audit and read-only performance audit;
  root owns numeric implementation, profiling, integration and delivery.
- Failed benchmark setup: `DTypePolicy.apply` does not exist; corrected to the
  documented process-wide JAX precision switch before array creation.
- RED3 expected missing-kernel failures; focused GREEN66. Independent review
  found no Critical/Important/Minor issues; all four deleted-case owners verified.
- Baseline/current 32-field24-frame warm measurements: 21.288/8.894ms point,
  25.101/8.664ms history, 85.599/59.120ms reduced. Isolated final repeat:
  22.170/9.057ms,23.076/8.631ms,85.920/58.511ms. Both pairs preserve246
  arrays byte-for-byte. Startup cost increases; compiled control has no clear gain.
- Perfetto executable launches2304->24 ordinary/history and8448->5400 reduced.
  XProf converts all six captures; TensorBoard Profile plugin lists all runs
  and serves both point overviews; point warm timelines show no compile events.
- Failed approaches: automatic Perfetto cache tried unwritable home directory;
  downloaded the manifest-pinned executable to /private/tmp, verified SHA256
  and passed its explicit bin_path. Initial Sphinx underline warning corrected.
  Two mypy issues corrected without changing numeric arithmetic. No installed
  environment, numerical tolerance or native model implementation was changed.
- First fast run:1757 passed and1 documentation-bound failure because adding a
  progress entry made the active log181lines. Folded an existing continuation
  without deleting content; focused bound/kernel checks4passed. Full stock
  coverage follows the corrected180-line log. Strict Sphinx then passes.
- Final stock2715/fast1758/exact-fork18 pass, failures/errors/skips0. The
  four-file CI AD command covers17 cases; the additional marked native scanned
  output case passes separately. JUnit/collection set comparison accounts for
  all2733 cases without overlap. Combined branch coverage92.59640102827764%;
  all331source/test hashes match. Black331/mypy331/repositoryflake8zero,
  compileall/strictSphinx/diff pass. Settled-source isolated repeat confirms
  host warm32-field medians8.915/8.652/59.092ms; original comparisons retained.
- Raw traces, tools responses, reproducible scripts, source hashes, coverage
  files and detailed logs live in /private/tmp/vercor-output-profile. The
  machine-readable repository report includes final validation and source hashes.
