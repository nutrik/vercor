# Gallery Case Names Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans for inline execution
> and request an independent whole-change review before delivery.

**Goal:** Remove `run_` from the twelve prefixed gallery filenames, adapting
imports, CLI examples, tests and current documentation, then commit and push.

**Architecture:** Keep the existing filename-based discovery and exclusive copy
implementation. Rename the modules directly without aliases or changes to their
physics, configuration, or keyword-only `run_setup` entry points.

**Tech Stack:** Python, Click, pytest, Black, flake8, mypy, Sphinx, Git/GitHub.

**Spec:** The user's explicit rename request and repository `AGENTS.md`.

## Global constraints

- Work in the current clean checkout on `feat/veris-seaice`, as requested.
- Preserve gallery bodies and existing external-template naming semantics.
- Preserve historical progress, plans and measured-results provenance.
- Full tests and requested static gates must pass before commit and push.
- Verify the remote branch SHA after the push.

## Review focus

- Every public gallery module imports under its new name.
- CLI listing and copying accept both new stems and `.py` filenames.
- Existing destination protection and external-template discovery still work.
- Wheel and sdist include the renamed scripts; copied scripts remain runnable.
- Current docs, benchmark imports and architecture checks reference real files.

### Task 1: Rename the gallery cases and deliver

**Files:** Rename `vercor/setups/gallery/run_*.py`; modify `vercor/cli.py`,
gallery consumers under `tests/` and `benchmarks/`, current gallery/API/researcher
documentation, `README.md`, `DESIGN.md`, `DEPENDENCIES.md`, and `PROGRESS.md`.

**Interfaces:** Existing `run_setup(*, loglevel, float_type)` signatures and CLI
commands stay the same; case names and Python module paths lose `run_`.

- [x] Update existing CLI test names and copy expectations first. Include all
  fourteen public gallery modules in the existing entry-point contract test.
- [x] Run focused listing/copy tests and confirm failure for unavailable names.
- [x] Rename twelve files, update live references and preserve architecture
  assertions that previously selected examples by the `run_` prefix.
- [x] Run focused tests, Black, strict flake8, mypy, compileall, strict Sphinx,
  fast/full stock suites, and the separate exact-fork autodiff lane.
- [x] Obtain independent review; resolve findings and record checks/failures.
**Delivery:** Commit only intended changes, push `feat/veris-seaice`, and verify
the remote SHA; the final handoff records that SHA.

## Execution notes

- Baseline: clean branch at `55fcb13a3ab201aa1992264f35c0e12dbd97679d`.
- Initial `conda run -n scipy python -m pytest tests/ -q --fast` aborted in
  native OpenMP/Python 3.14 initialization before collection. Use the existing
  isolated scipy312/Veros overlays for runtime tests; leave environments intact.

- CLI RED3/GREEN verified. Review found a vacuous `run_*.py` test selector;
  a nonempty assertion reproduced RED1, then public-file selection restored
  coverage of all gallery examples. Independent final review: no findings.
- Initial suite detected the existing 180-line progress bound after adding one
  line. Retained all prior records and attached the new status to an existing
  entry; focused architecture tests pass.
- The additional Sphinx nitpicky (`-n`) mode reports 398 pre-existing unresolved
  type references. The repository-standard `-W --keep-going -b html` build
  passes; no unrelated reference-policy changes made.

- Final stock2745/fast1776/exact-forkAD18 pass, zero failures/errors/skips.
  Stock coverage with branch collection: 92.0106%. Black/mypy331, strict
  repository flake8 zero, compileall, standard strict Sphinx, artifact smokes,
  and diff check pass. All331 source/test hashes match the final manifest.
