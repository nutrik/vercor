# Adapter Retirement Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans for the main change,
> delegate independent documentation cleanup, and obtain a fresh whole-change
> review before delivery. Steps use checkbox syntax for tracking.

**Goal:** Delete the unwanted host atmosphere integration completely and deliver
the validated change on a separate branch.

**Architecture:** Delete adapter owners and their exclusive consumers. Update
existing exact API/gallery contracts; preserve shared arithmetic and all other
component implementations without introducing compatibility layers.

**Tech Stack:** Python, JAX, pytest, Black, flake8, mypy, Sphinx, GitHub CLI.

**Spec:** docs/superpowers/specs/2026-11-10-adapter-retirement-design.md

## Global Constraints

- Work on a separate feature branch.
- Remove the retired integration from tracked code, dependencies, docs and tests.
- Preserve shared kernels, retained adapter behavior, native packages and history.
- Full tests and static checks must pass before committing and pushing.
- Use the `scipy` Conda environment as required by AGENTS.md.

## Review Focus

- Lightweight public imports and retained lazy factories still work.
- Exact manifests and signatures describe the remaining supported API.
- CLI discovery, copying and installed artifacts contain the supported gallery.
- Generic host/backend and payload contracts retain independent coverage.
- Historical document edits do not misrepresent benchmark provenance.

### Task 1: Retire the integration and all exclusive consumers

**Files:** vercor/setups/__init__.py, config.py, retired external modules and
gallery; existing mixed API/runtime/setup/output/distribution tests and public
signature fixture; .github/workflows/python-package.yml; README.md, DESIGN.md,
DEPENDENCIES.md, CHANGELOG.md, PROGRESS.md and affected docs.

**Interfaces:** The setup facade exposes only retained components. Gallery
discovery and distribution contents use the same supported catalog. Generic
runtime/output arithmetic remains unchanged.

- [x] Update exact public manifests/signatures/gallery contracts; record RED.
- [x] Delete integration modules/configuration/gallery and exclusive tests.
- [x] Remove CI-only dependencies and obsolete cases from shared tests.
- [x] Clean current/historical docs and dependency graph; refresh archive hashes.
- [x] Run focused and fast tests, formatting, strict lint, mypy, compileall,
  strict Sphinx, full tests, branch coverage and distribution checks.
- [x] Obtain independent source/test/documentation review and address findings.
- [x] Update this record and the bounded progress log with verified results.
- [x] Prepare intended-file commit, gh-authenticated push and associated pull
  request; record the final remote SHA, PR and clean tree in the external receipt.

## Execution notes

- Initial checkout: clean `main`, source commit `d6ba4eb`.
- The required initial `scipy` fast suite aborted in all four workers while
  importing the retired integration's native dependency. Its output is retained
  in `/private/tmp/vercor-removal-baseline.log`; it is not passing evidence.
- A branch creation attempt required the sandbox's authorized Git escalation;
  branch creation then succeeded without changing the source tree.
- Main source/test implementation: @root; independent test cleanup: @tests;
  independent documentation cleanup: @documentation.
- RED: exact gallery and signature contracts failed before deletion (two
  expected failures). Final supported-interpreter numerical/lifecycle/archive/
  signature focus passes 40 cases. Generic host/payload migration passes five
  targeted cases; documentation fast checks pass 21, including strict Sphinx.
- The user approved existing Python 3.12 runtime validation while retaining
  `scipy` for static checks. Temporary overlays provide JAX/JAXlib 0.11.2,
  Flax 0.12.10, JCM 3.0.0 and Veris 0.2.1 at its released commit; stock Veros
  1.7.0 and exact fork 7a8c964cf00b5aa0713c995edd760643a431b3c9 use separate
  process lanes. Installed packages are unchanged. The first runtime attempt
  exposed an incompatible existing Flax 0.12.8; its collection failures are
  excluded from final evidence, and the corrected stack passed the JCM probe.
- Review removed two private helpers whose only production callers belonged
  to the deleted integration, together with seven exclusive test cases. Existing
  general component output selection remains covered independently. Review
  also corrected historical subset provenance and two documentation sentences.
  The final independent review has no findings or declined behaviors.
- Final validation uses a frozen manifest of 324 tracked Python files. The
  earlier autodiff process was replaced after the last helper deletion so both
  marker lanes and their coverage describe the final source tree.

## Final validation

- Full stock: 2612/2612; exact-fork autodiff: 22/22; fast stock: 1726/1726;
  final wheel/sdist installation contracts: 29/29. Zero failures, errors or skips;
  both full marker lanes account for all 2634 cases.
- Combined coverage with branch measurement: 93.99255%; statements 95.61782%,
  branches 86.66307%. All 324 Python hashes match the frozen final source.
- Black and mypy pass 311 source/test files; strict repository flake8 is zero;
  compileall, strict Sphinx and whitespace checks pass. Black emits its known
  interpreter/target-version safety-parse advisory without changing files.
- Both distributions have 13 gallery setups, with no retired integration or
  exclusive dependency references in member names or bytes. The full maintained
  tracked text scan also has zero matches. Exact artifact hashes and per-lane
  counts are in `/private/tmp/vercor-retirement-final-verification.json`.
- The user also requested an associated pull request. Git delivery is authorized;
  its final SHA/URL and clean-tree result are recorded externally after mutation.
