# Physical coupling names implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans. Steps use checkbox syntax for tracking.

**Goal:** Complete the approved naming migration and publish the verified current branch.
**Architecture:** Keep the existing adapters/runtime and translate native names at their existing boundaries. Migrate physical field contracts and all their consumers atomically.
**Tech Stack:** Python, JAX, pytest, native Veris/Veros/JCM, Sphinx, GitHub CLI.
**Spec:** `docs/superpowers/specs/2026-10-21-coupling-names-design.md`.

## Global constraints

- Work on `feat/veris-seaice`; preserve/include existing `PROGRESS.md` changes.
- No numerical changes or compatibility aliases; retain native fields/layouts.
- Use a working Python 3.12 conda `scipy` lane; keep stock/fork Veros processes isolated.
- Commit only after full tests pass; push with GitHub CLI and verify remote SHA.

## Review focus

- A generic runtime component may still use arbitrary fields named `temperature`.
- Native atmospheric keys must survive exchange-key renaming unchanged.
- Full-column and lowest-level pressure must retain distinct physical ranks.
- Observed 10 m winds and unspecified-height phase temperature must stay distinct.
- User-supplied component names must reach routes, diagnostics and surface roles.

### Task 1: Physical field and adapter contracts

**Files:** `vercor/fields.py`, `vercor/recipes.py`, `vercor/setups/config.py`,
`vercor/setups/__init__.py`, bundled data/external factories and consumers;
existing mapper, forcing, numerical and gradient tests.
**Interfaces:** Produce the field/recipe/config names fixed by the spec; retain
native arrays and physical computations.

- [x] Update existing mapper tests first to assert new air keys, phase keyword,
  and full-column/lowest-level pressure shapes and literal pressure values.
- [x] Run focused tests; require failures at the missing new names/keyword.
- [x] Migrate shared recipe names, air keys, pressure fields, JCM concrete
  factory/config/owners, Veros forcing modes and CAMulator surface packer.
- [x] Update callers and existing tests without changing their physics assertions.
- [x] Run mapper, forcing, interchangeability and boundary tests; expect success.

### Task 2: Mask, gallery and documentation clarity

**Files:** `vercor/grid_masks.py`, `vercor/_runtime/surface_masks.py`,
`vercor/setups/gallery/*.py`, diagnostics, current docs, API snapshot,
`DEPENDENCIES.md`, existing gallery and mask tests.
**Interfaces:** Consume Task 1's names; retain arbitrary instance identities,
current topology, numerical masks and public import boundaries.

- [x] Update mask and gallery tests to require spelled-out helpers and custom
  component identities; run them before changing their production owners.
- [x] Spell out mask names; derive gallery identities from components and pass
  explicit surface-role names. Migrate related diagnostics and docs.
- [x] Refresh the explicit public manifest/signatures and dependency inventory.
- [x] Run focused masks/gallery/API/documentation tests; expect success.

### Task 3: Verify, review and deliver

**Files:** maintained source/tests/docs; `PROGRESS.md` and this plan.
**Interfaces:** Consume the completed naming refactor and preserve its full scope.

- [x] Run Black, strict maintained-source lint, mypy, compileall, whitespace,
  fast/full stock tests, isolated native AD, strict Sphinx, package/artifact checks.
- [x] Obtain a fresh independent review covering the spec and physical invariants;
  reproduce/fix findings and rerun affected checks.
- [x] Record exact evidence and environment limitations; audit every spec item.
- Delivery after recording verification: commit all intended changes including
  the original dirty progress entry, push `feat/veris-seaice` with GitHub CLI,
  and verify remote/local SHA equality. Git history and the GitHub branch ref
  are the authoritative evidence for this final post-commit step.

One commit is deliberate: field/config names and their consumers form one
inseparable refactor. The user's full-suite-before-commit rule takes precedence
over intermediate documentation or per-task commits.

## Verification record (2026-10-21)

Full stock 2295, fast 1368, exact-fork native AD 15 and installed-artifact 4
checks passed without failures/errors/skips. Combined coverage is 92.58985494%
(statements 94.62256722%, branches 83.24715615%); raw lane data is preserved.
Black, maintained lint, mypy (300 files), compileall, whitespace, strict Sphinx,
build and Twine passed. All 179 production modules match the final wheel.
Seven numerical modules are structurally identical after declared-name
normalization. Independent review and the scoped fixture rereview have no open
findings. The first full run exposed eight stale fixture expectations; their
keys/signatures were migrated while retaining values, shapes, tolerances and
gradient assertions. Evidence: `/private/tmp/vercor-coupling-names/`.
