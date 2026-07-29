# Task 1 report: packaged setup gallery relocation

## Implementation

Moved all eleven runnable setup-gallery files from `examples/` to
`vercor/setups/gallery/` with identical file contents. The old top-level
`examples/` directory is absent. The behavioral tests now import and inspect
the packaged gallery modules. The CI quality job checks `vercor` and `tests`
only for Black, mypy, and compileall.

The fast suite also exposed the executable private-module inventory and two
live source scans outside the brief's enumerated paths. They were updated to
refer to `vercor.setups.gallery`, so the relocated modules remain covered.

## Files changed

- Moved `examples/__init__.py` and ten setup scripts to `vercor/setups/gallery/`.
- Updated `.github/workflows/python-package.yml`.
- Updated gallery references in `tests/test_jcm_example.py`,
  `tests/test_runtime_run.py`, `tests/test_example_jax_helpers.py`,
  `tests/test_api_boundaries.py`, `tests/test_runtime_state.py`,
  `tests/test_setup_agnostic_api.py`, `tests/test_distribution_boundaries.py`,
  and `tests/test_v0_4_public_api.py`.
- Updated the private-module inventory in `docs/api-architecture-review.md`.

## TDD evidence

RED (before the moves):

```text
conda run -n scipy pytest tests/test_jcm_example.py tests/test_example_jax_helpers.py tests/test_api_boundaries.py tests/test_runtime_state.py tests/test_setup_agnostic_api.py tests/test_runtime_run.py::test_runtime_profile_harness_exposes_cli_entrypoint -q --tb=short
```

This failed as intended: `ModuleNotFoundError: No module named
'vercor.setups.gallery'`, plus file-not-found failures for the future gallery
paths.

GREEN (after moves and CI updates):

```text
conda run -n scipy pytest tests/test_jcm_example.py tests/test_example_jax_helpers.py tests/test_api_boundaries.py tests/test_runtime_state.py tests/test_setup_agnostic_api.py tests/test_runtime_run.py::test_runtime_profile_harness_exposes_cli_entrypoint tests/test_distribution_boundaries.py::test_ci_quality_job_enforces_static_full_and_coverage_gates -q --tb=short
```

Passed (exit 0).

## Verification

- `conda run -n scipy python -m black --check vercor tests` — passed (241 files unchanged; known Python 3.13/target-3.15 safety-parse advisory).
- `conda run -n scipy python -m mypy vercor tests` — passed: no issues in 241 source files.
- `conda run -n scipy python -m compileall -q vercor tests` — passed.
- `conda run -n scipy pytest tests/ -q --fast --tb=short` — passed (exit 0).
- `git diff --check` — passed.
- `rg --files examples` returned no paths; the `examples/` directory is absent.
- SHA-256 comparison against `HEAD:examples/<file>` confirmed every moved script is byte-identical.

## Self-review

Checked the moved script hashes, all changed test/workflow paths, the stale
module inventory, absence of `examples/`, and whitespace. No setup logic was
changed.

## Concerns

None. The Black Python-version advisory is pre-existing environment behavior;
Black exited successfully.
