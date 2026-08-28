# VerCOR Pre-1.0 Versioning Correction Implementation Record

**Date:** 2026-07-15
**Status:** Repository migration implemented; fresh artifact verification is a
separate follow-up.

## Goal

Correct every repository-owned VerCOR release label and artifact boundary to
the approved pre-1.0 sequence ending at `0.4.0a1`, without changing physics or
runtime behavior.

## Constraints

- Preserve external dependency versions, GitHub Action revisions, schema
  versions, dates, hashes, and independently versioned plugin releases.
- Preserve the pinned historical source SHA and normalize its package metadata
  only in the temporary extracted build tree.
- Keep the numerical interpolator variable unchanged.
- Do not rewrite history, tag, push, publish, upload artifacts, or create a
  hosted release.
- Use tests first: capture the focused policy RED before repository migration,
  then commit only the complete GREEN state.

## Completed migration unit

- Added `tests/test_versioning_policy.py` with current-metadata and exhaustive
  repository-label contracts. Its path helper scans cached and intended
  untracked repository text while ignoring deleted paths.
- Renamed the migration guide, architecture design and plan, current signature
  contract, frozen pre-redesign API contract, historical plugin fixture, and current
  test modules to their corrected paths.
- Set package and current artifact identity to `0.4.0a1`.
- Renamed CI plugin lanes to `native-v0.4` and
  `historical-pre-redesign-artifact` while preserving GitHub Action revisions.
- Renamed the frozen plugin distribution and package to the legacy identity and
  set its earlier-package dependency interval.
- Normalized the pinned historical wheel build to the pre-redesign version after archive
  extraction while preserving its SHA, exports, and signatures.
- Updated source/test identifiers, executable contracts, documentation, release
  commands, historical progress, and every renamed-path cross-reference.
- Removed stale candidate hash blocks instead of relabeling their evidence.
- Refreshed the progress-archive checksum after historical normalization.

## Verification contract

The migration is complete only when these commands pass with the direct
`scipy` environment interpreter if the Conda launcher is unavailable:

```bash
python -m pytest tests/test_versioning_policy.py tests/test_api_architecture_review.py tests/test_v0_4_compatibility_baseline.py tests/test_distribution_boundaries.py -q --tb=short
python -m pytest tests/ -q --fast --tb=short
git diff --check
```

Fresh build, installed-artifact, hash, and publication evidence is intentionally
outside this migration unit.
