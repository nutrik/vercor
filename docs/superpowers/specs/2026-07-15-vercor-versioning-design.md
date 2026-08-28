# VerCOR Pre-Stable Versioning Correction Design

**Date:** 2026-07-15
**Status:** Implemented locally; fresh release-artifact verification completed
locally as a separate follow-up. Nothing was published.

## Purpose

VerCOR may publish only minor and patch releases until an externally supervised
and approved first stable release. This correction makes the live candidate
`0.4.0a1` and normalizes earlier repository evidence to the approved pre-stable
series.

This is a release-contract and historical-documentation correction. It does not
restore an older API, add compatibility adapters, or change numerical or
runtime behavior.

## Corrected release sequence

The repository described a supervised sequence of superseded releases followed
by the current `0.4.0a1` alpha. Current-architecture prose uses VerCOR 0.4, and
historical fixture evidence uses the pre-redesign baseline. Neutral
fixture-generation markers use descriptive names instead of release labels.

## Scope

The correction covers package metadata, source docstrings, tests, contract
filenames and contents, plugin fixtures, GitHub workflow artifacts and lanes,
release documentation, historical progress, and every version-bearing path or
cross-reference.

Renamed Python modules and fixture packages leave no imports or build paths
pointing at their former names. Renamed Markdown and JSON files leave no stale
links or executable-contract paths.

## Boundaries

Only VerCOR's own release and API-generation labels are in scope. External and
independent identifiers remain unchanged, including Python and dependency
versions, GitHub Action revisions, the Apache License revision, schema versions,
dates, hashes, numerical values, and independently versioned plugin releases.

The checked-out repository is corrected without rewriting Git history, moving
tags, pushing, publishing packages, uploading artifacts, or creating a hosted
release.

## Artifact evidence

Candidate hashes produced before the metadata correction are obsolete and are
not relabeled. The separate release-verification follow-up built fresh corrected
wheel and source-distribution artifacts, passed the installed-artifact gates,
and recorded the exact local commands and SHA-256 evidence in `PROGRESS.md`.
No artifact was uploaded or published, and no tag or hosted release was created.

The historical compatibility fixture used an earlier-package dependency range
and a frozen pre-redesign public API baseline. The pinned source SHA is
unchanged; its project version is normalized only in the temporary extracted
build tree.

## Testing strategy

An executable repository policy checks intended repository text and existing
tracked paths for disallowed release labels while exempting external action
revisions and the numerical interpolator variable. Focused contracts validate
the current metadata, architecture documentation, frozen baseline, plugin
fixture, and distribution workflow. The full fast suite guards imports and
runtime behavior.

Fresh artifact and release verification was completed locally as the separate
follow-up recorded in `PROGRESS.md`; publication remains outside this task.
