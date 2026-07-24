# Automated Release Deployment Implementation Plan

**Date:** 2026-07-24
**Status:** Final-review corrected

## Goal

Make an authorized `v*.*.*` tag push publish the exact VerCOR wheel and sdist
built and tested by CI, then publish a matching GitHub Release without
overwriting, skipping, or silently repairing public state.

## Final-review corrections

The first implementation draft assumed the publisher could generate its own
checksum manifest and call `gh release create` with both assets. Final review
identified two provenance/state-machine gaps:

1. A publisher-generated manifest cannot authenticate CI-produced bytes.
2. GitHub CLI creates a draft, uploads assets separately, and then publishes;
   interruption can therefore leave an authenticated exact-tag draft with
   zero, one, or two assets.

This corrected plan supersedes the earlier inline workflow examples. The final
implementation has a producer-issued manifest artifact, immutable action pins,
same-run draft/upload/publish transitions, post-PyPI verification, and a
separately authorized exact-run recovery procedure. An ordinary workflow rerun
must fail when PyPI or any exact-tag draft/published release already exists.

The release re-review adds draft-aware pre-tag enumeration and a canonical `https://uploads.github.com` request target
generated with a safely encoded asset name; bounded missing-file recovery polling
requires PyPI's final exact two-file producer-manifest state.
Both pre-tag transcripts accept release absence only after the same token successfully calls the
non-mutating Release notes-generation endpoint requiring `contents: write`, then enumerates
releases so the authenticated request can see drafts.

## Non-negotiable boundaries

- A tag matching `v*.*.*` is the only deployment trigger.
- The tag must equal `v` plus `project.version` from `pyproject.toml`.
- `build-artifacts` is the sole builder.
- `vercor-distributions` contains exactly the wheel and sdist.
- `vercor-release-manifest` separately contains the producer-issued
  `SHA256SUMS`.
- The publisher downloads and verifies both artifacts before installing
  mutable tooling.
- Twine 6.2.0 is installed as the exact version; pip is not upgraded in the
  release job.
- Every action reference is a full immutable commit SHA.
- Every checkout uses `persist-credentials: false`.
- The existing PyPA publisher pin and `secrets.PYPI_API_TOKEN` remain
  unchanged.
- The workflow retains top-level `contents: read`, gives only
  `publish-release` `contents: write`, and grants no `id-token` permission.
- The protected environment remains `release`.
- Per-tag concurrency uses `cancel-in-progress: false`.
- Repository administrators separately protect `v*.*.*` tags with a ruleset.
- `skip-existing`, overwrite/clobber, release deletion, and tag mutation are
  forbidden.
- No local verification step pushes, tags, publishes, runs a hosted workflow,
  or changes a GitHub Release.

## Immutable action references

Resolve official major-version refs read-only from their repositories and pin
the resulting commits:

- `actions/checkout@11d5960a326750d5838078e36cf38b85af677262`
  (major 4)
- `actions/setup-python@a26af69be951a213d495a4c3e4e4022e16d87065`
  (major 5)
- `actions/upload-artifact@ea165f8d65b6e75b540449e92b4886f43607fa02`
  (major 4)
- `actions/download-artifact@d3f86a106a0bac45b974a628896c90dbdf5c8093`
  (major 4)
- `codecov/codecov-action@0fb7174895f61a3b6b78fc075e0cd60383518dac`
  (major 5)
- `pypa/gh-action-pypi-publish@ba38be9e461d3875417946c167d0b5f3d385a247`

## Task 1: Define the executable contracts first

Update:

- `tests/test_distribution_boundaries.py`
- `tests/test_api_architecture_review.py`
- `tests/test_release_state_validator.py`

Require:

- separate distribution and manifest uploads/downloads;
- full action SHAs and credential-free checkouts;
- per-tag concurrency;
- producer-only manifest generation;
- manifest verification before and after exact Twine installation;
- absence-only ordinary preflights for PyPI and every exact-tag GitHub Release,
  including drafts returned by authenticated release enumeration;
- peeled remote-tag equality with `HEAD`/`GITHUB_SHA` immediately before PyPI
  and GitHub mutations;
- bounded PyPI polling followed by exact filename/digest verification;
- explicit empty-draft creation, individual asset uploads, immediate
  revalidation, and `gh release edit --draft=false`;
- state-machine fixtures for draft asset counts 0/1/2, duplicate releases,
  duplicate/unexpected/bad assets, exact metadata, and published/draft
  distinction; and
- recovery documentation bound to an exact run ID, tag, commit, distribution
  artifact, and manifest artifact.

Run those focused tests before implementation and record the observed RED
failures. Per release instructions, the intentionally failing RED state is
evidence from the command output, not a retained failing commit.

## Task 2: Make the producer own the manifest

In `build-artifacts`:

1. Build the wheel and sdist.
2. Establish the exact two-file inventory.
3. Generate `release-manifest/SHA256SUMS` from those exact bytes.
4. Verify the manifest immediately.
5. Upload explicit wheel/sdist paths as `vercor-distributions`.
6. Upload only `release-manifest/SHA256SUMS` as
   `vercor-release-manifest`.

All installed-artifact and quality lanes continue downloading only
`vercor-distributions`; the publisher downloads both named artifacts.

## Task 3: Enforce the trusted-byte and namespace boundaries

In `publish-release`:

1. Check out the triggering SHA without persisted credentials.
2. Download both named artifacts.
3. Validate tag/version/notes and exact distribution inventory.
4. Authenticate the bytes with `tools/validate_release_state.py files`.
5. Authenticated-enumerate all releases and require exact-tag state `absent`.
6. Require PyPI HTTP 404.
7. Install `twine==6.2.0`, run `twine check`, and authenticate the bytes again.
8. Immediately re-query PyPI/release absence and compare the peeled remote tag
   with both checked-out `HEAD` and `GITHUB_SHA`.
9. Run the pinned PyPA publisher once, without `skip-existing`.

An exact PyPI version or any exact-tag draft/published release is a stop
condition. The ordinary workflow never changes that condition into recovery.

## Task 4: Verify PyPI before touching GitHub Releases

After the publisher returns:

1. Poll the PyPI JSON endpoint for a bounded number of attempts.
2. Accept only 404 while waiting and 200 for validation.
3. Fail immediately on any other status.
4. On 200, require exactly the expected wheel and sdist filenames and their
   producer-manifest SHA-256 digests.
5. Fail closed on timeout or any filename/digest discrepancy.

No GitHub Release mutation may run before this gate passes.

## Task 5: Publish through an explicit same-run draft state machine

Still in the uninterrupted ordinary run:

1. Re-enumerate authenticated releases and require exact-tag state `absent`.
2. Recheck the peeled remote tag.
3. Create an empty exact-tag draft with exact title, notes, and prerelease
   state.
4. Enumerate and validate the zero-asset draft.
5. Recheck the tag, generate the canonical uploads-host URL with a safely
   encoded fixed filename, upload the wheel by release ID, and revalidate its
   digest.
6. Recheck the tag, generate the corresponding sdist URL, upload the sdist by
   release ID, and revalidate both digests.
7. Authenticate the local bytes again.
8. Recheck the tag and publish with
   `gh release edit "$GITHUB_REF_NAME" --draft=false`.
9. Enumerate once more and require one published exact-tag release with exact
   metadata and the exact two assets/digests.

If any draft/upload/publish command fails, the job fails. A later ordinary
rerun stops on the existing PyPI version or draft.

## Task 6: Implement explicit recovery documentation

Update `docs/releasing.md` so recovery:

1. accepts a separately authorized `RELEASE_RUN_ID`;
2. proves its `headSha`, `push` event, and exact tag;
3. compares the peeled remote tag with `RELEASE_COMMIT`;
4. downloads that run's `vercor-distributions` and
   `vercor-release-manifest`;
5. authenticates those bytes before any tooling or mutation;
6. validates PyPI against that manifest;
7. authenticated-enumerates exact-tag releases, including drafts;
8. accepts only absent or a valid draft with 0/1/2 expected verified assets;
9. uploads only missing verified assets, revalidating immediately; and
10. publishes only an exact two-asset draft with `gh release edit
    --draft=false`.

Before either tag-creation transcript, authenticated paginated enumeration
must pass the manifest-free exact-tag absence validator so drafts cannot be
misclassified as absence. After either separately authorized one-file PyPI
upload, bounded polling must verify the final exact two-file producer-manifest
state.

Duplicate releases, a published release, wrong metadata, unexpected assets, or
bad digests stop recovery. Recovery never edits metadata, clobbers an asset,
deletes a release, or mutates a tag.

## Task 7: Verification and evidence

Before the cohesive commit:

1. Parse the workflow as YAML.
2. Feed every workflow `run:` block to `bash -n`.
3. Run the release validator and workflow/documentation boundaries.
4. Run Black, flake8, mypy, and compileall for changed Python.
5. Run `pytest tests/ -q --fast --tb=short`.
6. Run `git diff --check` and inspect the complete diff.
7. Keep `PROGRESS.md` at or below its 180-line active-memory limit.
8. Exercise installed `gh` with a synthetic token through a closed loopback
   proxy, proving the production URL targets `uploads.github.com` while the
   old hostname form targets `api.uploads.github.com`.
9. Append `.superpowers/sdd/final-review-fix-report.md` with exact RED/GREEN
   commands and counts, action-ref sources, changed files, commit SHA, remaining
   concerns, and a statement that no external mutation ran.
10. Commit the complete correction once, without pushing or creating/moving a
   tag.
