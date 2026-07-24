# Automated Release Deployment Design

**Date:** 2026-07-24
**Status:** Final-review corrected

## Final-review correction

The initial design treated a publisher-generated checksum file as sufficient
and modeled GitHub publication as one `gh release create` mutation. Final
review corrected both assumptions. The builder now creates the authoritative
manifest beside the exact two-file inventory and uploads it as a separate
workflow artifact. GitHub publication is an explicit same-run state machine:
create an empty draft, upload and revalidate each asset, then publish the
verified draft. Separately authorized recovery can resume a valid interrupted
draft with zero, one, or two expected assets only from the exact run's
distribution and manifest artifacts.

The release re-review additionally requires draft-aware pre-tag enumeration, a
canonical `https://uploads.github.com` request target for every raw asset
upload, and bounded missing-file recovery polling that proves PyPI eventually
contains the exact two producer-manifest digests.
Pre-tag absence is trustworthy only after the same token successfully calls the
non-mutating Release notes-generation endpoint requiring `contents: write`,
then enumerates releases so draft releases are visible to the authenticated
request.

## Purpose

Publish VerCOR package distributions to PyPI and create the corresponding
GitHub Release automatically after an authorized version tag is pushed. The
deployment must reuse the exact wheel and source distribution built and tested
by the existing GitHub Actions workflow.

## Scope

This change extends `.github/workflows/python-package.yml`, updates its
executable workflow contracts, and revises `docs/releasing.md` and
`PROGRESS.md`. It does not create or push a tag, publish a package, create a
hosted release, or change the package version.

The repository administrator must separately configure a protected GitHub
Actions environment named `release`. The existing repository secret
`PYPI_API_TOKEN` will authenticate the production upload. The distinct
`TEST_PYPI_API_TOKEN` secret will not be used. Administrators must also
configure a protected `v*.*.*` tag ruleset; workflow code does not manage
repository settings.

## Trigger and authorization

The existing workflow will keep its `main` push and pull-request triggers and
add a push trigger for tags matching `v*.*.*`.

A tag push authorizes the workflow to attempt deployment only when:

- every build, installed-artifact, extension, macOS, quality, and coverage
  gate succeeds;
- the triggering ref is a tag;
- the tag is exactly `v` followed by the version in `pyproject.toml`;
- the checked-out commit is the triggering commit;
- the release environment's protection rules permit the job to start;
- PyPI does not already contain that version; and
- authenticated enumeration finds no draft or published release for that tag.

Branch pushes and pull requests continue to run CI but never run deployment.
Per-tag deployment concurrency does not cancel an in-progress publisher. Tag
protection and release-environment reviewers remain repository settings, not
workflow-controlled policy.

## Workflow architecture

`build-artifacts` remains the sole distribution builder. It creates exactly:

- `dist/vercor-0.4.0-py3-none-any.whl`; and
- `dist/vercor-0.4.0.tar.gz`.

After establishing that inventory, the builder generates
`release-manifest/SHA256SUMS`. The two distributions are uploaded as
`vercor-distributions`; the manifest is uploaded separately as
`vercor-release-manifest`, so the distribution artifact remains exactly two
files.

A new `publish-release` job will depend on all validation jobs rather than
rebuilding either distribution. It will have job-scoped `contents: write`
permission and use the protected `release` environment. No other job receives
publication permissions or a reference to the PyPI credential.

The job will:

1. Check out the exact triggering commit without persisted credentials.
2. Download both named workflow artifacts.
3. Derive the project version from `pyproject.toml`.
4. Verify the tag/version relationship, exact two-file inventory, and
   producer-issued SHA-256 manifest before installing mutable tooling.
5. Authenticated-enumerate releases (including drafts), require exact-tag
   absence, and require PyPI HTTP 404.
6. Install exactly Twine 6.2.0, run metadata checks, and reverify the manifest.
7. Compare the peeled remote tag commit with checked-out `HEAD` and
   `GITHUB_SHA` immediately before mutation.
8. Publish both distributions through the PyPA publish action pinned at commit
   `ba38be9e461d3875417946c167d0b5f3d385a247`, authenticating as `__token__`
   with `secrets.PYPI_API_TOKEN`.
9. Poll PyPI for a bounded time and require the exact two manifest digests.
10. Reconfirm exact-tag release absence and the peeled tag, then create an
    empty draft with exact title, notes, and prerelease state.
11. Generate the canonical uploads-host URL with a safely encoded fixed
    filename, then upload the wheel and sdist separately by release ID,
    checking the tag before each mutation and immediately validating the draft
    asset inventory and GitHub-reported SHA-256 digests.
12. Publish only the exact two-asset draft with
    `gh release edit --draft=false`, then enumerate and validate the published
    release.

PyPI publication precedes GitHub Release creation, preserving the existing
release order. If PyPI or draft publication succeeds only partially, an
ordinary rerun fails on the existing public state. The fail-closed recovery
procedure in `docs/releasing.md` is separately authorized and binds recovery to
the exact `RELEASE_RUN_ID`, tag, commit, distributions, and manifest.
Manual one-file PyPI recovery polls for a bounded time after Twine returns and
requires the final exact wheel/sdist filename and digest set.

## Artifact and data flow

```text
tag push
  -> build wheel and sdist once
  -> generate authoritative SHA256SUMS from those exact bytes
  -> upload distributions and manifest as separate named artifacts
  -> run all CI gates against the uploaded artifact bundle
  -> protected publish-release job downloads and authenticates both artifacts
  -> verify tag, version, metadata, inventory, and exact namespace absence
  -> publish bundle to PyPI with the repository's production token
  -> poll PyPI until exact filenames and digests are visible
  -> create empty GitHub draft, upload/verify both assets, publish draft
```

The temporary external-extension fixture distribution remains outside
`dist/`, is not uploaded with the VerCOR artifact bundle, and is never
published.

## Failure behavior

The deployment job fails before publication when the tag is malformed, the
tag and package version differ, a required release-notes file is missing, the
artifact inventory differs from the exact wheel and sdist, metadata checks
fail, the authoritative manifest differs, a public version or draft/published
release already exists, the peeled tag changes, PyPI verification times out, a
remote API returns an unexpected status, or a required CI job fails.

The job will not use PyPI's `skip-existing` behavior. Treating an existing file
as success without verifying its digest could hide a conflicting publication.
GitHub Release creation likewise will not overwrite an existing release or
asset. Duplicate releases, wrong draft metadata, unexpected assets, and digest
mismatches stop both ordinary publication and recovery. No repair path deletes
or moves a tag, deletes a release, edits bad metadata, or clobbers an asset.

## Testing and verification

Tests will be written before the workflow change and will parse the workflow to
require:

- the version-tag trigger;
- a tag-only deployment condition;
- dependencies on every validation job;
- the protected `release` environment;
- job-scoped GitHub contents permission and no OIDC permission;
- exact triggering-commit checkout, immutable action pins, and credential-free
  checkout;
- separate producer-issued manifest upload/download and verification before
  and after locked Twine installation;
- tag-to-project-version validation;
- exact wheel/sdist inventory and metadata checks;
- per-tag concurrency and peeled remote-tag checks at mutation boundaries;
- authenticated, fail-closed PyPI and draft-aware GitHub preflights;
- authenticated paginated pre-tag release enumeration that rejects draft as
  well as published exact-tag state;
- an installed-`gh` request-construction regression using a synthetic token and
  closed loopback proxy, proving the canonical uploads host without contacting
  GitHub;
- bounded post-publisher PyPI filename/digest polling;
- production publication through `secrets.PYPI_API_TOKEN`, without exposing
  the token to any other step or job; and
- draft state-machine fixtures for 0/1/2 assets, duplicates, bad/unexpected
  assets, metadata, and published/draft distinction;
- explicit draft creation, per-asset upload/revalidation, and final publication
  with the exact two tested artifacts and tracked release notes; and
- exact-run recovery documentation that never uses local-build hashes.

Documentation contracts will require the administrator setup instructions and
replace the ordinary manual publication path with the automated tag workflow,
while retaining explicit partial-release recovery instructions.

Verification will run the focused release and distribution contract tests,
workflow YAML parsing, the fast test suite, static checks appropriate to the
changed Python tests, and `git diff --check`. No tag, push, package upload, or
GitHub Release creation is part of local verification.

## Alternatives considered

### Separate release workflow

A dedicated `release.yml` would either duplicate the substantial existing
build/test workflow or require a broader reusable-workflow refactor. That adds
unnecessary release risk and makes it harder to prove that publication uses
the exact artifacts already validated by CI.

### Publish after a manually created GitHub Release

Triggering on `release.published` is a common PyPI pattern, but it leaves GitHub
Release creation manual and therefore does not satisfy the requested automated
deployment.

### Extend the existing build workflow

This is the selected approach. A tag run builds once, validates once, and
publishes those same bytes. Publication permissions remain confined to one
protected, tag-only job.
