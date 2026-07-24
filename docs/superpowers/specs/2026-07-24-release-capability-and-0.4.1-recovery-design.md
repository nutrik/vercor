# Release Capability Probe and 0.4.1 Recovery Design

## Context

The `v0.4.0` tag workflow failed before publication while classifying public
release state. The job had effective `contents: write` permission, but
`tools/validate_release_state.py github-repository-push` required
`permissions.push: true` from `GET /repos/nutrik/vercor`. GitHub Actions'
`GITHUB_TOKEN` is an installation access token, and that repository response
does not reliably report the token's fine-grained `contents: write`
capability. The validator therefore rejected a correctly permissioned job.

The failed run did not invoke the PyPI publisher or create a GitHub Release.
PyPI has no VerCOR 0.4.0 project-version response, and GitHub has no Release
for the tag. The existing annotated `v0.4.0` tag remains immutable and points
to the commit containing the faulty workflow. Rerunning that run would execute
the same workflow revision.

## Approved Outcome

The release workflow will prove its GitHub Release write capability using the
same job token without mutating repository state. The failed `v0.4.0` attempt
will not be retagged or published. A patch release, VerCOR `0.4.1`, will carry
the workflow fix and become the next publication candidate through a new
annotated `v0.4.1` tag.

No external mutation occurs while implementing or locally verifying this
design. The eventual `v0.4.1` tag push is the authorized release trigger only
after the corrected commit passes local verification and is present on the
remote release branch.

## Capability-Probe Design

Both prepublication checks in `publish-release` will replace:

```bash
gh api "repos/${GITHUB_REPOSITORY}" > "$STATE_DIR/repository.json"
python tools/validate_release_state.py github-repository-push \
  --json "$STATE_DIR/repository.json"
```

with:

```bash
gh api --method POST \
  "repos/${GITHUB_REPOSITORY}/releases/generate-notes" \
  -f tag_name="$GITHUB_REF_NAME" \
  -f target_commitish="$GITHUB_SHA" \
  > "$STATE_DIR/release-capability.json"
```

GitHub documents this endpoint as requiring `contents: write` for installation
tokens and states that its generated content is not saved. A successful
request therefore tests the exact permission family needed later to create
and publish the Release without creating, editing, or deleting a Release.

The existing authenticated, paginated release enumeration remains immediately
after the capability probe. The same token must successfully enumerate
releases, including any draft visible to it, and the existing validator must
still prove the exact tag is absent. The PyPI 404 check, artifact digest check,
and remote peeled-tag check remain unchanged and fail closed.

The `github-repository-push` validator command and its unit tests will be
removed so future workflow or recovery documentation cannot reuse the
misleading signal. Current manual preflights in `docs/releasing.md` will use
the same non-mutating capability probe before authenticated release
enumeration.

## Alternatives Considered

1. Rely only on the workflow's static `contents: write` declaration. This is
   simpler but loses a runtime check that the effective token can call a
   Release write endpoint.
2. Supply a personal access token and retain `permissions.push`. This makes
   the repository field more user-oriented but introduces a long-lived
   credential when the scoped `GITHUB_TOKEN` is sufficient.
3. Delete and recreate `v0.4.0`. This would reuse the version but mutate an
   already pushed release tag and violate the repository's immutable-tag
   recovery policy.

The non-mutating endpoint plus a new `0.4.1` patch release is preferred because
it directly tests the required capability and preserves tag history.

## 0.4.1 Release Metadata

The active package version in `pyproject.toml` becomes `0.4.1`. Workflow build,
installation, checksum, and publication lanes must expect exactly:

- `vercor-0.4.1-py3-none-any.whl`
- `vercor-0.4.1.tar.gz`

`CHANGELOG.md` receives a dated `0.4.1` entry describing the corrected release
permission probe and preserved immutable `v0.4.0` failure history. A new
`docs/release-notes-0.4.1.md` presents `0.4.1` as the original stable 0.4
functionality plus release-automation repair. The existing 0.4.0 changelog and
release-notes files remain historical records.

The active commands and recovery transcript in `docs/releasing.md` move from
`0.4.0`/`v0.4.0` to `0.4.1`/`v0.4.1`. Compatibility statements, migration
guides, plugin floors such as `vercor>=0.4.0,<0.5`, historical design
documents, and validator fixture names remain at 0.4.0 where that is their
meaning.

## Execution Order and Failure Safety

The corrected `publish-release` sequence is:

1. Download and authenticate the CI-produced wheel, source distribution, and
   checksum manifest.
2. Probe GitHub Release write capability without saving state.
3. Enumerate GitHub Releases and require exact-tag absence.
4. Require PyPI version absence and reauthenticate the local files.
5. Run the locked metadata checker.
6. Repeat artifact, PyPI, capability, Release-absence, checked-out-SHA, and
   remote-tag checks immediately before publication.
7. Publish the exact two distributions to PyPI.
8. Verify PyPI filenames and SHA-256 digests.
9. Create a GitHub Release draft, upload and verify the exact two assets, then
   publish that draft.

The capability probe cannot alter package artifacts or PyPI. Every failure
before step 7 leaves PyPI and GitHub Releases unchanged. A failure after PyPI
publication uses the existing exact-state GitHub recovery path; it never
reuploads or overwrites accepted PyPI files.

## Test Strategy

Tests are changed before workflow and metadata implementation:

- Workflow structure tests require two `releases/generate-notes` calls, each
  using `POST`, `GITHUB_REF_NAME`, `GITHUB_SHA`, and output redirection.
- Sequence tests require each capability probe before its corresponding
  authenticated release enumeration and reject every
  `github-repository-push` or repository-JSON check.
- A validator CLI test first requires the obsolete subcommand to be rejected;
  it fails while that command still exists and passes only after the command
  and implementation are removed.
- Distribution and release-document tests require version `0.4.1`, exact
  0.4.1 artifact names, notes, tag commands, run selection, and verification
  commands while retaining the 0.4 compatibility floor.
- All workflow Bash blocks receive `bash -n` validation, and the workflow YAML
  is parsed.
- Focused release tests, fast tests, static checks, package builds, Twine
  checks, exact artifact-boundary checks, and the full suite run before the
  release commit is considered eligible for tagging.

The existing unrelated fast-suite setup-boundary failure observed before these
changes is recorded as baseline evidence and is not silently attributed to
this release repair.

## Release Handoff

Implementation may commit locally and push the corrected release commit to the
remote branch. Before creating `v0.4.1`, recheck that:

- the working tree is clean;
- remote `main` contains the exact release commit;
- PyPI `0.4.1` is absent;
- no GitHub Release targets `v0.4.1`;
- no local or remote `v0.4.1` tag exists.

Only then create and push the annotated `v0.4.1` tag. The tag-triggered
workflow owns package publication and GitHub Release creation; no manual
parallel upload is allowed.
