# Releasing VerCOR

This is the verification procedure for a release candidate. Preparing a
candidate does not authorize a commit, pull request, tag, push, upload,
publication, merge, or hosted release.

The historical `v0.4.0`, `v0.4.1`, and `v0.4.2` tags remain immutable. The
`v0.4.2` publication failed because PyPI rejected its invalid license
classifier; no PyPI 0.4.2 version or GitHub Release exists. VerCOR 0.4.3 is
the current published release. The version-specific commands below preserve
its verified recovery procedure as an operational record; do not rerun them to
republish 0.4.3. A future release must use a new version, tag, and release
branch throughout. Do not delete, overwrite, or repoint a historical tag.

## Repository deployment configuration

Tagged deployment uses the existing production repository secret
`PYPI_API_TOKEN`. Keep `TEST_PYPI_API_TOKEN` reserved for TestPyPI; the release
workflow never references it. Configure a GitHub Actions environment named
`release` and add the required reviewers or other deployment protection rules
before pushing a version tag. Repository administrators must also configure a
protected `v*.*.*` tag ruleset; the workflow does not change repository
settings. The workflow grants `contents: write` and exposes the production
token only in the tag-only `publish-release` job.

## 1. Confirm and review the candidate

- Work from the intended `release/vercor-0.4.3` branch with complete history.
- Confirm `pyproject.toml` and `CHANGELOG.md` use the intended version.
- Confirm the package root and canonical owner manifests match live signatures.
- Confirm optional JCM and Veros versions in the verification environment.
- Leave CAMulator uninstalled and unpinned until an exact compatible release is
  verified.

Perform the read-only review before requesting commit authority:

```bash
set -euo pipefail
RELEASE_BRANCH="release/vercor-0.4.3"
export RELEASE_BRANCH
test "$(git branch --show-current)" = "$RELEASE_BRANCH"
git status --short --untracked-files=all
git diff --check
git diff
git diff --cached --check
git diff --cached
git ls-files --others --exclude-standard
```

Only after explicit commit authorization may a maintainer stage the completely
reviewed release state and create the release commit. Run this transcript in one
shell so later gates retain `RELEASE_COMMIT`:

```text
set -euo pipefail
RELEASE_BRANCH="release/vercor-0.4.3"
export RELEASE_BRANCH
test "$(git branch --show-current)" = "$RELEASE_BRANCH"
git add -A
git diff --cached --check
git diff --cached
git commit -m "Release 0.4.3"
RELEASE_COMMIT="$(git rev-parse HEAD)"
export RELEASE_COMMIT
test -n "${RELEASE_COMMIT:-}"
printf 'Release commit: %s\n' "$RELEASE_COMMIT"
test -z "$(git status --porcelain --untracked-files=all)"
```

Do not infer the release SHA from a later moving branch name.

## 2. Run source gates from the release commit

Use the supported environment binaries directly if the Conda launcher is
unavailable. This single-shell transcript fails closed and binds every gate to
the recorded commit:

```bash
set -euo pipefail
RELEASE_BRANCH="release/vercor-0.4.3"
export RELEASE_BRANCH
test -n "${RELEASE_COMMIT:-}"
test "$(git branch --show-current)" = "$RELEASE_BRANCH"
test "$(git rev-parse HEAD)" = "$RELEASE_COMMIT"
test -z "$(git status --porcelain --untracked-files=all)"
python -m black --check vercor examples tests
python -m flake8 . --count --max-line-length=120 --statistics
python -m mypy vercor examples tests
python -m compileall -q vercor examples tests
python -m pytest tests/ -q --fast --tb=short
python -m pytest tests/ -q --tb=short
python -m pytest tests/ -q --cov=vercor --cov-branch --cov-report=term-missing --cov-fail-under=90
git diff --check
test "$(git rev-parse HEAD)" = "$RELEASE_COMMIT"
test -z "$(git status --porcelain --untracked-files=all)"
```

The clean-tree checks protect the source/ref state. They intentionally do not
inspect ignored `dist/` bytes; the fresh-directory and exact-inventory gates
below protect the build input, and the checksum manifest protects its bytes.

## 3. Build once and create the checksum manifest

Build the two publishable VerCOR distributions from the exact clean release
commit, inspect the bundle, and create the ignored `dist/SHA256SUMS` manifest:

```bash
set -euo pipefail
test -n "${RELEASE_COMMIT:-}"
test "$(git rev-parse HEAD)" = "$RELEASE_COMMIT"
test -z "$(git status --porcelain --untracked-files=all)"
test ! -e dist || test -d dist
shopt -s nullglob dotglob
DIST_ARTIFACTS=(dist/*)
test "${#DIST_ARTIFACTS[@]}" -eq 0
python -m build --outdir dist
DIST_ARTIFACTS=(dist/*)
test "${#DIST_ARTIFACTS[@]}" -eq 2
test -f dist/vercor-0.4.3-py3-none-any.whl
test -f dist/vercor-0.4.3.tar.gz
unzip -p dist/vercor-0.4.3-py3-none-any.whl vercor-0.4.3.dist-info/METADATA
tar -xOf dist/vercor-0.4.3.tar.gz vercor-0.4.3/PKG-INFO
python -m twine check dist/vercor-0.4.3-py3-none-any.whl dist/vercor-0.4.3.tar.gz
VERCOR_ARTIFACT_DIR="$(pwd)/dist" python -m pytest tests/test_distribution_boundaries.py -q --tb=short
(
  cd dist
  shasum -a 256 vercor-0.4.3-py3-none-any.whl vercor-0.4.3.tar.gz > SHA256SUMS
  shasum -a 256 -c SHA256SUMS
)
python -c 'import importlib.metadata as m; print("JCM", m.version("jcm")); print("Veros", m.version("veros"))'
test "$(git rev-parse HEAD)" = "$RELEASE_COMMIT"
test -z "$(git status --porcelain --untracked-files=all)"
```

`dist/SHA256SUMS` is ignored local release evidence, not a
source-cleanliness signal. The local `dist/SHA256SUMS` is not authoritative for
hosted publication because CI performs an independent build. The
`build-artifacts` job creates the authoritative manifest from its exact wheel
and sdist, then uploads it separately as `vercor-release-manifest`; only that
manifest can identify bytes published by the tag run.

## 4. Run local installed-artifact acceptance

Run the bounded optional-model nodes and explicit output-free gradient
acceptance against the exact source commit:

```bash
set -euo pipefail
test -n "${RELEASE_COMMIT:-}"
test "$(git rev-parse HEAD)" = "$RELEASE_COMMIT"
python -m pytest tests/test_setup_lifecycle_helpers.py::test_make_jcm_land_atmosphere_replaces_only_missing_forcing tests/test_external_components_coverage.py::test_jax_gcm_initialize_uses_provided_forcing_and_can_spin_up tests/test_external_components_coverage.py::test_jax_gcm_initialize_builds_default_forcing_when_missing tests/test_setup_boundaries.py::test_veros_implementation_import_does_not_configure_runtime tests/test_setup_boundaries.py::test_veros_factory_configures_once_before_implementation_import tests/test_external_components_coverage.py::test_veros_initialize_spinup_follows_enabled_only -q --tb=short
python -m pytest tests/test_v0_4_workflow_execution.py::test_output_free_workflow_preserves_jvp_and_reverse_mode_gradients tests/test_v0_4_workflow_execution.py::test_payload_dependent_multi_step_scan_preserves_treedef_jvp_and_grad tests/test_v0_4_output_providers.py::test_all_disabled_target_remains_jit_and_gradient_compatible -q --tb=short
(
  smoke_dir="$(mktemp -d)"
  external_extension_fixture_dir="$(mktemp -d)"
  python -m build --wheel \
    --outdir "$external_extension_fixture_dir" \
    tests/fixtures/external_extension_test_fixture
  python -m pip install --target "$smoke_dir/site" "dist/vercor-0.4.3-py3-none-any.whl"
  python -m pip install --target "$smoke_dir/site" \
    "$external_extension_fixture_dir/external_extension_test_fixture-0.1.0-py3-none-any.whl"
  cd "$smoke_dir"
  PYTHONPATH="$smoke_dir/site" \
    python -m external_extension_test_fixture.smoke \
    --output-dir "$smoke_dir/extension-output"
)
(cd dist && shasum -a 256 -c SHA256SUMS)
```

The hosted workflow repeats base, JCM, Veros, wheel/sdist, external-extension,
mypy, and macOS lanes on its configured Python matrix. Pull requests and
`main` pushes run validation only. A pushed version tag matching `v*.*.*` runs
the same gates and, after all pass, the protected `publish-release` job
publishes the tested artifact bundle.

## 5. Prepare the required release pull request

The workflow file runs validation on pushes to `main`, pull requests targeting
`main`, and version tags. Only a version tag can satisfy the deployment job's
condition. A push to `release/vercor-0.4.3` alone does not run it. Before any
GitHub preflight or pull-request creation, fetch the protected branch, prove it
is an ancestor of the reviewed release commit, push that exact commit to the
release branch, and verify the remote branch SHA. Then prove the same token can
invoke the non-mutating Release notes-generation endpoint and enumerate every
release page so an exact-tag draft cannot be mistaken for absence. PyPI 0.4.3
must also be absent:

```text
set -euo pipefail
RELEASE_BRANCH="release/vercor-0.4.3"
export RELEASE_BRANCH
test "$(git branch --show-current)" = "$RELEASE_BRANCH"
test -n "${RELEASE_COMMIT:-}"
test "$(git rev-parse HEAD)" = "$RELEASE_COMMIT"
git fetch --no-tags origin main
MAIN_COMMIT="$(git rev-parse refs/remotes/origin/main)"
export MAIN_COMMIT
test -n "${MAIN_COMMIT:-}"
git merge-base --is-ancestor "$MAIN_COMMIT" "$RELEASE_COMMIT"
git push --set-upstream origin "$RELEASE_BRANCH"
REMOTE_RELEASE_COMMIT="$(git ls-remote origin "refs/heads/${RELEASE_BRANCH}" | awk '{print $1}')"
export REMOTE_RELEASE_COMMIT
test "$REMOTE_RELEASE_COMMIT" = "$RELEASE_COMMIT"
GH_TOKEN="$(gh auth token)"
export GH_TOKEN
test -n "${GH_TOKEN:-}"
PREFLIGHT_DIR="$(mktemp -d)"
gh api --method POST repos/nutrik/vercor/releases/generate-notes \
  -f tag_name=v0.4.3 \
  -f target_commitish="$RELEASE_COMMIT" \
  > "$PREFLIGHT_DIR/release-capability.json"
gh api --paginate --slurp "repos/nutrik/vercor/releases?per_page=100" > "$PREFLIGHT_DIR/releases.json"
python tools/validate_release_state.py github-tag-absent --json "$PREFLIGHT_DIR/releases.json" --tag v0.4.3
PYPI_STATUS="$(curl -sS -L -o "$PREFLIGHT_DIR/pypi.json" -w '%{http_code}' https://pypi.org/pypi/vercor/0.4.3/json)"
export PYPI_STATUS
test "$PYPI_STATUS" = "404"
gh pr list --repo nutrik/vercor --state open --base main --head "$RELEASE_BRANCH" --json number,url,headRefName,baseRefName,headRefOid
```

If no authorized pull request exists, this is the exact preparation command.
Run it only with explicit pull-request creation authority:

```text
set -euo pipefail
RELEASE_BRANCH="release/vercor-0.4.3"
export RELEASE_BRANCH
test -n "${RELEASE_COMMIT:-}"
test "$(git rev-parse HEAD)" = "$RELEASE_COMMIT"
test -n "${MAIN_COMMIT:-}"
git merge-base --is-ancestor "$MAIN_COMMIT" "$RELEASE_COMMIT"
gh pr create --repo nutrik/vercor --base main --head "$RELEASE_BRANCH" --draft --title "Release VerCOR 0.4.3" --body "Prepare the immutable VerCOR 0.4.3 recovery release. The v0.4.2 workflow failed because PyPI rejected its invalid license classifier; v0.4.2 remains unchanged."
```

Confirm exactly one open matching pull request, reverify the remote branch SHA,
select the `pull_request` run of `python-package.yml` at the exact SHA, watch
it, and mechanically recheck the PR and run. After those checks pass, mark the
draft ready, merge through the repository's normal mechanism, bind
`RELEASE_COMMIT` to the fetched protected `main` merge commit, and detach the
worktree at that exact commit:

```text
set -euo pipefail
RELEASE_BRANCH="release/vercor-0.4.3"
export RELEASE_BRANCH
test "$(git branch --show-current)" = "$RELEASE_BRANCH"
test -n "${RELEASE_COMMIT:-}"
test "$(git rev-parse HEAD)" = "$RELEASE_COMMIT"
test -z "$(git status --porcelain --untracked-files=all)"
git fetch --no-tags origin main
MAIN_COMMIT="$(git rev-parse refs/remotes/origin/main)"
export MAIN_COMMIT
test -n "${MAIN_COMMIT:-}"
git merge-base --is-ancestor "$MAIN_COMMIT" "$RELEASE_COMMIT"
REMOTE_RELEASE_COMMIT="$(git ls-remote origin "refs/heads/${RELEASE_BRANCH}" | awk '{print $1}')"
export REMOTE_RELEASE_COMMIT
test "$REMOTE_RELEASE_COMMIT" = "$RELEASE_COMMIT"
RELEASE_PR_NUMBER="$(gh pr list --repo nutrik/vercor --state open --base main --head "$RELEASE_BRANCH" --json number --jq 'if length == 1 then .[0].number else empty end')"
export RELEASE_PR_NUMBER
test -n "${RELEASE_PR_NUMBER:-}"
test "$(gh pr view "$RELEASE_PR_NUMBER" --repo nutrik/vercor --json state --jq .state)" = "OPEN"
test "$(gh pr view "$RELEASE_PR_NUMBER" --repo nutrik/vercor --json isDraft --jq .isDraft)" = "true"
test "$(gh pr view "$RELEASE_PR_NUMBER" --repo nutrik/vercor --json baseRefName --jq .baseRefName)" = "main"
test "$(gh pr view "$RELEASE_PR_NUMBER" --repo nutrik/vercor --json headRefName --jq .headRefName)" = "$RELEASE_BRANCH"
test "$(gh pr view "$RELEASE_PR_NUMBER" --repo nutrik/vercor --json headRefOid --jq .headRefOid)" = "$RELEASE_COMMIT"
RELEASE_RUN_ID="$(gh run list --repo nutrik/vercor --workflow python-package.yml --event pull_request --branch "$RELEASE_BRANCH" --commit "$RELEASE_COMMIT" --limit 20 --json databaseId,event,headSha --jq 'map(select(.event == "pull_request" and .headSha == env.RELEASE_COMMIT)) | sort_by(.databaseId) | last | .databaseId // empty')"
export RELEASE_RUN_ID
test -n "${RELEASE_RUN_ID:-}"
gh run watch "$RELEASE_RUN_ID" --repo nutrik/vercor --exit-status
test "$(gh run view "$RELEASE_RUN_ID" --repo nutrik/vercor --json headSha --jq .headSha)" = "$RELEASE_COMMIT"
test "$(gh run view "$RELEASE_RUN_ID" --repo nutrik/vercor --json event --jq .event)" = "pull_request"
test "$(gh run view "$RELEASE_RUN_ID" --repo nutrik/vercor --json conclusion --jq .conclusion)" = "success"
(cd dist && shasum -a 256 -c SHA256SUMS)
gh pr ready "$RELEASE_PR_NUMBER" --repo nutrik/vercor
gh pr merge "$RELEASE_PR_NUMBER" --repo nutrik/vercor --merge
test "$(gh pr view "$RELEASE_PR_NUMBER" --repo nutrik/vercor --json state --jq .state)" = "MERGED"
MERGE_COMMIT="$(gh pr view "$RELEASE_PR_NUMBER" --repo nutrik/vercor --json mergeCommit --jq '.mergeCommit.oid // empty')"
export MERGE_COMMIT
test -n "${MERGE_COMMIT:-}"
git fetch --no-tags origin main
RELEASE_COMMIT="$(git rev-parse refs/remotes/origin/main)"
export RELEASE_COMMIT
test -n "${RELEASE_COMMIT:-}"
test "$MERGE_COMMIT" = "$RELEASE_COMMIT"
git switch --detach "$RELEASE_COMMIT"
test "$(git rev-parse HEAD)" = "$RELEASE_COMMIT"
test -z "$(git status --porcelain --untracked-files=all)"
```

If the run has not appeared yet, stop and rerun the selection transcript later.
Do not select a `push` run, a run for another workflow, or a run at another SHA.
Do not mark the PR ready or merge until every exact PR and workflow check above
has passed.

## 6. Create and verify the annotated tag

Immediately before tagging, fetch `main` again, require the remote `main`
commit to be an ancestor of and equal to the reviewed release commit, repeat
the authenticated public-namespace preflights, and confirm the local and
remote tag are absent.
The same token must successfully call the non-mutating Release
notes-generation endpoint and then enumerate every release page before the
manifest-free exact-tag validator accepts a well-formed authenticated release
listing with zero exact-tag draft or published matches:

Run the following transcript only with explicit tag-push and package-publication
authority. Pushing the annotated tag starts the automated publication after its
protected workflow gates pass.

```text
set -euo pipefail
test -n "${RELEASE_COMMIT:-}"
test "$(git rev-parse HEAD)" = "$RELEASE_COMMIT"
test -z "$(git status --porcelain --untracked-files=all)"
git fetch --no-tags origin main
MAIN_COMMIT="$(git rev-parse refs/remotes/origin/main)"
export MAIN_COMMIT
test -n "${MAIN_COMMIT:-}"
git merge-base --is-ancestor "$MAIN_COMMIT" "$RELEASE_COMMIT"
test "$MAIN_COMMIT" = "$RELEASE_COMMIT"
GH_TOKEN="$(gh auth token)"
export GH_TOKEN
test -n "${GH_TOKEN:-}"
TAG_PREFLIGHT_DIR="$(mktemp -d)"
gh api --method POST repos/nutrik/vercor/releases/generate-notes \
  -f tag_name=v0.4.3 \
  -f target_commitish="$RELEASE_COMMIT" \
  > "$TAG_PREFLIGHT_DIR/release-capability.json"
gh api --paginate --slurp "repos/nutrik/vercor/releases?per_page=100" > "$TAG_PREFLIGHT_DIR/releases.json"
python tools/validate_release_state.py github-tag-absent --json "$TAG_PREFLIGHT_DIR/releases.json" --tag v0.4.3
PYPI_STATUS="$(curl -sS -L -o "$TAG_PREFLIGHT_DIR/pypi.json" -w '%{http_code}' https://pypi.org/pypi/vercor/0.4.3/json)"
export PYPI_STATUS
test "$PYPI_STATUS" = "404"
test -z "$(git tag --list v0.4.3)"
REMOTE_TAG_PRECHECK="$(git ls-remote --tags origin refs/tags/v0.4.3 'refs/tags/v0.4.3^{}')"
export REMOTE_TAG_PRECHECK
test -z "$REMOTE_TAG_PRECHECK"
git tag -a v0.4.3 "$RELEASE_COMMIT" -m "VerCOR 0.4.3"
test "$(git cat-file -t v0.4.3)" = "tag"
test "$(git rev-parse 'v0.4.3^{commit}')" = "$RELEASE_COMMIT"
git show --stat v0.4.3
git push origin refs/tags/v0.4.3
REMOTE_TAG_STATE="$(git ls-remote --tags origin refs/tags/v0.4.3 'refs/tags/v0.4.3^{}')"
export REMOTE_TAG_STATE
REMOTE_TAG_COMMIT="$(printf '%s\n' "$REMOTE_TAG_STATE" | awk '$2 == "refs/tags/v0.4.3^{}" {print $1}')"
export REMOTE_TAG_COMMIT
test -n "${REMOTE_TAG_COMMIT:-}"
test "$REMOTE_TAG_COMMIT" = "$RELEASE_COMMIT"
```

Pushing the annotated `v0.4.3` tag starts `python-package.yml`. That tag push
is the publication authorization: after every CI lane passes, the protected
deployment job validates the tag against `pyproject.toml`, downloads the exact
two-file `vercor-distributions` artifact, checks both public namespaces, uses
`PYPI_API_TOKEN` to publish to PyPI, and creates the GitHub Release with the
same files. An existing local or remote tag is a stop condition. Never
overwrite or repoint a published release tag.

## 7. Publish packages and create the hosted release

Pushing the annotated `v0.4.3` tag starts the automated deployment. Do not run
a second local Twine upload or create the GitHub Release locally during the
ordinary release path. The repository secret `PYPI_API_TOKEN` is supplied only
to the production publish action in `python-package.yml`. The exact run carries
two named artifacts: `vercor-distributions` contains only the wheel and sdist,
while `vercor-release-manifest` contains their authoritative manifest.

```text
set -euo pipefail
test -n "${RELEASE_COMMIT:-}"
test "$(git rev-parse 'v0.4.3^{commit}')" = "$RELEASE_COMMIT"
RELEASE_RUN_ID="$(gh run list --repo nutrik/vercor --workflow python-package.yml --event push --commit "$RELEASE_COMMIT" --limit 20 --json databaseId,event,headBranch,headSha --jq 'map(select(.event == "push" and .headBranch == "v0.4.3" and .headSha == env.RELEASE_COMMIT)) | sort_by(.databaseId) | last | .databaseId // empty')"
export RELEASE_RUN_ID
test -n "${RELEASE_RUN_ID:-}"
gh run watch "$RELEASE_RUN_ID" --repo nutrik/vercor --exit-status
test "$(gh run view "$RELEASE_RUN_ID" --repo nutrik/vercor --json headSha --jq .headSha)" = "$RELEASE_COMMIT"
test "$(gh run view "$RELEASE_RUN_ID" --repo nutrik/vercor --json event --jq .event)" = "push"
test "$(gh run view "$RELEASE_RUN_ID" --repo nutrik/vercor --json headBranch --jq .headBranch)" = "v0.4.3"
test "$(gh run view "$RELEASE_RUN_ID" --repo nutrik/vercor --json conclusion --jq .conclusion)" = "success"
CI_VERIFY_ROOT="$(mktemp -d)"
gh run download "$RELEASE_RUN_ID" --repo nutrik/vercor --name vercor-distributions --dir "$CI_VERIFY_ROOT/dist"
gh run download "$RELEASE_RUN_ID" --repo nutrik/vercor --name vercor-release-manifest --dir "$CI_VERIFY_ROOT/manifest"
python tools/validate_release_state.py files --directory "$CI_VERIFY_ROOT/dist" --manifest "$CI_VERIFY_ROOT/manifest/SHA256SUMS" --expect vercor-0.4.3-py3-none-any.whl vercor-0.4.3.tar.gz
```

If the tag run has not appeared, is waiting for approval in the protected
`release` environment, or has not completed, stop and inspect that exact run.
Do not select a branch-push or pull-request run. If PyPI publication succeeds
but GitHub Release creation fails, use the exact-state recovery procedure
below; rerunning the ordinary deployment must fail when PyPI or an exact-tag
draft/published release already exists; it never silently repairs a partial
release. Resume only through the separately authorized recovery procedure with
the exact run's two artifacts.

## 8. Verify the published package and hosted release

```bash
set -euo pipefail
test -n "${RELEASE_COMMIT:-}"
test -n "${RELEASE_RUN_ID:-}"
test "$(gh run view "$RELEASE_RUN_ID" --repo nutrik/vercor --json headSha --jq .headSha)" = "$RELEASE_COMMIT"
test "$(gh run view "$RELEASE_RUN_ID" --repo nutrik/vercor --json event --jq .event)" = "push"
test "$(gh run view "$RELEASE_RUN_ID" --repo nutrik/vercor --json headBranch --jq .headBranch)" = "v0.4.3"
CI_VERIFY_ROOT="$(mktemp -d)"
gh run download "$RELEASE_RUN_ID" --repo nutrik/vercor --name vercor-distributions --dir "$CI_VERIFY_ROOT/dist"
gh run download "$RELEASE_RUN_ID" --repo nutrik/vercor --name vercor-release-manifest --dir "$CI_VERIFY_ROOT/manifest"
python tools/validate_release_state.py files --directory "$CI_VERIFY_ROOT/dist" --manifest "$CI_VERIFY_ROOT/manifest/SHA256SUMS" --expect vercor-0.4.3-py3-none-any.whl vercor-0.4.3.tar.gz
PYPI_VERIFY_JSON="$CI_VERIFY_ROOT/pypi.json"
PYPI_VERIFY_STATUS="$(curl -sS -L -o "$PYPI_VERIFY_JSON" -w '%{http_code}' https://pypi.org/pypi/vercor/0.4.3/json)"
test "$PYPI_VERIFY_STATUS" = "200"
python tools/validate_release_state.py pypi --json "$PYPI_VERIFY_JSON" --manifest "$CI_VERIFY_ROOT/manifest/SHA256SUMS" --expect vercor-0.4.3-py3-none-any.whl vercor-0.4.3.tar.gz
published_check_dir="$(mktemp -d)"
python -m venv "$published_check_dir/venv"
"$published_check_dir/venv/bin/python" -m pip install --upgrade pip
"$published_check_dir/venv/bin/python" -m pip install --no-cache-dir "vercor==0.4.3"
"$published_check_dir/venv/bin/python" -c 'import importlib.metadata as m; assert m.version("vercor") == "0.4.3"; print(m.version("vercor"))'
"$published_check_dir/venv/bin/python" -c 'from vercor import Clock, Coupler, Exchange, RectilinearGrid, RunState, RuntimeOptions'
RELEASE_VIEW_DIR="$(mktemp -d)"
RELEASE_VIEW_JSON="$RELEASE_VIEW_DIR/release.json"
export RELEASE_VIEW_JSON
gh release view v0.4.3 --repo nutrik/vercor --json tagName,name,isDraft,isPrerelease,assets > "$RELEASE_VIEW_JSON"
python - "$RELEASE_VIEW_JSON" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
expected = {
    "tagName": "v0.4.3",
    "name": "VerCOR 0.4.3",
    "isDraft": False,
    "isPrerelease": False,
}
for key, value in expected.items():
    if payload.get(key) != value:
        raise SystemExit(f"unexpected GitHub Release {key}: {payload.get(key)!r}")
PY
python tools/validate_release_state.py assets --json "$RELEASE_VIEW_JSON" --expect vercor-0.4.3-py3-none-any.whl vercor-0.4.3.tar.gz
release_verify_dir="$(mktemp -d)"
gh release download v0.4.3 --repo nutrik/vercor --dir "$release_verify_dir"
python tools/validate_release_state.py files --directory "$release_verify_dir" --manifest "$CI_VERIFY_ROOT/manifest/SHA256SUMS" --expect vercor-0.4.3-py3-none-any.whl vercor-0.4.3.tar.gz
```

## 9. Query exact public state before recovery

Recovery begins by selecting one run by `RELEASE_RUN_ID`, then proving its
event, tag, commit, remote peeled tag, artifacts, and authoritative manifest.
Never substitute a local build or a similarly named artifact from another run.

```text
set -euo pipefail
test -n "${RELEASE_COMMIT:-}"
test -n "${RELEASE_RUN_ID:-}"
test "$(gh run view "$RELEASE_RUN_ID" --repo nutrik/vercor --json headSha --jq .headSha)" = "$RELEASE_COMMIT"
test "$(gh run view "$RELEASE_RUN_ID" --repo nutrik/vercor --json event --jq .event)" = "push"
test "$(gh run view "$RELEASE_RUN_ID" --repo nutrik/vercor --json headBranch --jq .headBranch)" = "v0.4.3"
REMOTE_TAG_COMMIT="$(git ls-remote origin 'refs/tags/v0.4.3^{}' | awk '{print $1}')"
test "$REMOTE_TAG_COMMIT" = "$RELEASE_COMMIT"
CI_RECOVERY_ROOT="$(mktemp -d)"
export CI_RECOVERY_ROOT
CI_DIST_DIR="$CI_RECOVERY_ROOT/dist"
CI_MANIFEST="$CI_RECOVERY_ROOT/manifest/SHA256SUMS"
export CI_DIST_DIR CI_MANIFEST
gh run download "$RELEASE_RUN_ID" --repo nutrik/vercor --name vercor-distributions --dir "$CI_DIST_DIR"
gh run download "$RELEASE_RUN_ID" --repo nutrik/vercor --name vercor-release-manifest --dir "$CI_RECOVERY_ROOT/manifest"
python tools/validate_release_state.py files --directory "$CI_DIST_DIR" --manifest "$CI_MANIFEST" --expect vercor-0.4.3-py3-none-any.whl vercor-0.4.3.tar.gz
RECOVERY_STATE_DIR="$CI_RECOVERY_ROOT/state"
mkdir -p "$RECOVERY_STATE_DIR"
PYPI_STATUS="$(curl -sS -L -o "$RECOVERY_STATE_DIR/pypi.json" -w '%{http_code}' https://pypi.org/pypi/vercor/0.4.3/json)"
case "$PYPI_STATUS" in 200|404) ;; *) printf 'Unexpected PyPI HTTP status: %s\n' "$PYPI_STATUS" >&2; exit 1 ;; esac
GITHUB_TOKEN="$(gh auth token)"
export GITHUB_TOKEN
test -n "${GITHUB_TOKEN:-}"
gh api --method POST repos/nutrik/vercor/releases/generate-notes \
  -f tag_name=v0.4.3 \
  -f target_commitish="$RELEASE_COMMIT" \
  > "$RECOVERY_STATE_DIR/release-capability.json"
gh api --paginate --slurp "repos/nutrik/vercor/releases?per_page=100" > "$RECOVERY_STATE_DIR/releases.json"
python tools/validate_release_state.py github-releases --json "$RECOVERY_STATE_DIR/releases.json" --manifest "$CI_MANIFEST" --tag v0.4.3 --title "VerCOR 0.4.3" --notes-file docs/release-notes-0.4.3.md --expect vercor-0.4.3-py3-none-any.whl vercor-0.4.3.tar.gz --allow-state absent draft published --state-output "$RECOVERY_STATE_DIR/release-state.json"
printf 'PyPI status: %s\n' "$PYPI_STATUS"
```

## 10. Safe recovery

Recovery is a separately authorized path; an ordinary workflow rerun must
remain stopped. Rebind the exact tag run and download its two named artifacts
again immediately before choosing one alternative:

```text
set -euo pipefail
test -n "${RELEASE_COMMIT:-}"
test -n "${RELEASE_RUN_ID:-}"
test "$(gh run view "$RELEASE_RUN_ID" --repo nutrik/vercor --json headSha --jq .headSha)" = "$RELEASE_COMMIT"
test "$(gh run view "$RELEASE_RUN_ID" --repo nutrik/vercor --json event --jq .event)" = "push"
test "$(gh run view "$RELEASE_RUN_ID" --repo nutrik/vercor --json headBranch --jq .headBranch)" = "v0.4.3"
REMOTE_TAG_COMMIT="$(git ls-remote origin 'refs/tags/v0.4.3^{}' | awk '{print $1}')"
test "$REMOTE_TAG_COMMIT" = "$RELEASE_COMMIT"
CI_RECOVERY_ROOT="$(mktemp -d)"
export CI_RECOVERY_ROOT
CI_DIST_DIR="$CI_RECOVERY_ROOT/dist"
CI_MANIFEST="$CI_RECOVERY_ROOT/manifest/SHA256SUMS"
export CI_DIST_DIR CI_MANIFEST
gh run download "$RELEASE_RUN_ID" --repo nutrik/vercor --name vercor-distributions --dir "$CI_DIST_DIR"
gh run download "$RELEASE_RUN_ID" --repo nutrik/vercor --name vercor-release-manifest --dir "$CI_RECOVERY_ROOT/manifest"
python tools/validate_release_state.py files --directory "$CI_DIST_DIR" --manifest "$CI_MANIFEST" --expect vercor-0.4.3-py3-none-any.whl vercor-0.4.3.tar.gz
```

If PyPI returned 200 with only one verified file, run exactly one separately
labeled missing-file alternative after confirming which filename is absent.
Each alternative polls for a bounded time after upload and completes only when
PyPI reports the exact wheel and sdist digests from the CI manifest.

### Missing PyPI wheel only

```text
set -euo pipefail
test -n "${RELEASE_COMMIT:-}"
test -n "${RELEASE_RUN_ID:-}"
python tools/validate_release_state.py files --directory "$CI_DIST_DIR" --manifest "$CI_MANIFEST" --expect vercor-0.4.3-py3-none-any.whl vercor-0.4.3.tar.gz
python -m pip install twine==6.2.0
python tools/validate_release_state.py files --directory "$CI_DIST_DIR" --manifest "$CI_MANIFEST" --expect vercor-0.4.3-py3-none-any.whl vercor-0.4.3.tar.gz
PYPI_RECOVERY_DIR="$(mktemp -d)"
PYPI_STATUS="$(curl -sS -L -o "$PYPI_RECOVERY_DIR/pypi.json" -w '%{http_code}' https://pypi.org/pypi/vercor/0.4.3/json)"
test "$PYPI_STATUS" = "200"
python tools/validate_release_state.py pypi --json "$PYPI_RECOVERY_DIR/pypi.json" --manifest "$CI_MANIFEST" --expect vercor-0.4.3.tar.gz
REMOTE_TAG_COMMIT="$(git ls-remote origin 'refs/tags/v0.4.3^{}' | awk '{print $1}')"
test "$REMOTE_TAG_COMMIT" = "$RELEASE_COMMIT"
IMMEDIATE_PYPI_STATUS="$(curl -sS -L -o "$PYPI_RECOVERY_DIR/immediate-pypi.json" -w '%{http_code}' https://pypi.org/pypi/vercor/0.4.3/json)"
test "$IMMEDIATE_PYPI_STATUS" = "200"
python tools/validate_release_state.py pypi --json "$PYPI_RECOVERY_DIR/immediate-pypi.json" --manifest "$CI_MANIFEST" --expect vercor-0.4.3.tar.gz
python -m twine upload --repository-url https://upload.pypi.org/legacy/ "$CI_DIST_DIR/vercor-0.4.3-py3-none-any.whl"
PYPI_RECOVERY_VERIFIED=false
for attempt in {1..12}; do
  FINAL_PYPI_STATUS="$(curl -sS -L -o "$PYPI_RECOVERY_DIR/final-pypi.json" -w '%{http_code}' https://pypi.org/pypi/vercor/0.4.3/json)"
  case "$FINAL_PYPI_STATUS" in
    200)
      if python tools/validate_release_state.py pypi --json "$PYPI_RECOVERY_DIR/final-pypi.json" --manifest "$CI_MANIFEST" --expect vercor-0.4.3-py3-none-any.whl vercor-0.4.3.tar.gz; then
        PYPI_RECOVERY_VERIFIED=true
        break
      fi
      ;;
    404) ;;
    *) printf 'Unexpected PyPI HTTP status: %s\n' "$FINAL_PYPI_STATUS" >&2; exit 1 ;;
  esac
  if [ "$attempt" -lt 12 ]; then
    sleep 10
  fi
done
test "$PYPI_RECOVERY_VERIFIED" = "true"
```

### Missing PyPI sdist only

```text
set -euo pipefail
test -n "${RELEASE_COMMIT:-}"
test -n "${RELEASE_RUN_ID:-}"
python tools/validate_release_state.py files --directory "$CI_DIST_DIR" --manifest "$CI_MANIFEST" --expect vercor-0.4.3-py3-none-any.whl vercor-0.4.3.tar.gz
python -m pip install twine==6.2.0
python tools/validate_release_state.py files --directory "$CI_DIST_DIR" --manifest "$CI_MANIFEST" --expect vercor-0.4.3-py3-none-any.whl vercor-0.4.3.tar.gz
PYPI_RECOVERY_DIR="$(mktemp -d)"
PYPI_STATUS="$(curl -sS -L -o "$PYPI_RECOVERY_DIR/pypi.json" -w '%{http_code}' https://pypi.org/pypi/vercor/0.4.3/json)"
test "$PYPI_STATUS" = "200"
python tools/validate_release_state.py pypi --json "$PYPI_RECOVERY_DIR/pypi.json" --manifest "$CI_MANIFEST" --expect vercor-0.4.3-py3-none-any.whl
REMOTE_TAG_COMMIT="$(git ls-remote origin 'refs/tags/v0.4.3^{}' | awk '{print $1}')"
test "$REMOTE_TAG_COMMIT" = "$RELEASE_COMMIT"
IMMEDIATE_PYPI_STATUS="$(curl -sS -L -o "$PYPI_RECOVERY_DIR/immediate-pypi.json" -w '%{http_code}' https://pypi.org/pypi/vercor/0.4.3/json)"
test "$IMMEDIATE_PYPI_STATUS" = "200"
python tools/validate_release_state.py pypi --json "$PYPI_RECOVERY_DIR/immediate-pypi.json" --manifest "$CI_MANIFEST" --expect vercor-0.4.3-py3-none-any.whl
python -m twine upload --repository-url https://upload.pypi.org/legacy/ "$CI_DIST_DIR/vercor-0.4.3.tar.gz"
PYPI_RECOVERY_VERIFIED=false
for attempt in {1..12}; do
  FINAL_PYPI_STATUS="$(curl -sS -L -o "$PYPI_RECOVERY_DIR/final-pypi.json" -w '%{http_code}' https://pypi.org/pypi/vercor/0.4.3/json)"
  case "$FINAL_PYPI_STATUS" in
    200)
      if python tools/validate_release_state.py pypi --json "$PYPI_RECOVERY_DIR/final-pypi.json" --manifest "$CI_MANIFEST" --expect vercor-0.4.3-py3-none-any.whl vercor-0.4.3.tar.gz; then
        PYPI_RECOVERY_VERIFIED=true
        break
      fi
      ;;
    404) ;;
    *) printf 'Unexpected PyPI HTTP status: %s\n' "$FINAL_PYPI_STATUS" >&2; exit 1 ;;
  esac
  if [ "$attempt" -lt 12 ]; then
    sleep 10
  fi
done
test "$PYPI_RECOVERY_VERIFIED" = "true"
```

If an incorrect package file was accepted, yank 0.4.3 through package-index
administration, preserve the tag and evidence, and prepare a new patch release.
Published files cannot be replaced and a released version must not be reused
for different bytes.

### Resume an exact GitHub draft

Use only after PyPI contains the exact two CI-produced files. Authenticated
release enumeration includes drafts; a duplicate exact tag, a published
release, wrong metadata, an unexpected asset, or any digest mismatch is a stop
condition. A valid draft may contain zero, one, or both expected assets.
The URL helper safely encodes the fixed filename and emits the canonical
`https://uploads.github.com` request target; `gh api --hostname
uploads.github.com` is invalid because it addresses `api.uploads.github.com`.

```text
set -euo pipefail
test -n "${RELEASE_COMMIT:-}"
test -n "${RELEASE_RUN_ID:-}"
python tools/validate_release_state.py files --directory "$CI_DIST_DIR" --manifest "$CI_MANIFEST" --expect vercor-0.4.3-py3-none-any.whl vercor-0.4.3.tar.gz
GITHUB_TOKEN="$(gh auth token)"
export GITHUB_TOKEN
test -n "${GITHUB_TOKEN:-}"
GITHUB_RECOVERY_DIR="$(mktemp -d)"
PYPI_STATUS="$(curl -sS -L -o "$GITHUB_RECOVERY_DIR/pypi.json" -w '%{http_code}' https://pypi.org/pypi/vercor/0.4.3/json)"
test "$PYPI_STATUS" = "200"
python tools/validate_release_state.py pypi --json "$GITHUB_RECOVERY_DIR/pypi.json" --manifest "$CI_MANIFEST" --expect vercor-0.4.3-py3-none-any.whl vercor-0.4.3.tar.gz
check_tag_binding() {
  test "$(gh run view "$RELEASE_RUN_ID" --repo nutrik/vercor --json headSha --jq .headSha)" = "$RELEASE_COMMIT"
  test "$(gh run view "$RELEASE_RUN_ID" --repo nutrik/vercor --json event --jq .event)" = "push"
  test "$(gh run view "$RELEASE_RUN_ID" --repo nutrik/vercor --json headBranch --jq .headBranch)" = "v0.4.3"
  REMOTE_TAG_COMMIT="$(git ls-remote origin 'refs/tags/v0.4.3^{}' | awk '{print $1}')"
  test "$REMOTE_TAG_COMMIT" = "$RELEASE_COMMIT"
}
gh api --paginate --slurp "repos/nutrik/vercor/releases?per_page=100" > "$GITHUB_RECOVERY_DIR/releases.json"
python tools/validate_release_state.py github-releases --json "$GITHUB_RECOVERY_DIR/releases.json" --manifest "$CI_MANIFEST" --tag v0.4.3 --title "VerCOR 0.4.3" --notes-file docs/release-notes-0.4.3.md --expect vercor-0.4.3-py3-none-any.whl vercor-0.4.3.tar.gz --allow-state absent draft --state-output "$GITHUB_RECOVERY_DIR/release-state.json"
RELEASE_STATE="$(python -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["state"])' "$GITHUB_RECOVERY_DIR/release-state.json")"
if [ "$RELEASE_STATE" = "absent" ]; then
  check_tag_binding
  gh release create v0.4.3 --repo nutrik/vercor --verify-tag --draft --title "VerCOR 0.4.3" --notes-file docs/release-notes-0.4.3.md
  python tools/wait_for_github_release_state.py --repository nutrik/vercor --manifest "$CI_MANIFEST" --tag v0.4.3 --title "VerCOR 0.4.3" --notes-file docs/release-notes-0.4.3.md --expect vercor-0.4.3-py3-none-any.whl vercor-0.4.3.tar.gz --target-state draft --target-present --transitional-state absent --transitional-present --attempts 12 --interval-seconds 2 --state-output "$GITHUB_RECOVERY_DIR/release-state.json"
fi
RELEASE_ID="$(python -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["release_id"])' "$GITHUB_RECOVERY_DIR/release-state.json")"
WHEEL_MISSING="$(python -c 'import json,sys; print(sys.argv[2] in json.load(open(sys.argv[1], encoding="utf-8"))["missing"])' "$GITHUB_RECOVERY_DIR/release-state.json" vercor-0.4.3-py3-none-any.whl)"
SDIST_MISSING="$(python -c 'import json,sys; print(sys.argv[2] in json.load(open(sys.argv[1], encoding="utf-8"))["missing"])' "$GITHUB_RECOVERY_DIR/release-state.json" vercor-0.4.3.tar.gz)"
if [ "$WHEEL_MISSING" = "True" ]; then
  check_tag_binding
  WHEEL_UPLOAD_URL="$(python tools/validate_release_state.py github-upload-url --repository nutrik/vercor --release-id "$RELEASE_ID" --name vercor-0.4.3-py3-none-any.whl)"
  test "$WHEEL_UPLOAD_URL" = "https://uploads.github.com/repos/nutrik/vercor/releases/${RELEASE_ID}/assets?name=vercor-0.4.3-py3-none-any.whl"
  gh api --method POST -H "Content-Type: application/octet-stream" "$WHEEL_UPLOAD_URL" --input "$CI_DIST_DIR/vercor-0.4.3-py3-none-any.whl" > "$GITHUB_RECOVERY_DIR/upload-wheel.json"
  if [ "$SDIST_MISSING" = "True" ]; then
    python tools/wait_for_github_release_state.py --repository nutrik/vercor --manifest "$CI_MANIFEST" --tag v0.4.3 --title "VerCOR 0.4.3" --notes-file docs/release-notes-0.4.3.md --expect vercor-0.4.3-py3-none-any.whl vercor-0.4.3.tar.gz --target-state draft --target-present vercor-0.4.3-py3-none-any.whl --transitional-state draft --transitional-present --release-id "$RELEASE_ID" --attempts 12 --interval-seconds 2 --state-output "$GITHUB_RECOVERY_DIR/release-state.json"
  else
    python tools/wait_for_github_release_state.py --repository nutrik/vercor --manifest "$CI_MANIFEST" --tag v0.4.3 --title "VerCOR 0.4.3" --notes-file docs/release-notes-0.4.3.md --expect vercor-0.4.3-py3-none-any.whl vercor-0.4.3.tar.gz --target-state draft --target-present vercor-0.4.3-py3-none-any.whl vercor-0.4.3.tar.gz --transitional-state draft --transitional-present vercor-0.4.3.tar.gz --release-id "$RELEASE_ID" --attempts 12 --interval-seconds 2 --state-output "$GITHUB_RECOVERY_DIR/release-state.json"
  fi
fi
if [ "$SDIST_MISSING" = "True" ]; then
  check_tag_binding
  SDIST_UPLOAD_URL="$(python tools/validate_release_state.py github-upload-url --repository nutrik/vercor --release-id "$RELEASE_ID" --name vercor-0.4.3.tar.gz)"
  test "$SDIST_UPLOAD_URL" = "https://uploads.github.com/repos/nutrik/vercor/releases/${RELEASE_ID}/assets?name=vercor-0.4.3.tar.gz"
  gh api --method POST -H "Content-Type: application/octet-stream" "$SDIST_UPLOAD_URL" --input "$CI_DIST_DIR/vercor-0.4.3.tar.gz" > "$GITHUB_RECOVERY_DIR/upload-sdist.json"
  python tools/wait_for_github_release_state.py --repository nutrik/vercor --manifest "$CI_MANIFEST" --tag v0.4.3 --title "VerCOR 0.4.3" --notes-file docs/release-notes-0.4.3.md --expect vercor-0.4.3-py3-none-any.whl vercor-0.4.3.tar.gz --target-state draft --target-present vercor-0.4.3-py3-none-any.whl vercor-0.4.3.tar.gz --transitional-state draft --transitional-present vercor-0.4.3-py3-none-any.whl --release-id "$RELEASE_ID" --attempts 12 --interval-seconds 2 --state-output "$GITHUB_RECOVERY_DIR/release-state.json"
fi
test "$(python -c 'import json,sys; print(len(json.load(open(sys.argv[1], encoding="utf-8"))["missing"]))' "$GITHUB_RECOVERY_DIR/release-state.json")" -eq 0
check_tag_binding
gh release edit v0.4.3 --repo nutrik/vercor --draft=false
python tools/wait_for_github_release_state.py --repository nutrik/vercor --manifest "$CI_MANIFEST" --tag v0.4.3 --title "VerCOR 0.4.3" --notes-file docs/release-notes-0.4.3.md --expect vercor-0.4.3-py3-none-any.whl vercor-0.4.3.tar.gz --target-state published --target-present vercor-0.4.3-py3-none-any.whl vercor-0.4.3.tar.gz --transitional-state draft --transitional-present vercor-0.4.3-py3-none-any.whl vercor-0.4.3.tar.gz --release-id "$RELEASE_ID" --attempts 12 --interval-seconds 2 --state-output "$GITHUB_RECOVERY_DIR/release-state.json"
```

Do not edit metadata, overwrite assets, delete the tag, or delete a release to
make recovery pass. Preserve unexpected state as evidence and prepare a new
patch release when the exact-state validator rejects it.
