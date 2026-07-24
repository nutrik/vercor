# Release Capability Probe and 0.4.1 Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the false GitHub repository `permissions.push` preflight with
a non-mutating Release write-capability probe and publish the corrected
release automation as VerCOR 0.4.1 without moving `v0.4.0`.

**Architecture:** The tag-only `publish-release` job keeps job-scoped
`contents: write` and proves the effective installation token can call
`POST /repos/{owner}/{repo}/releases/generate-notes` before each authenticated,
draft-aware release enumeration. Package metadata, CI artifact names, active
release documentation, and release verification move together to 0.4.1; the
0.4 compatibility floor and historical 0.4.0 records do not move.

**Tech Stack:** GitHub Actions YAML, GitHub CLI/REST API, Python 3.12/3.13,
pytest, PyYAML, standard-library release validator, Flit, Twine 6.2.0.

## Global Constraints

- Never delete, overwrite, or repoint the existing local or remote `v0.4.0`
  tag.
- `GITHUB_TOKEN` remains the only GitHub Release credential and only
  `publish-release` receives `contents: write`.
- `secrets.PYPI_API_TOKEN`, the immutable PyPA action SHA, and
  `skip-existing: false` remain unchanged.
- The capability probe must be non-mutating and must run twice: during initial
  state classification and immediately before the PyPI mutation.
- Authenticated paginated release enumeration must follow each capability
  probe.
- The release inventory remains exactly one wheel and one source distribution.
- The active release is exactly `0.4.1`; plugin compatibility remains
  `vercor>=0.4.0,<0.5`.
- Use `/Users/romannuterman/miniforge3/envs/scipy/bin/python` for Python
  commands if the Conda launcher panics.
- Do not push a branch or tag until all local release gates pass.
- Preserve the observed unrelated fast-suite
  `test_setup_implementation_modules_are_private_after_boundary_redesign`
  baseline failure as separate evidence; do not mask it.

---

## File Map

- `.github/workflows/python-package.yml`: build the exact 0.4.1 artifacts and
  perform both non-mutating GitHub capability probes.
- `tools/validate_release_state.py`: remove the invalid
  `github-repository-push` command.
- `tests/test_release_state_validator.py`: prove the obsolete command is no
  longer accepted.
- `tests/test_distribution_boundaries.py`: enforce probe structure/order,
  current version, exact artifact names, and active release transcript.
- `tests/test_api_architecture_review.py`: bind active release metadata and
  recovery commands to 0.4.1 while preserving the stable 0.4 API contract.
- `tests/test_versioning_policy.py`: set the supervised current release to
  0.4.1.
- `pyproject.toml`: declare package version 0.4.1.
- `CHANGELOG.md`: record the 0.4.1 release-automation correction.
- `docs/release-notes-0.4.1.md`: provide the exact hosted Release notes.
- `docs/releasing.md`: make the active ordinary and recovery transcript target
  0.4.1 and use the non-mutating capability probe.
- `docs/superpowers/specs/2026-07-24-automated-release-deployment-design.md`:
  correct the superseded permission-proof statement.
- `docs/superpowers/plans/2026-07-24-automated-release-deployment.md`: correct
  the superseded permission-proof implementation statement.
- `PROGRESS.md`: record red/green and final release verification evidence.

---

### Task 1: Specify the Correct GitHub Capability Boundary

**Files:**

- Modify: `tests/test_distribution_boundaries.py`
- Modify: `tests/test_release_state_validator.py`
- Test:
  `tests/test_distribution_boundaries.py::test_version_tag_deploys_exact_tested_distributions`
- Test: new
  `tests/test_release_state_validator.py::test_cli_rejects_obsolete_github_repository_push_command`

**Interfaces:**

- Consumes: parsed `publish-release` workflow steps and the validator CLI.
- Produces: executable contracts for two non-mutating
  `releases/generate-notes` probes and removal of the misleading validator
  command.

- [ ] **Step 1: Replace the workflow permission assertions with the desired
  capability-probe assertions**

In `test_version_tag_deploys_exact_tested_distributions`, replace the
`repository_query`, `repository_push_check`, and `repository_json_argument`
block with:

```python
capability_endpoint = (
    '"repos/${GITHUB_REPOSITORY}/releases/generate-notes"'
)
capability_output = '> "$STATE_DIR/release-capability.json"'
release_enumeration = (
    "gh api --paginate --slurp "
    '"repos/${GITHUB_REPOSITORY}/releases?per_page=100"'
)
capability_preflights = tuple(
    step["run"]
    for step in publish["steps"]
    if step.get("name")
    in {
        "Classify exact public release state",
        "Revalidate immediately before PyPI mutation",
    }
)
assert len(capability_preflights) == 2
for preflight in capability_preflights:
    assert "gh api --method POST" in preflight
    assert capability_endpoint in preflight
    assert '-f tag_name="$GITHUB_REF_NAME"' in preflight
    assert '-f target_commitish="$GITHUB_SHA"' in preflight
    assert capability_output in preflight
    assert preflight.index(capability_endpoint) < preflight.index(
        release_enumeration
    )
    assert "github-repository-push" not in preflight
    assert 'repos/${GITHUB_REPOSITORY}" >' not in preflight
```

Also add these whole-workflow assertions after the loop:

```python
assert workflow_source.count("releases/generate-notes") == 2
assert "github-repository-push" not in workflow_source
assert "repository.json" not in workflow_source
```

- [ ] **Step 2: Require the obsolete validator command to be rejected**

Delete `_run_github_repository_push_validator` and its success/failure tests.
Add:

```python
@pytest.mark.fast_always
def test_cli_rejects_obsolete_github_repository_push_command(
    tmp_path: Path,
) -> None:
    payload = tmp_path / "repository.json"
    payload.write_text(
        json.dumps({"permissions": {"push": True}}),
        encoding="utf-8",
    )

    completed = _run_validator(
        "github-repository-push",
        "--json",
        str(payload),
    )

    assert completed.returncode == 2
    assert "invalid choice" in completed.stderr
    assert "github-repository-push" in completed.stderr
```

- [ ] **Step 3: Run the focused tests and verify RED**

Run:

```bash
/Users/romannuterman/miniforge3/envs/scipy/bin/python -m pytest \
  tests/test_release_state_validator.py::test_cli_rejects_obsolete_github_repository_push_command \
  tests/test_distribution_boundaries.py::test_version_tag_deploys_exact_tested_distributions \
  -q -n0
```

Expected: both tests fail. The CLI test reports return code `0` because the
obsolete command still exists; the workflow test reports missing
`releases/generate-notes`.

---

### Task 2: Implement the Non-Mutating Capability Probe

**Files:**

- Modify: `.github/workflows/python-package.yml`
- Modify: `tools/validate_release_state.py`
- Modify: `docs/releasing.md`
- Modify:
  `docs/superpowers/specs/2026-07-24-automated-release-deployment-design.md`
- Modify:
  `docs/superpowers/plans/2026-07-24-automated-release-deployment.md`
- Modify: `tests/test_distribution_boundaries.py`
- Test: `tests/test_release_state_validator.py`
- Test: `tests/test_distribution_boundaries.py`

**Interfaces:**

- Consumes: `GH_TOKEN`, `GITHUB_REPOSITORY`, `GITHUB_REF_NAME`, and
  `GITHUB_SHA`.
- Produces: an HTTP-success capability proof with response JSON at
  `$STATE_DIR/release-capability.json`; it saves no GitHub Release state.

- [ ] **Step 1: Replace both workflow checks**

In both `Classify exact public release state` and
`Revalidate immediately before PyPI mutation`, replace the repository query
and validator call with exactly:

```yaml
          gh api --method POST \
            "repos/${GITHUB_REPOSITORY}/releases/generate-notes" \
            -f tag_name="$GITHUB_REF_NAME" \
            -f target_commitish="$GITHUB_SHA" \
            > "$STATE_DIR/release-capability.json"
```

Keep the existing `gh api --paginate --slurp` release enumeration immediately
after this block.

- [ ] **Step 2: Remove the obsolete validator implementation**

Delete `_validate_github_repository_push` from
`tools/validate_release_state.py`:

```python
def _validate_github_repository_push(arguments: argparse.Namespace) -> None:
    payload = _read_json(arguments.json)
    permissions = payload.get("permissions")
    if not isinstance(permissions, dict) or permissions.get("push") is not True:
        raise ValueError("repository permissions.push must be true")
```

Delete its parser registration:

```python
github_repository_push = subparsers.add_parser("github-repository-push")
github_repository_push.add_argument("--json", type=Path, required=True)
github_repository_push.set_defaults(run=_validate_github_repository_push)
```

- [ ] **Step 3: Correct active manual preflights**

In each active GitHub preflight in `docs/releasing.md`, replace:

```bash
gh api repos/nutrik/vercor > "$STATE_DIR/repository.json"
python tools/validate_release_state.py github-repository-push --json "$STATE_DIR/repository.json"
```

The release-PR preflight becomes:

```bash
gh api --method POST repos/nutrik/vercor/releases/generate-notes \
  -f tag_name=v0.4.1 \
  -f target_commitish="$RELEASE_COMMIT" \
  > "$PREFLIGHT_DIR/release-capability.json"
```

The tag preflight becomes:

```bash
gh api --method POST repos/nutrik/vercor/releases/generate-notes \
  -f tag_name=v0.4.1 \
  -f target_commitish="$RELEASE_COMMIT" \
  > "$TAG_PREFLIGHT_DIR/release-capability.json"
```

The exact-state recovery query becomes:

```bash
gh api --method POST repos/nutrik/vercor/releases/generate-notes \
  -f tag_name=v0.4.1 \
  -f target_commitish="$RELEASE_COMMIT" \
  > "$RECOVERY_STATE_DIR/release-capability.json"
```

Keep each authenticated release-list query after its probe. Replace the
explanatory requirement to prove repository push permission with the
requirement to prove the same token can invoke the non-mutating Release
notes-generation endpoint and then enumerate every release page.

- [ ] **Step 4: Correct the superseded architecture text and its contract**

Replace the phrase ``repository `permissions.push` is true`` in the prior
automated-release design and plan with:

```text
the same token successfully calls the non-mutating Release notes-generation
endpoint requiring `contents: write`
```

Change `test_release_design_and_plan_describe_the_final_review_state_machine`
to require the exact marker:

```python
"non-mutating Release notes-generation",
```

and add:

```python
assert "repository `permissions.push` is true" not in document
```

- [ ] **Step 5: Run focused GREEN verification**

Run:

```bash
/Users/romannuterman/miniforge3/envs/scipy/bin/python -m pytest \
  tests/test_release_state_validator.py \
  tests/test_distribution_boundaries.py::test_version_tag_deploys_exact_tested_distributions \
  tests/test_distribution_boundaries.py::test_release_design_and_plan_describe_the_final_review_state_machine \
  -q -n0
```

Expected: all selected tests pass.

- [ ] **Step 6: Commit the capability repair**

Run:

```bash
git add \
  .github/workflows/python-package.yml \
  tools/validate_release_state.py \
  tests/test_release_state_validator.py \
  tests/test_distribution_boundaries.py \
  docs/releasing.md \
  docs/superpowers/specs/2026-07-24-automated-release-deployment-design.md \
  docs/superpowers/plans/2026-07-24-automated-release-deployment.md
git diff --cached --check
git commit -m "fix: probe GitHub release write capability"
```

Expected: one focused commit with no whitespace errors.

---

### Task 3: Specify VerCOR 0.4.1 as the Active Release

**Files:**

- Modify: `tests/test_distribution_boundaries.py`
- Modify: `tests/test_api_architecture_review.py`
- Modify: `tests/test_versioning_policy.py`
- Test: the focused metadata and release-transcript tests listed below.

**Interfaces:**

- Consumes: current release files and parsed project metadata.
- Produces: exact 0.4.1 package, artifact, tag, run-selection, PyPI, and hosted
  Release expectations.

- [ ] **Step 1: Update the supervised version assertions**

In `tests/test_versioning_policy.py`, set:

```python
CURRENT_VERSION = "0.4.1"
```

In `test_runtime_metadata_separates_test_and_development_dependencies`, set:

```python
assert project["version"] == "0.4.1"
```

Change the active release-notes entry in
`test_active_sources_do_not_use_retired_public_plugin_fixture_name` to:

```python
PROJECT_ROOT / "docs" / "release-notes-0.4.1.md",
```

- [ ] **Step 2: Update active release-document assertions**

In `test_release_bundle_contains_only_vercor_distributions`, require:

```python
assert checksum_line == (
    "shasum -a 256 vercor-0.4.1-py3-none-any.whl "
    "vercor-0.4.1.tar.gz > SHA256SUMS"
)
assert "Pushing the annotated `v0.4.1` tag" in publish
```

In `test_release_guide_binds_tag_authority_workflow_selection_and_hosted_state`,
replace active values with:

```python
tag_push = "git push origin refs/tags/v0.4.1"
```

and require:

```python
"Pushing the annotated `v0.4.1` tag starts `python-package.yml`."
'.headBranch == "v0.4.1"'
'"v0.4.1"'
'"VerCOR 0.4.1"'
"vercor-0.4.1-py3-none-any.whl"
"vercor-0.4.1.tar.gz"
```

- [ ] **Step 3: Update active architecture-review release assertions**

In `test_release_files_and_metadata_describe_the_stable_release`, require:

```python
assert project["version"] == "0.4.1"
assert re.search(r"^## \[0\.4\.1\] - 2026-07-24$", changelog, re.MULTILINE)
```

and these artifacts:

```python
(
    "vercor-0.4.1-py3-none-any.whl",
    "vercor-0.4.1.tar.gz",
)
```

In the release URL, recovery-state, missing-file, and hosted-draft assertions
from `test_release_preflight_uses_...` through
`test_release_recovery_commands_verify_exact_state_before_mutation`, replace
only active release values with:

```text
v0.4.1
VerCOR 0.4.1
vercor-0.4.1-py3-none-any.whl
vercor-0.4.1.tar.gz
https://pypi.org/pypi/vercor/0.4.1/json
docs/release-notes-0.4.1.md
```

Keep the module docstring, API review heading, public-signature contract
filename, migration paths, and plugin dependency floor at 0.4.0.

- [ ] **Step 4: Run focused tests and verify RED**

Run:

```bash
/Users/romannuterman/miniforge3/envs/scipy/bin/python -m pytest \
  tests/test_versioning_policy.py::test_current_vercor_release_is_the_approved_stable_release \
  tests/test_distribution_boundaries.py::test_runtime_metadata_separates_test_and_development_dependencies \
  tests/test_distribution_boundaries.py::test_release_bundle_contains_only_vercor_distributions \
  tests/test_distribution_boundaries.py::test_release_guide_binds_tag_authority_workflow_selection_and_hosted_state \
  tests/test_api_architecture_review.py::test_release_files_and_metadata_describe_the_stable_release \
  tests/test_api_architecture_review.py::test_release_recovery_commands_verify_exact_state_before_mutation \
  -q -n0
```

Expected: failures show the repository still declares and documents 0.4.0.

---

### Task 4: Implement the 0.4.1 Package and Release Metadata

**Files:**

- Modify: `pyproject.toml`
- Modify: `.github/workflows/python-package.yml`
- Modify: `CHANGELOG.md`
- Create: `docs/release-notes-0.4.1.md`
- Modify: `docs/releasing.md`
- Modify: `tests/test_distribution_boundaries.py`
- Modify: `tests/test_api_architecture_review.py`
- Modify: `tests/test_versioning_policy.py`

**Interfaces:**

- Consumes: the 0.4 stable implementation and the corrected release workflow.
- Produces: build metadata and release transcripts for exactly VerCOR 0.4.1.

- [ ] **Step 1: Bump project and workflow artifact versions**

Set:

```toml
version = "0.4.1"
```

Replace every hard-coded active VerCOR distribution name in
`.github/workflows/python-package.yml`:

```text
vercor-0.4.0-py3-none-any.whl -> vercor-0.4.1-py3-none-any.whl
vercor-0.4.0.tar.gz           -> vercor-0.4.1.tar.gz
```

Do not change the dynamic `${VERSION}` names in `publish-release`.

- [ ] **Step 2: Add the changelog entry**

Insert before the 0.4.0 entry:

```markdown
## [0.4.1] - 2026-07-24

### Fixed

- GitHub Actions now proves its effective `contents: write` capability through
  the non-mutating Release notes-generation endpoint instead of interpreting
  repository `permissions.push` for an installation token.
- The failed immutable `v0.4.0` workflow remains preserved; publication
  recovery proceeds through the new `v0.4.1` patch tag.
```

Add before the existing comparison links:

```markdown
[0.4.1]: https://github.com/nutrik/vercor/compare/v0.4.0...v0.4.1
```

- [ ] **Step 3: Add exact 0.4.1 hosted Release notes**

Create `docs/release-notes-0.4.1.md` with:

````markdown
# VerCOR 0.4.1

VerCOR 0.4.1 republishes the stable protocol-first VerCOR 0.4 functionality
with corrected release automation.

## Fixed

- The GitHub Actions release job now verifies its effective
  `contents: write` capability through GitHub's non-mutating Release
  notes-generation endpoint.
- The failed `v0.4.0` workflow and tag remain immutable; this patch uses new
  package and tag identities.

## Upgrade

```bash
python -m pip install --upgrade "vercor==0.4.1"
```

## Compatibility and migration

VerCOR requires Python 3.12 or 3.13. Version 0.4 is intentionally
source-breaking for 0.3 applications; follow
`docs/migration-0.3-to-0.4.md`. Third-party plugins should depend on
`vercor>=0.4.0,<0.5` and use the documented stable extension modules.

## Known limitations

CAMulator requires a separately installed compatible MILES-CREDIT environment;
an exact compatible release is not yet pinned. CAMulator spinup is not
implemented. No legacy 0.3 adapter namespace is included.
````

- [ ] **Step 4: Move the active release guide to 0.4.1**

In `docs/releasing.md`, replace all active candidate, artifact, PyPI, tag,
run-selection, GitHub Release, and recovery values:

```text
0.4.0 -> 0.4.1
v0.4.0 -> v0.4.1
docs/release-notes-0.4.0.md -> docs/release-notes-0.4.1.md
```

Retain compatibility floors and migration paths exactly:

```text
vercor>=0.4.0,<0.5
docs/migration-0.3-to-0.4.md
```

Ensure each `generate-notes` block uses `tag_name=v0.4.1` and its enclosing
candidate commit variable, and still precedes authenticated release
enumeration.

- [ ] **Step 5: Run focused GREEN verification**

Run:

```bash
/Users/romannuterman/miniforge3/envs/scipy/bin/python -m pytest \
  tests/test_versioning_policy.py \
  tests/test_distribution_boundaries.py \
  tests/test_api_architecture_review.py \
  -q -n0 --fast
```

Expected: all selected tests pass.

- [ ] **Step 6: Commit the 0.4.1 release metadata**

Run:

```bash
git add \
  pyproject.toml \
  .github/workflows/python-package.yml \
  CHANGELOG.md \
  docs/release-notes-0.4.1.md \
  docs/releasing.md \
  tests/test_distribution_boundaries.py \
  tests/test_api_architecture_review.py \
  tests/test_versioning_policy.py
git diff --cached --check
git commit -m "release: prepare VerCOR 0.4.1"
```

Expected: one versioned release-candidate commit with no whitespace errors.

---

### Task 5: Validate Workflow Syntax and Release Artifacts

**Files:**

- Modify: `PROGRESS.md`
- Verify: `.github/workflows/python-package.yml`
- Verify: `dist/vercor-0.4.1-py3-none-any.whl`
- Verify: `dist/vercor-0.4.1.tar.gz`

**Interfaces:**

- Consumes: the complete local 0.4.1 candidate.
- Produces: fresh local evidence that the workflow, package metadata,
  distributions, and tests satisfy the release boundary.

- [ ] **Step 1: Parse YAML and syntax-check every workflow Bash block**

Run:

```bash
/Users/romannuterman/miniforge3/envs/scipy/bin/python -c \
  'from pathlib import Path; import subprocess, yaml; p=Path(".github/workflows/python-package.yml"); workflow=yaml.safe_load(p.read_text()); blocks=[(job_name, step.get("name", "unnamed"), step["run"]) for job_name, job in workflow["jobs"].items() for step in job["steps"] if "run" in step]; [subprocess.run(["bash", "-n"], input=source, text=True, check=True) for _, _, source in blocks]; print(f"YAML parsed; {len(blocks)} Bash blocks passed bash -n")'
```

Expected: YAML parses and every block passes `bash -n`.

- [ ] **Step 2: Run static and focused release gates**

Run:

```bash
/Users/romannuterman/miniforge3/envs/scipy/bin/python -m black --check \
  tools/validate_release_state.py \
  tests/test_release_state_validator.py \
  tests/test_distribution_boundaries.py \
  tests/test_api_architecture_review.py \
  tests/test_versioning_policy.py
/Users/romannuterman/miniforge3/envs/scipy/bin/python -m flake8 \
  tools/validate_release_state.py \
  tests/test_release_state_validator.py \
  tests/test_distribution_boundaries.py \
  tests/test_api_architecture_review.py \
  tests/test_versioning_policy.py \
  --count --max-line-length=120 --statistics
/Users/romannuterman/miniforge3/envs/scipy/bin/python -m mypy \
  tools/validate_release_state.py \
  tests/test_release_state_validator.py \
  tests/test_distribution_boundaries.py \
  tests/test_api_architecture_review.py \
  tests/test_versioning_policy.py
/Users/romannuterman/miniforge3/envs/scipy/bin/python -m pytest \
  tests/test_release_state_validator.py \
  tests/test_distribution_boundaries.py \
  tests/test_api_architecture_review.py \
  tests/test_versioning_policy.py \
  -q -n4 --dist=loadscope --max-worker-restart=0 --fast
```

Expected: all static and focused release gates pass.

- [ ] **Step 3: Run the repository fast suite**

Run:

```bash
/Users/romannuterman/miniforge3/envs/scipy/bin/python -m pytest \
  tests/ -q --fast
```

Expected: no regression attributable to the release changes. If the observed
setup-boundary baseline failure recurs, rerun that test alone in a clean Python
process and record the result separately rather than changing release code.

- [ ] **Step 4: Run the full test and coverage gates**

Run:

```bash
/Users/romannuterman/miniforge3/envs/scipy/bin/python -m pytest \
  tests/ -q --tb=short
/Users/romannuterman/miniforge3/envs/scipy/bin/python -m pytest \
  tests/ -q --cov=vercor --cov-branch --tb=short
```

Expected: the full suite passes and branch coverage remains at least 90%.

- [ ] **Step 5: Build and authenticate the exact artifacts**

Use a fresh temporary output directory:

```bash
RELEASE_BUILD_DIR="$(mktemp -d /private/tmp/vercor-0.4.1-build.XXXXXX)"
/Users/romannuterman/miniforge3/envs/scipy/bin/python -m build \
  --no-isolation --outdir "$RELEASE_BUILD_DIR"
test "$(ls -1A "$RELEASE_BUILD_DIR" | wc -l | tr -d ' ')" = "2"
test -f "$RELEASE_BUILD_DIR/vercor-0.4.1-py3-none-any.whl"
test -f "$RELEASE_BUILD_DIR/vercor-0.4.1.tar.gz"
/Users/romannuterman/miniforge3/envs/scipy/bin/python -m twine check \
  "$RELEASE_BUILD_DIR/vercor-0.4.1-py3-none-any.whl" \
  "$RELEASE_BUILD_DIR/vercor-0.4.1.tar.gz"
```

Expected: exactly the named wheel and sdist exist and Twine accepts both.

- [ ] **Step 6: Verify package metadata and installed version**

Run:

```bash
/Users/romannuterman/miniforge3/envs/scipy/bin/python -c \
  'import sys, zipfile; wheel=sys.argv[1]; archive=zipfile.ZipFile(wheel); metadata=archive.read("vercor-0.4.1.dist-info/METADATA").decode(); assert "Version: 0.4.1" in metadata; assert "vercor/py.typed" in archive.namelist(); print("wheel metadata 0.4.1 and PEP 561 marker OK")' \
  "$RELEASE_BUILD_DIR/vercor-0.4.1-py3-none-any.whl"
SMOKE_DIR="$(mktemp -d /private/tmp/vercor-0.4.1-smoke.XXXXXX)"
/Users/romannuterman/miniforge3/envs/scipy/bin/python -m pip install \
  --no-deps --target "$SMOKE_DIR/site" \
  "$RELEASE_BUILD_DIR/vercor-0.4.1-py3-none-any.whl"
PYTHONPATH="$SMOKE_DIR/site" \
  /Users/romannuterman/miniforge3/envs/scipy/bin/python -c \
  'import importlib.metadata as metadata; assert metadata.version("vercor") == "0.4.1"; print(metadata.version("vercor"))'
```

Expected: archive and installed distribution metadata are exactly 0.4.1.

- [ ] **Step 7: Record evidence and commit the progress log**

After all verification commands finish, add a dated top-level `PROGRESS.md`
entry. State that the two `generate-notes` probes replaced the removed
`permissions.push` validator, then copy the literal RED/GREEN counts, static
results, fast/full counts, coverage percentage, artifact filenames, byte
sizes, and SHA-256 values from the command outputs. End the entry with the
verified statement that `v0.4.0` was not changed. Do not write the entry before
the evidence exists. Then run:

```bash
git add PROGRESS.md
git diff --cached --check
git commit -m "docs: record 0.4.1 release verification"
```

Expected: the progress entry contains no unverified claim or placeholder.

---

### Task 6: Publish the Corrected Commit Through the Required Release PR

**Files:**

- No source changes.
- Remote mutation: one release branch and one GitHub pull request.

**Interfaces:**

- Consumes: the locally verified commits.
- Produces: a reviewed remote `main` commit eligible for immutable tagging.

- [ ] **Step 1: Invoke the GitHub publication skill**

Use `github:yeet` to confirm the exact commit range, create a dedicated branch
such as `release/vercor-0.4.1`, push it, and open a draft pull request against
`main`. The PR title is:

```text
Release VerCOR 0.4.1
```

The body states:

```text
Fix the GitHub Release capability preflight and prepare the immutable VerCOR
0.4.1 recovery release. The existing v0.4.0 tag is unchanged.
```

- [ ] **Step 2: Inspect and wait for GitHub Actions**

Use the GitHub app for PR metadata and `gh` for Actions:

```bash
PR_NUMBER="$(gh pr view --json number --jq .number)"
test -n "$PR_NUMBER"
gh pr checks "$PR_NUMBER" --watch
```

Expected: every required GitHub Actions check passes. If a check fails, invoke
`github:gh-fix-ci`, inspect the exact logs, and do not merge or tag.

- [ ] **Step 3: Mark the PR ready and merge only after review**

After required checks and repository review requirements pass, mark the PR
ready and merge it through the repository's normal protected-branch mechanism.
Record the resulting remote `main` commit:

```bash
git fetch origin main
RELEASE_COMMIT="$(git rev-parse origin/main)"
export RELEASE_COMMIT
git merge-base --is-ancestor HEAD "$RELEASE_COMMIT"
```

Expected: `origin/main` contains all approved local release commits.

---

### Task 7: Create the Immutable v0.4.1 Release Trigger

**Files:**

- No source changes.
- Remote mutation: annotated tag `v0.4.1`; the workflow owns PyPI and GitHub
  Release mutations.

**Interfaces:**

- Consumes: the exact verified commit on remote `main`.
- Produces: one tag-triggered workflow run that publishes VerCOR 0.4.1.

- [ ] **Step 1: Recheck exact public absence and tag absence**

Run:

```bash
test "$(git rev-parse origin/main)" = "$RELEASE_COMMIT"
test -z "$(git tag --list v0.4.1)"
test -z "$(git ls-remote --tags origin refs/tags/v0.4.1 'refs/tags/v0.4.1^{}')"
PREFLIGHT_DIR="$(mktemp -d /private/tmp/vercor-0.4.1-preflight.XXXXXX)"
PYPI_STATUS="$(curl -sS -L -o "$PREFLIGHT_DIR/pypi.json" -w '%{http_code}' \
  https://pypi.org/pypi/vercor/0.4.1/json)"
test "$PYPI_STATUS" = "404"
GH_TOKEN="$(gh auth token)"
export GH_TOKEN
gh api --method POST repos/nutrik/vercor/releases/generate-notes \
  -f tag_name=v0.4.1 \
  -f target_commitish="$RELEASE_COMMIT" \
  > "$PREFLIGHT_DIR/release-capability.json"
gh api --paginate --slurp "repos/nutrik/vercor/releases?per_page=100" \
  > "$PREFLIGHT_DIR/releases.json"
/Users/romannuterman/miniforge3/envs/scipy/bin/python \
  tools/validate_release_state.py github-tag-absent \
  --json "$PREFLIGHT_DIR/releases.json" \
  --tag v0.4.1
```

Expected: local tag absent, remote tag absent, PyPI returns 404, capability
probe succeeds, and authenticated release enumeration contains no `v0.4.1`.

- [ ] **Step 2: Create and verify the local annotated tag**

Run:

```bash
git tag -a v0.4.1 "$RELEASE_COMMIT" -m "VerCOR 0.4.1"
test "$(git cat-file -t v0.4.1)" = "tag"
test "$(git rev-parse 'v0.4.1^{commit}')" = "$RELEASE_COMMIT"
git show --stat v0.4.1
```

Expected: `v0.4.1` is an annotated tag peeling to the exact remote-main
release commit.

- [ ] **Step 3: Push only the new tag**

Run:

```bash
git push origin refs/tags/v0.4.1
REMOTE_TAG_COMMIT="$(git ls-remote origin 'refs/tags/v0.4.1^{}' | awk '{print $1}')"
test "$REMOTE_TAG_COMMIT" = "$RELEASE_COMMIT"
```

Expected: the remote peeled tag equals the verified release commit.

- [ ] **Step 4: Bind and watch the exact tag workflow**

Run:

```bash
RELEASE_RUN_ID="$(gh run list \
  --repo nutrik/vercor \
  --workflow python-package.yml \
  --event push \
  --commit "$RELEASE_COMMIT" \
  --limit 20 \
  --json databaseId,event,headBranch,headSha \
  --jq 'map(select(.event == "push" and .headBranch == "v0.4.1" and .headSha == env.RELEASE_COMMIT)) | sort_by(.databaseId) | last | .databaseId // empty')"
test -n "$RELEASE_RUN_ID"
gh run watch "$RELEASE_RUN_ID" --repo nutrik/vercor --exit-status
```

Expected: the exact `v0.4.1` run succeeds. Do not start any manual publisher
while it runs.

- [ ] **Step 5: Verify exact public state**

Run the `docs/releasing.md` section 8 transcript for 0.4.1. It must prove:

```text
PyPI version: 0.4.1
PyPI filenames: vercor-0.4.1-py3-none-any.whl, vercor-0.4.1.tar.gz
GitHub tag: v0.4.1
GitHub Release title: VerCOR 0.4.1
GitHub Release draft: false
GitHub Release prerelease: false
GitHub assets: vercor-0.4.1-py3-none-any.whl, vercor-0.4.1.tar.gz
```

Download both hosted assets and validate their SHA-256 digests against the
exact workflow run's `vercor-release-manifest`.
