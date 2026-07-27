# Sphinx Pyproject Version Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `docs/conf.py` obtain the Sphinx version strictly from the repository-root `[project].version` in `pyproject.toml`.

**Architecture:** Resolve the metadata file from `Path(__file__)` so the lookup is independent of the Sphinx process working directory. Parse it with Python 3.12+'s standard-library `tomllib`, use the value for both Sphinx version settings, and let missing or malformed metadata fail during configuration.

**Tech Stack:** Python 3.12+, `pathlib`, `tomllib`, Sphinx configuration, pytest

## Global Constraints

- `pyproject.toml` remains the sole executable owner of the VerCOR package version.
- Do not import `vercor.__version__` while configuring the documentation version.
- Do not retain the `0+untagged` to `main` fallback.
- Missing `pyproject.toml`, invalid TOML, missing `[project]`, or missing `version` must fail explicitly.
- Do not change package metadata, runtime exports, release automation, or the current project version.
- Include every current and newly created untracked documentation file in the implementation commit.

---

### Task 1: Allow the official Read the Docs configuration URL

**Files:**
- Modify: `tests/test_versioning_policy.py`

**Interfaces:**
- Consumes: the official root `.readthedocs.yaml` reference URL `https://docs.readthedocs.io/en/stable/config-file/v2.html`.
- Produces: `_forbidden_api_tokens(relative_path: Path, line: str) -> tuple[str, ...]` that ignores only an API-like token whose span is inside that exact external URL.

- [ ] **Step 1: Write the failing policy tests**

Add these tests after the existing numerical-vector API-token tests in
`tests/test_versioning_policy.py`:

```python
@pytest.mark.fast_always
def test_integrated_scanner_allows_official_readthedocs_config_reference(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _run_integrated_scanner_for_line(
        monkeypatch,
        tmp_path,
        relative_path=Path(".readthedocs.yaml"),
        line=(
            "# See https://docs.readthedocs.io/en/stable/config-file/"
            + "v"
            + "2.html for details"
        ),
    )


@pytest.mark.fast_always
def test_readthedocs_reference_does_not_hide_a_stale_api_token() -> None:
    line = (
        "# See https://docs.readthedocs.io/en/stable/config-file/"
        + "v"
        + "2.html; stale "
        + "v"
        + "2 API"
    )

    assert _forbidden_api_tokens(Path(".readthedocs.yaml"), line) == ("v" + "2",)
```

- [ ] **Step 2: Run the policy tests and verify RED**

Run:

```bash
/Users/romannuterman/miniforge3/envs/scipy/bin/python -m pytest \
  tests/test_versioning_policy.py::test_integrated_scanner_allows_official_readthedocs_config_reference \
  tests/test_versioning_policy.py::test_readthedocs_reference_does_not_hide_a_stale_api_token \
  -q -n0 --tb=short
```

Expected: the integrated scanner test fails because `/v2.html` is reported as
a forbidden API token. The same-line test reports both tokens instead of only
the genuinely stale token.

- [ ] **Step 3: Implement the narrow span-aware exception**

Add these constants below `_NUMERICAL_VECTOR_LINES`:

```python
_READTHEDOCS_CONFIG_PATH = Path(".readthedocs.yaml")
_READTHEDOCS_CONFIG_REFERENCE = re.compile(
    r"https://docs\.readthedocs\.io/en/stable/config-file/v2\.html"
)
```

At the start of `_forbidden_api_tokens`, retain match spans and remove only a
match enclosed by the exact reference URL:

```python
matches = tuple(FORBIDDEN_API_TOKEN.finditer(line))
if relative_path == _READTHEDOCS_CONFIG_PATH:
    reference_spans = tuple(
        (match.start(), match.end())
        for match in _READTHEDOCS_CONFIG_REFERENCE.finditer(line)
    )
    matches = tuple(
        match
        for match in matches
        if not any(
            start <= match.start() and match.end() <= end
            for start, end in reference_spans
        )
    )
tokens = tuple(match.group() for match in matches)
```

Keep the existing numerical-vector exception after this new filtering.

- [ ] **Step 4: Run the policy tests and verify GREEN**

Run:

```bash
/Users/romannuterman/miniforge3/envs/scipy/bin/python -m pytest \
  tests/test_versioning_policy.py -q -n0 --tb=short
```

Expected: all version-policy tests pass.

- [ ] **Step 5: Commit the policy correction**

```bash
git add tests/test_versioning_policy.py
git diff --cached --check
git commit -m "test: allow Read the Docs config reference"
```

---

### Task 2: Read the Sphinx version from pyproject metadata

**Files:**
- Create: `.readthedocs.yaml`
- Create: `docs/Makefile`
- Create: `docs/requirements.txt`
- Create: `tests/test_docs_conf.py`
- Modify: `docs/conf.py:18-21,68-76`
- Modify: `PROGRESS.md`

**Interfaces:**
- Consumes: `pyproject.toml` containing a PEP 621 `[project]` table with a string `version` field.
- Produces: Sphinx globals `version: str` and `release: str`, both equal to `[project].version`.

- [ ] **Step 1: Write the failing configuration tests**

Create `tests/test_docs_conf.py`:

```python
"""Contracts for the Sphinx documentation configuration."""

from __future__ import annotations

import runpy
import shutil
import sys
from pathlib import Path
from typing import Any

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONF_PATH = PROJECT_ROOT / "docs" / "conf.py"


def _run_copied_conf(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    pyproject_text: str,
) -> dict[str, Any]:
    """Execute the real configuration from a synthetic project location."""
    docs_path = tmp_path / "docs"
    docs_path.mkdir()
    copied_conf = docs_path / "conf.py"
    shutil.copy2(CONF_PATH, copied_conf)
    (tmp_path / "pyproject.toml").write_text(pyproject_text, encoding="utf-8")

    unrelated_working_directory = tmp_path / "working-directory"
    unrelated_working_directory.mkdir()
    monkeypatch.chdir(unrelated_working_directory)

    original_sys_path = sys.path.copy()
    try:
        return runpy.run_path(str(copied_conf))
    finally:
        sys.path[:] = original_sys_path


def test_conf_reads_version_relative_to_its_own_project(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Use checkout metadata even when Sphinx starts elsewhere."""
    config = _run_copied_conf(
        tmp_path,
        monkeypatch,
        '[project]\nname = "synthetic-vercor"\nversion = "9.8.7"\n',
    )

    assert config["version"] == "9.8.7"
    assert config["release"] == "9.8.7"


def test_conf_rejects_missing_project_version(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fail instead of substituting a branch name for missing metadata."""
    with pytest.raises(KeyError, match="version"):
        _run_copied_conf(
            tmp_path,
            monkeypatch,
            '[project]\nname = "synthetic-vercor"\n',
        )
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```bash
/Users/romannuterman/miniforge3/envs/scipy/bin/python -m pytest \
  tests/test_docs_conf.py -q -n0 --tb=short
```

Expected: two failures. The synthetic project reports the checkout package
version instead of `9.8.7`, and missing metadata does not raise `KeyError`.

- [ ] **Step 3: Implement the minimal strict lookup**

In `docs/conf.py`, add the imports alongside the existing standard-library
imports:

```python
import tomllib
from pathlib import Path
```

Replace the `vercor.__version__` import and fallback with:

```python
pyproject_path = Path(__file__).resolve().parents[1] / "pyproject.toml"
vercor_version = tomllib.loads(
    pyproject_path.read_text(encoding="utf-8")
)["project"]["version"]
```

Retain:

```python
version = vercor_version
release = vercor_version
```

- [ ] **Step 4: Run the focused tests and verify GREEN**

Run:

```bash
/Users/romannuterman/miniforge3/envs/scipy/bin/python -m pytest \
  tests/test_docs_conf.py -q -n0 --tb=short
```

Expected: `2 passed`.

- [ ] **Step 5: Run focused version/documentation regression checks**

Run:

```bash
/Users/romannuterman/miniforge3/envs/scipy/bin/python -m pytest \
  tests/test_docs_conf.py \
  tests/test_distribution_boundaries.py \
  tests/test_api_architecture_review.py \
  tests/test_versioning_policy.py \
  -q --fast --tb=short
```

Expected: the new tests and existing relevant contracts pass.

- [ ] **Step 6: Update the progress log**

Add a concise dated item at the top of `PROGRESS.md` recording:

```markdown
- Sphinx project-version lookup completed locally (2026-07-27):
  `docs/conf.py` now resolves the repository-root `pyproject.toml` from
  `__file__`, parses `[project].version` with `tomllib`, and uses it for both
  Sphinx version fields without importing `vercor` or falling back to `main`.
  Focused configuration tests cover an unrelated working directory, synthetic
  metadata, and missing-version failure.
```

Append the actual focused and regression test counts and any unrelated
pre-existing failure after running verification.

- [ ] **Step 7: Run final static and fast verification**

Run:

```bash
/Users/romannuterman/miniforge3/envs/scipy/bin/python -m black \
  --check docs/conf.py tests/test_docs_conf.py
/Users/romannuterman/miniforge3/envs/scipy/bin/python -m flake8 \
  docs/conf.py tests/test_docs_conf.py --count --max-line-length=120 --statistics
/Users/romannuterman/miniforge3/envs/scipy/bin/python -m mypy \
  tests/test_docs_conf.py
/Users/romannuterman/miniforge3/envs/scipy/bin/python -m pytest \
  tests/ -q --fast --tb=short
git diff --check
```

Expected: static checks, new focused tests, and the fast suite pass.

- [ ] **Step 8: Commit the implementation**

Review and stage only the files in this task:

```bash
git diff -- .readthedocs.yaml docs/Makefile docs/conf.py docs/requirements.txt \
  tests/test_docs_conf.py PROGRESS.md
git add .readthedocs.yaml docs/Makefile docs/conf.py docs/requirements.txt \
  tests/test_docs_conf.py PROGRESS.md
git diff --cached --check
git commit -m "docs: read Sphinx version from pyproject"
```
