# Packaged Setup Gallery CLI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move the runnable setup scripts into the installed VerCOR package and add a Click CLI that copies a selected setup safely and runs a local Python setup file.

**Architecture:** `vercor.setups.gallery` is the single packaged resource owner. A focused `vercor.cli` module resolves gallery files through `importlib.resources`, performs exclusive non-overwriting copies, and launches local scripts with the active Python interpreter in a shell-free subprocess.

**Tech Stack:** Python 3.12+, Click 8.1+, `importlib.resources`, `pathlib`, `subprocess`, pytest, Click `CliRunner`, Flit.

## Global Constraints

- Preserve an existing copy destination and return an error.
- Accept a gallery setup name with or without its `.py` suffix.
- Copy only a direct public `.py` setup from `vercor.setups.gallery` into the current directory.
- Run only an existing local `.py` file with `sys.executable`, without a shell.
- Do not add setup listing, custom copy destinations, overwrite flags, direct bundled execution, or forwarded script arguments.
- Keep `vercor.__all__` unchanged and keep optional model imports lazy.
- Follow strict TDD: record the intended failure before each production behavior.
- Run `pytest tests/ -q --fast` before every commit.

---

### Task 1: Relocate the setup gallery without changing script behavior

**Files:**
- Move: `examples/__init__.py` → `vercor/setups/gallery/__init__.py`
- Move: `examples/custom_component_wrapping.py` → `vercor/setups/gallery/custom_component_wrapping.py`
- Move: `examples/profile_runtime.py` → `vercor/setups/gallery/profile_runtime.py`
- Move: `examples/run_camulator_with_veros.py` → `vercor/setups/gallery/run_camulator_with_veros.py`
- Move: `examples/run_data_driver.py` → `vercor/setups/gallery/run_data_driver.py`
- Move: `examples/run_jcm_with_era5data.py` → `vercor/setups/gallery/run_jcm_with_era5data.py`
- Move: `examples/run_jcm_with_slab.py` → `vercor/setups/gallery/run_jcm_with_slab.py`
- Move: `examples/run_jcm_with_veros.py` → `vercor/setups/gallery/run_jcm_with_veros.py`
- Move: `examples/run_jcm_with_verosdata.py` → `vercor/setups/gallery/run_jcm_with_verosdata.py`
- Move: `examples/run_slab_driver.py` → `vercor/setups/gallery/run_slab_driver.py`
- Move: `examples/run_veros_with_era5data.py` → `vercor/setups/gallery/run_veros_with_era5data.py`
- Modify: `tests/test_jcm_example.py`
- Modify: `tests/test_runtime_run.py`
- Modify: `tests/test_example_jax_helpers.py`
- Modify: `tests/test_api_boundaries.py`
- Modify: `tests/test_runtime_state.py`
- Modify: `tests/test_setup_agnostic_api.py`
- Modify: `.github/workflows/python-package.yml`
- Modify: `tests/test_distribution_boundaries.py`

**Interfaces:**
- Consumes: the existing runnable scripts and their public VerCOR imports.
- Produces: importable modules below `vercor.setups.gallery`; no top-level `examples` package.

- [ ] **Step 1: Point the existing behavioral tests at the future gallery**

In `tests/test_jcm_example.py`, replace the old directory and module owner:

```python
GALLERY_DIRECTORY = (
    Path(__file__).resolve().parents[1] / "vercor" / "setups" / "gallery"
)
GALLERY_MODULE = "vercor.setups.gallery.run_jcm_with_era5data"
```

Use `GALLERY_MODULE` in every `importlib.import_module(...)`, update the
subprocess source to:

```python
from vercor.setups.gallery import run_jcm_with_era5data as example
assert callable(example.build_coupler)
```

and scan `GALLERY_DIRECTORY.glob("*.py")` in the discarded-state regression.
In `tests/test_runtime_run.py`, import
`vercor.setups.gallery.profile_runtime`.

Replace every live `Path("examples/<name>.py")` source inspection in
`tests/test_api_boundaries.py`, `tests/test_runtime_state.py`, and
`tests/test_setup_agnostic_api.py` with
`Path("vercor/setups/gallery/<name>.py")`. Change
`tests/test_example_jax_helpers.py` to:

```python
from vercor.setups.gallery import custom_component_wrapping
```

- [ ] **Step 2: Run the focused tests and record the intended RED**

Run:

```bash
conda run -n scipy pytest tests/test_jcm_example.py tests/test_example_jax_helpers.py tests/test_api_boundaries.py tests/test_runtime_state.py tests/test_setup_agnostic_api.py tests/test_runtime_run.py::test_runtime_profile_harness_exposes_cli_entrypoint -q --tb=short
```

Expected: FAIL because `vercor.setups.gallery` does not exist.

- [ ] **Step 3: Move every file with `apply_patch`**

Use one move hunk per file:

```text
*** Update File: examples/<name>.py
*** Move to: vercor/setups/gallery/<name>.py
```

Do not alter setup logic during the move. Confirm `rg --files examples`
returns no paths and the now-empty directory is absent.

- [ ] **Step 4: Update active quality commands**

Change the three workflow commands to:

```text
python -m black --check vercor tests
python -m mypy vercor tests
python -m compileall -q vercor tests
```

Update the corresponding literals in
`test_ci_quality_job_enforces_static_full_and_coverage_gates`.

- [ ] **Step 5: Run the focused gallery and workflow tests**

Run:

```bash
conda run -n scipy pytest tests/test_jcm_example.py tests/test_example_jax_helpers.py tests/test_api_boundaries.py tests/test_runtime_state.py tests/test_setup_agnostic_api.py tests/test_runtime_run.py::test_runtime_profile_harness_exposes_cli_entrypoint tests/test_distribution_boundaries.py::test_ci_quality_job_enforces_static_full_and_coverage_gates -q --tb=short
```

Expected: PASS.

- [ ] **Step 6: Run the required pre-commit fast suite and commit**

Run:

```bash
conda run -n scipy pytest tests/ -q --fast --tb=short
git diff --check
git add -A examples vercor/setups/gallery tests/test_jcm_example.py tests/test_runtime_run.py tests/test_example_jax_helpers.py tests/test_api_boundaries.py tests/test_runtime_state.py tests/test_setup_agnostic_api.py .github/workflows/python-package.yml tests/test_distribution_boundaries.py
git commit -m "refactor: package runnable setup gallery"
```

Expected: the fast suite passes and Git records renames with no `examples/`
directory left.

---

### Task 2: Add the safe `copy-setup` happy path

**Files:**
- Create: `tests/test_cli.py`
- Create: `vercor/cli.py`
- Modify: `pyproject.toml`

**Interfaces:**
- Consumes: `importlib.resources.files("vercor.setups.gallery")` and the current directory.
- Produces: Click group `cli() -> None` and command `copy_setup(name: str) -> None`.

- [ ] **Step 1: Write the first copy behavior test**

Create `tests/test_cli.py`:

```python
from __future__ import annotations

from importlib import resources
from pathlib import Path

from click.testing import CliRunner

from vercor.cli import cli


def test_copy_setup_by_stem_copies_packaged_bytes() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        result = runner.invoke(cli, ["copy-setup", "run_jcm_with_veros"])
        copied = Path("run_jcm_with_veros.py")
        packaged = resources.files("vercor.setups.gallery").joinpath(copied.name)

        assert result.exit_code == 0, result.output
        assert copied.read_bytes() == packaged.read_bytes()
        assert "run_jcm_with_veros.py" in result.output
```

The break this catches is resolving the wrong package resource, destination, or
copy contents.

- [ ] **Step 2: Run the test and record the intended RED**

Run:

```bash
conda run -n scipy pytest tests/test_cli.py::test_copy_setup_by_stem_copies_packaged_bytes -q --tb=short
```

Expected: FAIL because `vercor.cli` does not exist.

- [ ] **Step 3: Add Click packaging and the minimal copy implementation**

Add to `[project].dependencies`:

```toml
"click>=8.1",
```

Add:

```toml
[project.scripts]
vercor = "vercor.cli:cli"
```

Create `vercor/cli.py` with:

```python
"""Command-line access to packaged VerCOR setup scripts."""

from __future__ import annotations

from importlib import resources
from pathlib import Path
import shutil

import click


@click.group()
def cli() -> None:
    """Copy and run VerCOR setup scripts."""


@cli.command("copy-setup")
@click.argument("name")
def copy_setup(name: str) -> None:
    """Copy bundled setup NAME into the current directory."""

    filename = name if name.endswith(".py") else f"{name}.py"
    source = resources.files("vercor.setups.gallery").joinpath(filename)
    destination = Path.cwd() / filename
    with source.open("rb") as source_stream, destination.open("xb") as target_stream:
        shutil.copyfileobj(source_stream, target_stream)
    click.echo(f"Copied {filename}")


if __name__ == "__main__":
    cli()
```

- [ ] **Step 4: Run the focused test and confirm GREEN**

Run:

```bash
conda run -n scipy pytest tests/test_cli.py::test_copy_setup_by_stem_copies_packaged_bytes -q --tb=short
```

Expected: PASS.

- [ ] **Step 5: Add and pass the explicit-suffix behavior**

Add:

```python
def test_copy_setup_accepts_python_filename() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        result = runner.invoke(cli, ["copy-setup", "run_slab_driver.py"])

        assert result.exit_code == 0, result.output
        assert Path("run_slab_driver.py").is_file()
```

Run the individual test before and after any minimal normalization correction,
then run all of `tests/test_cli.py`.

- [ ] **Step 6: Run the required pre-commit fast suite and commit**

Run:

```bash
conda run -n scipy pytest tests/ -q --fast --tb=short
git diff --check
git add pyproject.toml vercor/cli.py tests/test_cli.py
git commit -m "feat: copy packaged setup from CLI"
```

---

### Task 3: Make setup copying non-destructive and validate names

**Files:**
- Modify: `tests/test_cli.py`
- Modify: `vercor/cli.py`

**Interfaces:**
- Consumes: an untrusted setup name.
- Produces: private `_normalize_setup_name(name: str) -> str`; controlled Click errors; exclusive, cleanup-safe copying.

- [ ] **Step 1: Add the collision regression**

```python
def test_copy_setup_preserves_existing_destination() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        destination = Path("run_jcm_with_veros.py")
        destination.write_text("keep me", encoding="utf-8")

        result = runner.invoke(cli, ["copy-setup", "run_jcm_with_veros"])

        assert result.exit_code != 0
        assert "already exists" in result.output
        assert destination.read_text(encoding="utf-8") == "keep me"
```

Run it and record RED because raw `FileExistsError` does not produce the
required Click diagnostic.

- [ ] **Step 2: Add name-validation regressions**

```python
@pytest.mark.parametrize(
    "name",
    (
        "",
        ".py",
        "../run_slab_driver",
        "nested/run_slab_driver",
        r"nested\run_slab_driver",
        "run_slab_driver.txt",
        "__init__",
        "not_a_setup",
    ),
)
def test_copy_setup_rejects_invalid_or_unknown_name(name: str) -> None:
    result = CliRunner().invoke(cli, ["copy-setup", name])

    assert result.exit_code != 0
```

Add `import pytest`. Run the parameters and confirm the intended failures are
input-boundary failures rather than collection errors.

- [ ] **Step 3: Implement strict normalization and controlled copying**

Add:

```python
def _normalize_setup_name(name: str) -> str:
    if (
        not name
        or name != name.strip()
        or "/" in name
        or "\\" in name
    ):
        raise click.BadParameter("must be a direct setup name", param_hint="NAME")
    suffix = Path(name).suffix
    if suffix not in ("", ".py"):
        raise click.BadParameter("must name a Python setup", param_hint="NAME")
    filename = name if suffix else f"{name}.py"
    if filename in (".py", "__init__.py"):
        raise click.BadParameter("must name a Python setup", param_hint="NAME")
    return filename
```

Refactor `copy_setup` to verify `source.is_file()`, open the destination with
`"xb"`, translate `FileExistsError` and other `OSError` failures to
`click.ClickException`, and delete only a partially created destination:

```python
created = False
try:
    with source.open("rb") as source_stream:
        with destination.open("xb") as target_stream:
            created = True
            shutil.copyfileobj(source_stream, target_stream)
except FileExistsError as error:
    raise click.ClickException(f"{filename} already exists") from error
except OSError as error:
    if created:
        destination.unlink(missing_ok=True)
    raise click.ClickException(f"could not copy {filename}: {error}") from error
```

Unknown resources raise `click.ClickException(f"unknown setup: {name}")` before
opening either path.

- [ ] **Step 4: Run the CLI tests and confirm GREEN**

Run:

```bash
conda run -n scipy pytest tests/test_cli.py -q --tb=short
```

Expected: all copy and validation tests pass.

- [ ] **Step 5: Run the required pre-commit fast suite and commit**

Run:

```bash
conda run -n scipy pytest tests/ -q --fast --tb=short
git diff --check
git add vercor/cli.py tests/test_cli.py
git commit -m "fix: preserve existing setup copies"
```

---

### Task 4: Add the `run` command

**Files:**
- Modify: `tests/test_cli.py`
- Modify: `vercor/cli.py`

**Interfaces:**
- Consumes: existing readable local `Path` ending in `.py`.
- Produces: `run(setup_file: Path) -> None`, child execution with `sys.executable`, and exact nonzero status propagation.

- [ ] **Step 1: Write the real execution behavior test**

```python
def test_run_executes_python_file_with_current_interpreter() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        Path("setup.py").write_text(
            "from pathlib import Path\n"
            "import sys\n"
            "Path('interpreter.txt').write_text(sys.executable, encoding='utf-8')\n",
            encoding="utf-8",
        )

        result = runner.invoke(cli, ["run", "setup.py"])

        assert result.exit_code == 0, result.output
        assert Path("interpreter.txt").read_text(encoding="utf-8") == sys.executable
```

Add `import sys`. Run it and record RED because `run` is not registered.

- [ ] **Step 2: Implement minimal child-process execution**

Add `import subprocess` and `import sys`, then:

```python
@cli.command("run")
@click.argument(
    "setup_file",
    type=click.Path(
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        path_type=Path,
    ),
)
def run(setup_file: Path) -> None:
    """Run local Python SETUP_FILE with the active interpreter."""

    if setup_file.suffix != ".py":
        raise click.BadParameter(
            "must be a .py file",
            param_hint="SETUP_FILE",
        )
    completed = subprocess.run(
        [sys.executable, str(setup_file)],
        check=False,
    )
    if completed.returncode:
        raise click.exceptions.Exit(completed.returncode)
```

- [ ] **Step 3: Confirm the execution test is GREEN**

Run:

```bash
conda run -n scipy pytest tests/test_cli.py::test_run_executes_python_file_with_current_interpreter -q --tb=short
```

- [ ] **Step 4: Add exit and invalid-target tests**

```python
def test_run_propagates_script_exit_status() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        Path("setup.py").write_text("raise SystemExit(7)\n", encoding="utf-8")

        result = runner.invoke(cli, ["run", "setup.py"])

        assert result.exit_code == 7


@pytest.mark.parametrize("target", ("missing.py", "notes.txt"))
def test_run_rejects_missing_or_non_python_file(target: str) -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        if target == "notes.txt":
            Path(target).write_text("text", encoding="utf-8")

        result = runner.invoke(cli, ["run", target])

        assert result.exit_code != 0


def test_run_rejects_directory() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        Path("setup.py").mkdir()

        result = runner.invoke(cli, ["run", "setup.py"])

        assert result.exit_code != 0
```

Run each new behavior first to observe RED if the current boundary does not
already enforce it, then run the whole CLI file.

- [ ] **Step 5: Verify help exposes only the intended commands**

Add:

```python
def test_cli_help_lists_copy_and_run_commands() -> None:
    result = CliRunner().invoke(cli, ["--help"])

    assert result.exit_code == 0
    assert "copy-setup" in result.output
    assert "run" in result.output
```

Run `pytest tests/test_cli.py -q --tb=short`.

- [ ] **Step 6: Run the required pre-commit fast suite and commit**

Run:

```bash
conda run -n scipy pytest tests/ -q --fast --tb=short
git diff --check
git add vercor/cli.py tests/test_cli.py
git commit -m "feat: run local setup files from CLI"
```

---

### Task 5: Protect installed artifacts and document the workflow

**Files:**
- Modify: `tests/test_distribution_boundaries.py`
- Modify: `README.md`
- Modify: `docs/how-to/examples.rst`
- Modify: `docs/releasing.md`
- Modify: `DESIGN.md`
- Modify: `DEPENDENCIES.md`
- Modify: `CHANGELOG.md`
- Modify: `AGENTS.md`
- Modify: `docs/api-architecture-review.md` if its active inventory mentions the old gallery location
- Modify: `docs/superpowers/plans/2026-07-29-packaged-setup-gallery-cli.md`

**Interfaces:**
- Consumes: Flit metadata and the built wheel/source distribution.
- Produces: installed `vercor` console entry point, packaged gallery resources, and current user/developer instructions.

- [x] **Step 1: Extend the built-artifact regression and record RED**

In `test_built_distributions_run_external_extension_fixture_outside_checkout`,
define the ten public script names as literal expected inventory and assert:

```python
gallery_names = {
    "vercor/setups/gallery/custom_component_wrapping.py",
    "vercor/setups/gallery/profile_runtime.py",
    "vercor/setups/gallery/run_camulator_with_veros.py",
    "vercor/setups/gallery/run_data_driver.py",
    "vercor/setups/gallery/run_jcm_with_era5data.py",
    "vercor/setups/gallery/run_jcm_with_slab.py",
    "vercor/setups/gallery/run_jcm_with_veros.py",
    "vercor/setups/gallery/run_jcm_with_verosdata.py",
    "vercor/setups/gallery/run_slab_driver.py",
    "vercor/setups/gallery/run_veros_with_era5data.py",
}
assert gallery_names.issubset(wheel_names)
entry_points_name = next(
    name for name in wheel_names if name.endswith(".dist-info/entry_points.txt")
)
entry_points = wheel.read(entry_points_name).decode("utf-8")
assert "vercor = vercor.cli:cli" in entry_points
```

Assert the corresponding version-prefixed paths are in `sdist_names`. Run the
test before rebuilding and record the intended failure against the old cached
or pre-change artifact state.

- [x] **Step 2: Add an installed-resource copy probe**

After `install_local_target(...)`, run a child interpreter outside the checkout
with `PYTHONPATH=target`:

```python
probe = subprocess.run(
    [
        sys.executable,
        "-m",
        "vercor.cli",
        "copy-setup",
        "custom_component_wrapping",
    ],
    cwd=tmp_path,
    env=environment,
    check=True,
    capture_output=True,
    text=True,
)
assert (tmp_path / "custom_component_wrapping.py").is_file()
assert "custom_component_wrapping.py" in probe.stdout
```

This checks the installed package resource without importing from the checkout.

- [x] **Step 3: Update user documentation**

In `README.md`, replace the `Examples` repository link with a short Setup
gallery section showing:

```console
vercor copy-setup run_jcm_with_veros
vercor run run_jcm_with_veros.py
```

In `docs/how-to/examples.rst`, explain that setup scripts are packaged, copying
is non-overwriting, and the copied script is user-editable. Keep the dependency
table, rename its first column from module to setup, and use the same two CLI
commands as the canonical workflow.

- [x] **Step 4: Update active development and architecture records**

Replace active `examples` validation paths in `docs/releasing.md` with
`vercor tests`. Make the same source-path correction in `AGENTS.md`. Update:

- `DESIGN.md` section 9 to identify `vercor.setups.gallery` and `vercor.cli`;
- `DEPENDENCIES.md` to replace the old `examples/` layer with the gallery and
  CLI ownership;
- `CHANGELOG.md` with an unreleased entry for the packaged gallery and CLI;
- active references in `docs/api-architecture-review.md`; and
- this plan's checkboxes as tasks complete.

Do not rewrite archived progress or historical specifications.

- [x] **Step 5: Run focused documentation, architecture, CLI, and artifact tests**

Run:

```bash
conda run -n scipy pytest tests/test_cli.py tests/test_jcm_example.py tests/test_docs_build.py tests/test_distribution_boundaries.py -q --tb=short
```

Expected: PASS, including a fresh distribution build and installed-resource
probe.

- [x] **Step 6: Run static checks**

Run:

```bash
conda run -n scipy black vercor tests
conda run -n scipy flake8 . --count --max-line-length=120 --statistics
conda run -n scipy mypy vercor tests
conda run -n scipy python -m compileall -q vercor tests
git diff --check
```

Expected: all commands exit zero.

---

### Task 6: Complete verification, record progress, and commit

**Files:**
- Modify: `PROGRESS.md`
- Modify: `docs/superpowers/plans/2026-07-29-packaged-setup-gallery-cli.md`

**Interfaces:**
- Consumes: the final source tree and all project quality gates.
- Produces: concise durable verification evidence and a clean committed tree.

- [ ] **Step 1: Run the final fast suite**

Run:

```bash
conda run -n scipy pytest tests/ -q --fast --tb=short
```

Record exact pass/warning counts.

- [ ] **Step 2: Run the final full suite**

Run:

```bash
conda run -n scipy pytest tests/ -q --tb=short
```

Record exact pass/warning counts.

- [ ] **Step 3: Run branch coverage**

Run:

```bash
conda run -n scipy pytest tests/ -q --cov=vercor --cov-branch --cov-report=term-missing --cov-fail-under=90 --tb=short
```

Record the exact percentage and test counts.

- [ ] **Step 4: Build fresh distributions**

Run:

```bash
conda run -n scipy python -m build
conda run -n scipy pytest tests/test_distribution_boundaries.py -q --tb=short
```

Confirm the wheel and source distribution contain the gallery and console
entry-point metadata and the installed probe copies a resource outside the
checkout.

- [ ] **Step 5: Update the durable progress record**

Add a dated top entry to `PROGRESS.md` summarizing:

- the gallery relocation and removed `examples/`;
- `copy-setup` normalization, exclusive collision behavior, and cleanup;
- `run` interpreter and exit-code semantics;
- documentation and artifact coverage; and
- exact static, fast, full, coverage, build, and installed-probe results.

Mark every completed checkbox in this plan.

- [ ] **Step 6: Invoke verification-before-completion and re-run its required evidence**

Use `superpowers:verification-before-completion`. Do not claim success from
earlier output if any final source or documentation file changed afterward.

- [ ] **Step 7: Stage intentionally, inspect, and commit**

Run:

```bash
git status --short
git diff --stat
git diff --check
git add pyproject.toml vercor tests .github/workflows/python-package.yml AGENTS.md README.md CHANGELOG.md DESIGN.md DEPENDENCIES.md PROGRESS.md docs
git diff --cached --stat
git diff --cached --check
git commit -m "feat: add packaged setup gallery CLI"
git status --short
```

Expected: the commit succeeds and the working tree is clean.

- [ ] **Step 8: Report the result**

Return the committed hashes, command examples, moved gallery location, collision
policy, verification counts, and any known third-party warnings. Do not push,
tag, publish, or create a pull request.
