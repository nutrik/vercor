# Expanded Packaged Setup Gallery CLI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expand the installed VerCOR CLI with deterministic setup discovery, external setup directories, safe destination-directory copying, setup listing, version reporting, and isolated runtime logging and precision controls.

**Architecture:** `vercor.cli` owns Click presentation, a single setup catalog, duplicate detection, and non-overwriting copies. A private `vercor._setup_runner` loads an explicit `run_setup(*, loglevel, float_type)` contract in a child interpreter; bundled gallery scripts translate those values into `Coupler` logging and `RuntimeOptions` precision without hidden global overrides.

**Tech Stack:** Python 3.12+, Click 8.1+, `importlib.resources`, `runpy`, `pathlib`, `subprocess`, pytest, Click `CliRunner`, Black, flake8, mypy, Flit/build.

> **Final-review corrections (2026-07-30):** The final whole-branch reviewer
> governs the corrected runtime, catalog, artifact, and lowercase-choice
> instructions recorded below. They supersede the earlier task snippets where
> those snippets conflict.

## Global Constraints

- Work only in the isolated `feat/packaged-setup-gallery-cli` worktree and preserve the existing draft pull request.
- The root description is exactly `Vercor command-line tools`.
- `VERCOR_SETUP_DIR` is split with `os.pathsep`; discovery is direct, non-recursive, public `.py` files only.
- Reject duplicate setup stems across all bundled and external sources; never apply source precedence.
- `--to` names a directory, creates missing parents, reuses an existing directory, and never overwrites an existing setup file.
- `run_setup(*, loglevel: str, float_type: str) -> int | None` is the required setup-file contract.
- `--loglevel` choices are exactly `trace`, `debug`, `info`, `warning`, and `error`, defaulting to `info`.
- `--float-type` choices are exactly `float64` and `float32`, defaulting to `float64`.
- `trace` maps to logging level 5; float choices map to `DTypePolicy(enable_x64=True/False)`.
- Run setup code with `sys.executable` in a shell-free child process; preserve
  its integer exit status. Resolve the private runner from the already imported
  package and execute that file with Python `-P` safe-path isolation.
- Keep the setup file's parent at the front of `sys.path` for both loading and
  `run_setup` invocation, then restore the original list in `finally`.
- Resolve copy references against both each catalog stem and its canonical
  `.py` filename; validate unmatched references without rejecting cataloged
  dotted stems.
- Keep `vercor.__all__` unchanged and do not import optional model dependencies from `vercor.cli`.
- Follow strict TDD and run `conda run -n scipy pytest tests/ -q --fast --tb=short` before each implementation commit.

---

### Task 1: Centralize setup discovery and add listing/version help

**Files:**
- Modify: `tests/test_cli.py`
- Modify: `vercor/cli.py`

**Interfaces:**
- Consumes: `importlib.resources.files("vercor.setups.gallery")` and optional `VERCOR_SETUP_DIR`.
- Produces: `_SetupTemplate`, `_discover_setups() -> tuple[_SetupTemplate, ...]`, `show_setups() -> None`, dynamic copy help, and root `--version`.

- [ ] **Step 1: Write failing root-help, version, and listing tests**

Add imports:

```python
from collections.abc import Iterator
from importlib import metadata
import os
```

Add:

```python
def test_cli_help_exposes_required_description_options_and_commands() -> None:
    result = CliRunner().invoke(cli, ["--help"])

    assert result.exit_code == 0, result.output
    assert result.output == (
        "Usage: vercor [OPTIONS] COMMAND [ARGS]...\n"
        "\n"
        "  Vercor command-line tools\n"
        "\n"
        "Options:\n"
        "  --version  Show the version and exit.\n"
        "  --help     Show this message and exit.\n"
        "\n"
        "Commands:\n"
        "  copy-setup   Copy a standard setup to another directory.\n"
        "  run          Runs a Vercor setup from given file.\n"
        "  show-setups  Print a list of available pre-configured setups.\n"
    )


def test_cli_version_reports_installed_distribution_version() -> None:
    result = CliRunner().invoke(cli, ["--version"])

    assert result.exit_code == 0, result.output
    assert result.output.strip() == (
        f"vercor, version {metadata.version('vercor')}"
    )


def test_show_setups_lists_sorted_public_bundled_stems() -> None:
    result = CliRunner().invoke(cli, ["show-setups"])

    assert result.exit_code == 0, result.output
    names = result.output.splitlines()
    assert names == sorted(names)
    assert "run_jcm_with_veros" in names
    assert "__init__" not in names
```

Replace the old two-command help assertion with the new root-help test.

- [ ] **Step 2: Run the three tests and record RED**

Run:

```bash
conda run -n scipy pytest \
  tests/test_cli.py::test_cli_help_exposes_required_description_options_and_commands \
  tests/test_cli.py::test_cli_version_reports_installed_distribution_version \
  tests/test_cli.py::test_show_setups_lists_sorted_public_bundled_stems \
  -q --tb=short
```

Expected: FAIL because the description and version are absent and
`show-setups` is not registered.

- [ ] **Step 3: Add external-directory, direct-file, and stable-order tests**

Add:

```python
def test_show_setups_adds_direct_public_files_from_each_external_directory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    (first / "zeta.py").write_text("", encoding="utf-8")
    (first / "_private.py").write_text("", encoding="utf-8")
    (first / "notes.txt").write_text("", encoding="utf-8")
    (first / "nested").mkdir()
    (first / "nested" / "ignored.py").write_text("", encoding="utf-8")
    (second / "alpha.py").write_text("", encoding="utf-8")
    monkeypatch.setenv("VERCOR_SETUP_DIR", os.pathsep.join((str(first), str(second))))

    result = CliRunner().invoke(cli, ["show-setups"])

    assert result.exit_code == 0, result.output
    names = result.output.splitlines()
    assert names == sorted(names)
    assert "alpha" in names
    assert "zeta" in names
    assert "_private" not in names
    assert "ignored" not in names
```

- [ ] **Step 4: Add invalid-source and duplicate diagnostics tests**

Add:

```python
def test_show_setups_rejects_missing_external_directory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    missing = tmp_path / "missing"
    monkeypatch.setenv("VERCOR_SETUP_DIR", str(missing))

    result = CliRunner().invoke(cli, ["show-setups"])

    assert result.exit_code == 1
    assert "setup directory does not exist" in result.output
    assert str(missing) in result.output


def test_show_setups_rejects_non_directory_external_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source_file = tmp_path / "setups.txt"
    source_file.write_text("", encoding="utf-8")
    monkeypatch.setenv("VERCOR_SETUP_DIR", str(source_file))

    result = CliRunner().invoke(cli, ["show-setups"])

    assert result.exit_code == 1
    assert "setup path is not a directory" in result.output


def test_show_setups_reports_unreadable_external_directory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    unreadable = tmp_path / "unreadable"
    unreadable.mkdir()
    original_iterdir = Path.iterdir

    def failing_iterdir(path: Path) -> Iterator[Path]:
        if path == unreadable:
            raise OSError("permission denied")
        return original_iterdir(path)

    monkeypatch.setattr(Path, "iterdir", failing_iterdir)
    monkeypatch.setenv("VERCOR_SETUP_DIR", str(unreadable))

    result = CliRunner().invoke(cli, ["show-setups"])

    assert result.exit_code == 1
    assert f"could not read setup directory {unreadable}" in result.output
    assert "permission denied" in result.output


def test_show_setups_rejects_duplicate_names_and_reports_sources(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    (first / "custom.py").write_text("", encoding="utf-8")
    (second / "custom.py").write_text("", encoding="utf-8")
    monkeypatch.setenv("VERCOR_SETUP_DIR", os.pathsep.join((str(first), str(second))))

    result = CliRunner().invoke(cli, ["show-setups"])

    assert result.exit_code == 1
    assert "duplicate setup: custom" in result.output
    assert str(first / "custom.py") in result.output
    assert str(second / "custom.py") in result.output
```

Also add a packaged/external collision case using
`run_jcm_with_veros.py` and assert that both `vercor.setups.gallery` and the
external path appear.

- [ ] **Step 5: Implement the shared setup catalog and root commands**

In `vercor/cli.py`, add:

```python
from dataclasses import dataclass
from importlib.resources.abc import Traversable
import os


@dataclass(frozen=True)
class _SetupTemplate:
    """Describe one uniquely named copyable setup template."""

    stem: str
    filename: str
    source: Traversable
    origin: str


def _public_python_file(name: str) -> bool:
    return (
        name.endswith(".py")
        and name != "__init__.py"
        and not name.startswith("_")
    )


def _external_setup_directories() -> tuple[Path, ...]:
    raw_value = os.environ.get("VERCOR_SETUP_DIR", "")
    return tuple(
        Path(value).expanduser()
        for value in raw_value.split(os.pathsep)
        if value
    )


def _discover_setups() -> tuple[_SetupTemplate, ...]:
    candidates: list[_SetupTemplate] = []
    packaged = resources.files("vercor.setups.gallery")
    try:
        packaged_children = tuple(packaged.iterdir())
    except OSError as error:
        raise click.ClickException(
            f"could not read packaged setup gallery: {error}"
        ) from error
    for source in packaged_children:
        if source.is_file() and _public_python_file(source.name):
            candidates.append(
                _SetupTemplate(
                    stem=Path(source.name).stem,
                    filename=source.name,
                    source=source,
                    origin=f"vercor.setups.gallery/{source.name}",
                )
            )

    for directory in _external_setup_directories():
        if not directory.exists():
            raise click.ClickException(
                f"setup directory does not exist: {directory}"
            )
        if not directory.is_dir():
            raise click.ClickException(
                f"setup path is not a directory: {directory}"
            )
        try:
            children = tuple(directory.iterdir())
        except OSError as error:
            raise click.ClickException(
                f"could not read setup directory {directory}: {error}"
            ) from error
        for source in children:
            if source.is_file() and _public_python_file(source.name):
                candidates.append(
                    _SetupTemplate(
                        stem=source.stem,
                        filename=source.name,
                        source=source,
                        origin=str(source),
                    )
                )

    by_stem: dict[str, list[_SetupTemplate]] = {}
    for candidate in candidates:
        by_stem.setdefault(candidate.stem, []).append(candidate)
    duplicates = {
        stem: templates
        for stem, templates in by_stem.items()
        if len(templates) > 1
    }
    if duplicates:
        details = "; ".join(
            f"{stem}: {', '.join(item.origin for item in templates)}"
            for stem, templates in sorted(duplicates.items())
        )
        raise click.ClickException(f"duplicate setup: {details}")
    return tuple(sorted(candidates, key=lambda item: item.stem))
```

Decorate the group and add listing:

```python
@click.group(name="vercor", help="Vercor command-line tools")
@click.version_option(package_name="vercor")
def cli() -> None:
    """Provide Vercor command-line tools."""


@cli.command("show-setups")
def show_setups() -> None:
    """Print a list of available pre-configured setups."""

    for setup in _discover_setups():
        click.echo(setup.stem)
```

Keep helpers private and keep `__all__ == ("cli",)`.

- [ ] **Step 6: Add and pass dynamic copy-help coverage**

Add:

```python
def test_copy_help_lists_discovered_setups_and_environment_guidance(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    external = tmp_path / "external"
    external.mkdir()
    (external / "local_template.py").write_text("", encoding="utf-8")
    monkeypatch.setenv("VERCOR_SETUP_DIR", str(external))

    result = CliRunner().invoke(cli, ["copy-setup", "--help"])

    assert result.exit_code == 0, result.output
    assert "Available setups:" in result.output
    assert "run_jcm_with_veros" in result.output
    assert "local_template" in result.output
    assert "VERCOR_SETUP_DIR" in result.output
    assert "vercor copy-setup run_jcm_with_veros --to" in result.output
```

Implement a private Click command subclass:

```python
class _CopySetupCommand(click.Command):
    """Render the live setup catalog in copy command help."""

    def format_epilog(
        self,
        ctx: click.Context,
        formatter: click.HelpFormatter,
    ) -> None:
        formatter.write_paragraph()
        formatter.write_heading("Available setups")
        formatter.write_text("\n".join(item.stem for item in _discover_setups()))
        formatter.write_paragraph()
        formatter.write_text(
            "Example:\n"
            "    $ vercor copy-setup run_jcm_with_veros --to "
            "~/vercor-setups/run_jcm_with_veros\n\n"
            "Further directories containing setup templates can be added "
            "via the VERCOR_SETUP_DIR environment variable."
        )
```

Register `copy-setup` with `cls=_CopySetupCommand`.

- [ ] **Step 7: Run focused tests, the required fast suite, and commit**

Run:

```bash
conda run -n scipy pytest tests/test_cli.py -q --tb=short
conda run -n scipy pytest tests/ -q --fast --tb=short
git diff --check
git add vercor/cli.py tests/test_cli.py
git commit -m "feat: discover and list setup templates"
```

Expected: the complete CLI file and fast suite pass.

---

### Task 2: Add safe destination-directory copying

**Files:**
- Modify: `tests/test_cli.py`
- Modify: `vercor/cli.py`

**Interfaces:**
- Consumes: `_discover_setups()` and `_normalize_setup_name(name: str) -> str`.
- Produces: `copy_setup(setup: str, destination: Path) -> None` with directory creation and exclusive file creation.

- [ ] **Step 1: Add failing tests for new and existing `--to` directories**

Add:

```python
def test_copy_setup_creates_destination_directory_and_parents() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        destination = Path("new/target")

        result = runner.invoke(
            cli,
            ["copy-setup", "run_jcm_with_veros", "--to", str(destination)],
        )

        assert result.exit_code == 0, result.output
        assert (destination / "run_jcm_with_veros.py").is_file()


def test_copy_setup_reuses_existing_destination_directory() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        destination = Path("existing")
        destination.mkdir()

        result = runner.invoke(
            cli,
            ["copy-setup", "run_jcm_with_veros.py", "--to", str(destination)],
        )

        assert result.exit_code == 0, result.output
        assert (destination / "run_jcm_with_veros.py").is_file()
```

Run both and expect RED because `--to` is not defined.

- [ ] **Step 2: Add external-copy and canonical-name coverage**

Add:

```python
def test_copy_setup_copies_external_template_bytes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    external = tmp_path / "external"
    external.mkdir()
    source = external / "local_template.py"
    source.write_bytes(b"VALUE = 42\n")
    destination = tmp_path / "copied"
    monkeypatch.setenv("VERCOR_SETUP_DIR", str(external))

    result = CliRunner().invoke(
        cli,
        ["copy-setup", "local_template", "--to", str(destination)],
    )

    assert result.exit_code == 0, result.output
    assert (destination / "local_template.py").read_bytes() == source.read_bytes()
```

- [ ] **Step 3: Add invalid-target and collision-preservation tests**

Add:

```python
def test_copy_setup_rejects_file_as_destination_directory() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        target = Path("target")
        target.write_text("keep", encoding="utf-8")

        result = runner.invoke(
            cli,
            ["copy-setup", "run_jcm_with_veros", "--to", str(target)],
        )

        assert result.exit_code != 0
        assert target.read_text(encoding="utf-8") == "keep"


def test_copy_setup_preserves_existing_file_inside_to_directory() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        directory = Path("target")
        directory.mkdir()
        destination = directory / "run_jcm_with_veros.py"
        destination.write_text("keep me", encoding="utf-8")

        result = runner.invoke(
            cli,
            ["copy-setup", "run_jcm_with_veros", "--to", str(directory)],
        )

        assert result.exit_code == 1
        assert "already exists" in result.output
        assert destination.read_text(encoding="utf-8") == "keep me"
```

Retain the existing default-current-directory, name-validation, unknown-name,
and cleanup regressions.

- [ ] **Step 4: Implement catalog lookup and directory semantics**

Change the command to:

```python
@cli.command("copy-setup", cls=_CopySetupCommand)
@click.argument("setup")
@click.option(
    "--to",
    "destination",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("."),
    help="Target directory (default: current working directory).",
)
def _copy_setup(setup: str, destination: Path) -> None:
    """Copy a standard setup to another directory."""

    _validate_setup_reference(setup)
    catalog: dict[str, list[_SetupTemplate]] = {}
    for item in _discover_setups():
        for reference in (item.stem, item.filename):
            catalog.setdefault(reference, []).append(item)
    matches = catalog.get(setup, [])
    if not matches:
        _normalize_setup_name(setup)
        raise click.ClickException(f"unknown setup: {setup}")
    if len(matches) > 1:
        details = ", ".join(item.origin for item in matches)
        raise click.ClickException(
            f"ambiguous setup reference: {setup}: {details}"
        )
    template = matches[0]

    try:
        destination.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise click.ClickException(
            f"could not create target directory {destination}: {error}"
        ) from error
    if not destination.is_dir():
        raise click.ClickException(
            f"target path is not a directory: {destination}"
        )

    target = destination / template.filename
    created = False
    try:
        try:
            with template.source.open("rb") as source_stream:
                with target.open("xb") as target_stream:
                    created = True
                    shutil.copyfileobj(source_stream, target_stream)
        except BaseException:
            if created:
                target.unlink(missing_ok=True)
            raise
    except FileExistsError as error:
        raise click.ClickException(f"{target} already exists") from error
    except OSError as error:
        raise click.ClickException(
            f"could not copy {template.filename}: {error}"
        ) from error
    click.echo(f"Copied {template.filename} to {target}")
```

Update `_normalize_setup_name` diagnostics to use `SETUP` instead of `NAME`
and ensure the existing private-name behavior remains.

- [ ] **Step 5: Run focused tests, the required fast suite, and commit**

Run:

```bash
conda run -n scipy pytest tests/test_cli.py -q --tb=short
conda run -n scipy pytest tests/ -q --fast --tb=short
git diff --check
git add vercor/cli.py tests/test_cli.py
git commit -m "feat: copy setups to destination directories"
```

Expected: all copy behaviors pass without changing an existing file.

---

### Task 3: Add the isolated setup runner and runtime options

**Files:**
- Create: `vercor/_setup_runner.py`
- Create: `tests/test_setup_runner.py`
- Modify: `tests/test_cli.py`
- Modify: `vercor/cli.py`

**Interfaces:**
- Consumes: an existing resolved `.py` path, `loglevel: str`, and `float_type: str`.
- Produces: `_invoke_setup(path: Path, *, loglevel: str, float_type: str) -> int`, `main(argv: Sequence[str] | None = None) -> int`, and Click `run`.

- [ ] **Step 1: Write failing runner contract tests**

Create `tests/test_setup_runner.py`:

```python
from __future__ import annotations

from pathlib import Path

import pytest

from vercor import _setup_runner


def test_invoke_setup_passes_keyword_options_and_none_means_success(
    tmp_path: Path,
) -> None:
    setup = tmp_path / "setup.py"
    setup.write_text(
        "def run_setup(*, loglevel, float_type):\n"
        "    assert loglevel == 'debug'\n"
        "    assert float_type == 'float32'\n",
        encoding="utf-8",
    )

    assert (
        _setup_runner._invoke_setup(
            setup,
            loglevel="debug",
            float_type="float32",
        )
        == 0
    )


def test_invoke_setup_returns_integer_status(tmp_path: Path) -> None:
    setup = tmp_path / "setup.py"
    setup.write_text(
        "def run_setup(*, loglevel, float_type):\n"
        "    return 7\n",
        encoding="utf-8",
    )

    assert (
        _setup_runner._invoke_setup(
            setup,
            loglevel="info",
            float_type="float64",
        )
        == 7
    )
```

Run and expect import failure because `vercor._setup_runner` does not exist.

- [ ] **Step 2: Add missing-contract, invalid-return, main-guard, and local-import tests**

Add:

```python
@pytest.mark.parametrize(
    "source",
    (
        "VALUE = 1\n",
        "run_setup = 3\n",
    ),
)
def test_invoke_setup_requires_callable_contract(
    source: str,
    tmp_path: Path,
) -> None:
    setup = tmp_path / "setup.py"
    setup.write_text(source, encoding="utf-8")

    with pytest.raises(_setup_runner.SetupContractError, match="run_setup"):
        _setup_runner._invoke_setup(
            setup,
            loglevel="info",
            float_type="float64",
        )


@pytest.mark.parametrize("value", ("True", "'bad'", "object()"))
def test_invoke_setup_rejects_non_status_return(
    value: str,
    tmp_path: Path,
) -> None:
    setup = tmp_path / "setup.py"
    setup.write_text(
        "def run_setup(*, loglevel, float_type):\n"
        f"    return {value}\n",
        encoding="utf-8",
    )

    with pytest.raises(_setup_runner.SetupContractError, match="int or None"):
        _setup_runner._invoke_setup(
            setup,
            loglevel="info",
            float_type="float64",
        )


def test_invoke_setup_skips_main_guard_and_supports_adjacent_imports(
    tmp_path: Path,
) -> None:
    (tmp_path / "helper.py").write_text("VALUE = 9\n", encoding="utf-8")
    marker = tmp_path / "marker.txt"
    setup = tmp_path / "setup.py"
    setup.write_text(
        "from helper import VALUE\n"
        "from pathlib import Path\n"
        "if __name__ == '__main__':\n"
        "    raise AssertionError('main guard ran')\n"
        "def run_setup(*, loglevel, float_type):\n"
        f"    Path({str(marker)!r}).write_text(str(VALUE), encoding='utf-8')\n",
        encoding="utf-8",
    )

    status = _setup_runner._invoke_setup(
        setup,
        loglevel="info",
        float_type="float64",
    )

    assert status == 0
    assert marker.read_text(encoding="utf-8") == "9"
```

- [ ] **Step 3: Implement the private runner**

Create:

```python
"""Private child-process runner for VerCOR setup-file contracts."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
import runpy
import sys
from typing import Any


class SetupContractError(ValueError):
    """Report a setup file that does not implement the runner contract."""


def _load_setup(path: Path) -> dict[str, Any]:
    return runpy.run_path(str(path), run_name="_vercor_setup")


def _invoke_setup(
    path: Path,
    *,
    loglevel: str,
    float_type: str,
) -> int:
    original_sys_path = list(sys.path)
    sys.path.insert(0, str(path.parent))
    try:
        namespace = _load_setup(path)
        run_setup = namespace.get("run_setup")
        if not callable(run_setup):
            raise SetupContractError(
                f"{path} must define callable run_setup(*, loglevel, float_type)"
            )
        _validate_setup_signature(path, run_setup)
        result = run_setup(loglevel=loglevel, float_type=float_type)
        if result is None:
            return 0
        if isinstance(result, bool) or not isinstance(result, int):
            raise SetupContractError("run_setup must return int or None")
        return result
    finally:
        sys.path[:] = original_sys_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vercor setup runner")
    parser.add_argument("setup_file", type=Path)
    parser.add_argument(
        "--loglevel",
        required=True,
        choices=("trace", "debug", "info", "warning", "error"),
    )
    parser.add_argument(
        "--float-type",
        required=True,
        choices=("float64", "float32"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        return _invoke_setup(
            args.setup_file,
            loglevel=args.loglevel,
            float_type=args.float_type,
        )
    except SetupContractError as error:
        parser.exit(2, f"Error: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
```

Run `tests/test_setup_runner.py` and confirm GREEN.

- [ ] **Step 4: Replace direct-script CLI tests with contract-process tests**

In `tests/test_cli.py`, change temporary run scripts to define `run_setup`.
Add an option-propagation test:

```python
def test_run_passes_selected_loglevel_and_float_type() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        Path("setup.py").write_text(
            "from pathlib import Path\n"
            "def run_setup(*, loglevel, float_type):\n"
            "    Path('options.txt').write_text("
            "f'{loglevel},{float_type}', encoding='utf-8')\n",
            encoding="utf-8",
        )

        result = runner.invoke(
            cli,
            [
                "run",
                "--loglevel",
                "warning",
                "--float-type",
                "float32",
                "setup.py",
            ],
        )

        assert result.exit_code == 0, result.output
        assert Path("options.txt").read_text(encoding="utf-8") == (
            "warning,float32"
        )
```

Add a default-options case asserting `info,float64`, parametrize every allowed
choice, and retain current-interpreter, option-like filename, missing file,
directory, `.py` suffix, exception, and integer-status coverage.

Add explicit help coverage:

```python
def test_run_help_lists_runtime_options_and_defaults() -> None:
    result = CliRunner().invoke(cli, ["run", "--help"])

    assert result.exit_code == 0, result.output
    assert "-v, --loglevel [trace|debug|info|warning|error]" in result.output
    assert "[default: info]" in result.output
    assert "--float-type [float64|float32]" in result.output
    assert "[default: float64]" in result.output
```

- [ ] **Step 5: Implement Click runtime choices and private-runner subprocess**

Replace `run` with:

```python
_LOG_LEVELS = ("trace", "debug", "info", "warning", "error")
_FLOAT_TYPES = ("float64", "float32")


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
@click.option(
    "-v",
    "--loglevel",
    type=click.Choice(_LOG_LEVELS, case_sensitive=True),
    default="info",
    show_default=True,
)
@click.option(
    "--float-type",
    type=click.Choice(_FLOAT_TYPES, case_sensitive=True),
    default="float64",
    show_default=True,
)
def run(setup_file: Path, loglevel: str, float_type: str) -> None:
    """Runs a Vercor setup from given file."""

    if setup_file.suffix != ".py":
        raise click.BadParameter("must be a .py file", param_hint="SETUP_FILE")
    runner_path = Path(__file__).with_name("_setup_runner.py").resolve()
    completed = subprocess.run(
        [
            sys.executable,
            "-P",
            str(runner_path),
            str(setup_file.resolve()),
            "--loglevel",
            loglevel,
            "--float-type",
            float_type,
        ],
        check=False,
    )
    if completed.returncode:
        raise click.exceptions.Exit(completed.returncode)
```

- [ ] **Step 6: Run focused tests, the required fast suite, and commit**

Run:

```bash
conda run -n scipy pytest tests/test_setup_runner.py tests/test_cli.py -q --tb=short
conda run -n scipy pytest tests/ -q --fast --tb=short
git diff --check
git add vercor/_setup_runner.py vercor/cli.py tests/test_setup_runner.py tests/test_cli.py
git commit -m "feat: run explicit setup contracts"
```

Expected: runner and CLI tests pass, including exact child statuses.

---

### Task 4: Support trace logging and adopt the contract in testable gallery modules

**Files:**
- Modify: `vercor/_logging/config.py`
- Modify: `tests/test_coupler_coverage.py`
- Modify: `vercor/setups/gallery/custom_component_wrapping.py`
- Modify: `vercor/setups/gallery/profile_runtime.py`
- Modify: `vercor/setups/gallery/run_jcm_with_era5data.py`
- Modify: `tests/test_example_jax_helpers.py`
- Modify: `tests/test_runtime_run.py`
- Modify: `tests/test_jcm_example.py`

**Interfaces:**
- Consumes: `loglevel: str`, `float_type: str`.
- Produces: `normalize_log_level("trace") == 5` and three executable `run_setup` implementations that pass `DTypePolicy` and log level explicitly.

- [ ] **Step 1: Write and run the failing trace regression**

Add to `tests/test_coupler_coverage.py`:

```python
def test_normalize_log_level_accepts_trace_as_level_five() -> None:
    assert normalize_log_level("trace") == 5
    coupler = Coupler(
        Clock(start=datetime(2000, 1, 1), dt_seconds=60.0, steps=1),
        log_level="trace",
    )
    assert coupler.logger.isEnabledFor(5)
    assert coupler.logger.isEnabledFor(logging.DEBUG)
```

Add `normalize_log_level` to the file's existing
`from vercor.jax_logging import (...)` list. Run the individual test and expect
`ValueError: Unknown logging level: trace`.

- [ ] **Step 2: Implement trace normalization and confirm GREEN**

At the start of the string branch in `normalize_log_level`, add:

```python
if level.casefold() == "trace":
    return 5
```

Run the individual test, then the focused logging tests:

```bash
conda run -n scipy pytest \
  tests/test_coupler_coverage.py \
  tests/test_logging_boundaries.py \
  -q --tb=short
```

- [ ] **Step 3: Write failing option-propagation tests for the injectable JCM setup**

Extend the fake-coupler assertions in `tests/test_jcm_example.py`:

```python
coupler = example.build_coupler(
    ocean=ocean,
    jcm_inputs=jcm_inputs,
    clock=clock,
    log_level="warning",
    dtype=DTypePolicy(enable_x64=True),
)

assert coupler.kwargs["log_level"] == "warning"
assert coupler.kwargs["runtime"].dtype == DTypePolicy(enable_x64=True)
```

Import `DTypePolicy`. Add:

```python
def test_run_setup_maps_cli_values_to_coupler_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    example = importlib.import_module(GALLERY_MODULE)
    coupler = _RecordingRunCoupler()
    captured: dict[str, object] = {}

    def fake_build_coupler(**kwargs: object) -> _RecordingRunCoupler:
        captured.update(kwargs)
        return coupler

    monkeypatch.setattr(example, "build_coupler", fake_build_coupler)

    result = example.run_setup(loglevel="error", float_type="float32")

    assert result is None
    assert captured["log_level"] == "error"
    assert captured["dtype"] == DTypePolicy(enable_x64=False)
    assert [name for name, _ in coupler.events] == ["run"]
```

Run both and expect RED for unsupported arguments/missing `run_setup`.

- [ ] **Step 4: Implement the JCM setup contract while preserving `main`**

Import `DTypePolicy` from `vercor.dtypes`.

Change `build_coupler` to accept:

```python
def build_coupler(
    *,
    ocean: DataComponent | None = None,
    jcm_inputs: JCMInputs | None = None,
    clock: Clock | None = None,
    log_level: int | str = "INFO",
    dtype: DTypePolicy | None = None,
) -> Coupler:
```

Construct:

```python
runtime=RuntimeOptions(
    dtype=DTypePolicy() if dtype is None else dtype,
    topology=SurfaceMaskPolicy(),
),
log_level=log_level,
```

Add:

```python
def run_setup(*, loglevel: str, float_type: str) -> None:
    """Run the setup through the shared VerCOR CLI contract."""

    coupler = build_coupler(
        log_level=loglevel,
        dtype=DTypePolicy(enable_x64=float_type == "float64"),
    )
    coupler.run(output=OutputTarget("."))
```

Keep `main(arguments)` and its direct-script behavior unchanged.

- [ ] **Step 5: Add and implement contracts for custom wrapping and profiling**

In their existing tests, assert:

```python
assert callable(custom_component_wrapping.run_setup)
assert callable(profile_runtime.run_setup)
```

For `custom_component_wrapping`, extend `make_custom_coupler` with keyword
defaults `log_level: int | str = "INFO"` and
`dtype: DTypePolicy | None = None`, pass both into `Coupler`/`RuntimeOptions`,
and add:

```python
def run_setup(*, loglevel: str, float_type: str) -> None:
    """Run the custom-component demonstration through the CLI contract."""

    grid = make_example_grid()
    for component in (
        make_data_forcing(grid),
        make_differentiable_model(grid),
        make_host_model(grid),
    ):
        print(component)
    print(
        make_custom_coupler(
            grid,
            log_level=loglevel,
            dtype=DTypePolicy(enable_x64=float_type == "float64"),
        )
    )
```

Make the `__main__` block call
`run_setup(loglevel="info", float_type="float64")`.

For `profile_runtime`, thread `dtype: DTypePolicy | None = None` through
`build_slab_coupler` and `profile_runtime`, use it in `RuntimeOptions`, and add:

```python
def run_setup(*, loglevel: str, float_type: str) -> int:
    """Run the default profile through the shared CLI contract."""

    result = profile_runtime(
        steps=24,
        grid_nx=32,
        grid_ny=16,
        log_level=loglevel,
        dtype=DTypePolicy(enable_x64=float_type == "float64"),
    )
    for line in _format_result(result):
        print(line)
    return 0
```

Keep the profiling harness's existing argparse `main` entry point.

- [ ] **Step 6: Run focused tests, the required fast suite, and commit**

Run:

```bash
conda run -n scipy pytest \
  tests/test_coupler_coverage.py \
  tests/test_logging_boundaries.py \
  tests/test_jcm_example.py \
  tests/test_example_jax_helpers.py \
  tests/test_runtime_run.py \
  -q --fast --tb=short
conda run -n scipy pytest tests/ -q --fast --tb=short
git diff --check
git add \
  vercor/_logging/config.py \
  vercor/setups/gallery/custom_component_wrapping.py \
  vercor/setups/gallery/profile_runtime.py \
  vercor/setups/gallery/run_jcm_with_era5data.py \
  tests/test_coupler_coverage.py \
  tests/test_example_jax_helpers.py \
  tests/test_runtime_run.py \
  tests/test_jcm_example.py
git commit -m "feat: configure gallery setup runtime options"
```

---

### Task 5: Adopt the explicit contract across the remaining gallery

**Files:**
- Modify: `vercor/setups/gallery/run_camulator_with_veros.py`
- Modify: `vercor/setups/gallery/run_data_driver.py`
- Modify: `vercor/setups/gallery/run_jcm_with_slab.py`
- Modify: `vercor/setups/gallery/run_jcm_with_veros.py`
- Modify: `vercor/setups/gallery/run_jcm_with_verosdata.py`
- Modify: `vercor/setups/gallery/run_slab_driver.py`
- Modify: `vercor/setups/gallery/run_veros_with_era5data.py`
- Modify: `tests/test_cli.py`

**Interfaces:**
- Consumes: the approved `run_setup(*, loglevel: str, float_type: str) -> int | None` contract.
- Produces: every public bundled setup has the exact contract and explicitly supplies runtime precision and logging.

- [ ] **Step 1: Write a failing live contract test for every bundled setup**

Add:

```python
@pytest.mark.parametrize(
    "module_name",
    (
        "custom_component_wrapping",
        "profile_runtime",
        "run_camulator_with_veros",
        "run_data_driver",
        "run_jcm_with_era5data",
        "run_jcm_with_slab",
        "run_jcm_with_veros",
        "run_jcm_with_verosdata",
        "run_slab_driver",
        "run_veros_with_era5data",
    ),
)
def test_every_bundled_setup_exposes_keyword_only_run_contract(
    module_name: str,
) -> None:
    module = importlib.import_module(f"vercor.setups.gallery.{module_name}")

    run_setup = module.run_setup
    parameters = tuple(inspect.signature(run_setup).parameters.values())

    assert callable(run_setup)
    assert tuple(parameter.name for parameter in parameters) == (
        "loglevel",
        "float_type",
    )
    assert all(
        parameter.kind is inspect.Parameter.KEYWORD_ONLY
        for parameter in parameters
    )
```

Add `import importlib` and `import inspect`. Run the test and expect seven
missing-contract failures. This imports real modules without executing their
guarded setup bodies; the three representative modules from Task 4 already
have focused option-flow behavior tests.

- [ ] **Step 2: Convert each remaining guarded script body into `run_setup`**

For each listed script:

1. import `DTypePolicy` from `vercor.dtypes`;
2. replace the outer `if __name__ == "__main__":` body with:

```python
def run_setup(*, loglevel: str, float_type: str) -> None:
    """Run this setup through the shared VerCOR CLI contract."""

    dtype = DTypePolicy(enable_x64=float_type == "float64")
    # Existing guarded setup body remains here at this indentation.
```

3. change its `RuntimeOptions(...)` call to include `dtype=dtype`;
4. change its top-level `Coupler(...)` call to include
   `log_level=loglevel`; and
5. end the file with:

```python
if __name__ == "__main__":
    run_setup(loglevel="info", float_type="float64")
```

For example, the required `run_jcm_with_veros.py` constructor becomes:

```python
cpl = Coupler(
    clock=clock,
    components=components,
    exchanges=exchanges,
    run_order=run_order,
    runtime=RuntimeOptions(
        dtype=dtype,
        topology=SurfaceMaskPolicy(),
    ),
    log_level=loglevel,
)
```

Do not change model choices, clocks, exchanges, output paths, plotting, or
optional dependency imports.

- [ ] **Step 3: Run live gallery contract and option-flow tests**

Run:

```bash
conda run -n scipy pytest \
  tests/test_cli.py::test_every_bundled_setup_exposes_keyword_only_run_contract \
  tests/test_jcm_example.py::test_run_setup_maps_cli_values_to_coupler_runtime \
  tests/test_api_boundaries.py \
  tests/test_setup_agnostic_api.py \
  -q --fast --tb=short
conda run -n scipy python -m compileall -q vercor/setups/gallery
```

Expected: PASS. The live contract test imports all bundled setup modules
without executing model construction, while the representative JCM test
asserts real log-level and dtype propagation.

- [ ] **Step 4: Run the required fast suite and commit**

Run:

```bash
conda run -n scipy pytest tests/ -q --fast --tb=short
git diff --check
git add vercor/setups/gallery tests/test_cli.py
git commit -m "feat: adopt setup runner contract in gallery"
```

---

### Task 6: Update distribution contracts and user/developer documentation

**Files:**
- Modify: `tests/test_distribution_boundaries.py`
- Modify: `README.md`
- Modify: `docs/how-to/examples.rst`
- Modify: `docs/troubleshooting.rst`
- Modify: `DESIGN.md`
- Modify: `DEPENDENCIES.md`
- Modify: `PROGRESS.md`
- Modify: `CHANGELOG.md`

**Interfaces:**
- Consumes: completed CLI, private runner, and gallery contracts.
- Produces: packaged runner verification and current user/developer guidance.

- [ ] **Step 1: Add a failing distribution assertion for the private runner**

In the existing wheel/source build test, add:

```python
runner_name = "vercor/_setup_runner.py"
assert runner_name in wheel_names
assert f"vercor-{EXPECTED_VERSION}/{runner_name}" in sdist_names
```

Extend the installed-artifact probe with an isolated virtual environment that
inherits system site packages. Install the built wheel into it with
`pip --no-deps`, then execute the environment's real `vercor` console entry
point for `--version`, `show-setups`, `copy-setup --to`, and `run`. Run a small
external setup whose `run_setup` lazily imports an adjacent helper; do not
substitute `python -m vercor.cli` or direct private-runner invocation for the
console workflow.

- [ ] **Step 2: Run the artifact test and record RED if packaging is incomplete**

Run the exact existing distribution test that builds both artifacts:

```bash
conda run -n scipy pytest \
  tests/test_distribution_boundaries.py::test_built_distributions_run_external_extension_fixture_outside_checkout \
  -q --tb=short
```

Expected pre-change failure: the new runner assertion/probe is absent or the
installed probe still uses the old direct-script contract.

- [ ] **Step 3: Update README and Sphinx how-to commands**

Document:

```text
vercor show-setups
vercor copy-setup run_jcm_with_veros \
  --to ~/vercor-setups/run_jcm_with_veros
vercor run \
  --loglevel info \
  --float-type float64 \
  ~/vercor-setups/run_jcm_with_veros/run_jcm_with_veros.py
```

Explain `VERCOR_SETUP_DIR` as an `os.pathsep`-separated direct-directory list,
duplicate-name errors, create-or-reuse directory behavior, and non-overwrite
semantics. Document the `run_setup` signature for user-authored external
templates.

- [ ] **Step 4: Update architecture, dependencies, troubleshooting, and changelog**

Record these exact ownership statements:

- `vercor.cli`: Click presentation, shared discovery, duplicate rejection,
  and exclusive copying.
- `vercor._setup_runner`: private child loading and contract invocation.
- `vercor.setups.gallery`: model-specific setup construction and explicit
  translation of log level and dtype.
- `vercor._logging.config`: standard levels plus `trace == 5`.

Update the dependency order so the gallery depends on runtime/dtypes, the
private runner depends only on the standard library, and the CLI depends on
the gallery resource plus the runner subprocess boundary. Update
troubleshooting for missing contracts, duplicate templates, missing external
directories, and copy collisions. Add an unreleased changelog entry.

- [ ] **Step 5: Update `PROGRESS.md` with exact focused results**

Add a dated entry describing:

- the three commands and root version flag;
- external discovery and duplicate rejection;
- create-or-reuse `--to` behavior;
- the child setup contract and option defaults;
- `trace` and dtype mappings;
- focused test counts and commands actually observed at this point; and
- the remaining full-verification work.

Do not predict final counts before running them.

- [ ] **Step 6: Run focused docs, architecture, CLI, and distribution checks**

Run:

```bash
conda run -n scipy pytest \
  tests/test_cli.py \
  tests/test_setup_runner.py \
  tests/test_distribution_boundaries.py \
  tests/test_docs_build.py \
  tests/test_documentation_examples.py \
  -q --tb=short
conda run -n scipy sphinx-build -W -b html docs docs/_build/html
```

Expected: PASS, with no stale old-command documentation.

- [ ] **Step 7: Run the required fast suite and commit**

Run:

```bash
conda run -n scipy pytest tests/ -q --fast --tb=short
git diff --check
git add \
  tests/test_distribution_boundaries.py \
  README.md \
  docs/how-to/examples.rst \
  docs/troubleshooting.rst \
  DESIGN.md \
  DEPENDENCIES.md \
  PROGRESS.md \
  CHANGELOG.md
git commit -m "docs: describe expanded setup gallery CLI"
```

---

### Task 7: Perform final verification and update the draft pull request

**Files:**
- Modify: `PROGRESS.md`
- Modify: existing draft PR body for `feat/packaged-setup-gallery-cli`

**Interfaces:**
- Consumes: all implementation commits from Tasks 1-6.
- Produces: verified branch, final progress evidence, pushed commit, and updated draft PR.

- [ ] **Step 1: Run formatting and inspect resulting changes**

Run:

```bash
conda run -n scipy black vercor tests
git status --short
```

If Black changes files, run the focused tests for those files and include the
formatting in the owning implementation commit or a dedicated formatting
commit after the required fast suite.

- [ ] **Step 2: Run linting, type checking, compile checks, and whitespace checks**

Run:

```bash
conda run -n scipy flake8 . --count --max-line-length=120 --statistics
conda run -n scipy mypy vercor tests
conda run -n scipy python -m compileall -q vercor tests
git diff --check
```

Expected: zero flake8 errors, mypy success, compile success, and no whitespace
errors.

- [ ] **Step 3: Run fast and full parallel suites**

Run:

```bash
conda run -n scipy pytest tests/ -q --fast --tb=short
conda run -n scipy pytest tests/ -q --tb=short
```

Expected: both selections pass under the repository's configured four-worker
xdist settings.

- [ ] **Step 4: Run branch coverage**

Run:

```bash
conda run -n scipy pytest \
  --cov=vercor \
  --cov-branch \
  --cov-report=term-missing \
  tests/ -q --tb=short
```

Expected: all tests pass and configured coverage threshold is met.

- [ ] **Step 5: Build and probe wheel and source distribution**

Run:

```bash
conda run -n scipy python -m build
conda run -n scipy pytest \
  tests/test_distribution_boundaries.py \
  -q --tb=short
```

Confirm both artifacts contain `vercor/_setup_runner.py`, every gallery script,
and the `vercor` console entry point. Use the distribution test's isolated
installation probe to verify `--version`, `show-setups`, `copy-setup --to`,
and the lightweight `run_setup` contract.

- [ ] **Step 6: Record exact final evidence in `PROGRESS.md`**

Replace the preliminary verification note with observed counts, coverage,
artifact names, and the exact successful commands. Run:

```bash
git diff --check
conda run -n scipy pytest tests/ -q --fast --tb=short
git add PROGRESS.md
git commit -m "docs: record expanded CLI verification"
```

- [ ] **Step 7: Review the complete branch before publishing**

Run:

```bash
git status --short --branch
git log --oneline --decorate origin/main..HEAD
git diff --stat origin/main...HEAD
git diff --check origin/main...HEAD
```

Confirm the worktree is clean, only intended files changed, and the branch is
`feat/packaged-setup-gallery-cli`.

- [ ] **Step 8: Push and update the existing draft PR**

Run:

```bash
git push origin feat/packaged-setup-gallery-cli
```

Update the existing draft PR body with:

- the three-command interface and `--version`;
- external discovery and duplicate rejection;
- `--to` create-or-reuse semantics;
- explicit child-process contract and runtime options;
- logging trace support;
- test, type, lint, format, coverage, and artifact evidence; and
- the design and implementation-plan document paths.

Do not open a second pull request and do not mark the existing draft ready.
