from __future__ import annotations

from collections.abc import Iterator
from importlib import metadata
from importlib import resources
import os
from pathlib import Path
import sys

from click.testing import CliRunner
import pytest

from vercor.cli import cli


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


def test_run_executes_option_like_local_python_filename() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        Path("-c.py").write_text(
            "from pathlib import Path\n"
            "Path('ran.txt').write_text('ran', encoding='utf-8')\n",
            encoding="utf-8",
        )

        result = runner.invoke(cli, ["run", "--", "-c.py"])

        assert result.exit_code == 0, result.output
        assert Path("ran.txt").read_text(encoding="utf-8") == "ran"


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


def test_cli_help_exposes_required_description_options_and_commands() -> None:
    result = CliRunner().invoke(cli, ["--help"])

    assert result.exit_code == 0, result.output
    assert "Vercor command-line tools" in result.output
    assert "--version" in result.output
    assert "copy-setup" in result.output
    assert "show-setups" in result.output
    assert "run" in result.output


def test_cli_version_reports_installed_distribution_version() -> None:
    result = CliRunner().invoke(cli, ["--version"])

    assert result.exit_code == 0, result.output
    assert result.output.strip() == f"vercor, version {metadata.version('vercor')}"


def test_show_setups_lists_sorted_public_bundled_stems() -> None:
    result = CliRunner().invoke(cli, ["show-setups"])

    assert result.exit_code == 0, result.output
    names = result.output.splitlines()
    assert names == sorted(names)
    assert "run_jcm_with_veros" in names
    assert "__init__" not in names


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


def test_show_setups_rejects_packaged_and_external_duplicate_names(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    external = tmp_path / "external"
    external.mkdir()
    duplicate = external / "run_jcm_with_veros.py"
    duplicate.write_text("", encoding="utf-8")
    monkeypatch.setenv("VERCOR_SETUP_DIR", str(external))

    result = CliRunner().invoke(cli, ["show-setups"])

    assert result.exit_code == 1
    assert "duplicate setup: run_jcm_with_veros" in result.output
    assert "vercor.setups.gallery/run_jcm_with_veros.py" in result.output
    assert str(duplicate) in result.output


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


def test_copy_setup_by_stem_copies_packaged_bytes() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        result = runner.invoke(cli, ["copy-setup", "run_jcm_with_veros"])
        copied = Path("run_jcm_with_veros.py")
        packaged = resources.files("vercor.setups.gallery").joinpath(copied.name)

        assert result.exit_code == 0, result.output
        assert copied.read_bytes() == packaged.read_bytes()
        assert "run_jcm_with_veros.py" in result.output


def test_copy_setup_accepts_python_filename() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        result = runner.invoke(cli, ["copy-setup", "run_slab_driver.py"])

        assert result.exit_code == 0, result.output
        assert Path("run_slab_driver.py").is_file()


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


def test_copy_setup_rejects_duplicate_catalog_names(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    external = tmp_path / "external"
    external.mkdir()
    duplicate = external / "run_jcm_with_veros.py"
    duplicate.write_text("", encoding="utf-8")
    monkeypatch.setenv("VERCOR_SETUP_DIR", str(external))

    runner = CliRunner()
    with runner.isolated_filesystem():
        result = runner.invoke(cli, ["copy-setup", "run_jcm_with_veros"])

    assert result.exit_code == 1
    assert "duplicate setup: run_jcm_with_veros" in result.output
    assert "vercor.setups.gallery/run_jcm_with_veros.py" in result.output
    assert str(duplicate) in result.output


def test_copy_setup_preserves_existing_destination() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        destination = Path("run_jcm_with_veros.py")
        destination.write_text("keep me", encoding="utf-8")

        result = runner.invoke(cli, ["copy-setup", "run_jcm_with_veros"])

        assert result.exit_code != 0
        assert "already exists" in result.output
        assert destination.read_text(encoding="utf-8") == "keep me"


@pytest.mark.parametrize(
    ("name", "diagnostic"),
    (
        ("", "must be a direct setup name"),
        (".py", "must name a Python setup"),
        ("../run_slab_driver", "must be a direct setup name"),
        ("nested/run_slab_driver", "must be a direct setup name"),
        (r"nested\run_slab_driver", "must be a direct setup name"),
        ("run_slab_driver.txt", "must name a Python setup"),
        ("__init__", "must name a public setup"),
    ),
)
def test_copy_setup_rejects_malformed_name(
    name: str,
    diagnostic: str,
) -> None:
    result = CliRunner().invoke(cli, ["copy-setup", name])

    assert result.exit_code == 2
    assert "Invalid value for SETUP" in result.output
    assert diagnostic in result.output


def test_copy_setup_rejects_unknown_name_as_unknown_resource() -> None:
    result = CliRunner().invoke(cli, ["copy-setup", "not_a_setup"])

    assert result.exit_code == 1
    assert "Error: unknown setup: not_a_setup" in result.output


@pytest.mark.fast_always
@pytest.mark.parametrize("private_name", ("_internal", "_internal.py"))
def test_copy_setup_rejects_private_names_during_normalization(
    private_name: str,
) -> None:
    runner = CliRunner()

    private_result = runner.invoke(cli, ["copy-setup", private_name])

    assert private_result.exit_code == 2
    assert "must name a public setup" in private_result.output
