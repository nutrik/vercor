from __future__ import annotations

from importlib import resources
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


def test_cli_help_lists_copy_and_run_commands() -> None:
    result = CliRunner().invoke(cli, ["--help"])

    assert result.exit_code == 0
    assert "copy-setup" in result.output
    assert "run" in result.output


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


@pytest.mark.fast_always
@pytest.mark.parametrize("private_name", ("_internal", "_internal.py"))
def test_copy_setup_rejects_private_names_during_normalization(
    private_name: str,
) -> None:
    runner = CliRunner()

    private_result = runner.invoke(cli, ["copy-setup", private_name])
    unknown_result = runner.invoke(cli, ["copy-setup", "ordinary_unknown"])

    assert private_result.exit_code == 2
    assert "must name a public setup" in private_result.output
    assert unknown_result.exit_code == 1
    assert "unknown setup" in unknown_result.output
