from __future__ import annotations

from importlib import resources
from pathlib import Path

from click.testing import CliRunner
import pytest

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
