from __future__ import annotations

from collections.abc import Iterator
import importlib
from importlib import metadata
from importlib import resources
import inspect
import os
from pathlib import Path
import sys
from typing import IO

from click.testing import CliRunner
import pytest

from vercor.cli import cli


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
        parameter.kind is inspect.Parameter.KEYWORD_ONLY for parameter in parameters
    )


@pytest.fixture
def _use_worktree_package_for_child_runner(monkeypatch: pytest.MonkeyPatch) -> None:
    """Expose this worktree only to tests that start a real child process."""

    project_root = Path(__file__).parent.parent
    inherited_path = os.environ.get("PYTHONPATH")
    paths = (
        (str(project_root), inherited_path) if inherited_path else (str(project_root),)
    )
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(paths))


def test_run_executes_python_file_with_current_interpreter(
    _use_worktree_package_for_child_runner: None,
) -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        Path("setup.py").write_text(
            "from pathlib import Path\n"
            "import sys\n"
            "def run_setup(*, loglevel, float_type):\n"
            "    Path('interpreter.txt').write_text(sys.executable, encoding='utf-8')\n",
            encoding="utf-8",
        )

        result = runner.invoke(cli, ["run", "setup.py"])

        assert result.exit_code == 0, result.output
        assert Path("interpreter.txt").read_text(encoding="utf-8") == sys.executable


def test_run_executes_option_like_local_python_filename(
    _use_worktree_package_for_child_runner: None,
) -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        Path("-c.py").write_text(
            "from pathlib import Path\n"
            "def run_setup(*, loglevel, float_type):\n"
            "    Path('ran.txt').write_text('ran', encoding='utf-8')\n",
            encoding="utf-8",
        )

        result = runner.invoke(cli, ["run", "--", "-c.py"])

        assert result.exit_code == 0, result.output
        assert Path("ran.txt").read_text(encoding="utf-8") == "ran"


def test_run_propagates_script_exit_status(
    _use_worktree_package_for_child_runner: None,
) -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        Path("setup.py").write_text(
            "def run_setup(*, loglevel, float_type):\n" "    return 7\n",
            encoding="utf-8",
        )

        result = runner.invoke(cli, ["run", "setup.py"])

        assert result.exit_code == 7


def test_run_passes_selected_loglevel_and_float_type(
    _use_worktree_package_for_child_runner: None,
) -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        Path("setup.py").write_text(
            "from pathlib import Path\n"
            "def run_setup(*, loglevel, float_type):\n"
            "    Path('options.txt').write_text(\n"
            "        f'{loglevel},{float_type}', encoding='utf-8')\n",
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
        assert Path("options.txt").read_text(encoding="utf-8") == "warning,float32"


def test_run_passes_default_loglevel_and_float_type(
    _use_worktree_package_for_child_runner: None,
) -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        Path("setup.py").write_text(
            "from pathlib import Path\n"
            "def run_setup(*, loglevel, float_type):\n"
            "    Path('options.txt').write_text(\n"
            "        f'{loglevel},{float_type}', encoding='utf-8')\n",
            encoding="utf-8",
        )

        result = runner.invoke(cli, ["run", "setup.py"])

        assert result.exit_code == 0, result.output
        assert Path("options.txt").read_text(encoding="utf-8") == "info,float64"


@pytest.mark.parametrize("loglevel", ("trace", "debug", "info", "warning", "error"))
@pytest.mark.parametrize("float_type", ("float64", "float32"))
def test_run_accepts_every_runtime_option_choice(
    loglevel: str,
    float_type: str,
    _use_worktree_package_for_child_runner: None,
) -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        Path("setup.py").write_text(
            "from pathlib import Path\n"
            "def run_setup(*, loglevel, float_type):\n"
            "    Path('options.txt').write_text(\n"
            "        f'{loglevel},{float_type}', encoding='utf-8')\n",
            encoding="utf-8",
        )

        result = runner.invoke(
            cli,
            [
                "run",
                "--loglevel",
                loglevel,
                "--float-type",
                float_type,
                "setup.py",
            ],
        )

        assert result.exit_code == 0, result.output
        assert Path("options.txt").read_text(encoding="utf-8") == (
            f"{loglevel},{float_type}"
        )


def test_run_propagates_setup_exception_status(
    _use_worktree_package_for_child_runner: None,
) -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        Path("setup.py").write_text(
            "def run_setup(*, loglevel, float_type):\n"
            "    raise RuntimeError('setup failed')\n",
            encoding="utf-8",
        )

        result = runner.invoke(cli, ["run", "setup.py"])

        assert result.exit_code == 1


def test_run_uses_trusted_runner_when_working_directory_contains_vercor(
    _use_worktree_package_for_child_runner: None,
) -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        local_package = Path("vercor")
        local_package.mkdir()
        (local_package / "__init__.py").write_text("", encoding="utf-8")
        (local_package / "_setup_runner.py").write_text(
            "from pathlib import Path\n"
            "Path('shadow-runner.txt').write_text('shadowed', encoding='utf-8')\n"
            "raise SystemExit(23)\n",
            encoding="utf-8",
        )
        Path("setup.py").write_text(
            "from pathlib import Path\n"
            "def run_setup(*, loglevel, float_type):\n"
            "    Path('trusted-runner.txt').write_text('ran', encoding='utf-8')\n",
            encoding="utf-8",
        )

        result = runner.invoke(cli, ["run", "setup.py"])

        assert result.exit_code == 0, result.output
        assert Path("trusted-runner.txt").read_text(encoding="utf-8") == "ran"
        assert not Path("shadow-runner.txt").exists()


def test_run_help_lists_runtime_options_and_defaults() -> None:
    result = CliRunner().invoke(cli, ["run", "--help"])

    assert result.exit_code == 0, result.output
    assert "-v, --loglevel [trace|debug|info|warning|error]" in result.output
    assert "[default: info]" in result.output
    assert "--float-type [float64|float32]" in result.output
    assert "[default: float64]" in result.output


@pytest.mark.parametrize(
    ("option", "value"),
    (("--loglevel", "INFO"), ("--float-type", "FLOAT32")),
)
def test_run_rejects_uppercase_runtime_option_values(
    option: str,
    value: str,
) -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        Path("setup.py").write_text("", encoding="utf-8")

        result = runner.invoke(cli, ["run", option, value, "setup.py"])

        assert result.exit_code == 2
        assert "Invalid value" in result.output


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


@pytest.mark.parametrize("use_canonical_filename", (False, True))
def test_copy_setup_resolves_dotted_stem_and_canonical_filename(
    use_canonical_filename: bool,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    dotted_stem = "model.v" + "1"
    setup_reference = f"{dotted_stem}.py" if use_canonical_filename else dotted_stem
    external = tmp_path / "external"
    external.mkdir()
    source = external / f"{dotted_stem}.py"
    source.write_bytes(b"VALUE = 17\n")
    destination = tmp_path / setup_reference.replace(".", "-")
    monkeypatch.setenv("VERCOR_SETUP_DIR", str(external))

    listed = CliRunner().invoke(cli, ["show-setups"])
    result = CliRunner().invoke(
        cli,
        ["copy-setup", setup_reference, "--to", str(destination)],
    )

    assert listed.exit_code == 0, listed.output
    assert dotted_stem in listed.output.splitlines()
    assert result.exit_code == 0, result.output
    assert (destination / f"{dotted_stem}.py").read_bytes() == b"VALUE = 17\n"


def test_copy_setup_rejects_ambiguous_stem_filename_reference(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    external = tmp_path / "external"
    external.mkdir()
    canonical_match = external / "foo.py"
    stem_match = external / "foo.py.py"
    canonical_match.write_bytes(b"CANONICAL = True\n")
    stem_match.write_bytes(b"STEM = True\n")
    destination = tmp_path / "copied"
    monkeypatch.setenv("VERCOR_SETUP_DIR", str(external))

    result = CliRunner().invoke(
        cli,
        ["copy-setup", "foo.py", "--to", str(destination)],
    )

    assert result.exit_code == 1
    assert "ambiguous setup reference: foo.py" in result.output
    assert str(canonical_match) in result.output
    assert str(stem_match) in result.output
    assert not destination.exists()


@pytest.mark.parametrize("failure_type", (RuntimeError, KeyboardInterrupt))
def test_copy_setup_removes_partial_destination_after_stream_failure(
    failure_type: type[BaseException],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def failing_copyfileobj(
        _source_stream: IO[bytes],
        target_stream: IO[bytes],
        _length: int = 0,
    ) -> None:
        target_stream.write(b"partial")
        raise failure_type("stream failed")

    monkeypatch.setattr("vercor.cli.shutil.copyfileobj", failing_copyfileobj)
    runner = CliRunner()
    with runner.isolated_filesystem():
        target = Path("run_jcm_with_veros.py")

        result = runner.invoke(cli, ["copy-setup", "run_jcm_with_veros"])

        assert result.exit_code != 0
        assert not target.exists()


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
        (".", "must be a direct setup name"),
        ("..", "must be a direct setup name"),
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
