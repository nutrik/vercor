"""Command-line access to packaged VerCOR setup scripts."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import resources
from importlib.resources.abc import Traversable
import os
from pathlib import Path
import shutil
import subprocess
import sys

import click

__all__ = ("cli",)

_LOG_LEVELS = ("trace", "debug", "info", "warning", "error")
_FLOAT_TYPES = ("float64", "float32")


@dataclass(frozen=True)
class _SetupTemplate:
    """Describe one uniquely named copyable setup template."""

    stem: str
    filename: str
    source: Traversable
    origin: str


def _public_python_file(name: str) -> bool:
    return name.endswith(".py") and name != "__init__.py" and not name.startswith("_")


def _external_setup_directories() -> tuple[Path, ...]:
    raw_value = os.environ.get("VERCOR_SETUP_DIR", "")
    return tuple(
        Path(value).expanduser() for value in raw_value.split(os.pathsep) if value
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
            raise click.ClickException(f"setup directory does not exist: {directory}")
        if not directory.is_dir():
            raise click.ClickException(f"setup path is not a directory: {directory}")
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
        stem: templates for stem, templates in by_stem.items() if len(templates) > 1
    }
    if duplicates:
        details = "; ".join(
            f"{stem}: {', '.join(item.origin for item in templates)}"
            for stem, templates in sorted(duplicates.items())
        )
        raise click.ClickException(f"duplicate setup: {details}")
    return tuple(sorted(candidates, key=lambda item: item.stem))


@click.group(name="vercor")
@click.version_option(package_name="vercor")
def cli() -> None:
    """Vercor command-line tools."""


@cli.command("show-setups")
def show_setups() -> None:
    """Print a list of available pre-configured setups."""

    for setup in _discover_setups():
        click.echo(setup.stem)


def _normalize_setup_name(name: str) -> str:
    """Validate a direct setup name and return its Python filename."""

    if not name or name != name.strip() or "/" in name or "\\" in name:
        raise click.BadParameter("must be a direct setup name", param_hint="SETUP")
    if name == ".py":
        raise click.BadParameter("must name a Python setup", param_hint="SETUP")
    if name.startswith("_"):
        raise click.BadParameter("must name a public setup", param_hint="SETUP")
    suffix = Path(name).suffix
    if suffix not in ("", ".py"):
        raise click.BadParameter("must name a Python setup", param_hint="SETUP")
    filename = name if suffix else f"{name}.py"
    if filename in (".py", "__init__.py"):
        raise click.BadParameter("must name a Python setup", param_hint="SETUP")
    return filename


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


@cli.command("copy-setup", cls=_CopySetupCommand)
@click.argument("setup")
@click.option(
    "--to",
    "destination",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("."),
    help="Target directory (default: current working directory).",
)
def copy_setup(setup: str, destination: Path) -> None:
    """Copy a standard setup to another directory."""

    filename = _normalize_setup_name(setup)
    catalog = {item.filename: item for item in _discover_setups()}
    try:
        template = catalog[filename]
    except KeyError as error:
        raise click.ClickException(f"unknown setup: {setup}") from error

    try:
        destination.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise click.ClickException(
            f"could not create target directory {destination}: {error}"
        ) from error
    if not destination.is_dir():
        raise click.ClickException(f"target path is not a directory: {destination}")

    target = destination / template.filename
    created = False
    try:
        with template.source.open("rb") as source_stream:
            with target.open("xb") as target_stream:
                created = True
                shutil.copyfileobj(source_stream, target_stream)
    except FileExistsError as error:
        raise click.ClickException(f"{target} already exists") from error
    except OSError as error:
        if created:
            target.unlink(missing_ok=True)
        raise click.ClickException(
            f"could not copy {template.filename}: {error}"
        ) from error
    click.echo(f"Copied {template.filename} to {target}")


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
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "vercor._setup_runner",
            str(setup_file.resolve()),
            "--loglevel",
            loglevel.lower(),
            "--float-type",
            float_type.lower(),
        ],
        check=False,
    )
    if completed.returncode:
        raise click.exceptions.Exit(completed.returncode)


if __name__ == "__main__":
    cli()
