"""Command-line access to packaged VerCOR setup scripts."""

from __future__ import annotations

from importlib import resources
from pathlib import Path
import shutil
import subprocess
import sys

import click

__all__ = ("cli",)


@click.group()
def cli() -> None:
    """Copy and run VerCOR setup scripts."""


def _normalize_setup_name(name: str) -> str:
    """Validate a direct packaged setup name and return its Python filename."""

    if not name or name != name.strip() or "/" in name or "\\" in name:
        raise click.BadParameter("must be a direct setup name", param_hint="NAME")
    if name.startswith("_"):
        raise click.BadParameter("must name a public setup", param_hint="NAME")
    suffix = Path(name).suffix
    if suffix not in ("", ".py"):
        raise click.BadParameter("must name a Python setup", param_hint="NAME")
    filename = name if suffix else f"{name}.py"
    if filename in (".py", "__init__.py"):
        raise click.BadParameter("must name a Python setup", param_hint="NAME")
    return filename


@cli.command("copy-setup")
@click.argument("name")
def copy_setup(name: str) -> None:
    """Copy bundled setup NAME into the current directory."""

    filename = _normalize_setup_name(name)
    source = resources.files("vercor.setups.gallery").joinpath(filename)
    if not source.is_file():
        raise click.ClickException(f"unknown setup: {name}")
    destination = Path.cwd() / filename
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
    click.echo(f"Copied {filename}")


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


if __name__ == "__main__":
    cli()
