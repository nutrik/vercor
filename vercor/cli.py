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
