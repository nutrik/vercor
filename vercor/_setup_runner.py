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
    setup_directory = str(path.parent)
    original_sys_path = list(sys.path)
    sys.path.insert(0, setup_directory)
    try:
        return runpy.run_path(str(path), run_name="_vercor_setup")
    finally:
        sys.path[:] = original_sys_path


def _invoke_setup(
    path: Path,
    *,
    loglevel: str,
    float_type: str,
) -> int:
    namespace = _load_setup(path)
    run_setup = namespace.get("run_setup")
    if not callable(run_setup):
        raise SetupContractError(
            f"{path} must define callable run_setup(*, loglevel, float_type)"
        )
    result = run_setup(loglevel=loglevel, float_type=float_type)
    if result is None:
        return 0
    if isinstance(result, bool) or not isinstance(result, int):
        raise SetupContractError("run_setup must return int or None")
    return result


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
    """Run a setup-file contract with validated command-line arguments."""

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
