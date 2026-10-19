"""Install released coupled models with the reviewed JCM 3 JAX exception.

Veris 0.2.1 requires JAX >=0.11.2, whereas JCM 3.0.0 declares <0.11.
Only that upper bound is overridden, at exactly JAX 0.11.2. Every other
base requirement participates in normal pip resolution. Full model tests
remain the acceptance gate for this explicitly documented exception.
"""

from __future__ import annotations

import argparse
from email.parser import Parser
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from zipfile import ZipFile


def coupled_requirements(wheel: Path) -> list[str]:
    """Read base requirements, accepting only the reviewed release constraint."""
    with ZipFile(wheel) as archive:
        metadata = Parser().parsestr(
            archive.read("jcm-3.0.0.dist-info/METADATA").decode("utf-8")
        )
    if metadata["Name"] != "jcm" or metadata["Version"] != "3.0.0":
        raise ValueError("Expected the released JCM 3.0.0 wheel")
    requirements = []
    jax_requirements = []
    for requirement in metadata.get_all("Requires-Dist", []):
        # Keep platform/Python markers for pip; unrequested extras are absent.
        if re.search(r";.*\bextra\b", requirement):
            continue
        name = re.split(r"[<>=~!\[;\s]", requirement, maxsplit=1)[0]
        if name.lower() == "jax":
            jax_requirements.append(requirement)
        else:
            requirements.append(requirement)
    if jax_requirements != ["jax<0.11,>=0.10"]:
        raise ValueError(f"Unreviewed JCM JAX requirement: {jax_requirements!r}")
    return [*requirements, "jax==0.11.2"]


def verify_dependency_report(report: str, returncode: int) -> None:
    """Reject any pip-check conflict beyond the exact approved JCM exception."""
    expected = "jcm 3.0.0 has requirement jax<0.11,>=0.10, " "but you have jax 0.11.2."
    if returncode != 1 or report.splitlines() != [expected]:
        raise RuntimeError(f"Unexpected dependency validation result: {report}")


def main() -> None:
    """Resolve native dependencies and install only JCM without its upper bound."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--wheel", type=Path, help="Use an already downloaded JCM 3.0.0 wheel"
    )
    args = parser.parse_args()
    pip = [sys.executable, "-m", "pip"]
    with tempfile.TemporaryDirectory(prefix="vercor-jcm3-") as directory:
        wheel = args.wheel
        if wheel is None:
            subprocess.run(
                [
                    *pip,
                    "download",
                    "--no-deps",
                    "--only-binary=:all:",
                    "--dest",
                    directory,
                    "jcm==3.0.0",
                ],
                check=True,
            )
            wheel = Path(directory) / "jcm-3.0.0-py3-none-any.whl"
        subprocess.run(
            [*pip, "install", ".[dev,veros,veris]", *coupled_requirements(wheel)],
            check=True,
        )
        subprocess.run([*pip, "install", "--no-deps", str(wheel)], check=True)
    check = subprocess.run([*pip, "check"], text=True, capture_output=True)
    verify_dependency_report(check.stdout, check.returncode)
    print("Only the approved JCM 3.0.0 / JAX 0.11.2 metadata exception remains.")


if __name__ == "__main__":
    main()
