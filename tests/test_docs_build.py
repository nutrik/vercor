"""End-to-end contracts for the Sphinx documentation build."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("sphinx")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOCS_ROOT = PROJECT_ROOT / "docs"


def test_sphinx_builds_html_with_warnings_as_errors(tmp_path: Path) -> None:
    """Build the committed documentation without warnings or network access."""
    output_path = tmp_path / "html"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "sphinx",
            "-W",
            "--keep-going",
            "-b",
            "html",
            str(DOCS_ROOT),
            str(output_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert (output_path / "index.html").is_file()
