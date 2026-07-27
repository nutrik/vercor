"""End-to-end contracts for the Sphinx documentation build."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("sphinx")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOCS_ROOT = PROJECT_ROOT / "docs"

EXPECTED_ROOT_PAGES = (
    "introduction",
    "researchers/index",
    "developers/index",
    "how-to/index",
    "api/index",
    "troubleshooting",
    "project-resources",
)


@pytest.mark.fast_always
def test_documentation_has_two_learning_paths_and_reference_sections() -> None:
    """Keep every canonical top-level destination in the root toctree."""
    index_source = (DOCS_ROOT / "index.rst").read_text(encoding="utf-8")

    for page in EXPECTED_ROOT_PAGES:
        assert page in index_source

    assert "For Earth-system researchers" in index_source
    assert "For Python and JAX developers" in index_source


@pytest.mark.fast_always
def test_sphinx_builds_rst_and_selected_markdown_sources() -> None:
    """Keep MyST and archive exclusions explicit in the Sphinx policy."""
    conf_source = (DOCS_ROOT / "conf.py").read_text(encoding="utf-8")
    requirements = (DOCS_ROOT / "requirements.txt").read_text(encoding="utf-8")
    readthedocs = (PROJECT_ROOT / ".readthedocs.yaml").read_text(encoding="utf-8")

    assert '"myst_parser"' in conf_source
    assert '".rst": "restructuredtext"' in conf_source
    assert '".md": "markdown"' in conf_source
    assert "README.md" in conf_source
    assert "progress-archive-*.md" in conf_source
    assert "myst-parser==5.1.0" in requirements
    assert "method: pip" in readthedocs
    assert "path: ." in readthedocs


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
