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

API_PAGES = (
    "assembly.rst",
    "components.rst",
    "grids-exchanges.rst",
    "output-diagnostics.rst",
    "setups-physics-types.rst",
    "advanced.rst",
)

TUTORIAL_PAGES = (
    "researchers/getting-started.rst",
    "developers/data-components.rst",
    "developers/host-components.rst",
    "developers/jax-components.rst",
    "developers/coupling.rst",
)

PROJECT_RESOURCE_PAGES = (
    "migration-0.3-to-0.4",
    "plugin-authoring",
    "release-notes-0.4.3",
    "release-notes-0.4.2",
    "release-notes-0.4.1",
    "release-notes-0.4.0",
    "releasing",
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
@pytest.mark.parametrize("page", TUTORIAL_PAGES)
def test_tutorials_state_prerequisites_result_and_next_steps(page: str) -> None:
    """Keep each executable tutorial independently usable and navigable."""
    source = (DOCS_ROOT / page).read_text(encoding="utf-8")

    for section in ("Prerequisites", "Expected result", "Next steps"):
        assert section in source


@pytest.mark.fast_always
def test_exchange_troubleshooting_does_not_confuse_route_ids_with_fan_in() -> None:
    """Keep the route-identity remedy separate from unsupported field fan-in."""
    source = (DOCS_ROOT / "troubleshooting.rst").read_text(encoding="utf-8")

    assert "do not use distinct IDs to bypass fan-in rejection" in source
    assert "only one route may" in source
    assert "write a target field" in source


@pytest.mark.fast_always
def test_daily_data_uses_one_365_record_climatology_for_all_calendars() -> None:
    """Document the runtime's leap-day and 360-day daily-forcing mappings."""
    source = (DOCS_ROOT / "developers" / "data-components.rst").read_text(
        encoding="utf-8"
    )
    prose = " ".join(source.split())

    assert "365 records for every supported calendar" in prose
    assert "February 29 uses the same record as February 28" in prose
    assert "360-day month onto the corresponding no-leap month" in prose
    assert "360 records" not in prose


@pytest.mark.fast_always
def test_running_guide_does_not_claim_run_state_advances_the_clock_window() -> None:
    """Keep supplied-state reuse distinct from an unimplemented clock cursor."""
    source = (DOCS_ROOT / "researchers" / "running.rst").read_text(encoding="utf-8")

    assert "replays the configured ``Clock`` window" in source
    assert "does not store a clock cursor" in source
    assert "To continue a simulation" not in source


@pytest.mark.fast_always
def test_api_reference_is_curated_and_separates_advanced_contracts() -> None:
    """Document public owners explicitly and keep private modules absent."""
    sources = []
    for name in API_PAGES:
        source = (DOCS_ROOT / "api" / name).read_text(encoding="utf-8")
        assert ".. auto" in source
        assert "vercor._" not in source
        sources.append(source)

    stable_source = "\n".join(sources[:-1])
    advanced_source = sources[-1]
    assert "vercor.runtime" not in stable_source
    assert ".. automodule:: vercor.runtime" in advanced_source
    assert ".. automodule:: vercor.topology" in advanced_source
    assert ".. automodule:: vercor.dtypes" in stable_source
    assert ":members: DTypePolicy" in stable_source


@pytest.mark.fast_always
@pytest.mark.parametrize("page", API_PAGES)
def test_api_groups_introduce_their_usage_before_generated_members(page: str) -> None:
    """Give readers purpose and context before each page's generated API."""
    source = (DOCS_ROOT / "api" / page).read_text(encoding="utf-8")
    lines = source.splitlines()
    generated_group_count = 0

    for directive_index, line in enumerate(lines):
        if not line.startswith(".. automodule::"):
            continue
        generated_group_count += 1
        heading_underline_index = max(
            index
            for index, candidate in enumerate(lines[:directive_index])
            if candidate and set(candidate) <= {"=", "-", "^"}
        )
        introduction = lines[slice(heading_underline_index + 1, directive_index)]
        assert any(
            candidate.strip() for candidate in introduction
        ), f"{page} has no usage introduction before {line}"

    assert generated_group_count > 0


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


@pytest.mark.fast_always
def test_readme_is_a_concise_gateway_to_canonical_documentation() -> None:
    """Keep detailed guidance on Read the Docs instead of in the README."""
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")

    assert "https://vercor.readthedocs.io/" in readme
    for canonical_link in (
        "[Researcher guide](https://vercor.readthedocs.io/en/latest/researchers/)",
        "[Developer guide](https://vercor.readthedocs.io/en/latest/developers/)",
        "[Python API](https://vercor.readthedocs.io/en/latest/api/)",
        "[Migration guide](https://vercor.readthedocs.io/en/latest/migration-0.3-to-0.4.html)",
        "[Plugin authoring](https://vercor.readthedocs.io/en/latest/plugin-authoring.html)",
    ):
        assert canonical_link in readme
    assert len(readme.splitlines()) <= 180
    for removed_heading in (
        "### Create a custom JAX component",
        "### Run a host-side component",
        "### Couple components on different grids",
        "### Write output",
        "## Configuration",
        "## Troubleshooting",
    ):
        assert removed_heading not in readme


@pytest.mark.fast_always
def test_project_resources_publish_maintained_markdown() -> None:
    """Expose active project guides without publishing archives."""
    source = (DOCS_ROOT / "project-resources.rst").read_text(encoding="utf-8")
    published_pages = tuple(
        line.strip()
        for line in source.splitlines()
        if line.startswith("   ") and not line.lstrip().startswith(":")
    )

    assert published_pages == PROJECT_RESOURCE_PAGES
    for excluded_resource in (
        "api-architecture-review",
        "over-engineering-",
        "progress-archive-",
        "superpowers/",
    ):
        assert excluded_resource not in source


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
