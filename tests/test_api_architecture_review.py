"""Executable documentation and release contracts for stable VerCOR 0.4.0."""

from __future__ import annotations

import ast
from collections.abc import Callable
import hashlib
import importlib
import inspect
import json
from pathlib import Path
import re
import subprocess
import tomllib
from typing import Any, cast, get_type_hints

import pytest
import yaml

from tests._signature_support import canonicalize_external_typing_aliases

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REVIEW_PATH = PROJECT_ROOT / "docs" / "api-architecture-review.md"
README_PATH = PROJECT_ROOT / "README.md"
DESIGN_PATH = PROJECT_ROOT / "DESIGN.md"
MIGRATION_PATH = PROJECT_ROOT / "docs" / "migration-0.3-to-0.4.md"
RELEASING_PATH = PROJECT_ROOT / "docs" / "releasing.md"
WORKFLOW_PATH = PROJECT_ROOT / ".github" / "workflows" / "python-package.yml"
CHECKOUT_ACTION = "actions/checkout@11d5960a326750d5838078e36cf38b85af677262"
PYPI_PUBLISH_ACTION = (
    "pypa/gh-action-pypi-publish@ba38be9e461d3875417946c167d0b5f3d385a247"
)
TASK6_REPORT_PATH = PROJECT_ROOT / ".superpowers" / "sdd" / "task-6-report.md"
CHANGELOG_PATH = PROJECT_ROOT / "CHANGELOG.md"
PROGRESS_PATH = PROJECT_ROOT / "PROGRESS.md"
SIGNATURE_CONTRACT_PATH = (
    PROJECT_ROOT / "tests" / "contracts" / "vercor-0.4.0-public-signatures.json"
)
DEPENDENCIES_PATH = PROJECT_ROOT / "DEPENDENCIES.md"
PROGRESS_ARCHIVE_PATH = (
    PROJECT_ROOT / "docs" / "progress-archive-2026-05-16-to-2026-07-14.md"
)
PROGRESS_ARCHIVE_SHA256 = (
    "77a1d4a3c536901053718e9d7d31474a955922f8c2872f6e53f1c7fdbc70f69e"
)

REQUIRED_REVIEW_HEADINGS = (
    "1. Executive summary",
    "2. Duplication map",
    "3. Bad design decisions",
    "4. Public API redesign",
    "5. Private API redesign",
    "6. Setup-agnostic plugin architecture",
    "7. Compatibility plan",
    "8. Final rewritten API",
)


def _python_fences(markdown: str) -> tuple[str, ...]:
    """Return Python snippets from Markdown in source order."""

    return tuple(re.findall(r"```python\n(.*?)```", markdown, flags=re.DOTALL))


def _markdown_fences(markdown: str, *, owner: str) -> tuple[tuple[str, str], ...]:
    """Parse Markdown fences and reject ambiguous or unterminated transcripts."""

    fences: list[tuple[str, str]] = []
    opening: tuple[str, str] | None = None
    body: list[str] = []
    for line_number, line in enumerate(markdown.splitlines(), start=1):
        marker = re.fullmatch(r"(`{3,})([^`]*)", line)
        if opening is None:
            if marker is not None:
                opening = (marker.group(1), marker.group(2).strip())
                body = []
            continue
        opening_marker, language = opening
        if re.fullmatch(rf"`{{{len(opening_marker)},}}", line):
            source = "\n".join(body)
            assert not (
                len(opening_marker) == 3 and "```" in source
            ), f"{owner}:{line_number} has a literal triple fence inside a triple fence"
            fences.append((language, source))
            opening = None
            body = []
        else:
            body.append(line)
    assert opening is None, f"{owner} has an unterminated Markdown fence"
    return tuple(fences)


def _section(markdown: str, heading: str) -> str:
    """Return one Markdown subsection, excluding the next same-level heading."""

    level = len(heading) - len(heading.lstrip("#"))
    pattern = rf"^{re.escape(heading)}\n(.*?)(?=^#{{1,{level}}} |\Z)"
    match = re.search(pattern, markdown, flags=re.MULTILINE | re.DOTALL)
    assert match is not None, f"missing release-guide section {heading!r}"
    return match.group(1)


def _assert_public_imports_only(source: str, *, owner: str) -> None:
    """Reject imports from underscored VerCOR modules in a documentation snippet."""

    tree = ast.parse(source, filename=owner)
    for node in ast.walk(tree):
        modules: list[str] = []
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            modules.append(node.module)
        for module in modules:
            if module == "vercor" or module.startswith("vercor."):
                assert not any(
                    part.startswith("_") for part in module.split(".")[1:]
                ), f"{owner} imports private VerCOR module {module}"


def _documented_public_manifest(review: str) -> dict[str, tuple[str, ...]]:
    """Parse the JSON public manifest embedded in the architecture review."""

    match = re.search(
        r"<!-- public-api-manifest:start -->\n"
        r"```json\n(.*?)\n```\n"
        r"<!-- public-api-manifest:end -->",
        review,
        flags=re.DOTALL,
    )
    assert match is not None, "architecture review lacks its public API manifest"
    manifest = json.loads(match.group(1))
    assert isinstance(manifest, dict)
    return {name: tuple(exports) for name, exports in manifest.items()}


def _public_signature_contract() -> dict[str, dict[str, str]]:
    """Load the static callable-export and behavioral-method signature contract."""

    contract = json.loads(SIGNATURE_CONTRACT_PATH.read_text(encoding="utf-8"))
    assert contract["schema_version"] == 1
    sections = {
        name: values for name, values in contract.items() if name != "schema_version"
    }
    assert set(sections) == {"exports", "methods"}
    assert all(
        isinstance(values, dict)
        and all(isinstance(value, str) for value in values.values())
        for values in sections.values()
    )
    return cast(dict[str, dict[str, str]], sections)


def _canonical_public_callable_names(
    manifest: dict[str, tuple[str, ...]],
) -> tuple[str, ...]:
    """Return every concrete callable from canonical non-root owner manifests."""

    qualified_names: list[str] = []
    for module_name, exports in manifest.items():
        if module_name == "vercor":
            continue
        module = importlib.import_module(module_name)
        qualified_names.extend(
            f"{module_name}.{name}"
            for name in exports
            if inspect.isclass(getattr(module, name))
            or inspect.isroutine(getattr(module, name))
        )
    return tuple(qualified_names)


def _canonical_public_method_names(
    manifest: dict[str, tuple[str, ...]],
) -> tuple[str, ...]:
    """Return public class/protocol behavior, excluding inherited exceptions."""

    qualified_names: list[str] = []
    for owner_name in _canonical_public_callable_names(manifest):
        owner = _resolve_qualified_name(owner_name)
        if not inspect.isclass(owner):
            continue
        if issubclass(owner, BaseException):
            continue
        qualified_names.extend(
            f"{owner_name}.{method_name}"
            for method_name, method in inspect.getmembers(owner)
            if not method_name.startswith("_")
            and inspect.isroutine(method)
            and getattr(method, "__module__", None) is not None
        )
    qualified_names.append("vercor.regridding.RegridderFactory.__call__")
    return tuple(qualified_names)


def _resolve_qualified_name(qualified_name: str) -> object:
    """Resolve a documented object while keeping its canonical module explicit."""

    parts = qualified_name.split(".")
    for stop in range(len(parts), 0, -1):
        try:
            value: object = importlib.import_module(".".join(parts[:stop]))
        except ModuleNotFoundError:
            continue
        for attribute in parts[stop:]:
            value = getattr(value, attribute)
        return value
    raise AssertionError(f"cannot resolve documented signature {qualified_name}")


def _normalized_signature(value: object) -> str:
    """Return a stable, resolved signature including defaults and annotations."""

    callable_value = cast(Callable[..., object], value)
    hint_target = value.__init__ if inspect.isclass(value) else callable_value
    hints = get_type_hints(hint_target)
    try:
        signature = inspect.signature(callable_value)
    except ValueError:
        assert inspect.isclass(value) and issubclass(value, BaseException)
        init_signature = inspect.signature(value.__init__)
        signature = init_signature.replace(
            parameters=tuple(init_signature.parameters.values())[1:]
        )
    signature = signature.replace(
        parameters=[
            parameter.replace(
                annotation=hints.get(parameter.name, parameter.annotation)
            )
            for parameter in signature.parameters.values()
        ],
        return_annotation=hints.get("return", signature.return_annotation),
    )
    rendered = str(signature)
    rendered = re.sub(
        r"<function ([^ >]+) at 0x[0-9a-fA-F]+>",
        r"<function \1>",
        rendered,
    )
    rendered = re.sub(r"<object object at 0x[0-9a-fA-F]+>", "<object>", rendered)
    normalized = (
        rendered.replace("vercor.components.contracts.", "vercor.components.")
        .replace("vercor.components.contexts.", "vercor.components.")
        .replace("vercor.components.data.", "vercor.components.")
        .replace("vercor.setups.config.", "vercor.setups.")
        .replace("vercor.setups._jcm.", "vercor.setups.")
        .replace("pathlib._local.Path", "pathlib.Path")
        .replace(" -> NoneType", " -> None")
    )
    return canonicalize_external_typing_aliases(normalized)


@pytest.mark.parametrize(
    "rendered",
    (
        "numpy.ndarray[tuple[typing.Any, ...], numpy.dtype[typing.Any]]",
        "NDArray[typing.Any]",
        "numpy.typing.NDArray[typing.Any]",
    ),
)
def test_external_numpy_ndarray_renderings_have_one_public_token(
    rendered: str,
) -> None:
    """Keep equivalent NumPy aliases stable across dependency renderings."""

    assert (
        canonicalize_external_typing_aliases(rendered)
        == "numpy.typing.NDArray[typing.Any]"
    )


@pytest.mark.parametrize(
    "rendered",
    (
        "Union[jax.Array, numpy.ndarray, numpy.bool, numpy.number, bool, int, "
        "float, complex, jax._src.literals.TypedNdArray]",
        "Union[jax.Array, numpy.ndarray, numpy.bool, numpy.number, bool, int, "
        "float, complex]",
        "jax.typing.ArrayLike",
    ),
)
def test_external_jax_arraylike_renderings_have_one_public_token(
    rendered: str,
) -> None:
    """Keep equivalent JAX aliases stable without freezing private names."""

    assert canonicalize_external_typing_aliases(rendered) == "jax.typing.ArrayLike"


@pytest.mark.parametrize(
    "rendered",
    (
        "numpy.ndarray[tuple[typing.Any, ...], numpy.dtype[numpy.float64]]",
        "NDArray[numpy.float64]",
        "SomeNDArray[typing.Any]",
        "Union[jax.Array, numpy.ndarray, numpy.bool, numpy.number, bool, int, float]",
        "collections.abc.Sequence[NDArray[numpy.float64]]",
    ),
)
def test_external_typing_near_misses_remain_unchanged(rendered: str) -> None:
    """Avoid canonicalizing strings that are not the evidenced aliases."""

    assert canonicalize_external_typing_aliases(rendered) == rendered


@pytest.mark.fast_always
def test_architecture_review_has_exact_v0_4_title_and_eight_sections() -> None:
    """Keep the approved review shape exact without asserting explanatory prose."""

    review = REVIEW_PATH.read_text(encoding="utf-8")
    assert review.startswith("# VerCOR 0.4.0 API architecture review\n")
    assert tuple(re.findall(r"^## (.+)$", review, flags=re.MULTILINE)) == (
        REQUIRED_REVIEW_HEADINGS
    )


@pytest.mark.fast_always
def test_architecture_review_has_prioritized_decisions_and_complete_final_api() -> None:
    """Require the requested per-issue decisions and two-part final API."""

    review = REVIEW_PATH.read_text(encoding="utf-8")
    decisions = review.split("## 3. Bad design decisions", 1)[1].split(
        "## 4. Public API redesign", 1
    )[0]
    for column in ("Problem", "Consequence", "Concrete fix", "Priority"):
        assert f"| {column} " in decisions
    assert decisions.count("**must change**") >= 10
    assert decisions.count("**nice to improve**") >= 2

    final_api = review.split("## 8. Final rewritten API", 1)[1]
    assert "### 8.1 Complete public API" in final_api
    assert "### 8.2 Complete private API" in final_api
    for reference in (
        "section 4",
        "section 5",
        "vercor-0.4.0-public-signatures.json",
        "plugin-authoring.md",
    ):
        assert reference in final_api
    assert "Public-to-private relationships" in final_api


@pytest.mark.fast_always
def test_documented_public_manifest_matches_live_canonical_owners() -> None:
    """Execute the review's public inventory against live ``__all__`` values."""

    manifest = _documented_public_manifest(REVIEW_PATH.read_text(encoding="utf-8"))
    assert tuple(manifest) == ("vercor",) + tuple(sorted(manifest)[1:])
    for module_name, documented_exports in manifest.items():
        module = importlib.import_module(module_name)
        assert tuple(module.__all__) == documented_exports, module_name
        assert all(hasattr(module, name) for name in documented_exports), module_name


@pytest.mark.fast_always
def test_static_public_signature_contract_is_complete_and_matches_source() -> None:
    """Freeze every canonical callable export and relevant behavioral method."""

    manifest = _documented_public_manifest(REVIEW_PATH.read_text(encoding="utf-8"))
    contract = _public_signature_contract()
    serialized_contract = json.dumps(contract)
    assert "numpy.typing.NDArray[typing.Any]" in serialized_contract
    assert "jax.typing.ArrayLike" in serialized_contract
    assert (
        "numpy.ndarray[tuple[typing.Any, ...], numpy.dtype[typing.Any]]"
        not in serialized_contract
    )
    assert "jax._src.literals.TypedNdArray" not in serialized_contract
    assert tuple(contract["exports"]) == _canonical_public_callable_names(manifest)
    assert tuple(contract["methods"]) == _canonical_public_method_names(manifest)
    for qualified_name, documented_signature in {
        **contract["exports"],
        **contract["methods"],
    }.items():
        value = _resolve_qualified_name(qualified_name)
        assert _normalized_signature(value) == documented_signature


@pytest.mark.fast_always
def test_documented_private_inventory_matches_all_nonpublic_modules() -> None:
    """Keep the descriptive private inventory complete without freezing behavior."""

    review = REVIEW_PATH.read_text(encoding="utf-8")
    private_section = review.split("## 5. Private API redesign", 1)[1].split(
        "### Foundations and numerical implementations", 1
    )[0]
    match = re.search(r"```text\n(.*?)\n```", private_section, flags=re.DOTALL)
    assert match is not None
    documented = set(match.group(1).splitlines())

    public_modules = {"vercor", *_documented_public_manifest(review)}
    discovered: set[str] = set()
    for path in (PROJECT_ROOT / "vercor").rglob("*.py"):
        relative = path.relative_to(PROJECT_ROOT / "vercor")
        parts = (
            relative.parent.parts
            if relative.name == "__init__.py"
            else relative.with_suffix("").parts
        )
        discovered.add("vercor" + (f".{'.'.join(parts)}" if parts else ""))

    assert documented == discovered - public_modules


@pytest.mark.fast_always
def test_readme_python_snippets_run_as_one_public_quick_start(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Run every README Python block together, outside the repository directory."""

    snippets = _python_fences(README_PATH.read_text(encoding="utf-8"))
    assert snippets
    source = "\n\n".join(snippets)
    _assert_public_imports_only(source, owner="README.md")
    monkeypatch.chdir(tmp_path)

    namespace: dict[str, object] = {}
    exec(compile(source, str(README_PATH), "exec"), namespace)

    assert (tmp_path / "output" / "output.snapshot.nc").is_file()
    assert tuple((tmp_path / "output").glob("output.averages.*.nc"))


@pytest.mark.fast_always
def test_migration_v0_4_snippet_runs_without_private_or_compat_imports(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Execute the supported 0.4 migration result and verify its observable state."""

    snippets = _python_fences(MIGRATION_PATH.read_text(encoding="utf-8"))
    assert len(snippets) == 1
    source = snippets[0]
    _assert_public_imports_only(source, owner="docs/migration-0.3-to-0.4.md")
    assert "vercor.compat" not in source
    monkeypatch.chdir(tmp_path)

    namespace: dict[str, object] = {}
    exec(compile(source, str(MIGRATION_PATH), "exec"), namespace)

    migrated_temperature = cast(Any, namespace["migrated_temperature"])
    assert float(migrated_temperature[0, 0]) == pytest.approx(282.0)
    assert not tuple(tmp_path.iterdir())


@pytest.mark.fast_always
def test_release_files_and_metadata_describe_the_stable_release() -> None:
    """Bind release documentation to installed project metadata and artifact names."""

    project = tomllib.loads(
        (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )["project"]
    assert project["version"] == "0.4.1"
    assert "Development Status :: 5 - Production/Stable" in project["classifiers"]
    assert "Development Status :: 3 - Alpha" not in project["classifiers"]
    assert "Development Status :: 4 - Beta" not in project["classifiers"]

    changelog = CHANGELOG_PATH.read_text(encoding="utf-8")
    assert re.search(r"^## \[0\.4\.1\] - 2026-07-24$", changelog, re.MULTILINE)
    releasing = RELEASING_PATH.read_text(encoding="utf-8")
    commands = "\n".join(re.findall(r"```bash\n(.*?)```", releasing, re.DOTALL))
    for command in (
        "python -m build",
        "python -m pytest tests/ -q --tb=short",
        "python -m pytest tests/ -q --cov=vercor --cov-branch",
        "VERCOR_ARTIFACT_DIR",
        "tests/test_distribution_boundaries.py",
        "test_output_free_workflow_preserves_jvp_and_reverse_mode_gradients",
        "python -m external_extension_test_fixture.smoke",
        "shasum -a 256",
        "git diff --check",
    ):
        assert command in commands
    assert "git push" not in commands
    assert "twine upload" not in commands

    for artifact in (
        "vercor-0.4.1-py3-none-any.whl",
        "vercor-0.4.1.tar.gz",
    ):
        assert artifact in commands


@pytest.mark.fast_always
def test_readme_installation_metadata_matches_project_version() -> None:
    """Keep active installation guidance bound to the packaged release version."""

    project = tomllib.loads(
        (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )["project"]
    version = project["version"]
    readme = README_PATH.read_text(encoding="utf-8")

    assert readme.count(f"Version `{version}` is the current release.") == 2
    assert f'python -m pip install "vercor=={version}"' in readme
    assert f'python -m pip install --upgrade "vercor=={version}"' in readme


@pytest.mark.fast_always
def test_release_transcripts_are_well_formed_and_shell_syntax_valid() -> None:
    """Keep active release transcripts syntactically executable and unambiguous."""

    with pytest.raises(AssertionError, match="unterminated Markdown fence"):
        _markdown_fences("````bash\nprintf ok\n", owner="malformed.md")
    with pytest.raises(AssertionError, match="literal triple fence"):
        _markdown_fences(
            "```bash\npython -c 'print(\"```text\")'\n```\n",
            owner="malformed.md",
        )

    documents = [RELEASING_PATH]
    if TASK6_REPORT_PATH.is_file():
        documents.append(TASK6_REPORT_PATH)
        report = TASK6_REPORT_PATH.read_text(encoding="utf-8")
        for unsafe_command in (
            "python -m twine upload dist/",
            "git push --delete origin v0.4.0",
            "DOWNLOADED_WHEEL_SHA",
            "DOWNLOADED_SDIST_SHA",
        ):
            assert unsafe_command not in report
        assert report.count("`docs/releasing.md`") >= 1
        assert "tests/test_release_state_validator.py" in report
        assert "binds all four checkouts" in report
    for path in documents:
        fences = _markdown_fences(path.read_text(encoding="utf-8"), owner=str(path))
        transcripts = tuple(source for language, source in fences if language == "text")
        assert transcripts, f"{path} has no executable text transcript"
        for transcript_number, source in enumerate(transcripts, start=1):
            completed = subprocess.run(
                ["bash", "-n"],
                input=source,
                text=True,
                capture_output=True,
                check=False,
            )
            assert completed.returncode == 0, (
                f"{path} transcript {transcript_number} fails bash -n:\n"
                f"{completed.stderr}"
            )
            assert source.startswith("set -euo pipefail\n")
            assert 'test -n "${RELEASE_COMMIT:-}"' in source


@pytest.mark.fast_always
def test_release_workflow_checks_out_the_exact_triggering_commit() -> None:
    """Prevent pull-request CI from silently testing GitHub's synthetic merge SHA."""

    workflow = yaml.safe_load(WORKFLOW_PATH.read_text(encoding="utf-8"))
    checkout_steps = tuple(
        step
        for job in workflow["jobs"].values()
        for step in job["steps"]
        if step.get("uses") == CHECKOUT_ACTION
    )
    assert len(checkout_steps) == 6
    for step in checkout_steps:
        assert step.get("with", {}).get("ref") == (
            "${{ github.event.pull_request.head.sha || github.sha }}"
        )
        assert step.get("with", {}).get("persist-credentials") is False


@pytest.mark.fast_always
def test_release_publication_preflights_are_authenticated_and_fail_closed() -> None:
    """Require identity, ancestry, and absence checks at every publication boundary."""

    guide = RELEASING_PATH.read_text(encoding="utf-8")
    prepare = _section(guide, "## 5. Prepare the required release pull request")
    tag = _section(guide, "## 6. Create and verify the annotated tag")
    repo_url = "https://api.github.com/repos/nutrik/vercor"
    published_only_release_url = f"{repo_url}/releases/tags/v0.4.1"
    release_enumeration = (
        'gh api --paginate --slurp "repos/nutrik/vercor/releases?per_page=100"'
    )
    capability_probe = (
        "gh api --method POST repos/nutrik/vercor/releases/generate-notes"
    )
    pypi_url = "https://pypi.org/pypi/vercor/0.4.1/json"
    ancestry_check = 'git merge-base --is-ancestor "$MAIN_COMMIT" "$RELEASE_COMMIT"'
    exact_main_check = 'test "$MAIN_COMMIT" = "$RELEASE_COMMIT"'

    for section in (prepare, tag):
        assert "git fetch --no-tags origin main" in section
        assert 'MAIN_COMMIT="$(git rev-parse refs/remotes/origin/main)"' in section
        assert 'GH_TOKEN="$(gh auth token)"' in section
        assert capability_probe in section
        assert "-f tag_name=v0.4.1" in section
        assert '-f target_commitish="$RELEASE_COMMIT"' in section
        assert "release-capability.json" in section
        assert release_enumeration in section
        assert "tools/validate_release_state.py github-tag-absent" in section
        assert "--tag v0.4.1" in section
        assert published_only_release_url not in section
        assert "RELEASE_STATUS" not in section
        assert pypi_url in section
        assert 'test "$PYPI_STATUS" = "404"' in section
        capability_index = section.index(capability_probe)
        enumeration_index = section.index(release_enumeration)
        absence_index = section.index(
            "tools/validate_release_state.py github-tag-absent"
        )
        assert capability_index < enumeration_index < absence_index

    assert ancestry_check in prepare
    assert exact_main_check not in prepare
    assert ancestry_check not in tag
    assert exact_main_check in tag
    assert tag.index("git fetch --no-tags origin main") < tag.index(
        'MAIN_COMMIT="$(git rev-parse refs/remotes/origin/main)"'
    )
    assert tag.index(
        'MAIN_COMMIT="$(git rev-parse refs/remotes/origin/main)"'
    ) < tag.index(exact_main_check)
    assert tag.index(exact_main_check) < tag.index("git tag -a")
    assert prepare.index(release_enumeration) < guide.index("git tag -a")

    workflow = yaml.safe_load(WORKFLOW_PATH.read_text(encoding="utf-8"))
    publish_steps = workflow["jobs"]["publish-release"]["steps"]
    publish_commands = "\n".join(
        step.get("run", "") for step in publish_steps if isinstance(step, dict)
    )
    publish_action_steps = tuple(
        (index, step)
        for index, step in enumerate(publish_steps)
        if step.get("uses") == PYPI_PUBLISH_ACTION
    )
    assert len(publish_action_steps) == 1
    publish_action_index, _ = publish_action_steps[0]
    github_release_index = next(
        index
        for index, step in enumerate(publish_steps)
        if "gh release edit" in step.get("run", "")
    )

    assert "https://pypi.org/pypi/vercor/${VERSION}/json" in publish_commands
    assert 'test "$PYPI_STATUS" = "404"' in publish_commands
    assert publish_commands.count("releases/generate-notes") == 2
    assert "github-repository-push" not in publish_commands
    assert "repository.json" not in publish_commands
    assert (
        'gh api --paginate --slurp "repos/${GITHUB_REPOSITORY}/releases?per_page=100"'
        in publish_commands
    )
    assert "tools/validate_release_state.py github-releases" in publish_commands
    assert publish_commands.count("--allow-state absent") >= 2
    assert "--allow-state absent draft" not in publish_commands
    assert "--allow-state published" in publish_commands
    assert "PYPI_UPLOAD_REQUIRED" not in publish_commands
    assert "if" not in publish_action_steps[0][1]
    assert publish_action_index < github_release_index

    run_steps = tuple(
        (index, step["run"])
        for index, step in enumerate(publish_steps)
        if isinstance(step, dict) and "run" in step
    )
    initial_validation_index = next(
        index
        for index, command in run_steps
        if "tools/validate_release_state.py files" in command
    )
    twine_install_index = next(
        index
        for index, command in run_steps
        if "python -m pip install twine==6.2.0" in command
    )
    pre_publish_index = publish_action_index - 1
    pre_publish = publish_steps[pre_publish_index]["run"]
    post_publish_index = next(
        index for index, command in run_steps if "for attempt in {1..12}" in command
    )
    assert initial_validation_index < twine_install_index < pre_publish_index
    assert "tools/validate_release_state.py files" in pre_publish
    assert "git fetch --no-tags origin main" in pre_publish
    assert 'MAIN_COMMIT="$(git rev-parse refs/remotes/origin/main)"' in pre_publish
    assert 'test "$MAIN_COMMIT" = "$GITHUB_SHA"' in pre_publish
    assert pre_publish.rstrip().endswith('test "$REMOTE_TAG_COMMIT" = "$GITHUB_SHA"')
    assert publish_action_index < post_publish_index < github_release_index
    assert "python -m build" not in publish_commands
    assert "> SHA256SUMS" not in publish_commands
    assert "pip install --upgrade pip twine" not in publish_commands
    assert "python -m build" not in publish_commands


@pytest.mark.fast_always
def test_release_pr_transcript_uses_release_branch_and_draft_metadata() -> None:
    """Bind the active release PR transcript to the approved branch and copy."""

    guide = RELEASING_PATH.read_text(encoding="utf-8")
    prepare = _section(guide, "## 5. Prepare the required release pull request")
    release_branch = "release/vercor-0.4.1"
    branch_assignment = f'RELEASE_BRANCH="{release_branch}"'
    exact_title = '--title "Release VerCOR 0.4.1"'
    exact_body = (
        '--body "Fix the GitHub Release capability preflight and prepare the '
        "immutable VerCOR 0.4.1 recovery release. The existing v0.4.0 tag is "
        'unchanged."'
    )

    assert "refactor" not in guide
    branch_checks = tuple(
        line for line in guide.splitlines() if "git branch --show-current)" in line
    )
    assert branch_checks
    assert all('"$RELEASE_BRANCH"' in line for line in branch_checks)

    transcripts = tuple(
        source
        for language, source in _markdown_fences(prepare, owner="release PR section")
        if language == "text"
    )
    assert len(transcripts) == 3
    for transcript in transcripts:
        assert branch_assignment in transcript
        assert "export RELEASE_BRANCH" in transcript

    assert (
        "gh pr list --repo nutrik/vercor --state open --base main "
        '--head "$RELEASE_BRANCH" '
        "--json number,url,headRefName,baseRefName,headRefOid"
    ) in prepare
    assert (
        'gh pr create --repo nutrik/vercor --base main --head "$RELEASE_BRANCH" '
        "--draft"
    ) in prepare
    assert exact_title in prepare
    assert exact_body in prepare
    assert (
        "gh pr list --repo nutrik/vercor --state open --base main "
        '--head "$RELEASE_BRANCH" --json number'
    ) in prepare
    assert 'git push origin "$RELEASE_BRANCH"' in prepare
    assert '--event pull_request --branch "$RELEASE_BRANCH"' in prepare


@pytest.mark.fast_always
def test_release_recovery_preflight_proves_capability_before_enumerating() -> None:
    """Keep exact-state recovery bound to the non-mutating capability probe."""

    guide = RELEASING_PATH.read_text(encoding="utf-8")
    recovery_state = _section(guide, "## 9. Query exact public state before recovery")
    normalized_lines = tuple(
        line.strip() for line in recovery_state.splitlines() if line.strip()
    )
    capability_output = '> "$RECOVERY_STATE_DIR/release-capability.json"'
    release_enumeration = (
        'gh api --paginate --slurp "repos/nutrik/vercor/releases?per_page=100" '
        '> "$RECOVERY_STATE_DIR/releases.json"'
    )

    for required in (
        "gh api --method POST repos/nutrik/vercor/releases/generate-notes",
        "-f tag_name=v0.4.1",
        '-f target_commitish="$RELEASE_COMMIT"',
        capability_output,
        release_enumeration,
    ):
        assert required in recovery_state
    assert normalized_lines[normalized_lines.index(capability_output) + 1] == (
        release_enumeration
    )


@pytest.mark.fast_always
def test_workflow_run_blocks_are_bash_syntax_valid() -> None:
    """Parse every workflow shell script before GitHub Actions can execute it."""

    workflow = yaml.safe_load(WORKFLOW_PATH.read_text(encoding="utf-8"))
    for job_name, job in workflow["jobs"].items():
        for step_index, step in enumerate(job["steps"]):
            command = step.get("run")
            if not isinstance(command, str):
                continue
            completed = subprocess.run(
                ["bash", "-n"],
                input=command,
                text=True,
                capture_output=True,
                check=False,
            )
            step_name = step.get("name", "unnamed run step")
            assert completed.returncode == 0, (
                f"workflow job {job_name!r}, step {step_index} ({step_name!r}) "
                f"fails bash -n:\n{completed.stderr}"
            )


@pytest.mark.fast_always
def test_release_recovery_commands_verify_exact_state_before_mutation() -> None:
    """Bind every recovery mutation to one exact CI run and verified bytes."""

    guide = RELEASING_PATH.read_text(encoding="utf-8")
    recovery = _section(guide, "## 10. Safe recovery")
    for required in (
        'test -n "${RELEASE_RUN_ID:-}"',
        'gh run view "$RELEASE_RUN_ID"',
        "--json headSha --jq .headSha)",
        "--json event --jq .event)",
        "--json headBranch --jq .headBranch)",
        'gh run download "$RELEASE_RUN_ID"',
        "--name vercor-distributions",
        "--name vercor-release-manifest",
        'CI_DIST_DIR="$CI_RECOVERY_ROOT/dist"',
        'CI_MANIFEST="$CI_RECOVERY_ROOT/manifest/SHA256SUMS"',
        "tools/validate_release_state.py files",
        "vercor-0.4.1-py3-none-any.whl",
        "vercor-0.4.1.tar.gz",
    ):
        assert required in recovery
    assert "The local `dist/SHA256SUMS` is not authoritative" in guide

    for heading in ("### Missing PyPI wheel only", "### Missing PyPI sdist only"):
        section = _section(guide, heading)
        assert 'test -n "${RELEASE_COMMIT:-}"' in section
        assert 'test -n "${RELEASE_RUN_ID:-}"' in section
        assert "$CI_MANIFEST" in section
        assert "$CI_DIST_DIR" in section
        assert "https://pypi.org/pypi/vercor/0.4.1/json" in section
        assert section.count("tools/validate_release_state.py pypi") == 3
        assert "for attempt in {1..12}" in section
        assert 'case "$FINAL_PYPI_STATUS" in' in section
        assert "sleep 10" in section
        assert 'test "$PYPI_RECOVERY_VERIFIED" = "true"' in section
        assert (
            'REMOTE_TAG_COMMIT="$(git ls-remote origin '
            "'refs/tags/v0.4.1^{}' | awk '{print $1}')\""
        ) in section
        assert 'test "$REMOTE_TAG_COMMIT" = "$RELEASE_COMMIT"' in section
        assert (
            "python -m twine upload --repository-url " "https://upload.pypi.org/legacy/"
        ) in section
        assert section.index("python -m twine upload") < section.index(
            "for attempt in {1..12}"
        )

    github = _section(guide, "### Resume an exact GitHub draft")
    for required in (
        'test -n "${RELEASE_COMMIT:-}"',
        'test -n "${RELEASE_RUN_ID:-}"',
        "$CI_MANIFEST",
        "$CI_DIST_DIR",
        'gh api --paginate --slurp "repos/nutrik/vercor/releases?per_page=100"',
        "tools/validate_release_state.py github-releases",
        "--allow-state absent draft",
        "--allow-state draft",
        "--allow-state published",
        "gh release create v0.4.1",
        "--draft",
        "tools/validate_release_state.py github-upload-url",
        "https://uploads.github.com/repos/nutrik/vercor/releases/",
        "gh release edit v0.4.1",
        "--draft=false",
    ):
        assert required in github
    assert github.count("tools/validate_release_state.py github-upload-url") == 2
    assert github.count("https://uploads.github.com/repos/nutrik/vercor/releases/") == 2
    assert "--hostname uploads.github.com" not in github
    assert github.count("tools/validate_release_state.py github-releases") >= 4
    assert 'test "$REMOTE_TAG_COMMIT" = "$RELEASE_COMMIT"' in github
    assert github.count("check_tag_binding") >= 5

    for forbidden in (
        "--clobber",
        "gh release delete",
        "git push --delete",
        "git tag --delete",
        "DESTRUCTIVE_ASSET_CLOBBER_APPROVED",
    ):
        assert forbidden not in recovery


@pytest.mark.fast_always
def test_active_memory_is_current_and_historical_detail_is_archived() -> None:
    """Keep active orientation bounded while preserving the detailed history."""

    progress = PROGRESS_PATH.read_text(encoding="utf-8")
    assert len(progress.splitlines()) <= 180
    archive_paths = tuple(
        PROJECT_ROOT / path
        for path in re.findall(r"`(docs/progress-archive-[^`]+\.md)`", progress)
    )
    assert archive_paths
    assert all(path.is_file() for path in archive_paths)
    assert PROGRESS_ARCHIVE_PATH in archive_paths
    assert (
        hashlib.sha256(PROGRESS_ARCHIVE_PATH.read_bytes()).hexdigest()
        == PROGRESS_ARCHIVE_SHA256
    )
    assert "VerCOR 0.4.0 release verification" in progress

    design = DESIGN_PATH.read_text(encoding="utf-8")
    dependencies = DEPENDENCIES_PATH.read_text(encoding="utf-8")
    assert "vercor.compat.v0_3" not in design
    assert "vercor.compat.v0_3" not in dependencies
