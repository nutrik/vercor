"""Static and artifact-level tests for VerCOR distribution boundaries."""

from __future__ import annotations

import ast
import inspect
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import tarfile
import tomllib
import zipfile

import pytest
import yaml

import tests._distribution_support as distribution_support
from tests._distribution_support import (
    BuiltDistributions,
    EXPECTED_EXTENSION_FIXTURE_WHEEL_NAME,
    EXPECTED_SDIST_NAME,
    EXPECTED_VERSION,
    EXPECTED_WHEEL_NAME,
    build_distributions,
    build_external_extension_fixture,
    install_local_target,
)
from tests._signature_support import EXTERNAL_TYPING_ALIAS_REPLACEMENTS
from tests.test_api_architecture_review import (
    _public_signature_contract,
    _section,
)
from tests.test_setup_boundaries import _run_setup_probe
from tests.test_v0_4_public_api import PUBLIC_MODULE_EXPORTS

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXTERNAL_EXTENSION_FIXTURE_ROOT = (
    PROJECT_ROOT / "tests" / "fixtures" / "external_extension_test_fixture"
)
RELEASING_PATH = PROJECT_ROOT / "docs" / "releasing.md"
CHECKOUT_ACTION = "actions/checkout@11d5960a326750d5838078e36cf38b85af677262"
SETUP_PYTHON_ACTION = "actions/setup-python@a26af69be951a213d495a4c3e4e4022e16d87065"
UPLOAD_ARTIFACT_ACTION = (
    "actions/upload-artifact@ea165f8d65b6e75b540449e92b4886f43607fa02"
)
DOWNLOAD_ARTIFACT_ACTION = (
    "actions/download-artifact@d3f86a106a0bac45b974a628896c90dbdf5c8093"
)
PYPI_PUBLISH_ACTION = (
    "pypa/gh-action-pypi-publish@ba38be9e461d3875417946c167d0b5f3d385a247"
)
CODECOV_ACTION = "codecov/codecov-action@0fb7174895f61a3b6b78fc075e0cd60383518dac"
EXPECTED_INSTALLED_ROOT = (
    "Clock",
    "Coupler",
    "Exchange",
    "RectilinearGrid",
    "RunState",
    "RuntimeOptions",
)
EXPECTED_INSTALLED_OWNER_MANIFESTS = PUBLIC_MODULE_EXPORTS
EXPECTED_INSTALLED_SIGNATURES = _public_signature_contract()
STABLE_EXTENSION_MODULES = {
    "vercor.components",
    "vercor.coupler",
    "vercor.exchanges",
    "vercor.grids",
    "vercor.output",
    "vercor.physics",
    "vercor.regridding",
    "vercor.runtime",
    "vercor.state",
    "vercor.topology",
    "vercor.types",
}
REMOVED_PRIMARY_MODULES = (
    "vercor.coupling",
    "vercor.settings",
    "vercor.physical_constants",
    "vercor.host_arrays",
    "vercor.pytree",
    "vercor.interpolators",
)
FORBIDDEN_ARCHIVE_PARTS = {
    ".DS_Store",
    ".pytest_cache",
    "__MACOSX",
    "__pycache__",
}


@pytest.mark.fast_always
def test_active_sources_do_not_use_retired_public_plugin_fixture_name() -> None:
    active_paths = (
        PROJECT_ROOT / ".github" / "workflows" / "python-package.yml",
        PROJECT_ROOT / "DESIGN.md",
        PROJECT_ROOT / "DEPENDENCIES.md",
        PROJECT_ROOT / "CHANGELOG.md",
        *sorted((PROJECT_ROOT / "docs").glob("release-notes-*.md")),
        PROJECT_ROOT / "docs" / "plugin-authoring.md",
        PROJECT_ROOT / "docs" / "api-architecture-review.md",
        PROJECT_ROOT / "docs" / "releasing.md",
        PROJECT_ROOT / "tests" / "_distribution_support.py",
        PROJECT_ROOT / "tests" / "test_distribution_boundaries.py",
        PROJECT_ROOT / "tests" / "test_api_architecture_review.py",
        PROJECT_ROOT / "tests" / "test_v0_4_public_api.py",
    )
    retired_markers = (
        "tests/fixtures/" + "public_plugin",
        "vercor_" + "public_plugin",
        "public-" + "plugin fixture",
        "public " + "plugin fixture",
    )
    violations = {
        str(path.relative_to(PROJECT_ROOT)): marker
        for path in active_paths
        for marker in retired_markers
        if marker in path.read_text(encoding="utf-8")
    }
    assert violations == {}


def _forbidden_archive_members(names: set[str]) -> tuple[str, ...]:
    """Return generated platform/cache members that must never ship."""

    return tuple(
        sorted(
            name
            for name in names
            if Path(name).suffix == ".pyc"
            or FORBIDDEN_ARCHIVE_PARTS.intersection(Path(name).parts)
        )
    )


@pytest.fixture(scope="module")
def built_distributions(tmp_path_factory: pytest.TempPathFactory) -> BuiltDistributions:
    """Build once locally or reuse the explicitly supplied CI artifact bundle."""

    return build_distributions(
        PROJECT_ROOT,
        tmp_path_factory.mktemp("distribution-build") / "dist",
    )


@pytest.fixture(scope="module")
def external_extension_fixture_wheel(
    tmp_path_factory: pytest.TempPathFactory,
) -> Path:
    """Build the independently installed extension fixture once per module."""

    return build_external_extension_fixture(
        PROJECT_ROOT,
        tmp_path_factory.mktemp("external-extension-fixture-build"),
    )


@pytest.mark.fast_always
def test_runtime_metadata_separates_test_and_development_dependencies() -> None:
    metadata = tomllib.loads(
        (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )
    project = metadata["project"]
    runtime_dependencies = tuple(project["dependencies"])
    extras = project["optional-dependencies"]

    assert not any(
        dependency.lower().startswith("pytest") for dependency in runtime_dependencies
    )
    assert {"jcm", "veros", "test", "dev"}.issubset(extras)
    assert any(dependency.lower().startswith("pytest") for dependency in extras["test"])
    assert any(
        dependency.lower().startswith("pytest-cov") for dependency in extras["test"]
    )
    assert any(
        dependency.lower().startswith("pytest-cov") for dependency in extras["dev"]
    )
    assert "pytest-xdist>=3.7" in extras["test"]
    assert "pytest-xdist>=3.7" in extras["dev"]
    for tool in ("black", "build", "flake8", "mypy"):
        assert any(dependency.lower().startswith(tool) for dependency in extras["dev"])

    license_classifiers = tuple(
        classifier
        for classifier in project["classifiers"]
        if classifier.startswith("License ::")
    )
    assert license_classifiers == (
        "License :: OSI Approved :: Apache Software License",
    )

    coverage = metadata["tool"]["coverage"]
    assert coverage["run"]["branch"] is True
    assert coverage["report"]["fail_under"] == 90


@pytest.mark.fast_always
def test_pytest_defaults_use_measured_parallel_policy() -> None:
    metadata = tomllib.loads(
        (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )

    assert shlex.split(metadata["tool"]["pytest"]["ini_options"]["addopts"]) == [
        "-q",
        "-n4",
        "--dist=loadscope",
        "--max-worker-restart=0",
    ]


@pytest.mark.fast_always
def test_external_extension_test_fixture_is_present() -> None:
    assert (PROJECT_ROOT / "vercor" / "py.typed").is_file()
    required_files = (
        "pyproject.toml",
        "src/external_extension_test_fixture/__init__.py",
        "src/external_extension_test_fixture/plugin.py",
        "src/external_extension_test_fixture/smoke.py",
        "src/external_extension_test_fixture/py.typed",
        "use_site.py",
    )
    for relative_path in required_files:
        assert (
            EXTERNAL_EXTENSION_FIXTURE_ROOT / relative_path
        ).is_file(), relative_path


@pytest.mark.fast_always
def test_external_extension_fixture_is_isolated_and_never_imports_private_modules() -> (
    None
):
    assert EXTERNAL_EXTENSION_FIXTURE_ROOT.is_dir()
    python_paths = sorted(EXTERNAL_EXTENSION_FIXTURE_ROOT.rglob("*.py"))
    assert python_paths

    for path in python_paths:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
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
                    ), f"{path} imports private VerCOR module {module}"
                    assert (
                        module == "vercor" or module in STABLE_EXTENSION_MODULES
                    ), f"{path} imports unstable VerCOR module {module}"
                    if isinstance(node, ast.ImportFrom) and module == "vercor":
                        assert all(
                            alias.name in EXPECTED_INSTALLED_ROOT
                            for alias in node.names
                        ), f"{path} imports a non-root contract from vercor"


@pytest.mark.fast_always
def test_current_external_extension_fixture_uses_canonical_owners_and_v0_4_workflows() -> (
    None
):
    fixture_project = EXTERNAL_EXTENSION_FIXTURE_ROOT / "pyproject.toml"
    assert fixture_project.is_file()
    project = tomllib.loads(fixture_project.read_text(encoding="utf-8"))["project"]
    source = (
        EXTERNAL_EXTENSION_FIXTURE_ROOT
        / "src/external_extension_test_fixture/plugin.py"
    ).read_text(encoding="utf-8")

    assert project["version"] == "0.1.0"
    assert project["dependencies"] == ["vercor>=0.4.0,<0.5"]
    for owner in (
        "vercor.components",
        "vercor.coupler",
        "vercor.exchanges",
        "vercor.grids",
        "vercor.output",
        "vercor.regridding",
        "vercor.runtime",
        "vercor.state",
        "vercor.topology",
        "vercor.types",
    ):
        assert f"from {owner} import" in source, owner
    assert "from vercor import Clock" in source
    for contract in (
        "DataComponent",
        "PluginConfig",
        "PluginRegridderFactory",
        "PluginWorkflow",
        "SetupResult(",
        "Exchange(",
        "ExchangeTopologyPatch(",
        "fractional_masks=",
        "PeriodOutput(",
        "StepResult(",
        ".replace_fields(",
    ):
        assert contract in source, contract


@pytest.mark.fast_always
def test_tag_release_rejects_version_mismatch_before_build() -> None:
    workflow = yaml.safe_load(
        (PROJECT_ROOT / ".github/workflows/python-package.yml").read_text(
            encoding="utf-8"
        )
    )
    steps = workflow["jobs"]["build-artifacts"]["steps"]
    guard_index, guard = next(
        (index, step)
        for index, step in enumerate(steps)
        if step.get("name") == "Reject mismatched release tag"
    )
    build_index = next(
        index
        for index, step in enumerate(steps)
        if step.get("name") == "Build VerCOR distributions once"
    )
    assert guard_index < build_index
    assert guard["if"] == "startsWith(github.ref, 'refs/tags/v')"
    assert guard["env"]["VERSION"] == ("${{ steps.project-metadata.outputs.version }}")

    for tag, returncode in (
        (f"v{EXPECTED_VERSION}", 0),
        (f"v{EXPECTED_VERSION}-mismatch", 1),
    ):
        completed = subprocess.run(
            ["bash"],
            input=guard["run"],
            cwd=PROJECT_ROOT,
            env={
                **os.environ,
                "GITHUB_REF_NAME": tag,
                "VERSION": EXPECTED_VERSION,
            },
            text=True,
            capture_output=True,
            check=False,
        )
        assert completed.returncode == returncode
        if returncode:
            assert "Release tag/version mismatch" in completed.stdout
            assert f"package version {EXPECTED_VERSION}" in completed.stdout


@pytest.mark.fast_always
def test_ci_project_metadata_step_derives_outputs_from_pyproject(
    tmp_path: Path,
) -> None:
    """Execute metadata extraction against a non-repository project version."""

    workflow = yaml.safe_load(
        (PROJECT_ROOT / ".github/workflows/python-package.yml").read_text(
            encoding="utf-8"
        )
    )
    build_job = workflow["jobs"]["build-artifacts"]
    assert build_job["outputs"] == {
        "version": "${{ steps.project-metadata.outputs.version }}",
        "wheel_name": "${{ steps.project-metadata.outputs.wheel_name }}",
        "sdist_name": "${{ steps.project-metadata.outputs.sdist_name }}",
    }
    metadata_step = next(
        step for step in build_job["steps"] if step.get("id") == "project-metadata"
    )

    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "vercor"\nversion = "9.8.7"\n',
        encoding="utf-8",
    )
    github_output = tmp_path / "github-output"
    completed = subprocess.run(
        ["bash"],
        input=metadata_step["run"],
        cwd=tmp_path,
        env={**os.environ, "GITHUB_OUTPUT": str(github_output)},
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert github_output.read_text(encoding="utf-8").splitlines() == [
        "version=9.8.7",
        "wheel_name=vercor-9.8.7-py3-none-any.whl",
        "sdist_name=vercor-9.8.7.tar.gz",
    ]


@pytest.mark.fast_always
def test_ci_validates_installed_artifacts_across_supported_environments() -> None:
    workflow_path = PROJECT_ROOT / ".github/workflows/python-package.yml"
    workflow = yaml.safe_load(workflow_path.read_text(encoding="utf-8"))
    jobs = workflow["jobs"]
    build_job = jobs["build-artifacts"]
    installed_job = jobs["installed-artifact-tests"]
    assert "plugin-contract-tests" not in jobs
    assert "external-extension-contract-tests" in jobs
    extension_job = jobs["external-extension-contract-tests"]
    macos_job = jobs["macos-smoke"]
    expected_outputs = {
        "version": "${{ steps.project-metadata.outputs.version }}",
        "wheel_name": "${{ steps.project-metadata.outputs.wheel_name }}",
        "sdist_name": "${{ steps.project-metadata.outputs.sdist_name }}",
    }
    assert build_job["outputs"] == expected_outputs

    build_steps = build_job["steps"]
    build_commands = "\n".join(
        step.get("run", "") for step in build_steps if isinstance(step, dict)
    )
    assert "python -m build --outdir dist" in build_commands
    assert "shopt -s nullglob dotglob" in build_commands
    assert "DIST_ARTIFACTS=(dist/*)" in build_commands
    assert 'test "${#DIST_ARTIFACTS[@]}" -eq 2' in build_commands
    assert 'test -f "dist/${WHEEL_NAME}"' in build_commands
    assert 'test -f "dist/${SDIST_NAME}"' in build_commands
    assert "tests/fixtures/external_extension_test_fixture" not in build_commands
    assert "external_extension_test_fixture" not in build_commands
    upload_steps = {
        step["with"]["name"]: step
        for step in build_steps
        if step.get("uses") == UPLOAD_ARTIFACT_ACTION
    }
    assert set(upload_steps) == {
        "vercor-distributions",
        "vercor-release-manifest",
    }
    assert set(upload_steps["vercor-distributions"]["with"]["path"].splitlines()) == {
        "dist/${{ steps.project-metadata.outputs.wheel_name }}",
        "dist/${{ steps.project-metadata.outputs.sdist_name }}",
    }
    assert (
        upload_steps["vercor-release-manifest"]["with"]["path"]
        == "release-manifest/SHA256SUMS"
    )
    assert (
        'sha256sum "$WHEEL_NAME" "$SDIST_NAME" ' "> ../release-manifest/SHA256SUMS"
    ) in build_commands

    matrix = installed_job["strategy"]["matrix"]
    assert matrix["python-version"] == ["3.12", "3.13"]
    assert matrix["environment"] == ["base", "jcm", "veros"]
    assert matrix["artifact"] == ["wheel"]
    included = {item["environment"]: item for item in matrix["include"]}
    assert {"base", "jcm", "veros"}.issubset(included)
    assert included["jcm"]["extra"] == "[jcm,veros]"
    assert included["veros"]["extra"] == "[jcm,veros]"
    assert (
        "test_make_jcm_land_atmosphere_replaces_only_missing_forcing"
        in included["jcm"]["pytest-target"]
    )
    assert (
        "test_jax_gcm_initialize_uses_provided_forcing_and_can_spin_up"
        in included["jcm"]["pytest-target"]
    )
    assert (
        "test_veros_initialize_spinup_follows_enabled_only"
        in included["veros"]["pytest-target"]
    )

    installed_steps = installed_job["steps"]
    installed_commands = "\n".join(
        step.get("run", "") for step in installed_steps if isinstance(step, dict)
    )
    download_step = next(
        step for step in installed_steps if step.get("uses") == DOWNLOAD_ARTIFACT_ACTION
    )
    assert download_step["with"]["path"] == "dist/"
    assert "python -m build" not in installed_commands
    assert "VERCOR_ARTIFACT_DIR" in installed_commands
    assert "VERCOR_TEST_PACKAGE_ROOT" in installed_commands
    installed_step = next(
        step
        for step in installed_steps
        if step.get("name") == "Install selected artifact environment"
    )
    assert installed_step["env"] == {
        "WHEEL_NAME": "${{ needs.build-artifacts.outputs.wheel_name }}",
        "SDIST_NAME": "${{ needs.build-artifacts.outputs.sdist_name }}",
    }
    assert 'WHEEL_PATH="${GITHUB_WORKSPACE}/dist/${WHEEL_NAME}"' in installed_commands
    assert 'SDIST_PATH="${GITHUB_WORKSPACE}/dist/${SDIST_NAME}"' in installed_commands
    assert "vercor-0.3.0-py3-none-any.whl" not in installed_commands
    assert (
        "tests/fixtures/external_extension_test_fixture/src" not in installed_commands
    )
    assert "pip install ." not in installed_commands
    install_tools_line = next(
        line.strip()
        for line in installed_commands.splitlines()
        if "pip install --upgrade pip" in line
    )
    installed_tools = set(shlex.split(install_tools_line))
    assert {"build", "flit_core<4"}.issubset(installed_tools)
    assert "pytest-xdist>=3.7" in installed_tools

    assert extension_job["strategy"]["matrix"] == {
        "python-version": ["3.12", "3.13"],
    }
    extension_commands = "\n".join(
        step.get("run", "") for step in extension_job["steps"] if isinstance(step, dict)
    )
    assert (
        'python -m build --wheel --outdir "$RUNNER_TEMP/'
        'external-extension-fixture-dist" '
        "tests/fixtures/external_extension_test_fixture"
    ) in extension_commands
    assert EXPECTED_EXTENSION_FIXTURE_WHEEL_NAME in extension_commands
    extension_step = next(
        step
        for step in extension_job["steps"]
        if step.get("name") == "Run installed external extension contract"
    )
    assert extension_step["env"] == {
        "WHEEL_NAME": "${{ needs.build-artifacts.outputs.wheel_name }}",
    }
    assert (
        'python -m pip install "${GITHUB_WORKSPACE}/dist/${WHEEL_NAME}"'
        in extension_commands
    )
    assert "external_extension_test_fixture.smoke" in extension_commands
    assert (
        'python -P -m mypy --strict "$RUNNER_TEMP/extension-typecheck/'
        'external_extension_test_fixture" external_extension_use_site.py'
        in extension_commands
    )

    assert macos_job["runs-on"] == "macos-latest"
    macos_commands = "\n".join(
        step.get("run", "") for step in macos_job["steps"] if isinstance(step, dict)
    )
    macos_step = next(
        step
        for step in macos_job["steps"]
        if step.get("name") == "Run installed macOS external extension smoke"
    )
    assert macos_step["env"] == {
        "WHEEL_NAME": "${{ needs.build-artifacts.outputs.wheel_name }}",
    }
    assert 'python -m pip install "dist/${WHEEL_NAME}"' in macos_commands
    assert "external_extension_test_fixture.smoke" in macos_commands
    assert "tests/fixtures/external_extension_test_fixture" in macos_commands
    macos_checkout = next(
        step for step in macos_job["steps"] if step.get("uses") == CHECKOUT_ACTION
    )
    assert (
        macos_checkout["with"]["ref"]
        == "${{ github.event.pull_request.head.sha || github.sha }}"
    )


@pytest.mark.fast_always
def test_version_tag_deploys_exact_tested_distributions() -> None:
    workflow_path = PROJECT_ROOT / ".github/workflows/python-package.yml"
    workflow_source = workflow_path.read_text(encoding="utf-8")
    workflow = yaml.safe_load(workflow_source)
    triggers = workflow[True]
    publish = workflow["jobs"]["publish-release"]
    expected_outputs = {
        "version": "${{ steps.project-metadata.outputs.version }}",
        "wheel_name": "${{ steps.project-metadata.outputs.wheel_name }}",
        "sdist_name": "${{ steps.project-metadata.outputs.sdist_name }}",
    }

    assert triggers["push"] == {
        "branches": ["main"],
        "tags": ["v*.*.*"],
    }
    assert publish["if"] == (
        "github.event_name == 'push' && " "startsWith(github.ref, 'refs/tags/v')"
    )
    assert publish["needs"] == [
        "build-artifacts",
        "installed-artifact-tests",
        "external-extension-contract-tests",
        "macos-smoke",
        "quality",
    ]
    assert publish["runs-on"] == "ubuntu-latest"
    assert publish["environment"] == {
        "name": "release",
        "url": "https://pypi.org/p/vercor",
    }
    assert publish["concurrency"] == {
        "group": "release-${{ github.ref_name }}",
        "cancel-in-progress": False,
    }
    assert workflow["permissions"] == {"contents": "read"}
    assert publish["permissions"] == {"contents": "write"}
    assert workflow["jobs"]["build-artifacts"]["outputs"] == expected_outputs
    permission_owners = (
        ("workflow", workflow.get("permissions", {})),
        *workflow["jobs"].items(),
    )
    for owner, permissions in permission_owners:
        permission_map = (
            permissions if owner == "workflow" else permissions.get("permissions", {})
        )
        assert "id-token" not in permission_map, owner
        if owner != "publish-release":
            assert permission_map.get("contents") != "write", owner

    checkout = next(
        step for step in publish["steps"] if step.get("uses") == CHECKOUT_ACTION
    )
    setup = next(
        step for step in publish["steps"] if step.get("uses") == SETUP_PYTHON_ACTION
    )
    downloads = {
        step["with"]["name"]: step
        for step in publish["steps"]
        if step.get("uses") == DOWNLOAD_ARTIFACT_ACTION
    }
    assert checkout["with"]["ref"] == (
        "${{ github.event.pull_request.head.sha || github.sha }}"
    )
    assert checkout["with"]["persist-credentials"] is False
    assert setup["with"]["python-version"] == "3.12"
    assert downloads["vercor-distributions"]["with"] == {
        "name": "vercor-distributions",
        "path": "dist/",
    }
    assert downloads["vercor-release-manifest"]["with"] == {
        "name": "vercor-release-manifest",
        "path": "release-manifest/",
    }

    run_steps = tuple(
        (index, step["run"])
        for index, step in enumerate(publish["steps"])
        if isinstance(step, dict) and "run" in step
    )
    commands = "\n".join(command for _, command in run_steps)
    normalized_commands = " ".join(commands.replace("\\\n", " ").split())
    inventory_step = next(
        step
        for step in publish["steps"]
        if step.get("name") == "Verify CI-produced release inventory"
    )
    assert inventory_step["env"] == {
        "PROJECT_VERSION": "${{ needs.build-artifacts.outputs.version }}",
        "WHEEL_NAME": "${{ needs.build-artifacts.outputs.wheel_name }}",
        "SDIST_NAME": "${{ needs.build-artifacts.outputs.sdist_name }}",
    }
    for required in (
        'test "$GITHUB_REF_TYPE" = "tag"',
        'test "$PROJECT_VERSION" = "$VERSION"',
        'test "$WHEEL_NAME" = "vercor-${VERSION}-py3-none-any.whl"',
        'test "$SDIST_NAME" = "vercor-${VERSION}.tar.gz"',
        'test "$GITHUB_REF_NAME" = "v${VERSION}"',
        'test -f "docs/release-notes-${VERSION}.md"',
        "DIST_ARTIFACTS=(dist/*)",
        'test "${#DIST_ARTIFACTS[@]}" -eq 2',
        'WHEEL="dist/${WHEEL_NAME}"',
        'SDIST="dist/${SDIST_NAME}"',
        'python -m twine check "$WHEEL" "$SDIST"',
        "release-manifest/SHA256SUMS",
        "tools/validate_release_state.py files",
        "https://pypi.org/pypi/vercor/${VERSION}/json",
        "tools/validate_release_state.py pypi",
        "for attempt in {1..12}",
        'case "$PYPI_STATUS" in',
        "sleep 10",
        'REMOTE_TAG_COMMIT="$(git ls-remote origin '
        '"refs/tags/${GITHUB_REF_NAME}^{}"',
        'test "$REMOTE_TAG_COMMIT" = "$GITHUB_SHA"',
        'test "$(git rev-parse HEAD)" = "$GITHUB_SHA"',
        'gh api --paginate --slurp "repos/${GITHUB_REPOSITORY}/releases?per_page=100"',
        "tools/validate_release_state.py github-releases",
        'gh release create "$GITHUB_REF_NAME"',
        "--draft",
        '--notes-file "docs/release-notes-${RELEASE_VERSION}.md"',
        "tools/validate_release_state.py github-upload-url",
        "https://uploads.github.com/repos/${GITHUB_REPOSITORY}/releases/",
        '--input "$RELEASE_WHEEL"',
        '--input "$RELEASE_SDIST"',
        'gh release edit "$GITHUB_REF_NAME"',
        "--draft=false",
    ):
        assert required in commands

    version_step_index, version_command = next(
        (index, command)
        for index, command in run_steps
        if "VERSION=" in command and "pyproject.toml" in command
    )
    assert "tomllib" in version_command
    assert '"project"]["version"]' in version_command
    tag_check_index = next(
        index
        for index, command in run_steps
        if 'test "$GITHUB_REF_NAME" = "v${VERSION}"' in command
    )
    assert version_step_index <= tag_check_index

    twine_install_index = next(
        index
        for index, command in run_steps
        if "python -m pip install twine==" in command
    )
    assert "python -m pip install twine==6.2.0" in commands
    assert "pip install --upgrade pip twine" not in commands
    manifest_verification_indices = tuple(
        index
        for index, command in run_steps
        if "tools/validate_release_state.py files" in command
    )
    assert any(index < twine_install_index for index in manifest_verification_indices)
    assert any(index > twine_install_index for index in manifest_verification_indices)
    producer_manifest = (
        'sha256sum "$WHEEL_NAME" "$SDIST_NAME" ' "> ../release-manifest/SHA256SUMS"
    )
    assert workflow_source.count(producer_manifest) == 1
    assert producer_manifest not in commands
    assert "> SHA256SUMS" not in commands
    assert "python -m build" not in commands

    capability_endpoint = '"repos/${GITHUB_REPOSITORY}/releases/generate-notes"'
    capability_output = '> "$STATE_DIR/release-capability.json"'
    release_enumeration = (
        "gh api --paginate --slurp "
        '"repos/${GITHUB_REPOSITORY}/releases?per_page=100"'
    )
    capability_preflights = tuple(
        step["run"]
        for step in publish["steps"]
        if step.get("name")
        in {
            "Classify exact public release state",
            "Revalidate immediately before PyPI mutation",
        }
    )
    assert len(capability_preflights) == 2
    for preflight in capability_preflights:
        normalized_lines = tuple(
            line.strip() for line in preflight.splitlines() if line.strip()
        )
        assert "gh api --method POST" in preflight
        assert capability_endpoint in preflight
        assert '-f tag_name="$GITHUB_REF_NAME"' in preflight
        assert '-f target_commitish="$GITHUB_SHA"' in preflight
        assert capability_output in preflight
        assert preflight.index(capability_endpoint) < preflight.index(
            release_enumeration
        )
        assert normalized_lines[
            normalized_lines.index(capability_output) + 1
        ].startswith(release_enumeration)
        assert "github-repository-push" not in preflight
        assert 'repos/${GITHUB_REPOSITORY}" >' not in preflight

    assert workflow_source.count("releases/generate-notes") == 2
    assert "github-repository-push" not in workflow_source
    assert "repository.json" not in workflow_source

    pypi_publish_steps = tuple(
        (index, step)
        for index, step in enumerate(publish["steps"])
        if step.get("uses") == PYPI_PUBLISH_ACTION
    )
    assert len(pypi_publish_steps) == 1
    pypi_publish = pypi_publish_steps[0]
    pypi_publish_index, pypi_publish_step = pypi_publish
    assert "if" not in pypi_publish_step
    assert pypi_publish_step["with"] == {
        "user": "__token__",
        "password": "${{ secrets.PYPI_API_TOKEN }}",
        "packages-dir": "dist/",
        "skip-existing": False,
        "attestations": False,
    }
    pre_publish_index = pypi_publish_index - 1
    pre_publish = publish["steps"][pre_publish_index]["run"]
    assert "tools/validate_release_state.py files" in pre_publish
    assert 'test "$PYPI_STATUS" = "404"' in pre_publish
    main_fetch = "git fetch --no-tags origin main"
    main_binding = 'MAIN_COMMIT="$(git rev-parse refs/remotes/origin/main)"'
    exact_main_check = 'test "$MAIN_COMMIT" = "$GITHUB_SHA"'
    assert main_fetch in pre_publish
    assert main_binding in pre_publish
    assert exact_main_check in pre_publish
    assert pre_publish.index(main_fetch) < pre_publish.index(main_binding)
    assert pre_publish.index(main_binding) < pre_publish.index(exact_main_check)
    assert pre_publish.rstrip().endswith('test "$REMOTE_TAG_COMMIT" = "$GITHUB_SHA"')

    post_pypi_index = next(
        index for index, command in run_steps if "for attempt in {1..12}" in command
    )
    github_release_index = next(
        index
        for index, command in run_steps
        if 'gh release edit "$GITHUB_REF_NAME"' in command
    )
    assert pypi_publish_index < post_pypi_index < github_release_index
    assert commands.count("tools/validate_release_state.py github-releases") == 3
    assert commands.count("tools/wait_for_github_release_state.py") == 4
    for visibility_contract in (
        "--target-state draft --target-present "
        "--transitional-state absent --transitional-present",
        '--target-state draft --target-present "$RELEASE_WHEEL_NAME" '
        "--transitional-state draft --transitional-present",
        "--target-state draft --target-present "
        '"$RELEASE_WHEEL_NAME" "$RELEASE_SDIST_NAME" '
        "--transitional-state draft --transitional-present "
        '"$RELEASE_WHEEL_NAME"',
        "--target-state published --target-present "
        '"$RELEASE_WHEEL_NAME" "$RELEASE_SDIST_NAME" '
        "--transitional-state draft --transitional-present "
        '"$RELEASE_WHEEL_NAME" "$RELEASE_SDIST_NAME"',
    ):
        assert visibility_contract in normalized_commands
    assert normalized_commands.count("--attempts 12 --interval-seconds 2") == 4
    assert commands.count("tools/validate_release_state.py github-upload-url") == 2
    assert (
        commands.count(
            "https://uploads.github.com/repos/${GITHUB_REPOSITORY}/releases/"
        )
        == 2
    )
    assert commands.count('test "$REMOTE_TAG_COMMIT" = "$GITHUB_SHA"') >= 4
    assert commands.count('gh release edit "$GITHUB_REF_NAME"') == 1
    assert "--allow-state absent draft" not in commands
    assert "PYPI_UPLOAD_REQUIRED" not in commands
    assert commands.count('test "$PYPI_STATUS" = "404"') >= 2
    for forbidden in (
        "--clobber",
        "gh release delete",
        "gh api -x delete",
        "gh api --method delete",
        "--hostname uploads.github.com",
        "git push --delete",
        "git tag --delete",
        "twine upload",
        "--skip-existing",
        "skip-existing: true",
    ):
        assert forbidden not in workflow_source.lower()
    assert workflow_source.count("pypa/gh-action-pypi-publish@") == 1
    assert "TEST_PYPI_API_TOKEN" not in workflow_source
    assert workflow_source.count("secrets.PYPI_API_TOKEN") == 1


@pytest.mark.fast_always
def test_release_provenance_actions_are_immutable_and_checkouts_drop_credentials() -> (
    None
):
    """Pin every provenance action and keep release-path checkouts credential-free."""

    workflow_source = (PROJECT_ROOT / ".github/workflows/python-package.yml").read_text(
        encoding="utf-8"
    )
    workflow = yaml.safe_load(workflow_source)
    required_actions = {
        CHECKOUT_ACTION,
        SETUP_PYTHON_ACTION,
        UPLOAD_ARTIFACT_ACTION,
        DOWNLOAD_ARTIFACT_ACTION,
        PYPI_PUBLISH_ACTION,
        CODECOV_ACTION,
    }
    used_actions = {
        step["uses"]
        for job in workflow["jobs"].values()
        for step in job["steps"]
        if "uses" in step
    }
    assert required_actions.issubset(used_actions)
    assert all(
        len(reference.rsplit("@", maxsplit=1)[1]) == 40
        and all(character in "0123456789abcdef" for character in reference[-40:])
        for reference in used_actions
    )
    for mutable_reference in (
        "actions/checkout@v",
        "actions/setup-python@v",
        "actions/upload-artifact@v",
        "actions/download-artifact@v",
    ):
        assert mutable_reference not in workflow_source
    checkout_steps = tuple(
        step
        for job in workflow["jobs"].values()
        for step in job["steps"]
        if step.get("uses") == CHECKOUT_ACTION
    )
    assert len(checkout_steps) == 6
    assert all(
        step.get("with", {}).get("persist-credentials") is False
        for step in checkout_steps
    )


@pytest.mark.fast_always
def test_release_design_and_plan_describe_the_final_review_state_machine() -> None:
    """Keep release architecture documents aligned with executable provenance."""

    design = (
        PROJECT_ROOT
        / "docs"
        / "superpowers"
        / "specs"
        / "2026-07-24-automated-release-deployment-design.md"
    ).read_text(encoding="utf-8")
    plan = (
        PROJECT_ROOT
        / "docs"
        / "superpowers"
        / "plans"
        / "2026-07-24-automated-release-deployment.md"
    ).read_text(encoding="utf-8")
    for document in (design, plan):
        for required in (
            "Final-review corrected",
            "`vercor-release-manifest`",
            "producer-issued manifest",
            "Twine 6.2.0",
            "per-tag",
            "peeled remote tag",
            "zero, one, or two",
            "`gh release edit",
            "--draft=false",
            "ordinary",
            "rerun",
            "canonical `https://uploads.github.com` request target",
            "draft-aware pre-tag",
            "bounded missing-file recovery polling",
            "non-mutating Release notes-generation",
        ):
            assert required in document
        assert "repository `permissions.push` is true" not in document
        for stale in (
            "actions/checkout@v4",
            "actions/setup-python@v5",
            "actions/upload-artifact@v4",
            "actions/download-artifact@v4",
            "pip install --upgrade pip twine",
        ):
            assert stale not in document


@pytest.mark.fast_always
def test_release_bundle_contains_only_vercor_distributions() -> None:
    expected_tag = f"v{EXPECTED_VERSION}"
    releasing = RELEASING_PATH.read_text(encoding="utf-8")
    checksum_line = next(
        line.strip()
        for line in releasing.splitlines()
        if line.strip().startswith("shasum -a 256 vercor-")
    )

    assert checksum_line == (
        f"shasum -a 256 {EXPECTED_WHEEL_NAME} " f"{EXPECTED_SDIST_NAME} > SHA256SUMS"
    )
    assert "shopt -s nullglob dotglob" in releasing
    assert "DIST_ARTIFACTS=(dist/*)" in releasing
    assert 'test "${#DIST_ARTIFACTS[@]}" -eq 0' in releasing
    assert 'test "${#DIST_ARTIFACTS[@]}" -eq 2' in releasing
    assert f"test -f dist/{EXPECTED_WHEEL_NAME}" in releasing
    assert f"test -f dist/{EXPECTED_SDIST_NAME}" in releasing
    publish = _section(
        releasing,
        "## 7. Publish packages and create the hosted release",
    )
    assert f"Pushing the annotated `{expected_tag}` tag" in publish
    assert "`PYPI_API_TOKEN`" in publish
    assert "python-package.yml" in publish
    assert "gh run watch" in publish
    assert "python -m twine upload" not in publish
    assert "gh release create" not in publish
    assert "dist/external_extension_test_fixture" not in releasing


@pytest.mark.fast_always
def test_progress_records_refreshed_corrected_release_artifacts() -> None:
    """Reject package evidence that predates the active README and runbook fixes."""

    progress = (PROJECT_ROOT / "PROGRESS.md").read_text(encoding="utf-8")
    current_status = _section(progress, "## Current Status")
    marker = "evidence was refreshed after the README and release-runbook corrections"
    evidence = next(
        (
            entry.group()
            for entry in re.finditer(
                r"^- .*?(?=^- |\Z)",
                current_status,
                flags=re.MULTILINE | re.DOTALL,
            )
            if marker in entry.group()
        ),
        None,
    )
    assert evidence is not None
    assert marker in evidence
    assert "commit `32b8276`" not in evidence
    for artifact in (EXPECTED_WHEEL_NAME, EXPECTED_SDIST_NAME):
        assert re.search(
            rf"`{re.escape(artifact)}`, [1-9][0-9]* B, SHA-256 `[0-9a-f]{{64}}`",
            evidence,
        )
    for stale_digest in (
        "653adb66c1507aa3ead8abb34a7f18a7dfe243ee8c11500e28a8fa33ea042112",
        "5ae1cf3146fb4aae9b289f1da6bcf228a18d534183cd38108f5d15f16dd5d71f",
    ):
        assert stale_digest not in evidence


@pytest.mark.fast_always
def test_release_guide_binds_tag_authority_workflow_selection_and_hosted_state() -> (
    None
):
    """Require the ordinary tagged-release handoff to remain fail-closed."""

    expected_tag = f"v{EXPECTED_VERSION}"
    expected_title = f"VerCOR {EXPECTED_VERSION}"
    releasing = RELEASING_PATH.read_text(encoding="utf-8")
    deployment = _section(releasing, "## Repository deployment configuration")
    prepare = _section(releasing, "## 5. Prepare the required release pull request")
    tag = _section(releasing, "## 6. Create and verify the annotated tag")
    publish = _section(
        releasing, "## 7. Publish packages and create the hosted release"
    )
    verify = _section(
        releasing, "## 8. Verify the published package and hosted release"
    )
    prepare_text = " ".join(prepare.split())
    tag_text = " ".join(tag.split())
    publish_text = " ".join(publish.split())

    for required in (
        "`PYPI_API_TOKEN`",
        "`TEST_PYPI_API_TOKEN`",
        "`release`",
        "protected `v*.*.*` tag ruleset",
    ):
        assert required in deployment
    for required in (
        "pushes to `main`",
        "pull requests targeting `main`",
        "version tags",
        "Only a version tag can satisfy the deployment job's condition.",
    ):
        assert required in prepare_text

    authority = "explicit tag-push and package-publication authority"
    tag_push = f"git push origin refs/tags/{expected_tag}"
    assert authority in tag_text
    assert tag_text.index(authority) < tag_text.index(tag_push)
    assert (
        f"Pushing the annotated `{expected_tag}` tag starts `python-package.yml`."
        in tag_text
    )
    assert tag_text.count("Never overwrite or repoint a published release tag.") == 1

    for required in (
        '--event push --commit "$RELEASE_COMMIT"',
        '.event == "push"',
        f'.headBranch == "{expected_tag}"',
        ".headSha == env.RELEASE_COMMIT",
        'gh run watch "$RELEASE_RUN_ID" --repo nutrik/vercor --exit-status',
        "--json headSha --jq .headSha)",
        "--json event --jq .event)",
        "--json headBranch --jq .headBranch)",
        "--json conclusion --jq .conclusion)",
        "If PyPI publication succeeds but GitHub Release creation fails",
        "vercor-distributions",
        "vercor-release-manifest",
        "authoritative manifest",
        "exact run",
        "rerunning the ordinary deployment must fail",
        "separately authorized recovery",
    ):
        assert required in publish_text

    for required in (
        "tagName",
        "name",
        "isDraft",
        "isPrerelease",
        f'"{expected_tag}"',
        f'"{expected_title}"',
        "tools/validate_release_state.py assets",
        EXPECTED_WHEEL_NAME,
        EXPECTED_SDIST_NAME,
    ):
        assert required in verify


@pytest.mark.fast_always
def test_ci_quality_job_installs_canonical_docs_environment() -> None:
    """Install the dependencies required by the documentation tests in CI."""

    workflow = yaml.safe_load(
        (PROJECT_ROOT / ".github/workflows/python-package.yml").read_text(
            encoding="utf-8"
        )
    )
    quality_steps = workflow["jobs"]["quality"]["steps"]
    install_index, install = next(
        (index, step)
        for index, step in enumerate(quality_steps)
        if step.get("name") == "Install quality and optional-model dependencies"
    )
    test_index = next(
        index
        for index, step in enumerate(quality_steps)
        if step.get("name") == "Run full test suite"
    )

    assert install_index < test_index
    assert "python -m pip install -r docs/requirements.txt" in install["run"]


@pytest.mark.fast_always
def test_ci_quality_job_enforces_static_full_and_coverage_gates() -> None:
    workflow = yaml.safe_load(
        (PROJECT_ROOT / ".github/workflows/python-package.yml").read_text(
            encoding="utf-8"
        )
    )
    quality = workflow["jobs"]["quality"]
    assert quality["needs"] == "build-artifacts"
    download = next(
        step
        for step in quality["steps"]
        if step.get("uses") == DOWNLOAD_ARTIFACT_ACTION
    )
    assert download["with"] == {
        "name": "vercor-distributions",
        "path": "dist/",
    }
    assert quality["env"]["VERCOR_ARTIFACT_DIR"] == ("${{ github.workspace }}/dist")
    checkout = next(
        step for step in quality["steps"] if step.get("uses") == CHECKOUT_ACTION
    )
    setup = next(
        step for step in quality["steps"] if step.get("uses") == SETUP_PYTHON_ACTION
    )
    commands = "\n".join(
        step.get("run", "") for step in quality["steps"] if isinstance(step, dict)
    )

    assert checkout.get("with", {}).get("fetch-depth") == 0
    assert setup["with"]["python-version"] == "3.12"
    assert 'pip install ".[dev,jcm,veros]"' in commands
    assert "black --check vercor examples tests" in commands
    assert "flake8 ." in commands
    assert "--exit-zero" not in commands
    assert "mypy vercor examples tests" in commands
    assert "compileall" in commands
    assert "pytest tests/ -q --tb=short" in commands
    assert "--cov=vercor" in commands
    assert "--cov-branch" in commands
    assert "--cov-fail-under=90" in commands


def test_distribution_helper_reuses_explicit_artifact_directory_without_building(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert "plugin_wheel" not in BuiltDistributions.__dataclass_fields__
    artifact_dir = tmp_path / "downloaded-dist"
    artifact_dir.mkdir()
    wheel = artifact_dir / EXPECTED_WHEEL_NAME
    sdist = artifact_dir / EXPECTED_SDIST_NAME
    wheel.touch()
    sdist.touch()

    def unexpected_build(*args: object, **kwargs: object) -> object:
        _ = args, kwargs
        pytest.fail("downloaded artifacts must bypass local build tooling")

    monkeypatch.setattr(subprocess, "run", unexpected_build)

    distributions = build_distributions(
        PROJECT_ROOT,
        tmp_path / "unused-build-output",
        artifact_dir=artifact_dir,
    )

    assert distributions.wheel == wheel
    assert distributions.sdist == sdist
    assert distributions.build_pythonpath == ""
    assert set(artifact_dir.iterdir()) == {wheel, sdist}


def test_distribution_helper_rejects_extra_artifact(
    tmp_path: Path,
) -> None:
    artifact_dir = tmp_path / "stale-dist"
    artifact_dir.mkdir()
    (artifact_dir / EXPECTED_WHEEL_NAME).touch()
    (artifact_dir / EXPECTED_SDIST_NAME).touch()
    (artifact_dir / "vercor-0.4.0a0-py3-none-any.whl").touch()

    with pytest.raises(ValueError, match="exactly"):
        build_distributions(
            PROJECT_ROOT,
            tmp_path / "unused-build-output",
            artifact_dir=artifact_dir,
        )


@pytest.mark.parametrize(
    ("wheel_name", "sdist_name"),
    (
        (
            "vercor-0.4.0a0-py3-none-any.whl",
            EXPECTED_SDIST_NAME,
        ),
        (
            EXPECTED_WHEEL_NAME,
            "vercor-0.4.0a0.tar.gz",
        ),
    ),
)
def test_distribution_helper_rejects_wrong_artifact_version(
    tmp_path: Path,
    wheel_name: str,
    sdist_name: str,
) -> None:
    assert "plugin_wheel" not in BuiltDistributions.__dataclass_fields__
    artifact_dir = tmp_path / "wrong-dist"
    artifact_dir.mkdir()
    (artifact_dir / wheel_name).touch()
    (artifact_dir / sdist_name).touch()

    with pytest.raises(ValueError, match=f"VerCOR {EXPECTED_VERSION}"):
        build_distributions(
            PROJECT_ROOT,
            tmp_path / "unused-build-output",
            artifact_dir=artifact_dir,
        )


def test_built_distributions_run_external_extension_fixture_outside_checkout(
    built_distributions: BuiltDistributions,
    external_extension_fixture_wheel: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    distributions = built_distributions

    assert distributions.wheel.name == EXPECTED_WHEEL_NAME
    assert distributions.sdist.name == EXPECTED_SDIST_NAME
    assert (
        external_extension_fixture_wheel.name == EXPECTED_EXTENSION_FIXTURE_WHEEL_NAME
    )
    assert external_extension_fixture_wheel.is_file()
    with zipfile.ZipFile(distributions.wheel) as wheel:
        wheel_names = set(wheel.namelist())
        metadata_name = next(
            name for name in wheel_names if name.endswith(".dist-info/METADATA")
        )
        metadata = wheel.read(metadata_name).decode("utf-8")
    assert "vercor/py.typed" in wheel_names
    assert _forbidden_archive_members(wheel_names) == ()
    assert f"Version: {EXPECTED_VERSION}" in metadata
    pytest_requirements = [
        line
        for line in metadata.splitlines()
        if line.lower().startswith("requires-dist: pytest")
    ]
    assert pytest_requirements
    assert all("extra ==" in line for line in pytest_requirements)
    assert "Provides-Extra: test" in metadata
    assert "Provides-Extra: dev" in metadata

    with zipfile.ZipFile(external_extension_fixture_wheel) as plugin_archive:
        plugin_metadata_name = next(
            name
            for name in plugin_archive.namelist()
            if name.endswith(".dist-info/METADATA")
        )
        plugin_metadata = plugin_archive.read(plugin_metadata_name).decode("utf-8")
    assert "Version: 0.1.0" in plugin_metadata
    assert "Requires-Dist: vercor>=0.4.0,<0.5" in plugin_metadata

    with tarfile.open(distributions.sdist, "r:gz") as sdist:
        sdist_names = set(sdist.getnames())
    assert f"vercor-{EXPECTED_VERSION}/vercor/py.typed" in sdist_names
    assert _forbidden_archive_members(sdist_names) == ()

    target = tmp_path / "installed-target"
    install_local_target(
        wheel=distributions.wheel,
        extension_fixture_wheel=external_extension_fixture_wheel,
        target=target,
    )
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(target)
    probe_source = f"""
import importlib
import importlib.metadata
import inspect
import json
import pathlib
import re
import typing
import vercor

external_typing_alias_replacements = {EXTERNAL_TYPING_ALIAS_REPLACEMENTS!r}
external_typing_alias_pattern = re.compile(
    r"(?<![\\w.])(?:"
    + "|".join(
        re.escape(dependency_rendering)
        for dependency_rendering, _ in external_typing_alias_replacements
    )
    + r")(?![\\w.])"
)

owners = {{}}
for module_name in {tuple(EXPECTED_INSTALLED_OWNER_MANIFESTS)!r}:
    module = importlib.import_module(module_name)
    owners[module_name] = {{
        "all": list(module.__all__),
        "file": module.__file__,
    }}

removed = {{}}
for module_name in {REMOVED_PRIMARY_MODULES!r}:
    try:
        importlib.import_module(module_name)
    except ModuleNotFoundError as error:
        removed[module_name] = error.name
    else:
        removed[module_name] = None

def resolve(qualified_name):
    parts = qualified_name.split(".")
    for stop in range(len(parts), 0, -1):
        try:
            value = importlib.import_module(".".join(parts[:stop]))
        except ModuleNotFoundError:
            continue
        for attribute in parts[stop:]:
            value = getattr(value, attribute)
        return value
    raise AssertionError(qualified_name)


def normalized_signature(value):
    hint_target = value.__init__ if inspect.isclass(value) else value
    hints = typing.get_type_hints(hint_target)
    try:
        signature = inspect.signature(value)
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
    rendered = re.sub(
        r"<function ([^ >]+) at 0x[0-9a-fA-F]+>",
        r"<function \\1>",
        str(signature),
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
    replacements = dict(external_typing_alias_replacements)
    return external_typing_alias_pattern.sub(
        lambda match: replacements[match.group(0)],
        normalized,
    )


callable_exports = []
method_names = []
for module_name, exports in {EXPECTED_INSTALLED_OWNER_MANIFESTS!r}.items():
    if module_name == "vercor":
        continue
    module = importlib.import_module(module_name)
    for name in exports:
        value = getattr(module, name)
        if not (inspect.isclass(value) or inspect.isroutine(value)):
            continue
        owner_name = f"{{module_name}}.{{name}}"
        callable_exports.append(owner_name)
        if inspect.isclass(value) and not issubclass(value, BaseException):
            method_names.extend(
                f"{{owner_name}}.{{method_name}}"
                for method_name, method in inspect.getmembers(value)
                if not method_name.startswith("_")
                and inspect.isroutine(method)
                and getattr(method, "__module__", None) is not None
            )
method_names.append("vercor.regridding.RegridderFactory.__call__")

signatures = {{section: {{}} for section in ("exports", "methods")}}
for section, expected in {EXPECTED_INSTALLED_SIGNATURES!r}.items():
    for qualified_name in expected:
        signatures[section][qualified_name] = normalized_signature(
            resolve(qualified_name)
        )

print(json.dumps({{
    "file": vercor.__file__,
    "version": importlib.metadata.version("vercor"),
    "typed": str(pathlib.Path(vercor.__file__).with_name("py.typed")),
    "root": list(vercor.__all__),
    "owners": owners,
    "removed": removed,
    "callable_exports": callable_exports,
    "method_names": method_names,
    "signatures": signatures,
}}))
"""
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            probe_source,
        ],
        cwd=tmp_path,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    installed = json.loads(probe.stdout)
    assert Path(installed["file"]).is_relative_to(target)
    assert installed["version"] == EXPECTED_VERSION
    assert Path(installed["typed"]).is_file()
    assert installed["root"] == list(EXPECTED_INSTALLED_ROOT)
    for module_name, expected_manifest in EXPECTED_INSTALLED_OWNER_MANIFESTS.items():
        owner = installed["owners"][module_name]
        assert owner["all"] == list(expected_manifest)
        assert Path(owner["file"]).is_relative_to(target)
    assert installed["removed"] == {
        module_name: module_name for module_name in REMOVED_PRIMARY_MODULES
    }
    assert set(installed["callable_exports"]) == set(
        EXPECTED_INSTALLED_SIGNATURES["exports"]
    )
    assert set(installed["method_names"]) == set(
        EXPECTED_INSTALLED_SIGNATURES["methods"]
    )
    assert installed["signatures"] == EXPECTED_INSTALLED_SIGNATURES

    monkeypatch.setenv("VERCOR_TEST_PACKAGE_ROOT", str(target))
    setup_probe = _run_setup_probe("import vercor")
    setup_probe_path = setup_probe["vercor_file"]
    assert isinstance(setup_probe_path, str)
    assert Path(setup_probe_path).is_relative_to(target)

    smoke_output = tmp_path / "external-extension-fixture-output"
    smoke = subprocess.run(
        [
            sys.executable,
            "-m",
            "external_extension_test_fixture.smoke",
            "--output-dir",
            str(smoke_output),
        ],
        cwd=tmp_path,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    evidence = json.loads(smoke.stdout.splitlines()[-1])
    assert evidence["temperature"] == 12.0
    assert evidence["host_value"] == 17.0
    assert evidence["exchange_forcing"] == 2.0
    assert evidence["state_replacement"] is True
    assert evidence["config"] == {
        "forcing": 2.0,
        "initial_temperature": 3.0,
        "steps": 3,
    }
    assert evidence["config_frozen"] is True
    assert evidence["factory"] == ["FORCING", "JAX", "HOST"]
    assert evidence["lifecycle"] == ["user-setup", "hook-setup"]
    assert evidence["period_files"] == [
        "jax.averages.2000-01-01T000000.000000.step00000000.schema0000.nc",
        "jax.averages.2000-01-01T000100.000000.step00000001.schema0000.nc",
        "jax.averages.2000-01-01T000200.000000.step00000002.schema0000.nc",
    ]
    assert evidence["regridder_calls"] == ["plugin-forcing"]
    assert evidence["topology"] == ["build:plugin-forcing"]
    assert evidence["topology_patch_routes"] == ["plugin-forcing"]
    assert evidence["workflow"] == ["build"]
    assert evidence["snapshot"] == {"component": "JAX", "temperature": 12.0}

    mypy_environment = environment.copy()
    mypy_environment["MYPYPATH"] = str(target)
    external_use_site = tmp_path / "external_extension_fixture_use_site.py"
    shutil.copyfile(EXTERNAL_EXTENSION_FIXTURE_ROOT / "use_site.py", external_use_site)
    mypy = subprocess.run(
        [
            str(Path(sys.executable).with_name("mypy")),
            "--strict",
            "--verbose",
            str(target / "external_extension_test_fixture"),
            str(external_use_site),
        ],
        cwd=tmp_path,
        env=mypy_environment,
        check=True,
        capture_output=True,
        text=True,
    )
    mypy_evidence = mypy.stdout + mypy.stderr
    assert str(PROJECT_ROOT) not in mypy_evidence
    assert str(target) in mypy_evidence


@pytest.mark.fast_always
def test_installed_default_slab_factory_runs_v0_4_component(
    built_distributions: BuiltDistributions,
    tmp_path: Path,
) -> None:
    """Run the dependency-free slab default strictly from the installed wheel."""

    target = tmp_path / "installed-slab-target"
    install_local_target(
        wheel=built_distributions.wheel,
        target=target,
    )
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(target)
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import json
from datetime import datetime

from vercor import Clock, Coupler, RectilinearGrid
from vercor.components import Component
from vercor.setups import make_slab_ocean

grid = RectilinearGrid.uniform(
    "installed-slab",
    nlon=2,
    nlat=2,
    longitude=(0.0, 360.0),
    latitude=(-90.0, 90.0),
)
component = make_slab_ocean(grid)
coupler = Coupler(
    Clock(start=datetime(2000, 1, 1), dt_seconds=60.0, steps=1),
    components=(component,),
    run_order=(component.name,),
)
state = coupler.run()
print(json.dumps({
    "component": component.name,
    "is_component": isinstance(component, Component),
    "shape": list(state.component(component.name).field(
        "sea_surface_temperature"
    ).shape),
}))
""",
        ],
        cwd=tmp_path,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )

    evidence = json.loads(probe.stdout.splitlines()[-1])
    assert evidence == {
        "component": "OCN",
        "is_component": True,
        "shape": [2, 2],
    }


@pytest.mark.fast_always
def test_dependency_free_slab_contract_does_not_request_extension_fixture() -> None:
    slab_parameters = inspect.signature(
        test_installed_default_slab_factory_runs_v0_4_component
    ).parameters
    install_parameters = inspect.signature(install_local_target).parameters

    assert "external_extension_fixture_wheel" not in slab_parameters
    assert install_parameters["extension_fixture_wheel"].default is None


def test_built_sdist_installs_and_imports_outside_checkout(
    built_distributions: BuiltDistributions,
    external_extension_fixture_wheel: Path,
    tmp_path: Path,
) -> None:
    """Install the sdist and compose the external fixture outside the checkout."""

    assert external_extension_fixture_wheel.is_file()
    target = tmp_path / "installed-sdist-target"
    install_environment = os.environ.copy()
    build_pythonpath = distribution_support._cached_build_pythonpath()
    if build_pythonpath:
        install_environment["PYTHONPATH"] = build_pythonpath
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--no-deps",
            "--no-build-isolation",
            "--target",
            str(target),
            str(built_distributions.sdist),
        ],
        cwd=tmp_path,
        env=install_environment,
        check=True,
        capture_output=True,
        text=True,
    )
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(target)
    probe = subprocess.run(
        [
            sys.executable,
            "-P",
            "-c",
            """
import importlib.metadata
import json
from pathlib import Path
import vercor

print(json.dumps({
    "file": vercor.__file__,
    "root": list(vercor.__all__),
    "version": importlib.metadata.version("vercor"),
    "typed": Path(vercor.__file__).with_name("py.typed").is_file(),
}))
""",
        ],
        cwd=tmp_path,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    evidence = json.loads(probe.stdout)
    assert Path(evidence["file"]).is_relative_to(target)
    assert evidence["root"] == list(EXPECTED_INSTALLED_ROOT)
    assert evidence["version"] == EXPECTED_VERSION
    assert evidence["typed"] is True

    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--no-deps",
            "--only-binary=:all:",
            "--target",
            str(target),
            str(external_extension_fixture_wheel),
        ],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    smoke = subprocess.run(
        [
            sys.executable,
            "-m",
            "external_extension_test_fixture.smoke",
            "--output-dir",
            str(tmp_path / "sdist-external-extension-fixture-output"),
        ],
        cwd=tmp_path,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    plugin_evidence = json.loads(smoke.stdout.splitlines()[-1])
    assert plugin_evidence["temperature"] == 12.0
    assert plugin_evidence["host_value"] == 17.0


def test_supplied_wheels_install_and_run_without_build_environment(
    built_distributions: BuiltDistributions,
    external_extension_fixture_wheel: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert external_extension_fixture_wheel.is_file()
    artifact_dir = tmp_path / "supplied-artifacts"
    artifact_dir.mkdir()
    for artifact in (
        built_distributions.wheel,
        built_distributions.sdist,
    ):
        shutil.copyfile(artifact, artifact_dir / artifact.name)

    def unavailable_build_environment() -> str:
        pytest.fail("supplied wheels must not inspect build/flit_core/Conda fallback")

    monkeypatch.setattr(
        distribution_support,
        "_cached_build_pythonpath",
        unavailable_build_environment,
    )
    monkeypatch.setenv("VERCOR_ARTIFACT_DIR", str(artifact_dir))

    supplied = build_distributions(PROJECT_ROOT, tmp_path / "must-not-build")
    target = tmp_path / "clean-installed-target"
    install_local_target(
        wheel=supplied.wheel,
        extension_fixture_wheel=external_extension_fixture_wheel,
        target=target,
    )

    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(target)
    smoke = subprocess.run(
        [
            sys.executable,
            "-m",
            "external_extension_test_fixture.smoke",
            "--output-dir",
            str(tmp_path / "clean-external-extension-fixture-output"),
        ],
        cwd=tmp_path,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )

    evidence = json.loads(smoke.stdout.splitlines()[-1])
    assert evidence["temperature"] == 12.0
    assert evidence["host_value"] == 17.0
