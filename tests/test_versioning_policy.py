"""Repository-wide contracts for VerCOR's supervised pre-1.0 versioning."""

from __future__ import annotations

from pathlib import Path
import re
import subprocess
import tomllib

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CURRENT_VERSION = "0.4.2"
TEXT_SUFFIXES = {".json", ".md", ".py", ".toml", ".yaml", ".yml"}
FORBIDDEN_RELEASE_LABELS = (
    ".".join(("1", "0", "0")),
    ".".join(("2", "0", "0")),
    ".".join(("3", "0", "0")),
    ".".join(("3", "1", "0")),
    ".".join(("3", "1", "1")),
    ".".join(("4", "0", "0")) + "a1",
)
FORBIDDEN_API_TOKEN = re.compile(
    r"(?<![@A-Za-z0-9])[vV][" + "1234" + r"](?![A-Za-z0-9])"
)
_NUMERICAL_VECTOR_PATH = Path("vercor/_interpolators/bilinear_rectilinear.py")
_NUMERICAL_VECTOR_LINES = (
    re.compile(r"^\s*" + "v" + r"3\s*="),
    re.compile(r"^\s*v(?:00|10|01|11)\s*=\s*" + "v" + r"3\["),
)
FORBIDDEN_VERCOR_MAJOR = re.compile(r"\bVerCOR [" + "1234" + r"](?:\b|\.)")
_RELEASE_SHORTHAND = r"(?<![\d.])(?:[12]\.0|3\." + r"[01]|4\.0|[1234]\.x)(?![\d.])"
_RELEASE_SHORTHAND_TOKEN = re.compile(_RELEASE_SHORTHAND, flags=re.IGNORECASE)
FORBIDDEN_PATH_FRAGMENTS = (
    "migration-" + "3-to-" + "4",
    "vercor-" + "4-api",
    "test_v" + "4_",
    "test_v" + "2_",
    "public_plugin_" + "3_0",
    "vercor-3." + "1.1",
    "vercor-4." + "0.0a1",
)
_EXACT_RELEASE_PATTERNS = tuple(
    (
        label,
        re.compile(rf"(?<![\d.]){re.escape(label)}(?![\dA-Za-z.])"),
    )
    for label in FORBIDDEN_RELEASE_LABELS
)
_VERSION_QUALIFIER = (
    r"(?:current|previous|historical|frozen|stable|first|major|later|native)"
)
_VERCOR_VERSION_PREFIX = re.compile(
    r"\bvercor(?:['’]s)?"
    rf"(?:[ \t_-]+(?:{_VERSION_QUALIFIER}|version|releases?|APIs?|history|"
    r"migrations?|artifacts?|manifests?|"
    r"plugins?|fixtures?|line|candidate))*[ \t:`\"'=[_-]*$",
    flags=re.IGNORECASE,
)
_EXTERNAL_VERSION_PREFIX = re.compile(
    r"\b(?:external|independent)"
    rf"(?:[ \t_-]+{_VERSION_QUALIFIER})*"
    r"(?:[ \t_-]+(?:artifacts?|releases?|schemas?|plugins?|APIs?|versions?|"
    r"fixtures?|lines?|dependencies?))*[ \t:`\"'=[_-]*$",
    flags=re.IGNORECASE,
)
_REPOSITORY_VERSION_PREFIX = re.compile(
    r"(?:"
    rf"{_VERSION_QUALIFIER}[ \t_-]+"
    r"|(?:(?:current|previous|historical|frozen|stable|first|major)[ \t-]+)*"
    r"(?:releases?|APIs?|history|migrations?|artifacts?|manifests?|"
    r"plugin[ \t-]+fixtures?)"
    r"(?:[ \t-]+(?:release|version|label|line|history|candidate))*"
    r"[ \t:`\"'=[_-]*"
    r")$",
    flags=re.IGNORECASE,
)
_REPOSITORY_VERSION_SUFFIX = re.compile(
    r"^[ \t`\"'\])}:_-]*(?:releases?|APIs?|history|migrations?|artifacts?|"
    r"manifests?|plugins?|fixtures?|lines?)\b",
    flags=re.IGNORECASE,
)
_VERSION_ASSIGNMENT = re.compile(
    r"(?:^|[\"'])\s*(?:__version__|version)[\"']?\s*[:=]",
    flags=re.IGNORECASE,
)


def _tracked_text_paths() -> tuple[Path, ...]:
    """Return existing tracked or intended repository text paths."""

    result = subprocess.run(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
        check=True,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    paths = (
        Path(name)
        for name in result.stdout.split("\0")
        if name and Path(name).suffix in TEXT_SUFFIXES
    )
    return tuple(path for path in paths if (PROJECT_ROOT / path).is_file())


def _forbidden_exact_release_labels(
    relative_path: Path,
    line: str,
) -> tuple[str, ...]:
    """Return exact old labels only when the line establishes VerCOR ownership."""

    metadata_context = bool(
        _VERSION_ASSIGNMENT.search(line)
        and (
            relative_path == Path("pyproject.toml")
            or relative_path.parts[:1] == ("vercor",)
            or (
                relative_path.parts[:2] == ("tests", "contracts")
                and relative_path.name.startswith("vercor-")
            )
        )
    )
    changelog_context = bool(
        relative_path == Path("CHANGELOG.md") and re.match(r"^\s*(?:##\s+)?\[", line)
    )
    labels: list[str] = []
    for label, pattern in _EXACT_RELEASE_PATTERNS:
        for match in pattern.finditer(line):
            owner = _version_context_owner(line, match.start(), match.end())
            if owner == "external":
                continue
            if owner == "vercor" or metadata_context or changelog_context:
                labels.append(label)
                break
    return tuple(labels)


def _version_context_owner(line: str, start: int, end: int) -> str | None:
    """Classify only the narrow ownership syntax adjacent to a version span."""

    prefix = line[:start]
    suffix = line[end:]
    if _EXTERNAL_VERSION_PREFIX.search(prefix):
        return "external"
    if (
        _VERCOR_VERSION_PREFIX.search(prefix)
        or _REPOSITORY_VERSION_PREFIX.search(prefix)
        or _REPOSITORY_VERSION_SUFFIX.search(suffix)
    ):
        return "vercor"
    return None


def _forbidden_release_shorthand_labels(line: str) -> tuple[str, ...]:
    """Return shorthand spans whose adjacent context belongs to VerCOR."""

    return tuple(
        match.group()
        for match in _RELEASE_SHORTHAND_TOKEN.finditer(line)
        if _version_context_owner(line, match.start(), match.end()) == "vercor"
    )


def _forbidden_api_tokens(relative_path: Path, line: str) -> tuple[str, ...]:
    """Return stale API tokens while preserving the interpolator's numeric vector."""

    tokens = tuple(FORBIDDEN_API_TOKEN.findall(line))
    if relative_path != _NUMERICAL_VECTOR_PATH:
        return tokens
    if not any(pattern.search(line) for pattern in _NUMERICAL_VECTOR_LINES):
        return tokens
    return tuple(token for token in tokens if token.lower() != "v" + "3")


@pytest.mark.fast_always
@pytest.mark.parametrize(
    "line",
    (
        "VerCOR " + ".".join(("1", "0")) + " release",
        "VerCOR " + ".".join(("2", "0")) + " API",
        "frozen " + ".".join(("3", "0")) + " plugin",
        "current-" + ".".join(("3", "1")),
        "Compatibility within the " + "4" + ".x line",
        "2" + ".x migration",
        "vercor-release-" + ".".join(("3", "1")) + "-final",
    ),
)
def test_release_shorthand_matcher_rejects_repository_labels(line: str) -> None:
    assert _forbidden_release_shorthand_labels(line)


@pytest.mark.fast_always
@pytest.mark.parametrize(
    "line",
    (
        "pre-1.0",
        "plugin timeout is 3.0 seconds",
        "external plugin version 3.0",
        "Python 3.12 and Python 3.13",
        "actions/checkout@v4",
        "schema version 1",
        "JCM 1.1.1 and Veros 1.6.2",
        "external_extension_test_fixture-0.1.0-py3-none-any.whl",
        "dependency release 0.2.1",
        "v" + "3 = eastward_vector_component",
    ),
)
def test_release_shorthand_matcher_allows_external_and_numeric_labels(
    line: str,
) -> None:
    assert not _forbidden_release_shorthand_labels(line)


@pytest.mark.fast_always
def test_current_vercor_release_is_the_approved_stable_release() -> None:
    """Require the repository's approved stable release version."""

    project = tomllib.loads(
        (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )["project"]
    assert project["version"] == CURRENT_VERSION


def _run_integrated_scanner_for_line(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    relative_path: Path,
    line: str,
) -> None:
    """Run the repository scanner against one isolated candidate line."""

    candidate = tmp_path / relative_path
    candidate.parent.mkdir(parents=True, exist_ok=True)
    candidate.write_text(line + "\n", encoding="utf-8")
    monkeypatch.setattr("tests.test_versioning_policy.PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        "tests.test_versioning_policy._tracked_text_paths",
        lambda: (relative_path,),
    )
    test_tracked_repository_has_no_forbidden_vercor_release_labels()


@pytest.mark.fast_always
@pytest.mark.parametrize(
    "line",
    (
        "numpy==" + "2" + ".0.0",
        "independent plugin version " + "3" + ".1.1",
        "external schema " + "3" + ".0.0",
        "VerCOR depends on numpy==" + "2" + ".0.0",
        "VerCOR supports independent plugin version " + "3" + ".1.1",
        "VerCOR documents external schema " + "3" + ".0.0",
    ),
)
def test_integrated_scanner_allows_external_exact_version_collisions(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    line: str,
) -> None:
    _run_integrated_scanner_for_line(
        monkeypatch,
        tmp_path,
        relative_path=Path("external-metadata.md"),
        line=line,
    )


@pytest.mark.fast_always
@pytest.mark.parametrize(
    "line",
    (
        "external artifact version " + "3" + ".0.0",
        "external release " + "3" + ".1.1",
        'independent release: "' + "3" + '.1.1"',
        "independent artifact version " + "2" + ".0.0",
        "external " + "3" + ".0 artifact",
        "external " + "2" + ".0 API",
        "independent " + "3" + ".1 plugin fixture",
    ),
)
def test_integrated_scanner_allows_explicit_external_version_contexts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    line: str,
) -> None:
    _run_integrated_scanner_for_line(
        monkeypatch,
        tmp_path,
        relative_path=Path("external-release-metadata.yml"),
        line=line,
    )


@pytest.mark.fast_always
@pytest.mark.parametrize(
    "line",
    (
        "VerCOR version `" + "4" + ".0.0a1`",
        'VERCOR_VERSION: "' + "4" + '.0.0a1"',
        "export VERCOR_VERSION='" + "3" + ".1.1'",
        "VerCOR version " + "4" + ".0",
    ),
)
def test_integrated_scanner_rejects_quoted_and_env_vercor_versions(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    line: str,
) -> None:
    with pytest.raises(AssertionError):
        _run_integrated_scanner_for_line(
            monkeypatch,
            tmp_path,
            relative_path=Path("release-config.yml"),
            line=line,
        )


def _ownership_matrix_line(
    owner: str,
    qualifier: str,
    concept: str,
    *,
    shorthand: bool,
) -> str:
    """Render one exact or shorthand line for the ownership grammar matrix."""

    fields: tuple[str, ...]
    if shorthand:
        label = "3" + (".0" if concept == "plugin fixture" else ".1")
        fields = (owner, qualifier, label, concept)
    else:
        label = {
            "release": "3" + ".1.1",
            "API": "3" + ".0.0",
            "artifact": "3" + ".0.0",
            "plugin fixture": "2" + ".0.0",
        }[concept]
        version_word = "version" if concept == "artifact" else ""
        fields = (owner, qualifier, concept, version_word, label)
    return " ".join(field for field in fields if field)


@pytest.mark.fast_always
@pytest.mark.parametrize(
    ("owner", "qualifier", "concept", "shorthand"),
    tuple(
        (owner, qualifier, concept, shorthand)
        for owner in ("external", "independent", "VerCOR")
        for qualifier in ("", "current", "historical", "frozen")
        for concept in ("release", "API", "artifact", "plugin fixture")
        for shorthand in (False, True)
    ),
)
def test_integrated_scanner_version_ownership_matrix(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    owner: str,
    qualifier: str,
    concept: str,
    shorthand: bool,
) -> None:
    line = _ownership_matrix_line(
        owner,
        qualifier,
        concept,
        shorthand=shorthand,
    )
    if owner == "VerCOR":
        with pytest.raises(AssertionError):
            _run_integrated_scanner_for_line(
                monkeypatch,
                tmp_path,
                relative_path=Path("ownership-matrix.md"),
                line=line,
            )
    else:
        _run_integrated_scanner_for_line(
            monkeypatch,
            tmp_path,
            relative_path=Path("ownership-matrix.md"),
            line=line,
        )


@pytest.mark.fast_always
def test_integrated_scanner_later_vercor_owner_overrides_external_qualifier(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    with pytest.raises(AssertionError):
        _run_integrated_scanner_for_line(
            monkeypatch,
            tmp_path,
            relative_path=Path("ownership-matrix.md"),
            line="external current VerCOR release " + "3" + ".1.1",
        )


@pytest.mark.fast_always
@pytest.mark.parametrize(
    ("relative_path", "line"),
    (
        (Path("pyproject.toml"), 'version = "' + "4" + '.0.0a1"'),
        (
            Path("docs/releasing.md"),
            "VerCOR release " + "3" + ".1.1",
        ),
        (
            Path("docs/api-history.md"),
            "frozen API history " + "3" + ".0.0",
        ),
        (
            Path("tests/fixture-notes.md"),
            "historical plugin fixture " + "2" + ".0.0",
        ),
        (
            Path(".github/workflows/python-package.yml"),
            "artifact: vercor-" + "1" + ".0.0-py3-none-any.whl",
        ),
    ),
)
def test_integrated_scanner_rejects_vercor_owned_exact_versions(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    relative_path: Path,
    line: str,
) -> None:
    with pytest.raises(AssertionError, match="labels="):
        _run_integrated_scanner_for_line(
            monkeypatch,
            tmp_path,
            relative_path=relative_path,
            line=line,
        )


@pytest.mark.fast_always
@pytest.mark.parametrize(
    "line",
    (
        "        " + "v" + "3 = (u_src_array[..., None] * basis)",
        "        v00 = " + "v" + "3[self.j0, self.i0, :]",
    ),
)
def test_integrated_scanner_allows_numerical_vector_interpolator_context(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    line: str,
) -> None:
    _run_integrated_scanner_for_line(
        monkeypatch,
        tmp_path,
        relative_path=Path("vercor/_interpolators/bilinear_rectilinear.py"),
        line=line,
    )


@pytest.mark.fast_always
def test_integrated_scanner_still_rejects_stale_api_in_interpolator(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    with pytest.raises(AssertionError, match="api_tokens=.*" + "v" + "4"):
        _run_integrated_scanner_for_line(
            monkeypatch,
            tmp_path,
            relative_path=Path("vercor/_interpolators/bilinear_rectilinear.py"),
            line="# stale " + "v" + "4 API",
        )


@pytest.mark.fast_always
def test_tracked_repository_has_no_forbidden_vercor_release_labels() -> None:
    violations: list[str] = []
    for relative_path in _tracked_text_paths():
        rendered_path = relative_path.as_posix()
        for fragment in FORBIDDEN_PATH_FRAGMENTS:
            if fragment in rendered_path:
                violations.append(
                    f"{rendered_path}: forbidden path fragment {fragment!r}"
                )

        text = (PROJECT_ROOT / relative_path).read_text(encoding="utf-8")
        for line_number, line in enumerate(text.splitlines(), start=1):
            labels = _forbidden_exact_release_labels(relative_path, line)
            api_tokens = _forbidden_api_tokens(relative_path, line)
            major_names = tuple(FORBIDDEN_VERCOR_MAJOR.findall(line))
            shorthand_labels = _forbidden_release_shorthand_labels(line)
            if labels or api_tokens or major_names or shorthand_labels:
                violations.append(
                    f"{rendered_path}:{line_number}: "
                    f"labels={labels}, api_tokens={api_tokens}, "
                    f"major_names={major_names}, shorthand={shorthand_labels}"
                )

    assert not violations, "ERROR forbidden VerCOR release labels:\n" + "\n".join(
        violations
    )
