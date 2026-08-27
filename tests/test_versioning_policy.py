"""Repository-wide contracts for VerCOR's supervised pre-1.0 versioning."""

from __future__ import annotations

from pathlib import Path
import re
import subprocess

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
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
_READTHEDOCS_CONFIG_PATH = Path(".readthedocs.yaml")
_READTHEDOCS_CONFIG_REFERENCE = re.compile(
    r"https://docs\.readthedocs\.io/en/stable/config-file/" + "v" + r"2\.html"
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
_VERCOR_API_IDENTIFIER_PREFIX = re.compile(
    r"\bvercor(?:\.[A-Za-z_][A-Za-z0-9_]*)*\.$",
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
    r"^[ \t`\"'\])}:_-]*(?:only[ \t_-]+)?(?:releases?|APIs?|history|migrations?|artifacts?|"
    r"manifests?|plugins?|fixtures?|lines?)\b",
    flags=re.IGNORECASE,
)
_VERSION_ASSIGNMENT = re.compile(
    r"(?:^|[\"'])\s*(?:__version__|version)[\"']?\s*[:=]",
    flags=re.IGNORECASE,
)
_PRE_V0_4_TOKEN = re.compile(
    r"(?:"
    r"(?<![\d.])(?:[vV])?0\.(?:[0-3])"
    r"(?:\.\d+(?:[A-Za-z][0-9A-Za-z.-]*)?)?(?![\d.])"
    r"|"
    r"(?<![A-Za-z0-9_])(?:[vV])?0_(?:[0-3])(?:_\d+)?(?![A-Za-z0-9_])"
    r")"
)
_PRE_V0_4_PATH = re.compile(
    r"(?:migration|vercor|compat|api|release)[^/]*0\.(?:[0-3])",
    flags=re.IGNORECASE,
)
_EXTERNAL_ARTIFACT_STEMS = (
    "external_extension_test_fixture-",
    "vercor_public_plugin-",
)


def _legacy_version(
    *,
    minor: int,
    major: int = 0,
    patch: int | None = 0,
    prefix: str = "",
    suffix: str = "",
) -> str:
    """Construct an unsupported VerCOR label without source-hiding fragments."""

    parts = (major, minor) if patch is None else (major, minor, patch)
    return prefix + ".".join(str(part) for part in parts) + suffix


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
        or _VERCOR_API_IDENTIFIER_PREFIX.search(prefix)
        or _REPOSITORY_VERSION_PREFIX.search(prefix)
        or _REPOSITORY_VERSION_SUFFIX.search(suffix)
    ):
        return "vercor"
    return None


def _forbidden_pre_v0_4_labels(relative_path: Path, line: str) -> tuple[str, ...]:
    """Return unsupported labels when their path or context belongs to VerCOR."""

    metadata_context = bool(
        _VERSION_ASSIGNMENT.search(line)
        and (
            relative_path == Path("pyproject.toml")
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
    for match in _PRE_V0_4_TOKEN.finditer(line):
        owner = _version_context_owner(line, match.start(), match.end())
        external_artifact_context = any(
            line[: match.start()].endswith(stem) for stem in _EXTERNAL_ARTIFACT_STEMS
        )
        if owner == "external" or external_artifact_context:
            continue
        if owner == "vercor" or metadata_context or changelog_context:
            labels.append(match.group())
    return tuple(labels)


def _forbidden_pre_v0_4_path(relative_path: Path) -> bool:
    """Return whether a path itself identifies an unsupported release series."""

    return bool(_PRE_V0_4_PATH.search(relative_path.as_posix()))


def _forbidden_release_shorthand_labels(line: str) -> tuple[str, ...]:
    """Return shorthand spans whose adjacent context belongs to VerCOR."""

    return tuple(
        match.group()
        for match in _RELEASE_SHORTHAND_TOKEN.finditer(line)
        if _version_context_owner(line, match.start(), match.end()) == "vercor"
    )


@pytest.mark.fast_always
@pytest.mark.parametrize("minor", range(4))
@pytest.mark.parametrize("prefix", ("", "v", "V"))
def test_pre_v0_4_matcher_rejects_vercor_owned_labels(
    minor: int,
    prefix: str,
) -> None:
    label = _legacy_version(minor=minor, patch=2, prefix=prefix)
    assert _forbidden_pre_v0_4_labels(
        Path("docs/history.md"),
        f"Historical VerCOR release {label}",
    ) == (label,)


@pytest.mark.fast_always
@pytest.mark.parametrize("minor", range(4))
def test_pre_v0_4_matcher_rejects_encoded_vercor_api_identifiers(
    minor: int,
) -> None:
    label = f"v0_{minor}"
    assert _forbidden_pre_v0_4_labels(
        Path("docs/history.md"),
        f"vercor.compat.{label}",
    ) == (label,)


@pytest.mark.fast_always
@pytest.mark.parametrize("minor", range(4))
def test_pre_v0_4_matcher_rejects_only_api_suffix(minor: int) -> None:
    label = _legacy_version(minor=minor)
    assert _forbidden_pre_v0_4_labels(
        Path("docs/history.md"),
        f"the {label}-only API",
    ) == (label,)


@pytest.mark.fast_always
@pytest.mark.parametrize(
    ("relative_path", "line"),
    (
        (
            Path("docs/external-dependencies.md"),
            "external dependency version " + _legacy_version(minor=2, patch=1),
        ),
        (
            Path("dist/artifacts.md"),
            "external_extension_test_fixture-"
            + _legacy_version(minor=1)
            + "-py3-none-any.whl",
        ),
        (Path("docs/numerics.md"), "one-quarter numerical weights are 0.25"),
        (Path("docs/development.md"), "Python 3.12 and Python 3.13"),
        (
            Path(".readthedocs.yaml"),
            "https://docs.readthedocs.io/en/stable/config-file/" + "v" + "2.html",
        ),
    ),
)
def test_pre_v0_4_matcher_allows_external_and_numeric_contexts(
    relative_path: Path,
    line: str,
) -> None:
    assert not _forbidden_pre_v0_4_labels(relative_path, line)


@pytest.mark.fast_always
def test_pre_v0_4_matcher_rejects_vercor_label_beside_external_artifact() -> None:
    """Keep an external artifact exemption scoped to its own version token."""

    vercor_label = _legacy_version(minor=2, patch=1)
    external_label = _legacy_version(minor=1)
    line = (
        f"Historical VerCOR release {vercor_label}; "
        "external_extension_test_fixture-"
        f"{external_label}-py3-none-any.whl"
    )

    assert _forbidden_pre_v0_4_labels(Path("docs/history.md"), line) == (vercor_label,)


@pytest.mark.fast_always
@pytest.mark.parametrize("minor", range(4))
def test_pre_v0_4_matcher_rejects_legacy_migration_paths(minor: int) -> None:
    assert _forbidden_pre_v0_4_path(Path(f"docs/migration-0.{minor}-to-0.4.md"))


def _forbidden_api_tokens(relative_path: Path, line: str) -> tuple[str, ...]:
    """Return stale API tokens while preserving the interpolator's numeric vector."""

    matches = tuple(FORBIDDEN_API_TOKEN.finditer(line))
    if relative_path == _READTHEDOCS_CONFIG_PATH:
        reference_spans = tuple(
            (match.start(), match.end())
            for match in _READTHEDOCS_CONFIG_REFERENCE.finditer(line)
        )
        matches = tuple(
            match
            for match in matches
            if not any(
                start <= match.start() and match.end() <= end
                for start, end in reference_spans
            )
        )
    tokens = tuple(match.group() for match in matches)
    if relative_path != _NUMERICAL_VECTOR_PATH:
        return tokens
    if not any(pattern.search(line) for pattern in _NUMERICAL_VECTOR_LINES):
        return tokens
    return tuple(token for token in tokens if token.lower() != "v" + "3")


@pytest.mark.fast_always
@pytest.mark.parametrize(
    "line",
    (
        "VerCOR " + _legacy_version(major=1, minor=0, patch=None) + " release",
        "VerCOR " + _legacy_version(major=2, minor=0, patch=None) + " API",
        "frozen " + _legacy_version(major=3, minor=0, patch=None) + " plugin",
        "current-" + _legacy_version(major=3, minor=1, patch=None),
        "Compatibility within the " + "4" + ".x line",
        "2" + ".x migration",
        "vercor-release-" + _legacy_version(major=3, minor=1, patch=None) + "-final",
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
        "dependency release " + _legacy_version(minor=2, patch=1),
        "v" + "3 = eastward_vector_component",
    ),
)
def test_release_shorthand_matcher_allows_external_and_numeric_labels(
    line: str,
) -> None:
    assert not _forbidden_release_shorthand_labels(line)


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
        "numpy==" + _legacy_version(major=2, minor=0),
        "independent plugin version " + _legacy_version(major=3, minor=1, patch=1),
        "external schema " + _legacy_version(major=3, minor=0),
        "VerCOR depends on numpy==" + _legacy_version(major=2, minor=0),
        "VerCOR supports independent plugin version "
        + _legacy_version(major=3, minor=1, patch=1),
        "VerCOR documents external schema " + _legacy_version(major=3, minor=0),
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
        "external artifact version " + _legacy_version(major=3, minor=0),
        "external release " + _legacy_version(major=3, minor=1, patch=1),
        'independent release: "' + _legacy_version(major=3, minor=1, patch=1) + '"',
        "independent artifact version " + _legacy_version(major=2, minor=0),
        "external " + _legacy_version(major=3, minor=0, patch=None) + " artifact",
        "external " + _legacy_version(major=2, minor=0, patch=None) + " API",
        "independent "
        + _legacy_version(major=3, minor=1, patch=None)
        + " plugin fixture",
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
        "VerCOR version `" + _legacy_version(major=4, minor=0, suffix="a1") + "`",
        'VERCOR_VERSION: "' + _legacy_version(major=4, minor=0, suffix="a1") + '"',
        "export VERCOR_VERSION='" + _legacy_version(major=3, minor=1, patch=1) + "'",
        "VerCOR version " + _legacy_version(major=4, minor=0, patch=None),
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
        label = _legacy_version(
            major=3,
            minor=0 if concept == "plugin fixture" else 1,
            patch=None,
        )
        fields = (owner, qualifier, label, concept)
    else:
        label = {
            "release": _legacy_version(major=3, minor=1, patch=1),
            "API": _legacy_version(major=3, minor=0),
            "artifact": _legacy_version(major=3, minor=0),
            "plugin fixture": _legacy_version(major=2, minor=0),
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
            line="external current VerCOR release "
            + _legacy_version(major=3, minor=1, patch=1),
        )


@pytest.mark.fast_always
@pytest.mark.parametrize(
    ("relative_path", "line"),
    (
        (
            Path("pyproject.toml"),
            'version = "' + _legacy_version(major=4, minor=0, suffix="a1") + '"',
        ),
        (
            Path("docs/releasing.md"),
            "VerCOR release " + _legacy_version(major=3, minor=1, patch=1),
        ),
        (
            Path("docs/api-history.md"),
            "frozen API history " + _legacy_version(major=3, minor=0),
        ),
        (
            Path("tests/fixture-notes.md"),
            "historical plugin fixture " + _legacy_version(major=2, minor=0),
        ),
        (
            Path(".github/workflows/python-package.yml"),
            "artifact: vercor-"
            + _legacy_version(major=1, minor=0)
            + "-py3-none-any.whl",
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
def test_integrated_scanner_allows_official_readthedocs_config_reference(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _run_integrated_scanner_for_line(
        monkeypatch,
        tmp_path,
        relative_path=Path(".readthedocs.yaml"),
        line=(
            "# See https://docs.readthedocs.io/en/stable/config-file/"
            + "v"
            + "2.html for details"
        ),
    )


@pytest.mark.fast_always
def test_readthedocs_reference_does_not_hide_a_stale_api_token() -> None:
    line = (
        "# See https://docs.readthedocs.io/en/stable/config-file/"
        + "v"
        + "2.html; stale "
        + "v"
        + "2 API"
    )

    assert _forbidden_api_tokens(Path(".readthedocs.yaml"), line) == ("v" + "2",)


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
        if _forbidden_pre_v0_4_path(relative_path):
            violations.append(f"{rendered_path}: forbidden pre-v0.4 path")

        text = (PROJECT_ROOT / relative_path).read_text(encoding="utf-8")
        for line_number, line in enumerate(text.splitlines(), start=1):
            labels = _forbidden_exact_release_labels(relative_path, line)
            pre_v0_4_labels = _forbidden_pre_v0_4_labels(relative_path, line)
            api_tokens = _forbidden_api_tokens(relative_path, line)
            major_names = tuple(FORBIDDEN_VERCOR_MAJOR.findall(line))
            shorthand_labels = _forbidden_release_shorthand_labels(line)
            if (
                labels
                or pre_v0_4_labels
                or api_tokens
                or major_names
                or shorthand_labels
            ):
                violations.append(
                    f"{rendered_path}:{line_number}: "
                    f"labels={labels}, pre_v0_4_labels={pre_v0_4_labels}, "
                    f"api_tokens={api_tokens}, "
                    f"major_names={major_names}, shorthand={shorthand_labels}"
                )

    assert not violations, "ERROR forbidden VerCOR release labels:\n" + "\n".join(
        violations
    )
