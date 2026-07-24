"""Executable contracts for fail-closed release recovery state validation."""

from __future__ import annotations

import hashlib
import json
import shutil
import socket
from pathlib import Path
import subprocess
import sys
from urllib.parse import parse_qs, urlsplit

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VALIDATOR_PATH = PROJECT_ROOT / "tools" / "validate_release_state.py"
WHEEL = "vercor-0.4.0-py3-none-any.whl"
SDIST = "vercor-0.4.0.tar.gz"
UNEXPECTED = "unexpected-0.4.0-py3-none-any.whl"


def _run_validator(*arguments: str) -> subprocess.CompletedProcess[str]:
    """Run the tracked release-state validator as a maintainer would."""

    return subprocess.run(
        [sys.executable, str(VALIDATOR_PATH), *arguments],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def _write_manifest(path: Path) -> None:
    """Write deterministic synthetic release hashes."""

    path.write_text(f"{'a' * 64}  {WHEEL}\n{'b' * 64}  {SDIST}\n", encoding="utf-8")


def _release_asset(name: str, *, digest: str | None = None) -> dict[str, object]:
    """Return one uploaded GitHub asset fixture with a manifest digest."""

    expected_digest = "a" * 64 if name == WHEEL else "b" * 64
    return {
        "id": 10 if name == WHEEL else 11,
        "name": name,
        "state": "uploaded",
        "digest": digest or f"sha256:{expected_digest}",
    }


def _release(
    assets: list[dict[str, object]],
    *,
    draft: bool = True,
    **overrides: object,
) -> dict[str, object]:
    """Return one exact-tag GitHub Release fixture."""

    payload: dict[str, object] = {
        "id": 42,
        "tag_name": "v0.4.0",
        "name": "VerCOR 0.4.0",
        "body": "Release notes.\n",
        "draft": draft,
        "prerelease": False,
        "assets": assets,
    }
    payload.update(overrides)
    return payload


def _run_github_release_validator(
    tmp_path: Path,
    releases: object,
    *,
    allow_states: tuple[str, ...] = ("absent", "draft"),
) -> tuple[subprocess.CompletedProcess[str], Path]:
    """Run the exact-tag GitHub Release state-machine validator."""

    manifest = tmp_path / "SHA256SUMS"
    _write_manifest(manifest)
    notes = tmp_path / "notes.md"
    notes.write_text("Release notes.\n", encoding="utf-8")
    payload = tmp_path / "releases.json"
    payload.write_text(json.dumps(releases), encoding="utf-8")
    state = tmp_path / "state.json"
    completed = _run_validator(
        "github-releases",
        "--json",
        str(payload),
        "--manifest",
        str(manifest),
        "--tag",
        "v0.4.0",
        "--title",
        "VerCOR 0.4.0",
        "--notes-file",
        str(notes),
        "--expect",
        WHEEL,
        SDIST,
        "--allow-state",
        *allow_states,
        "--state-output",
        str(state),
    )
    return completed, state


def _run_github_tag_absent_validator(
    tmp_path: Path,
    releases: object,
) -> subprocess.CompletedProcess[str]:
    """Run the manifest-free exact-tag absence validator."""

    payload = tmp_path / "releases.json"
    payload.write_text(json.dumps(releases), encoding="utf-8")
    return _run_validator(
        "github-tag-absent",
        "--json",
        str(payload),
        "--tag",
        "v0.4.0",
    )


def _run_isolated_gh_request(
    tmp_path: Path,
    *arguments: str,
) -> subprocess.CompletedProcess[str]:
    """Run gh through a closed loopback proxy with only a synthetic token."""

    gh = shutil.which("gh")
    if gh is None:
        pytest.skip("gh is not installed")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        proxy_port = int(probe.getsockname()[1])
    proxy_url = f"http://127.0.0.1:{proxy_port}"
    config_dir = tmp_path / "gh-config"
    state_dir = tmp_path / "gh-state"
    config_dir.mkdir(exist_ok=True)
    state_dir.mkdir(exist_ok=True)
    return subprocess.run(
        [gh, "api", *arguments],
        cwd=PROJECT_ROOT,
        env={
            "GH_CONFIG_DIR": str(config_dir),
            "GH_DEBUG": "api",
            "GH_TOKEN": "synthetic-request-target-token",
            "HTTPS_PROXY": proxy_url,
            "https_proxy": proxy_url,
            "NO_PROXY": "",
            "XDG_STATE_HOME": str(state_dir),
            "no_proxy": "",
        },
        text=True,
        capture_output=True,
        check=False,
        timeout=5,
    )


@pytest.mark.fast_always
def test_cli_rejects_obsolete_github_repository_push_command(
    tmp_path: Path,
) -> None:
    payload = tmp_path / "repository.json"
    payload.write_text(
        json.dumps({"permissions": {"push": True}}),
        encoding="utf-8",
    )

    completed = _run_validator(
        "github-repository-push",
        "--json",
        str(payload),
    )

    assert completed.returncode == 2
    assert "invalid choice" in completed.stderr
    assert "github-repository-push" in completed.stderr


@pytest.mark.fast_always
@pytest.mark.parametrize(
    "filename",
    [
        WHEEL,
        SDIST,
        "release notes+evidence.txt",
    ],
)
def test_github_upload_url_uses_canonical_host_and_safe_filename_query(
    filename: str,
) -> None:
    """Construct the exact uploads host target and encode its filename query."""

    completed = _run_validator(
        "github-upload-url",
        "--repository",
        "nutrik/vercor",
        "--release-id",
        "42",
        "--name",
        filename,
    )

    assert completed.returncode == 0, completed.stderr
    target = urlsplit(completed.stdout.strip())
    assert target.scheme == "https"
    assert target.netloc == "uploads.github.com"
    assert target.path == "/repos/nutrik/vercor/releases/42/assets"
    assert parse_qs(target.query, strict_parsing=True) == {"name": [filename]}


@pytest.mark.fast_always
def test_github_upload_url_rejects_path_like_asset_name() -> None:
    """Keep the upload query bound to one filename rather than a path."""

    completed = _run_validator(
        "github-upload-url",
        "--repository",
        "nutrik/vercor",
        "--release-id",
        "42",
        "--name",
        "../unexpected.whl",
    )

    assert completed.returncode != 0
    assert "asset name must be a plain filename" in completed.stderr


@pytest.mark.fast_always
def test_installed_gh_uses_canonical_upload_request_target(
    tmp_path: Path,
) -> None:
    """Exercise gh's actual request construction without contacting GitHub."""

    old_construction = _run_isolated_gh_request(
        tmp_path,
        "--hostname",
        "uploads.github.com",
        f"repos/nutrik/vercor/releases/42/assets?name={WHEEL}",
        "--method",
        "GET",
    )
    assert old_construction.returncode != 0
    assert "synthetic-request-target-token" not in old_construction.stderr
    assert "Request to https://api.uploads.github.com/" in old_construction.stderr
    assert "Host: api.uploads.github.com" in old_construction.stderr
    assert "127.0.0.1" in old_construction.stderr

    target = _run_validator(
        "github-upload-url",
        "--repository",
        "nutrik/vercor",
        "--release-id",
        "42",
        "--name",
        WHEEL,
    )
    assert target.returncode == 0, target.stderr
    completed = _run_isolated_gh_request(
        tmp_path,
        target.stdout.strip(),
        "--method",
        "GET",
    )
    assert completed.returncode != 0
    assert "synthetic-request-target-token" not in completed.stderr
    assert "Request to https://uploads.github.com/" in completed.stderr
    assert "Host: uploads.github.com" in completed.stderr
    assert "api.uploads.github.com" not in completed.stderr
    assert "127.0.0.1" in completed.stderr
    assert (tmp_path / "gh-state" / "gh" / "device-id").is_file()
    assert not (PROJECT_ROOT / ".local" / "state" / "gh" / "device-id").exists()


@pytest.mark.fast_always
def test_github_tag_absence_accepts_paginated_unrelated_releases(
    tmp_path: Path,
) -> None:
    """Accept a well-formed authenticated listing with no exact-tag release."""

    completed = _run_github_tag_absent_validator(
        tmp_path,
        [[{"id": 7, "tag_name": "v0.3.0", "draft": False}], []],
    )

    assert completed.returncode == 0, completed.stderr


@pytest.mark.fast_always
@pytest.mark.parametrize("draft", [True, False])
def test_github_tag_absence_rejects_draft_or_published_exact_tag(
    tmp_path: Path,
    draft: bool,
) -> None:
    """Treat both draft and published exact-tag releases as occupied state."""

    completed = _run_github_tag_absent_validator(
        tmp_path,
        [[{"id": 42, "tag_name": "v0.4.0", "draft": draft}]],
    )

    assert completed.returncode != 0
    assert "exact-tag release already exists" in completed.stderr


@pytest.mark.fast_always
def test_github_tag_absence_rejects_duplicate_exact_tag_releases(
    tmp_path: Path,
) -> None:
    """Reject an ambiguous listing with duplicate exact-tag releases."""

    completed = _run_github_tag_absent_validator(
        tmp_path,
        [
            {"id": 42, "tag_name": "v0.4.0", "draft": True},
            {"id": 43, "tag_name": "v0.4.0", "draft": False},
        ],
    )

    assert completed.returncode != 0
    assert "duplicate exact-tag releases" in completed.stderr


@pytest.mark.fast_always
@pytest.mark.parametrize(
    "releases",
    [
        {"not": "a release list"},
        [{"id": 42, "draft": True}],
    ],
)
def test_github_tag_absence_rejects_malformed_release_listing(
    tmp_path: Path,
    releases: object,
) -> None:
    """Reject malformed enumeration data instead of treating it as absence."""

    completed = _run_github_tag_absent_validator(tmp_path, releases)

    assert completed.returncode != 0
    assert "expected GitHub release" in completed.stderr


@pytest.mark.fast_always
@pytest.mark.parametrize("present", [WHEEL, SDIST])
def test_pypi_recovery_state_requires_exact_expected_filename_set(
    tmp_path: Path,
    present: str,
) -> None:
    """Reject an unexpected third PyPI file even when the expected file is valid."""

    manifest = tmp_path / "SHA256SUMS"
    _write_manifest(manifest)
    expected_digest = "a" * 64 if present == WHEEL else "b" * 64
    payload = tmp_path / "pypi.json"
    payload.write_text(
        json.dumps(
            {
                "urls": [
                    {"filename": present, "digests": {"sha256": expected_digest}},
                    {
                        "filename": UNEXPECTED,
                        "digests": {"sha256": "c" * 64},
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    completed = _run_validator(
        "pypi",
        "--json",
        str(payload),
        "--manifest",
        str(manifest),
        "--expect",
        present,
    )

    assert completed.returncode != 0
    assert "exact filename set" in completed.stderr


@pytest.mark.fast_always
@pytest.mark.parametrize("present", [WHEEL, SDIST])
def test_hosted_recovery_state_requires_exact_expected_asset_set(
    tmp_path: Path,
    present: str,
) -> None:
    """Reject an unexpected third hosted asset beside the expected asset."""

    payload = tmp_path / "release.json"
    payload.write_text(
        json.dumps({"assets": [{"name": present}, {"name": UNEXPECTED}]}),
        encoding="utf-8",
    )

    completed = _run_validator(
        "assets",
        "--json",
        str(payload),
        "--expect",
        present,
    )

    assert completed.returncode != 0
    assert "exact asset set" in completed.stderr


@pytest.mark.fast_always
def test_release_state_validator_accepts_exact_sets_and_verified_bytes(
    tmp_path: Path,
) -> None:
    """Accept exact package/asset sets and bytes matching the manifest."""

    downloaded = tmp_path / "downloaded"
    downloaded.mkdir()
    wheel = downloaded / WHEEL
    sdist = downloaded / SDIST
    wheel.write_bytes(b"wheel")
    sdist.write_bytes(b"sdist")
    manifest = tmp_path / "SHA256SUMS"
    completed_hashes = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (wheel, sdist)
    }
    manifest.write_text(
        "".join(f"{completed_hashes[name]}  {name}\n" for name in (WHEEL, SDIST)),
        encoding="utf-8",
    )
    pypi = tmp_path / "pypi.json"
    pypi.write_text(
        json.dumps(
            {
                "urls": [
                    {
                        "filename": name,
                        "digests": {"sha256": completed_hashes[name]},
                    }
                    for name in (WHEEL, SDIST)
                ]
            }
        ),
        encoding="utf-8",
    )
    release = tmp_path / "release.json"
    release.write_text(
        json.dumps({"assets": [{"name": WHEEL}, {"name": SDIST}]}),
        encoding="utf-8",
    )

    commands = (
        (
            "pypi",
            "--json",
            str(pypi),
            "--manifest",
            str(manifest),
            "--expect",
            WHEEL,
            SDIST,
        ),
        ("assets", "--json", str(release), "--expect", WHEEL, SDIST),
        (
            "files",
            "--directory",
            str(downloaded),
            "--manifest",
            str(manifest),
            "--expect",
            WHEEL,
            SDIST,
        ),
    )
    for command in commands:
        completed = _run_validator(*command)
        assert completed.returncode == 0, completed.stderr


@pytest.mark.fast_always
def test_release_state_validator_rejects_unexpected_manifest_entries(
    tmp_path: Path,
) -> None:
    """Reject a producer manifest that names anything beyond the two distributions."""

    downloaded = tmp_path / "downloaded"
    downloaded.mkdir()
    wheel = downloaded / WHEEL
    sdist = downloaded / SDIST
    wheel.write_bytes(b"wheel")
    sdist.write_bytes(b"sdist")
    digests = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (wheel, sdist)
    }
    manifest = tmp_path / "SHA256SUMS"
    manifest.write_text(
        "".join(f"{digests[name]}  {name}\n" for name in (WHEEL, SDIST))
        + f"{'c' * 64}  {UNEXPECTED}\n",
        encoding="utf-8",
    )
    completed = _run_validator(
        "files",
        "--directory",
        str(downloaded),
        "--manifest",
        str(manifest),
        "--expect",
        WHEEL,
        SDIST,
    )

    assert completed.returncode != 0
    assert "exact manifest entry set" in completed.stderr


@pytest.mark.fast_always
@pytest.mark.parametrize(
    "present",
    [
        (),
        (WHEEL,),
        (WHEEL, SDIST),
    ],
)
def test_github_draft_recovery_accepts_zero_one_or_two_verified_assets(
    tmp_path: Path,
    present: tuple[str, ...],
) -> None:
    """Accept only the three resumable exact-draft interruption states."""

    releases = [[_release([_release_asset(name) for name in present])]]

    completed, state_path = _run_github_release_validator(tmp_path, releases)

    assert completed.returncode == 0, completed.stderr
    assert json.loads(state_path.read_text(encoding="utf-8")) == {
        "state": "draft",
        "release_id": 42,
        "present": list(present),
        "missing": [name for name in (WHEEL, SDIST) if name not in present],
    }


@pytest.mark.fast_always
@pytest.mark.parametrize(
    ("case", "error"),
    [
        ("duplicate-releases", "duplicate exact-tag releases"),
        ("duplicate-assets", "duplicate asset names"),
        ("unexpected-asset", "unexpected draft asset"),
        ("bad-digest", "GitHub SHA-256 does not match manifest"),
    ],
)
def test_github_draft_recovery_rejects_ambiguous_or_bad_asset_state(
    tmp_path: Path,
    case: str,
    error: str,
) -> None:
    """Stop instead of overwriting duplicate, unexpected, or bad draft state."""

    exact = _release([_release_asset(WHEEL)])
    if case == "duplicate-releases":
        releases: object = [exact, {**exact, "id": 43}]
    elif case == "duplicate-assets":
        releases = [_release([_release_asset(WHEEL), _release_asset(WHEEL)])]
    elif case == "unexpected-asset":
        releases = [
            _release(
                [
                    _release_asset(WHEEL),
                    {
                        "id": 12,
                        "name": UNEXPECTED,
                        "state": "uploaded",
                        "digest": f"sha256:{'c' * 64}",
                    },
                ]
            )
        ]
    else:
        releases = [_release([_release_asset(WHEEL, digest=f"sha256:{'c' * 64}")])]

    completed, state_path = _run_github_release_validator(tmp_path, releases)

    assert completed.returncode != 0
    assert error in completed.stderr
    assert not state_path.exists()


@pytest.mark.fast_always
@pytest.mark.parametrize(
    ("overrides", "error"),
    [
        ({"name": "Wrong title"}, "unexpected release title"),
        ({"body": "Wrong notes\n"}, "unexpected release notes"),
        ({"prerelease": True}, "release must not be a prerelease"),
    ],
)
def test_github_draft_recovery_validates_exact_metadata(
    tmp_path: Path,
    overrides: dict[str, object],
    error: str,
) -> None:
    """Reject a draft whose title, notes, or prerelease state differs."""

    release = _release([])
    release.update(overrides)
    completed, _ = _run_github_release_validator(
        tmp_path,
        [release],
    )

    assert completed.returncode != 0
    assert error in completed.stderr


@pytest.mark.fast_always
def test_github_release_state_distinguishes_published_from_draft(
    tmp_path: Path,
) -> None:
    """Never treat an already-published exact release as a resumable draft."""

    published = [_release([_release_asset(WHEEL), _release_asset(SDIST)], draft=False)]

    rejected, rejected_state = _run_github_release_validator(
        tmp_path,
        published,
        allow_states=("absent", "draft"),
    )

    assert rejected.returncode != 0
    assert "published release state is not allowed" in rejected.stderr
    assert not rejected_state.exists()

    accepted, accepted_state = _run_github_release_validator(
        tmp_path,
        published,
        allow_states=("published",),
    )

    assert accepted.returncode == 0, accepted.stderr
    assert json.loads(accepted_state.read_text(encoding="utf-8")) == {
        "state": "published",
        "release_id": 42,
        "present": [WHEEL, SDIST],
        "missing": [],
    }
