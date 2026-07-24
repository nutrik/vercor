"""Fail-closed validation for VerCOR release recovery state.

This script is intentionally standard-library-only so maintainers can validate
fresh PyPI JSON, GitHub release JSON, and downloaded hosted assets before a
recovery mutation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any
from urllib.parse import quote, urlencode


def _read_json(path: Path) -> dict[str, Any]:
    """Read one JSON object or terminate with a concise error."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return payload


def _read_manifest(path: Path) -> dict[str, str]:
    """Read a shasum-compatible manifest and reject duplicate names."""

    manifest: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        digest, name = line.split(maxsplit=1)
        name = name.lstrip("*")
        if name in manifest:
            raise ValueError(f"{path}: duplicate manifest name: {name}")
        manifest[name] = digest
    return manifest


def _require_exact_names(
    actual_names: list[str],
    expected_names: list[str],
    *,
    description: str,
) -> None:
    """Require an exact, duplicate-free filename set."""

    if len(actual_names) != len(set(actual_names)):
        raise ValueError(f"{description}: duplicate names: {actual_names}")
    if set(actual_names) != set(expected_names) or len(actual_names) != len(
        expected_names
    ):
        raise ValueError(
            f"{description}: expected exact {description} "
            f"{sorted(expected_names)}, got {sorted(actual_names)}"
        )


def _sha256(path: Path) -> str:
    """Return the SHA-256 digest of one file."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_pypi(arguments: argparse.Namespace) -> None:
    """Validate an exact PyPI filename set and every expected digest."""

    payload = _read_json(arguments.json)
    urls = payload.get("urls")
    if not isinstance(urls, list) or not all(isinstance(item, dict) for item in urls):
        raise ValueError(f"{arguments.json}: expected a PyPI urls list")
    filenames = [str(item.get("filename")) for item in urls]
    _require_exact_names(
        filenames,
        arguments.expect,
        description="filename set",
    )
    manifest = _read_manifest(arguments.manifest)
    for item in urls:
        filename = str(item["filename"])
        digests = item.get("digests")
        if not isinstance(digests, dict):
            raise ValueError(f"{filename}: missing PyPI digests")
        expected_digest = manifest.get(filename)
        if expected_digest is None:
            raise ValueError(f"{filename}: missing from checksum manifest")
        if digests.get("sha256") != expected_digest:
            raise ValueError(f"{filename}: PyPI SHA-256 does not match manifest")


def _validate_assets(arguments: argparse.Namespace) -> None:
    """Validate an exact hosted-release asset-name set."""

    payload = _read_json(arguments.json)
    assets = payload.get("assets")
    if not isinstance(assets, list) or not all(
        isinstance(item, dict) for item in assets
    ):
        raise ValueError(f"{arguments.json}: expected a release assets list")
    names = [str(item.get("name")) for item in assets]
    _require_exact_names(names, arguments.expect, description="asset set")


def _release_list(path: Path) -> list[dict[str, Any]]:
    """Read a flat REST release list or ``gh api --paginate --slurp`` pages."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"{path}: expected GitHub releases list")
    if payload and all(isinstance(page, list) for page in payload):
        releases = [item for page in payload for item in page]
    else:
        releases = payload
    if not all(isinstance(item, dict) for item in releases):
        raise ValueError(f"{path}: expected GitHub release objects")
    return releases


def _validate_github_tag_absent(arguments: argparse.Namespace) -> None:
    """Require an authenticated release listing to contain no exact tag."""

    releases = _release_list(arguments.json)
    matching: list[dict[str, Any]] = []
    for index, release in enumerate(releases):
        tag_name = release.get("tag_name")
        if not isinstance(tag_name, str) or not tag_name:
            raise ValueError(
                f"{arguments.json}: expected GitHub release {index} "
                "to have a non-empty tag_name"
            )
        if tag_name == arguments.tag:
            matching.append(release)
    if len(matching) > 1:
        raise ValueError(f"{arguments.tag}: duplicate exact-tag releases")
    if matching:
        raise ValueError(f"{arguments.tag}: exact-tag release already exists")


def _github_upload_url(arguments: argparse.Namespace) -> None:
    """Print one canonical, safely encoded GitHub release-asset upload URL."""

    repository_parts = arguments.repository.split("/")
    if (
        len(repository_parts) != 2
        or any(not part for part in repository_parts)
        or any(
            not all(character.isalnum() or character in "._-" for character in part)
            for part in repository_parts
        )
    ):
        raise ValueError("repository must be OWNER/REPOSITORY")
    if arguments.release_id <= 0:
        raise ValueError("release id must be positive")
    if (
        not arguments.name
        or arguments.name in {".", ".."}
        or "/" in arguments.name
        or "\\" in arguments.name
        or "\x00" in arguments.name
    ):
        raise ValueError("asset name must be a plain filename")
    query = urlencode({"name": arguments.name}, quote_via=quote)
    print(
        f"https://uploads.github.com/repos/{arguments.repository}"
        f"/releases/{arguments.release_id}/assets?{query}"
    )


def _validate_github_releases(arguments: argparse.Namespace) -> None:
    """Validate the unique exact-tag GitHub Release recovery state."""

    releases = _release_list(arguments.json)
    matching = [
        release for release in releases if release.get("tag_name") == arguments.tag
    ]
    if len(matching) > 1:
        raise ValueError(f"{arguments.tag}: duplicate exact-tag releases")

    manifest = _read_manifest(arguments.manifest)
    _require_exact_names(
        list(manifest),
        arguments.expect,
        description="manifest entry set",
    )
    if not matching:
        state = "absent"
        release_id = None
        present: list[str] = []
    else:
        release = matching[0]
        if release.get("name") != arguments.title:
            raise ValueError(f"{arguments.tag}: unexpected release title")
        expected_notes = arguments.notes_file.read_text(encoding="utf-8")
        if release.get("body") != expected_notes:
            raise ValueError(f"{arguments.tag}: unexpected release notes")
        if release.get("prerelease") is not False:
            raise ValueError(f"{arguments.tag}: release must not be a prerelease")
        draft = release.get("draft")
        if not isinstance(draft, bool):
            raise ValueError(f"{arguments.tag}: release draft state is not boolean")
        state = "draft" if draft else "published"
        release_id = release.get("id")
        if (
            not isinstance(release_id, int)
            or isinstance(release_id, bool)
            or release_id <= 0
        ):
            raise ValueError(f"{arguments.tag}: invalid release id")
        assets = release.get("assets")
        if not isinstance(assets, list) or not all(
            isinstance(asset, dict) for asset in assets
        ):
            raise ValueError(f"{arguments.tag}: expected a release assets list")
        names = [str(asset.get("name")) for asset in assets]
        if len(names) != len(set(names)):
            raise ValueError(f"{arguments.tag}: duplicate asset names")
        unexpected = sorted(set(names).difference(arguments.expect))
        if unexpected:
            raise ValueError(
                f"{arguments.tag}: unexpected draft asset: {unexpected[0]}"
            )
        for asset in assets:
            name = str(asset["name"])
            if asset.get("state") != "uploaded":
                raise ValueError(f"{name}: GitHub asset is not uploaded")
            expected_digest = manifest[name]
            if asset.get("digest") != f"sha256:{expected_digest}":
                raise ValueError(f"{name}: GitHub SHA-256 does not match manifest")
        present = [name for name in arguments.expect if name in names]

    if state not in arguments.allow_state:
        raise ValueError(f"{state} release state is not allowed")
    output = {
        "state": state,
        "release_id": release_id,
        "present": present,
        "missing": [name for name in arguments.expect if name not in present],
    }
    arguments.state_output.write_text(
        json.dumps(output, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _validate_files(arguments: argparse.Namespace) -> None:
    """Validate exact downloaded filenames and their manifest digests."""

    actual_names = sorted(
        path.name for path in arguments.directory.iterdir() if path.is_file()
    )
    _require_exact_names(actual_names, arguments.expect, description="file set")
    manifest = _read_manifest(arguments.manifest)
    _require_exact_names(
        list(manifest),
        arguments.expect,
        description="manifest entry set",
    )
    for name in arguments.expect:
        expected_digest = manifest.get(name)
        if expected_digest is None:
            raise ValueError(f"{name}: missing from checksum manifest")
        if _sha256(arguments.directory / name) != expected_digest:
            raise ValueError(f"{name}: downloaded SHA-256 does not match manifest")


def _validate_differs(arguments: argparse.Namespace) -> None:
    """Require a downloaded selected asset to differ from the manifest."""

    manifest = _read_manifest(arguments.manifest)
    expected_digest = manifest.get(arguments.name)
    if expected_digest is None:
        raise ValueError(f"{arguments.name}: missing from checksum manifest")
    if arguments.file.name != arguments.name:
        raise ValueError(
            f"selected filename mismatch: {arguments.file.name} != {arguments.name}"
        )
    if _sha256(arguments.file) == expected_digest:
        raise ValueError(
            f"{arguments.name}: selected asset unexpectedly matches manifest"
        )


def _parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    pypi = subparsers.add_parser("pypi")
    pypi.add_argument("--json", type=Path, required=True)
    pypi.add_argument("--manifest", type=Path, required=True)
    pypi.add_argument("--expect", nargs="+", required=True)
    pypi.set_defaults(run=_validate_pypi)

    assets = subparsers.add_parser("assets")
    assets.add_argument("--json", type=Path, required=True)
    assets.add_argument("--expect", nargs="+", required=True)
    assets.set_defaults(run=_validate_assets)

    github_tag_absent = subparsers.add_parser("github-tag-absent")
    github_tag_absent.add_argument("--json", type=Path, required=True)
    github_tag_absent.add_argument("--tag", required=True)
    github_tag_absent.set_defaults(run=_validate_github_tag_absent)

    github_upload_url = subparsers.add_parser("github-upload-url")
    github_upload_url.add_argument("--repository", required=True)
    github_upload_url.add_argument("--release-id", type=int, required=True)
    github_upload_url.add_argument("--name", required=True)
    github_upload_url.set_defaults(run=_github_upload_url)

    github_releases = subparsers.add_parser("github-releases")
    github_releases.add_argument("--json", type=Path, required=True)
    github_releases.add_argument("--manifest", type=Path, required=True)
    github_releases.add_argument("--tag", required=True)
    github_releases.add_argument("--title", required=True)
    github_releases.add_argument("--notes-file", type=Path, required=True)
    github_releases.add_argument("--expect", nargs="+", required=True)
    github_releases.add_argument(
        "--allow-state",
        nargs="+",
        choices=("absent", "draft", "published"),
        required=True,
    )
    github_releases.add_argument("--state-output", type=Path, required=True)
    github_releases.set_defaults(run=_validate_github_releases)

    files = subparsers.add_parser("files")
    files.add_argument("--directory", type=Path, required=True)
    files.add_argument("--manifest", type=Path, required=True)
    files.add_argument("--expect", nargs="+", required=True)
    files.set_defaults(run=_validate_files)

    differs = subparsers.add_parser("differs")
    differs.add_argument("--file", type=Path, required=True)
    differs.add_argument("--manifest", type=Path, required=True)
    differs.add_argument("--name", required=True)
    differs.set_defaults(run=_validate_differs)
    return parser


def main() -> int:
    """Validate one release-state boundary."""

    arguments = _parser().parse_args()
    try:
        arguments.run(arguments)
    except (KeyError, OSError, TypeError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
