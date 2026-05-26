from __future__ import annotations

import hashlib
import os
from pathlib import Path
import shutil
from urllib.request import urlopen

from vercor.exceptions import AssetError

VERCOR_ASSETS_BASE_URL = (
    os.environ.get("VERCOR_ASSETS_BASE_URL")
    or "https://sid.erda.dk/share_redirect/bC5N6nQcbY/"
)

_ASSETS_CACHE_DIR = Path.home() / ".vercor" / "assets"


def _md5sum(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_asset(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(url) as response, target.open("wb") as out:
        shutil.copyfileobj(response, out)


def _asset_base_url() -> str | None:
    base_url = VERCOR_ASSETS_BASE_URL
    if base_url is None:
        return None
    stripped = base_url.strip().rstrip("/")
    return stripped if stripped else None


def ensure_registered_asset(
    asset_key: str,
    registry: dict[str, dict[str, str]],
) -> Path:
    """Resolve a registered asset to a verified local cache path."""

    asset = registry[asset_key]
    filename = asset["filename"]
    expected_md5 = asset["md5"]

    cached_path = _ASSETS_CACHE_DIR / filename
    if cached_path.exists():
        if _md5sum(cached_path) == expected_md5:
            return cached_path
        cached_path.unlink()

    _ASSETS_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    base_url = _asset_base_url()
    if base_url is None:
        raise AssetError(
            "Forcing asset not found in cache and no remote base URL configured. "
            "Set VERCOR_ASSETS_BASE_URL to a server hosting VerCOR forcing assets. "
            f"Missing asset: '{filename}'"
        )

    url = f"{base_url}/{filename}"
    try:
        _download_asset(url, cached_path)
    except Exception as e:
        raise AssetError(
            f"Failed to download forcing asset '{filename}' from '{url}': {e}"
        ) from e

    actual_md5 = _md5sum(cached_path)
    if actual_md5 != expected_md5:
        if cached_path.exists():
            cached_path.unlink()
        raise AssetError(
            f"MD5 mismatch for forcing asset '{filename}': expected {expected_md5}, got {actual_md5}"
        )

    return cached_path
