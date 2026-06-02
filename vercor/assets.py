from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
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


@dataclass(frozen=True)
class _RegisteredAsset:
    """Normalized asset registry entry used by the generic cache layer."""

    filename: str
    md5: str


def _registered_asset(
    asset_key: str,
    registry: Mapping[str, Mapping[str, str]],
) -> _RegisteredAsset:
    asset = registry[asset_key]
    return _RegisteredAsset(filename=asset["filename"], md5=asset["md5"])


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


def _cached_asset_path(asset: _RegisteredAsset) -> Path:
    return _ASSETS_CACHE_DIR / asset.filename


def _verified_cached_asset_path(asset: _RegisteredAsset) -> Path | None:
    cached_path = _cached_asset_path(asset)
    if not cached_path.exists():
        return None
    if _md5sum(cached_path) == asset.md5:
        return cached_path
    cached_path.unlink()
    return None


def _download_registered_asset(asset: _RegisteredAsset, cached_path: Path) -> None:
    base_url = _asset_base_url()
    if base_url is None:
        raise AssetError(
            "Asset not found in cache and no remote base URL configured. "
            "Set VERCOR_ASSETS_BASE_URL to a server hosting VerCOR assets. "
            f"Missing asset: '{asset.filename}'"
        )

    url = f"{base_url}/{asset.filename}"
    try:
        _download_asset(url, cached_path)
    except Exception as e:
        raise AssetError(
            f"Failed to download asset '{asset.filename}' from '{url}': {e}"
        ) from e


def ensure_registered_asset(
    asset_key: str,
    registry: dict[str, dict[str, str]],
) -> Path:
    """Resolve a registered asset to a verified local cache path."""

    asset = _registered_asset(asset_key, registry)

    cached_path = _verified_cached_asset_path(asset)
    if cached_path is not None:
        return cached_path

    _ASSETS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cached_path = _cached_asset_path(asset)

    _download_registered_asset(asset, cached_path)

    actual_md5 = _md5sum(cached_path)
    if actual_md5 != asset.md5:
        if cached_path.exists():
            cached_path.unlink()
        raise AssetError(
            f"MD5 mismatch for asset '{asset.filename}': expected {asset.md5}, got {actual_md5}"
        )

    return cached_path
