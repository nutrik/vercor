from __future__ import annotations

from io import BytesIO
import hashlib
from pathlib import Path
from typing import Any, Literal

import pytest
import vercor.assets as assets_module

from vercor.assets import _asset_base_url, _download_asset, ensure_registered_asset
from vercor.exceptions import AssetError

pytestmark = pytest.mark.fast_always


def test_asset_base_url_normalizes_and_handles_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        assets_module, "VERCOR_ASSETS_BASE_URL", " https://example.test/assets// "
    )
    assert _asset_base_url() == "https://example.test/assets"

    monkeypatch.setattr(assets_module, "VERCOR_ASSETS_BASE_URL", "   ")
    assert _asset_base_url() is None

    monkeypatch.setattr(assets_module, "VERCOR_ASSETS_BASE_URL", None)
    assert _asset_base_url() is None


def test_download_asset_writes_response_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = b"asset-content"

    class DummyResponse:
        def __enter__(self) -> BytesIO:
            return BytesIO(payload)

        def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> Literal[False]:
            return False

    monkeypatch.setattr(assets_module, "urlopen", lambda _url: DummyResponse())

    target = tmp_path / "nested" / "asset.nc"
    _download_asset("https://example.test/asset.nc", target)

    assert target.exists()
    assert target.read_bytes() == payload


def test_ensure_registered_asset_uses_valid_cached_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    filename = "cached.nc"
    payload = b"valid-cached"
    (tmp_path / filename).write_bytes(payload)

    monkeypatch.setattr(assets_module, "_ASSETS_CACHE_DIR", tmp_path)
    monkeypatch.setattr(
        assets_module,
        "_download_asset",
        lambda _url, _target: (_ for _ in ()).throw(
            AssertionError("unexpected download")
        ),
    )

    resolved = ensure_registered_asset(
        "k",
        {"k": {"filename": filename, "md5": hashlib.md5(payload).hexdigest()}},
    )
    assert resolved == tmp_path / filename


def test_ensure_registered_asset_downloads_when_cached_md5_invalid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    filename = "downloaded.nc"
    cached = tmp_path / filename
    cached.write_bytes(b"stale")
    downloaded = b"fresh-content"

    monkeypatch.setattr(assets_module, "_ASSETS_CACHE_DIR", tmp_path)
    monkeypatch.setattr(
        assets_module, "VERCOR_ASSETS_BASE_URL", "https://example.test/base"
    )

    def fake_download(url: str, target: Path) -> None:
        assert url == "https://example.test/base/downloaded.nc"
        target.write_bytes(downloaded)

    monkeypatch.setattr(assets_module, "_download_asset", fake_download)

    resolved = ensure_registered_asset(
        "k",
        {"k": {"filename": filename, "md5": hashlib.md5(downloaded).hexdigest()}},
    )
    assert resolved == cached
    assert cached.read_bytes() == downloaded


def test_ensure_registered_asset_errors_without_base_url(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(assets_module, "_ASSETS_CACHE_DIR", tmp_path)
    monkeypatch.setattr(assets_module, "VERCOR_ASSETS_BASE_URL", None)

    with pytest.raises(AssetError, match="Asset not found in cache") as exc_info:
        ensure_registered_asset(
            "k",
            {"k": {"filename": "missing.nc", "md5": hashlib.md5(b"x").hexdigest()}},
        )

    message = str(exc_info.value)
    assert "forcing" not in message.lower()
    assert "VERCOR_ASSETS_BASE_URL" in message
    assert "missing.nc" in message


def test_ensure_registered_asset_raises_and_deletes_on_md5_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    filename = "bad.nc"
    target = tmp_path / filename

    monkeypatch.setattr(assets_module, "_ASSETS_CACHE_DIR", tmp_path)
    monkeypatch.setattr(assets_module, "VERCOR_ASSETS_BASE_URL", "https://example.test")
    monkeypatch.setattr(
        assets_module, "_download_asset", lambda _url, tgt: tgt.write_bytes(b"wrong")
    )

    with pytest.raises(AssetError, match="MD5 mismatch"):
        ensure_registered_asset(
            "k",
            {"k": {"filename": filename, "md5": hashlib.md5(b"expected").hexdigest()}},
        )

    assert not target.exists()
