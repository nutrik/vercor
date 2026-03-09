from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
import hashlib
from io import BytesIO
from pathlib import Path
from typing import Any, Literal, cast

import matplotlib
import numpy as np
import pytest
import vercor.tools as tools_module

from vercor.clock import Clock, DateTime360, DateTime365
from vercor.exceptions import AssetError, CouplerError, RegridderError
from vercor.grid import RectilinearGrid
from vercor.settings import VercorSettings
from vercor.tools import (
    _asset_base_url,
    _append_unique,
    _download_asset,
    _ensure_forcing_asset,
    _flatten_fields,
    check_remap_conservation,
    check_total_lnd_ocn_mask_sum,
    compute_ocn_lnd_masks_on_atm_grid,
    datetime_to_seconds_in_year,
    get_component,
    get_forcing_data,
    get_field_at_specific_time,
    get_field_time_slice,
    get_periodic_interval,
    grids_identical,
    is_leap_year,
    plot_component_scalar_vector_comparison,
    print_component_field_means_table,
    safe_component_nanmean,
)

matplotlib.use("Agg")


@dataclass
class DummyCoupler:
    clock: Clock
    settings: VercorSettings


def make_coupler(year_in_seconds: float) -> DummyCoupler:
    clock = Clock(start=datetime(2000, 1, 1), dt_seconds=1.0, steps=1)
    settings = VercorSettings(year_in_seconds=year_in_seconds)
    return DummyCoupler(clock=clock, settings=settings)


@dataclass
class DummyComponentA:
    name: str = "a"


@dataclass
class DummyComponentB:
    name: str = "b"


@dataclass
class DummyGridComponent:
    grid: RectilinearGrid
    fields: dict[str, np.ndarray]

    def get(self, field_name: str) -> np.ndarray:
        if field_name not in self.fields:
            raise KeyError(field_name)
        return self.fields[field_name]


def test_get_field_at_specific_time_weights_and_interpolation() -> None:
    coupler = make_coupler(year_in_seconds=12.0)

    # Build a (lat, lon, time) array with constant values per record
    lat, lon, nrec = 2, 3, 12
    arr = np.zeros((lat, lon, nrec), dtype=float)
    arr[..., 0] = 0.0
    arr[..., 1] = 10.0
    data = {"foo": arr}

    current_time = coupler.clock.start + timedelta(seconds=0.25)

    total_seconds = (current_time - coupler.clock.start).total_seconds()
    (n1, f1), (n2, f2) = get_periodic_interval(
        current_time=total_seconds,
        cycle_length=coupler.settings.year_in_seconds,
        rec_spacing=coupler.settings.year_in_seconds / 12.0,
        n_rec=12,
    )

    assert n1 == 0
    assert n2 == 1
    assert np.isclose(f1 + f2, 1.0)

    out = get_field_at_specific_time("foo", data, coupler, current_time=current_time)  # type: ignore
    assert np.allclose(out, 2.5)


def test_get_field_at_specific_time_boundary_record() -> None:
    coupler = make_coupler(year_in_seconds=120.0)

    lat, lon, nrec = 2, 2, 12
    arr = np.zeros((lat, lon, nrec), dtype=float)
    arr[..., 1] = 7.0
    data = {"foo": arr}

    rec_spacing = coupler.settings.year_in_seconds / 12.0
    current_time = coupler.clock.start + timedelta(seconds=rec_spacing)

    out = get_field_at_specific_time("foo", data, coupler, current_time=current_time)  # type: ignore
    assert np.allclose(out, 7.0)


def test_get_field_at_specific_time_axis_ordering() -> None:
    coupler = make_coupler(year_in_seconds=12.0)

    # lat-major data; output is expected to be (lon, lat) due to swapaxes
    arr = np.zeros((2, 3, 12), dtype=float)
    arr[:, :, 0] = np.array([[0.0, 1.0, 2.0], [10.0, 11.0, 12.0]])
    data = {"foo": arr}

    out = get_field_at_specific_time("foo", data, coupler, current_time=coupler.clock.start)  # type: ignore
    expected = arr[:, :, 0].swapaxes(-2, -1)

    assert out.shape == (3, 2)
    assert np.allclose(out, expected)


def test_get_field_at_specific_time_uses_coupler_clock_start_when_time_is_none() -> (
    None
):
    coupler = make_coupler(year_in_seconds=12.0)
    arr = np.zeros((2, 2, 12), dtype=float)
    arr[..., 0] = 3.0
    data = {"foo": arr}

    out = get_field_at_specific_time("foo", data, coupler, current_time=None)  # type: ignore
    assert np.allclose(out, 3.0)


def test_get_field_time_slice_basic_indexing() -> None:
    data = {"foo": np.arange(365 * 2, dtype=float).reshape(365, 2)}

    time = datetime(2001, 1, 1)
    out = get_field_time_slice("foo", data, time)
    assert np.allclose(out, data["foo"][0, :])

    time = datetime(2001, 12, 31)
    out = get_field_time_slice("foo", data, time)
    assert np.allclose(out, data["foo"][364, :])


def test_get_field_time_slice_no_leap_year_adjustment() -> None:
    data = {"foo": np.arange(366, dtype=float)}

    time = datetime(2000, 2, 29)
    out = get_field_time_slice("foo", data, time, no_leap=True)

    # Feb 29 in a leap year is mapped to Feb 28 when no_leap=True
    assert np.isclose(out, data["foo"][58])


def test_get_field_time_slice_leap_day_retained_when_requested() -> None:
    data = {"foo": np.arange(366, dtype=float)}

    time = datetime(2000, 2, 29)
    out = get_field_time_slice("foo", data, time, no_leap=False)

    assert np.isclose(out, data["foo"][59])


def test_get_field_time_slice_model_datetime_360_maps_to_real_month_lengths() -> None:
    data = {"foo": np.arange(365, dtype=float)}

    time = DateTime360(
        year=2001,
        month=1,
        day=30,
        hour=0,
        minute=0,
        second=0,
        microsecond=0,
        day_of_year=30,
    )
    out = get_field_time_slice("foo", data, time, no_leap=True)
    assert np.isclose(out, data["foo"][30])

    time = DateTime360(
        year=2001,
        month=2,
        day=3,
        hour=0,
        minute=0,
        second=0,
        microsecond=0,
        day_of_year=33,
    )
    out = get_field_time_slice("foo", data, time, no_leap=True)
    assert np.isclose(out, data["foo"][32])


def test_get_field_time_slice_model_datetime_360_february_non_leap() -> None:
    data = {"foo": np.arange(365, dtype=float)}

    time = DateTime360(
        year=2001,
        month=2,
        day=30,
        hour=0,
        minute=0,
        second=0,
        microsecond=0,
        day_of_year=60,
    )
    out = get_field_time_slice("foo", data, time, no_leap=True)

    assert np.isclose(out, data["foo"][58])


def test_get_field_time_slice_model_datetime_360_february_leap_allowed() -> None:
    data = {"foo": np.arange(366, dtype=float)}

    time = DateTime360(
        year=2000,
        month=2,
        day=30,
        hour=0,
        minute=0,
        second=0,
        microsecond=0,
        day_of_year=60,
    )
    out = get_field_time_slice("foo", data, time, no_leap=False)

    assert np.isclose(out, data["foo"][59])


def test_get_field_time_slice_model_datetime_365_uses_day_of_year_directly() -> None:
    data = {"foo": np.arange(365, dtype=float)}
    time = DateTime365(
        year=2001,
        month=3,
        day=1,
        hour=0,
        minute=0,
        second=0,
        microsecond=0,
        day_of_year=60,
    )
    out = get_field_time_slice("foo", data, time, no_leap=True)
    assert np.isclose(out, data["foo"][59])


def test_datetime_to_seconds_in_year_for_model_datetime_with_arithmetic() -> None:
    base = DateTime360(
        year=2001,
        month=1,
        day=1,
        hour=0,
        minute=0,
        second=0,
        microsecond=0,
        day_of_year=1,
    )
    shifted = base + timedelta(days=1, hours=2, minutes=3, seconds=4, microseconds=5)
    assert shifted - base == timedelta(
        days=1, hours=2, minutes=3, seconds=4, microseconds=5
    )

    seconds = datetime_to_seconds_in_year(shifted)
    assert np.isclose(seconds, 1 * 86400 + 2 * 3600 + 3 * 60 + 4 + 5e-6)


def test_get_periodic_interval_wraps_with_time_beyond_cycle() -> None:
    (n1, f1), (n2, f2) = get_periodic_interval(
        current_time=13.25,
        cycle_length=12.0,
        rec_spacing=1.0,
        n_rec=12,
    )

    assert n1 == 1
    assert n2 == 2
    assert np.isclose(f1 + f2, 1.0)


def test_is_leap_year_cases() -> None:
    assert is_leap_year(2000)
    assert not is_leap_year(1900)
    assert is_leap_year(2004)
    assert not is_leap_year(2001)


def test_get_forcing_data_valid_and_invalid_file_type(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_name = "model.nc"
    surface_name = "surface.nc"
    model_bytes = b"model-level-data"
    surface_bytes = b"surface-data"

    (tmp_path / model_name).write_bytes(model_bytes)
    (tmp_path / surface_name).write_bytes(surface_bytes)

    monkeypatch.setattr(tools_module, "_ASSETS_CACHE_DIR", tmp_path)
    monkeypatch.setattr(
        tools_module,
        "_FORCING_ASSETS",
        {
            "era5_model_levels": {
                "filename": model_name,
                "md5": hashlib.md5(model_bytes).hexdigest(),
            },
            "era5_surface": {
                "filename": surface_name,
                "md5": hashlib.md5(surface_bytes).hexdigest(),
            },
        },
    )

    model_level = get_forcing_data("era5_model_levels")
    surface = get_forcing_data("era5_surface")

    assert isinstance(model_level, Path)
    assert isinstance(surface, Path)
    assert str(model_level).endswith(model_name)
    assert str(surface).endswith(surface_name)

    with pytest.raises(AssetError, match="Unknown file_type"):
        get_forcing_data("unknown")


def test_flatten_fields_and_append_unique() -> None:
    flattened = _flatten_fields(["a", ("b", "c"), "d"])
    assert flattened == ["a", "b", "c", "d"]

    target = ["a", "b"]
    _append_unique(target, ["b", "c", "d", "a"])
    assert target == ["a", "b", "c", "d"]


def test_grids_identical_detects_equal_and_unequal_grids() -> None:
    lon = np.array([0.0, 1.0, 2.0])
    lat = np.array([-1.0, 0.0])
    g0 = RectilinearGrid("g0", longitude=lon, latitude=lat)
    g1 = RectilinearGrid("g1", longitude=lon.copy(), latitude=lat.copy())
    g2 = RectilinearGrid("g2", longitude=np.array([0.0, 1.5, 2.0]), latitude=lat.copy())

    assert grids_identical(g0, g1)
    assert not grids_identical(g0, g2)


def test_get_component_returns_single_and_raises_for_ambiguous_or_missing() -> None:
    allcomponents: dict[str, object] = {
        "a": DummyComponentA(name="ATM"),
        "b": DummyComponentB(name="OCN"),
    }

    selected = get_component(cast(Any, allcomponents), "ATM")
    assert isinstance(selected, DummyComponentA)

    with pytest.raises(CouplerError, match="No component"):
        get_component(cast(Any, allcomponents), "UNKNOWN")

    with pytest.raises(CouplerError, match="Multiple"):
        get_component(
            cast(
                Any,
                {
                    "a": DummyComponentA(name="OCN"),
                    "b": DummyComponentA(name="OCN"),
                },
            ),
            "OCN",
        )


def test_safe_component_nanmean_returns_nan_for_missing_fields() -> None:
    grid = RectilinearGrid(
        "dummy",
        longitude=np.array([0.0, 1.0]),
        latitude=np.array([0.0, 1.0]),
    )
    comp = DummyGridComponent(
        grid=grid, fields={"foo": np.array([[1.0, np.nan], [3.0, 5.0]])}
    )

    assert np.isclose(safe_component_nanmean(comp, "foo"), 3.0)
    assert np.isnan(safe_component_nanmean(comp, "does_not_exist"))


def test_print_component_field_means_table_with_callable_metric(
    capsys: pytest.CaptureFixture[str],
) -> None:
    grid = RectilinearGrid(
        "dummy",
        longitude=np.array([0.0, 1.0]),
        latitude=np.array([0.0, 1.0]),
    )
    atm = DummyGridComponent(
        grid=grid,
        fields={
            "u": np.array([[3.0, 4.0], [0.0, 0.0]]),
            "v": np.array([[4.0, 3.0], [0.0, 0.0]]),
            "temp": np.array([[280.0, 282.0], [284.0, 286.0]]),
        },
    )
    ocn = DummyGridComponent(
        grid=grid,
        fields={
            "u": np.array([[1.0, 2.0], [0.0, 0.0]]),
            "v": np.array([[2.0, 1.0], [0.0, 0.0]]),
            "temp": np.array([[270.0, 271.0], [272.0, 273.0]]),
        },
    )

    print_component_field_means_table(
        components={"ATM": atm, "OCN": ocn},
        fields=[
            ("temp", "temp"),
            (lambda c: np.sqrt(c.get("u") ** 2 + c.get("v") ** 2), "speed"),
        ],
        component_order=["ATM", "OCN"],
    )

    captured = capsys.readouterr().out
    assert "Variable" in captured
    assert "ATM" in captured
    assert "OCN" in captured
    assert "temp" in captured
    assert "speed" in captured


def test_plot_component_scalar_vector_comparison_aligns_axes_and_shapes() -> None:
    import matplotlib.pyplot as plt

    atm_grid = RectilinearGrid(
        "atm",
        longitude=np.array([0.0, 1.0, 2.0]),
        latitude=np.array([-1.0, 1.0]),
    )
    ocn_grid = RectilinearGrid(
        "ocn",
        longitude=np.array([0.0, 2.0]),
        latitude=np.array([-2.0, 0.0, 2.0]),
    )

    atm = DummyGridComponent(
        grid=atm_grid,
        fields={
            "scalar": np.array([[1.0, 3.0, 5.0], [2.0, 4.0, 6.0]]),
            "u": np.ones((2, 3)),
            "v": np.zeros((2, 3)),
        },
    )
    ocn = DummyGridComponent(
        grid=ocn_grid,
        fields={
            "scalar": np.array([[7.0, 10.0], [8.0, 11.0], [9.0, 12.0]]),
            "u": np.zeros((3, 2)),
            "v": np.ones((3, 2)),
        },
    )

    fig, axs, scalar_mappable = plot_component_scalar_vector_comparison(
        rows=[
            ("ATM", atm, "scalar", "u", "v"),
            ("OCN", ocn, "scalar", "u", "v"),
        ],
        figsize=(8.0, 5.0),
        quiver_scale=10.0,
    )

    assert axs.shape == (2, 2)
    assert scalar_mappable is not None
    assert np.allclose(axs[0, 0].get_xlim(), axs[1, 0].get_xlim())
    assert np.allclose(axs[0, 1].get_xlim(), axs[1, 1].get_xlim())
    assert np.allclose(axs[0, 0].get_ylim(), axs[1, 0].get_ylim())

    plt.close(fig)


def test_plot_component_scalar_vector_comparison_rejects_empty_rows() -> None:
    with pytest.raises(ValueError, match="at least one component"):
        plot_component_scalar_vector_comparison(rows=[])


def test_asset_base_url_normalizes_and_handles_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        tools_module, "VERCOR_ASSETS_BASE_URL", " https://example.test/assets// "
    )
    assert _asset_base_url() == "https://example.test/assets"

    monkeypatch.setattr(tools_module, "VERCOR_ASSETS_BASE_URL", "   ")
    assert _asset_base_url() is None

    monkeypatch.setattr(tools_module, "VERCOR_ASSETS_BASE_URL", None)
    assert _asset_base_url() is None


def test_download_asset_writes_response_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = b"forcing-content"

    class DummyResponse:
        def __enter__(self) -> BytesIO:
            return BytesIO(payload)

        def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> Literal[False]:
            return False

    monkeypatch.setattr(tools_module, "urlopen", lambda _url: DummyResponse())

    target = tmp_path / "nested" / "asset.nc"
    _download_asset("https://example.test/asset.nc", target)

    assert target.exists()
    assert target.read_bytes() == payload


def test_ensure_forcing_asset_uses_valid_cached_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    filename = "cached.nc"
    payload = b"valid-cached"
    (tmp_path / filename).write_bytes(payload)

    monkeypatch.setattr(tools_module, "_ASSETS_CACHE_DIR", tmp_path)
    monkeypatch.setattr(
        tools_module,
        "_FORCING_ASSETS",
        {"k": {"filename": filename, "md5": hashlib.md5(payload).hexdigest()}},
    )

    # Must not be called when cache hash is valid.
    monkeypatch.setattr(
        tools_module,
        "_download_asset",
        lambda _url, _target: (_ for _ in ()).throw(
            AssertionError("unexpected download")
        ),
    )

    resolved = _ensure_forcing_asset("k")
    assert resolved == tmp_path / filename


def test_ensure_forcing_asset_downloads_when_cached_md5_invalid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    filename = "downloaded.nc"
    cached = tmp_path / filename
    cached.write_bytes(b"stale")

    downloaded = b"fresh-content"

    monkeypatch.setattr(tools_module, "_ASSETS_CACHE_DIR", tmp_path)
    monkeypatch.setattr(
        tools_module,
        "_FORCING_ASSETS",
        {"k": {"filename": filename, "md5": hashlib.md5(downloaded).hexdigest()}},
    )
    monkeypatch.setattr(
        tools_module, "VERCOR_ASSETS_BASE_URL", "https://example.test/base"
    )

    def fake_download(url: str, target: Path) -> None:
        assert url == "https://example.test/base/downloaded.nc"
        target.write_bytes(downloaded)

    monkeypatch.setattr(tools_module, "_download_asset", fake_download)

    resolved = _ensure_forcing_asset("k")
    assert resolved == cached
    assert cached.read_bytes() == downloaded


def test_ensure_forcing_asset_errors_without_base_url(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(tools_module, "_ASSETS_CACHE_DIR", tmp_path)
    monkeypatch.setattr(
        tools_module,
        "_FORCING_ASSETS",
        {"k": {"filename": "missing.nc", "md5": hashlib.md5(b"x").hexdigest()}},
    )
    monkeypatch.setattr(tools_module, "VERCOR_ASSETS_BASE_URL", None)

    with pytest.raises(AssetError, match="no remote base URL configured"):
        _ensure_forcing_asset("k")


def test_ensure_forcing_asset_raises_and_deletes_on_md5_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    filename = "bad.nc"
    target = tmp_path / filename

    monkeypatch.setattr(tools_module, "_ASSETS_CACHE_DIR", tmp_path)
    monkeypatch.setattr(
        tools_module,
        "_FORCING_ASSETS",
        {"k": {"filename": filename, "md5": hashlib.md5(b"expected").hexdigest()}},
    )
    monkeypatch.setattr(tools_module, "VERCOR_ASSETS_BASE_URL", "https://example.test")
    monkeypatch.setattr(
        tools_module, "_download_asset", lambda _url, tgt: tgt.write_bytes(b"wrong")
    )

    with pytest.raises(AssetError, match="MD5 mismatch"):
        _ensure_forcing_asset("k")

    assert not target.exists()


def test_compute_ocn_lnd_masks_on_atm_grid_clips_and_builds_binary_land_mask() -> None:
    class DummyRegridder:
        def __call__(self, _arr: np.ndarray) -> np.ndarray:
            return np.array([[1.2, -0.2], [0.4, 0.0]])

    ocean_binary_mask = np.array([[1.0, 0.0], [1.0, 0.0]])
    ocn_fmask, lnd_fmask, lnd_bmask = compute_ocn_lnd_masks_on_atm_grid(
        ocean_binary_mask,
        cast(Any, DummyRegridder()),
    )

    assert np.allclose(ocn_fmask, np.array([[1.0, 0.0], [0.4, 0.0]]))
    assert np.allclose(lnd_fmask, np.array([[0.0, 1.0], [0.6, 1.0]]))
    assert np.array_equal(lnd_bmask, np.array([[0, 1], [1, 1]]))


def test_check_total_lnd_ocn_mask_sum_success_and_failure() -> None:
    lnd_good = np.array([[0.3, 1.0], [0.0, 0.8]])
    ocn_good = np.array([[0.7, 0.0], [1.0, 0.2]])
    check_total_lnd_ocn_mask_sum(lnd_good, ocn_good)

    lnd_bad = np.array([[0.3, 1.0], [0.0, 0.8]])
    ocn_bad = np.array([[0.7, 0.0], [1.0, 0.0]])
    with pytest.raises(RegridderError, match="must sum to approx. 1"):
        check_total_lnd_ocn_mask_sum(lnd_bad, ocn_bad)


def test_check_remap_conservation_handles_skip_and_mismatch(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    class DummyRemapper:
        def __init__(
            self,
            src_lat_b: np.ndarray,
            dst_lat_b: np.ndarray,
            src_mass: float,
            dst_mass: float,
        ) -> None:
            self.src_lat_b = src_lat_b
            self.dst_lat_b = dst_lat_b
            self._src_mass = src_mass
            self._dst_mass = dst_mass

        def get_src_total_mass(self, _arr: np.ndarray) -> float:
            return self._src_mass

        def get_dst_total_mass(self, _arr: np.ndarray) -> float:
            return self._dst_mass

    class DummyRegridder:
        def __init__(self, interpolator: Any) -> None:
            self.interpolator = interpolator

    monkeypatch.setattr(tools_module, "ConservativeRectilinearRemapper", DummyRemapper)

    # Different lat bounds: should skip strict conservation check and not raise.
    skip_interp = DummyRemapper(
        src_lat_b=np.array([-90.0, 0.0, 90.0]),
        dst_lat_b=np.array([-80.0, 0.0, 80.0]),
        src_mass=10.0,
        dst_mass=1.0,
    )
    check_remap_conservation(
        cast(Any, DummyRegridder(skip_interp)),
        np.ones((2, 2)),
        np.ones((2, 2)),
    )
    assert "Skipping mass conservation check" in capsys.readouterr().out

    # Same bounds with unequal mass: must raise.
    mismatch_interp = DummyRemapper(
        src_lat_b=np.array([-90.0, 0.0, 90.0]),
        dst_lat_b=np.array([-90.0, 0.0, 90.0]),
        src_mass=10.0,
        dst_mass=9.0,
    )
    with pytest.raises(RegridderError, match="does not conserve total mass"):
        check_remap_conservation(
            cast(Any, DummyRegridder(mismatch_interp)),
            np.ones((2, 2)),
            np.ones((2, 2)),
        )
