from __future__ import annotations

from pathlib import Path

from vercor.assets import ensure_registered_asset
from vercor.exceptions import AssetError

_FORCING_ASSETS: dict[str, dict[str, str]] = {
    "era5_model_levels": {
        "filename": "era5_198x_ml_4x4deg_monthly_mean.nc",
        "md5": "2ada464b2eb2bf3a7abec7f77a18634c",
    },
    "era5_surface": {
        "filename": "era5_198x_sfc_4x4deg_monthly_mean.nc",
        "md5": "304d547b72b3677f7bc44c71bcf7cb8f",
    },
    "era5_land": {
        "filename": "era5_lnd_skt_1980.nc",
        "md5": "b0877a7715c438b7a17593ad00bb8218",
    },
    "era5_land_masked": {
        "filename": "era5_lnd_skt_masked_1980.nc",
        "md5": "cea9349ee88f1ecb55572f87f065ff9b",
    },
    "erainterim_ocean_4deg": {
        "filename": "forcing_4deg_global_open_itf.nc",
        "md5": "cfcc6d8cde8da5a74ecec00309d92dd7",
    },
    "erainterim_ocean_1deg": {
        "filename": "forcing_1deg_global.nc",
        "md5": "1fc86f88acd820da078c8da5873cfa01",
    },
    "ecmwf_4deg_monthly": {
        "filename": "ecmwf_4deg_monthly_nc4.nc",
        "md5": "d1b4e0e199d7a5883cf7c88d3d6bcb27",
    },
}


def get_forcing_data(file_type: str) -> Path:
    """Resolve setup forcing data to cached assets in $HOME/.vercor/assets."""

    if file_type not in _FORCING_ASSETS:
        allowed = ", ".join(sorted(_FORCING_ASSETS.keys()))
        raise AssetError(
            f"Unknown file_type '{file_type}'. Allowed values are: {allowed}"
        )
    return ensure_registered_asset(file_type, _FORCING_ASSETS)
