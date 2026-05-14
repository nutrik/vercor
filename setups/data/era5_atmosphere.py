from pathlib import Path
from typing import Any, Optional, cast

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from vercor.components.base import DataComponent, data_component
from vercor.dtypes import as_jax_real_array
from vercor.field_layout import (
    canonicalize_time_last_level_field,
    canonicalize_time_last_surface_field,
)
from vercor.fluxes.utilities import (
    compute_air_density,
    get_altitudes_hybrid_sigma_levels,
    compute_pressure_levels,
    compute_potential_temperature,
)
from vercor.grid import RectilinearGrid
from vercor.runtime.contexts import ComponentInitContext
from vercor.settings import VercorSettings
from vercor.assets import get_forcing_data
from setups.data.forcing import read_forcing as _read_forcing

_ERA5_ATMOSPHERE_FIELD_NAMES = (
    "surface_pressure",
    "specific_humidity_3d",
    "temperature_3d",
    "u_velocity",
    "v_velocity",
    "net_shortwave_radiation_flux",
    "downward_longwave_radiation_flux",
    "specific_humidity",
    "temperature",
    "model_level_height",
    "density",
    "potential_temperature",
)


def _decode_surface_pressure(lnsp: ArrayLike) -> jax.Array:
    """Convert log surface pressure to physical pressure in Pascals."""
    return jnp.exp(as_jax_real_array(lnsp))


def _compute_monthly_diagnostics(
    settings: VercorSettings,
    surface_pressure: ArrayLike,
    hyai: ArrayLike,
    hybi: ArrayLike,
    hyam: ArrayLike,
    hybm: ArrayLike,
    temperature_3d: ArrayLike,
    specific_humidity_3d: ArrayLike,
    temperature: ArrayLike,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Compute ERA5 diagnostics for one monthly slice on the runtime JAX path."""

    surface_pressure_array = as_jax_real_array(surface_pressure, settings)
    temperature_3d_array = as_jax_real_array(temperature_3d, settings).transpose(
        (1, 2, 0)
    )
    specific_humidity_3d_array = as_jax_real_array(
        specific_humidity_3d,
        settings,
    ).transpose((1, 2, 0))
    temperature_array = as_jax_real_array(temperature, settings)
    hyai_array = as_jax_real_array(hyai, settings)
    hybi_array = as_jax_real_array(hybi, settings)
    hyam_array = as_jax_real_array(hyam, settings)
    hybm_array = as_jax_real_array(hybm, settings)

    ph = compute_pressure_levels(surface_pressure_array, hyai_array, hybi_array)
    pf = compute_pressure_levels(surface_pressure_array, hyam_array, hybm_array)
    model_level_height = get_altitudes_hybrid_sigma_levels(
        settings,
        temperature_3d_array,
        specific_humidity_3d_array,
        ph,
    )[..., 1]
    density = compute_air_density(settings, pf[..., 0], temperature_array)
    potential_temperature = compute_potential_temperature(
        settings,
        temperature_array,
        pf[..., 0],
    )

    return model_level_height, density, potential_temperature


def make_era5_atmosphere(
    name: str = "ATM",
    model_level_file: Optional[Path] = None,
    surface_file: Optional[Path] = None,
) -> DataComponent:
    """Return an ERA5 atmosphere forcing component."""

    if model_level_file is None:
        model_level_file = get_forcing_data("era5_model_levels")
    if surface_file is None:
        surface_file = get_forcing_data("era5_surface")

    data_files = {
        "model_level": str(model_level_file),
        "surface": str(surface_file),
    }

    longitude = _read_forcing(data_files, "longitude", where="model_level")
    latitude = _read_forcing(data_files, "latitude", where="model_level")[::-1]

    grid = RectilinearGrid(
        name=f"{name.lower()}-grid",
        longitude=longitude,
        latitude=latitude,
    )

    hyai = as_jax_real_array(
        _read_forcing(data_files, "hyai", where="model_level")[-3:]
    )
    hybi = as_jax_real_array(
        _read_forcing(data_files, "hybi", where="model_level")[-3:]
    )
    hyam = as_jax_real_array(
        _read_forcing(data_files, "hyam", where="model_level")[-2:]
    )
    hybm = as_jax_real_array(
        _read_forcing(data_files, "hybm", where="model_level")[-2:]
    )

    lnsp = _read_forcing(data_files, "lnsp", where="model_level", flip_y=True)[
        ..., 0, :
    ]
    surface_pressure = _decode_surface_pressure(
        canonicalize_time_last_surface_field(lnsp)
    )
    specific_humidity_3d = canonicalize_time_last_level_field(
        _read_forcing(data_files, "q", where="model_level", flip_y=True)[..., 1:, :]
    )
    temperature_3d = canonicalize_time_last_level_field(
        _read_forcing(data_files, "t", where="model_level", flip_y=True)[..., 1:, :]
    )

    fields = {
        "surface_pressure": surface_pressure,
        "specific_humidity_3d": specific_humidity_3d,
        "temperature_3d": temperature_3d,
        "u_velocity": canonicalize_time_last_surface_field(
            _read_forcing(data_files, "u", where="model_level", flip_y=True)[:, :, 1, :]
        ),
        "v_velocity": canonicalize_time_last_surface_field(
            _read_forcing(data_files, "v", where="model_level", flip_y=True)[:, :, 1, :]
        ),
        "net_shortwave_radiation_flux": canonicalize_time_last_surface_field(
            _read_forcing(data_files, "msnswrf", where="surface", flip_y=True)
        ),
        "downward_longwave_radiation_flux": canonicalize_time_last_surface_field(
            _read_forcing(data_files, "msdwlwrf", where="surface", flip_y=True)
        ),
        "specific_humidity": specific_humidity_3d[:, 0, :, :],
        "temperature": temperature_3d[:, 0, :, :],
    }

    def initialize(component: DataComponent, context: ComponentInitContext) -> None:
        diagnostics = [
            _compute_monthly_diagnostics(
                context.settings,
                component.data["surface_pressure"][month_index],
                hyai,
                hybi,
                hyam,
                hybm,
                component.data["temperature_3d"][month_index],
                component.data["specific_humidity_3d"][month_index],
                component.data["temperature"][month_index],
            )
            for month_index in range(int(component.data["surface_pressure"].shape[0]))
        ]
        component.seed_fields(
            {
                "model_level_height": jnp.stack(
                    [item[0] for item in diagnostics],
                    axis=0,
                ),
                "density": jnp.stack([item[1] for item in diagnostics], axis=0),
                "potential_temperature": jnp.stack(
                    [item[2] for item in diagnostics],
                    axis=0,
                ),
            }
        )

    component = data_component(
        name=name,
        grid=grid,
        fields=fields,
        initialize=initialize,
    )
    component.declare_fields(outputs=_ERA5_ATMOSPHERE_FIELD_NAMES)
    component.update_settings(apply_time_interpolation=True)
    cast(Any, component).DATA_FILES = data_files
    cast(Any, component).hyai = hyai
    cast(Any, component).hybi = hybi
    cast(Any, component).hyam = hyam
    cast(Any, component).hybm = hybm
    return component
