from pathlib import Path
from typing import Optional

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from vercor.components import Component, LifecycleHooks, DataComponent, SetupResult
from vercor.dtypes import DTypePolicy, as_jax_real_array
from vercor.field_layout import (
    canonicalize_time_last_level_field,
    canonicalize_time_last_surface_field,
)
from vercor.fluxes.utilities import (
    compute_air_density,
    compute_potential_temperature,
)
from vercor.fluxes.vertical_coordinates import (
    compute_hybrid_pressure_levels,
    get_altitudes_hybrid_sigma_levels,
)
from vercor.grids import RectilinearGrid
from vercor.components import SetupContext
from vercor.forcing_data import read_forcing as _read_forcing
from vercor.output import OutputSpec
from vercor.physics import PhysicalConstants
from vercor.setups._data.assets import get_forcing_data
from vercor.setups._data._component_helpers import time_interpolated_data_component

_ERA5_ATMOSPHERE_FIELD_NAMES = (
    "surface_pressure",
    "specific_humidity_3d",
    "temperature_3d",
    "air_u_velocity",
    "air_v_velocity",
    "u_velocity_10m",
    "v_velocity_10m",
    "downward_shortwave_radiation_flux",
    "net_shortwave_radiation_flux",
    "downward_longwave_radiation_flux",
    "air_specific_humidity",
    "air_temperature",
    "model_level_height",
    "air_density",
    "air_potential_temperature",
    "rain_rate",
    "snow_rate",
)


def _decode_surface_pressure(lnsp: ArrayLike) -> jax.Array:
    """Convert log surface pressure to physical pressure in Pascals."""
    return jnp.exp(as_jax_real_array(lnsp))


def _compute_monthly_diagnostics(
    constants: PhysicalConstants,
    dtype: DTypePolicy,
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

    surface_pressure_array = as_jax_real_array(surface_pressure, dtype)
    temperature_3d_array = as_jax_real_array(temperature_3d, dtype).transpose((1, 2, 0))
    specific_humidity_3d_array = as_jax_real_array(
        specific_humidity_3d,
        dtype,
    ).transpose((1, 2, 0))
    temperature_array = as_jax_real_array(temperature, dtype)
    hyai_array = as_jax_real_array(hyai, dtype)
    hybi_array = as_jax_real_array(hybi, dtype)
    hyam_array = as_jax_real_array(hyam, dtype)
    hybm_array = as_jax_real_array(hybm, dtype)

    ph = compute_hybrid_pressure_levels(surface_pressure_array, hyai_array, hybi_array)
    pf = compute_hybrid_pressure_levels(surface_pressure_array, hyam_array, hybm_array)
    model_level_height = get_altitudes_hybrid_sigma_levels(
        constants,
        temperature_3d_array,
        specific_humidity_3d_array,
        ph,
    )[..., 1]
    density = compute_air_density(constants, pf[..., 0], temperature_array)
    potential_temperature = compute_potential_temperature(
        constants,
        temperature_array,
        pf[..., 0],
    )

    return model_level_height, density, potential_temperature


def make_era5_atmosphere(
    name: str = "ATM",
    model_level_file: Optional[Path] = None,
    surface_file: Optional[Path] = None,
    *,
    output: OutputSpec | None = None,
) -> DataComponent:
    """Return an ERA5 atmosphere prescribed forcing component with no physical feedback inputs."""

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

    # ECMWF rates are already water-equivalent kg/m²/s. Aggregate on the
    # source grid before the runtime selects months or regrids either phase.
    precipitation_fields = {
        output_name: canonicalize_time_last_surface_field(
            _read_forcing(data_files, first, where="surface", flip_y=True)
            + _read_forcing(data_files, second, where="surface", flip_y=True)
        )
        for output_name, first, second in (
            ("rain_rate", "crr", "lsrr"),
            ("snow_rate", "csfr", "lssfr"),
        )
    }

    fields = {
        "surface_pressure": surface_pressure,
        "specific_humidity_3d": specific_humidity_3d,
        "temperature_3d": temperature_3d,
        "u_velocity_10m": canonicalize_time_last_surface_field(
            _read_forcing(data_files, "u10", where="surface", flip_y=True)
        ),
        "v_velocity_10m": canonicalize_time_last_surface_field(
            _read_forcing(data_files, "v10", where="surface", flip_y=True)
        ),
        "downward_shortwave_radiation_flux": canonicalize_time_last_surface_field(
            _read_forcing(data_files, "msdwswrf", where="surface", flip_y=True)
        ),
        "air_u_velocity": canonicalize_time_last_surface_field(
            _read_forcing(data_files, "u", where="model_level", flip_y=True)[:, :, 1, :]
        ),
        "air_v_velocity": canonicalize_time_last_surface_field(
            _read_forcing(data_files, "v", where="model_level", flip_y=True)[:, :, 1, :]
        ),
        "net_shortwave_radiation_flux": canonicalize_time_last_surface_field(
            _read_forcing(data_files, "msnswrf", where="surface", flip_y=True)
        ),
        "downward_longwave_radiation_flux": canonicalize_time_last_surface_field(
            _read_forcing(data_files, "msdwlwrf", where="surface", flip_y=True)
        ),
        "air_specific_humidity": specific_humidity_3d[:, 0, :, :],
        "air_temperature": temperature_3d[:, 0, :, :],
        "rain_rate": jnp.zeros_like(surface_pressure),
        "snow_rate": jnp.zeros_like(surface_pressure),
    }

    def setup(component: Component, context: SetupContext) -> SetupResult:
        _ = component
        diagnostics = [
            _compute_monthly_diagnostics(
                context.constants,
                context.dtype,
                fields["surface_pressure"][month_index],
                hyai,
                hybi,
                hyam,
                hybm,
                fields["temperature_3d"][month_index],
                fields["specific_humidity_3d"][month_index],
                fields["air_temperature"][month_index],
            )
            for month_index in range(int(fields["surface_pressure"].shape[0]))
        ]
        return SetupResult(
            fields={
                **precipitation_fields,
                "model_level_height": jnp.stack(
                    [item[0] for item in diagnostics],
                    axis=0,
                ),
                "air_density": jnp.stack([item[1] for item in diagnostics], axis=0),
                "air_potential_temperature": jnp.stack(
                    [item[2] for item in diagnostics],
                    axis=0,
                ),
            }
        )

    component = time_interpolated_data_component(
        name=name,
        grid=grid,
        fields=fields,
        outputs=_ERA5_ATMOSPHERE_FIELD_NAMES,
        lifecycle=LifecycleHooks(setup=setup),
        output=output,
    )
    return component
