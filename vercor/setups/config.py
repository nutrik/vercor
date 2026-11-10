from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import timedelta
from math import isfinite
from numbers import Real
from types import MappingProxyType
from typing import Any, Literal

from vercor.output import OutputSpec, PeriodOutput


@dataclass(frozen=True)
class Spinup:
    """Spinup policy shared by bundled setup factories."""

    enabled: bool = False
    duration: timedelta = timedelta(days=2)


@dataclass(frozen=True)
class JCMAtmosphereConfig:
    """Configuration for the bundled JAXGCM/JCM atmosphere setup factory.

    Exchanged fluxes integrate every native step and evolving air fields use
    the endpoint. ``save_interval`` remains a validated compatibility setting;
    it cannot thin exchange samples. ``output`` owns file cadence and selection.
    """

    name: str = "ATM"
    custom_parameters: Mapping[str, float] | None = None
    model_timestep: timedelta = timedelta(minutes=30)
    save_interval: timedelta = timedelta(days=1)
    forcing_data: Any | None = None
    spinup: Spinup = field(default_factory=Spinup)
    output: OutputSpec = field(default_factory=OutputSpec)
    jitted: bool = True
    seaice_coupling: bool = field(default=False, kw_only=True)

    def __post_init__(self) -> None:
        """Copy mutable caller-provided mappings into owned config state."""

        if not isinstance(self.seaice_coupling, bool):
            raise TypeError("seaice_coupling must be bool")
        if self.custom_parameters is not None:
            object.__setattr__(
                self,
                "custom_parameters",
                MappingProxyType(dict(self.custom_parameters)),
            )


@dataclass(frozen=True)
class VerosConfig:
    """Configuration for the bundled Veros ocean setup factory.

    The ``global_4deg`` ocean's ``restore_to_climatology_temp`` enables the SST
    correction (off by default) and ``restore_to_climatology_salt`` independently
    adds SSS restoring to the coupled virtual salt flux (on by default).
    ``salinity_restoring_timescale`` is a positive finite duration in seconds,
    defaulting to four 365-day model years (126144000 s). These controls apply
    during spinup and coupled stepping in either execution lane.
    Explicit ``surface_forcing="surface_fluxes"``
    accepts only total sea-ice ocean fluxes on the global grid, keeps independent
    restoring, and requires native pressure/free-surface execution for real SSH.
    Its default momentum and tracer timesteps are both 1800 seconds so physical
    interval stress and tracer fluxes advance over the same duration. An explicit
    ``custom_parameters["dt_mom"]`` also sets the implicit tracer timestep;
    explicit ``dt_tracer`` preserves a deliberately accelerated native policy.
    ``backend="jax"`` selects the native numerical kernels by default; host
    orchestration works with released Veros 1.7.0. Explicit ``backend="numpy"``
    retains NumPy kernels. ``execution="jax"`` requires the differentiable
    fork and native JAX; explicit None retains execution-based selection.
    """

    name: str = "OCN"
    execution: Literal["host", "jax"] = "host"
    custom_parameters: Mapping[str, Any] | None = None
    restore_to_climatology_temp: bool = False
    restore_to_climatology_salt: bool = True
    spinup: Spinup = field(default_factory=Spinup)
    output: OutputSpec = field(default_factory=OutputSpec)
    surface_forcing: Literal["bulk_meteorology", "surface_fluxes"] = field(
        default="bulk_meteorology", kw_only=True
    )

    export_open_water_surface: bool = field(default=False, kw_only=True)
    backend: Literal["numpy", "jax"] | None = field(default="jax", kw_only=True)
    salinity_restoring_timescale: float = field(default=4 * 365 * 86400.0, kw_only=True)

    def __post_init__(self) -> None:
        """Validate execution policy and copy caller mappings into owned state."""

        if isinstance(self.salinity_restoring_timescale, bool) or not isinstance(
            self.salinity_restoring_timescale, Real
        ):
            raise TypeError(
                "salinity_restoring_timescale must be a real number in seconds"
            )
        if (
            not isfinite(self.salinity_restoring_timescale)
            or self.salinity_restoring_timescale <= 0
        ):
            raise ValueError("salinity_restoring_timescale must be positive and finite")
        if not isinstance(self.export_open_water_surface, bool):
            raise TypeError("export_open_water_surface must be bool")
        if self.export_open_water_surface and self.surface_forcing != "surface_fluxes":
            raise ValueError(
                "export_open_water_surface requires surface_fluxes surface_forcing"
            )
        if self.execution not in ("host", "jax"):
            raise ValueError("execution must be 'host' or 'jax'")
        if self.backend not in (None, "numpy", "jax"):
            raise ValueError("backend must be 'numpy', 'jax', or None")
        if self.execution == "jax" and self.backend == "numpy":
            raise ValueError("JAX execution requires the native JAX backend")

        if not isinstance(self.surface_forcing, str):
            raise TypeError("surface_forcing must be a string")
        if self.surface_forcing not in ("bulk_meteorology", "surface_fluxes"):
            raise ValueError(
                "surface_forcing must be 'bulk_meteorology' or 'surface_fluxes'"
            )
        if self.surface_forcing == "surface_fluxes":
            if (self.custom_parameters or {}).get("enable_streamfunction", False):
                raise ValueError(
                    "surface_fluxes surface_forcing requires enable_streamfunction=False for native SSH"
                )

        if self.custom_parameters is not None:
            object.__setattr__(
                self,
                "custom_parameters",
                MappingProxyType(dict(self.custom_parameters)),
            )


@dataclass(frozen=True)
class VerisConfig:
    """Standalone sea-ice policy; runtime dtype/grid and coupled clock are owned by VerCOR.

    Native initial conditions use physical (x, y) arrays without halos, with
    hIceMean/hSnowMean in grid-cell volume-equivalent metres and Area in [0, 1].
    Numerical configuration is static. Native physical constants, evolving state
    and prescribed forcing are differentiable runtime payload inputs.
    ``atmosphere_forcing="model_level"`` derives actual 10 m winds through the
    shared CESM solve and absorbs raw shortwave with the open-water albedo.
    """

    name: str = "CICE"
    model_timestep: timedelta = timedelta(hours=1)
    custom_parameters: Mapping[str, object] | None = None
    physical_constants: Mapping[str, object] | None = field(default=None, kw_only=True)
    initial_conditions: Mapping[str, object] | None = None
    output: OutputSpec = field(default_factory=OutputSpec)
    atmosphere_forcing: Literal["observed_10m", "model_level"] = field(
        default="observed_10m", kw_only=True
    )

    def __post_init__(self) -> None:
        """Validate the native clock and copy caller-owned mappings/arrays."""
        if not isinstance(self.atmosphere_forcing, str):
            raise TypeError("atmosphere_forcing must be a string")
        if self.atmosphere_forcing not in ("observed_10m", "model_level"):
            raise ValueError(
                "atmosphere_forcing must be 'observed_10m' or 'model_level'"
            )
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("name must be a non-empty string")
        if not isinstance(self.model_timestep, timedelta):
            raise TypeError("model_timestep must be a timedelta")
        if self.model_timestep.total_seconds() <= 0:
            raise ValueError("model_timestep must be positive")
        if not isinstance(self.output, OutputSpec):
            raise TypeError("output must be OutputSpec")
        for name in ("custom_parameters", "physical_constants", "initial_conditions"):
            supplied = getattr(self, name)
            if supplied is not None:
                owned = deepcopy(dict(supplied))
                for value in owned.values():
                    setflags = getattr(value, "setflags", None)
                    if callable(setflags):
                        setflags(write=False)
                object.__setattr__(self, name, MappingProxyType(owned))


def _default_jcm_atmosphere_config() -> JCMAtmosphereConfig:
    """Return the historic defaults for the paired JCM setup factory."""

    return JCMAtmosphereConfig(
        spinup=Spinup(enabled=True),
        output=OutputSpec(period=PeriodOutput(frequency="month")),
        jitted=True,
    )


@dataclass(frozen=True)
class JCMLandAtmosphereConfig:
    """Configuration for the bundled paired JCM land/atmosphere setup."""

    atmosphere: JCMAtmosphereConfig = field(
        default_factory=_default_jcm_atmosphere_config
    )
    land_name: str = "LND"
    land_output: OutputSpec = field(default_factory=OutputSpec)

    def __post_init__(self) -> None:
        """Validate paired setup component names."""

        if not isinstance(self.land_name, str) or not self.land_name:
            raise ValueError("land_name must be a non-empty string")


__all__ = [
    "JCMAtmosphereConfig",
    "JCMLandAtmosphereConfig",
    "Spinup",
    "VerosConfig",
    "VerisConfig",
]
