from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import timedelta
from types import MappingProxyType
from typing import Any, Literal

from vercor.jax_logging import LoggerLike
from vercor.output import OutputSpec, PeriodOutput


@dataclass(frozen=True)
class Spinup:
    """Spinup policy shared by bundled setup factories."""

    enabled: bool = False
    duration: timedelta = timedelta(days=2)


@dataclass(frozen=True)
class JAXGCMConfig:
    """Configuration for the bundled JAXGCM/JCM atmosphere setup factory."""

    name: str = "ATM"
    custom_parameters: Mapping[str, float] | None = None
    model_timestep: timedelta = timedelta(minutes=30)
    save_interval: timedelta = timedelta(days=1)
    forcing_data: Any | None = None
    spinup: Spinup = field(default_factory=Spinup)
    output: OutputSpec = field(default_factory=OutputSpec)
    jitted: bool = True

    def __post_init__(self) -> None:
        """Copy mutable caller-provided mappings into owned config state."""

        if self.custom_parameters is not None:
            object.__setattr__(
                self,
                "custom_parameters",
                MappingProxyType(dict(self.custom_parameters)),
            )


@dataclass(frozen=True)
class VerosConfig:
    """Configuration for the bundled Veros ocean setup factory.

    For ``global_4deg``, ``restore_to_climatology_temp`` enables the SST
    correction and ``restore_to_climatology_salt`` independently adds 30-day
    SSS restoring to the coupled virtual salt flux. Both default to False
    and apply during spinup and coupled stepping in either execution lane.
    The ACC setup retains its native internal forcing. Explicit ``surface_forcing="coupled"``
    accepts only total sea-ice ocean fluxes on the global grid, keeps independent
    restoring, and requires native pressure/free-surface execution for real SSH.
    """

    name: str = "OCN"
    setup: Literal["global_4deg", "acc"] = "global_4deg"
    execution: Literal["host", "jax"] = "host"
    custom_parameters: Mapping[str, Any] | None = None
    restore_to_climatology_temp: bool = False
    restore_to_climatology_salt: bool = False
    spinup: Spinup = field(default_factory=Spinup)
    output: OutputSpec = field(default_factory=OutputSpec)
    surface_forcing: Literal["atmosphere", "coupled"] = field(
        default="atmosphere", kw_only=True
    )

    def __post_init__(self) -> None:
        """Validate execution policy and copy caller mappings into owned state."""

        if self.setup not in ("global_4deg", "acc"):
            raise ValueError("setup must be 'global_4deg' or 'acc'")
        if self.execution not in ("host", "jax"):
            raise ValueError("execution must be 'host' or 'jax'")

        if not isinstance(self.surface_forcing, str):
            raise TypeError("surface_forcing must be a string")
        if self.surface_forcing not in ("atmosphere", "coupled"):
            raise ValueError("surface_forcing must be 'atmosphere' or 'coupled'")
        if self.surface_forcing == "coupled":
            if self.setup != "global_4deg":
                raise ValueError("coupled surface_forcing requires global_4deg")
            if (self.custom_parameters or {}).get("enable_streamfunction", False):
                raise ValueError(
                    "coupled surface_forcing requires enable_streamfunction=False for native SSH"
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
    """

    name: str = "CICE"
    model_timestep: timedelta = timedelta(hours=1)
    custom_parameters: Mapping[str, object] | None = None
    physical_constants: Mapping[str, object] | None = field(default=None, kw_only=True)
    initial_conditions: Mapping[str, object] | None = None
    output: OutputSpec = field(default_factory=OutputSpec)

    def __post_init__(self) -> None:
        """Validate the native clock and copy caller-owned mappings/arrays."""
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


@dataclass(frozen=True)
class CAMulatorConfig:
    """Configuration for the bundled CAMulator atmosphere setup factory."""

    config_path: str
    name: str = "ATM"
    model_weights_path: str = "checkpoint.pt00091.pt"
    init_noise: float | None = None
    spinup: Spinup = field(default_factory=Spinup)
    output: OutputSpec = field(default_factory=OutputSpec)
    device: str = "cuda"
    time_alignment: Literal["strict", "forcing_start"] = "strict"
    logger: LoggerLike | None = None

    def __post_init__(self) -> None:
        """Validate CAMulator forcing-time alignment policy."""

        if self.time_alignment not in ("strict", "forcing_start"):
            raise ValueError("time_alignment must be 'strict' or 'forcing_start'")


def _default_jcm_atmosphere_config() -> JAXGCMConfig:
    """Return the historic defaults for the paired JCM setup factory."""

    return JAXGCMConfig(
        spinup=Spinup(enabled=True),
        output=OutputSpec(period=PeriodOutput(frequency="month")),
        jitted=True,
    )


@dataclass(frozen=True)
class JCMLandAtmosphereConfig:
    """Configuration for the bundled paired JCM land/atmosphere setup."""

    atmosphere: JAXGCMConfig = field(default_factory=_default_jcm_atmosphere_config)
    land_name: str = "LND"
    land_output: OutputSpec = field(default_factory=OutputSpec)

    def __post_init__(self) -> None:
        """Validate paired setup component names."""

        if not isinstance(self.land_name, str) or not self.land_name:
            raise ValueError("land_name must be a non-empty string")


__all__ = [
    "CAMulatorConfig",
    "JAXGCMConfig",
    "JCMLandAtmosphereConfig",
    "Spinup",
    "VerosConfig",
    "VerisConfig",
]
