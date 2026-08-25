from __future__ import annotations

from collections.abc import Mapping
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
    """Configuration for the bundled Veros ocean setup factory."""

    name: str = "OCN"
    setup: Literal["global_4deg", "acc"] = "global_4deg"
    execution: Literal["host", "jax"] = "host"
    custom_parameters: Mapping[str, Any] | None = None
    restore_to_climatology: bool = False
    spinup: Spinup = field(default_factory=Spinup)
    output: OutputSpec = field(default_factory=OutputSpec)
    jitted: bool = False

    def __post_init__(self) -> None:
        """Validate execution policy and copy caller mappings into owned state."""

        if self.setup not in ("global_4deg", "acc"):
            raise ValueError("setup must be 'global_4deg' or 'acc'")
        if self.execution not in ("host", "jax"):
            raise ValueError("execution must be 'host' or 'jax'")

        if self.custom_parameters is not None:
            object.__setattr__(
                self,
                "custom_parameters",
                MappingProxyType(dict(self.custom_parameters)),
            )


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
]
