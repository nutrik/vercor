"""Public CAMulator wind artifact filtering facade."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch

from vercor.jax_logging import get_default_logger
import vercor.setups.external._camulator_wind_filtering as _wind_filtering


@dataclass
class WindArtifactFilterConfig:
    """Configuration for CAMulator wind artifact post-processing."""

    activate: bool = True
    mask_level: int = 14
    target_levels: Sequence[int] = tuple(range(9, 21))
    target_vars: Sequence[str] = ("U", "V", "T", "Qtot")
    speed_threshold: float = 3.0193274566643846
    smooth_sigma: float = 1.0
    dilation_zonal: int = 13
    dilation_meridional: int = 5
    falloff_sigma: float = 4.0

    def validate(self) -> None:
        """Validate configuration values before runtime post-processing."""

        if self.dilation_zonal <= 0 or self.dilation_meridional <= 0:
            raise ValueError("Dilations must be positive")
        if self.smooth_sigma <= 0 or self.falloff_sigma <= 0:
            raise ValueError("Sigmas must be positive")
        if not isinstance(self.target_levels, Sequence):
            raise ValueError("target_levels must be a sequence")
        if not isinstance(self.target_vars, Sequence):
            raise ValueError("target_vars must be a sequence")


def load_wind_filter_config(conf: dict[str, Any]) -> WindArtifactFilterConfig:
    """Return wind-filter configuration from the optional CAMulator config block."""

    raw = conf.get("postprocessing", {}).get("wind_artifact_filter", {})
    cfg = WindArtifactFilterConfig(
        activate=raw.get("activate", True),
        mask_level=raw.get("mask_level", 14),
        target_levels=tuple(raw.get("target_levels", list(range(9, 21)))),
        target_vars=tuple(raw.get("target_vars", ["U", "V", "T", "Qtot"])),
        speed_threshold=raw.get("speed_threshold", 3.0193274566643846),
        smooth_sigma=raw.get("smooth_sigma", 1.0),
        dilation_zonal=raw.get("dilation_zonal", 13),
        dilation_meridional=raw.get("dilation_meridional", 5),
        falloff_sigma=raw.get("falloff_sigma", 4.0),
    )
    cfg.validate()
    return cfg


def wind_filter(
    field: torch.Tensor,
    gaussian_2d: torch.Tensor,
    kernel_size: int,
    smooth_blend_mask: torch.Tensor,
) -> torch.Tensor:
    """Apply wind filtering to a 2D field or conv2d-compatible tensor."""

    return _wind_filtering.filter_field(
        field,
        gaussian_2d,
        kernel_size,
        smooth_blend_mask,
    )


def post_process_wind_artifacts(
    x: torch.Tensor, conf: dict[str, Any], enable_filtering: bool = True
) -> None:
    """Apply CAMulator wind artifact filtering in place when configured."""

    if not enable_filtering:
        return

    try:
        wf_cfg = load_wind_filter_config(conf)
        if not wf_cfg.activate:
            return

        apply_wind_artifact_filter_to_tensor(
            x=x,
            varname_upper=conf["data"]["variables"],
            levels_per_var=conf["model"]["levels"],
            mask_level=wf_cfg.mask_level,
            target_levels=wf_cfg.target_levels,
            target_vars=wf_cfg.target_vars,
            speed_threshold=wf_cfg.speed_threshold,
            smooth_sigma=wf_cfg.smooth_sigma,
            dilation_zonal=wf_cfg.dilation_zonal,
            dilation_meridional=wf_cfg.dilation_meridional,
            falloff_sigma=wf_cfg.falloff_sigma,
        )
    except Exception as e:
        get_default_logger().warning(f"Wind artifact filtering failed: {e}")


def apply_wind_artifact_filter_to_tensor(
    x: torch.Tensor,
    varname_upper: list[str],
    levels_per_var: int,
    mask_level: int = 14,
    target_levels: Sequence[int] | None = None,
    target_vars: Sequence[str] | None = None,
    speed_threshold: float = 3.0193274566643846,
    smooth_sigma: float = 1.5,
    dilation_zonal: int = 15,
    dilation_meridional: int = 5,
    falloff_sigma: float = 4.0,
) -> None:
    """Apply CAMulator wind artifact filtering to selected tensor channels."""

    _wind_filtering.apply_wind_filter_to_tensor(
        x=x,
        varname_upper=varname_upper,
        levels_per_var=levels_per_var,
        mask_level=mask_level,
        target_levels=target_levels,
        target_vars=target_vars,
        speed_threshold=speed_threshold,
        smooth_sigma=smooth_sigma,
        dilation_zonal=dilation_zonal,
        dilation_meridional=dilation_meridional,
        falloff_sigma=falloff_sigma,
        logger=get_default_logger(),
    )


def simple_wind_artifact_filter(
    u_wind: torch.Tensor,
    v_wind: torch.Tensor,
    speed_threshold: float = 25.0,
    smooth_sigma: float = 2.0,
    dilation_zonal: int = 9,
    dilation_meridional: int = 3,
    falloff_sigma: float = 3.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, torch.Tensor]:
    """Return filtered winds plus reusable kernel and blend-mask artifacts."""

    artifacts = _wind_filtering.build_wind_filter_artifacts(
        u_wind,
        v_wind,
        speed_threshold=speed_threshold,
        smooth_sigma=smooth_sigma,
        dilation_zonal=dilation_zonal,
        dilation_meridional=dilation_meridional,
        falloff_sigma=falloff_sigma,
    )
    return (
        artifacts.u_filtered,
        artifacts.v_filtered,
        artifacts.gaussian_2d,
        artifacts.kernel_size,
        artifacts.smooth_blend_mask,
    )


__all__ = [
    "WindArtifactFilterConfig",
    "apply_wind_artifact_filter_to_tensor",
    "load_wind_filter_config",
    "post_process_wind_artifacts",
    "simple_wind_artifact_filter",
    "wind_filter",
]
