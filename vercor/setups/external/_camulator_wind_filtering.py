"""Private CAMulator wind artifact tensor filtering mechanics."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from vercor.jax_logging import LoggerLike


@dataclass(frozen=True)
class WindFilterArtifacts:
    """Reusable mask and kernel artifacts for CAMulator wind filtering."""

    u_filtered: torch.Tensor
    v_filtered: torch.Tensor
    gaussian_2d: torch.Tensor
    kernel_size: int
    smooth_blend_mask: torch.Tensor


def filter_field(
    field: torch.Tensor,
    gaussian_2d: torch.Tensor,
    kernel_size: int,
    smooth_blend_mask: torch.Tensor,
) -> torch.Tensor:
    """Return ``field`` smoothed through an existing Gaussian/blend mask pair."""

    if field.dim() == 2:
        field = field.unsqueeze(0).unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False

    if gaussian_2d.dim() == 2:
        gaussian_2d = gaussian_2d.unsqueeze(0).unsqueeze(0)
    elif gaussian_2d.dim() == 3:
        gaussian_2d = gaussian_2d.unsqueeze(1)

    field_smooth = F.conv2d(field, gaussian_2d, padding=kernel_size // 2)
    field_filtered: torch.Tensor = (
        smooth_blend_mask * field_smooth + (1 - smooth_blend_mask) * field
    )

    if squeeze_output:
        return field_filtered.squeeze()
    return field_filtered


def build_wind_filter_artifacts(
    u_wind: torch.Tensor,
    v_wind: torch.Tensor,
    speed_threshold: float = 25.0,
    smooth_sigma: float = 2.0,
    dilation_zonal: int = 9,
    dilation_meridional: int = 3,
    falloff_sigma: float = 3.0,
) -> WindFilterArtifacts:
    """Build filtered winds, smoothing kernel, and blend mask for reuse."""

    if u_wind.dim() == 2:
        u_wind = u_wind.unsqueeze(0).unsqueeze(0)
        v_wind = v_wind.unsqueeze(0).unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False

    device = u_wind.device
    dtype = u_wind.dtype

    wind_speed = torch.sqrt(u_wind**2 + v_wind**2)
    high_speed_mask = wind_speed > speed_threshold
    mask_float = high_speed_mask.float().to(dtype=dtype)

    dilation_kernel = torch.ones(
        1, 1, dilation_meridional, dilation_zonal, device=device, dtype=dtype
    )
    expanded_mask_float = F.conv2d(
        mask_float,
        dilation_kernel,
        padding=(dilation_meridional // 2, dilation_zonal // 2),
    )
    expanded_mask_float = torch.clamp(expanded_mask_float, 0, 1)

    falloff_kernel_size_lat = _odd_kernel_size(2 * falloff_sigma * 2 + 1)
    falloff_kernel_size_lon = _odd_kernel_size(2 * falloff_sigma * 4 + 1)
    gaussian_2d_falloff = _anisotropic_gaussian_kernel(
        falloff_kernel_size_lat,
        falloff_kernel_size_lon,
        falloff_sigma,
        dtype=dtype,
        device=device,
    )
    smooth_blend_mask = F.conv2d(
        expanded_mask_float,
        gaussian_2d_falloff,
        padding=(falloff_kernel_size_lat // 2, falloff_kernel_size_lon // 2),
    )

    kernel_size = _odd_kernel_size(2 * smooth_sigma * 3 + 1)
    gaussian_2d = _isotropic_gaussian_kernel(
        kernel_size,
        smooth_sigma,
        dtype=dtype,
        device=device,
    )
    u_filtered = filter_field(u_wind, gaussian_2d, kernel_size, smooth_blend_mask)
    v_filtered = filter_field(v_wind, gaussian_2d, kernel_size, smooth_blend_mask)

    if squeeze_output:
        return WindFilterArtifacts(
            u_filtered=u_filtered.squeeze(),
            v_filtered=v_filtered.squeeze(),
            gaussian_2d=gaussian_2d.squeeze(),
            kernel_size=kernel_size,
            smooth_blend_mask=smooth_blend_mask,
        )
    return WindFilterArtifacts(
        u_filtered=u_filtered,
        v_filtered=v_filtered,
        gaussian_2d=gaussian_2d,
        kernel_size=kernel_size,
        smooth_blend_mask=smooth_blend_mask,
    )


def apply_wind_filter_to_tensor(
    x: torch.Tensor,
    varname_upper: Sequence[str],
    levels_per_var: int,
    mask_level: int = 14,
    target_levels: Sequence[int] | None = None,
    target_vars: Sequence[str] | None = None,
    speed_threshold: float = 3.0193274566643846,
    smooth_sigma: float = 1.5,
    dilation_zonal: int = 15,
    dilation_meridional: int = 5,
    falloff_sigma: float = 4.0,
    *,
    logger: LoggerLike | None = None,
) -> None:
    """Apply wind artifact filtering to selected variable levels in place."""

    selected_target_levels = range(10, 20) if target_levels is None else target_levels
    selected_target_vars = ("U", "V", "T", "Q") if target_vars is None else target_vars
    var_dict = _split_tensor_variables(x, varname_upper, levels_per_var)

    if "U" not in var_dict or "V" not in var_dict:
        raise ValueError("U and V winds required for mask calculation")

    artifacts = build_wind_filter_artifacts(
        var_dict["U"][:, mask_level, :, :].squeeze(),
        var_dict["V"][:, mask_level, :, :].squeeze(),
        speed_threshold=speed_threshold,
        smooth_sigma=smooth_sigma,
        dilation_zonal=dilation_zonal,
        dilation_meridional=dilation_meridional,
        falloff_sigma=falloff_sigma,
    )

    for var_name in selected_target_vars:
        if var_name not in var_dict:
            _warn(logger, f"{var_name} not found, skipping")
            continue

        var_idx = list(varname_upper).index(var_name)
        start_idx = var_idx * levels_per_var

        for level in selected_target_levels:
            if level >= var_dict[var_name].shape[1]:
                _warn(logger, f"Level {level} exceeds available levels for {var_name}")
                continue

            level_slice = var_dict[var_name][:, level, :, :].squeeze()
            filtered_slice = filter_field(
                level_slice,
                artifacts.gaussian_2d,
                artifacts.kernel_size,
                artifacts.smooth_blend_mask,
            )
            x[:, start_idx + level, :, :] = filtered_slice.unsqueeze(0)


def _split_tensor_variables(
    x: torch.Tensor,
    varname_upper: Sequence[str],
    levels_per_var: int,
) -> dict[str, torch.Tensor]:
    """Return variable views into a CAMulator level-major tensor."""

    variables = {}
    for idx, var_name in enumerate(varname_upper):
        start = idx * levels_per_var
        end = (idx + 1) * levels_per_var
        variables[var_name] = x[:, start:end, :, :]
    return variables


def _odd_kernel_size(raw_size: float) -> int:
    """Return the nearest odd integer size not smaller than ``raw_size``."""

    kernel_size = int(raw_size)
    if kernel_size % 2 == 0:
        kernel_size += 1
    return kernel_size


def _anisotropic_gaussian_kernel(
    size_lat: int,
    size_lon: int,
    sigma: float,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Return a 4D anisotropic Gaussian kernel for conv2d falloff."""

    x_lat = torch.arange(size_lat, dtype=dtype, device=device)
    x_lat = x_lat - size_lat // 2
    gaussian_1d_lat = torch.exp(-0.5 * (x_lat / sigma) ** 2)
    gaussian_1d_lat = gaussian_1d_lat / gaussian_1d_lat.sum()

    x_lon = torch.arange(size_lon, dtype=dtype, device=device)
    x_lon = x_lon - size_lon // 2
    gaussian_1d_lon = torch.exp(-0.5 * (x_lon / (sigma * 2)) ** 2)
    gaussian_1d_lon = gaussian_1d_lon / gaussian_1d_lon.sum()

    gaussian_2d = gaussian_1d_lat.unsqueeze(1) * gaussian_1d_lon.unsqueeze(0)
    return gaussian_2d.unsqueeze(0).unsqueeze(0)


def _isotropic_gaussian_kernel(
    kernel_size: int,
    sigma: float,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Return a 4D isotropic Gaussian kernel for conv2d smoothing."""

    x = torch.arange(kernel_size, dtype=dtype, device=device)
    x = x - kernel_size // 2
    gaussian_1d = torch.exp(-0.5 * (x / sigma) ** 2)
    gaussian_1d = gaussian_1d / gaussian_1d.sum()
    gaussian_2d = gaussian_1d.unsqueeze(0) * gaussian_1d.unsqueeze(1)
    return gaussian_2d.unsqueeze(0).unsqueeze(0)


def _warn(logger: LoggerLike | None, message: str) -> None:
    """Emit an optional warning without coupling tensor mechanics to logging setup."""

    if logger is not None:
        logger.warning(message)


__all__ = [
    "WindFilterArtifacts",
    "apply_wind_filter_to_tensor",
    "build_wind_filter_artifacts",
    "filter_field",
]
