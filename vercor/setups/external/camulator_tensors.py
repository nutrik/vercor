"""CAMulator tensor indexing and xarray-to-Torch staging helpers."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal, Optional

import torch
import xarray as xr


def _append_indexed_variables(
    indices: dict[str, dict[str, Any]],
    variable_names: Sequence[str],
    *,
    start_index: int,
    n_channels: int,
    is_3d: bool,
) -> int:
    """Append available tensor-variable indices and return the next channel index."""

    idx = start_index
    for var in variable_names:
        indices[var] = {
            "start_idx": idx,
            "end_idx": idx + n_channels,
            "n_channels": n_channels,
            "is_3d": is_3d,
            "available": True,
        }
        idx += n_channels
    return idx


def _mark_unavailable_variables(
    indices: dict[str, dict[str, Any]],
    variable_names: Sequence[str],
    *,
    reason: str,
) -> None:
    """Mark variables that are recognized by config but absent from a tensor type."""

    for var in variable_names:
        indices[var] = {"available": False, "reason": reason}


def _prepare_static_forcing_tensor(
    forcing_ds: xr.Dataset,
    static_variables: list[str],
    device: Any,
) -> torch.Tensor:
    """Prepare static CAMulator forcing through an explicit xarray/Torch boundary."""

    static_values = forcing_ds[static_variables].to_array(dim="static_variable").values
    return (
        torch.as_tensor(static_values)
        .unsqueeze(0)
        .unsqueeze(2)
        .to(
            device,
            non_blocking=True,
        )
    )


class StateVariableAccessor:
    """Access variables by name in CAMulator state, input, and output tensors."""

    def __init__(
        self,
        conf: dict[str, Any],
        tensor_type: Literal["state", "input", "output"] = "state",
    ) -> None:
        """Initialize a variable accessor for the requested tensor type."""

        self.conf = conf
        self.tensor_type = tensor_type
        self.prognostic_vars = conf["data"]["variables"]
        self.surface_vars = conf["data"]["surface_variables"]
        self.diagnostic_vars = conf["data"]["diagnostic_variables"]
        self.dynamic_forcing_vars = conf["data"]["dynamic_forcing_variables"]
        self.forcing_vars = conf["data"]["forcing_variables"]
        self.static_vars = conf["data"]["static_variables"]
        self.levels = conf["model"]["levels"]
        self.static_first = conf["data"]["static_first"]
        self._build_index_maps()

    def _build_index_maps(self) -> None:
        """Build index mappings for every supported tensor type."""

        self.var_indices: dict[str, dict[str, dict[str, Any]]] = {}
        self._build_state_indices()
        self._build_input_indices()
        self._build_output_indices()

    def _build_state_indices(self) -> None:
        """Build indices for pure state tensors with no forcing or diagnostics."""

        indices: dict[str, dict[str, Any]] = {}
        idx = _append_indexed_variables(
            indices,
            self.prognostic_vars,
            start_index=0,
            n_channels=self.levels,
            is_3d=True,
        )
        _append_indexed_variables(
            indices,
            self.surface_vars,
            start_index=idx,
            n_channels=1,
            is_3d=False,
        )
        _mark_unavailable_variables(
            indices,
            self.diagnostic_vars,
            reason="Diagnostics not in state tensor",
        )
        _mark_unavailable_variables(
            indices,
            (*self.dynamic_forcing_vars, *self.forcing_vars, *self.static_vars),
            reason="Forcing not in state tensor",
        )
        self.var_indices["state"] = indices

    def _build_input_indices(self) -> None:
        """Build indices for model input tensors with forcing appended to state."""

        indices: dict[str, dict[str, Any]] = {}
        idx = _append_indexed_variables(
            indices,
            self.prognostic_vars,
            start_index=0,
            n_channels=self.levels,
            is_3d=True,
        )
        idx = _append_indexed_variables(
            indices,
            self.surface_vars,
            start_index=idx,
            n_channels=1,
            is_3d=False,
        )
        if self.static_first:
            forcing_order = (
                self.static_vars + self.dynamic_forcing_vars + self.forcing_vars
            )
        else:
            forcing_order = (
                self.dynamic_forcing_vars + self.forcing_vars + self.static_vars
            )
        _append_indexed_variables(
            indices,
            forcing_order,
            start_index=idx,
            n_channels=1,
            is_3d=False,
        )
        _mark_unavailable_variables(
            indices,
            self.diagnostic_vars,
            reason="Diagnostics not in input tensor",
        )
        self.var_indices["input"] = indices

    def _build_output_indices(self) -> None:
        """Build indices for model output tensors with diagnostics appended."""

        indices: dict[str, dict[str, Any]] = {}
        idx = _append_indexed_variables(
            indices,
            self.prognostic_vars,
            start_index=0,
            n_channels=self.levels,
            is_3d=True,
        )
        idx = _append_indexed_variables(
            indices,
            self.surface_vars,
            start_index=idx,
            n_channels=1,
            is_3d=False,
        )
        _append_indexed_variables(
            indices,
            self.diagnostic_vars,
            start_index=idx,
            n_channels=1,
            is_3d=False,
        )
        _mark_unavailable_variables(
            indices,
            (*self.dynamic_forcing_vars, *self.forcing_vars, *self.static_vars),
            reason="Forcing not in output tensor",
        )
        self.var_indices["output"] = indices

    def get_var_info(self, var_name: str) -> Any:
        """Return indexing metadata for a configured CAMulator variable."""

        indices = self.var_indices[self.tensor_type]
        if var_name not in indices:
            all_vars = (
                self.prognostic_vars
                + self.surface_vars
                + self.diagnostic_vars
                + self.dynamic_forcing_vars
                + self.forcing_vars
                + self.static_vars
            )
            raise ValueError(
                f"Variable '{var_name}' not found in config. Available variables: {all_vars}"
            )
        return indices[var_name]

    def get_state_var(
        self,
        state_tensor: torch.Tensor,
        var_name: str,
        time_idx: Optional[int] = None,
    ) -> torch.Tensor:
        """Extract a named variable view from a CAMulator tensor."""

        info = self.get_var_info(var_name)
        if not info["available"]:
            raise ValueError(
                f"Variable '{var_name}' not available in '{self.tensor_type}' tensor. "
                f"Reason: {info.get('reason', 'Unknown')}"
            )

        var_slice = state_tensor[
            :, info["start_idx"] : info["end_idx"], ...  # noqa: E203
        ]
        if time_idx is not None:
            if time_idx >= state_tensor.shape[2]:
                raise IndexError(
                    f"Time index {time_idx} out of bounds for tensor with {state_tensor.shape[2]} time steps"
                )
            var_slice = var_slice[:, :, time_idx, :, :]
        return var_slice

    def set_state_var(
        self,
        state_tensor: torch.Tensor,
        var_name: str,
        var_data: torch.Tensor,
        time_idx: Optional[int] = None,
    ) -> None:
        """Set a named variable in a CAMulator tensor in place."""

        info = self.get_var_info(var_name)
        if not info["available"]:
            raise ValueError(
                f"Variable '{var_name}' not available in '{self.tensor_type}' tensor. "
                f"Reason: {info.get('reason', 'Unknown')}"
            )

        expected_shape: tuple[int, ...]
        if time_idx is None:
            expected_shape = (
                state_tensor.shape[0],
                info["n_channels"],
                state_tensor.shape[2],
                state_tensor.shape[3],
                state_tensor.shape[4],
            )
        else:
            expected_shape = (
                state_tensor.shape[0],
                info["n_channels"],
                state_tensor.shape[3],
                state_tensor.shape[4],
            )
        if var_data.shape != expected_shape:
            raise ValueError(
                f"Shape mismatch for '{var_name}'. Expected {expected_shape}, got {var_data.shape}"
            )

        if time_idx is None:
            state_tensor[:, info["start_idx"] : info["end_idx"], ...] = (  # noqa: E203
                var_data
            )
        else:
            state_tensor[
                :, info["start_idx"] : info["end_idx"], time_idx, :, :  # noqa: E203
            ] = var_data

    def list_available_vars(self) -> dict[str, dict[str, Any]]:
        """Return variables available in this accessor's tensor type."""

        return {
            var: info
            for var, info in self.var_indices[self.tensor_type].items()
            if info.get("available", False)
        }


__all__ = [
    "StateVariableAccessor",
    "_append_indexed_variables",
    "_mark_unavailable_variables",
    "_prepare_static_forcing_tensor",
]
