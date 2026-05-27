"""CAMulator tensor indexing and xarray-to-Torch staging helpers."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal, Optional

import torch
import xarray as xr

from vercor.host_arrays import runtime_array_to_host
from vercor.types import RuntimeArray


@dataclass(frozen=True)
class TensorVariableIndex:
    """Typed channel metadata for one CAMulator tensor variable."""

    start_idx: int = 0
    end_idx: int = 0
    n_channels: int = 0
    is_3d: bool = False
    available: bool = True
    reason: str | None = None

    @classmethod
    def for_channels(
        cls,
        *,
        start_idx: int,
        n_channels: int,
        is_3d: bool,
    ) -> "TensorVariableIndex":
        """Return metadata for a variable present in a tensor."""

        return cls(
            start_idx=start_idx,
            end_idx=start_idx + n_channels,
            n_channels=n_channels,
            is_3d=is_3d,
            available=True,
        )

    @classmethod
    def unavailable(cls, *, reason: str) -> "TensorVariableIndex":
        """Return metadata for a known variable absent from a tensor."""

        return cls(available=False, reason=reason)

    @property
    def channel_slice(self) -> slice:
        """Return the channel slice occupied by this variable."""

        return slice(self.start_idx, self.end_idx)

    def require_available(self, *, tensor_type: str, var_name: str) -> None:
        """Raise a user-facing error if the variable is absent from the tensor."""

        if self.available:
            return
        raise ValueError(
            f"Variable '{var_name}' not available in '{tensor_type}' tensor. "
            f"Reason: {self.reason or 'Unknown'}"
        )

    def to_mapping(self) -> dict[str, Any]:
        """Return the legacy dictionary representation used by callers."""

        if not self.available:
            return {
                "available": False,
                "reason": self.reason or "Unknown",
            }
        return {
            "start_idx": self.start_idx,
            "end_idx": self.end_idx,
            "n_channels": self.n_channels,
            "is_3d": self.is_3d,
            "available": True,
        }


def _torch_tensor_from_jax_array(
    array: RuntimeArray,
    device: str,
    *,
    pin_memory: bool = False,
) -> torch.Tensor:
    """Transfer a JAX-compatible array through an explicit host-to-Torch boundary."""

    tensor = torch.as_tensor(runtime_array_to_host(array).copy())
    if pin_memory and device != "cpu" and torch.cuda.is_available():
        tensor = tensor.pin_memory()
    return tensor.to(device, non_blocking=True)


def _append_indexed_variables(
    indices: dict[str, TensorVariableIndex],
    variable_names: Sequence[str],
    *,
    start_index: int,
    n_channels: int,
    is_3d: bool,
) -> int:
    """Append available tensor-variable indices and return the next channel index."""

    idx = start_index
    for var in variable_names:
        indices[var] = TensorVariableIndex.for_channels(
            start_idx=idx,
            n_channels=n_channels,
            is_3d=is_3d,
        )
        idx += n_channels
    return idx


def _mark_unavailable_variables(
    indices: dict[str, TensorVariableIndex],
    variable_names: Sequence[str],
    *,
    reason: str,
) -> None:
    """Mark variables that are recognized by config but absent from a tensor type."""

    for var in variable_names:
        indices[var] = TensorVariableIndex.unavailable(reason=reason)


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

        self.var_indices: dict[str, dict[str, TensorVariableIndex]] = {}
        self._build_state_indices()
        self._build_input_indices()
        self._build_output_indices()

    def _build_state_indices(self) -> None:
        """Build indices for pure state tensors with no forcing or diagnostics."""

        indices: dict[str, TensorVariableIndex] = {}
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

        indices: dict[str, TensorVariableIndex] = {}
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

        indices: dict[str, TensorVariableIndex] = {}
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

    def get_var_index(self, var_name: str) -> TensorVariableIndex:
        """Return typed indexing metadata for a configured CAMulator variable."""

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

    def get_var_info(self, var_name: str) -> dict[str, Any]:
        """Return legacy dictionary metadata for a configured variable."""

        return self.get_var_index(var_name).to_mapping()

    def get_state_var(
        self,
        state_tensor: torch.Tensor,
        var_name: str,
        time_idx: Optional[int] = None,
    ) -> torch.Tensor:
        """Extract a named variable view from a CAMulator tensor."""

        index = self.get_var_index(var_name)
        index.require_available(tensor_type=self.tensor_type, var_name=var_name)

        var_slice = state_tensor[:, index.channel_slice, ...]  # noqa: E203
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

        index = self.get_var_index(var_name)
        index.require_available(tensor_type=self.tensor_type, var_name=var_name)

        expected_shape: tuple[int, ...]
        if time_idx is None:
            expected_shape = (
                state_tensor.shape[0],
                index.n_channels,
                state_tensor.shape[2],
                state_tensor.shape[3],
                state_tensor.shape[4],
            )
        else:
            expected_shape = (
                state_tensor.shape[0],
                index.n_channels,
                state_tensor.shape[3],
                state_tensor.shape[4],
            )
        if var_data.shape != expected_shape:
            raise ValueError(
                f"Shape mismatch for '{var_name}'. Expected {expected_shape}, got {var_data.shape}"
            )

        if time_idx is None:
            state_tensor[:, index.channel_slice, ...] = var_data  # noqa: E203
        else:
            state_tensor[:, index.channel_slice, time_idx, :, :] = (  # noqa: E203
                var_data
            )

    def list_available_vars(self) -> dict[str, dict[str, Any]]:
        """Return variables available in this accessor's tensor type."""

        return {
            var: info.to_mapping()
            for var, info in self.var_indices[self.tensor_type].items()
            if info.available
        }


__all__ = [
    "StateVariableAccessor",
    "TensorVariableIndex",
    "_append_indexed_variables",
    "_mark_unavailable_variables",
    "_prepare_static_forcing_tensor",
    "_torch_tensor_from_jax_array",
]
