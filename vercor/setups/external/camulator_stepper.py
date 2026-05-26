"""CAMulator state transformation and model stepping helpers."""

from __future__ import annotations

from typing import Any, Literal, Optional

import torch

from vercor.jax_logging import get_default_logger
from vercor.setups.external import camulator_imports
from vercor.setups.external.camulator_tensors import StateVariableAccessor

logger = get_default_logger()


class StateManager:
    """Manage CAMulator state tensor shifts and forcing concatenation."""

    def __init__(self, conf: dict[str, Any]) -> None:
        """Initialize tensor-shape metadata from a CAMulator configuration."""

        self.conf = conf
        self.history_len = conf["data"]["history_len"]
        self.varnum_diag = len(conf["data"]["diagnostic_variables"])
        self.static_dim = (
            len(conf["data"]["static_variables"])
            if not conf["data"]["static_first"]
            else 0
        )
        self.static_first = conf["data"]["static_first"]

    def shift_state_forward(
        self,
        state: torch.Tensor,
        prediction: torch.Tensor,
    ) -> torch.Tensor:
        """Roll the state tensor forward by one timestep."""

        if self.history_len == 1:
            if self.varnum_diag > 0:
                return prediction[:, : -self.varnum_diag, ...].detach()
            return prediction.detach()

        if self.static_dim == 0:
            state_detach = state[:, :, 1:, ...].detach()
        else:
            state_detach = state[:, : -self.static_dim, 1:, ...].detach()

        if self.varnum_diag > 0:
            new_pred = prediction[:, : -self.varnum_diag, ...].detach()
        else:
            new_pred = prediction.detach()

        return torch.cat([state_detach, new_pred], dim=2)

    def build_input_with_forcing(
        self,
        state: torch.Tensor,
        dynamic_forcing: torch.Tensor,
        static_forcing: torch.Tensor,
    ) -> torch.Tensor:
        """Combine state with dynamic and static forcing variables."""

        if self.static_first:
            forcing = torch.cat((static_forcing, dynamic_forcing), dim=1)
        else:
            forcing = torch.cat((dynamic_forcing, static_forcing), dim=1)

        return torch.cat((state, forcing), dim=1)


class CAMulatorStepper:
    """Core CAMulator time-stepper with optional post-processing fixers."""

    def __init__(
        self, model: torch.nn.Module, conf: dict[str, Any], device: torch.device
    ):
        """Initialize model, state helpers, and post-processing hooks."""

        self.model = model
        self.conf = conf
        self.device = device
        self.state_manager = StateManager(conf)
        self.state_accessor = StateVariableAccessor(conf, tensor_type="state")
        self.input_accessor = StateVariableAccessor(conf, tensor_type="input")
        self.output_accessor = StateVariableAccessor(conf, tensor_type="output")
        self._setup_postprocessing()

    def _setup_postprocessing(self) -> None:
        """Initialize conservation fixers and wind filtering if available."""

        post_conf = self.conf["model"]["post_conf"]
        postblock_available = camulator_imports._load_postblock_modules()
        windpp_available = camulator_imports._load_windpp_module()

        self.flag_mass = (
            postblock_available
            and post_conf["activate"]
            and post_conf["global_mass_fixer"]["activate"]
        )
        self.flag_water = (
            postblock_available
            and post_conf["activate"]
            and post_conf["global_water_fixer"]["activate"]
        )
        self.flag_energy = (
            postblock_available
            and post_conf["activate"]
            and post_conf["global_energy_fixer"]["activate"]
        )

        if self.flag_mass:
            self.opt_mass = camulator_imports.GlobalMassFixer(post_conf)
            logger.info("Global mass fixer initialized")
        if self.flag_water:
            self.opt_water = camulator_imports.GlobalWaterFixer(post_conf)
            logger.info("Global water fixer initialized")
        if self.flag_energy:
            self.opt_energy = camulator_imports.GlobalEnergyFixer(post_conf)
            logger.info("Global energy fixer initialized")

        self.enable_wind_filtering = windpp_available

    def step(
        self,
        state: torch.Tensor,
        dynamic_forcing: torch.Tensor,
        static_forcing: torch.Tensor,
    ) -> torch.Tensor:
        """Advance the atmospheric state by one CAMulator model timestep."""

        model_input = self.state_manager.build_input_with_forcing(
            state,
            dynamic_forcing,
            static_forcing,
        )
        with torch.no_grad():
            prediction = self.model(model_input.float())
        return self._apply_postprocessing(prediction, model_input)

    def _apply_postprocessing(
        self,
        prediction: torch.Tensor,
        model_input: torch.Tensor,
    ) -> torch.Tensor:
        """Apply wind artifact filtering and conservation fixers."""

        if self.enable_wind_filtering:
            camulator_imports.post_process_wind_artifacts(
                prediction,
                self.conf,
                enable_filtering=True,
            )

        if self.flag_mass:
            prediction = self.opt_mass({"y_pred": prediction, "x": model_input})[
                "y_pred"
            ]
        if self.flag_water:
            prediction = self.opt_water({"y_pred": prediction, "x": model_input})[
                "y_pred"
            ]
        if self.flag_energy:
            prediction = self.opt_energy({"y_pred": prediction, "x": model_input})[
                "y_pred"
            ]
        return prediction

    def get_state_var(
        self,
        tensor: torch.Tensor,
        var_name: str,
        tensor_type: Literal["state", "input", "output"] = "state",
        time_idx: Optional[int] = None,
    ) -> torch.Tensor:
        """Return a named variable from a state, input, or output tensor."""

        accessor = {
            "state": self.state_accessor,
            "input": self.input_accessor,
            "output": self.output_accessor,
        }[tensor_type]
        return accessor.get_state_var(tensor, var_name, time_idx)

    def set_state_var(
        self,
        tensor: torch.Tensor,
        var_name: str,
        var_data: torch.Tensor,
        tensor_type: Literal["state", "input", "output"] = "state",
        time_idx: Optional[int] = None,
    ) -> None:
        """Set a named variable in a state, input, or output tensor."""

        accessor = {
            "state": self.state_accessor,
            "input": self.input_accessor,
            "output": self.output_accessor,
        }[tensor_type]
        accessor.set_state_var(tensor, var_name, var_data, time_idx)


__all__ = [
    "CAMulatorStepper",
    "StateManager",
]
