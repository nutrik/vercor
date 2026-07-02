"""CAMulator state transformation and model stepping helpers."""

from __future__ import annotations

from typing import Any

import torch

from vercor.jax_logging import get_default_logger
from vercor.setups.external import camulator_imports
from vercor.setups.external.camulator_wind_filter import post_process_wind_artifacts

logger = get_default_logger()


class CAMulatorStepper:
    """Core CAMulator time-stepper with optional post-processing fixers."""

    def __init__(
        self, model: torch.nn.Module, conf: dict[str, Any], device: torch.device
    ):
        """Initialize model, state helpers, and post-processing hooks."""

        self.model = model
        self.conf = conf
        self.device = device
        self.history_len = conf["data"]["history_len"]
        self.varnum_diag = len(conf["data"]["diagnostic_variables"])
        self.static_dim = (
            len(conf["data"]["static_variables"])
            if not conf["data"]["static_first"]
            else 0
        )
        self.static_first = conf["data"]["static_first"]
        self._setup_postprocessing()

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

    def _setup_postprocessing(self) -> None:
        """Initialize conservation fixers and wind filtering if available."""

        post_conf = self.conf["model"]["post_conf"]
        postblock_available = camulator_imports.load_postblock_modules()

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

    def _apply_postprocessing(
        self,
        prediction: torch.Tensor,
        model_input: torch.Tensor,
    ) -> torch.Tensor:
        """Apply wind artifact filtering and conservation fixers."""

        post_process_wind_artifacts(
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


__all__ = [
    "CAMulatorStepper",
]
