"""CAMulator host-runtime atmosphere component factory."""

from __future__ import annotations

from datetime import timedelta
from functools import partial
from typing import Optional

from vercor.components import HostRuntimeComponent
from vercor.jax_logging import LoggerLike
import vercor.setups.external.camulator_contracts as _camulator_contracts
import vercor.setups.external.camulator_runtime as _camulator_runtime
from vercor.setups.external.camulator_gcm_state import CAMulatorGCMSetupState


def make_camulator_gcm(
    config_path: str,
    name: str = "ATM",
    model_weights_path: str = "checkpoint.pt00091.pt",
    output_subfolder_name: Optional[str] = None,
    init_noise: Optional[float] = None,
    spinup_time: timedelta = timedelta(days=2),
    do_spinup: bool = False,
    device: str = "cuda",
    output_cpus_number: int = 8,
    output_frequency: str | None = None,
    logger: LoggerLike | None = None,
) -> HostRuntimeComponent:
    """Return a host-backed CAMulator atmosphere component."""

    state = CAMulatorGCMSetupState(
        config_path=config_path,
        name=name,
        model_weights_path=model_weights_path,
        output_subfolder_name=output_subfolder_name,
        init_noise=init_noise,
        spinup_time=spinup_time,
        do_spinup=do_spinup,
        device=device,
        output_cpus_number=output_cpus_number,
        output_frequency=output_frequency,
        logger=logger,
    )
    return HostRuntimeComponent.from_model(
        name=name,
        grid=state.grid,
        step=partial(_camulator_runtime.step_camulator_runtime, state),
        inputs=("sea_surface_temperature", "land_surface_temperature"),
        outputs=_camulator_contracts.CAMULATOR_RUNTIME_FIELD_NAMES,
        default_fields=_camulator_contracts.camulator_runtime_field_defaults(),
        initialize=state.initialize,
    )


__all__ = [
    "make_camulator_gcm",
]
