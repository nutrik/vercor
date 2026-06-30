"""Veros ocean component factory."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import timedelta
from functools import partial
from typing import Any

from vercor.components import HostRuntimeComponent, host_component
import vercor.setups.external.veros_gcm_state as _veros_gcm_state
import vercor.setups.external.veros_runtime as _veros_runtime
from vercor.setups.external.veros_gcm_state import VerosGCMSetupState

try:
    import veros  # noqa: F401
except ImportError:
    raise ImportError(
        "The VerosGCM component requires the Veros package. Please install it with `pip install veros`."
    )


def make_veros_gcm(
    name: str = "OCN",
    spinup_time: timedelta = timedelta(days=2),
    custom_parameters: dict[str, Any] | None = None,
    restore_to_climatology: bool = False,
    do_spinup: bool = False,
    output_frequency: str | None = None,
    output_variables: Sequence[str] | None = None,
    jitted: bool = False,
) -> HostRuntimeComponent:
    """Return a host-backed Veros GCM component."""

    state = VerosGCMSetupState(
        name=name,
        spinup_time=spinup_time,
        custom_parameters=custom_parameters,
        restore_to_climatology=restore_to_climatology,
        do_spinup=do_spinup,
        output_frequency=output_frequency,
        output_variables=output_variables,
        jitted=jitted,
    )
    return host_component(
        name=name,
        grid=state.grid,
        step=partial(_veros_runtime.step_veros_runtime, state),
        inputs=_veros_gcm_state.VEROS_INPUT_FIELD_NAMES,
        outputs=("sea_surface_temperature",),
        default_fields=_veros_gcm_state.veros_default_fields(),
        initialize=state.initialize,
    )


__all__ = [
    "make_veros_gcm",
]
