from vercor._deprecation import deprecated_getattr
from vercor import fluxes
from vercor.calendar import DateTime360, DateTime365, ModelDateTime
from vercor.clock import Clock
from vercor.components.base import (
    Component,
)
from vercor.components.contracts import (
    ComponentHooks,
    ComponentCreatePayloadHook,
    ComponentInitializeHook,
    ComponentPrefillHook,
    ComponentValidateHook,
    FieldSpec,
    KEEP_PAYLOAD,
    StepResult,
)
from vercor.components.contexts import (
    SetupContext,
    StepContext,
)
from vercor.components.data import (
    DataComponent,
)
from vercor.components.host import (
    HostComponent,
)
from vercor.coupler import Coupler
from vercor.exchange import Exchange
from vercor.grid import RectilinearGrid
from vercor.runtime.state import CouplerState
from vercor.runtime.views import ComponentView
from vercor.settings import VercorSettings as Settings

__all__ = [
    "Coupler",
    "Component",
    "ComponentCreatePayloadHook",
    "ComponentHooks",
    "ComponentInitializeHook",
    "ComponentPrefillHook",
    "ComponentValidateHook",
    "ComponentView",
    "CouplerState",
    "DataComponent",
    "FieldSpec",
    "HostComponent",
    "KEEP_PAYLOAD",
    "Settings",
    "SetupContext",
    "StepContext",
    "StepResult",
    "Clock",
    "DateTime360",
    "DateTime365",
    "RectilinearGrid",
    "Exchange",
    "fluxes",
    "ModelDateTime",
]


__getattr__ = deprecated_getattr(
    __name__,
    {
        "ComponentFieldSpec": ("vercor.FieldSpec", FieldSpec),
        "ComponentSetupContext": ("vercor.SetupContext", SetupContext),
        "ComponentStepContext": ("vercor.StepContext", StepContext),
        "ComponentStepResult": ("vercor.StepResult", StepResult),
        "HostRuntimeComponent": ("vercor.HostComponent", HostComponent),
    },
    remove_in="0.2.0",
)
