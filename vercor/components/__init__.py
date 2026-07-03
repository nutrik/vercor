from vercor._deprecation import deprecated_getattr
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

__all__ = [
    "Component",
    "ComponentCreatePayloadHook",
    "ComponentHooks",
    "ComponentInitializeHook",
    "ComponentPrefillHook",
    "ComponentValidateHook",
    "DataComponent",
    "FieldSpec",
    "HostComponent",
    "KEEP_PAYLOAD",
    "SetupContext",
    "StepContext",
    "StepResult",
]


__getattr__ = deprecated_getattr(
    __name__,
    {
        "ComponentFieldSpec": ("vercor.components.FieldSpec", FieldSpec),
        "ComponentSetupContext": ("vercor.components.SetupContext", SetupContext),
        "ComponentStepContext": ("vercor.components.StepContext", StepContext),
        "ComponentStepResult": ("vercor.components.StepResult", StepResult),
        "HostRuntimeComponent": ("vercor.components.HostComponent", HostComponent),
    },
    remove_in="0.2.0",
)
