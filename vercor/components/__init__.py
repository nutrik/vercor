"""Protocol-first public component authoring API."""

from vercor.components.base import CallableComponent
from vercor.components.contexts import SetupContext, StepContext
from vercor.components.contracts import (
    Component,
    ComponentMaskSpec,
    ComponentSpec,
    LifecycleHooks,
    PrefillContext,
    PrefillResult,
    SetupResult,
    StepResult,
    TransferPolicy,
    ValidationContext,
)
from vercor.components.data import DataComponent

__all__ = [
    "CallableComponent",
    "Component",
    "ComponentMaskSpec",
    "ComponentSpec",
    "DataComponent",
    "LifecycleHooks",
    "PrefillContext",
    "PrefillResult",
    "SetupContext",
    "SetupResult",
    "StepContext",
    "StepResult",
    "TransferPolicy",
    "ValidationContext",
]

# Importing package children attaches them to this facade automatically.  They
# are implementation modules, not additional public owners; keep only the
# explicit contract above visible from the facade namespace.
for _module_name in (
    "_protocol",
    "base",
    "contexts",
    "contracts",
    "data",
    "runtime_execution",
    "setup_validation",
):
    globals().pop(_module_name, None)
del _module_name
