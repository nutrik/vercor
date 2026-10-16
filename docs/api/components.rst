Components
==========

Use these contracts to declare component fields and lifecycle behavior, adapt
callable steps, or supply time-selected data.

.. automodule:: vercor.components
   :members: CallableComponent, Component, ComponentMaskSpec, ComponentSpec, DataComponent, LifecycleHooks, PrefillContext, PrefillResult, SetupContext, SetupResult, StepContext, StepResult, TransferPolicy, ValidationContext
   :show-inheritance:

Dynamic component masks
-----------------------

``ComponentSpec(masks=ComponentMaskSpec(binary_field="active",
fractional_field="ice_fraction"))`` references ordinary declared runtime fields.
Setup and step results initialize or replace these fields using the existing
lifecycle. Scalars expand at preparation; runtime mask fields must match the
grid shape and dtype and use ``TransferPolicy(time_selection="current")``.

Binary fields contain finite zeros or ones and remain within the immutable
grid wet mask. Fractions contain finite values in [0, 1] and are zero outside
the binary support. Omitted binary support uses the grid wet mask; omitted
fractional coverage is one on active support. Fractional values remain traced
for differentiation; binary membership changes are discrete.
