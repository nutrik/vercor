Component concepts
==================

VerCOR accepts a structural component contract.  A component author provides a
``name``, a :class:`~vercor.grids.RectilinearGrid`, a
:class:`~vercor.components.ComponentSpec`, and a ``step`` callable.  The
specification declares the fields that the step reads (``inputs``), the fields
it may update (``outputs``), their initial values, its execution capability,
and lifecycle policies.  :class:`~vercor.components.CallableComponent` 
and :class:`~vercor.components.DataComponent` are convenient ways to provide 
that same contract.

Fields are declarations, not ad-hoc mutable dictionaries.  The runtime
normalizes declared scalar initial values to the component grid and validates
field names, shapes, and dtypes at the component boundary.  A component
step returns a mapping of declared output updates, or a 
:class:`~vercor.components.StepResult` when it must also replace its 
runtime payload.

Runtime state is immutable.  ``Coupler.initial_state()`` creates a
:class:`~vercor.RunState`; methods such as ``RunState.replace_fields`` 
return a new state instead of mutating the current one.  This makes 
experiments reproducible and keeps JAX transformations well defined.

Author configuration versus runtime payload
-------------------------------------------

The component author object is configuration: it owns the fixed name, grid,
specification, and step implementation.  Evolving per-run state belongs in a
payload returned by :class:`~vercor.components.SetupResult` and then 
replaced through :class:`~vercor.components.StepResult`.
Do not keep changing counters, caches, or model state as hidden mutable
attributes on the author object.  Keeping configuration and per-run payload
separate lets independent runs start from independent immutable states.
