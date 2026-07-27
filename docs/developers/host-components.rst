Host components
===============

A component that needs ordinary Python or host-library execution declares
``execution="host"``.  ``RuntimeOptions(backend="auto")`` then selects the
host driver when that component is scheduled.

The setup hook returns the first payload and each step functionally replaces
it in ``StepResult``.  The example increments a counter while returning a new
``HostPayload`` each time.  Never place evolving hidden mutable state on the
component author object: it would be shared configuration rather than
per-runtime state.

.. literalinclude:: ../_examples/host_component.py
   :language: python
   :linenos:
