Host components
===============

Prerequisites
-------------

Install the core VerCOR package and read :doc:`concepts` for the distinction
between fixed author configuration and per-run payload state. This tutorial
uses no optional model package.

A component that needs ordinary Python or host-library execution declares
``execution="host"``.  :class:`~vercor.runtime.RuntimeOptions` ``(backend="auto")``
then selects the host driver when that component is scheduled.

The setup hook returns the first payload and each step functionally replaces
it in :class:`~vercor.components.StepResult`.  The example uses ``HostPayload.calls`` to add ``1`` on
the first step and ``2`` on the second while returning a new payload each time.
Never place evolving hidden mutable state on the component author object: it
would be shared configuration rather than per-runtime state.

.. literalinclude:: ../_examples/host_component.py
   :language: python
   :linenos:

Expected result
---------------

The program completes without output after two host steps. Its assertion
confirms that every value in the final ``counter`` field is ``3.0``. The
second increment depends on receiving the replacement payload from the first
step.

Next steps
----------

Continue to :doc:`jax-components` to implement a compiled and differentiable
component with pure array operations.
