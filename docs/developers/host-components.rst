Host components
===============

Prerequisites
-------------

Install the core VerCOR package and read :doc:`concepts` for the distinction
between fixed author configuration and per-run payload state. This tutorial
uses no optional model package.

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

Expected result
---------------

The program completes without output after two host steps. Its assertion
confirms that every value in the final ``counter`` field is ``2.0`` while each
step has returned a replacement ``HostPayload``.

Next steps
----------

Continue to :doc:`jax-components` to implement a compiled and differentiable
component with pure array operations.
