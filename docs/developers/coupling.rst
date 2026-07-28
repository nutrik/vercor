Coupling components
===================

Prerequisites
-------------

Install the core VerCOR package and read :doc:`data-components` and
:doc:`jax-components` for the two component styles used here. The example uses
the built-in bilinear regridder and needs no optional model package.

An :class:`~vercor.exchanges.Exchange` transfers named fields from a source
component to a target component. Both endpoints must declare every exchanged
field: the source as an output and the target as an input. The following
complete example sends a static heat flux to a model and verifies that the
model uses it.

.. literalinclude:: ../_examples/coupled_components.py
   :language: python
   :linenos:

Expected result
---------------

The program completes without output. Its assertion confirms that the model's
initial temperature of ``280.0`` becomes ``285.0`` in every target-grid cell
after receiving the ``5.0`` heat flux for one step.

Scheduling and route identity
-----------------------------

``run_order`` controls the receive, step, and send sequence within each clock
step. Put a target after its source when the target must use that source's
current-step result. An exchange without ``route_id`` uses
``"source->target"``; give every route an explicit, unique ID when more than
one route connects the same component pair.

Regridding and topology
-----------------------

Use :func:`vercor.regridding.bilinear` for scalar fields and paired vector
components. Use :func:`vercor.regridding.conservative` only for scalar fields.
The coupler rejects ambiguous fan-in: two routes may not write the same target
field. Combine values in an explicit component step instead.

``SurfaceMaskPolicy`` describes the bundled atmosphere/ocean/land topology.
It is not a general policy for ordinary setup-agnostic component graphs; use
plain exchanges for those graphs.

Next steps
----------

Choose an execution policy in :doc:`../how-to/backends`, then configure
optional files and diagnostics in :doc:`../how-to/output`.
