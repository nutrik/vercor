Coupling components
===================

An ``Exchange`` transfers named fields from a source component to a target
component. Both endpoints must declare every exchanged field: the source as
an output and the target as an input. The following complete example sends a
static heat flux to a model and verifies that the model uses it.

.. literalinclude:: ../_examples/coupled_components.py
   :language: python
   :linenos:

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
