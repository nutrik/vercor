Advanced extension API
======================

.. warning::

   These contracts are intended for custom schedulers, execution backends, and
   exchange topology policies. Most applications should use the stable
   user-facing API.

Runtime execution
-----------------

Implement these contracts only when the default sequential workflow and
built-in host or JAX execution are insufficient.

.. automodule:: vercor.runtime
   :members: ExecutionBackend, ExecutionChunk, ExecutionContext, ExecutionPlan, RuntimeDriver, RuntimeOptions, SequentialWorkflow, StepPlan, Workflow, WorkflowContext
   :show-inheritance:

Exchange topology
-----------------

Implement a topology policy to derive validated route masks during
preparation.

.. automodule:: vercor.topology
   :members: ExchangeTopologyPatch, SurfaceMaskPolicy, TopologyContext, TopologyPolicy
   :show-inheritance:
