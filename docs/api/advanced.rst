Advanced extension API
======================

.. warning::

   These contracts are intended for custom schedulers, execution backends, and
   exchange topology policies. Most applications should use the stable
   user-facing API.

Runtime execution
-----------------

.. automodule:: vercor.runtime
   :members: ExecutionBackend, ExecutionChunk, ExecutionContext, ExecutionPlan, RuntimeDriver, RuntimeOptions, SequentialWorkflow, StepPlan, Workflow, WorkflowContext
   :show-inheritance:

Exchange topology
-----------------

.. automodule:: vercor.topology
   :members: ExchangeTopologyPatch, SurfaceMaskPolicy, TopologyContext, TopologyPolicy
   :show-inheritance:
