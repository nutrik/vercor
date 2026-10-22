Execution backend
=================

Set the backend through :class:`~vercor.runtime.RuntimeOptions` when
constructing a :class:`~vercor.Coupler`.
For a graph that contains both JAX and host components, use the default
recommendation:

.. code-block:: python

   runtime = RuntimeOptions(backend="auto")

.. list-table:: Backend settings
   :header-rows: 1
   :widths: 17 52 31

   * - Setting
     - Behavior
     - Valid components
   * - ``auto``
     - Selects the host driver when any scheduled component is host-backed.
     - Mixed
   * - ``jax``
     - Uses compiled JAX execution.
     - JAX only
   * - ``host``
     - Uses the Python driver.
     - JAX and host

Forcing ``backend="jax"`` rejects host components during configuration before
the coupler takes a step. See :doc:`../developers/host-components` for the
host-component contract.

With fixed array schemas, VerCOR reuses its compiled kernels during a run.
Initial compilation of a schedule or kernel is expected. Changes to shapes,
dtypes or static configuration can require another executable. Separate runs
retain their existing run-local compilation lifetime.

Host components and custom backends keep their Python execution contract.
Their authors should construct reusable JIT functions during setup and pass
changing numerical values, including step indices, as dynamic arguments.
Defining a new JIT function or local JAX control-flow body on every Python step
can trigger compilation inside that component even when VerCOR's kernels are
stable.
