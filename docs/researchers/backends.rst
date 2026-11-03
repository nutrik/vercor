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

Veros numerical backend
-----------------------

The bundled Veros adapter has a separate numerical-backend choice:

.. code-block:: python

   from vercor.setups import VerosConfig, make_veros_gcm

   ocean = make_veros_gcm(
       config=VerosConfig(execution="host", backend="jax")
   )

Released Veros 1.7.0 supports JAX-compiled native kernels under the host driver.
The native backend now defaults to ``"jax"`` after the measured coupled
throughput comparison. Use ``backend="numpy"`` for native NumPy kernels.
Explicit ``backend=None`` retains execution-based selection (NumPy for host,
JAX for scanned execution). ``execution="host"`` keeps
Python orchestration in either case; it does not make the ocean differentiable
through a compiled VerCOR scan. VerCOR retains ownership of coupling, temporal
means, continuation state and output.

``execution="jax"`` selects the existing differentiable-fork adapter and
requires the pinned fork capabilities, with native JAX kernels. A NumPy backend
with that execution policy is rejected. With fork-backed JAX components,
``RuntimeOptions(backend="host")`` forces host orchestration for a controlled
comparison against ``RuntimeOptions(backend="jax")`` using the same model
distribution. Comparing stock Veros with the older fork also changes model
code and must not be attributed solely to orchestration.

Veros locks its native backend process-wide. Run NumPy, stock JAX and fork JAX
comparisons in separate Python processes. Choose the backend before importing
native implementation modules. See :doc:`../veris-veros-jcm-validation` for
the measured release comparison and its startup/throughput limitations.
