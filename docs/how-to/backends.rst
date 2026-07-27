Choose an execution backend
===========================

Set the backend through ``RuntimeOptions`` when constructing a ``Coupler``.
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
