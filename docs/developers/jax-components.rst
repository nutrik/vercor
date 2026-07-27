JAX components
==============

Declare ``execution="jax"`` for a component whose step is expressed with pure
array operations.  Keep its PyTree structure, array shapes, and dtypes stable
from step to step.  Do not use Python branching on traced physical values;
express array-level decisions with JAX operations instead.

For differentiable or compiled runs, use ``output=None`` to avoid I/O.  The
example runs with the JAX backend, applies ``jax.jit``, and differentiates the
final temperature with ``jax.grad``.

.. literalinclude:: ../_examples/jax_component.py
   :language: python
   :linenos:
