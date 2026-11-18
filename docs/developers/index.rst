For developers
=============================

Use this path to wrap / build data, host, and differentiable JAX components, then
connect them through explicit exchanges.

Running the test suites
-----------------------

From a checkout, activate a fresh Python 3.12 or 3.13 environment as described
in :doc:`../researchers/getting-started`. Install the development, native-model,
and documentation dependencies before running the fast suite.

.. code-block:: console

   python -m pip install -r docs/requirements.txt
   python -m pip install ".[dev,jcm,veros,veris]"
   python -m pip check
   python -m pytest tests/ --fast

Omit ``--fast`` to run the full stock suite. For native Veros autodiff, use a
separate environment with the differentiable Veros fork pinned in
``.github/workflows/python-package.yml`` and run
``python -m pytest tests/ -n0 -m veros_autodiff``.

.. toctree::
   :maxdepth: 1

   concepts
   data-components
   host-components
   jax-components
   coupling
   /output-profiling
   /exchange-profiling
   /runtime-profiling
