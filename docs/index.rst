VerCOR documentation
====================

VerCOR is a JAX-first coupler for composing atmosphere, ocean, sea-ice, land,
and forcing-data components on a shared clock. It moves fields between model
grids, supports host-side and JAX-native components, and keeps output-free JAX
workflows differentiable.

For Earth-system researchers
----------------------------

Start with :doc:`researchers/index` to install VerCOR, run a small bundled
model, and inspect coupled state and output.

For Python and JAX developers
-----------------------------

Start with :doc:`developers/index` to implement data, host, or differentiable
JAX components and connect them with exchanges.

.. toctree::
   :maxdepth: 2
   :caption: Learn

   introduction
   researchers/index
   developers/index

.. toctree::
   :maxdepth: 2
   :caption: Use and extend

   how-to/index
   troubleshooting
   api/index
   project-resources
