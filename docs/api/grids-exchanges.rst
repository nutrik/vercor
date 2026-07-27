Grids, exchanges, and regridding
================================

Grids
-----

Use a rectilinear grid to declare component coordinates, edges, and masks.

.. automodule:: vercor.grids
   :members: RectilinearGrid
   :show-inheritance:

Exchanges
---------

Use exchanges to route declared source fields to declared target fields.

.. automodule:: vercor.exchanges
   :members: Exchange
   :show-inheritance:

Fields
------

Use scalar names or explicit vector declarations to describe an exchange's
field capabilities.

.. automodule:: vercor.fields
   :members: COMMON_FIELD_NAMES, ExchangeField, VectorField, vector
   :show-inheritance:

Regridding
----------

Use the bundled factories for common transfers or implement the public
protocols for custom scalar and vector regridding.

.. automodule:: vercor.regridding
   :members: Regridder, RegridderFactory, VectorRegridder, bilinear, conservative
   :show-inheritance:
