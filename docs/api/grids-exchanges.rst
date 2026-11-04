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

Dynamic masks and quantities
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Every route must explicitly select ``Exchange(quantity=...)``. Each route
applies one quantity meaning to all of its
scalar and vector fields; use separate route IDs for quantities with different
meanings.

Let ``R`` be the fixed remapper, ``a`` the source coverage, ``b`` the target
binary computational support, and ``c`` its fractional coverage. The modes are:

* ``intensive``: ``b * R(a * field) / R(a)``, with safe zero on empty overlap.
  Temperature, humidity, pressure and per-area forcing keep their intensive
  values. A wet ice-free target still receives forcing so ice can form.
* ``source_flux``: ``b * R(a * field)``, the source-area contribution per target
  grid-cell area. Choose conservative remapping when integral conservation is
  required; bilinear interpolation does not provide that guarantee.
* ``target_flux``: ``c * R(a * field) / R(a)``, the covered-source mean weighted
  by the target coverage.

The default ``source_coverage="component"`` uses the current fractional mask.
Use ``source_coverage="binary"`` with an explicit quantity for fields already
integrated over total computational area, such as Veris ocean boundary fluxes.
This selects source binary support for weighting, overlap and finite-value
validation, so a wet ice-free source still sends its total flux. Do not multiply
these fields by ice concentration again. Stored imports retain the support of
their last dispatch independently of current component coverage.

Source weighting occurs before scalar/vector remapping, retaining vector
transformation. Inactive NaNs are neutralized before arithmetic; infinity is
invalid everywhere. Missing remapped coverage becomes zero, and mask diagnostics
are bounded to [0, 1] without changing the physical field numerator/denominator.
Prepared binary topology restrictions intersect these explicit modes. A
conflicting fractional topology patch is rejected to prevent double weighting.

See :doc:`../exchange-coverage` for the complete coverage contract.

Fields
------

Use scalar names or explicit vector declarations to describe an exchange's
field capabilities. :doc:`../coupling-fields` lists the bundled physical names,
units, vertical levels and migration from former names.

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
