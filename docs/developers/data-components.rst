Data components
===============

Prerequisites
-------------

Install the core VerCOR package on Python 3.12 or 3.13. This tutorial uses no
optional model package or external data.

Use ``DataComponent`` for forcing or observations that have no active model
step.  A scalar field expands to the component grid, while a leading record
axis is preserved.  The executable example defines both forms.

``TransferPolicy("current")`` exports the stored field as-is.  ``linear``
selects adjacent monthly records and interpolates during exchange.  ``daily``
selects daily records; it expects 365 records for the Gregorian and no-leap
calendars, or 360 records for the 360-day calendar.

.. literalinclude:: ../_examples/data_component.py
   :language: python
   :linenos:

Expected result
---------------

The program completes without output. Its assertions confirm that the scalar
forcing expands to the two-dimensional grid and the monthly forcing retains
its 12-record leading axis.

Next steps
----------

Continue to :doc:`host-components` when a model needs ordinary Python
execution and per-run payload state.
