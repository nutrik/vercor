Data components
===============

Prerequisites
-------------

Install the core VerCOR package on Python 3.12 or 3.13. This tutorial uses no
optional model package or external data.

Use :class:`~vercor.components.DataComponent` for forcing or observations 
that have no active model step.  A scalar field expands to the component grid,
while a leading record axis is preserved.  The executable example defines both
forms.

:class:`~vercor.components.TransferPolicy` ``("current")`` exports the stored
field as-is.  ``linear`` selects adjacent monthly records and interpolates 
during exchange.  ``daily`` selects from one no-leap climatology with 365
records for every supported calendar. 
In a Gregorian leap year, February 29 uses the same record as
February 28.  For a 360-day clock, VerCOR maps the relative position of each
day in a 360-day month onto the corresponding no-leap month; it does not
expect a separate 360-record dataset.

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
