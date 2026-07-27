Data components
===============

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
