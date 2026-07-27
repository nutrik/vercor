Assembly and state
==================

Core assembly
-------------

Use these root exports to define a clock, assemble a coupler, run it, and
choose its ordinary runtime policy.

.. automodule:: vercor
   :members: Clock, Coupler, Exchange, RectilinearGrid, RunState, RuntimeOptions
   :show-inheritance:

Component state
---------------

Use component views to inspect fields or construct immutable field
replacements without depending on private runtime stores.

.. automodule:: vercor.state
   :members: ComponentState, FieldLookupScope, FieldScope
   :show-inheritance:

Calendars
---------

Use the calendar types when a model clock follows no-leap or 360-day dates
instead of host ``datetime`` values.

.. automodule:: vercor.calendar
   :members: CalendarDate, DateTime360, DateTime365, ModelDateTime, YearType
   :show-inheritance:
