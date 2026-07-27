Configure output
================

Use ``coupler.run(output=None)`` when a run should perform no I/O, including
when it is transformed with JAX. It returns the final state directly and keeps
the run suitable for compilation and differentiation.

Choose run-level output kinds
-----------------------------

Pass an :class:`vercor.output.OutputTarget` to enable selected files beneath a
directory. Its flags independently enable or disable period files, final
fields, and component snapshots:

.. code-block:: python

   from vercor.output import OutputTarget

   target = OutputTarget(
       "output",
       write_period=True,
       write_final_fields=True,
       write_snapshots=False,
   )
   final_state = coupler.run(output=target)

Final fields record the state returned by the run. Period files contain
provider samples accumulated at the selected cadence. Component snapshots are
written only by components that declare a snapshot writer.

Choose component output policy
------------------------------

:class:`vercor.output.OutputSpec` belongs in a component's specification. Its
``period`` policy uses :class:`vercor.output.PeriodOutput`; select ``"step"``
with ``PeriodOutput(frequency="step")`` for every-step period output, or use
``PeriodOutput(frequency="month")`` for monthly output.

.. code-block:: python

   from vercor.output import OutputSpec, PeriodOutput

   output = OutputSpec(period=PeriodOutput(frequency="month"))

Enabled file output rejects traced runtime state because writing files is a
host-side effect. Keep transformed runs output-free and write results from an
ordinary Python call instead.
