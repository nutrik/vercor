Configure output
================

Use ``coupler.run(output=None)`` when a run should perform no I/O, including
when it is transformed with JAX. It returns the final state directly and keeps
the run suitable for compilation and differentiation.

Choose run-level output kinds
-----------------------------

Pass an :class:`~vercor.output.OutputTarget` to enable selected files beneath a
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

:class:`~vercor.output.OutputSpec` belongs in a component's specification. Its
``period`` policy uses :class:`~vercor.output.PeriodOutput`; select ``"step"``
with ``PeriodOutput(frequency="step")`` for every-step period output, or use
``PeriodOutput(frequency="month")`` for monthly output.

.. code-block:: python

   from vercor.output import OutputSpec, PeriodOutput

   output = OutputSpec(period=PeriodOutput(frequency="month"))

Enabled file output rejects traced runtime state because writing files is a
host-side effect. Keep transformed runs output-free and write results from an
ordinary Python call instead.

Use the same policy with every backend
----------------------------------------

``OutputSpec``, ``PeriodOutput`` and ``OutputTarget`` work the same way for
all-JAX, forced-host, mixed-component and custom-backend runs. Each completed
coupling step contributes a post-step sample. The shared coordinator selects
variables, checks their schema, averages finite samples, writes complete
periods and retains independent component cadences. Incomplete final periods
are not written.

Providers may implement :class:`~vercor.output.JaxOutputProvider` in addition
to :class:`~vercor.output.OutputProvider`. Period output prefers their pure
``sample_jax`` method on every backend, even alongside ordinary providers.
Its :class:`~vercor.output.JaxOutputContext` supplies an absolute int32 step
and the coupling interval in seconds. Ordinary providers use ``sample`` with
a concrete :class:`~vercor.output.OutputContext`, including the end-of-step
calendar timestamp. Neither interface performs file I/O.

When the runtime uses JAX and every enabled provider supports numeric sampling,
sampling and accumulation run inside scans between output boundaries. Host
components execute and sample in Python, so their native payloads need not be
JAX PyTrees. Both paths use the same validation and averaging implementation;
files and calendar formatting remain with the host coordinator.
