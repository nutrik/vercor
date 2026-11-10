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

With the default empty ``variables`` tuple, period output includes every
available state variable, flux and diagnostic. Bundled native providers combine
the native registry with canonical component fields; native names retain their
existing meaning where both expose the same name. For a subset, specify names
in the desired order:

.. code-block:: python

   output = OutputSpec(
       period=PeriodOutput("month", variables=("sea_surface_temperature",)),
   )

Providers receive the selection before extraction, so unselected arrays are
not converted or diagnosed for output. The default runtime provider includes
both inputs and outputs. Received inputs keep their current values; a component's
climatological export policy applies only to its outputs. Custom providers own
which variables they expose and should honor ``context.variables`` before
extracting arrays. Unknown names raise a component-scoped error.

Veris includes every initialized native state and diagnostic registry entry.
Veros includes initialized diagnostic arrays with names such as ``energy.k_m``;
output does not run or reset unscheduled native diagnostic routines. Repeated
native axes have distinct dimension names, and implicit dimensions use index
coordinates. Native endpoint snapshots retain their own temporal meaning.

Enabled file output rejects traced runtime state because writing files is a
host-side effect. Keep transformed runs output-free and write results from an
ordinary Python call instead.

Use the same policy with every backend
----------------------------------------

``OutputSpec``, ``PeriodOutput`` and ``OutputTarget`` work the same way for
all-JAX, forced-host, mixed-component and custom-backend runs. Each completed
coupling step contributes a post-step sample. The shared coordinator selects
variables, checks their schema, accumulates finite sums and per-cell counts,
and computes means only when writing complete periods and retains independent component cadences. Incomplete final periods
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

For a fixed provider schema, unequal calendar periods and incomplete tails
reuse the schedule's compiled scan. A common exact block size divides every
window, so shorter windows need no padding or extra samples. Coprime window
lengths use single-step blocks. Numeric scalar coordinates use an explicit
dtype inside the scan so period resets preserve the JAX input signature.
Initial compilation is expected; new provider shapes, dtypes or static
metadata require a new configuration.

Streaming native samples
------------------------

The output session stores only fixed-size sums, finite counts and current
coordinates.

Providers that already reduced native samples can supply
``OutputFrame(reduction_counts={...})``. For each counted variable, its values
must be finite sums and its counts must be same-shaped nonnegative integers;
no sample axis is included. The session merges those sums/counts directly.
Uncounted variables retain ordinary sampling semantics, so canonical fields can
share the same frame. A cell with no finite samples writes NaN. Incomplete final
windows are retained only for the run and are not written.
