Output and diagnostics
======================

Output
------

Use these objects to select output, provide sampled variables, and configure
period or snapshot writing at run boundaries.

.. automodule:: vercor.output
   :members: OutputContext, OutputFrame, OutputProvider, OutputSpec, OutputTarget, OutputVariable, PeriodOutput, SnapshotContext, SnapshotWriter
   :show-inheritance:

Diagnostics
-----------

Use the diagnostic helpers to summarize or compare component fields outside
the coupled physics step.

.. automodule:: vercor.diagnostics
   :members: ComponentMetric, combine_surface_temperatures, component_vector_speed, plot_component_scalar_vector_comparison, print_component_field_means_table, safe_component_nanmean, total_surface_temperature
   :show-inheritance:

Native Veris output uses the shared provider/coordinator path. Its supported
variables are ``Area``, ``hIceMean``, ``hSnowMean``, ``TSurf``, ``uIce``,
``vIce`` and native coupling diagnostics. Arrays omit halos and preserve native
tracer/face locations. ``hIceMean`` and ``hSnowMean`` remain volume per total
grid-cell area in metres. ``forc_salt_surface`` retains upstream unknown units.
The runtime alias ``ice_fraction`` is the same final concentration as native
``Area``. Empty period selections include all supported native variables;
subsets and unknown names follow the existing output coordinator's contract.
Output-free JIT/AD runs do not sample providers or invoke writers.
