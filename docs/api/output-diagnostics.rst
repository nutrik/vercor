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
