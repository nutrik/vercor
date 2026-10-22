Output and diagnostics
======================

Output
------

Use these objects to select output, provide sampled variables, and configure
period or snapshot writing at run boundaries.

.. automodule:: vercor.output
   :members: JaxOutputContext, JaxOutputProvider, OutputContext, OutputFrame, OutputProvider, OutputSpec, OutputTarget, OutputVariable, PeriodOutput, SnapshotContext, SnapshotWriter
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
grid-cell area in metres. ``forc_salt_surface`` uses local virtual-salinity units (g/kg)*m/s.
Native ``ocean_mass_flux`` and ``ocean_salt_mass_flux`` are actual downward
bulk/salt mass inputs in kg/m2/s; independent external mass and enthalpy inputs
are also available from the native diagnostic registry.
The canonical variables ``sea_surface_heat_flux``, ``sea_surface_mass_flux``,
``sea_surface_salt_mass_flux`` and ``ocean_u_stress``/``ocean_v_stress`` are
coupling-interval means, formed as sum(dt_native * flux) / dt_coupling.
Heat and actual mass/salt input are downward positive; stress is total force
per computational face area in Pa, including direct open-water stress.
Canonical stresses use ``x_east_face``/``y_north_face`` at upper cell edges,
distinct from native west/south ``x_face``/``y_face``. Final native diagnostics
remain instantaneous. Empty selections include both native and canonical
variables.

The runtime alias ``ice_fraction`` is the same final concentration as native
``Area``. Empty period selections include all supported native variables;
subsets and unknown names follow the existing output coordinator's contract.
Output-free JIT/AD runs do not sample providers or invoke writers.
