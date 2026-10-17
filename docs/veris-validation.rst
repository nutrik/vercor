Veris scientific validation
============================

This integration validates numerical conservation, spherical operators and
parameter derivatives. It does not establish observed climate skill. The native
dependency is pinned to ``4551fc86181c660ceb0010160ec7f569d97e52c6`` on
``team-ocean/veris:feat/vercor-scientific-validation``. Local CPU correctness, coverage, documentation and packaging checks pass;
exact-commit hosted evidence is recorded with the measured results.

Physics and units
-----------------

Native optional spherical coordinates include tracer/edge latitudes and radius.
Supplied native cell areas remain authoritative. Zonal distances include the
latitude cosine, tensor strain includes spherical curvature, and stress divergence
is the stored-area weighted negative transpose of strain. Cartesian setups retain
their existing convention. Spherical y boundaries are closed.

Native physical constants are traced PyTree leaves; derived values are recomputed
from base parameters. Precision, shapes, iteration counts and binary transitions
are static. ``replace_veris_parameters`` creates a new runtime state without
reinitializing geometry or changing the original state. Derivatives are local to
the executed smooth branch; hard ice-formation and clipping boundaries are not
smooth calibration targets.

The coupling exports downward-positive heat (W/m2), total bulk mass (kg/m2/s),
actual salt mass (kg/m2/s), and canonical east/north stresses (Pa), each averaged
over every native substep. They are fluxes per total wet-cell area. Binary source
coverage prevents a second concentration factor, including after complete melt.
Native salinity remains g/kg. Fixed bulk ice salinity must not exceed ocean
salinity during gross ice formation; melting salty ice into fresher water remains
valid. Snow falling on open water melts with its latent cooling, vapor exchange is
limited by available solid mass, and cleanup is budgeted once.

VerCOR owns freshwater density. Prognostic Veros uses the fixed-reference virtual
salinity forcing ``(-S_ref*M + 1000*J)/rho_fw`` in m (g/kg)/s, with M and J the
actual downward mass and salt fluxes. This rigid-lid tracer budget differs from a
combined physical ocean/ice salt inventory when local salinity differs from the
reference. Native ocean density, heat capacity, timestep and layer depth convert
heat/stress/salinity sources once. Independent restoring remains explicit.

Reproduce extended checks
-------------------------

From the source checkout, with the pinned native dependency and forcing assets:

.. code-block:: console

   python -m benchmarks.validate_veris --days 30 --model-seconds 3600 --output test_logs/science/30d-3600
   python -m benchmarks.validate_veris --days 30 --model-seconds 1800 --output test_logs/science/30d-1800
   python -m benchmarks.validate_veris --days 30 --model-seconds 900 --output test_logs/science/30d-900
   python -m benchmarks.validate_veris --days 365 --model-seconds 3600 --segment-days 90 --output test_logs/science/365d-3600
   python -m benchmarks.validate_veris --days 365 --model-seconds 1800 --output test_logs/science/365d-1800

The default static momentum precision is 400 EVP iterations with alpha=beta=500.
``--evp-steps 1600`` repeats a case at increased solver precision without changing
physical parameters. ``--adaptive-evp`` selects the existing adaptive numerical
solver. Reports include the resolved numerical configuration and the physical
constants used in inventory calculations. ``--minimum-ice-thickness`` explicitly
authors the native ``hIce_min`` removal threshold (m); its default remains 1e-5.

The original native threshold does not pass this experiment's coupled temporal
refinement criterion. It repeatedly removes small transported stocks, making
propagation across an ice edge depend on the native timestep. Only two cells
near 74 degrees south account for almost all the 900/450-second volume shift.
Momentum iteration is not the remaining cause: all seven reported quantities at adaptive 400/800 iterations agree
within the declared roundoff floors, while thermodynamics-only refinement passes.

The validated configuration uses float64, adaptive EVP with 400 iterations and
an explicitly authored 1e-9 m removal threshold. All seven 3600/1800/900-second
quantity differences decrease there. A further reduction to 1e-11 m leaves ice,
snow and salt inventories unchanged and changes cumulative inputs by at most
1.02e-10 relative. Native defaults and phase equations are preserved. This
validation applies to the stated configuration; it does not certify the original
1e-5 m cutoff or arbitrary timesteps.

Repeat each 30-day command above with these explicit settings:

.. code-block:: console

   python -m benchmarks.validate_veris --days 30 --model-seconds 900 --adaptive-evp --minimum-ice-thickness 1e-9 --output test_logs/science/validated-900

The harness reuses the prescribed gallery and ``Coupler.run``. Initial polar wet
cells at absolute latitude >=60 degrees have concentration 0.5, mean ice thickness
0.25 m, surface temperature 260 K and fixed ice salinity 4 g/kg. Both daily incoming
routes are intensive; ERA5 and monthly native Veros SST/SSS use unchanged forcing
timestamps. Ocean currents, height and runoff are prescribed zero. These are
forced numerical experiments, not an equilibrated climate.

The native runtime checks all floating leaves, concentration bounds, nonnegative
ice/snow storage and physics status at every substep. An integer carry counts
actual executed transitions. A pure output provider reduces interval rates on
device through existing step-period output. The verifier integrates those scalar
means with exact coupling duration and removes temporary scalar files; no native
trajectory is retained. Reports record commits, source/forcing/area hashes, dtype,
geometry, actual checked steps, inventories, fluxes and normalized residuals.
Segmented runs retain all payload/flux/parameter carries and compare every state
leaf at the same model-calendar forcing dates.

Independent closures use native stored areas and modeled solid enthalpy
``H = -L*(rho_ice*V_ice + rho_snow*V_snow)``. For each quantity, the residual
``abs(delta_inventory + ocean_input - external_input)`` is normalized by the
largest absolute term or 1. Actual salt has no external source. Float64 acceptance
is <=1e-10. Negative controls reject lost flux, duplicated concentration/density/
timestep factors, final-substep substitution and inconsistent geometry weighting.

A simple constant-force, zero-drag momentum balance shows why iteration accuracy
must be checked separately: n fixed-beta iterations advance the fraction
``1 - (beta/(beta+1))**n`` of the fully converged backward-Euler increment.
At beta=500 this is about 0.55 for 400 iterations and 0.96 for 1600. This is a
numerical solver error, not a change to atmospheric forcing or physical drag.

Thirty-day refinement compares ice/snow volume and five cumulative inputs at
3600/1800/900 seconds. Successive differences must decrease outside the reported
roundoff floor ``64*eps*checked_native_steps*quantity_scale``. Annual runs report
sensitivity without claiming that two resolutions establish an order of accuracy.
The native 1000-step spherical transport test separately measures volume drift,
rigid-body rotation and second-order tensor refinement.

Calibration and prognostic feedback
-----------------------------------

The reusable workflow and invocation are in :doc:`setup-gallery`. Synthetic h0
recovery tests span three truths and two starts, fixed wet weights including
ice-free cells, a distinct holdout forcing, and float32/float64. Zero/all-dry
weights and an insensitive observation window fail clearly. JVP, VJP and a finite
difference sweep agree for native physical parameters and the actual coupled
h0-to-ocean-salinity response.

Small real prognostic integrations verify heat, bulk mass and fixed-reference
virtual-salt inventories using native areas and vertical layers. Shared-grid
stress transfer is exact; different-grid staggered stress conservation needs
separate validation. The long prescribed runs do not validate long prognostic
ocean stability, observational parameter identification or smooth gradients at
binary phase transitions.

Measured acceptance evidence
----------------------------

The committed :download:`machine-readable summary <veris-validation-results.json>`
includes exact provenance, source hashes, failed controls and the full decision log.
Detailed local logs and JSON reports are retained under the ignored
``test_logs/veris-scientific-validation`` directory. Native transport at the clean
pinned revision completed 1000 steps with maximum relative volume drift
2.11497e-15 and tensor refinement orders 1.999736 and 1.999934.

.. figure:: /_images/veris-refinement.svg
   :width: 100%

   Recorded final volume and successive timestep differences. The original
   removal threshold fails refinement; the explicit 1e-9 m configuration resolves
   it. Thermodynamics-only is a separate diagnostic, not an alternative climate
   prediction. Historical controls retain their actual native revision.

.. list-table:: Float64 acceptance measurements
   :header-rows: 1

   * - Experiment
     - Checked native steps
     - Maximum normalized budget residual
   * - Validated 30-day, 3600 s
     - 720
     - 1.052e-15
   * - Validated 30-day, 1800 s
     - 1440
     - 1.052e-15
   * - Validated 30-day, 900 s
     - 2880
     - 7.514e-16
   * - Annual stability, 3600 s
     - 8760
     - 1.835e-15
   * - Annual stability, 1800 s
     - 17520
     - 7.472e-16

Both 30-day and annual segmented comparisons have zero scaled state difference.
Annual 3600/1800-second quantity sensitivity is at most 5.4533e-4 relative; two
resolutions alone establish no temporal order. The 1000-step transport test is
at native ee9614e. Final-pin 4551fc8 is used for the convergent 30-day sweep and
installed correction tests; older runs are not relabeled as that revision.
