.. Adapted from https://veros.readthedocs.io/en/latest/reference/setup-gallery.html

Setup gallery
=============

This page gives an overview of the available model setups. To copy the setup file to the current working directory, you can make use of the ``vercor copy-setup`` command.

Example:

.. code-block:: console

   vercor copy-setup run_jcm_with_veros \
     --to ~/vercor-setups/run_jcm_with_veros


.. list-table:: Available setups
   :header-rows: 1
   :widths: 29 39 16 16

   * - Setup
     - Purpose
     - Optional dependency
     - External data
   * - ``custom_component_wrapping``
     - Build data, JAX, and host component wrappers.
     - None
     - No
   * - ``profile_runtime``
     - Profile a synthetic slab coupling run.
     - None
     - No
   * - ``run_slab_driver``
     - Run and plot the bundled slab topology.
     - Matplotlib
     - No
   * - ``run_data_driver``
     - Couple ERA data components and plot diagnostics.
     - Matplotlib
     - Yes
   * - ``run_jcm_with_slab``
     - Couple JCM to slab ocean and land components.
     - JCM, Matplotlib
     - Yes
   * - ``run_jcm_with_era5data``
     - Couple JCM to ERA5 ocean forcing.
     - JCM
     - Yes
   * - ``run_jcm_with_veros``
     - Couple JCM atmosphere/land to Veros.
     - JCM, Veros
     - Yes
   * - ``run_jcm_with_verosdata``
     - Couple JCM to Veros-format ocean data.
     - JCM
     - Yes
   * - ``run_veros_with_era5data``
     - Couple Veros to ERA5 atmosphere and land data.
     - Veros
     - Yes
   * - ``run_veris_dataveros_era5``
     - Run Veris with prescribed Veros ocean and ERA5 data.
     - Veris, Veros
     - Yes
   * - ``run_veris_veros_era5``
     - Couple prognostic Veris and Veros with prescribed ERA5 air.
     - Veris, Veros
     - Yes
   * - ``run_veris_veros_jcm``
     - Couple JCM land/atmosphere, prognostic Veros and native Veris sea ice.
     - JCM, Veris, Veros
     - Yes
   * - ``run_veris_calibration``
     - Recover native h0 from synthetic concentration with a cold-forcing holdout.
     - Veris
     - No
   * - ``run_camulator_with_veros``
     - Couple CAMulator atmosphere/land to Veros.
     - CAMulator, Veros
     - Yes

The table identifies the extra dependency and input-data expectation for each
setup. ``None`` means that the setup has no model-specific optional package
or external input requirement, although plotting examples need
`Matplotlib <https://matplotlib.org>`_.

Coupled models
--------------

Veros
+++++

`Veros <https://veros.readthedocs.io/en/latest/>`__, the versatile ocean simulator, is a full-fledged primitive equation ocean model
that supports anything between idealized toy models and realistic, high-resolution, global ocean simulations.

The ERA5/Veros and JCM/Veros examples exchange ``rain_rate`` and ``snow_rate``
in kg/m²/s, positive into the ocean. ERA5 sums ``crr + lsrr`` for rain and
``csfr + lssfr`` for water-equivalent snow before interpolation. JCM sums
``precls + precnv`` and divides by 1000 to convert its g/m²/s diagnostics.
It diagnoses water-equivalent rain and snow on the native JCM grid before
interpolating each phase, using SPEEDY's near-surface air temperature ``t0``.
See the `ECMWF parameter definitions <https://codes.ecmwf.int/grib/param-db/228218>`_
and `JCM diagnostic units <https://jax-gcm.readthedocs.io/en/latest/speedy_translation.html>`_.

JCM's snow fraction is a sigmoid approximation to the calibrated
linear-transition (LT) model in
`Bedard-Therrien et al. (2025), Appendix C <https://doi.org/10.5194/hess-29-1135-2025>`_:

.. math::

   f_{\mathrm{snow}} = \operatorname{sigmoid}\left(
       \frac{4(T_{\mathrm{mid}} - T_0)}{T_{\mathrm{rain}} - T_{\mathrm{snow}}}
       \right), \qquad
   T_{\mathrm{mid}} = \frac{T_{\mathrm{snow}} + T_{\mathrm{rain}}}{2}.

The traced ``PhysicalConstants.precipitation_snow_temperature`` and
``precipitation_rain_temperature`` default to 272.77 K (-0.38 degrees C)
and 278.15 K (5 degrees C), respectively, with a required positive span.
The sigmoid matches LT's 50 percent point (275.46 K) and midpoint slope;
at the two thresholds its snow fractions are approximately 88.08 percent
and 11.92 percent. It retains smooth phase tails, rather than LT's hard
100/0 percent cutoffs. These thresholds were calibrated over boreal eastern
Canadian land stations; this adaptation has not been calibrated for the
global ocean or SPEEDY.

With total water-equivalent precipitation ``P``, snow is ``P * f_snow``
and rain is ``P - snow``. Their sum and hence their contribution to virtual
salt flux are preserved. The split does not introduce snow storage or
melting-energy terms.

The wrapper calls ``surface_flux.t0`` ``temperature_2m`` for phase diagnosis;
SPEEDY documents it as near-surface temperature without specifying a 2 m
height. Exported ``temperature``, ``specific_humidity``, winds, density, and
potential temperature still describe the bottom model level at the exported
``model_level_height``. Their local names use ``*_bottom`` to distinguish
them from t0; radiative fluxes are surface quantities.

VerCOR combines these rates with bulk evaporation on the ocean grid and
computes a virtual salt flux using the fixed reference salinity:

.. math::

   F_{S} = -\frac{S_{\mathrm{ref}}}{\rho_{\mathrm{fw}}}
       (R + S_{\mathrm{snow}} - E).

Here ``E`` is positive for water leaving the ocean. The traced
``PhysicalConstants.surface_salinity_reference`` defaults to 34.7 g/kg and
``PhysicalConstants.freshwater_density`` to 1000 kg/m³. The resulting
``forc_salt_surface`` has units m (g/kg)/s: precipitation lowers salinity and
evaporation raises it. Veros applies its own tracer timestep and top-cell
depth. This follows the freshwater term of the
`rigid-lid conversion in Veros PR 81 <https://github.com/nutrik/veros/pull/81>`_,
implemented entirely in the VerCOR adapter.

The examples disable both climatological restoring terms by default.
``VerosConfig.restore_to_climatology_temp`` controls the SST correction,
and ``VerosConfig.restore_to_climatology_salt`` independently controls
30-day surface salinity restoring. Both default to ``False``.

.. list-table:: Independent global Veros restoring controls
   :header-rows: 1

   * - ``restore_to_climatology_temp``
     - ``restore_to_climatology_salt``
     - Enabled restoring
   * - ``False``
     - ``False``
     - None
   * - ``True``
     - ``False``
     - Temperature only
   * - ``False``
     - ``True``
     - Salinity only
   * - ``True``
     - ``True``
     - Temperature and salinity

These controls apply to the global setup during spinup and coupled stepping
in both host and JAX execution. The native callback applies the independent
weights; ``qnec`` remains the raw heat-correction coefficient even when its
contribution is disabled. Coupled heat and freshwater fluxes remain active.
Replace uses of the former shared ``restore_to_climatology`` argument with
the appropriate new argument(s). The ACC setup retains its native internal
forcing.

JCM
+++

`JAX-GCM <https://jax-gcm.readthedocs.io/en/latest/>`__ (JCM) is a differentiable atmospheric general circulation model written in JAX.
Its pluggable dynamical-core interface currently ships with the `Dinosaur <https://github.com/neuralgcm/dinosaur>`__
spectral backend and couples it to modular SPEEDY, Held-Suarez, and ECHAM-style physics packages.

CAMulator
+++++++++

`CAMulator <https://doi.org/10.48550/arXiv.2504.06007>`__ is an auto-regressive
machine-learned (ML) emulator of the `Community Atmosphere Model version 6 <https://www.cesm.ucar.edu/models/cam>`__ (CAM6)
that simulates the next atmospheric state given prescribed sea-surface temperatures
and incoming solar radiation. It is developed as part of the
`NSF NCAR Community Research Earth Digital Intelligence Twin project <https://miles-credit.readthedocs.io/en/latest/>`__.
CAMulator achieves these results with a 350 times speedup over CAM6, making it an efficient alternative for generating large ensembles.

CAMulator configuration paths and checkpoint paths are machine-specific.
Reconfigure those paths, their associated data, and the checkpoint before
running a CAMulator driver.

Veris forcing conventions
-------------------------

The Veris adapter reuses the Veros CESM bulk calculation. ``ATemp`` uses
``tref`` and ``aqh`` uses ``qref`` at
``PhysicalConstants.air_temperature_reference_height`` (2 m by default).
The bulk inputs retain colocated model-level temperature, humidity, wind,
density and height. These diagnostics use open-ocean similarity beneath
prescribed ice; they are not observed ERA5 near-surface values.

ERA5 separately supplies actual ``u10/v10`` winds, downward shortwave
``msdwswrf`` and downward longwave ``msdwlwrf``. Net shortwave ``msnswrf``
remains distinct. Native ``Qnet`` and ``Qsw`` negate VerCOR's downward-positive
heat fluxes. Separate rain and snow mass rates and signed CESM evaporation
convert to water-equivalent m/s using the traced freshwater density; runoff
and the prescribed ocean currents/height anomaly are explicitly zero.
Forcing is reapplied before each native substep because native thermodynamics
replaces heat inputs with residual coupling fluxes.

Veris with prescribed ocean
---------------------------

The standalone `Veris sea-ice model <https://github.com/team-ocean/veris>`_
uses the existing global 4-degree Veros ocean grid and its binary wet mask.
The example schedules ``LND -> CICE -> OCN -> ATM`` each day. Both incoming
ice routes are intensive, so wet cells with zero concentration still receive
forcing and can form ice. Concentration ``Area`` evolves as ``ice_fraction``
without rebuilding ocean geometry or remappers.

From a source checkout containing this integration, install and copy the setup:

.. code-block:: console

   python -m pip install ".[veris,veros]"
   vercor copy-setup run_veris_dataveros_era5 --to ~/vercor-setups/run_veris_dataveros_era5
   vercor run ~/vercor-setups/run_veris_dataveros_era5/run_veris_dataveros_era5.py --loglevel info --float-type float64

The native revision is pinned by the ``veris`` extra. Native integration uses
one-hour coupled dynamics/thermodynamics steps inside daily coupling intervals.
The default experiment starts with native zero ice and writes monthly native
ice variables plus final runtime/snapshot files. ``VerisConfig.initial_conditions``
accepts native physical (x, y) arrays, and ``VerisConfig.output`` selects period
variables/cadence. Thickness outputs remain volume per total grid-cell area.

The ocean provides monthly Veros SST/SSS climatology with explicitly zero
currents and height anomaly. Runoff is also prescribed zero. Land and atmosphere
are ERA5 data components; CICE sends no feedback to these prescribed components.
CESM reference-air diagnostics and heat/freshwater conventions are described
above. Numerical conservation, spherical operators, extended trajectories and local
derivatives are measured in :doc:`veris-validation`; these checks do not
establish observed climate skill.

Prognostic sea ice and ocean
----------------------------

``run_veris_veros_era5`` uses ``make_veris_veros`` to initialize one native ocean
and share its exact geometry with sea ice. The daily ``LND -> CICE -> OCN -> ATM``
sequence is explicitly lagged: ice receives the latest ocean/air state, ocean
receives the current ice interval means, then ERA5 advances. Heat, total bulk
mass and actual salt mass use a conservative scalar route. Stress uses the
vector-capable shared-grid identity route. Both outgoing routes use binary
source support and together carry ``VERIS_TO_OCEAN_FLUX_FIELDS``; concentration
is not applied twice. This same-grid proof does not establish different-grid
staggered stress conservation.

Synthetic parameter recovery
-----------------------------

``run_veris_calibration`` needs no external data assets. It fits h0 (m) from
synthetic concentration using public state replacement, a fixed wet observation
mask including ice-free cells, and a residual scale of 1e-3. Its reusable
``calibrate_lead_closing_thickness`` accepts a Coupler, fixed initial RunState,
observed_fraction, optional weights, initial_h0, positive bounds and a component
name. The default bounds are 0.25--0.8 m. Invalid observations/weights, all-dry
weights and an insensitive observation window fail before minimization.

The JAX-jitted value/gradient drives SciPy L-BFGS-B with ftol=1e-14, gtol=1e-10
and maxiter=100. The frozen PyTree result contains fitted h0, scaled loss and
gradient, optimizer iteration/evaluation counts and convergence. The example
prints recovery of truth 0.43 m and a distinct cold-forcing holdout error.
This validates numerical recovery rather than observed climate skill; float32
concentration quantization limits recovery accuracy relative to float64.

Coupled Veris, Veros and JCM
----------------------------

Copy ``run_veris_veros_jcm`` with the standard CLI, or import its public
``build_coupler`` function. The default is 365 daily noleap intervals starting
2000-01-03, after two days of native JCM climatological spinup. The builder
selects precision before constructing models, initializes the ocean once, and
reuses ``make_veris_veros`` and ``make_jcm_land_atmosphere``. Optional models
remain lazy imports. Inject ``clock``, native ``jcm_inputs`` or an already built
``pair`` for shorter runs; pair configuration overrides cannot accompany a pair.

The execution order is ``LND -> CICE -> OCN -> ATM``. Veris uses the latest
available ocean and atmospheric fields; its interval-mean total ocean heat,
mass, salt and stress enter Veros once. There is no direct ATM-to-OCN route.
JCM then receives current Veris ice temperature/albedo/concentration and
post-advance Veros open-water SST contributions. JCM land supplies separate
land temperature and soil availability. Land remains prescribed: receiving
heat fluxes does not make it a prognostic land model.

For wet mask W and current ice fraction A, conservative remapping transfers
``R(W*(1-A)*SST)``, ``R(W*A*Tice)``, ``R(W*A*alpha_ice)`` and ``R(W*A)``.
Products are formed on their source grid, preserving covariance. With native
land fraction fl, ocean fraction fo=1-fl and remapped ice fraction fi, JCM uses
marine temperature ``(water_temperature_contribution+ice_temperature_contribution)/fo``
and marine albedo ``((fo-fi)*alpha_water+ice_albedo_contribution)/fo``.
Concentration supplied to native JCM is ``fi/fo``. The configured native
open-water albedo is used wherever water is free of ice. Absent branches use
finite reference temperatures and zero inactive contributions. The existing
static mask helper's 0.001 threshold defines absent land/ocean support.

Enable the three explicit policies when supplying configurations:
``VerisConfig(atmosphere_forcing="model_level")``,
``VerosConfig(surface_forcing="coupled", export_open_water_surface=True)`` and
``JAXGCMConfig(seaice_coupling=True)``. Incompatible supplied modes fail early.
The builder requires noleap dates; coupled JCM binds its native 365-day ordinal
epoch and requires whole-second dates and spinup aligned to coupling intervals.
Initial air/radiation is evaluated from initialized native JCM physics without
advancing its dycore or cross-step carry. Its initial boundary is the native
climatology; subsequent ATM intervals use the incoming coupled surfaces. Thus
the first CICE interval has physical air/radiation and the explicit air lag is
retained. JCM model-level winds are converted to actual 10 m vectors using the
same CESM bulk solution that diagnoses temperature/humidity for Veris.

JCM's SPEEDY scheme has one effective marine temperature. This coupling does
not resolve separate marine turbulent fluxes over ice and water. Raw downward
shortwave radiation is exported separately: Veris applies the configured water
albedo to its open-water branch and native ice albedo to ice. JCM's already
combined turbulent flux channel is selected once; land receives its separate
per-land-area channels with downward-positive heat signs. Dynamic surface
weighting never adds another concentration or ocean-area factor.

Shared JCM period files add canonical ``sea_surface_temperature`` (K),
``marine_surface_albedo`` (1) and ``marine_ice_fraction`` (1) on ``lat, lon``;
native diagnostics remain available. Output is owned by the existing shared
coordinator. The two-day global smoke and smooth native AD tests establish
functional coupling and derivative behavior, not long-term climate accuracy.
