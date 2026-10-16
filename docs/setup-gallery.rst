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

The examples disable climatological restoring by default. Set
``VerosConfig(restore_to_climatology=True)`` to enable the existing SST
correction and add 30-day surface salinity restoring to the coupled salt
flux. The ACC setup retains its native internal forcing.

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
