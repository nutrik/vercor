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
