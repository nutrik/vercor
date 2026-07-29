Run packaged setup gallery
==========================

VerCOR packages longer runnable setup scripts in ``vercor.setups.gallery`` in
addition to the small, dependency-light programs in the learning paths. Copy a
setup into a working directory, then run the copied script:

.. code-block:: console

   vercor copy-setup run_jcm_with_veros
   vercor run run_jcm_with_veros.py

``copy-setup`` never overwrites an existing destination. The copied script is
your local, user-editable setup; modify it before running it again as needed.

The table identifies the extra dependency and input-data expectation for each
setup. ``None`` means that the setup has no model-specific optional package
or external input requirement, although plotting examples need
`Matplotlib <https://matplotlib.org>`_.


.. list-table:: Packaged setup gallery
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

CAMulator configuration paths and checkpoint paths are machine-specific.
Reconfigure those paths, their associated data, and the checkpoint before
running a CAMulator driver.
