Run packaged setup gallery
==========================

.. note::

   The setup-gallery CLI is unreleased development functionality. PyPI
   ``vercor==0.4.3`` does not contain these commands; use a current
   development-source installation until a later published release includes
   them.

The development version packages longer runnable setup scripts in
``vercor.setups.gallery`` in addition to the small, dependency-light programs
in the learning paths. List the available templates, copy one into a working
directory, then run the copied script:

.. code-block:: console

   vercor show-setups
   vercor copy-setup run_jcm_with_veros \
     --to ~/vercor-setups/run_jcm_with_veros
   vercor run \
     --loglevel info \
     --float-type float64 \
     ~/vercor-setups/run_jcm_with_veros/run_jcm_with_veros.py

``vercor --version`` reports the installed distribution version. To include
your own template directories, set ``VERCOR_SETUP_DIR`` to an
``os.pathsep``-separated list of direct directories (``:`` on POSIX and ``;``
on Windows). The catalog includes only direct public ``.py`` files. Every
template name must be unique across those directories and the packaged gallery;
a duplicate is an error instead of selecting a source implicitly.

``copy-setup --to`` creates missing parent directories and reuses an existing
directory, but opens its destination exclusively and never overwrites a file.
The copied script is your local, user-editable setup; modify it before running
it again as needed. ``run`` accepts only lowercase ``trace``, ``debug``,
``info``, ``warning``, and ``error`` log levels (default ``info``), and
``float64`` or ``float32`` precision choices (default ``float64``).

External templates use the same child-process contract. They must define
exactly the keyword-only callable below; it may return ``None`` for success or
an integer process status:

.. code-block:: python

   def run_setup(*, loglevel: str, float_type: str) -> int | None:
       ...

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
