Setups, physics, and types
==========================

Bundled setups
--------------

Use the lightweight slab factories directly; model-specific factories require
their documented optional packages or input data.

.. automodule:: vercor.setups
   :members: CAMulatorConfig, JAXGCMConfig, JCMLandAtmosphereConfig, JCMLandAtmosphereSetup, JCMInputs, Spinup, VerosConfig, load_jcm_inputs, make_slab_atmosphere, make_slab_land, make_slab_ocean, make_slab_seaice, make_jcm_land_atmosphere, make_camulator_gcm, make_camulator_land, make_era5_atmosphere, make_era5_land, make_era5_ocean, make_erainterim_ocean, make_jax_gcm, make_jcm_land, make_veros_gcm
   :show-inheritance:

Physics
-------

Use physical constants as traced inputs shared across component steps.

.. automodule:: vercor.physics
   :members: PhysicalConstants
   :show-inheritance:

Precision policy
----------------

Use ``DTypePolicy`` as ``RuntimeOptions.dtype`` to select VerCOR's real-array
precision while keeping index arrays 32-bit.

.. automodule:: vercor.dtypes
   :members: DTypePolicy
   :show-inheritance:

Runtime array type
------------------

Use this alias when annotating arrays that may be NumPy or JAX values at a
public runtime boundary.

.. automodule:: vercor.types
   :members: RuntimeArray
   :show-inheritance:
