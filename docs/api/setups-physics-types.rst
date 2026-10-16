Setups, physics, and types
==========================

Bundled setups
--------------

Use the lightweight slab factories directly; model-specific factories require
their documented optional packages or input data.

.. automodule:: vercor.setups
   :members: CAMulatorConfig, JAXGCMConfig, JCMLandAtmosphereConfig, JCMLandAtmosphereSetup, JCMInputs, Spinup, VerosConfig, VerisConfig, VerisVerosDataSetup, load_jcm_inputs, make_slab_atmosphere, make_slab_land, make_slab_ocean, make_slab_seaice, make_jcm_land_atmosphere, make_camulator_gcm, make_camulator_land, make_era5_atmosphere, make_era5_land, make_era5_ocean, make_erainterim_ocean, make_jax_gcm, make_jcm_land, make_veros_gcm, make_veris, make_veris_verosdata
   :show-inheritance:


``make_veris_verosdata`` creates standalone sea ice and a prescribed ocean
from one setup-only Veros global 4-degree snapshot, sharing its exact wet grid.
``make_veris`` accepts custom native ``OceanGeometry`` on the same physical grid.
Install the pinned native model with ``pip install 'vercor[veris]'``; the paired
factory additionally needs the existing ``veros`` extra. Initialization uses a
host-compatible Veros process; start a fresh process if its JAX backend was
already locked by another model.

``VerisConfig.model_timestep`` defaults to one hour and must divide the positive
coupling interval. Runtime dtype and grid size remain coupler-owned. Native
initial conditions use unhaloed (x, y) arrays; ``hIceMean`` and ``hSnowMean``
are grid-cell volume-equivalent metres. Native numerical configuration and
physical constants are static; native state, forcing and ice fractions retain
JAX differentiation. No CICE feedback is sent to the prescribed ocean or ERA5.

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
