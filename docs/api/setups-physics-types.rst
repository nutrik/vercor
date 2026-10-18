Setups, physics, and types
==========================

Bundled setups
--------------

Use the lightweight slab factories directly; model-specific factories require
their documented optional packages or input data.

.. automodule:: vercor.setups
   :members: CAMulatorConfig, JAXGCMConfig, JCMLandAtmosphereConfig, JCMLandAtmosphereSetup, JCMInputs, Spinup, VerosConfig, VerisConfig, VerisVerosDataSetup, VerisVerosSetup, load_jcm_inputs, make_slab_atmosphere, make_slab_land, make_slab_ocean, make_slab_seaice, make_jcm_land_atmosphere, make_camulator_gcm, make_camulator_land, make_era5_atmosphere, make_era5_land, make_era5_ocean, make_erainterim_ocean, make_jax_gcm, make_jcm_land, make_veros_gcm, make_veris, make_veris_verosdata, make_veris_veros
   :show-inheritance:


``make_veris_verosdata`` creates standalone sea ice and a prescribed ocean
from one setup-only Veros global 4-degree snapshot, sharing its exact wet grid.
``make_veris`` accepts custom native ``OceanGeometry`` on the same physical grid.
Its U/V geometry uses native west/south faces. Canonical incoming ocean currents
use Veros east/north faces; packing closes coastal faces and shifts them once.
Install the pinned native model with ``pip install 'vercor[veris]'``; the paired
factory additionally needs the existing ``veros`` extra. Initialization uses a
host-compatible Veros process; start a fresh process if its JAX backend was
already locked by another model.

``VerisConfig.model_timestep`` defaults to one hour and must divide the positive
coupling interval. Runtime dtype and grid size remain coupler-owned. Native
initial conditions use unhaloed (x, y) arrays; ``hIceMean`` and ``hSnowMean``
are grid-cell volume-equivalent metres. Native numerical configuration and
physical constants are traced payload leaves; native state, forcing and ice fractions retain
JAX differentiation. No CICE feedback is sent to the prescribed ocean or ERA5.

Veris exports current ``ice_surface_temperature`` in kelvin and
``ice_surface_albedo`` per ice area, diagnosed with native snow thickness,
surface temperature, hemisphere and thickness categories. Both exports follow
the current payload; the five ocean boundary fluxes remain interval means.
``VERIS_TO_JCM_SURFACE_FIELDS`` requires source ice coverage, while
``VERIS_TO_JCM_COVERAGE_FIELDS`` uses binary wet support to transfer concentration
without multiplying it by itself.

``VerisConfig.atmosphere_forcing`` defaults to ``"observed_10m"``. The
``"model_level"`` mode uses ``JCM_ATMOSPHERE_TO_VERIS_FORCING_FIELDS`` and
diagnoses actual 10 m winds from the same converged CESM solve used for air
temperature, humidity and open-water fluxes. It retains the moving-ocean
reference frame and uses ``(1 - open_water_albedo) * SW_down`` for open water.
Raw downward shortwave remains the separate native ice forcing.

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

Veris parameter updates
-----------------------

``VerisConfig.physical_constants`` supplies independent native physical values
separately from numerical ``custom_parameters``. It is keyword-only and owns
copies of caller mappings/arrays. Start calibration candidates from one fixed
initial state with ``replace_veris_parameters(initial, {"h0": value})``, then
call ``coupler.run(candidate)``. The helper is immutable and supports JIT, JVP
and VJP. Geometry, precision and hard binary transitions retain their native
meaning. VerCOR's traced ``freshwater_density`` owns native rhoFresh and its
derived ratios at setup and every step; direct helper overrides are rejected.

.. autofunction:: vercor.setups.replace_veris_parameters

Veris ocean feedback
--------------------

``VERIS_TO_OCEAN_FLUX_FIELDS`` contains downward heat (W/m2), total bulk mass
(kg/m2/s), actual salt mass (kg salt/m2/s), and east/north total stress (Pa).
These are interval means over all native substeps on the ocean's shared grid.
Ice traction uses the momentum-time face concentration and complementary
open-water traction from one shared bulk calculation. Export these total-area
fields with ``quantity="source_flux", source_coverage="binary"`` so complete
melt does not suppress ocean input or apply ice concentration twice. The final
instantaneous native diagnostics remain separate from the canonical means.

Total feedback into Veros
-------------------------

``VerosConfig(surface_forcing="coupled")`` selects total ocean boundary inputs
from ``VERIS_TO_OCEAN_FLUX_FIELDS`` for the global setup. It requires pressure/
free-surface execution (``enable_streamfunction=False``) to export actual native
SSH; ACC retains internal forcing and rejects this mode. The default atmosphere
mode preserves existing behavior. Coupled mode exports SST in K, SSS in native
g/kg, actual east/north surface currents in m/s and native SSH in m. Streamfunction
or surface pressure is never used as SSH.

Native ocean kernels apply timestep, layer depth, ocean density and heat capacity
once. Virtual salt input is ``(-S_ref*M + 1000*J)/rho_fw`` in (g/kg)*m/s using
fixed reference salinity (34.7 g/kg by default), total downward bulk mass M and
actual downward salt mass J in kg/m2/s. This Boussinesq reference budget differs
from physical ocean salt inventory when local salinity differs from S_ref.
No atmospheric bulk, rain, snow or evaporation is recomputed in this mode.

Temperature and salinity restoring remain independent. Omitting the coupled
qnec coefficient preserves its native climatology. Climatological spinup starts
with total feedback inactive and retains the native cold-water cooling proxy.
Installing total feedback activates it even for zero input; thereafter the
proxy cannot discard heat or salt already diagnosed by the sea-ice model.

Shared prognostic setup
------------------------

``make_veris_veros(config=VerisConfig(...), ocean_config=VerosConfig(surface_forcing="coupled"))``
returns frozen ``VerisVerosSetup(seaice, ocean)``. It initializes one native ocean,
extracts its authoritative edges, areas, radius and masks once, and shares the
exact grid object with sea ice. The ocean retains the existing setup/runtime
owner; no second ocean or prescribed current/height proxy is created. The default
ocean configuration is global coupled mode; incompatible explicit modes fail
before initialization. The prescribed ``make_veris_verosdata`` workflow remains
available for experiments without ocean feedback.
