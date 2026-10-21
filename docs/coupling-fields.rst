Coupling fields and naming migration
====================================

Bundled exchange names describe a physical quantity and its location. Concrete
model names remain on native adapters, configuration and dependencies. A matching
name alone does not establish compatibility: producers and consumers must also
agree on units, vertical level, grid geometry, coverage and time meaning.

Lowest-model-level air
----------------------

The bulk meteorology contract uses colocated air at ``model_level_height``
(metres above the surface). ERA5, JCM and compatible CAMulator inputs provide
the following horizontal fields. JCM and CAMulator map these to
``(latitude, longitude)`` arrays.

.. list-table:: Air exchange-key migration
   :header-rows: 1
   :widths: 25 35 40

   * - Former key
     - Current key
     - Units and meaning
   * - ``temperature``
     - ``air_temperature``
     - K; lowest-level air temperature
   * - ``density``
     - ``air_density``
     - kg/m³; colocated air density
   * - ``u_velocity`` / ``v_velocity``
     - ``air_u_velocity`` / ``air_v_velocity``
     - m/s; colocated horizontal air velocity vector
   * - ``potential_temperature``
     - ``air_potential_temperature``
     - K; colocated air potential temperature
   * - ``specific_humidity``
     - ``air_specific_humidity``
     - kg/kg; water-vapor specific humidity

These keys do not imply a 2 m temperature or 10 m wind. Actual observed
``temperature_2m`` and ``u_velocity_10m`` / ``v_velocity_10m`` keep their names.
Generic components may still declare their own fields such as ``temperature``;
the migration applies to the bundled physical air contract.

``VerisConfig(atmosphere_forcing="observed_10m")`` imports both model-level air
and separate observed 10 m winds. ``atmosphere_forcing="model_level"`` instead
diagnoses 10 m winds through the same CESM bulk solve that supplies reference
air temperature/humidity and open-water fluxes, retaining the moving-ocean
reference frame. Neither mode relabels a model-level wind as an observation.

Pressure and precipitation phase
--------------------------------

The former ``pressure`` key had different ranks in different adapters:

* JCM exports ``air_pressure_3d`` in Pa with shape
  ``(sigma_level, latitude, longitude)``. Native sigma order is preserved;
  the final level supplies the lowest-level thermodynamic calculations.
* CAMulator exports ``model_level_air_pressure`` in Pa with shape
  ``(latitude, longitude)`` for its lowest hybrid-sigma level.
* ``surface_pressure`` remains the distinct horizontal surface pressure in Pa.

These are renamed existing arrays; the migration adds no pressure export.
JCM's phase-only mapper keyword is ``near_surface_air_temperature``, formerly
``temperature_2m``. Its input remains native SPEEDY ``surface_flux.t0`` on the
native grid, with no specified 2 m height. It partitions precipitation into
rain and snow before transposition/remapping. The six air fields above still
use the bottom sigma-center state for bulk fluxes; phase temperature does not
replace them. Native JCM and CAMulator variable names remain unchanged at their
translation boundaries. CAMulator humidity requires a declared vapor-specific
channel; total water ``Qtot`` cannot substitute for vapor.

Recipes and setup names
-----------------------

Update imports, exchange declarations, state lookups and diagnostics together.
The former names have no compatibility aliases.

.. list-table:: Renamed recipes in ``vercor.recipes``
   :header-rows: 1
   :widths: 50 50

   * - Former recipe
     - Current recipe
   * - ``ATMOSPHERE_TO_VEROS_FORCING_FIELDS``
     - ``ATMOSPHERE_TO_OCEAN_BULK_FORCING_FIELDS``
   * - ``ATMOSPHERE_TO_VERIS_FORCING_FIELDS``
     - ``OBSERVED_10M_ATMOSPHERE_TO_SEAICE_FORCING_FIELDS``
   * - ``JCM_ATMOSPHERE_TO_VERIS_FORCING_FIELDS``
     - ``MODEL_LEVEL_ATMOSPHERE_TO_SEAICE_FORCING_FIELDS``
   * - ``OCEAN_TO_VERIS_FORCING_FIELDS``
     - ``OCEAN_TO_SEAICE_FORCING_FIELDS``
   * - ``VERIS_TO_OCEAN_FLUX_FIELDS``
     - ``SEAICE_TO_OCEAN_TOTAL_FLUX_FIELDS``
   * - ``VERIS_TO_JCM_SURFACE_FIELDS``
     - ``SEAICE_TO_ATMOSPHERE_SURFACE_FIELDS``
   * - ``VERIS_TO_JCM_COVERAGE_FIELDS``
     - ``SEAICE_COVERAGE_FIELDS``
   * - ``OCEAN_TO_JCM_OPEN_WATER_FIELDS``
     - ``OCEAN_TO_ATMOSPHERE_OPEN_WATER_FIELDS``
   * - ``JCM_LAND_TO_ATMOSPHERE_FIELDS``
     - ``LAND_TO_ATMOSPHERE_STATE_FIELDS``
   * - ``SLAB_ATMOSPHERE_TO_OCEAN_FLUX_FIELDS``
     - ``ATMOSPHERE_TO_OCEAN_TURBULENT_HEAT_FLUX_FIELDS``
   * - ``SLAB_ATMOSPHERE_TO_LAND_FLUX_FIELDS``
     - ``ATMOSPHERE_TO_LAND_LATENT_HEAT_FLUX_FIELDS``

The specifically JCM-backed public setup is now
``make_jcm_atmosphere(coords, terrain, config=JCMAtmosphereConfig(...))``, replacing
``make_jax_gcm`` and ``JAXGCMConfig``. Its private setup/runtime owners are
``JCMAtmosphereSetupState`` and ``JCMAtmosphereRuntimePayload``; existing private
module paths remain unchanged. These names identify the implementation, while
the reusable recipes identify the exchanged physical contract.

For ``VerosConfig.surface_forcing``, replace ``"atmosphere"`` with
``"bulk_meteorology"`` (the default), and ``"coupled"`` with
``"surface_fluxes"``. Global bulk meteorology computes the existing CESM
surface forcing from air, radiation and precipitation.
``"surface_fluxes"`` consumes complete sea-ice/open-water boundary rates once,
requires the global setup and pressure/free-surface execution, and retains
independent temperature/salinity restoring. ACC retains native internal forcing
and rejects ``"surface_fluxes"``. ``export_open_water_surface=True`` requires
``"surface_fluxes"``. The corresponding private predicate is
``uses_bulk_meteorology``.

The public grid-mask helpers migrate as follows:

* ``compute_ocn_lnd_masks_on_atm_grid`` →
  ``compute_ocean_land_masks_on_atmosphere_grid``
* ``check_total_lnd_ocn_mask_sum`` → ``check_land_ocean_fraction_sum``
* ``create_lnd_mask_from_ocn`` → ``create_land_mask_from_ocean``

They retain the existing complementary land/ocean topology and 0.001 support threshold.
The first returns ocean fraction, land fraction and land binary support on the
atmosphere grid; the last returns land binary support and land fraction.
CAMulator's private ``prepare_camulator_sst_input`` becomes
``prepare_camulator_surface_temperature_input``: its input is rescaled combined
land/ocean surface temperature, rather than SST alone.

Fluxes, coverage and geometry
-----------------------------

.. list-table:: Bundled field meanings retained by the migration
   :header-rows: 1
   :widths: 38 22 40

   * - Fields
     - Units
     - Location, sign or time meaning
   * - ``sea_surface_temperature`` / ``sea_surface_salinity`` /
       ``sea_surface_height``
     - K / g/kg / m
     - Ocean tracer surface; actual native SSH in prognostic feedback mode
   * - ``ocean_u_velocity`` / ``ocean_v_velocity``
     - m/s
     - Canonical east/north ocean faces
   * - Radiation; ``rain_rate`` / ``snow_rate``
     - W/m²; kg/m²/s
     - Downward positive; snow is water-equivalent mass
   * - ``sensible_heat_flux`` / ``latent_heat_flux``
     - W/m²
     - Downward positive; JCM 3 exports grid means
   * - ``sea_surface_heat_flux``
     - W/m²
     - Downward into ocean; total-area interval mean
   * - ``sea_surface_mass_flux`` / ``sea_surface_salt_mass_flux``
     - kg/m²/s / kg salt/m²/s
     - Downward into ocean; total-area interval means
   * - ``ocean_u_stress`` / ``ocean_v_stress``
     - Pa
     - Signed east/north total-area stress interval means
   * - ``ice_surface_temperature`` / ``ice_surface_albedo``
     - K / 1
     - Current properties per ice-covered area
   * - ``ice_fraction``
     - 1
     - Current ice concentration on the wet ocean domain
   * - ``open_water_temperature_contribution``
     - K
     - Current ``(1-A)*SST``, formed after the ocean advance

Atmospheric bulk inputs and ocean state use ``quantity="intensive"``. Ice
surface properties sent to JCM use ``quantity="source_flux"`` with component
ice coverage, forming ``R(W*A*Tice)`` and ``R(W*A*albedo)``. Concentration,
the preweighted open-water contribution, and total ocean feedback use binary
source coverage: their values already contain their area factors. Sending
concentration or ocean state across the full wet domain also uses binary
coverage so zero ice does not remove valid forcing.

Source products are formed before remapping. JCM's marine compositor divides
the received marine contributions by ocean fraction once. See
:doc:`exchange-coverage` for the exact ``intensive``, ``source_flux`` and
``target_flux`` formulas, inactive-value handling and normalization ownership.

Ocean currents retain Veros east/north-face staggering. The Veris boundary
converts geometry, masks and currents to native west/south faces once, closes
coastal faces, and converts ocean feedback stresses back to east/north faces.
A generic recipe name does not accept arbitrary staggering or establish
staggered conservation across different ocean grids.

Current component coverage is refreshed at lifecycle boundaries. Stored imports
retain the support recorded at their last dispatch, even if the source changes
coverage before the recipient steps. The coupled gallery retains the physical
land → ice → ocean → atmosphere order, with routes, state lookup and topology
roles derived from actual component names. Configurable instance names such as
``ATM`` and ``OCN`` remain instance identities rather than physical field names.
