Exchange coverage and quantity semantics
========================================

Ownership
---------

A component's grid declares immutable binary support. ``ComponentMaskSpec``
references current grid-shaped binary and fractional fields; fractions must be
finite, in [0, 1], and zero outside static support. Declared mask fields remain
current even when other data fields select daily or interpolated monthly records.
JCM and CAMulator land factories retain their computed ``land_fraction``.

Topology validates geometry and may restrict binary route support. It cannot
provide fractional coverage or expand the target's static support. The existing
``SurfaceMaskPolicy`` derives static geometry from an ocean wet mask and checks
the corresponding land grid. Shared native JCM/slab tiles instead declare their
native fractions directly and use no ocean-binary-derived policy.

The runtime resolves coverage and remapped overlap after lifecycle boundaries.
Current route coverage and the coverage saved with the last dispatch are separate:
a source can disappear or reactivate before the recipient runs without changing
the validity of that recipient's stored imports.

Formulas
--------

Every ``Exchange`` requires ``quantity``. Let ``s`` be source coverage, ``b`` target
binary support including topology restrictions, ``t`` target fractional coverage,
and ``R`` the configured remapper. Define ``d = R(s)`` and
``N = R(where(s > 0, x, 0) * s)``. On supported targets with ``d > 0``:

.. list-table::
   :header-rows: 1

   * - Quantity
     - Received value
     - Meaning
   * - ``intensive``
     - ``N / d``
     - Local value per covered area
   * - ``source_flux``
     - ``N * b``
     - Source-area contribution per target grid-cell area
   * - ``target_flux``
     - ``(N / d) * t``
     - Local value weighted by target coverage

Empty overlap returns finite zero values and derivatives. The same dispatcher
weights both vector components before vector remapping/rotation. Inactive source
NaNs are neutralized before multiplication; active NaNs and all infinities fail
validation. Diagnostic coverage is bounded to [0, 1]; physical numerators and
denominators are not clipped. Conservative remapping preserves the integral of
source contributions where target support covers the source domain.

``source_coverage="binary"`` uses component binary support in place of its
fraction. Use this for fields whose source already formed the area product;
applying component fractional coverage again would square that area factor.

Bundled field audit
-------------------

.. list-table::
   :header-rows: 1
   :widths: 38 22 40

   * - Fields and consumers
     - Declaration
     - Physical reason
   * - Atmospheric wind, air state, pressure, radiation and precipitation;
       ocean SST, salinity, currents and height; land temperature and moisture
     - ``intensive``
     - Native closures consume local values or local flux densities. A field
       named flux does not by itself require an area contribution.
   * - CAMulator land temperature and SST
     - ``source_flux``; conservative
     - The existing consumer sums land/ocean temperature contributions.
   * - Veris temperature and albedo to coupled JCM
     - ``source_flux``; component coverage
     - Forms ``R(W*A*Tice)`` and ``R(W*A*alpha)`` before marine composition.
   * - Veris concentration to JCM; ocean open-water temperature contribution
     - ``source_flux``; binary coverage
     - Source values already contain ``A`` or ``(1-A)*SST``.
   * - Veris total ocean heat, mass, salt and stress feedback
     - ``source_flux``; binary coverage
     - Veris already combines ice and open-water area contributions once.
   * - Concentration and ocean state sent over the full wet domain
     - ``intensive``; binary coverage
     - Zero ice concentration and under-ice ocean state remain defined.

No bundled consumer needs ``target_flux``. Scalar/vector tests cover all three
semantics, fractional coasts, evolving and zero coverage, conservation and AD.

Input use and component substitution
------------------------------------

Prescribed ERA5 atmosphere/land/ocean, ERA-Interim ocean and CAMulator land
factories have no physical feedback inputs. Their outputs follow the prescribed
records, even when another component changes. The data gallery explicitly wraps
its factory declarations in ``DataComponent`` declarations with diagnostic
imports: those received fields feed its tables and comparison plots, not a
prognostic tendency. General data imports and forwarding remain supported.

.. list-table::
   :header-rows: 1
   :widths: 26 36 38

   * - Producer -> consumer
     - Fields / units
     - Input use and sign
   * - Slab atmosphere -> slab ocean
     - sensible/latent heat, W/m²
     - Both enter the mixed-layer heat tendency; positive warms ocean.
   * - Slab atmosphere -> slab land
     - latent heat, W/m²
     - Positive increases soil moisture; negative dries it.
   * - Ocean -> slab atmosphere / sea ice
     - SST, K
     - Air heat exchange / ice-fraction diagnosis; neither imports land soil
       moisture or unused ice feedback.
   * - ERA5 / JCM / supported custom atmosphere -> Veros / Veris
     - T and theta, K; q, kg/kg; density, kg/m³; height, m; winds, m/s
     - Colocated air feeds the same CESM bulk closure. A colder, drier air
       column gives negative downward sensible and latent heat over warm water.
   * - Atmosphere -> Veros / Veris
     - shortwave/longwave, W/m²; rain/snow, kg/m²/s
     - Radiation and precipitation are downward positive. CESM evaporation
       is downward negative. Veros virtual salt is
       ``-34.7 * (rain + snow + evap) / 1000`` in m (g/kg)/s;
       net freshwater input freshens the ocean.
   * - Prescribed / prognostic Veros -> Veris
     - SST, K; SSS, g/kg; native face currents, m/s; SSH, m
     - Shared native geometry and explicit face conversion; prognostic
       velocities and free surface replace prescribed proxies without
       changing physical field meaning.

The substitution matrix independently constructs data, callable JAX, structural
JAX and host producers with identical physical exports, exercises their steps,
and checks received fields and CESM heat/mass signs. Shared independent coastal
expectations cover three quantities, nonuniform fractions, rotated vectors,
inactive NaNs, disappearance/reactivation and retained received support.
Preweighted Veris exports use binary source coverage and receive one area
factor. Continuous JAX fields and fractions retain analytic JVP/VJP and finite
difference checks; binary transitions are discrete. Host components retain
physical parity on the host path and fail early on the differentiable backend.
Native ERA5/JCM conversions, Veros/Veris face geometry and CAMulator vapor-species
capability checks remain separate regressions. No native CAMulator execution is
certified by synthetic substitution.

Intentional numerical and API changes
-------------------------------------

There is no default or ``legacy`` quantity and no topology-owned fractional map.
Explicit declarations replace the previous generic post-remap attenuation.
Ordinary JCM receives independent intensive land temperature, SST and soil
availability. Its diagnostic total temperature is ``f_land*Tland + (1-f_land)*SST``;
land and sea temperatures are no longer summed and supplied to both native tiles.
Absent tiles use the existing 288.15 K reference to keep native branches finite.
The standalone land default uses that reference rather than zero kelvin.

JCM 3.0.0 uses Gregorian time and an explicit native time/step carry. Resumed
runs preserve that clock, including leap days. ``SetupContext.calendar`` allows
setup-time rejection of unsupported calendars. Native public humidity is kg/kg;
SPEEDY evaporation and precipitation diagnostics still require gram conversion.
JCM 3 publishes grid-mean turbulent fluxes and removed separate land/marine
channels. Unused imports into prescribed JCM land and the unavailable land-only
recipe are removed. The JCM/slab demonstration applies grid-mean turbulent flux
densities to both toy tiles; it does not certify separate tile heat budgets.
Slab turbulent heat fluxes are downward-positive W/m²: sensible heat is
``10 * (T_air - SST)`` and latent heat is half of sensible heat. Positive
fluxes warm the mixed layer; the independent ocean restoring tendency is
``-(SST - 288.15 K) / (30 days)``. Slab soil moisture changes by
``1e-9 * latent_heat_flux * dt_seconds``, clipped to [0, 1], so negative
latent heat (evaporation) dries soil and positive latent heat wets it.
The latent and moisture coefficients remain toy closures and do not certify
a physical water budget.
CAMulator's existing ML land-mask branch is retained. Its inverse-transformed
``FSNS`` and ``FLNS`` are downward-positive energies in J/m² accumulated over
one native forecast step. The mapper requires ``native_step_seconds`` from the
existing model timestep (``3600 * data.lead_time_periods``), independently of
how many native steps fit in a coupling interval. It rejects non-positive and
non-finite durations, divides each energy by that native duration once, and
recovers downward longwave as ``sigma * TS**4 + FLNS / native_step_seconds``.
Surface emission is added exactly once. The adapter does not infer rate units
or guess a different training contract.
Startup uses timestamp-matched native ``TS/FSNS/FLNS`` diagnostics for channels
absent from the initial input, with the same forecast-step duration. Initial
and predicted meteorology share one pure hybrid-coordinate mapping, including
canonical ``surface_pressure`` and lowest-level ``pressure`` in Pa. Specific
humidity must come from an explicitly declared vapor-specific native channel
available in both input and prediction; ``Qtot`` is rejected as total water.

This follows CREDIT revision
``bc078d7a39300fb8c8fe867a22ec218a5e02da11``:
`native energy conversion <https://github.com/NCAR/miles-credit/blob/bc078d7a39300fb8c8fe867a22ec218a5e02da11/credit/postblock/gen1.py>`_,
`energy units and signs <https://github.com/NCAR/miles-credit/blob/bc078d7a39300fb8c8fe867a22ec218a5e02da11/credit/postblock/conservation.py>`_,
and `CAMulator timing <https://github.com/NCAR/miles-credit/blob/bc078d7a39300fb8c8fe867a22ec218a5e02da11/climate/camulator_config.yml>`_.
The 1-, 3-, and 6-hour mapper checks and two-substep host runtime test use
stand-in predictions. CREDIT and its checkpoint are unavailable locally;
these checks certify conversion and adapter behavior, not native climate skill.

Released models and the dependency exception
--------------------------------------------

The Veris extra selects tag ``v0.2.1``, commit
``6e3b56ee14922f4bae760d57772f610078aa1886``. JCM is pinned to ``3.0.0`` and
Dinosaur is at least ``1.5.0``. Veris requires JAX >=0.11.2; JCM declares
``jax>=0.10,<0.11``. These upstream constraints have no common solution.

The reviewed combined validation lane uses exactly JAX 0.11.2. From the checkout:

.. code-block:: console

   conda activate scipy
   python tools/install_coupled_models.py

The installer resolves every other JCM base requirement normally, installs the
released JCM wheel without resolving its incompatible upper bound, and requires
``pip check`` to report exactly that one known mismatch. Other conflicts fail.
Normal JCM-only and Veris-only extras retain their upstream constraints; the
combined extras are not presented as normally resolvable.

Acceptance uses released native model rollouts, independent surface/output
checks, conservation, JIT/JVP/VJP and finite differences, plus separate stock and
exact-fork Veros test processes. This establishes the tested functional coupling;
it does not certify long-term climate behavior or remove the upstream metadata
conflict. Final commands and results are recorded in ``PROGRESS.md``.

Validation on 2026-10-19 passed 2,209 full stock tests, 1,284 fast tests and
15 exact-fork native AD tests, with no skips. Combined statement/branch coverage
is 92.48% (94.60% statements, 82.69% branches). The source and isolated wheel
CLI-copy global two-day runs match exactly across seven saved fields: 2,319 wet
ocean cells, 48 guarded native ice steps, nine routes and 13 NetCDF files.
Independent review's obsolete AD fixture route was reproduced and removed;
all native gradient assertions remain. Black, maintained-source flake8, mypy,
compileall, strict Sphinx, wheel/sdist metadata and installed extension/CLI
boundaries pass. Validation used Python 3.12.15 in the isolated conda scipy
environment with Dinosaur 1.5.1 and Flax 0.12.10.
