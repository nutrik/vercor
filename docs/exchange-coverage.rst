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
CAMulator's existing ML land-mask branch is retained.

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
