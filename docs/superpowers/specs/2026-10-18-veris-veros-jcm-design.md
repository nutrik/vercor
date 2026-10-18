# Coupled Veris, Veros and JCM gallery

Status: adapter-based approach approved by the user on 2026-10-18. The user's
requested revision is applied: native Veris changes must use the existing
`feat/vercor-scientific-validation` branch, starting from its latest remote
commit before changes. The user approved this revised written specification
on 2026-10-18 and requested the implementation plan.

## Intended result and scope

Add the packaged, copyable `vercor/setups/gallery/run_veris_veros_jcm.py` setup.
It composes JCM land (`LND`), standalone Veris (`CICE`), prognostic global-four-degree
Veros (`OCN`) and JCM atmosphere (`ATM`), in exactly that execution order. Reuse
`make_veris_veros`, `make_jcm_land_atmosphere`, existing explicit exchange
quantities, native model transitions, CESM bulk physics, immutable runtime
payloads and the shared output coordinator. No new workflow, generic fan-in
reducer or component lifecycle is needed.

VerCOR changes, verification, commits and the final push stay on the existing
`feat/veris-seaice` branch. A small Veris source extraction uses an isolated
temporary checkout on the existing `feat/vercor-scientific-validation` branch
of `team-ocean/veris`. Fetch and resolve that branch's latest remote tip
immediately before native changes, and record the actual starting SHA. Do not
start from VerCOR's older dependency pin merely because it was the audit base.
Preserve the user's separate editable Veris checkout.

GitHub CLI verified remote tip `6c205a356dd1b25c26430f899cb8998523ab3447` on
2026-10-18. It is one commit ahead of VerCOR's current native pin
`4551fc86181c660ceb0010160ec7f569d97e52c6`; its changes are validation records,
the changelog and a GPU runner test expectation, with no solver changes.
Recheck the remote before editing and review any further intervening changes.
Push verified native changes back to `feat/vercor-scientific-validation` with
an ordinary fast-forward push before pinning VerCOR to its exact reachable
commit. If the remote advances, incorporate its new tip and repeat affected
verification; do not force-push. Use GitHub CLI authentication and verify both
remote SHAs. No changes to Veros or JCM source, release, tag or merge are required.

## Evidence and approach

The prognostic Veris/Veros pair already owns one ocean initialization, exact
shared geometry and interval-mean total ocean boundary fluxes. Its scalar
feedback uses conservative `source_flux` exchanges with binary source support;
its staggered stress feedback uses the vector-capable shared-grid identity.
Keep those numerical contracts.

The JCM audit identified four relevant adapter gaps: imported soil moisture is
not applied to native forcing; coastal land coverage is replaced by binary
land support; land and ocean temperatures are summed and reused for both native
surface types; and turbulent flux mapping sums land, marine and the already
combined native channels. Native SPEEDY also assumes constant ice albedo.
Correct the channel-selection defect with a regression, and provide explicit
sea-ice coupling for the new surface path. Keep unrelated gallery behavior
outside that path unchanged.

A gallery-only wrapper would duplicate model/runtime logic. Copying native
thermodynamics or replacing SPEEDY radiation would create competing physics
owners. The selected approach extends the existing adapters at their surface
boundaries and uses JCM's native composable forcing term.

## 1. Surface ownership and conservative composition

Let `W` be the static native ocean wet mask, `A` the current Veris concentration,
and `R` the existing conservative scalar remapper to the atmosphere grid.
`T_w` is the current post-step Veros SST, `T_i` native Veris surface temperature,
`alpha_i` its diagnosed ice/snow albedo and `alpha_w` JCM's configured open-water
albedo (`mod_radcon.albsea`).

The atmosphere receives separate, distinctly named contributions:

| Source | Quantity on the atmosphere grid | Exchange support |
| --- | --- | --- |
| Veros | `R(W * (1-A) * T_w)` | Binary source, `source_flux` |
| Veris | `R(W * A * T_i)` | Component coverage, `source_flux` |
| Veris | `R(W * A * alpha_i)` | Component coverage, `source_flux` |
| Veris | `R(W * A)` | Binary source, `source_flux` |
| JCM land | Intensive land temperature and soil moisture | Land support, `intensive` |

Products are formed before remapping. In particular, do not substitute
`R(A)*R(T)` for `R(A*T)`, or multiply an already weighted contribution by
concentration a second time. Source-flux here denotes the existing area-weighted
transfer operation; temperature contributions are in kelvin times area
fraction, not heat flux.

Derive static ocean fraction `f_o` and land fraction `f_l=1-f_o` through the
existing ocean-to-atmosphere mask helper. Define `f_i=R(W*A)` and
`f_w=f_o-f_i`. Supply native JCM with:

```
T_marine = (R(W*(1-A)*T_w) + R(W*A*T_i)) / f_o
alpha_marine = (alpha_w*f_w + R(W*A*alpha_i)) / f_o
A_marine = f_i / f_o
```

Use safe masked division at zero ocean coverage. Follow the existing helper's
static coverage threshold convention: marine contributions are inactive where
that helper classifies ocean coverage as absent. Test threshold endpoints as
well as ordinary fractional coasts. Native JCM receives the
independent intensive `T_land` in `stl_am`, `T_marine` in
`sea_surface_temperature`, JCM-land moisture in `soilw_am`, the actual
`A_marine` in `sice_am`, and fractional `f_l` in `terrain.fmask`. Its native
land albedo/snow forcing remains owned by JCM. Diagnostic total temperature
is `f_l*T_land + f_o*T_marine`.

JCM's existing SPEEDY surface scheme has one marine channel. This design uses
the effective area-weighted marine temperature above; nonlinear fluxes evaluated
at that temperature are not an explicit two-tile water/ice flux calculation.
This choice reuses native SPEEDY physics. The implementation must document it
and must not claim exact atmosphere/ice/ocean energy-budget closure or climate
validation from a short smoke run.

## 2. Native Veris exports

Export ordinary current runtime fields `ice_surface_temperature` and
`ice_surface_albedo` alongside existing `ice_fraction` and interval ocean fluxes.
The first is physical-interior `TSurf` transposed once to VerCOR `(lat, lon)`
layout. Both are current end-of-interval state diagnostics; ocean boundary
fluxes retain their independent interval-mean meaning.

Extract the native snow/ice albedo selection into a pure shared helper used by
the temperature solver and by a native current-state albedo diagnostic. Reuse
the native thickness-category factors and concentration regularization when
diagnosing snow thickness; do not invent a `hSnowMean/Area` approximation.
Preserve hemispheric constants, wet/dry temperature threshold, linear snow
transition, native category averaging and zero-ice growth behavior exactly.
The instantaneous diagnostic uses current state and is not represented as the
historical albedo used during the preceding interval's Newton solves.

Native tests must prove unchanged solver outputs against the pre-extraction
implementation and verify both hemispheres, dry/wet ice, snow transitions,
multiple categories, zero ice and physical-parameter gradients. Normalize new
VerCOR exports to the sole runtime dtype; seed them during setup and update
them with each native transition without hidden mutable state.

## 3. Open-water SST contribution in the ocean adapter

Add an opt-in `VerosConfig.export_open_water_surface` boolean, default false and
valid only with `surface_forcing="coupled"`. The enabled ocean imports
`ice_fraction` on the exact shared grid using a separate intensive, binary-source
CICE route. After advancing native Veros, form
`open_water_temperature_contribution=(1-A)*T_w` from that imported current
concentration and the newly produced SST. Export this field to ATM through
the conservative binary-source route above. Keep full SST available to CICE.

The ocean's computational mask stays its static wet mask. An ice-covered cell
still contains evolving ocean physics and can receive total fluxes. Do not
replace the ocean component mask with open-water coverage. Imported concentration
is immutable runtime state, not a setup-owner lookup into CICE.

## 4. JCM forcing extension and native hook

Add opt-in `JAXGCMConfig.seaice_coupling`, default false. In the enabled paired
setup retain the existing helper's fractional land coverage for native terrain,
while land's binary support continues to identify cells where land exists.
Apply land moisture explicitly and keep intensive land/marine temperatures
separate. Coupled input names distinguish weighted contributions from ordinary
intensive values; no ambiguous target-field fan-in is introduced.

Use a small frozen registered forcing PyTree containing native `ForcingData`
and the already composed marine surface albedo. Delegate native attributes;
`select(date, calendar)` preserves the extension while using native time
selection, and `copy(...)` returns a new wrapper. Do not subclass `ForcingData`,
whose native copy method returns the base type and discards extensions.

Install a narrow subclass of native `SpeedyForcing` through
`speedy_physics(...).replace("forcing", ...)` before constructing the model.
It first invokes the native implementation, then functionally replaces only
`_mod_radcon.alb_s` and `.albsfc` from the composed marine albedo and fractional
land coverage. Keep the real concentration in `sice_am`. Ordinary native
forcing must delegate unchanged, because native bootstrap probes physics terms
with `ForcingData.zeros`.

Keep baseline native forcing and all evolving native state in the existing
`JAXGCMRuntimePayload`. Derive applied coupled forcing only from that payload,
static resources and the current received fields; do not retain runtime ice
arrays on a model/physics/setup object.

## 5. Atmosphere-to-surface fields and flux masks

Provide raw downward shortwave (`shortwave_rad.rsds`), downward longwave,
surface pressure (`1e5 * normalized_surface_pressure`), model-level wind/air
state, and separately diagnosed rain/snow. Veris receives these intensively
over all wet ocean cells, including zero concentration, so new ice can form.
Native pressure-use behavior is unchanged; exporting pressure does not imply
a new pressure feedback into native ice thermodynamics.

Add `VerisConfig.atmosphere_forcing` with `"observed_10m"` as the existing
default and `"model_level"` for JCM. The latter selects a specific forcing
recipe without observed 10 m winds or mixed-surface absorbed shortwave. It
receives JCM's `open_water_albedo`, derives water absorbed shortwave as
`(1-alpha_w)*SW_down`, and diagnoses actual reference-height wind vectors from
the same CESM similarity/bulk state used for its forcing pack. Reuse existing
stability functions and bulk diagnostics, with stability corrections at the
model level and 10 m. Extend a shared internal bulk result/helper to retain the
stability state used by the final iteration, preserving the existing public
bulk return tuple and all existing outputs. Diagnose the reference vector from
that same solve; neither perform an independent bulk solve nor reconstruct
stability with an extra iteration. Preserve vector direction and the moving-water reference.
Do not relabel native JCM `u0/v0` as 10 m winds, or use neutral wind speed squared
as an observed vector. The observed ERA5 path keeps its existing wind source.

Raw downward shortwave feeds native Veris ice/snow absorption. Water absorbed
shortwave feeds its open-water branch exactly once. Never feed JCM's
land/ice/water-mixed absorbed `rsns` into the open-water calculation.

Correct native heat/evaporation channel mapping: channel zero is land, channel
one is marine, channel two is their native fraction-weighted total. Total
diagnostics select channel two, without summation. Export
`land_sensible_heat_flux` and `land_latent_heat_flux` from channel zero in
downward-positive W/m2 per unit land area. JCM land accepts these distinctly
named inputs in addition to the existing legacy fields. The new gallery uses
their dedicated recipe with intensive transfer and land support: no fractional
land-area multiplication precedes use as an intensive land flux.
JCM land remains the existing prescribed data component; receiving fluxes does
not turn it into a prognostic land model.

Atmosphere does not separately force Veros in this gallery. Veris's existing
interval-mean total fluxes include open water and ice, and are delivered to
Veros using the established binary-source scalar/stress routes. Avoid a second
open-water flux or concentration multiplication at the ocean boundary.

## 6. Gallery, initialization and output

The gallery exposes the standard `run_setup(loglevel, float_type)` contract and
an injectable builder for short tests. Enable dtype policy before constructing
native resources. Use paired factory/configuration patterns, daily noleap
coupling and the existing native substep alignment. The default coupling epoch
is 2000-01-03 after two days of native spinup beginning 2000-01-01. The injectable
builder validates the noleap calendar, and coupled JCM setup binds its native
epoch to coupling start minus the configured spinup duration before bootstrap
or compilation. Reject reuse of resources with a conflicting bound epoch;
keep calendar/epoch changes outside this path out of scope.

Evaluate native physics diagnostics without advancing at the actual initialized
or spun-up dynamics/time/forcing, and use those diagnostics to initialize the
coupled atmosphere exports. Keep the native dycore time and cross-step physics
carry unchanged by this diagnostic evaluation. This also covers disabled
spinup: native bootstrap diagnostics are structural zero templates and must
not be mistaken for physically evaluated SW/LW. Verify exported pressure,
air-state and radiation provenance on the first CICE step.

Declare all new fields during setup for stable compiled PyTrees. Keep runtime
output optional, use shared period/snapshot output, and expose enough surface
fields to inspect concentration, marine temperature, and albedo in the smoke
run. CLI discovery/copying and installed-wheel imports must work with lazy
optional dependencies. Reuse existing public owners and update field metadata,
recipes, API inventories, `DESIGN.md`, `DEPENDENCIES.md`, gallery documentation
and `PROGRESS.md` where their contracts change.

## 7. Acceptance and verification

Write regression tests before implementation. Required evidence includes:

1. Independent native albedo truth and pre/post extraction equivalence, including
   category geometry and zero-ice initiation.
2. Distinct land/water/ice temperatures and albedos at zero, partial and full
   concentration, both hemispheres, all-land/all-ocean and fractional coasts.
3. Nonuniform correlated SST/concentration on differing grids, proving source
   products precede remapping and post-OCN SST reaches the same ATM interval.
4. Land moisture reaches native forcing; native terrain retains fractional
   coverage; total versus land turbulent channels are selected once.
5. Raw/absorbed shortwave identities, reference-height winds under stable,
   neutral and unstable conditions, identity at model height 10 m, covariance
   when adding a uniform ocean/air velocity, finite zero-relative-wind handling,
   pressure units and forcing provenance.
6. Ice disappearance/reactivation, inactive NaNs, finite JIT/JVP/VJP and
   finite-difference agreement away from hard wet/dry and phase transitions.
7. Immutable payload/repeated-run independence, native bootstrap, legacy JCM
   compatibility, strict declared field shapes/dtypes and invalid active values.
8. Exact gallery order/routes, one native ocean initialization, no duplicate
   ocean forcing, standard CLI copy/discovery and installed-package smoke.
9. A real short four-component coupled run with evolving finite ice/ocean/air
   fields and inspected shared output, without claiming long-term climate skill.

Use the isolated Python 3.12 Conda environment named `scipy` with the exact new
native pin. The first orientation run used an older installed Veris and
demonstrated capability/geometry mismatches; it is environment evidence, not
feature regression evidence. Keep stock and exact-fork Veros autodiff checks
in separate processes and put pytest basetemps under `/private/tmp`.

Before commits, pass focused RED/GREEN evidence, Black, strict flake8, mypy,
compileall, fast and full stock suites, affected exact-fork autodiff tests,
native Veris tests, whitespace checks, and gallery/installed-artifact smoke.
Use concise stdout and detailed logs under `test_logs`. Per `AGENTS.md`, defer
all commits, including this specification, until the required full suite passes.
Push only the verified native and requested VerCOR branches using GitHub CLI
authentication and verify the remote commit IDs before marking the goal done.
