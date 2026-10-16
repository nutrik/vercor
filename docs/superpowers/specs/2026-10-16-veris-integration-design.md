# Dynamic component masks and Veris integration

Status: approved by the user on 2026-10-16, including CESM `tref` for `ATemp`;
implementation plan approved; implemented and locally verified on
`feat/veris-seaice`; source pushed with GitHub SHA verification.

## Intent and acceptance criteria

Implement two sequential stages on `feat/veris-seaice`: first, time-varying
binary/fractional component masks with correct exchange semantics; second, a
prognostic Veris callable component and the packaged
`run_veris_dataveros_era5.py` example. Preserve the existing uncommitted
`PROGRESS.md` changes in this branch. Complete formatting, lint, typing, tests,
small commits, and the user-authorized GitHub push after implementation passes.

The example prescribes ocean and atmosphere data. Its execution order is
`LND -> CICE -> OCN -> ATM`. Sea ice uses the authentic Veros global-four-degree
ocean grid and static wet mask. `Area` supplies evolving fractional coverage.
There are no CICE feedback routes to the prescribed data components. Generic
mask/exchange infrastructure and callable-component tests prepare for future
prognostic feedback without adding those routes now.

## Evidence and alternatives

VerCOR already has immutable component payloads, lazy external factories,
monthly data selection, route masks, CESM bulk fluxes, output providers, and
fixed-shape JAX scans. Reuse these owners rather than introduce a second
component lifecycle, output session, or model runner.

1. Recommended: declare dynamic masks by references to ordinary runtime fields,
   reuse existing lifecycle/results, and wrap standalone Veris's pure `advance`.
   This keeps evolving data in immutable runtime state and geometry static.
2. Mutate component grids and reconstruct regridders each step. Reject: grid
   identity is prepared configuration, and sparse remapping geometry must remain
   fixed under JIT/AD.
3. Embed the older Veros sea-ice plugin or a Veris `Simulation.run` inside a
   component. Reject: this duplicates scheduling, state ownership, and output,
   and does not match the current standalone Veris API.

Live GitHub API inspection found both upstream `main` and `jax-only` at
`a9d6c0ade52fbae4c0d1da694878943d6dbf7e6e`. Target that immutable standalone
revision, not the older plugin described by cached web/ReadTheDocs pages.
Use a lazy optional dependency with a capability-oriented installation error;
keep the existing Veros and differentiable-Veros dependency contracts intact.

## Stage 1: runtime masks

### Declaration and ownership

Add a frozen `ComponentMaskSpec`, owned/exported by `vercor.components`, and an
optional `ComponentSpec.masks` declaration. Its `binary_field` and
`fractional_field` reference declared, current-time, grid-shaped component
fields. A missing binary field means the static wet support; a missing fraction
means unit coverage on active support. Existing components retain their present
behavior when the declaration is omitted.

Setup seeds mask fields through `SetupResult`; stepping changes them through
ordinary mapping results or `StepResult`. Do not add a parallel mask callback,
mutable adapter attribute, or independently evolving copy. The fractional
mask remains a traced array. A binary mask is discrete membership, so no
derivative through its threshold is promised.

`RectilinearGrid.binary_mask` remains immutable computational support. Dynamic
binary masks and fractional coverage must be zero outside that support. Veris
uses the ocean wet support as its binary domain and exposes native `Area` as
`ice_fraction`. An ice-free wet cell remains a computational cell where ice can
form; do not replace the wet mask with `Area > 0`.

### Exchange quantities

Keep the current scalar/vector route behavior as the default compatibility
mode for routes without dynamic declarations. Routes touching a declared
dynamic component mask must select an explicit quantity mode: intensive values,
source-area flux contributions, or target-area flux contributions. Each route
has one mode; split fields into distinct named routes when their meanings differ.

Let `R` be the fixed remap, `a` the effective source fraction, `c` the target
fraction, and `b` the target binary/domain gate. Fractions include their
component's binary support. The explicit modes are:

- Intensive: covered-source mean `b * R(a * f) / R(a)`, gated by target binary
  computational support, independently of target concentration. In particular,
  CICE receives forcing on ice-free wet cells so ice can form. Return zero on
  empty overlap with a safe denominator. Do not scale
  temperature, humidity, pressure, velocity, or per-area forcing by coverage.
- Source-area flux: `b * R(a * f)`, a contribution per destination grid-cell
  area. Use conservative remapping when area-integral conservation is required.
- Target-area flux: `c * R(a * f) / R(a)`, a covered-source mean integrated over
  the target coverage. This is distinct from a source-area contribution.

Apply these explicit modes consistently to scalars and vector pairs, retaining
the regridder's vector transformation. Binary gating applies in these modes;
the default static compatibility mode retains its existing vector behavior.
Never replace
`R(a * f)` with `R(a) * R(f)`. Neutralize inactive NaNs before multiplication,
remapping, or division. Infinity remains invalid everywhere.

Prepared topology binary restrictions intersect the runtime binary support.
Existing fractional topology patches retain their current meaning in the
compatibility mode. Explicit quantity modes own their coverage normalization;
reject a nontrivial conflicting fractional patch rather than silently applying
coverage twice. The gallery uses explicit intensive incoming CICE routes and
does not need a surface fractional patch on those routes.

### Refresh, validation, and output

Resolve masks after setup/prefill and before initial outgoing stores are
primed. After a component step, validate its mask fields, refresh derived route
masks, and then send. A later recipient sees the new mask; an earlier recipient
uses the prior state. An omitted component retains its current mask.

Rebuild derived masks from authoritative component fields for supplied states,
including states updated through `RunState.replace_fields`. Synchronize
replaced current exports with their outgoing copies; record
arrays retain time selection at the next source send. Preserve received values
and their last-dispatch support independently. Keep route order, leaf
shapes/dtypes, and carry schemas fixed. Check binary values in `{0, 1}`,
fractions in `[0, 1]`, finiteness, shape, dtype, and static-domain containment
with transform-safe checks. Newly active exported fields must be finite.

Preserve one runtime source for masks in receive/step/send validation and
diagnostics. Final output must sample final runtime masks; currently the facade
passes initial prepared topology masks, which would be stale after evolution.

Primary owners: `components/contracts.py`, component binding/validation,
`_runtime/driver.py`, preparation, dispatch, field transfer, state validation,
and output handoff. Put shared mask resolution in a small private runtime
module. Update public manifests/signatures and dependency documentation for
the intentional additive API.

## Stage 2: Veris wrapper and ocean data

### Lifecycle and geometry

Use canonical `vercor/setups/_external/`, consistent with the Veros and JCM
wrappers and the active IDE files. This resolves the request's
`setups/external/` spelling to the repository's existing owner. Expose the
factory/configuration lazily from `vercor.setups`.

Initialize native Veris with `Experiment`, `Grid`, `OceanGeometry`, and
`initialize_experiment`. Hold native `State` and fixed-structure diagnostics in
a frozen PyTree payload. Bind immutable native configuration/constants as
setup resources. Call `veris.advance(..., physics="coupled")` for each native
substep, returning fields plus replacement payload through `StepResult`.
Use existing timestep-alignment checks; reject a coupling timestep that is not
a positive integer multiple of the native timestep. Host I/O stays outside
JIT/AD. Do not call native runners or writers from stepping.

Veris supplies island, dynamics, and growth presets, but no global-four-degree
geometry preset. Initialize the existing Veros global-four-degree setup once
without spinup/integration. Snapshot its grid, surface wet masks, monthly
SST/SSS climatologies, and authentic metric arrays; discard the mutable native
model owner after extraction. Share the snapshot between construction of the
data ocean and Veris; do not initialize a second ocean for the ice wrapper.
Keep this preparation in a focused helper and provide a paired setup factory,
following the existing paired JCM setup pattern, so the gallery needs no
private imports or duplicated geometry assembly.

Supply physical-interior `maskT/U/V`, `ht`, `coriolis_t`, `dxt/dxu/dyt/dyu`, and
`area_t/u/v` through `OceanGeometry`. Strip the native two-cell halos once;
Veris builds its own. Veros setup spacings are already in meters and areas
already include spherical factors. Pass negative native depth as the Veris
bottom-elevation convention. Convert Veros east/north face masks, areas and
spacing samples to native west/south indices; tracer coordinates stay fixed.
Native face output coordinates use the lower edge of each tracer cell.

The data ocean uses existing linear monthly interpolation of `sst_clim` and
`sss_clim`, converting SST from Celsius to Kelvin. Veros's initial currents
are allocation zeros, not observations: document zero prescribed currents,
SSH, and runoff as example idealizations. Use distinct canonical ocean-current
names rather than atmospheric `u_velocity`/`v_velocity`.

### Forcing and physical conventions

Pass physical-interior `(nx, ny)` arrays into Veris, transposing once from
VerCOR's `(lat, lon)` layout. All incoming CICE fields are intensive values or
per-area forcing; native Veris applies its own `Area` weighting.

| Veris input | Source/conversion | Units and sign |
| --- | --- | --- |
| `theta` | ocean SST | K |
| `ocSalt` | ocean SSS | g/kg |
| `uOcean`, `vOcean` | prescribed Veros east/north currents, coastal faces closed, shifted to native west/south | m/s |
| `ssh_an` | prescribed SSH, zero in example | m |
| `surfPress` | ERA5 surface pressure | Pa |
| `uWind`, `vWind` | ERA5 `u10`, `v10` | m/s, 10 m |
| `wSpeed` | derive through native `advance` | m/s |
| `ATemp` | CESM diagnosed 2 m air temperature `tref` | K |
| `aqh` | CESM diagnosed 2 m specific humidity `qref` | kg/kg |
| `SWdown` | ERA5 `msdwswrf` | W/m2, downward |
| `LWdown` | ERA5 `msdwlwrf` | W/m2, downward |
| `Qsw` | negative ERA5 net shortwave | W/m2, upward |
| `Qnet` | negative CESM bulk plus radiative ocean heat flux | W/m2, upward |
| `precip` | rain mass flux / freshwater density | m/s, into surface |
| `snowfall` | snow water-equivalent mass flux / freshwater density | m/s |
| `evap` | signed CESM bulk evaporation / freshwater density | m/s, negative loss |
| `runoff` | prescribed runoff, zero in example | m/s, into ocean |

The actual cached ERA5 surface asset contains `msdwswrf`, `msnswrf`,
`msdwlwrf`, `u10`, `v10`, and `sp`. Add downward shortwave to the ERA5
adapter along with 10 m wind, retaining existing model-level and net-shortwave
fields. Use the existing ocean CESM `tref` and `qref` diagnostics for `ATemp`
and `aqh`, respectively, documenting their shared open-ocean surface-similarity
assumption. Both are diagnosed at the configured air-temperature reference
height, which defaults to 2 m. The bulk calculation
must retain the original colocated model-level wind/temperature/humidity and
model-level height, rather than mixing those with `u10`. Native ice
surface thermodynamics consumes the supplied near-surface values directly;
changing a height setting does not transform model-level temperature or
humidity. Add independent height/conversion tests. Do not relabel diagnosed
`tref`/`qref` as measured ERA5 temperature/humidity or neutral relative speed
squared as observed wind speed.

Extract only necessary shared array-level bulk/radiative calculations from the
Veros-specific owner, preserving existing Veros numerical behavior. Do not
require a mutable Veros state to compute Veris forcing. Reapply atmospheric
forcing, especially `Qnet`/`Qsw`, before every native substep because native
thermodynamics replaces those values with coupling residuals.

Keep rain and snow separate: upstream combines their contributions, so passing
total precipitation as rain would count snow twice. Native salt-flux diagnostics
currently declare unknown units; do not expose a scientifically unverified
ocean-feedback conversion or route in this task.

### Output and gallery

Implement a native-payload `OutputProvider` with interior coordinates and
registry-derived units for ice concentration, volume-equivalent thickness,
snow, ice velocity, temperature, and supported diagnostics. Route period output
and final snapshots through VerCOR's existing output coordinator. Output-free
runs perform no I/O and remain compatible with supported JAX transforms.

Create `vercor/setups/gallery/run_veris_dataveros_era5.py` with the standard
`run_setup(*, loglevel, float_type)` interface, an explicit finite daily clock,
the required run order, and visible native/coupling timesteps and output policy.
Use native configurable initial conditions rather than introduce a separate ice
initialization model. Use ERA5 atmosphere/land and the shared Veros data/geometry
construction. No CICE-to-OCN, CICE-to-ATM, or CICE-to-LND exchange is configured.

## Verification and handoff

Write failing tests before each implementation unit. Stage 1 covers static
compatibility; binary activation/deactivation; fractions 0/interior/1; refresh
timing; initial and supplied states; empty coverage; NaNs/infinity; source
weighting before remapping; scalar/vector parity; current-mask output; host/JAX
agreement; and JIT/JVP/VJP/finite differences through interior fractions.

Stage 2 covers lazy optional imports, immutable payload ownership, authentic
geometry/staggering/units, forcing heights/signs/conversions, substep alignment
and forcing reset, output metadata/selection, exact run order, and absence of
feedback to data components. Compare a real two-step native Veris rollout to
direct `advance` with identical converted forcing. Run a real short global
four-degree example, checking finite wet fields, zero dry-cell ice, bounded
`Area`, state evolution, and produced files. This validates operation; it does
not establish long-term climate or spherical-dynamics accuracy.

Run Black, strict flake8, mypy, compileall, whitespace checks, the stock fast/full
suites, and isolated native Veros AD tests. Upstream supports differentiation
through state and forcing; its configuration and native constants are static.
Do not claim gradients through those constants or across binary mask changes.

The default Python 3.14 `scipy` environment aborts importing Torch with duplicate
OpenMP initialization. The existing Python 3.12 environment at
`/private/tmp/vercor-time-ci/envs/scipy` is named `scipy` and supports the baseline.
Stock and JAX-native Veros tests require separate Python processes because
Veros locks its backend on first use. The installed-GitHub-CLI upload test needs
localhost socket access. Record all evidence and unsuccessful approaches in
`PROGRESS.md`, preserving its existing edits, before commits and final push.

## Upstream references

- [Standalone public API](https://github.com/team-ocean/veris/blob/a9d6c0ade52fbae4c0d1da694878943d6dbf7e6e/veris/__init__.py)
- [Native advancement](https://github.com/team-ocean/veris/blob/a9d6c0ade52fbae4c0d1da694878943d6dbf7e6e/veris/execution/advance.py)
- [Ocean geometry](https://github.com/team-ocean/veris/blob/a9d6c0ade52fbae4c0d1da694878943d6dbf7e6e/docs/quickstart/custom-ocean.rst)
- [Forcing contract](https://github.com/team-ocean/veris/blob/a9d6c0ade52fbae4c0d1da694878943d6dbf7e6e/veris/model/forcing.py)
- [Native diagnostics](https://github.com/team-ocean/veris/blob/a9d6c0ade52fbae4c0d1da694878943d6dbf7e6e/veris/model/diagnostics.py)
- [Native AD contract](https://github.com/team-ocean/veris/blob/a9d6c0ade52fbae4c0d1da694878943d6dbf7e6e/docs/reference/automatic-differentiation.rst)
