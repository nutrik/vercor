# Native-substep means for exchanged state

Status: approved by the user on 2026-11-02. The user selected adapter-level
native-substep means and approved validation in the existing `scipy312`
environment after the required `scipy` baseline aborted its pytest workers.
Implementation and delivery stay on the existing `feat/veris-seaice` branch.

## Intent and numerical contract

For a component taking N equal native steps inside one coupling interval,
export the arithmetic mean of its N post-native-step canonical state samples:

    exchanged_field = sum(mapped_native_field[k] for k in range(N)) / N

Equivalently this is a duration-weighted mean because native timesteps are
fixed and divide the coupling interval. Do not include the interval's initial
state. N=1 exports the native endpoint. This is a discrete sampling convention,
not a claim of exact continuous-time quadrature.

Diagnose nonlinear derived fields at each native sample before averaging:
density, potential temperature, model-level height, ice albedo, and open-water
temperature contribution. Preserve existing flux integration and nonlinear
precipitation/radiation ordering. Native prognostic state, continuation carry,
clock, integer/categorical fields, diagnostic histories, and native snapshots
retain their original endpoint/sample meanings. Initialization diagnoses the
initial instant; it has no completed interval to average.

## Approaches

1. **Selected by the user: adapter-level means.** Each native adapter already
   owns its native timestep alignment, loop/history, and field conversion. Add
   state reduction at those boundaries, reusing canonical mappers. This keeps
   native continuation independent of exchanged samples and adds no public API.
2. **Central dispatcher averaging.** The dispatcher sees only one returned
   sample per coupling interval. It cannot reconstruct missing native history;
   changing it would require moving native stepping into the generic runtime.
3. **A public temporal policy registry.** This would make a requested default
   behavior optional and introduce configuration/protocol machinery without
   supplying the missing samples. It is unnecessary for this change.

## Existing owners and changes

- **JCM:** extend the compact native exchange sample projection to include the
  surface-nearest dynamics and pressure information needed by its existing
  canonical mapper. Map every sample, then average all evolving atmospheric
  exports, including the pressure profile. Keep endpoint `JCMState` and exact
  native carry. Do not return complete vertical/physics histories from the
  compiled wrapper. Sampling remains independent of `save_interval`.
- **CAMulator:** reuse the already mapped native prediction block and reduce
  every canonical prediction field rather than only radiation. Preserve
  native model state, cursor, output samples, and applied surface forcing.
- **Veros:** extract canonical surfaces after each native step and accumulate
  only these small arrays. Return their means and the final native payload in
  host and differentiable JAX paths. Diagnose the open-water temperature
  contribution at each sample with the coupling interval's received ice field.
- **Veris:** extend the existing pure native loop's carry with accumulated
  canonical ice fraction, temperature, and diagnosed albedo. Return their means
  alongside the existing mean fluxes. Retain sticky native validity and final
  payload diagnostics. Evolving component/route masks follow the exported mean
  ice fraction; existing spatial coverage weighting remains dispatcher-owned.

Prescribed monthly/daily data and prescribed land components retain their
recipient-clock/record selection: they have no advancing native prognostic
trajectory to reduce. Slab components take one coupling-sized step and retain
their endpoint. Third-party component authors continue to own internal
subcycling; this change introduces no generic runtime subcycling or registry.

## Scope and implementation judgment

Reuse existing field mappers and loops; add private helpers only when they
remove actual duplication. Keep lazy optional imports, SI units, dtype policy,
immutable payloads, masks, public manifests, and dependency layers. Keep current
sequential component ordering. Do not change native numerics, physical
tolerances, installed dependencies, forcing interpolation, output cadence, or
spatial remapping to implement temporal reduction. Averaging separate exported
fields does not introduce a new covariance-preserving temporal remapping law.
The builder owns small helper/signature choices within these constraints.

## Validation and delivery

Write failing analytic regressions before implementation. Cover one, two, and
several native steps; changing sample sequences; nonlinear derived fields;
float32/float64; independent sample-before-mean oracles; mask/inactive values;
unchanged native endpoints, clocks and category dtypes; interval reset and
continuation; native output history; JIT/JVP/reverse gradients and compilation
reuse. Include route-level evidence that a recipient receives the mean.

Run focused fast tests during development. Before committing, run Black,
maintained strict flake8 plus repository lint accounting, mypy, compileall,
fast and full stock pytest, separate exact-fork Veros gradient tests,
appropriate coverage and documentation checks, and short native coupled
host/JAX integrations. Use the approved `scipy312` environment and isolated
temporary stock/fork launchers if native process-wide backend rules require
them. Record failures and evidence in the plan and bounded `PROGRESS.md`;
update `DESIGN.md`, `DEPENDENCIES.md`, and temporal-contract documentation.

Review implementation and tests independently, make focused passing commits,
push with GitHub CLI authentication, and verify the remote branch SHA against
local HEAD. No release, tag, merge, or messages to others are requested.
