# Dinosaur 1.3.6 and JCM 2.0.1 Migration Design

## Summary

VerCOR's optional JCM integration currently pins Dinosaur 1.3.5 and JCM 1.1.1.
This migration updates the integration to the latest stable releases, Dinosaur
1.3.6 and JCM 2.0.1, without changing VerCOR's public component, setup, field,
or output contracts.

JCM 2.0.1 is the material compatibility boundary. It replaces the concrete
`SpeedyPhysics` class with composable SPEEDY physics, moves Dinosaur-specific
state conversion behind a pluggable dynamical-core interface, represents
physics diagnostics as mappings, and exposes cross-step physics carry through
`Model.run_from_state_with_carry`. VerCOR must preserve that carry across
coupling calls so subcycled radiation and other stateful physics do not reset at
each component step.

## Goals

- Pin the JCM optional dependency set to `dinosaur==1.3.6` and `jcm==2.0.1`.
- Keep `vercor.setups` signatures and lazy optional-import behavior unchanged.
- Preserve the existing VerCOR/JCM exchange field names, shapes, units, and
  sign conventions.
- Preserve JAX JIT, JVP, and reverse-mode differentiation through coupled JCM
  execution.
- Carry both the JCM native dycore state and cross-step physics state
  functionally in the immutable VerCOR runtime payload.
- Adapt JCM land forcing and native output to JCM 2 structures.
- Verify the migration with focused unit tests and real JCM integration tests.

## Non-goals

- Supporting JCM 1.x and JCM 2.x in the same VerCOR release.
- Adding ECHAM/RRTMGP configuration to VerCOR's bundled JCM setup.
- Exposing JCM's pluggable dycore or physics composition as new VerCOR public
  configuration.
- Changing VerCOR's public root or setup API.
- Refactoring unrelated slab, Veros, CAMulator, runtime, or output code.
- Adding compatibility shims for unreleased JCM or Dinosaur development heads.

## Dependency Policy

`pyproject.toml` will continue to use exact pins for the coupled JCM stack:

```toml
jcm = ["dinosaur==1.3.6", "jcm==2.0.1"]
```

Exact pins keep the tested upstream API and numerical behavior reproducible.
The repository's dependency-boundary tests and optional-model CI lane will be
updated to assert these exact versions. Core VerCOR imports will remain usable
without either optional package installed.

## Adapter Architecture

The existing private JAXGCM modules remain the ownership boundary:

- `jax_gcm_state.py` constructs JCM resources and initializes model state.
- `jax_gcm_runtime.py` owns immutable runtime payload and stepping.
- `jax_gcm_fields.py` translates native JCM diagnostics into VerCOR coupling
  fields.
- `jax_gcm_output.py` translates native state into VerCOR output variables.
- `jax_gcm_tools.py` loads coordinates, terrain, forcing, and SPEEDY parameter
  values.
- `jcm_land.py` translates JCM forcing into a VerCOR data component.

No version-dispatch module will be added. Each adapter will use the JCM 2.0.1
API directly, keeping the implementation small and making unsupported versions
fail at dependency resolution instead of inside a model run.

## Model Construction and Initial State

The setup state will construct SPEEDY physics with
`jcm.physics.speedy.speedy_terms.speedy_physics(parameters=...)` and pass it to
`jcm.model.Model`. `TerrainData` and `ForcingData` will be imported from their
JCM 2 owner modules rather than the old compatibility exports.

Initialization will call `Model.bootstrap_state()` to materialize the native
dycore state and the exact physics-carry PyTree required by JCM 2. VerCOR will
read the two bootstrapped state slots through one private helper, validate that
they were populated, and place normalized copies in its runtime state. The
gridpoint dynamics snapshot will be obtained through the model's dycore bridge.
This contains the unavoidable upstream-private attribute access in one tested
location; JCM's own checkpoint API documents the same bootstrap/state-slot
workflow.

## Immutable Runtime State

The private `JCMState` PyTree will contain four distinct responsibilities:

- `dynamics`: the latest gridpoint `PhysicsState` used for coupling output and
  native output.
- `physics`: the latest physics diagnostics mapping used for field and native
  output extraction.
- `dycore_state`: the native state used to resume integration.
- `physics_carry`: JCM's cross-step physics carry used to resume integration
  without resetting cached physics.

`JAXGCMRuntimePayload` will continue to carry the `JCMState` and the unchanged
forcing template. Both remain normalized to `RuntimeOptions.dtype`, while
integer and boolean leaves retain their original semantics.

## Step Data Flow

One VerCOR JCM component step will perform the following flow:

1. Combine the received land and sea surface temperatures and split the result
   with JCM's terrain mask.
2. Create an applied forcing value with two-dimensional land and sea surface
   temperatures while retaining the original forcing template in the payload.
3. Call `Model.run_from_state_with_carry` with the payload's native dycore state
   and physics carry.
4. Reduce saved dynamics and physics frames over the coupling interval using
   the existing PyTree mean helpers.
5. Return a new immutable `JCMState` containing the reduced output snapshots,
   final dycore state, and final physics carry.
6. Map the reduced SPEEDY diagnostics and dynamics into VerCOR's existing
   exchange fields.

The step function will remain optionally JIT compiled. No evolving JCM state
will be stored only on the mutable setup object; host-side mirrors remain for
logging and snapshot fallback only.

## Field Translation

JCM 2 SPEEDY diagnostics are mappings whose typed internal values include
`_surface_flux` and `_shortwave_rad`. A small private lookup helper will obtain
and validate these required structures before the numerical field mapper is
called. This keeps upstream shape knowledge at the adapter boundary and keeps
`map_jcm_output_fields` as a pure, JIT-compiled numerical function.

The existing VerCOR fields remain unchanged:

- lowest-level wind, temperature, and specific humidity;
- sensible and latent heat flux;
- net shortwave and downward longwave radiation flux;
- pressure, density, potential temperature, and model-level height.

Existing orientation, unit conversion, and flux sign conventions will be
preserved and tested against compact reference values. No numerical correction
or fudge factor will be introduced.

## JCM Land Forcing

JCM 2 wraps time-varying forcing fields in `TimeSeries` objects whose `values`
layout is `(time, longitude, latitude)`. The JCM land adapter will use a
JCM-specific canonicalization helper:

- static `(longitude, latitude)` arrays become `(latitude, longitude)`;
- time-first `(time, longitude, latitude)` arrays become
  `(time, latitude, longitude)`;
- malformed shapes raise a field-specific `ValueError`.

The resulting `DataComponent` retains the existing daily transfer policy. The
generic data-field canonicalizer will not gain JCM-specific behavior.

## Native Output

The native output adapter will use the model physics object's
`data_struct_to_dict` implementation to flatten JCM 2 diagnostics mappings.
Snapshot generation will read `JCMState.dynamics` and `JCMState.physics` rather
than the JCM 1 `prog` and `phydata` attributes.

Dynamics unit metadata will continue to come from
`jcm/dynamics_units_table.csv`. SPEEDY physics metadata will fall back to the
packaged `jcm/physics/speedy/units_table.csv` because JCM 2 composable physics
inherits a `None` `UNITS_TABLE_CSV_PATH`. User-provided output providers and
snapshot writers remain authoritative and unchanged.

## Errors and Optional Imports

- Missing JCM continues to raise the existing factory-oriented `ImportError`.
- Missing bootstrapped state or required diagnostics will raise a component-
  scoped error before field translation.
- Invalid JCM forcing layout will report the field name and observed shape.
- Imports of JCM and Dinosaur remain confined to lazy setup modules, so core
  VerCOR and dependency-free slab workflows remain importable.
- Exact dependency pins intentionally replace runtime support for JCM 1.x.

## Test Strategy

Implementation will follow test-driven development in these layers:

1. Dependency and source-boundary tests will assert the new exact pins, JCM 2
   import owners, composable SPEEDY factory, and absence of removed JCM 1 APIs.
2. Unit tests with compact fakes will cover bootstrapped state creation,
   cross-step carry threading, diagnostics lookup, immutable payload
   replacement, spinup, and error paths.
3. JCM land tests will cover static fields, JCM 2 time-first `TimeSeries`
   values, malformed shapes, and daily selection.
4. Field and output tests will cover JCM 2 mapping diagnostics, metadata,
   snapshot sampling, shapes, dtypes, and flux conventions.
5. Real JCM tests will cover input loading, initial payload structure, one
   component step, spinup dtype normalization, JIT, reverse-mode gradients,
   and JVP.
6. Regression verification will run Black, strict flake8, mypy, compileall,
   the configured fast and full suites, branch coverage, build and installed
   artifact checks, optional-model tests, and `git diff --check`.

Successful tests will stay concise. Detailed command evidence belongs in the
task report; durable results and any failed approaches will be recorded in
`PROGRESS.md`.

## Documentation and Dependency Order

`PROGRESS.md` will record the completed migration, exact versions, focused and
full verification results, and any known upstream warnings. `DEPENDENCIES.md`
will be created or updated as required by `AGENTS.md` to place the optional JCM
adapter modules after the core component, dtype, grid, output, and setup
configuration modules they consume.

## Acceptance Criteria

- Installing `.[jcm]` resolves Dinosaur 1.3.6 and JCM 2.0.1.
- All existing public VerCOR setup signatures remain unchanged.
- The bundled JCM atmosphere and paired land-atmosphere setup initialize and
  execute against JCM 2.0.1.
- Cross-step physics carry is present in runtime state and advances
  functionally across coupling steps.
- Coupling outputs preserve existing names, shapes, signs, and dtype policy.
- Native period and snapshot output work with JCM 2 diagnostics.
- Real JCM execution remains compatible with JIT, JVP, and reverse-mode AD.
- Core imports remain dependency-free with respect to JCM and Dinosaur.
- Formatting, linting, type checking, fast/full tests, coverage, and package
  verification pass before the implementation commit and PR are created.
