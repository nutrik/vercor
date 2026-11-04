# Physical coupling contract names

The user approved the preceding branch naming audit and requested all table
findings, smaller clarity opportunities, documentation, complete quality gates,
and a commit/push on `feat/veris-seaice`. Preserve the dirty `PROGRESS.md`.

## Scope and decisions

Use physical names for reusable coupling interfaces; keep concrete model names
on adapters, native owners and dependencies. This is one atomic naming refactor,
with no compatibility aliases, registry, new execution path or numerical change.

- Lowest-model-level exchange keys become `air_temperature`, `air_density`,
  `air_u_velocity`, `air_v_velocity`, `air_potential_temperature`, and
  `air_specific_humidity`. Keep `model_level_height`, actual `temperature_2m`
  observations/toy diagnostics, and actual `u_velocity_10m`/`v_velocity_10m`
  diagnostics distinct. Do not rename native JCM/CAMulator variables.
- JCM's full vertical pressure field becomes `air_pressure_3d`; CAMulator's
  lowest-level field becomes `model_level_air_pressure`. Preserve shapes,
  ordering and Pa units. No additional exported field is needed.
- JCM's phase-only mapper argument becomes `near_surface_air_temperature`.
  It remains native SPEEDY t0, without an asserted 2 m height, and never replaces
  the lowest-level air fields supplied to bulk fluxes.
- Shared recipes describe ocean bulk meteorology, observed-10m/model-level
  atmosphere-to-sea-ice forcing, ocean-to-sea-ice forcing, total sea-ice-to-ocean
  fluxes, ice surface properties/coverage, open-water contributions, land state,
  and turbulent/latent heat fluxes rather than concrete implementations.
- Veros `surface_forcing` becomes `bulk_meteorology` or `surface_fluxes`.
  Rename `uses_atmosphere_forcing` to `uses_bulk_meteorology` and preserve the
  global/ACC restrictions and native forcing behavior.
- Spell out ocean/land/atmosphere and binary/fractional mask helper names and
  local variables. Keep the existing complementary land/ocean topology.
- Galleries derive routes, run order, state lookup and policy roles from the
  actual components' names. Explicit instance-name defaults remain configurable.
- Rename `prepare_camulator_sst_input` to
  `prepare_camulator_surface_temperature_input` because it packs the combined
  surface. Retain native channel names at their translation boundaries.
- The specifically JCM-backed public factory/configuration become
  `make_jcm_atmosphere` and `JCMAtmosphereConfig`. Corresponding private setup
  state/payload owners explicitly identify JCM; existing module paths can stay.

## Invariants

Keep all units, flux signs, source/target coverage choices, last-dispatch mask
ownership, inactive-NaN neutralization, vertical locations, native staggering,
precision, lazy optional imports, payload ownership, and output-free JIT/JVP/VJP.
In particular ocean currents remain canonical east/north faces translated to
Veris west/south faces; generic naming does not accept arbitrary staggering.

## Evidence and delivery

Migrate existing numerical and gradient assertions rather than weakening them.
Protect mapper pressure ranks and values and the near-surface/model-level
temperature separation. Check custom component names in bundled galleries.
Update current documentation, API manifests/signature snapshots, and dependency
inventory. Preserve historical audit records. Run focused tests, fast/full stock
and isolated exact-fork AD lanes, formatting, lint, typing, compile/whitespace,
strict docs, and packaging/artifact checks before the single refactor commit.
Push the current branch with GitHub CLI authentication and verify remote SHA.
