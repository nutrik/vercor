# Global-only Veros setup

The user explicitly requested removal of ACCSetup, standardization on
global_4deg, test migration, source cleanup, full quality gates, and a commit
and verified push on the existing feat/veris-seaice branch. This replaces
the preceding read-only scope. The required result is fully specified.

## Approach

Remove ACC support completely rather than retaining a private ACC fixture or
a one-value setup selector. VerosConfig no longer accepts setup; callers
remove setup="global_4deg". Native global geometry, parameter defaults,
restoring controls, backend choices, host/scanned execution, and both existing
forcing modes remain owned by their current implementations.

The native class selector depends only on execution. Remove the ACC adapter,
setup checks, and the internal-forcing bypass. Derive forcing behavior directly
from surface_forcing instead of storing uses_bulk_meteorology.

## Test and benchmark migration

Backend tests initialize the real 90 x 40 x 15 global ocean and advance it.
Migrate the three-day initial-temperature and twenty-day c_k derivative tests
to active global cells; retain finite differences, JVP/VJP comparisons,
payload stability, and actual multi-step advancement. Choose wet cells from
the initialized global mask, not ACC coordinates. Existing coupled global
adjoint/flux, native float32, and continuation checks remain required.

Replace the obsolete internal-forcing quiet-step regression with forcing-mode
coverage. Native probe fixtures explicitly exercise surface-flux mode where
they supply preinstalled native source terms.

The benchmark uses --case ocean or --case coupled. Both report
setup=global_4deg separately from case. The ocean case uses explicit zero
coupled fluxes and synchronous 1800-second native/coupling clocks so it needs
no artificial atmosphere. Comparison rejects different cases. Historical
ACC measurements remain labeled as historical evidence; current commands
and documentation describe only the global setup.

## Acceptance

- No ACC class/import/selection or internal-forcing bypass in runnable code.
- No redundant public setup option; maintained signature fixture matches.
- Tests preserve backend, forcing, gradients, quiet stepping, and benchmark
  attribution with meaningful global behavior.
- Current docs, design, dependencies, changelog, and progress agree.
- Formatting, lint, typing, fast/full stock tests, exact-fork AD/native tests,
  coverage, compileall, documentation, and independent review pass.
- One bounded removal/migration commit on feat/veris-seaice; push origin and
  verify its SHA matches the clean local tip. Preserve unrelated work.

## Constraints

No native model source patches, physics changes, relaxed tolerances, numerical
fudge factors, or reduced validation scope. The known scipy/Python3.14 native
worker abort is handled with the existing isolated Python3.12 validation
environment and exact native dependency overlays. No installed packages change.
