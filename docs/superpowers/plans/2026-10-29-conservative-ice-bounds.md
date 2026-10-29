# Conservative ice-area bounds repair

Goal: fix the delayed JCM bounds failure for the supplied Veris parameters,
with every component executing in JAX, and deliver on `feat/veris-seaice`.

Root-cause evidence: a full wet field remapped to T31 reaches
1.0000000000000082 (37 float64 eps above one) before any model advance.
Longitude overlaps convert degree differences to radians; destination areas
instead subtract radian coordinates. Static masks clip the overshoot while
physical ice fields retain it. Full ice can therefore trip the two-eps guard.

- [x] Add failing raw-remapper and remap-to-JCM endpoint tests for float32/64,
  periodic longitude seams and shifted longitude origins.
- [x] Use the same longitude interval arithmetic for overlaps and destination
  areas in the existing remapper. Retain conservation and JCM bounds checks.
- [x] Verify focused conservation, bounds, inactive support and JVP/VJP tests.
- [x] Prepare one instrumented global gallery validation with the exact seven
  supplied parameters, all-JAX components, ordinary monthly output, and local
  failure evidence captured before a callback can abort execution. Run once.
- [x] Run formatting, lint, typing, fast/full stock and isolated exact-fork AD
  tests; obtain independent review and record results in PROGRESS.md.
- [ ] Commit only intended paths, push using GitHub CLI authentication, and
  verify the remote branch SHA.

Constraints: current branch and existing component/exchange ownership remain;
no broad clipping, tolerance increases, native physics changes or repeated
long exploratory runs. No module/dependency layer changes are required.

Validation preparation uses temporary Python3.12 dependency targets because
the required scipy environment aborts in native-library loading. Installed
environments remain untouched. The provided request already authorizes
implementation, Git commit and push; execute this repair inline.

Execution evidence: six meaningful RED failures before the arithmetic change;
89 focused remapping/surface cases GREEN afterwards. Black308, maintained
flake8 zero, mypy308, compileall and whitespace pass. Repository lint retains
the same 200 historical findings under ignored scientific snippets. Independent
review confirms masked conservation and finite adjoint/JVP/VJP behavior in both
precisions. The first global harness preparation stopped after initialization
on an incorrect public payload lookup; zero coupled steps ran. Corrected to
the existing private state accessor before the single expensive trajectory.
The capture uses an identity-returning I/O callback so operands are saved
before the production bounds check, rather than depending on callback order.

Final global result: 365/365 coupling steps, 8,760 native ice substeps,
zero violations, 43 output files, 1,120.25 seconds. The old-area replay first
flags one cell on day 253, then one on day 270; this diagnoses the delayed
trigger without a second trajectory. Fast stock 1,497 and exact-fork AD 15
pass; full stock 2,446 plus its corrected RST-only failure 1/1 pass. Strict
Sphinx passes after fixing the title underline. Regular scipy runs all six
new regressions successfully. Provenance/socket/progress preparation failures
are resolved, and no source numerical tolerance was changed.
