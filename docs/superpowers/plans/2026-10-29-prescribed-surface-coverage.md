# Prescribed surface coverage implementation plan

> **For agentic workers:** Use superpowers:executing-plans to implement this plan task by task. Steps use checkbox syntax for tracking.

**Goal:** Preserve native atmospheric surface forcing outside a coupled ocean's domain and use fractional area blending consistently in physics and output.

**Architecture:** Keep the existing component, exchange, immutable-payload and native-forcing owners. For geometric ocean footprint G, remapped wet fraction C and native land fraction N, use prescribed marine area U=(1-G)*(1-N) and effective land L=1-C-U. Coupled surface products remain source weighted; add U times the native marine value and normalize by C+U. Blend native time-series values without changing their dates or selection policy. Native diagnosed fields supply the actual physics and output samples.

**Tech stack:** Python, JAX, JCM, Veros, Veris, pytest, Black, flake8, mypy, GitHub CLI.

**Spec:** The user's polar-output bug report in this session and DESIGN.md component, quantity and output contracts.

## Constraints and decisions

- Work in the existing clean `feat/veris-seaice` checkout as requested; no branch/worktree switch.
- Preserve the established remapped coastline inside the ocean grid and native terrain outside it.
- Keep dispatcher zero-fill, quantity weighting, dynamic component masks and last-dispatch coverage unchanged.
- Pure native arrays and time series must both work; preserve exact native time selection at every model substep.
- No physics clipping, tolerance relaxation, duplicated native albedo physics or new runner.
- The requested scipy baseline crashes workers before collection. Use the existing Python 3.12 lane for meaningful validation, without altering installed environments.
- Execute the authorized repair continuously; geometry specialist and independent final reviewer follow AGENTS.md roles.

## Review focus

- Fractional domain-edge cells and native coastal/pure-land polar cells.
- Undefined absent-surface operands, genuine zero forcing and float32/64 precision.
- Native forcing time-series changes within a coupling interval, including ice-derived albedo.
- Ordinary ocean/atmosphere coupling as well as the sea-ice coupling gallery.
- Native diagnostics, runtime fields, period output and final snapshots must agree.

### Task 1: Separate geometric footprint from physical land

Files: `vercor/grid_masks.py`, `vercor/setups/_data/jcm_land.py`, `vercor/_runtime/surface_masks.py`, `tests/test_partial_surface_masks.py`.

- [x] Add RED regressions for uncovered native poles, partial cells, native pure land, default compatibility and topology consistency.
- [x] Reuse one remapper for G/C. Add optional native land fraction to existing mask/land factories and preserve default behavior.
- [x] Validate topology inside the coupled domain and retain actual native land support outside it.
- [x] Run focused GREEN mask/topology tests.

### Task 2: Blend prescribed and coupled marine forcing

Files: `vercor/setups/_jcm.py`, `vercor/setups/_external/jax_gcm.py`, `jax_gcm_state.py`, `jax_gcm_fields.py`, `jax_gcm_surface.py`, `jax_gcm_runtime.py` in that directory, and focused JCM tests.

- [x] Add RED tests for polar preservation, hand-computed partial cells, exact native time selection and actual integration/output agreement.
- [x] Carry static prescribed marine area from paired construction into the atmosphere setup owner using an optional factory keyword; preserve existing standalone defaults.
- [x] Compose coupled contributions over total marine area while checking ice against coupled ocean area. Blend original native SST/ice arrays or time-series values using prescribed marine fraction.
- [x] Extend the existing native forcing wrapper to blend native albedo without duplicating its diagnosis; keep the original forcing in the payload.
- [x] Export the surface fields actually used by native physics, sharing them with output.
- [x] Run focused GREEN tests, inactive-NaN and JIT/JVP/VJP checks, and a short real global integration in separate native execution lanes.

### Task 3: Verify and deliver

Files: existing setup helper tests, signature inventory if changed, DESIGN.md, PROGRESS.md and existing coupled validation documentation.

- [x] Update intentional contracts and documentation; retain bounded PROGRESS.md.
- [x] Run Black, maintained-source lint, repository lint report, mypy, fast and full stock tests, and separate exact-fork AD tests. Keep logs/artifacts under /private/tmp.
- [x] Obtain independent review and resolve material findings with RED/GREEN evidence.
- [x] Prepare authorized Git delivery after full verification: intended paths and remote branch checked. Commit/push follow this local record; the final response records the verified remote SHA.

## Execution evidence

- Clean starting tree at `6648b1a`; current branch matches the user's request.
- Baseline scipy: all four xdist workers crashed before collection; no tests ran.
- Read-only geometry probe confirmed uncovered latitude bands become land under the old helper.

- Geometry RED 10 failures, GREEN 14 new and 60 focused cases. Native composition RED 9 expected failures; GREEN 55 native cases. Snapshot omission independently failed before restoration.
- Original transient native forcing diagnostic is deliberately removed by JCM. The adapter now samples exact saved-physics dates with a stable JIT kernel instead of depending on that transient key. Native TimeSeries uses dataclass replacement, not a copy method.
- Independent review reproduced float32 land 0.8/prescribed 0.2 rejection; shared two-epsilon full-prescribed reconciliation and targeted RED/GREEN regression resolve it. Native time-series and saved-snapshot-date coverage was expanded.
- Real host and all-JAX global runs each completed two days, 96 native atmospheric evaluations, 252 prescribed polar cells and 9 files. Applied polar SST/ice agree exactly with native selected forcing at every native evaluation. Preparation asset locking and a harness assertion that confused wet coverage with geometric footprint were corrected before either trajectory.
- Temporary Python3.12 overlays pin reviewed JAX/JAXlib 0.11.2, Flax 0.12.10, released Veris and isolated stock/exact-fork Veros; original environments remain unchanged.

- The original end-to-end albedo finite-difference stencil no longer resolved the local slope after native background restoration: h=1e-2/1e-3/1e-4 gave 1.01246/0.999763/15.01246 while JVP/VJP agreed at -0.552228598. Deterministic repeats and h=1e-5/1e-6/1e-7 gave -0.552228319/-0.552226235/-0.552195919. Keep native geography/physics and the unchanged 0.003 relative/1e-7 absolute tolerance; record both stencil ranges. This certifies the local alpha=0.70 derivative; the exact cause of larger-perturbation regime changes is unproven.
- Full/fast checks initially found the bounded PROGRESS log grew to181 lines. Removed one duplicate historical release-authority bullet to restore180; its restriction remains in Durable Constraints, with current commit/push explicitly authorized by the user.

- Final stock 2472/2472 and fast 1522/1522 pass with zero errors/skips. Exact-fork AD:14 pass, then corrected local-stencil acceptance1/1 passes with original tolerances. Combined statement/branch coverage92.89226642825213% exceeds90%; all three raw data SHA256 hashes are preserved.
- Black310, strict maintained flake8 zero, mypy310, compileall, diff, strict Sphinx, wheel/sdist build and Twine pass. All180 packaged Python files match source byte-for-byte. Repository lint retains200 historical snippet findings. Full suite includes installed-artifact and extension probes.
- Verification receipts use /private/tmp/vercor-mask-*. The first coverage report invocation used a CLI-only argument on the Python API; rerun with the correct API and independently enforce the CLI90% gate. Final full/fast runs correct the earlier PROGRESS181-line failure.
