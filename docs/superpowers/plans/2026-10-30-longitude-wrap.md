# Conservative longitude-wrap repair

> **For agentic workers:** Use superpowers:executing-plans for the test-first repair and superpowers:requesting-code-review before delivery.

**Goal:** Restore the native wet 0–4 degree ocean strip in atmosphere feedback on `feat/veris-seaice`.

**Architecture:** Standardize eastward longitude edges once in the existing remapper constructor, before both overlap and destination-area calculations. Preserve field/mask column order, explicit broad monotonic cells, and the identical-grid bypass. No new public API, dependencies, or physics correction.

**Tech Stack:** Python, JAX, NumPy, pytest, Black, flake8, mypy, GitHub CLI.

**Spec:** The user's active longitude-wrap defect report; `DESIGN.md` and `AGENTS.md`.

## Review focus

- Wrapped source and destination seams, shifted origins, and legitimate monotonic cells wider than 180 degrees.
- Wet masks, first-strip integrals, nonconstant fields, and column order.
- Float32/float64, JIT, JVP, and VJP through the repaired seam.
- Native/coupled surface fractions: remove spurious prescribed marine coverage inside the footprint and retain polar forcing.
- Identical Veris/Veros grids retain their conservative identity bypass.

## Task 1: Verify and repair longitude geometry

- [x] Add regressions against independently known monotonic geometry, analytic strip overlaps/integrals, masks, and derivatives.
- [x] Run new tests unchanged against production code; record expected failures.
- [x] Standardize longitude bounds in `vercor/_interpolators/conservative_remap_rectilinear.py` before geometry assembly, using one private helper for both axes' longitude bounds.
- [x] Run focused remapping and surface-fraction tests in both precision policies.
- [x] Independently review the change and fix any substantive findings with RED/GREEN evidence.
- [x] Run formatting, maintained-source/repository lint, type checking, fast/full suites, and native geometry/integration checks. Keep receipts under `/private/tmp/vercor-longitude-*`.
- [x] Update `PROGRESS.md`, dependency notes, and scientific validation documentation with verified results and limits.
- **Authorized delivery:** Commit only intended files after the full suite passes, push the requested branch with GitHub CLI credentials, and report the independently verified remote SHA in the final response.

## Evidence and decisions

- Initial worktree clean on `feat/veris-seaice`; origin is `nutrik/vercor`.
- Verified native adapter uses `vs.xu[1:-2]` verbatim, and remapper stores source endpoints unchanged before calculating overlaps.
- Baseline `scipy` parallel fast run aborts all workers before collection. Focused `scipy -n0 --fast` remapper/regridder run passes 27 cases.
- Ruling: implement inline on the explicitly requested branch; the user's detailed defect report supplies the repair scope and authorization.
- Ruling: keep native grid coordinates as provenance; repair shared remapping geometry rather than adding a setup-specific edge patch. Conservation diagnostics already consume the remapper's stored bounds.

- RED26 reproduce the defect, including target column coverage 0.43333333333333335 in float64. Review found three compatibility cases (descending ordering, descending anchor alignment, and a shifted 420-degree cell), pinned by four additional RED cases. Final independent focused review passes101; no remaining material finding.
- The normalizer preserves explicit positive widths and descending destinations. Actual static support limits select periodic copies; blindly applying shortest-arc unwrap or always using three fixed shifts would alter existing broad-cell behavior.
- Live native replay matches all supplied rounded areas; corrected NumPy/production wet coverage maximum absolute difference4.44e-16. Original four archived arrays were not supplied, so their archive-equality claim is not certified.
- Initial broad checks: stock fast/full hit a sandbox-only loopback socket restriction; final runs use authorized localhost access. Native AD15 passed. First type check found five NumPy-scalar tolerance annotations in new tests; converted them to Python floats.
- Strict Sphinx with extra nitpicky `-n` flags396 existing unresolved references; the repository's ordinary strict `-W --keep-going` build passes. Maintained lint0; repository-wide lint keeps200 pre-existing test_logs snippet findings.
- Two preliminary coverage runs were interrupted to separate stock/AD coverage paths under /private/tmp before the final run. Raw final lane data will be retained separately.

- Final gates: stock2502, fast1552 and exact-forkAD15 pass with zero failures/errors/skips. Combined coverage92.89480740012512% exceeds unchanged90%; raw stock/AD SHA-256 values retained in /private/tmp/vercor-longitude-final-results.json. All311 source/test hashes are unchanged from final-run capture.
- Final host and all-JAX two-day runs each pass96 native atmospheric observations, zero selected-native SST/ice mismatch,252 prescribed polar cells and9 shared output files; geometric receipts agree. Elapsed72.61s host/87.05s JAX.
- Black/mypy311/maintainedflake8=0/compileall/strictSphinx/build/diff pass. Wheel/source comparison matches180 production modules. Full output logs and JUnit receipts are under /private/tmp/vercor-longitude-*; installed environments and physics tolerances are unchanged.
