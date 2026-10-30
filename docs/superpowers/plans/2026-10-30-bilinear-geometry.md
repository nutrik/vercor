# Bilinear geometry audit and repair

> **For agentic workers:** Use superpowers:executing-plans for the test-first repair and superpowers:requesting-code-review before delivery.

**Goal:** Audit bilinear longitude handling for defects related to the conservative seam failure, fix verified defects, and push `feat/veris-seaice`.

**Architecture:** Retain the existing geometry, weights, extrapolation, and public regridder owners. Compute fractions from degree differences. A cached construction-only binary search compares source candidates in each target's whole-turn frame; precise target phases stay near zero. Preserve original columns, raw nonwrapped widths, stable seam widths, endpoint conventions, source orientation once, and identity bypasses.

**Spec:** The user's bilinear audit request; `AGENTS.md` and `DESIGN.md`.

## Review focus

- Native T31/4-degree periodic seams, exact points, whole turns, shifted origins, descending source order, and independently known scalar/vector results.
- Scalar/vector source masks, target masks, inactive NaNs, nearest/IDW fallback, field JIT/JVP/VJP, and zero-support behavior.
- Nonperiodic endpoint clipping, irregular spacing, ascending/descending latitude, and public regridder integration.
- No new public API, dependency, numerical fudge factor, or weakening of tolerances.
- Preserve construction-only geometry and runtime compilation boundaries.

## Task 1: Audit and repair verified defects

- [x] Trace source/target coordinates, periodic lookup/fractions, vector bases, masks, extrapolation, and exchange call sites.
- [x] Add regressions against analytic seam weights and spherical vectors, plus descending masked extrapolation, and observe RED.
- [x] Repair the existing owners with one source-orientation boundary and stable geometry arithmetic.
- [x] Run focused interpolation, regridder, compilation, and coupling tests; obtain independent read-only review.
- [x] Run formatting, lint, mypy, full/fast stock tests, separate exact-fork AD tests, coverage, docs, packaging, and native checks.
- [x] Update dependency/progress/audit documentation with verified findings and limitations.
- **Authorized delivery:** Commit only intended paths after the full suite passes, push with GitHub CLI credentials, and verify the remote SHA.

## Evidence and decisions

- Started clean on `feat/veris-seaice` at `13c31ee69d6450f1b3517eeea6d03c9b8a3ebb8e`.
- Bilinear geometry consumes source centers, not longitude edges. Nonmonotonic source centers are rejected before interpolation; the native conservative-edge defect does not silently drop a bilinear strip.
- Focused baseline fast run:15 pass in `scipy -n0`. Reuse the currently verified Python3.12 stock/AD launchers for native/full validation after the known scipy3.14 parallel abort, without modifying installed environments.
- Verified descending fallback reproduction: one valid source at longitude3 with u=30 gives scalar30 but vector0 in both nearest/IDW modes and both precision policies. `_finish_vector` supplies already-oriented values to `_extrapolate_scalar`, which flips them again against an already-oriented mask.
- Verified seam-weight cancellation: expected2/3.75; float32 gives0.5333347916603088 at origin0 and0.5333236455917358 at origin720. Float64 origin720 gives0.5333333333333117. Ratios do not require conversion to radians.
- Ruling: preserve the established identical-grid no-op; explicit exchange dispatch already owns source weighting and target support. Do not add a second masking owner.
- Ruling: implement inline on the explicitly requested branch using the supplied audit scope; no unrelated design or branch change.

- Initial fixtures incorrectly attempted descending RectilinearGrid construction; corrected them to use the supported low-level descending interpolator. Isolated unchanged baseline source reproduces34 failures/10 passing controls with no errors. Affine RED8, micro-coordinate reviewRED6/8, in-range-originRED4, aliased-phaseRED2, fine-seamRED3/4 are retained under /private/tmp/vercor-bilinear-*.
- Superseded approach: a fine seam cannot encode a small query phase by adding360 in float32. Proper periodic grids now share canonical[-180,180) phases for lookup and ratios, with original-column index mapping; eager fsum classifies span accurately. Inclusive/broad periodic inputs retain legacy first-turn behavior. No runtime metadata, PyTree leaves, public API, or dependency is added.
- Review caught offset-by180 canonicalization losing near-zero phase; signed fmod fixed it. A separate source-origin modulo cancellation is avoided in the legacy wrapper by subtracting whole turns directly and retaining in-range inputs.
- Focused final gate passes145 cases including76 new regressions and runtime-compilation checks. Earlier broad runs were explicitly stopped as superseded when the final fine-seam review finding required a geometry correction; they are not counted as final gates.

- Final ruling: canonical sorting plus a legacy span path left the same verified alias defect in inclusive/wider grids. One cached fixed-iteration binary search now compares source candidates after subtracting the target-frame integer turn, never adding it to the target. Raw widths retain all first-turn semantics; wrapped widths and target offsets use stable positive phase differences. No span branch, sorting, dense target/source array, or extra runtime state is needed.
- Added exact-coordinate inclusive/wide alias RED8; final focused153 includes84 new regression cases. The source-turn search prototype matches Fraction-based seam oracles in both precisions, including target0/first-half/360 and endpoint360.

- Final source review is clear:153 focused cases,1028 exact-coordinate scalar/index/order oracles per precision, and masked vector/VJP oracles across both precisions/all axis orders. Final unified search also resolves exact360 and wider-domain alias RED8 without sorting, span rules, or target translation.
- Final native host/all-JAX runs each complete2days/96 atmospheric observations/252 prescribed polar cells/9 output files. Both atmosphere-to-ocean and ocean-to-atmosphere all-wet geometry agree with SciPy RGI within9.992e-16 for vectors and3.331e-16 for scalars; unit coverage is exactly1, identity routes retained, selected-native polar SST/ice errors are0. Native receipts are /private/tmp/vercor-bilinear-native-{host,jax}/verification.json.
- Black/mypy312/maintainedflake8=0/compileall/strictSphinx/build pass. Repository lint retains200 historical test_logs findings. Final broad gates use separate /private/tmp/vercor-bilinear-final-coverage-{stock,ad} files and a frozen312-file source/test hash manifest.

- Final gates: full stock2586, fast1636, exact-forkAD15; zero failures/errors/skips. Combined coverage92.9221706533381% exceeds unchanged90%; raw lane SHA-256 values remain unchanged and are recorded in /private/tmp/vercor-bilinear-final-results.json. All312 source/test hashes match the frozen manifest. Strict Sphinx, Black, mypy312, maintained lint0, compileall, build, wheel/source identity180 and diff checks pass.
- Final native host/all-JAX geometry receipts are identical: maximum scalar error3.331e-16, vector error9.992e-16 versus independent SciPy, exact unit coverage, native ocean/ice identity, zero prescribed polar SST/ice mismatch. Native run elapsed24.99s host/32.64s JAX.
