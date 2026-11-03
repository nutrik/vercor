# Exchange remapping profiling design

The user requests source optimization beyond output, using XProf, Perfetto and
TensorBoard, with scientific behavior preserved and delivery on the current
`feat/veris-seaice` branch. This is maintenance of existing behavior. Root owns
implementation and profiling; an independent performance specialist inspects
candidates and a fresh reviewer checks the final change.

## Approach

Reuse the already resolved source/target coverage when retaining each dispatch's
received support. Keep one shared diagnostic formula for dispatch and refresh.
Fuse conservative remapping's existing vectorized NaN neutralization, gathers,
weighting and sparse scatter sums in a private pure JIT method. Pass the existing
remapper PyTree, preserving dynamic weights and stable executable reuse across
equivalent instances. Keep shape and `safe_masked_divide` validation outside this
kernel to preserve eager error classes and ordering.

Alternatives considered: compiling the entire remapper risks changing direct
validation exceptions; refreshing only routes incident to a stepped component
changes validation of unrelated invalid states supplied to internal helpers.
Both are excluded. No persistent mask cache, new public API, padding, dependencies,
physical approximations or changes to output handling are needed.

## Evidence and acceptance

Use isolated processes against an archive of baseline `abc4494` and the working
source. Measure synchronized cold and nine warm calls with separate trace capture
for both normalization modes, scalar/vector dispatch and complete output-free
host/scanned component runs, at float64 and float32. Save every resulting leaf and
compare byte-for-byte where the arithmetic ordering permits it; quantify any
roundoff rather than weakening existing tolerances. Inspect actual XSpace and
Perfetto captures, and load both profiles through TensorBoard's Profile plugin.

Tests must pin a single coverage remap per dispatch, changing mask/field values,
last-dispatch support, source/target flux semantics, sparse NaNs, denominator
changes, orientation, gradients, eager errors and JIT reuse. Required gates are
Black, strict repository flake8, mypy, fast/full stock tests and the separate exact
Veros-fork lane. Keep existing assertions and update PROGRESS/DEPENDENCIES. Commit
only after the full suite passes, push normally, then verify the remote SHA.
