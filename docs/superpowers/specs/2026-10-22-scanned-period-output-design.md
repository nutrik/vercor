# Scanned period output

The user approved the previously presented recommendation to accumulate numeric
period output inside `lax.scan`, and requested implementation, complete checks,
commit and GitHub CLI push on the existing `feat/veris-seaice` branch.

## Required behavior

For the default all-JAX workflow, output-free execution retains its existing
single checkpointed scan. Period output with JAX-compatible providers executes
one scan between successive output boundaries, splitting additionally where
the ordered component schedule changes. Boundaries are the union of all
component cadences. Every clock step still samples every declared provider
after stepping, using the same absolute step and selected runtime fields.

Carry the existing immutable sum/count accumulators alongside RunState. Reuse
their finite-count and native sample-dimension reductions, latest coordinate
values, independent resets, selected variables and static metadata. Do not
retain a trajectory. File writes, calendar conversion and final/snapshot output
remain with the existing host coordinator. Incomplete periods are not flushed.

## Provider contract

Keep `OutputProvider.sample(OutputContext)` unchanged. Add an optional structural
`JaxOutputProvider` protocol with `sample_jax(JaxOutputContext) -> OutputFrame`.
The frozen context contains component, state, payload, traced absolute step and
static dt_seconds; it intentionally has no Python datetime. This method is pure,
supports abstract tracing, returns numeric JAX-compatible arrays and has fixed
variable/coordinate schema, metadata, shapes and dtypes for the whole run.

The default field provider and bundled JCM, Veris and Veros providers share
numeric extraction between the two methods. Their ordinary method supplies the
real timestamp coordinate; the numeric method omits that coordinate. Veros
selects the native time slot with JAX indexing instead of host integer coercion.
Any enabled provider without the optional contract preserves the existing
single-step host sampling behavior. Host/custom execution stays unchanged.

## Runtime ownership

Extract a reusable private JAX runtime step builder from backends.py. The
output-free executor and a private output scan executor both use that step;
there is one exchange/receive/physics/mask/send implementation. The backend
does not receive an output session or own paths/cadences. Output's private scan
module composes the step with numeric sampling and accumulator updates.

Initialize a fixed-shape zero accumulator template by abstractly tracing the
first scheduled step plus explicitly opted-in sampling. This performs no actual
physics advancement, sampling side effect or initial-state accumulation. The
real first step belongs to the first scan. Normalize accumulator coordinate
leaves to arrays. Zero-step runs do not sample or compile an output executor.

Keep cancellation and numerical validation behavior. This work does not reduce
checks, change error latency policy, add asynchronous writes, change physical
time steps or alter mask/exchange ownership. Reuse compiled executors per
component schedule; JAX owns length-specific compiled variants.

## Acceptance

- Ten six-hour steps with daily means execute scans [4,4,2], write means
  2.5 and 6.5, finish at 10 and do not write the incomplete tail.
- Independent cadences, schedule changes, Gregorian/leap/noleap/360-day clocks,
  omitted components and repeated runs preserve values and timestamps.
- Numeric providers preserve finite-count/sample-axis reductions, coordinate
  updates and schema diagnostics; ordinary providers preserve exact Python
  step/time context and existing errors.
- Disabled/final-only output retains existing execution and AD behavior.
  Host/custom backend driver validation and cancellation remain covered.
- Real bundled JAX extraction and native smoke checks prove that the accelerated
  lane is not restricted to dummy components. Compare warm runtime with the
  previous singleton behavior while preserving numeric output work.
- Run formatting, lint, mypy, fast/full tests and relevant native AD acceptance.
  Record evidence, obtain independent review, commit intended files only, push
  with GitHub CLI and verify the remote SHA. Preserve the pre-existing audit log.
