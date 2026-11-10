# Shared exchange and output reduction

Approved in chat on 2026-11-03: proceed with the shared reducer design.

Exchange samples and period-output samples have separate owners and clocks.
Exchange field names remain in existing component contracts and adapter mappers;
`OutputSpec` and `PeriodOutput` select only optional diagnostic output. No public
configuration or sampling-cadence change is needed.

The chosen approach is a small private numerical module below both owners. It
provides finite sample sums/counts, safe division, and an immutable equal-duration
sample accumulator. Output retains its metadata/schema validation and per-cell
finite counts. Native exchanges retain strict sums and a scalar sample count:
nonfinite native samples must still propagate to existing physics checks, rather
than silently disappear as missing diagnostic samples. This avoids allocating
one redundant count array per native field whose samples are all required.

Veros uses the same streaming accumulator in its existing native
loop. JCM scans its compact native sample history, diagnosing nonlinear canonical
fields before adding each sample; it no longer materializes a second trajectory
of mapped fields. Its native API still returns compact histories, so this does
not claim to remove native-model trajectory allocation. Veris keeps its validated
native loop and rate integrals; the shared safe division computes its state,
flux, and budget means with the existing duration weights.

Moving sampling into the output session was rejected because optional output
selection and coupling-step cadence cannot own mandatory native-step exchanges.
Keeping separate reducers was rejected because it preserves duplicated mean
arithmetic. Replacing the native JCM stepping API is outside this narrow repair.

Validation covers finite/missing values, empty reductions, dtype, JIT, JVP/VJP,
nonlinear mapping before means, singleton endpoints, continuation/reset, native
invalid-sample rejection, bounded storage, and independent output selections
across JAX, host, mixed and custom backends. Preserve native payload endpoints,
prescribed-record selection, source weighting/remapping, and existing masks.

Work stays on `feat/veris-seaice`. Run formatting, lint, typing, focused, fast,
full stock and isolated exact-fork AD tests before committing and pushing.
