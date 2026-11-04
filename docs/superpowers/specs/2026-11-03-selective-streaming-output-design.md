# Selective streaming component output

## Intent and constraints

Include all available component state variables, fluxes and diagnostics in enabled
period output by default. A nonempty `PeriodOutput.variables` selects an ordered
subset before extraction. Output remains opt-in (`OutputSpec.period=None` and
`output=None` remain disabled). Preserve native endpoint snapshots, exchange
semantics, calendars, finite-value counts, and backend-independent output behavior.
Work on the existing `feat/veris-seaice` branch and retain current owners.

## Approaches

1. **Recommended:** carry the requested variable names in the existing ordinary
   and numeric output contexts. Bundled providers select before array extraction;
   the shared session validates names/schema and owns running sums/counts. This
   preserves structural providers and gives third-party authors the same policy.
2. Add optional provider-selection methods or bind new providers per run. This
   duplicates the existing contexts/capability dispatch without improving reduction.
3. Filter complete frames after sampling. This preserves today's extraction cost
   and does not meet the user's requirement to avoid sampling unselected fields.

The user approved the recommended approach and implementation on 2026-11-03.

## Behavior and ownership

- The default runtime provider includes all declared state fields, including
  inputs and outputs, without duplicate stores.
- Bundled native providers enumerate their native available state/flux/diagnostic
  registries plus canonical runtime state fields and select names before reading,
  converting, diagnosing or transferring arrays. Native names retain their existing
  meaning on collisions; canonical-only fields retain grid dimensions. Preserve
  model metadata and active-variable rules. Support repeated Veros dimensions
  through unique axis names and index coordinates for implicit native dimensions.
  Runtime inputs use their received values; export transfer selection applies only
  to declared outputs.
- Ordinary and numeric contexts expose a default-empty immutable variables tuple;
  existing provider signatures continue to work. Third-party providers must honor
  this tuple to avoid their own unnecessary extraction. Core validation remains
  available for providers that return a complete frame.
- Reuse the existing immutable `_OutputAccumulator` for finite sums/counts. Compute
  period means only at completed write boundaries; retain partial windows across
  execution chunks and reset only completed windows. No step trajectory is stored.
- Replace CAMulator's native prediction history and stack-based mapped means with
  immediate per-prediction reduction. Decode nonlinear diagnostics before summing,
  retain only endpoint prediction/model continuation plus fixed-size reductions,
  and avoid native output extraction when period output is disabled. A default-
  disabled static `StepContext.output_variables` request (None disabled, empty
  tuple all) carries the run-specific policy into native stepping. A reduced frame
  can supply `reduction_counts` per variable: counted variable values are reduced
  sums, counts have the same shape, and counts are nonnegative integers. The shared
  session validates these arrays and merges sums/counts without an intermediate
  mean. Variables without reduction counts retain ordinary sampling semantics. Snapshots use
  the endpoint. Do not alter other adapters' physics integration histories where
  those histories are required by their native stepping API.

## Validation and delivery

Write failing behavior tests first for default completeness, selective extraction
(including poisoned unselected values), unknown names, finite counts, reset and
constant accumulator size. Exercise JAX scans, host, mixed and custom drivers;
verify native endpoint behavior and streaming CAMulator equivalence. Run formatting,
linting, typing, fast/full tests and applicable isolated AD checks; document any
pre-existing environment failures. Update design/dependencies/progress, review the
final diff, commit only intended paths, push with GitHub CLI authentication and
verify the remote branch SHA.

## Builder review

Read-only review identified the required native/canonical union, run-specific
CAMulator step policy, and preservation of per-cell finite counts. These are
incorporated above. Internal callback signatures and helper placement are left
to implementation; no additional model or backend policy is introduced.

## Acceptance cases

- Input-only runtime fields are included; monthly/daily export policies do not
  reinterpret received grid arrays. Native/canonical collisions preserve existing
  native temporal meaning and metadata.
- Poisoned unselected arrays and endpoint diagnostic callbacks are never accessed.
  Unknown selections fail with component context; subset ordering is retained.
- Counted native frames match independent finite native-sample sums/counts,
  including changing missing-data patterns and all-missing cells.
- CAMulator continuation and endpoint snapshots match existing results, native
  nonlinear exchange diagnostics are diagnosed before reduction, and repeated runs
  with output enabled/disabled do not mutate setup resources or retain history.
- Accumulator leaf shapes and storage bytes remain constant as step count grows;
  means are evaluated only at completed cadence boundaries.
