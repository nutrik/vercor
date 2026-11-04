# Explicit Veros precision and measured runtime performance

The user selected the focused approach on 2026-11-02 and authorized autonomous
implementation and verification. Work starts from clean `feat/veris-seaice`
commit `7c774d487cd23811b3b94e4588c5cabbed2b5433`.

## Intent and scope

Close the explicitly deferred Veros float32 validation, obtain reproducible
Perfetto evidence for VerCOR performance, optimize measured overhead, resolve
the historical repository lint findings, and remove redundant functional tests.
Preserve independent numerical, physical, autodiff, backend, boundary, and error
coverage. Preserve public APIs, native physics, temporal means, native endpoint
continuation, masks, SI units, and the runtime dtype policy.

## Approaches

The user selected a focused change in existing owners: separate precision and
native-backend processes, synchronized cold/warm profiles, small measured
optimizations, generated-log lint exclusion, and explicit retained-test maps.
A broad runtime redesign adds compatibility risk without profiler evidence.
Simply counting fewer tests or hiding lint error codes would not establish the
requested correctness or maintenance improvement.

## Existing evidence and intended changes

- All 200 historical lint findings are in four ignored `test_logs/**/*.py`
  snippets. All 327 tracked Python files pass the 120-column check. Explicitly
  exclude generated evidence in `.flake8`; keep all lint codes enabled and
  leave archived evidence intact.
- Veros temporal fixtures currently force float64. Native precision must be
  selected before backend import/lock. Exercise actual native allocation and
  advancement in float32, including independent surface/mean/continuation and
  source-budget checks. Use a separate exact-fork JAX process for JVP/VJP.
  Avoid fixture-created float64 arrays and assert actual floating dtypes.
- The existing slab profiler times one compile-inclusive run. Extend its
  existing module to distinguish setup, compilation and synchronized warm
  execution, retain compact CLI output, support repeatable Perfetto capture,
  and expose machine-readable measurements where useful. Profile actual small
  coupled native host/JAX runs separately, with identical inputs per comparison.
- Initial CPU traces show frequent successful validation host callbacks.
  Investigate avoiding host calls when a device-side validation predicate is
  satisfied. Preserve failure messages, invalid-value rejection, JIT/JVP/VJP,
  inactive NaN handling, interruption and output behavior. Keep changes only
  after a controlled comparison demonstrates a benefit.
- Remove functional/API repetitions only after recording the retained nodes
  and their assertions. Preserve independent contract oracles when consolidating
  manifest/signature tests. No blanket test deletion or selection changes.

## Validation and reporting

Use the existing `scipy312` interpreter with verified stock and exact-fork
overlays, separately. No profiler installation is needed for JAX Perfetto
capture. Warm and synchronize before timing; collect traces separately from
untraced timings. Report hardware, dependency versions, grid/steps/dtype,
repetitions, median and spread, compilation/setup cost, callback evidence and
limitations. Native miniatures do not establish climate-run throughput.

Before local commits run focused regressions, Black, repository-wide flake8,
mypy, compileall, fast/full stock suites, exact-fork autodiff and explicit
float32 lanes, combined branch coverage at the unchanged 90% floor, and
appropriate documentation checks. Review the complete diff independently.
Keep detailed evidence outside the checkout and update bounded `PROGRESS.md`.
Use the existing feature checkout; no release, tag, merge or publication is
part of this task. GitHub read access has been verified.
