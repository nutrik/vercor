# Veros Salinity Restoring Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans for inline execution
> and obtain an independent whole-change review before delivery.

**Goal:** Add an independent salinity restoring time scale, default to four
365-day years in seconds, enable salinity restoring by default, validate,
commit and push on `feat/veris-seaice`.

**Architecture:** Add keyword-only `VerosConfig.salinity_restoring_timescale`
with default `4 * 365 * 86400.0` seconds. Pass it through the existing factory
and setup owner into a native scalar variable, keeping the independent
restoring weights and differentiable forcing kernel. The existing config
owns defaults; the native wrappers reuse them. No new runtime module.

**Tech Stack:** Python, JAX, Veros, pytest, Black, flake8, mypy, Sphinx, GitHub.

**Spec:** The user's explicit parameter/default/test/documentation request and
repository `AGENTS.md`. This is one bounded change with fully specified behavior.

## Global Constraints

- Work in the current checkout and branch as explicitly requested.
- Preserve positional config arguments, native ACC forcing, coupled fluxes,
  masks, nonaccumulating forcing and the existing temperature correction.
- Salinity restoring defaults on; temperature restoring remains off.
- Reject nonnumeric, nonfinite and nonpositive public time scales.
- Full tests and static checks must pass before the authorized commit/push.

## Review Focus

- Default and custom seconds survive factory/setup forwarding.
- Each independent flag combination preserves coupled fluxes.
- Top-cell tracer response uses the configured time scale exactly once.
- Native time scale remains differentiable with JVP/VJP/finite differences.
- Galleries and public signature documentation reflect the new defaults.

### Task 1: Configure, apply and deliver salinity restoring

**Files:** `vercor/setups/config.py`, `_external/veros_gcm.py`,
`_external/veros_gcm_state.py`, `_external/veros_setup.py`, gallery examples;
existing freshwater/ice-feedback/native-AD/constructor tests and
`tests/test_veris_prognostic_gallery.py`;
public signature fixture; `docs/setup-gallery.rst`, `DESIGN.md`,
`DEPENDENCIES.md`, `CHANGELOG.md`, `PROGRESS.md`.

**Interfaces:** Public keyword-only `salinity_restoring_timescale: float` in
seconds; corresponding native scalar in seconds divides the additive SSS
restoring term. Existing boolean controls remain separate.

- [x] Add/update tests and observe expected failures before source changes.
- [x] Pass the parameter to native state and replace the 30-day constant.
- [x] Update galleries, current documentation and public signature fixture.
- [x] Run focused/fast/full stock and separate native-JAX/AD validation.
- [x] Run formatting, strict lint, typing, compileall, strict Sphinx and diff checks.
- [x] Obtain an independent review and address any findings.
**Delivery:** Commit intended files, push and verify the remote SHA and
clean tree. The final handoff and external validation receipt record delivery.

## Execution Notes

- Baseline `conda run -n scipy pytest tests/ -q --fast` aborted all four
  Python 3.14 native workers during optional native dependency import before useful results.
- Pre-flight: one task; no cross-task interface conflicts. User authorization
  fixes the checkout, branch and commit/push delivery method.

- RED: 29 expected host failures and 8 native-AD failures. Focused GREEN:
  38 host cases and 8 native-AD cases. A later integration test retained
  the former no-restoring expected salinity; its independent two-step Euler
  mean now includes default restoring (corrective2/2, focusedfast39/39).
- Python3.14 fixture generation changed typing rendering; regenerated only
  the VerosConfig signature with the canonical Python3.12 interpreter.
  A temporary kwargs annotation was corrected for mypy. Initial lint used
  its79-column default and Sphinx added nitpicky references; project checks
  use the AGENTS120-column limit and Sphinx -W without the extra -n mode.
- Black331/mypy331, repository flake8zero, compileall, strict Sphinx and
  whitespace checks pass. Independent review reports no findings.

- The initial fast run exposed three failures: Python-version signature
  rendering, the public ocean two-step expected mean, and the prognostic
  ice/ocean salt budget omitting climatological restoring. Corrective tests
  pass2/2 and1/1 respectively; the budget now includes the independently
  derived600-second restoring source. No production changes were needed.

- All22 native autodiff cases are accounted for:21 passed in the complete
  run; the remaining precipitation-gradient case exposed the old no-restoring
  analytic expectation. Its independently derived geometric attenuation
  passed a focused rerun at unchanged tolerances. Native implementation
  sources are unchanged. Supplemental review of all three analytic test
  corrections reports no findings. Final fast suite:1801/1801, no skips.

- Final stock suite:2770/2770, no failures/errors/skips; nativeAD22/22
  across the complete and corrective runs, fast1801/1801. All2792 collected
  cases are accounted for. All331 source/test hashes remain unchanged after
  the final test-only correction and formatting. Stock coverage from the
  earlier expectation-correction run is92.02%; final stock run is green.
- Evidence: `/private/tmp/vercor-salinity-validation.json`, corresponding
  JUnit reports/logs, and `/private/tmp/vercor-salinity-source-hashes.json`.
