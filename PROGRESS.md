# VerCOR Progress

This is the bounded orientation log for active development. Detailed history is preserved in `docs/progress-archive-2026-04-23-to-2026-05-15.md`, `docs/progress-archive-2026-05-16-to-2026-07-14.md`, and `docs/progress-archive-2026-07-22.md`.

## Current Status

- VerCOR 0.4.2 release-candidate evidence was refreshed after the README and release-runbook corrections (2026-07-25): supervised TDD recorded three intended 0.4.1-versus-0.4.2 metadata failures, followed by two intended recovery-record and remote-main-ancestry failures; focused GREEN passed 180/180 and fast passed 711/711 with four known Flax/JAX deprecation warnings. Final verification completed locally from commit `5e99397`: workflow YAML parsed and all 20 Bash blocks passed `bash -n`; Black left 238 files unchanged (with its Python 3.13/target-Python-3.15 safety-parse advisory), flake8 reported 0, mypy/compileall/whitespace passed, and the focused release gates passed 209/209. Fresh configured suites passed fast 711/711 (four Flax/JAX `jax.core.Effect` deprecation warnings), full 1,349/1,349 (those four plus the JCM/xarray `compat` future warning), and coverage 1,349/1,349 at 91.13% branch coverage (7,404 statements; 1,562 branches). A fresh exactly-two-file build in `/private/tmp/vercor-0.4.2-build.ypvaAX` passed Twine, wheel metadata/PEP 561, and an outside-checkout dependency-free installed-version 0.4.2 smoke (`/private/tmp/vercor-0.4.2-smoke.0eJlbq`): `vercor-0.4.2-py3-none-any.whl`, 213305 B, SHA-256 `271a726a25fde21e7c68104117b19d52b39bd1b591fc52dc22dbf462cf5da52e`; `vercor-0.4.2.tar.gz`, 156886 B, SHA-256 `3771217cdb59fa11910223355c726ce27b1b79f35444f4fe9a4484255236bcdc`. Neither `v0.4.0` (`8b67b06773810b5b669426939d746834701a65e3`) nor the stale `v0.4.1` tag (`f4fc2a5661b73e549e6076cf14431242984abc50`, peeled `e0748ee5d31e40cf08a2066013981f35c91bea0f`) was changed; no tag, push, publication, upload, release creation, or other remote mutation was performed.
- VerCOR 0.4.1 release candidate evidence was refreshed after the README and release-runbook corrections (2026-07-24): the release transcript now pushes and verifies the exact remote release branch before GitHub preflights, reauthenticates that branch before selecting the exact PR run, and merges, rebinds, and detaches at protected `main` before the independent tag preflight. Wave-2 TDD recorded `4 failed, 34 passed, 15 deselected in 3.82s` RED and `38 passed, 15 deselected in 3.67s` focused GREEN. Fresh final gates passed: YAML plus all 19 workflow Bash blocks; Black left both edited test files unchanged with its known Python-version advisory; flake8 reported 0; fast printed `710 passed, 4 warnings in 23.30s`; full printed `1348 passed, 5 warnings in 47.93s`; and branch coverage printed `1348 passed, 5 warnings in 52.40s`. The four fast warnings were the known Flax/JAX deprecations; full and coverage added one known JCM/xarray future warning. Branch coverage was 91.13% across 7,404 statements and 1,562 branches. A fresh exact two-file build in `/private/tmp/vercor-0.4.1-wave2-build.UqUTPD` passed Twine, wheel metadata/PEP 561, and isolated no-dependency installed-version 0.4.1 checks (`/private/tmp/vercor-0.4.1-wave2-smoke.WBu1NL`): `vercor-0.4.1-py3-none-any.whl`, 213304 B, SHA-256 `df6d0073538868c1eec8b799b6b31d6934aa279085572781b27c6e61e36c93fa`; `vercor-0.4.1.tar.gz`, 156862 B, SHA-256 `7dd953e1501d2a033e22985c4dd3c3585559dcf01343dbf2cf3a9f4a3cffcb19`. No parent-checkout, remote, tag, push, pull-request creation, merge, upload, publication, or hosted state was changed. The existing tag object is `8b67b06773810b5b669426939d746834701a65e3`, peeled commit `d298e895981037684e4c8576796e62620434b395`. Verified: `v0.4.0` was not changed.
- Automated tagged release final-review and re-review corrections completed locally (2026-07-24): exact producer artifacts, immutable/credential-free actions, per-tag serialization, exact Twine, peeled-tag gates, absence-only ordinary reruns, bounded PyPI verification, and the zero/one/two-asset draft state machine remain enforced. A follow-up closes the missed workflow integration: both draft-visibility preflights now query authenticated repository JSON, require boolean `permissions.push` true, and only then enumerate releases; the regression contract preserves that order. The original final fixes also reject false/missing/malformed permission state and commit an installed-`gh` request-construction regression. With a synthetic token and closed loopback proxy, the test proves the old hostname form targets invalid `api.uploads.github.com`, while the production helper's full URL targets `uploads.github.com` and never contacts GitHub or leaks credentials; GH config/state is isolated under `tmp_path`, and the discovered `.local/state/gh/device-id` artifact was removed and remains absent. Follow-up TDD RED was the expected missing workflow call; GREEN passed 35/35 focused, 80/80 release/API/validator, and 708/708 fast with the four known Flax/JAX warnings. YAML and all 19 workflow Bash blocks, Black (with the known Python-version advisory), flake8, and focused mypy passed. Original verification also passed full 1,346/1,346. No push, tag, upload, publication, hosted release mutation, or other external mutation was performed.
- Final release-inventory review fixes completed locally (2026-07-24): directory-backed distribution reuse/local builds, CI upload, and the release transcript now fail closed unless the bundle contains exactly the VerCOR 0.4.0 wheel and sdist; the release build also requires an absent/empty `dist/`, while publication keeps explicit two-file arguments. The dependency-free slab test installs only the VerCOR wheel. RED was four intended assertions; focused GREEN and the slab/helper focus passed 4/4 each; distribution/architecture passed 27/27; fast passed 675/675. Fresh post-fix gates passed: Black 237 unchanged with its known Python-version advisory, flake8 0, mypy 237, strict fixture mypy 4, compileall, full/coverage 1,313/1,313 at 91.13% (7,404 statements, 1,562 branches), and final review with no findings; warnings were four Flax/JAX deprecations and one JCM/xarray future warning. The user's ignored `dist/` files were not modified; no push, tag, upload, publication, merge, or hosted release was performed.
- External-extension fixture isolation verified locally at commit `0a4c1dc` (2026-07-24): the complete fixture rename is `tests/fixtures/external_extension_test_fixture` / `external_extension_test_fixture`; the release, CI upload, and checksum boundary contains exactly the two VerCOR artifacts (`vercor-0.4.0-py3-none-any.whl` and `vercor-0.4.0.tar.gz`), while the fixture is built and tested separately. TDD RED evidence was 12 fixture/boundary assertions (old embedded `public_plugin` ownership), 3 CI/release assertions (old job/smoke/checksum boundary), and 1 active-source assertion naming four stale documents; GREEN was 27/27, 3/3, and 1/1 respectively. Fresh gates: Black left 237 files unchanged (exit 0, with its Python 3.13/target-Python-3.15 safety-parse advisory); flake8 0; mypy 237 source files; strict fixture mypy 4 source files; and compileall all exited 0. Fast/full/branch-coverage pytest passed 674/674 and 1,311/1,311; branch coverage was 91.13% (7,404 statements, 1,562 branches), with four Flax/JAX `jax.core.Effect` deprecations and one JCM/xarray `compat` future warning on full/coverage. Separate temporary builds passed: `/private/tmp/vercor-task4-release.wKdcps` contains exactly the VerCOR wheel and sdist, and `/private/tmp/vercor-task4-fixture.yvechu` contains only `external_extension_test_fixture-0.1.0-py3-none-any.whl`. No push, publication, upload, tag, or hosted release was performed.
- VerCOR 0.4.0 release verification completed locally from uncommitted local release changes (2026-07-23) on Python 3.13.13: metadata 4/4, documentation 12/12, fast 664/664, final full 1,311/1,311, earlier branch coverage 1,302/1,302 at 91.13%, and optional-model/gradient acceptance 12/12 passed.
  Python 3.12 and hosted CI were not run locally; the GitHub connector returned 404 for configured origin `nutrik/vercor`, and remote configuration was not changed; artifacts built once with the prescribed `scipy` commands are not yet tied to an exact release commit.
  `dist/vercor-0.4.0-py3-none-any.whl`: 213305 B, SHA-256 `a50773003b9d74cf1670f887b0951487bac87528c6ed3e4f7c4e54cfc16495d8`;
  `dist/vercor-0.4.0.tar.gz`: 158745 B, SHA-256 `e2f3633991bb983b7144d6f7c631a274b4188013c962df6d28c8b6465c39250d`;
  `dist/vercor_public_plugin-0.1.0-py3-none-any.whl`: 6087 B, SHA-256 `b463de7a8f0f12fc3ae443ab0ebc4afe73d8c072aabb230d293b15ba055181ef`.
  Archive metadata/PEP 561 and main wheel/sdist 152+152-member forbidden-file inspection passed (plugin wheel: 7); supplied bundle boundary 16/16 and outside-checkout installation/import version 0.4.0 passed; artifacts are ignored.
  Release safety TDD captured 4 structural RED failures and 5 exact-state validator RED failures, then passed 4/4 and 5/5: transcripts are fence/`bash -n` checked, all CI checkouts bind the triggering SHA, authenticated repository/release/PyPI and `main`-ancestry preflights repeat at mutation boundaries, and recovery rejects extra names while re-querying exact tag/package/asset state and freshly validating hosted bytes.
  Preflight found branch `refactor` and no local or remotely advertised `v0.4.0` tag; Twine passed both VerCOR artifacts while the evidence-only plugin wheel had nonblocking missing-long-description warnings; other warnings were pip user-cache disabled, four Flax/JAX-effect instances, and one JCM/xarray instance.
  No commit, tag, push, publication, upload, merge, or hosted release was performed.
- Time-selected runtime output verification completed locally (2026-07-23): period accumulation had sampled raw climatology/forcing-record axes rather than the scalar time-selected field; shared transfer-policy selection now supplies the default provider's precomputed `RuntimeStepInfo`. Focused selector/output checks passed 3/3 and 95/95; fast 664/664; full and branch coverage 1,302/1,302 at 91.13% (7,404 statements; 1,562 branches); maintained output-free JVP/reverse gradients 3/3. The stale `tests/test_gradients.py` path does not exist. Black left 236 files unchanged; flake8 0; mypy 236 files; compileall, installed wheel/sdist/plugin artifact boundary (16 tests), optional-model nodes (9 parameterized tests), and whitespace passed. Warnings unchanged: four Flax/JAX `jax.core.Effect` deprecations and one JCM/xarray merge-default future warning.
- Bundled output defaults aligned locally (2026-07-23): omitted slab, ERA5,
  ERA-Interim, and direct/paired JCM-land declarations now resolve to
  `OutputSpec()`, matching external configs with no period policy; explicit
  step/month policies and final-field output remain independent. TDD RED was
  14 expected failures; focused passed 57/57 and the applicable
  output/setup/API/artifact fast set passed 51/51. Fast passed 664/664 with four
  known Flax/JAX warnings; full and branch coverage passed 1,298/1,298 with one
  additional JCM/xarray warning. Coverage is 91.10% across 7,398 statements and
  1,560 branches. Black left 236 files unchanged; flake8 was 0, mypy passed 236
  files, compileall, installed artifacts, and whitespace checks passed.
- Installed plugin evidence and author guidance completed locally (2026-07-21): the fixture accepts `0.4.0a1` through `<0.5`, its non-default configuration produces independently observable JAX, host, exchange, and backend results, and CI now resolves the plugin against the installed VerCOR wheel. The new public-only author guide executes configuration, payload, regridding, topology, workflow/backend, output, and fake-testing examples in order. The reviewed guide composes through a plugin-owned factory, samples payload state, and release instructions preserve normal plugin dependency resolution.
- Stable extension and factory typing completed locally (2026-07-21):
  `RegridderFactory` is one runtime-checkable public protocol, the plugin
  fixture may import only the six-symbol root/stable extension tier, and the
  review/design distinguish that tier from retained alpha inventory and JAX
  integration hooks. RED was the expected `TYPE_CHECKING` source-boundary and
  extension-import failures; focused GREEN and strict plugin mypy passed, as
  did the full suite. The user-approved follow-up narrowed the factory protocol
  to its two required grids, restoring built-in/default factory compatibility.
  No public export was removed.
- CAMulator atmosphere payload ownership completed locally (2026-07-21): setup seeds a frozen native payload; functional stepping clones and advances payload-owned model state, cursor, forecast hour, and predictions; providers and snapshots sample only context payloads. TDD RED was 5 expected failures; exact focus passed 18/18, complete focus passed 90/90, static gates passed, and full passed 1,242/1,242 with five known third-party warnings.
- Explicit CAMulator forcing alignment and functional land cursor completed locally (2026-07-21): `strict` now rejects coupler/forcing start mismatches, `forcing_start` opts in without warnings, typed configuration carries the policy, the immutable cursor advances functionally, and land owns its cursor in runtime payload; TDD RED was 4 expected failures and focused GREEN passed all selected tests.
- Veros payload ownership completed locally (2026-07-21): setup seeds the native payload; runtime returns `StepResult` without mutating setup resources; provider and snapshot output read context payloads. RED was 4 transition/provider plus 2 setup failures; focused GREEN passed 79/79 (8/8 non-fast ownership), and full passed 1,242/1,242 with five known third-party warnings.
- Static component identity through setup completed locally (2026-07-21): setup cannot replace declared name, grid, or spec; the adapter revalidates after the hook before examining its result. TDD RED was 3/3 expected failures; focused GREEN passed 42/42.
- CI artifact and NetCDF backend stability completed locally (2026-07-21): the quality job now reuses the build-once artifact bundle, forcing fixtures explicitly use h5netcdf, and JCM packaged input loading temporarily prefers h5netcdf without leaking xarray configuration. After the formatting follow-up, the static gates passed: Black left 234 files unchanged (with its known Python 3.13/target-Python-3.15 safety-parse advisory), flake8 was 0, mypy passed 234 source files, and compileall/whitespace checks were clean. The exact direct-`scipy`-interpreter workflow contract command with `-n0` passed 1/1. Fast passed 638/638 with four Flax/JAX-effect deprecation warnings; full and branch coverage passed 1,235/1,235 without NetCDF/HDF failures, with those four warnings plus one JCM/xarray merge-default future warning. Branch coverage was 90.75% across 7,287 statements and 1,524 branches.
- JAXGCM runtime dtype warning fixed locally (2026-07-21): the adapter applies the runtime-owned `DTypePolicy` before pressure and altitude calculations, preventing `float64` promotion and the incompatible JAX scatter into `float32`; the warning-as-error and mapped-field dtype regression had 1/1 RED, focused GREEN passed 11/11, fast/full passed 638/638 and 1,235/1,235 without the scatter warning, Black left 234 files unchanged, flake8 was 0, mypy passed 234 source files, and whitespace checks were clean.
- Codebase simplification completed locally (2026-07-20): Tasks 1-8 reduced
  private grid/regridding/runtime/component state and focused setup paths while
  preserving public behavior; all task focuses, fast/full/coverage, and static
  gates passed. Detailed evidence is retained in the task reports and archive.
- Period-average window-start identity completed locally (2026-07-17): filenames and NetCDF times use each schema's actual start across partial/subsequent periods, mixed cadences, and Gregorian/no-leap/360-day clocks while preserving calendar ISO formats, post-step provider times, means, and incomplete-period behavior. TDD RED/GREEN was 4/4; output focus 25/25.
  Black/flake8/mypy/compileall passed; fast passed 636/636 and full/coverage 1224/1224 (90.50% across 7,361 statements and 1,538 branches), with known third-party warnings.
- VerCOR 0.4 deprecation cleanup completed locally (2026-07-17) in commits `f82588f` through `04e6f45`. Obsolete evidence,
  mutable helpers, absence-only guards, and old adapter tests are gone; positive contracts cover public ownership, lifecycle,
  immutable state, artifacts, output, numerics, JIT, and gradients. Supported foreign-state, calendar, transform, lazy-import,
  payload-copy, and offline-artifact behavior remains. Focused gates passed 34/34, 94/94, 136/136, and 301/301; docs 175/175;
  fast 632/632; full/coverage 1220/1220 at 90.49%. Black, flake8, mypy, compileall, and whitespace passed.
- Controlled pytest parallelization completed locally (2026-07-16): fixed n4
  loadscope reduced the measured 124.38s serial mean to 61.66s (50.43%) while
  preserving selection, warnings, and 90.52% combined coverage. Fast/full,
  Black, flake8, mypy, compileall, and whitespace gates passed; production and
  release behavior did not change.
- Calendar-owned runtime year metadata completed locally (2026-07-15):
  commits `9ade80c` and `a9b079c`. Runtime forcing metadata now derives year
  type and duration per timestamp; the duplicate runtime owner and private
  mapper are gone, and the common-year stdlib `datetime` boundary is fixed.
  Focused GREEN was 136/136; mutation/restoration was 1 failure then 1 pass;
  final fast was 660/660 with 596 deselected and full was 1256/1256. Earlier
  implementation coverage was 90.52% across 7,355 statements and 1,534
  branches; Black, flake8, mypy, compileall, and whitespace gates passed.
- Matcher-level versioning review completed locally (2026-07-15): ordered,
  explicit repository-release contexts replace the broad proximity heuristic.
  RED isolated 4 incorrect cases among 17 parameters; matcher GREEN is 17/17
  and the complete policy/architecture focus is 28/28. External dependency,
  plugin, action, schema, and numerical labels remain accepted.
- Versioning-review follow-up completed locally (2026-07-15): the repository
  policy now rejects contextual major-series shorthand without matching
  numerical values or external versions. RED reported exactly 10 remaining
  repository-owned labels; the corrected policy/documentation focus passes
  11/11 and the progress-archive checksum is refreshed.
- Completed locally (2026-07-15): corrected the unsupervised historical
  release labels to the approved pre-1.0 sequence, ending at `0.4.0a1`. The
  policy and architecture focus passes 30/30 and the fast suite passes 524/524.
  The Conda launcher panic occurred before pytest, so actual checks used the
  direct `scipy` environment interpreter. The approved repository-wide scope is recorded in
  `docs/superpowers/specs/2026-07-15-vercor-versioning-design.md`; the execution
  sequence is in `docs/superpowers/plans/2026-07-15-vercor-versioning.md`. No
  tag, push, publication, or Git-history rewrite is authorized.
- VerCOR 0.4.0a1 Task 10 candidate preparation was completed and committed in
  repository history on 2026-07-14. Tagging, pushing, and publication remain
  intentionally unperformed pending separate authority.
  Task 9 was explicitly skipped: no legacy adapter namespace is implemented.
- Tasks 1-8 plus Task 10 form the complete alpha series. The current API has a six-symbol root,
  protocol-first components, constructor-only coupling, traced physical
  constants, stable route IDs, strict state validation, workflow-planned chunk
  execution, unified output providers, migrated bundled setups/examples, and a
  public-only installed 0.4 plugin.
- VerCOR 0.4.0a1 release verification completed locally (2026-07-15) from
  build HEAD `31e803c06a4e65e8e72ee77937b056eac540eb44`. Black warned Python 3.13
  cannot perform its safety parse for configured Python 3.15, while exit
  remained 0 and 242 files were unchanged; strict flake8 reported 0; mypy
  passed 238 source files; compileall and whitespace checks were clean.
  The fast suite passed 543/543; the full and branch-coverage suites passed
  1139/1139; coverage was 90.51% (7,352 statements and 1,532 branches). The
  full and coverage runs emitted five third-party warning instances: one Flax
  JAX-effect deprecation, one JAX scatter-cast future warning, one JCM/xarray
  merge-default future warning, and two xarray NumPy-shape deprecations. The
  optional JCM/Veros focus passed 9/9 with only the Flax warning; output-free
  JVP/reverse differentiation passed 3/3; supplied-artifact boundaries passed
  16/16. Fresh offline no-isolation builds are in
  `/private/tmp/vercor-0.4.0a1-dist/` with SHA-256 values:
  `vercor-0.4.0a1-py3-none-any.whl`
  `a713f10c3722145d1dd0e0886c266e264d098dc7f30276b99bb027fdc246bff7`;
  `vercor-0.4.0a1.tar.gz`
  `119717648950a04d89fe28a2522a2c6ae5fc699d8725ae0cdc788691c6c529a2`;
  `vercor_public_plugin-0.1.0-py3-none-any.whl`
  `198a7e2d7d4873d3550ff3ffe41aa8b6c41ab38e80347b501e6f04e43766db74`.
  JCM 1.1.1 and Veros 1.6.2 remain the installed optional-model versions.
  Tag, push, publication, and upload remain unperformed.
- Post-review version-policy hardening completed locally (2026-07-15): the
  ownership matrix preserves qualified external/independent identifiers while
  rejecting exact and shorthand VerCOR labels. Policy passed 141/141,
  policy/architecture passed 161/161, and the full suite passed 1261/1261 with
  the same five third-party warning instances. Independent final review found
  no Critical, Important, or Minor issues. Release artifacts and hashes above
  remain unchanged; no tag, push, publication, or upload was performed.

## Implemented 0.4 Architecture

- `vercor.__all__` is exactly `Clock`, `Coupler`, `Exchange`,
  `RectilinearGrid`, `RunState`, and `RuntimeOptions`.
- `Component` is structural. `ComponentSpec` owns fields, lifecycle, execution,
  transfer, and output; `CallableComponent` and `DataComponent` are the only
  convenience adapters.
- `PhysicalConstants` is the frozen traced PyTree; `RuntimeOptions.dtype` is
  the sole precision policy.
- `Coupler(...)` owns complete immutable assembly. Reconfiguration constructs a
  new coupler.
- Exchange and topology identity is the stable route ID. Ambiguous target-field
  fan-in is rejected.
- Workflows produce exact plans; the core owns chunks and validates every
  backend driver call and returned state.
- `RunState` exposes only component views and immutable field replacement.
- One output coordinator owns all enabled provider selection, accumulation,
  cadence, host transfer, paths, period files, final fields, and snapshots.
  `output=None` performs no I/O and remains differentiable.
- JCM, Veros, and CAMulator imports remain lazy. CAMulator is not installed or
  pinned.

## Release Candidate Handoff

- The executable review validates the exact eight sections, all canonical
  public manifests, central/root signatures, all 119 non-public modules,
  runnable README/migration snippets, archive SHA-256, Task 9 absence, and
  release metadata.
- CI encodes Python 3.12/3.13 base/JCM/Veros artifact lanes, Python 3.12/3.13
  native-v0.4 plugin lanes, and a macOS installed-plugin smoke. GitHub-hosted
  jobs have not run locally.
- The Task 10 documentation/release commit is present in repository history; do not tag, push, or publish without separate authority.

## Durable Constraints

- Never add numerical fudge factors; trace discrepancies to their first source.
- Write behavior/contract tests before implementation changes.
- Preserve exact public owner manifests and keep primary 0.4 modules alias-free.
- Keep output opt-in and optional-model imports lazy.
- No registry, entry-point discovery, Pydantic, fan-in reducer, public prepared
  graph, fractional subcycling, or CAMulator dependency pin.
- Do not tag, push, publish, or create a release without separate authority.
- The Conda launcher can panic through `conda-rattler`; use the direct `scipy`
  environment executable when that occurs.

## Validation Policy

Use concise pytest output. Run focused tests while iterating, then Black,
strict flake8, mypy, compileall, fast and full pytest, branch coverage, build,
installed wheel/sdist/plugin smokes, and `git diff --check` before a release
candidate commit. Put detailed command evidence in the active task report and
archive only durable outcomes here.
