# Expanded Packaged Setup Gallery CLI Design

## Status and Scope

This specification extends and supersedes the command behavior described in
`2026-07-29-packaged-setup-gallery-cli-design.md`. The packaged gallery remains
the built-in source of setup templates, while the CLI gains setup listing,
external template discovery, explicit copy destinations, version reporting,
and runtime logging and precision controls.

The console interface is:

```text
vercor [OPTIONS] COMMAND [ARGS]...

Commands:
  copy-setup
  show-setups
  run
```

The root help description is `Vercor command-line tools`. The root
`--version` option reports the installed distribution version through package
metadata.

## Setup Discovery

Setup discovery has two source types:

1. Python files directly inside the packaged `vercor.setups.gallery` resource.
2. Python files directly inside directories named by `VERCOR_SETUP_DIR`.

`VERCOR_SETUP_DIR` is an operating-system path list, split with
`os.pathsep` (`:` on Unix and `;` on Windows). Discovery does not recurse.
Only direct `.py` files whose stems do not begin with `_` are public setups;
`__init__.py` is excluded.

Discovery returns setup stems in deterministic sorted order. It retains each
setup's canonical `.py` filename and source so listing and copying share one
implementation. A configured external directory that is missing, is not a
directory, or cannot be read is reported as a concise Click error.

Setup stems must be unique across every source. A duplicate between the
packaged gallery and an external directory, or between two external
directories, is an error. The diagnostic names the duplicate stem and every
conflicting source. There is no implicit override or precedence rule.

## Root and Listing Commands

The root command exposes:

```text
Options:
  --version  Show the version and exit.
  --help     Show this message and exit.
```

`show-setups` prints the sorted available setup stems, one per line. It uses
the same discovery function as `copy-setup`; invalid directories and duplicate
stems therefore behave consistently.

The `copy-setup --help` text includes the available setup stems discovered for
the current environment, an example invocation, and guidance that additional
template directories can be supplied through `VERCOR_SETUP_DIR`.

## Copying a Setup

The command is:

```text
vercor copy-setup [OPTIONS] SETUP

Options:
  --to PATH
```

`SETUP` accepts either a public setup stem or its `.py` filename. Existing
normalization and traversal protections remain: paths, private names, empty
names, bare `.` and `..` markers, and non-Python suffixes are rejected. Catalog
lookup precedes suffix diagnostics so a public dotted stem such as
`model.profile`, as printed by `show-setups`, remains copyable by either
`model.profile` or `model.profile.py`. If one template's canonical filename is
another template's public stem, that shared reference is rejected as ambiguous
instead of selecting either source implicitly.

`--to` names a destination directory and defaults to the current working
directory. If the directory does not exist, the command creates it, including
missing parents. If it already exists, the command reuses it. A path that
exists but is not a directory is rejected.

The destination filename is always the setup's canonical `.py` filename. The
file is created exclusively so an existing destination is never overwritten.
If copying fails after creating a new file, the incomplete file is removed;
the directory itself is retained.

Examples:

```text
vercor copy-setup run_jcm_with_veros
vercor copy-setup run_jcm_with_veros \
  --to ~/vercor-setups/run_jcm_with_veros
```

## Runnable Setup Contract

Every setup executed by `vercor run` must expose:

```python
def run_setup(*, loglevel: str, float_type: str) -> int | None:
    ...
```

All bundled gallery setups implement this contract. A copied setup preserves
the same contract and remains directly editable by the user. External setup
files may implement it without being registered or imported into the VerCOR
package.

The function explicitly owns runtime configuration:

- it passes `loglevel` to each top-level `Coupler` it constructs; and
- it constructs `DTypePolicy(enable_x64=float_type == "float64")` and passes
  the policy through `RuntimeOptions`.

This explicit boundary avoids environment-driven overrides and hidden changes
to core runtime defaults.

## Running a Setup

The command is:

```text
vercor run [OPTIONS] SETUP_FILE

Options:
  -v, --loglevel [trace|debug|info|warning|error]
  --float-type [float64|float32]
```

The defaults are `info` and `float64`. Click validates the choices and the
existing local `.py` file.

The CLI resolves the private runner file beside the already imported
`vercor.cli` module and executes that file with `sys.executable -P` in a child
process. It passes the resolved setup path and selected values as separate
arguments without a shell, so a local `vercor.py` or `vercor/` cannot replace
the trusted runner. The child loads the file without triggering its
`if __name__ == "__main__"` block, validates that `run_setup` is callable, and
invokes it with keyword arguments.

The setup file's directory remains on `sys.path` throughout loading and
invocation, matching ordinary script execution even when `run_setup` performs
a lazy adjacent import. The original path is restored in `finally`. Missing
contracts and invalid return values produce concise runner diagnostics. `None`
means success; an integer return value becomes the process exit status.
Unhandled setup exceptions remain visible and cause a nonzero exit. The outer
Click command returns the child's status unchanged.

Loading and invocation stay in the child so setup globals, optional
dependencies, JAX initialization, and failures do not contaminate the CLI
process.

## Logging and Precision

VerCOR's existing log-level normalization recognizes `trace` as numeric level
5 in addition to Python's standard `debug`, `info`, `warning`, and `error`
levels. Trace therefore enables every currently implemented VerCOR log
message, including debug messages, without requiring a new logger protocol
method.

`float64` maps to `DTypePolicy(enable_x64=True)` and `float32` maps to
`DTypePolicy(enable_x64=False)`. Each setup supplies the resulting policy to
its runtime options instead of relying on ambient JAX configuration.

## Module Boundaries

`vercor.cli` owns Click presentation, input validation, shared discovery,
duplicate checks, and copying. One private child-runner module owns setup-file
loading, contract validation, and invocation. The gallery scripts own their
model-specific construction and translate the two contract values into their
`Coupler` and `RuntimeOptions`.

Discovery is centralized so `show-setups`, copy help, and `copy-setup` cannot
develop different setup catalogs. The private runner is packaged with the
distribution but is not added to VerCOR's public API.

No physics, model, workflow, or numerical contracts change. The only core
logging extension is acceptance of the `trace` name.

## Errors

Click usage errors cover malformed names, invalid choices, missing run files,
directories supplied as files, non-Python run files, and invalid `--to`
targets. Command errors cover unreadable setup sources, missing external
directories, duplicate setup names, unknown setups, copy collisions, and
filesystem failures.

The private runner reports a missing or non-callable `run_setup`, an invalid
return value, and loading failures without masking setup tracebacks.

All errors return a nonzero status and preserve existing destination files.

## Test-Driven Implementation

Behavioral tests are written before implementation and cover:

- root description, version output, commands, and option help;
- deterministic bundled and external setup listing;
- platform-native multiple-directory parsing;
- invalid external directories and duplicate-name diagnostics;
- dynamic available-setup help;
- copying by stem and filename;
- default, newly created, and existing `--to` directories;
- rejection of non-directory targets and preservation of existing files;
- child-process isolation and use of the active interpreter;
- setup-contract loading and local imports;
- explicit log-level and float-type propagation;
- default option values and every accepted choice;
- missing/non-callable contracts, exceptions, and return statuses;
- contract presence in every bundled setup;
- wheel and source-distribution inclusion of the gallery and private runner; and
- a real installed `vercor` console workflow covering version, listing,
  destination copying, and lightweight external setup execution.

Documentation tests and architecture checks are updated with the new commands
and private boundary. `README.md`, user documentation, `DESIGN.md`,
`DEPENDENCIES.md`, and `PROGRESS.md` describe the implemented behavior.

## Verification and Delivery

Development occurs only in the isolated
`feat/packaged-setup-gallery-cli` worktree and uses small passing commits.
Before delivery, verification includes focused tests, Black, strict flake8,
mypy, compile checks, the fast suite, the full suite, branch coverage,
wheel/source-distribution builds, installed-artifact probes, and
`git diff --check`.

The verified branch is pushed to `feat/packaged-setup-gallery-cli`, and the
existing draft pull request is updated rather than replaced.
