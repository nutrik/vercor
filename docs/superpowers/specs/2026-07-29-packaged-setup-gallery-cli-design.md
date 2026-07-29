# Packaged Setup Gallery CLI Design

## Goal

Ship VerCOR's runnable setup scripts inside the installed package and provide a
small Click command-line interface that lets users copy a selected setup into
their working directory and run a local setup file.

## User Interface

The package installs one console command:

```text
vercor copy-setup run_jcm_with_veros
vercor run run_jcm_with_veros.py
```

`copy-setup` accepts either a setup stem or its `.py` filename. It copies the
matching bundled file from `vercor.setups.gallery` to the current working
directory. The destination always uses the setup's canonical `.py` filename.
If that destination already exists, the command fails without changing it.

`run` accepts one local `.py` file. It starts that file as a child process with
the same Python interpreter that is running the VerCOR CLI and returns the
child's exit status. Script output and errors remain connected to the invoking
terminal.

Listing setups, copying to a custom destination, overwriting files, forwarding
additional script arguments, and running a bundled setup without first copying
it are outside this change.

## Package Layout

All files currently in `examples/` move to:

```text
vercor/setups/gallery/
```

The existing `examples/__init__.py` becomes the gallery package initializer,
and the top-level `examples/` directory is removed. The scripts remain ordinary
runnable Python files and keep their current setup logic.

`vercor/cli.py` owns the Click group and both commands. It locates gallery files
with `importlib.resources.files("vercor.setups.gallery")`, so source checkouts,
wheels, and source distributions use the same lookup. The gallery remains the
single source of setup contents; the CLI does not embed a duplicate name or
source registry.

`pyproject.toml` adds Click as a runtime dependency and registers
`vercor.cli:cli` as the `vercor` project script. The root `vercor.__all__`
contract remains unchanged.

## Command Boundaries and Data Flow

For `copy-setup`:

1. Normalize a bare stem to `<stem>.py`.
2. Reject empty names, path separators, parent traversal, and non-`.py` names.
3. Resolve the normalized name directly below the packaged gallery.
4. Reject a name that does not identify a bundled setup file.
5. Create the destination in the current directory exclusively.
6. Stream the packaged bytes to the new destination and report its path.

Exclusive destination creation makes collision handling authoritative even if
the filesystem changes between validation and copying. If copying fails after
the CLI creates the destination, the incomplete new file is removed; a
pre-existing path is never removed or modified.

For `run`:

1. Let Click validate that the supplied path is an existing local file.
2. Reject files whose suffix is not `.py`.
3. invoke `[sys.executable, str(path)]` without a shell.
4. Return the child process status unchanged.

The shell is not involved, so filenames are passed as data and cannot inject
shell syntax. Running a setup is intentionally isolated in a child interpreter,
matching direct `python setup.py` behavior and keeping script globals and
optional dependency failures out of the CLI process.

## Errors

All user input failures are concise Click usage errors with nonzero status:

- invalid or nested setup names;
- unknown gallery setup names;
- an existing copy destination;
- a missing run target;
- a directory or non-Python run target; and
- filesystem failures while reading or creating a copy.

The CLI does not reinterpret exceptions from a setup process. Its stdout,
stderr, and exit status remain observable at the command line.

## Testing

Implementation follows red-green-refactor cycles. Behavioral tests use Click's
test runner and real temporary files to cover:

- copying a setup by stem;
- copying a setup by `.py` filename;
- copied bytes matching the packaged resource;
- preserving an existing destination and its contents;
- rejecting traversal, nested, unknown, and non-Python setup names;
- executing a local Python file with the current interpreter;
- propagating a nonzero script exit status;
- rejecting missing, directory, and non-Python run targets; and
- displaying the two commands in CLI help.

Distribution tests build the wheel and source distribution and verify that the
console entry point and every gallery script are present and usable from an
installed artifact. Existing tests that import the injectable JCM example move
to its `vercor.setups.gallery` module path.

## Documentation and Verification

The README and Sphinx example guide describe the copy-and-run workflow and stop
directing users to the removed top-level directory. Active release commands,
architecture contracts, and tests replace `examples` paths with
`vercor/setups/gallery`. `DESIGN.md` records the packaged gallery and CLI,
`DEPENDENCIES.md` places the dependency-light CLI after its gallery resource,
and `PROGRESS.md` records the completed behavior and exact validation results.

Verification includes focused CLI, example, architecture, documentation, and
distribution tests; Black; strict flake8; mypy; compileall; fast and full
pytest; branch coverage; wheel/source-distribution builds and installed
artifact probes; and `git diff --check`.
