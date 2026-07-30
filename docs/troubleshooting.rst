Troubleshooting
===============

An optional setup cannot be imported
------------------------------------

**Symptom.** Importing an optional JCM, Veros, or CAMulator setup fails.

**Cause.** The selected setup needs its optional model package or its required
external data. CAMulator has no documented compatible dependency pin.

**Action.** Install and configure the dependency required by the selected
driver; see :doc:`how-to/examples` for the packaged setup gallery and its data
requirements.

A setup file does not satisfy ``vercor run``
--------------------------------------------

**Symptom.** ``vercor run`` reports that a setup must define ``run_setup`` or
that its return value is invalid.

**Cause.** The local Python file does not define exactly
``run_setup(*, loglevel, float_type)``, or the callable returned a value other
than ``None`` or an integer status.

**Action.** Define the keyword-only callable with exactly those two parameters.
Use the lowercase CLI values for ``loglevel`` (``trace``, ``debug``, ``info``,
``warning``, or ``error``) and ``float_type`` (``float64`` or ``float32``).

Duplicate setup templates are rejected
--------------------------------------

**Symptom.** ``show-setups`` or ``copy-setup`` reports ``duplicate setup``.

**Cause.** A public direct ``.py`` file in a ``VERCOR_SETUP_DIR`` directory has
the same stem as another external template or a packaged gallery template.

**Action.** Rename or remove one template so every catalog name is unique. Set
``VERCOR_SETUP_DIR`` to an ``os.pathsep``-separated list of existing directories;
it does not search nested directories.

An external setup directory is missing
--------------------------------------

**Symptom.** The CLI reports that a setup directory does not exist or is not a
directory.

**Cause.** ``VERCOR_SETUP_DIR`` contains a missing path or a file.

**Action.** Correct each direct directory in the ``os.pathsep``-separated list,
or unset the variable when only packaged templates are needed.

Copying a setup collides with a local file
------------------------------------------

**Symptom.** ``copy-setup`` reports that the destination setup file already
exists.

**Cause.** The target directory already contains a file with the selected
template filename.

**Action.** Choose a different ``--to`` directory or rename the existing local
file. ``copy-setup --to`` creates missing parent directories and reuses an
existing directory, but never overwrites a destination file.

An exchange field is rejected
-----------------------------

**Symptom.** Coupler construction rejects a field or reports ambiguous target
field fan-in.

**Cause.** An exchange field is not declared by both endpoints, or multiple
routes write one target field.

**Action.** Declare the field as a source output and target input. Give routes
between the same component pair distinct IDs when their identities collide,
but do not use distinct IDs to bypass fan-in rejection: only one route may
write a target field. Combine multiple values in an explicit component step;
see :doc:`developers/coupling`.

A host component fails with the JAX backend
--------------------------------------------

**Symptom.** A graph containing a host component is rejected before stepping.

**Cause.** ``RuntimeOptions(backend="jax")`` permits only JAX components.

**Action.** Use ``RuntimeOptions(backend="auto")`` for a mixed graph, or use
``backend="host"``; see :doc:`how-to/backends`.

A compiled payload changes structure
------------------------------------

**Symptom.** A compiled run fails after a component step returns its payload.

**Cause.** JAX compilation requires stable payload PyTree structure, array
shapes, and dtypes from step to step.

**Action.** Keep the payload structure fixed and return replacement values
with unchanged layouts; see :doc:`developers/jax-components`.

Output fails under JIT or differentiation
-----------------------------------------

**Symptom.** A transformed run fails when file output is enabled.

**Cause.** File output is a host-side effect and cannot consume traced runtime
state.

**Action.** Run with ``output=None`` under JIT or differentiation, then write
results in ordinary Python; see :doc:`how-to/output`.
