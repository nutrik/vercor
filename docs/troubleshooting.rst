Troubleshooting
===============

An optional setup cannot be imported
------------------------------------

**Symptom.** Importing an optional JCM, Veros, or CAMulator setup fails.

**Cause.** The selected setup needs its optional model package or its required
external data. CAMulator has no documented compatible dependency pin.

**Action.** Install and configure the dependency required by the selected
driver; see :doc:`how-to/examples` for the repository examples and their data
requirements.

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
