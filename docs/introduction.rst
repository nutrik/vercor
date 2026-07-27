Introduction
============

What VerCOR is
--------------

VerCOR is a coupler for assembling Earth-system components, such as atmosphere,
ocean, sea-ice, land, and forcing-data models, into one run. A component owns
its model state and fields. Grids describe where those fields live, and
exchanges transfer named fields between component grids.

Why JAX
-------

JAX lets compatible numerical workflows be compiled, vectorized, and
differentiated. VerCOR preserves this capability when every scheduled component
and exchange can run in JAX.

How a run fits together
-----------------------

A clock advances the shared model time and determines when components are due
to run. VerCOR stores the evolving coupled system in an immutable ``RunState``:
each step returns a new state rather than changing the old one. Output is
opt-in, so simulations only write files or diagnostics when an output session
is configured.

Execution and differentiation
------------------------------

Output-free JAX workflows can be differentiated end to end. Host-side
components are supported for models that require ordinary Python or external
libraries; scheduling any host component selects host execution when
``backend="auto"``.
