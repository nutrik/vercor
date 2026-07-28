Running a coupled configuration
===============================

Assemble once
-------------

Create a :class:`~vercor.Coupler` with its clock, components, exchanges,
runtime policy, and component order before running it. A coupler owns that
complete configuration. Create a new :class:`~vercor.Coupler` to change
configured components, exchanges, runtime policy, or run order.

Choose component order
----------------------

For the default sequential workflow, the ``run_order`` passed to
:class:`~vercor.Coupler` is the unique subset of configured components scheduled
at each clock step, in order. Put a component after the components whose
received fields it needs for that step. The quick start schedules only the ocean
name. An empty order is valid, and a custom workflow may schedule registered
components differently.

Run from a new or existing state
--------------------------------

Ask the coupler for its initialized state when you want to inspect or retain
the exact state used as the run input. Pass that state to ``run`` to reuse its
fields and payload; the returned state is a new immutable
:class:`~vercor.RunState`.

.. code-block:: python

   initial_state = coupler.initial_state()
   final_state = coupler.run(initial_state)

Calling ``coupler.run()`` without an argument also creates and advances a new
initial state. A :class:`~vercor.RunState` does not store a clock cursor,
so every call to ``run`` replays the configured :class:`~vercor.Clock` window.
Passing a prior state changes the input fields and payload, not the run's
start time or step indices.

Inspect component fields
------------------------

Select one component by name, then read a declared field from its component
view. ``components()`` returns the views for every configured component.

.. code-block:: python

   ocean_state = final_state.component("OCN")
   all_components = final_state.components()

For example, ``ocean_state.field("sea_surface_temperature")`` returns the
ocean sea-surface-temperature array. ``all_components`` is useful when a
diagnostic needs fields from several components.
