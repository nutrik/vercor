Running a coupled configuration
===============================

Assemble once
-------------

Create a ``Coupler`` with its clock, components, exchanges, runtime policy,
and component order before running it. A coupler owns that complete
configuration. Create a new ``Coupler`` to change configured components,
exchanges, runtime policy, or run order.

Choose component order
----------------------

The ``run_order`` passed to ``Coupler`` names the components that run at each
clock step, in order. Put a component after the components whose received
fields it needs for that step. The quick start has one ocean component, so its
order contains only the ocean name; a multi-component configuration uses each
configured name once in its intended sequence.

Run from a new or existing state
--------------------------------

Ask the coupler for its initialized state when you want to inspect or retain
the exact state used as the run input. Pass that state to ``run`` to advance it;
the returned state is a new immutable ``RunState``.

.. code-block:: python

   initial_state = coupler.initial_state()
   final_state = coupler.run(initial_state)

Calling ``coupler.run()`` without an argument also creates and advances a new
initial state. To continue a simulation, pass the prior ``RunState`` to the
same configured coupler.

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
