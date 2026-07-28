Getting started
===============

Prerequisites
-------------

Use Python 3.12 or 3.13. This first run uses VerCOR's bundled slab ocean, so
it needs no external data or optional model packages.

Installation
------------

Install the release in an isolated Python environment:

.. code-block:: console

   python -m pip install "vercor==0.4.3"

Run your first component
------------------------

Save the following complete program as ``quickstart.py`` and run it with the
same Python environment used for installation:

.. literalinclude:: ../_examples/quickstart.py
   :language: python
   :linenos:

Expected result
---------------

:class:`~vercor.grids.RectilinearGrid` defines the longitude-latitude locations
for a component's fields. :class:`~vercor.Clock` defines the shared start time,
time step, and number of steps. :class:`~vercor.Coupler` assembles the configured components
and advances them in the requested order.

``coupler.run()`` returns an immutable :class:`~vercor.RunState`. Use
``RunState.component(name)`` to select a component and then ``field(name)`` to
read one of its fields. The assertions in the program verify that the slab
ocean's sea-surface-temperature field has the grid shape and finite values.

Next steps
----------

Continue to :doc:`running` to reuse an initial state, choose a component order,
and inspect fields from one or more components. See :doc:`../troubleshooting`
when a configuration or run does not behave as expected.
