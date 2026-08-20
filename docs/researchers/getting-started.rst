.. Adapted from https://raw.githubusercontent.com/team-ocean/veros/refs/heads/main/doc/introduction/get-started.rst

Getting started
===============

Installation
------------

Quick installation via pip
++++++++++++++++++++++++++

.. warning::

  Use this installation method when you want to get started quickly and do not
  need to modify VerCOR's source code. For development, use a source
  installation as described below.

Prerequisites
+++++++++++++

Before installing VerCOR, ensure that Python 3.12 or 3.13 is available.

Installation
++++++++++++

The quickest way to get a working VerCOR installation is to run::

  $ pip install vercor

or, optionally ::

  $ pip install "vercor[jcm,veros]"

to install VerCOR with the optional
`JCM <https://jax-gcm.readthedocs.io/en/latest/>`_ and
`Veros <https://veros.readthedocs.io/en/latest/>`_ components.

If you want to install a specific release version execute:

.. code-block:: console

   python -m pip install "vercor==0.4.4"

Using Conda (multi-platform)
++++++++++++++++++++++++++++

1. `Download and install Miniconda <https://docs.conda.io/en/latest/miniconda.html>`__. If you are using Windows, you may use the Anaconda prompt to execute the following steps.

2. Clone the VerCOR repository:

   .. code-block:: console

      git clone https://github.com/nutrik/vercor.git

   If you do not have git installed, you can do so via ``conda install git``.

3. Create a new Conda environment for VerCOR and install its dependencies::

       $ cd vercor/  # if not already in the vercor directory
       $ conda env create -f conda-environment.yml

4. Activate the new environment::

       $ conda activate vercor

Using pip (Linux / OSX)
+++++++++++++++++++++++

1. Clone the repository:

   .. code-block:: console

      git clone https://github.com/nutrik/vercor.git

2. Install VerCOR in editable mode, preferably in a virtual environment::

      $ pip install -e .

   The ``-e`` flag ensures that changes to the code are immediately reflected without reinstalling.

3. Optionally install JCM and Veros::

      $ pip install -e ".[jcm,veros]"

Setting up a coupled run
------------------------

In order to perform your first coupled run, you can copy a pre-configured setup script from the :doc:`../setup-gallery` into a working directory:

.. code-block:: console

   vercor copy-setup run_veros_with_era5data \
     --to ~/vercor-setups/run_veros_with_era5data

Alternatively, you can create your own setup script from scratch. In this case, you need to define the longitude-latitude locations of a component's fields
with :class:`~vercor.grids.RectilinearGrid`; the shared start time, coupler time step, and number of steps with
:class:`~vercor.Clock`; and assemble the configured components and the order in which they are advanced with :class:`~vercor.Coupler`.
All of the above is demonstrated in the following program, which you can copy and save as `quickstart.py`:

.. literalinclude:: ../_examples/quickstart.py
   :language: python
   :linenos:

Running VerCOR
--------------

To run coupled setups, use the ``vercor run`` command from :doc:`../cli` in the
same Python environment used to install VerCOR.
If you choose to run the copied above pre-configured setup, execute:

.. code-block:: console

   vercor run \
     --loglevel info \
     --float-type float64 \
     ~/vercor-setups/run_veros_with_era5data/run_veros_with_era5data.py

For the alternative setup (given above) execute:

.. code-block:: console

   python quickstart.py

Expected result
+++++++++++++++

``coupler.run()`` returns an immutable :class:`~vercor.RunState`. Use
``RunState.component(name)`` to select a component and then ``field(name)`` to
read one of its fields. The assertions in the `quickstart.py` program verify that
the slab ocean's sea-surface-temperature field has the grid shape and finite values.

Next steps
----------

Continue to :doc:`running` from :doc:`../setup-gallery`, reuse an initial state,
choose a component order, and inspect fields from one or more components.
See :doc:`../troubleshooting` when a configuration or run does not behave as expected.
