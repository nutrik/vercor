.. Adapted from https://github.com/team-ocean/veros/blob/main/doc/reference/cli.rst

Command line tools
==================

After installing VerCOR, you can call these commands from any location.

All CLI options
---------------

The ``vercor`` entry point provides access to all VerCOR command-line tools.

.. run-click:: vercor.cli:cli
   :args: --help

Show preconfigured setups
-------------------------

.. run-click:: vercor.cli:cli
   :args: show-setups --help

Copy a selected setup
---------------------

.. run-click:: vercor.cli:cli
   :args: copy-setup --help

Run a selected/copied setup
---------------------------

.. run-click:: vercor.cli:cli
   :args: run --help
