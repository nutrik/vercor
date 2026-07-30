# VerCOR

<p align="center">
  <a href="http://vercor.readthedocs.io/?badge=latest">
    <img src="https://readthedocs.org/projects/vercor/badge/?version=latest" alt="Documentation status">
  </a>
  <a href="https://github.com/nutrik/vercor/actions/workflows/python-package.yml">
    <img src="https://github.com/nutrik/vercor/actions/workflows/python-package.yml/badge.svg" alt="Test status">
  </a>
  <a href="https://codecov.io/github/nutrik/vercor" > 
    <img src="https://codecov.io/github/nutrik/vercor/graph/badge.svg"/> 
  </a>
</p>

Versatile Earth system COupleR (VerCOR) connects atmosphere, ocean, sea-ice,
land, and forcing-data components on a shared clock. It helps Earth-system
researchers compose models, exchange fields between grids, and collect
diagnostics and output.

VerCOR is built on [JAX](https://github.com/jax-ml/jax). Output-free JAX
workflows remain differentiable end to end, supporting sensitivity analysis,
automatic differentiation, and gradient-based experiments.

> **Stable release:** Version `0.4.3` is the current release. VerCOR 0.3
> applications should follow the
> [Migration guide](https://vercor.readthedocs.io/en/latest/migration-0.3-to-0.4.html).

## Key capabilities

- Combine JAX-native and host-side models in one coupled simulation.
- Exchange scalar and vector fields between rectilinear grids.
- Use bilinear or conservative regridding.
- Run Gregorian, no-leap, or 360-day calendars.
- Use bundled slab atmosphere, ocean, land, and sea-ice components.
- Connect optional JCM, Veros, and CAMulator models.
- Supply static or time-dependent forcing data.
- Preserve immutable state across JAX transformations.
- Enable period averages, final fields, and snapshots when requested.

## Requirements and installation

VerCOR supports Python 3.12 and 3.13 and requires a JAX installation supported
by your platform.

Install the core package:

```bash
python -m pip install "vercor==0.4.3"
```

Install an optional extra before using a bundled JCM or Veros setup:

```bash
python -m pip install "vercor[jcm]"
python -m pip install "vercor[veros]"
```

CAMulator additionally requires NCAR's
[MILES-CREDIT](https://github.com/NCAR/miles-credit). A compatible CREDIT
release has not yet been confirmed or pinned.

Version `0.4.3` is the current release. Upgrade an older installation for the
published stable API:

```bash
python -m pip install --upgrade "vercor==0.4.3"
```

Features explicitly marked development-only require a current source checkout
and are not part of `0.4.3`.

## Quick start

This dependency-light example runs a slab ocean for two one-hour steps:

```python
from datetime import datetime

import jax.numpy as jnp

from vercor import Clock, Coupler, RectilinearGrid
from vercor.setups import make_slab_ocean


grid = RectilinearGrid.uniform(
    "quickstart",
    nlon=2,
    nlat=2,
    longitude=(0.0, 360.0),
    latitude=(-90.0, 90.0),
)
clock = Clock(datetime(2000, 1, 1), dt_seconds=3600.0, steps=2)
ocean = make_slab_ocean(grid)
coupler = Coupler(clock, components=(ocean,), run_order=(ocean.name,))

final_state = coupler.run()
sea_surface_temperature = final_state.component(ocean.name).field(
    "sea_surface_temperature"
)

assert sea_surface_temperature.shape == grid.shape
assert bool(jnp.all(jnp.isfinite(sea_surface_temperature)))
```

## Setup gallery

> **Development-only:** The setup-gallery CLI is currently unreleased and is
> not included in the PyPI `0.4.3` distribution. The commands below apply to a
> current development-source installation until a later release publishes
> them.

The development version packages runnable setup scripts with its distribution.
List the available templates, copy one into a user-editable directory, then run
that local copy:

```console
vercor show-setups
vercor copy-setup run_jcm_with_veros \
  --to ~/vercor-setups/run_jcm_with_veros
vercor run \
  --loglevel info \
  --float-type float64 \
  ~/vercor-setups/run_jcm_with_veros/run_jcm_with_veros.py
```

`vercor --version` reports the installed distribution version. `VERCOR_SETUP_DIR`
may add external templates as an `os.pathsep`-separated list of direct
directories. Every template name must be unique across the packaged gallery and
those directories; duplicates are errors. `copy-setup --to` creates missing
parent directories or reuses an existing directory, but never overwrites an
existing setup file.

An external template must define exactly
`run_setup(*, loglevel, float_type)`. `vercor run` passes lowercase-only
`loglevel` choices (`trace`, `debug`, `info`, `warning`, or `error`) and
`float_type` choices (`float64` or `float32`); their defaults are `info` and
`float64`. The function returns `None` for success or an integer process status.

## Documentation

Read the canonical documentation at
[https://vercor.readthedocs.io/](https://vercor.readthedocs.io/):

- [Researcher guide](https://vercor.readthedocs.io/en/latest/researchers/)
- [Developer guide](https://vercor.readthedocs.io/en/latest/developers/)
- [Python API](https://vercor.readthedocs.io/en/latest/api/)
- [Migration guide](https://vercor.readthedocs.io/en/latest/migration-0.3-to-0.4.html)
- [Plugin authoring](https://vercor.readthedocs.io/en/latest/plugin-authoring.html)

Repository resources:

- [Changelog](CHANGELOG.md)
- [License](LICENSE)
