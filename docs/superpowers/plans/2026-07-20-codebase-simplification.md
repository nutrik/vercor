# VerCOR Codebase Simplification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove approved private over-engineering and reduce bilinear/Veros runtime overhead without changing VerCOR's public API, numerical behavior, JAX transforms, output formats, or plugin contracts.

**Architecture:** Keep the public protocol-first design and immutable prepared runtime boundary. Simplify only private owners: delete unread state, put shared behavior on existing cohesive classes/helpers, return topology maps directly, remove one-use modules, shrink interpolator PyTrees, and copy Veros state once per forcing update.

**Tech Stack:** Python 3.13, JAX, NumPy, pytest/pytest-xdist, mypy, Black, flake8, h5netcdf/xarray, optional Veros adapters.

## Global Constraints

- Preserve the six-symbol package root and every canonical public owner, signature, protocol, and output filename format.
- Preserve exact runtime validation coverage, exception behavior, immutable PyTrees, traced physics values, and runtime dtype ownership.
- Preserve lazy optional-framework imports and the installed public-plugin contract.
- Do not change public physics parameters, public catalogs, logging exports, `RectilinearGrid.from_coordinates()`, seeded topology maps, or validation placement.
- Write or update the focused test first, observe RED, implement the smallest change, observe GREEN, then commit that unit.
- Use `/Users/romannuterman/miniforge3/envs/scipy/bin/python -m pytest` for tests in the `scipy` environment.
- Do not commit a unit until its focused tests and `pytest tests/ -q --fast` pass.

## File Structure

- `vercor/grids.py`: sole concrete public grid and its validation/display behavior.
- `vercor/_regridders/base.py`: shared scalar transfer behavior; concrete modules construct interpolators.
- `vercor/components/*`, `vercor/setups/_data/*`, `vercor/_runtime/stores.py`, `vercor/fluxes/utilities.py`: dead-state cleanup.
- `vercor/output/_dataset.py`: shared output-dimension inference; `_session.py` owns immutable accumulation.
- `vercor/_runtime/topology*.py`, `surface_masks.py`, `initialization.py`, `prepared.py`: direct immutable topology-map flow.
- `vercor/_runtime/facade.py`: constructs run context and coordinates execution directly; `runner.py` is removed.
- `vercor/_interpolators/bilinear_rectilinear.py`: only operational interpolation leaves and static metadata.
- `vercor/setups/_external/veros_state.py`: one copy/unlock for a four-field forcing update.
- Existing focused tests remain with their subsystem; private layout assertions are replaced with behavioral invariants.

---

### Task 1: Collapse the private grid hierarchy and shared scalar regridding

**Files:**
- Modify: `tests/test_helpers_coverage.py`
- Modify: `tests/test_bilinear_rectilinear_regridder.py`
- Modify: `tests/test_conservative_rectilinear_regridder.py`
- Modify: `vercor/grids.py`
- Modify: `vercor/_regridders/base.py`
- Modify: `vercor/_regridders/bilinear.py`
- Modify: `vercor/_regridders/conservative.py`

**Interfaces:**
- Consumes: existing `RectilinearGrid`, `_BaseRegridder`, and interpolator `apply_scalar(field)` methods.
- Produces: unchanged public grid and regridder behavior, with `_BaseRegridder.regrid(field: Any) -> Any` as the single scalar implementation.

- [ ] **Step 1: Write the failing ownership tests**

Replace the `_Grid` fixture in `tests/test_helpers_coverage.py` with direct `RectilinearGrid` assertions and add this ownership assertion:

```python
def test_rectilinear_grid_owns_grid_behavior_without_private_base() -> None:
    import vercor.grids as grids

    grid = RectilinearGrid(
        name="rect",
        longitude=np.asarray([0.0, 120.0, 240.0]),
        latitude=np.asarray([-45.0, 45.0]),
        binary_mask=np.ones((2, 3)),
    )

    assert not hasattr(grids, "_Grid")
    assert "Grid name:  rect" in str(grid)
    assert "shape=(2, 3)" in repr(grid)
    with pytest.raises(GridError, match="Mask must be a 2D array"):
        RectilinearGrid(
            name="bad-mask",
            longitude=np.asarray([0.0, 120.0, 240.0]),
            latitude=np.asarray([-45.0, 45.0]),
            binary_mask=np.ones((2, 3, 1)),
        )
```

Add to both regridder test files an assertion that scalar dispatch is inherited:

```python
from vercor._regridders.base import _BaseRegridder


def test_scalar_regrid_is_owned_by_shared_private_base() -> None:
    assert "regrid" not in BilinearRectilinearRegridder.__dict__
    assert BilinearRectilinearRegridder.regrid is _BaseRegridder.regrid
```

The conservative-file version is:

```python
def test_scalar_regrid_is_owned_by_shared_private_base() -> None:
    assert "regrid" not in ConservativeRectilinearRegridder.__dict__
    assert ConservativeRectilinearRegridder.regrid is _BaseRegridder.regrid
```

- [ ] **Step 2: Run tests to verify RED**

Run:

```bash
/Users/romannuterman/miniforge3/envs/scipy/bin/python -m pytest \
  tests/test_helpers_coverage.py \
  tests/test_bilinear_rectilinear_regridder.py \
  tests/test_conservative_rectilinear_regridder.py -q --tb=short
```

Expected: failures because `_Grid` still exists and both subclasses still define `regrid`.

- [ ] **Step 3: Move behavior to the concrete grid and shared regridder**

In `vercor/grids.py`, remove `abc` and `_Grid`, inherit only from `_PyTreeNodeMixin`, validate the mask in `RectilinearGrid.__init__`, and keep the existing display output:

```python
@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True, init=False, repr=False, kw_only=True)
class RectilinearGrid(_PyTreeNodeMixin):
    pytree_children = (
        "longitude",
        "latitude",
        "longitude_edges",
        "latitude_edges",
        "binary_mask",
    )
    pytree_aux_data = ("name",)

    name: str
    binary_mask: _RuntimeArray | None
    longitude: _RuntimeArray
    latitude: _RuntimeArray
    longitude_edges: _RuntimeArray | None
    latitude_edges: _RuntimeArray | None
```

After creating `binary_mask_array` in the existing constructor, replace the
base-class post-init call with:

```python
    if binary_mask_array is not None and binary_mask_array.ndim != 2:
        raise _GridError("Mask must be a 2D array.")
    object.__setattr__(self, "name", name)
    object.__setattr__(self, "binary_mask", binary_mask_array)
```

Add the existing display behavior directly to `RectilinearGrid`:

```python

    def __str__(self) -> str:
        return (
            f"{self.__class__.__name__}:\n"
            f"├── Grid name:  {self.name}\n"
            f"├── Grid shape: {self.shape}\n"
            f"└── Binary mask: "
            f"{'Provided' if self.binary_mask is not None else 'Not provided'}\n"
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name}, shape={self.shape})"
```

In `_regridders/base.py`, import `RegridderError` and replace the current base method:

```python
def regrid(self, field: Any) -> Any:
    """Transfer one scalar field from the source grid to the target grid."""
    if self.has_identical_grids:
        return field
    if self.interpolator is None:
        raise RegridderError("Regridder not properly set up")
    return self.interpolator.apply_scalar(field)
```

Delete the identical `regrid` methods and now-unused `RegridderError` imports from both concrete modules. Retain `BilinearRectilinearRegridder.regrid_vector` unchanged.

- [ ] **Step 4: Run focused tests and fast suite**

Run the Step 2 command, then:

```bash
/Users/romannuterman/miniforge3/envs/scipy/bin/python -m pytest tests/ -q --fast --tb=short
```

Expected: all focused tests and all fast tests pass.

- [ ] **Step 5: Commit**

```bash
git add vercor/grids.py vercor/_regridders tests/test_helpers_coverage.py \
  tests/test_bilinear_rectilinear_regridder.py \
  tests/test_conservative_rectilinear_regridder.py
git commit -m "refactor: simplify grid and scalar regridding"
```

### Task 2: Remove unread component, data, store, and flux state

**Files:**
- Modify: `tests/test_component_base_coverage.py`
- Modify: `tests/test_component_models_coverage.py`
- Modify: `tests/test_fluxes_utilities.py`
- Modify: `tests/test_runtime_state.py`
- Modify: `vercor/components/base.py`
- Modify: `vercor/components/_adapter.py`
- Modify: `vercor/setups/_data/_component_helpers.py`
- Modify: `vercor/setups/_data/era5_atmosphere.py`
- Modify: `vercor/setups/_data/era5_land.py`
- Modify: `vercor/setups/_data/era5_ocean.py`
- Modify: `vercor/setups/_data/erainterim_ocean.py`
- Modify: `vercor/_runtime/stores.py`
- Modify: `vercor/fluxes/utilities.py`

**Interfaces:**
- Consumes: normalized component step callable, `DataComponent.spec`, and existing `FieldStore.get/set/replace` methods.
- Produces: the same component and data behavior without `_author_step`, `_data_files`, `_hybrid_coefficients`, `get_or_zeros_like`, or a duplicate virtual-temperature kernel.

- [ ] **Step 1: Write failing minimal-state tests**

Add to `tests/test_component_base_coverage.py`:

```python
from vercor.components import StepContext


def test_callable_component_preparation_retains_only_normalized_step() -> None:
    component = CallableComponent("MODEL", make_test_grid(), lambda fields: {})
    prepared = _coupler(component)._ensure_prepared().components["MODEL"]

    assert not hasattr(component, "_author_step")
    assert not hasattr(prepared, "_author_step")
    assert prepared.step(
        {},
        StepContext(dt_seconds=60.0, time=datetime(2000, 1, 1)),
    ) == {}
```

Replace the four `_data_files` value assertions in
`tests/test_component_models_coverage.py` with minimal-state assertions; retain
their immediately following positive component-contract assertions. The ERA5
atmosphere case also verifies the unread hybrid-coefficient cache is absent:

```python
assert not hasattr(component, "_data_files")
assert not hasattr(component, "_hybrid_coefficients")  # ERA5 atmosphere only
assert component.spec.transfer.time_selection == "linear"
```

Add to `tests/test_fluxes_utilities.py`:

```python
import vercor.fluxes.utilities as flux_utilities_module


def test_flux_utilities_do_not_duplicate_virtual_temperature_kernel() -> None:
    assert not hasattr(
        flux_utilities_module,
        "_virtual_temperature_from_specific_humidity",
    )
```

In `test_runtime_field_store_exposes_mapping_membership_without_default_fallbacks`,
delete both `get_or_zeros_like` assertions and replace the existing weak
`get_or` assertion with:

```python
assert not hasattr(store, "get_or_zeros_like")
```

Rename `test_runtime_field_store_new_helpers_are_jit_compatible` to
`test_runtime_field_store_replace_helpers_are_jit_compatible` and use the
ordinary JAX operation at its real call site:

```python
def update(value: FieldStore) -> FieldStore:
    return value.replace_many(
        {
            "temperature": value.get("temperature") + 2.0,
            "humidity": jnp.zeros_like(value.get("temperature")) + 0.25,
        }
    )
```

- [ ] **Step 2: Run tests to verify RED**

```bash
/Users/romannuterman/miniforge3/envs/scipy/bin/python -m pytest \
  tests/test_component_base_coverage.py \
  tests/test_component_models_coverage.py \
  tests/test_fluxes_utilities.py \
  tests/test_runtime_state.py -q --tb=short
```

Expected: minimal-state assertions fail while the old fields/method remain.

- [ ] **Step 3: Delete unread state and propagation**

Apply these exact structural changes:

```python
# components/base.py
self._normalized_step: _ComponentStepCallable = normalize_component_step_callable(step)

# components/_adapter.py
@dataclass(frozen=True)
class _ComponentDeclaration:
    component: Component
    name: str
    grid: RectilinearGrid
    spec: ComponentSpec
    step: _ComponentStepCallable

@dataclass(frozen=True)
class _ComponentBinding:
    _component: Component
    name: str
    grid: RectilinearGrid
    spec: ComponentSpec
    _step: _ComponentStepCallable
    _data: Mapping[str, RuntimeArray]
    _payload: Any | None
    _dtype_policy: DTypePolicy
```

Remove the `author_step=` assignment from `normalize_component` and `_author_step=` from `prepare_component`.

Remove `data_files` and `cast` from `time_interpolated_data_component`, remove `data_files=data_files` from its four call sites, and delete the `_hybrid_coefficients` assignment. Delete `FieldStore.get_or_zeros_like` and the unused helper in `fluxes/utilities.py`; clean now-unused imports.

- [ ] **Step 4: Run focused tests and fast suite**

Run the Step 2 command and then the fast-suite command from Task 1. Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add vercor/components vercor/setups/_data vercor/_runtime/stores.py \
  vercor/fluxes/utilities.py tests/test_component_base_coverage.py \
  tests/test_component_models_coverage.py tests/test_fluxes_utilities.py \
  tests/test_runtime_state.py
git commit -m "refactor: remove unread component and data state"
```

### Task 3: Centralize output dimensions and simplify immutable reconstruction

**Files:**
- Modify: `tests/test_period_averages.py`
- Modify: `tests/test_v0_4_output_providers.py`
- Modify: `vercor/output/_dataset.py`
- Modify: `vercor/output/_runtime.py`
- Modify: `vercor/output/_session.py`

**Interfaces:**
- Produces: `grid_field_dims(name: str, shape: tuple[int, ...], grid_shape: tuple[int, int] | None) -> tuple[str, ...]`.
- Consumers: `_RuntimeFieldProvider.sample` and `_runtime_output_variable`.

- [ ] **Step 1: Write failing shared-layout and PyTree tests**

Add to `tests/test_period_averages.py`:

```python
from vercor.output._dataset import grid_field_dims


@pytest.mark.parametrize(
    ("shape", "grid_shape", "expected"),
    [
        ((2, 3), (2, 3), ("nlat", "nlon")),
        ((4, 2, 3), (2, 3), ("temperature_dim_0", "nlat", "nlon")),
        ((4,), (2, 3), ("temperature_dim_0",)),
        ((2, 3), None, ("temperature_dim_0", "temperature_dim_1")),
    ],
)
def test_grid_field_dims_is_the_single_output_layout_rule(
    shape: tuple[int, ...],
    grid_shape: tuple[int, int] | None,
    expected: tuple[str, ...],
) -> None:
    assert grid_field_dims("temperature", shape, grid_shape) == expected


def test_output_accumulator_replace_preserves_pytree_structure() -> None:
    empty = _OutputAccumulator.zeros_from_frame(_frame(jnp.asarray([0.0, 0.0])))
    updated = jax.jit(
        lambda accumulator: accumulator.add_frame(_frame(jnp.asarray([1.0, 3.0])))
    )(empty)
    reset = jax.jit(lambda accumulator: accumulator.reset())(updated)

    assert jax.tree_util.tree_structure(empty) == jax.tree_util.tree_structure(updated)
    assert jax.tree_util.tree_structure(updated) == jax.tree_util.tree_structure(reset)
    assert_allclose_compact(reset.counts[0], [0, 0])
```

Add an assertion to the existing output-plan test that `_OutputSchema` has exactly `component`, `provider`, and `period` dataclass fields.

- [ ] **Step 2: Run tests to verify RED**

```bash
/Users/romannuterman/miniforge3/envs/scipy/bin/python -m pytest \
  tests/test_period_averages.py tests/test_v0_4_output_providers.py -q --tb=short
```

Expected: import failure for `grid_field_dims` and schema-field mismatch.

- [ ] **Step 3: Implement one dimension rule and `replace` updates**

Add to `_dataset.py` and export privately only through direct internal import:

```python
def grid_field_dims(
    name: str,
    shape: tuple[int, ...],
    grid_shape: tuple[int, int] | None,
) -> tuple[str, ...]:
    """Return stable generic dimensions for one optional grid-shaped field."""
    if grid_shape is not None and len(shape) >= 2 and shape[-2:] == grid_shape:
        prefix = tuple(f"{name}_dim_{index}" for index in range(len(shape) - 2))
        return (*prefix, "nlat", "nlon")
    return tuple(f"{name}_dim_{index}" for index in range(len(shape)))
```

Use it in both output modules. Delete `_generic_field_dims`, delete `_OutputSchema.index`, and construct `_OutputSchema(component, provider, period)`.

Import `replace` from `dataclasses` and reduce accumulator reconstruction to:

```python
return replace(
    self,
    sum_values=tuple(sums),
    counts=tuple(counts),
    coordinate_shapes=_coordinate_shapes(coordinate_values),
    coordinate_dtypes=_coordinate_dtypes(coordinate_values),
    coordinate_values=coordinate_values,
)
```

and:

```python
return replace(
    self,
    sum_values=tuple(jnp.zeros_like(value) for value in self.sum_values),
    counts=tuple(jnp.zeros_like(value) for value in self.counts),
)
```

If `replace` changes the JIT PyTree structure, revert only the `replace` part, retain the dimension/schema cleanup, and record the JAX reason in `PROGRESS.md`.

- [ ] **Step 4: Run focused tests and fast suite**

Run the Step 2 command and the fast suite. Expected: all pass, including JIT accumulator tests.

- [ ] **Step 5: Commit**

```bash
git add vercor/output tests/test_period_averages.py tests/test_v0_4_output_providers.py
git commit -m "refactor: simplify output schema plumbing"
```

### Task 4: Return runtime topology maps directly and inline role lookup

**Files:**
- Delete: `vercor/_runtime/component_topology.py`
- Modify: `vercor/_runtime/topology_state.py`
- Modify: `vercor/_runtime/topology.py`
- Modify: `vercor/_runtime/initialization.py`
- Modify: `vercor/_runtime/prepared.py`
- Modify: `vercor/_runtime/surface_masks.py`
- Modify: `tests/test_coupler_coverage.py`
- Modify: `tests/test_runtime_state.py`
- Modify: `tests/test_runtime_facade_boundaries.py`
- Modify: `tests/test_tools_components_and_plotting.py`

**Interfaces:**
- Produces: `build_exchange_topology` returning `RuntimeTopologyMaps`.
- Preserves: frozen mapping proxies and the surface-role error `Surface mask policy requires role component {role_name!r} to be registered`.

- [ ] **Step 1: Change tests to the direct-map behavior**

Update the topology result tests to use:

```python
topology_maps = build_exchange_topology(
    components=cast(Any, components),
    exchanges=(exchange,),
    dtype=DTypePolicy(),
    topology_policy=SurfaceMaskPolicy(),
    logger=cast(Any, _RecordingLogger()),
)
assert isinstance(topology_maps, RuntimeTopologyMaps)
assert set(topology_maps.regridders) == {"OCN->ATM"}
assert_allclose_compact(
    topology_maps.fractional_masks["OCN->ATM"],
    np.full((2, 2), 0.4),
)
```

Replace the standalone component-topology tests with a behavioral test of `_require_surface_role`:

```python
def test_surface_role_lookup_checks_mapping_key_and_component_name() -> None:
    components = {"ATM": SimpleNamespace(name="WRONG", grid=make_test_grid())}
    with pytest.raises(
        CouplerError,
        match="Surface mask policy requires role component 'ATM' to be registered",
    ):
        surface_masks_module._require_surface_role(components, "ATM")
```

Update source-boundary tests to assert `component_topology.py` is absent, `surface_masks.py` owns `_require_surface_role`, and `ExchangeTopologyState` is absent.
Also assert `not hasattr(RuntimeTopologyMaps, "empty")`.

- [ ] **Step 2: Run tests to verify RED**

```bash
/Users/romannuterman/miniforge3/envs/scipy/bin/python -m pytest \
  tests/test_coupler_coverage.py tests/test_runtime_state.py \
  tests/test_runtime_facade_boundaries.py \
  tests/test_tools_components_and_plotting.py -q --tb=short
```

Expected: result-type and private-owner assertions fail.

- [ ] **Step 3: Implement the direct topology flow**

Make `build_exchange_topology` return `prepared_maps`. Remove `ExchangeTopologyState` and `RuntimeTopologyMaps.empty`. Change initialization to:

```python
@dataclass(frozen=True)
class RuntimeInitializationState:
    components: MappingProxyType[str, _ComponentBinding]
    runtime_contracts: dict[str, ExchangeContract]
    topology_maps: RuntimeTopologyMaps
```

Assign the topology result directly in `initialize_coupler_runtime`:

```python
topology_maps = build_exchange_topology(
    components=prepared_components,
    exchanges=exchanges,
    topology_maps=topology_maps,
    topology_policy=topology_policy,
    dtype=dtype,
    logger=logger,
)
return RuntimeInitializationState(
    components=MappingProxyType(prepared_components),
    runtime_contracts=runtime_contracts,
    topology_maps=topology_maps,
)
```

In `prepared.py` use:

```python
topology_maps = initialized.topology_maps
```

Move the checked lookup into `surface_masks.py`:

```python
def _require_surface_role(
    components: Mapping[str, _SurfaceRoleComponent],
    role_name: str,
) -> _SurfaceRoleComponent:
    try:
        component = components[role_name]
    except KeyError as exc:
        raise CouplerError(
            f"Surface mask policy requires role component {role_name!r} to be registered"
        ) from exc
    if component.name != role_name:
        raise CouplerError(
            f"Surface mask policy requires role component {role_name!r} to be registered"
        )
    return component
```

Delete `component_topology.py` and clean imports/`__all__` lists.

- [ ] **Step 4: Run focused tests and fast suite**

Run Step 2 and the fast suite. Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add vercor/_runtime tests/test_coupler_coverage.py tests/test_runtime_state.py \
  tests/test_runtime_facade_boundaries.py tests/test_tools_components_and_plotting.py
git commit -m "refactor: simplify runtime topology state"
```

### Task 5: Remove one-use runtime execution wrappers

**Files:**
- Delete: `vercor/_runtime/runner.py`
- Modify: `vercor/_runtime/facade.py`
- Modify: `tests/_runtime_helpers.py`
- Modify: `tests/test_runtime_state.py`
- Modify: `tests/test_runtime_facade_boundaries.py`
- Modify: `DEPENDENCIES.md`

**Interfaces:**
- Consumes: `RuntimeRunContext`, `build_validated_execution_plan(context)`, and `execute_plan(state, *, plan, context)`.
- Produces: unchanged `runtime_facade.run(runtime_state, *, prepared, logger, output=None) -> RunState` with direct context construction and plan execution.

- [ ] **Step 1: Rewrite tests around the observable facade boundary**

Update `run_scanned_coupler` to force the JAX backend through the facade:

```python
jax_coupling = replace(
    coupling,
    runtime=replace(coupling.runtime, backend="jax"),
)
return runtime_facade.run(
    prepared_state,
    prepared=jax_coupling,
    logger=coupler.logger,
)
```

Update architecture tests to require `RuntimeRunContext` construction plus `build_validated_execution_plan` and `execute_plan` in `facade.py`, and require `runner.py` to be absent.

- [ ] **Step 2: Run tests to verify RED**

```bash
/Users/romannuterman/miniforge3/envs/scipy/bin/python -m pytest \
  tests/test_runtime_state.py tests/test_runtime_facade_boundaries.py \
  tests/test_runtime_run.py tests/test_v0_4_workflow_execution.py -q --tb=short
```

Expected: facade ownership assertions fail before implementation.

- [ ] **Step 3: Inline context construction and execution**

In `facade.py`, remove `runtime_run_context` and import the two execution functions. Replace the call with:

```python
context = RuntimeRunContext(
    run_order=prepared.run_order,
    clock=prepared.clock,
    logger=logger,
    dispatch_context=prepared.dispatch_context,
    interrupts=prepared.interrupts,
    options=prepared.runtime,
    output=output,
)
plan = build_validated_execution_plan(context)
final_state = execute_plan(runtime_state, plan=plan, context=context)
```

Delete `runner.py`, remove its test imports, and remove it from `DEPENDENCIES.md` while retaining `run_context.py` before `execution.py`/`backends.py`.

- [ ] **Step 4: Run focused tests and fast suite**

Run Step 2 and the fast suite. Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add vercor/_runtime/facade.py vercor/_runtime/runner.py tests/_runtime_helpers.py \
  tests/test_runtime_state.py tests/test_runtime_facade_boundaries.py DEPENDENCIES.md
git commit -m "refactor: remove runtime execution wrappers"
```

### Task 7: Shrink bilinear interpolator PyTree state

**Files:**
- Modify: `tests/test_bilinear_rectilinear_interpolator.py`
- Modify: `tests/test_bilinear_interpolator_boundaries.py`
- Modify: `vercor/_interpolators/bilinear_rectilinear.py`

**Interfaces:**
- Preserves: constructor and `apply_scalar`/`apply_vector` signatures.
- Produces: 26 dynamic leaves instead of 28 for the representative interpolator; removes `fx`, `fy`, `nx_source`, `ny_source`, and post-construction `lat_descending` state.

- [ ] **Step 1: Write failing minimal-PyTree and descending-grid behavior tests**

Replace the direct `lat_descending` assertion with a behavior assertion, and extend the round-trip test:

```python
def test_descending_latitude_is_constructor_only_and_interpolates_correctly() -> None:
    interp = _scalar_interp(
        np.asarray([0.0, 1.0]),
        np.asarray([1.0, 0.0]),
        np.asarray([0.5]),
        np.asarray([0.5]),
        periodic_longitude=False,
    )
    result = interp.apply_scalar(np.asarray([[3.0, 5.0], [1.0, 3.0]]))
    assert_allclose_compact(result, np.asarray([[3.0]]))
    assert not hasattr(interp, "lat_descending")


def test_interpolator_pytree_omits_construction_only_weights() -> None:
    interp = _scalar_interp(
        np.asarray([0.0, 1.0]),
        np.asarray([0.0, 1.0]),
        np.asarray([0.25]),
        np.asarray([0.75]),
        periodic_longitude=False,
    )
    leaves, treedef = jax.tree_util.tree_flatten(interp)
    restored = jax.tree_util.tree_unflatten(treedef, leaves)

    assert len(leaves) == 26
    for name in ("fx", "fy", "nx_source", "ny_source"):
        assert not hasattr(restored, name)
    assert_allclose_compact(
        jax.jit(restored.apply_scalar)(np.asarray([[5.0, 7.0], [8.0, 10.0]])),
        np.asarray([[7.75]]),
    )
```

- [ ] **Step 2: Run tests to verify RED**

```bash
/Users/romannuterman/miniforge3/envs/scipy/bin/python -m pytest \
  tests/test_bilinear_rectilinear_interpolator.py \
  tests/test_bilinear_interpolator_boundaries.py -q --tb=short
```

Expected: leaf count is 28 and removed attributes still exist.

- [ ] **Step 3: Remove construction-only state**

Remove `fx`/`fy` from `pytree_children`; remove redundant names and `lat_descending` from `pytree_aux_data`. Use locals and canonical dimensions:

```python
lat_ascending = _geometry.all_positive(lat_diff)
lat_descending = _geometry.all_negative(lat_diff)
if not (lat_ascending or lat_descending):
    raise ValueError("lat_src must be strictly monotonic (ascending or descending).")
self.lat_ascending = lat_ascending
src_mask_array = jnp.ones((self.nlat, self.nlon), dtype=bool)
```

Do not assign `weights.fx` or `weights.fy`; the four final weights already
contain the operational values. Remove the duplicate `self.nlon` assignment.
Keep the separate validity calculations in `_apply_bilinear_scalar` and
`_extrapolate_scalar`: they serve different execution paths.

- [ ] **Step 4: Run interpolation, regridding, gradient, and fast tests**

```bash
/Users/romannuterman/miniforge3/envs/scipy/bin/python -m pytest \
  tests/test_bilinear_rectilinear_interpolator.py \
  tests/test_bilinear_interpolator_boundaries.py \
  tests/test_bilinear_rectilinear_regridder.py \
  tests/test_gradients.py -q --tb=short
/Users/romannuterman/miniforge3/envs/scipy/bin/python -m pytest tests/ -q --fast --tb=short
```

Expected: all pass with 26 representative leaves and unchanged numerical results.

- [ ] **Step 5: Commit**

```bash
git add vercor/_interpolators/bilinear_rectilinear.py \
  tests/test_bilinear_rectilinear_interpolator.py \
  tests/test_bilinear_interpolator_boundaries.py
git commit -m "perf: shrink bilinear interpolator pytree"
```

### Task 8: Copy Veros state once per forcing update

**Files:**
- Modify: `tests/test_external_components_coverage.py`
- Modify: `tests/test_external_tools_coverage.py`
- Modify: `vercor/setups/_external/veros_state.py`

**Interfaces:**
- Consumes: `copy_state`, `update_veros_interior`, `runtime_array_to_host`, and Veros variable unlock context.
- Produces: unchanged `apply_veros_forcing_fields(state, forcing_fields, *, jitted) -> VerosState` with exactly one copy and four assignments.

- [ ] **Step 1: Write the failing copy-count test**

Replace the isolated `set_variable` tests in both listed test modules with a
direct `apply_veros_forcing_fields` test. Use `_FakeVariableStore` (or the
equivalent existing unlock-capable fake) with zero-filled `taux`, `tauy`,
`qnet`, and `qnec` arrays:

```python
def test_apply_veros_forcing_fields_copies_once_and_updates_all_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    variables = _FakeVariableStore(
        **{
            name: np.zeros((8, 8, 1), dtype=float)
            for name in ("taux", "tauy", "qnet", "qnec")
        }
    )
    state = SimpleNamespace(variables=variables)
    copy_calls: list[bool] = []

    def recording_copy(value: Any, jitted: bool = True) -> Any:
        copy_calls.append(jitted)
        return deepcopy(value)

    monkeypatch.setattr(veros_state_module, "copy_state", recording_copy)
    forcing = veros_state_module.VerosForcingFields(
        taux=jnp.ones((4, 4, 1)),
        tauy=jnp.full((4, 4, 1), 2.0),
        qnet=jnp.full((4, 4, 1), 3.0),
        qnec=jnp.full((4, 4, 1), 4.0),
    )

    result = veros_state_module.apply_veros_forcing_fields(
        state,
        forcing,
        jitted=True,
    )

    assert not hasattr(veros_state_module, "set_variable")
    assert copy_calls == [True]
    for name, expected in zip(("taux", "tauy", "qnet", "qnec"), (1, 2, 3, 4)):
        assert_allclose_compact(
            getattr(result.variables, name)[2:-2, 2:-2, :],
            np.full((4, 4, 1), expected),
        )
```

In the two `step_veros_runtime` tests, monkeypatch
`apply_veros_forcing_fields` instead of `set_variable`. Record and assert the
four fields from the received `VerosForcingFields`, return the supplied state,
and retain the existing SST, transpose, and NaN-cleaning assertions.

- [ ] **Step 2: Run tests to verify RED**

```bash
/Users/romannuterman/miniforge3/envs/scipy/bin/python -m pytest \
  tests/test_external_components_coverage.py \
  tests/test_external_tools_coverage.py -q --tb=short
```

Expected: copy count is four under the current loop.

- [ ] **Step 3: Copy and unlock once**

Replace the implementation with:

```python
updated_state = copy_state(state, jitted=jitted)
variables = updated_state.variables
with variables.unlock():
    for variable_name, variable_value in zip(
        ("taux", "tauy", "qnet", "qnec"),
        forcing_fields,
        strict=True,
    ):
        current = getattr(variables, variable_name)
        updated = update_veros_interior(current, variable_value)
        setattr(variables, variable_name, runtime_array_to_host(updated))
return updated_state
```

Remove `set_variable` and its `__all__` entry after migrating all direct and
monkeypatched test references to the four-field behavior.

- [ ] **Step 4: Run Veros, optional-adapter, and fast tests**

```bash
/Users/romannuterman/miniforge3/envs/scipy/bin/python -m pytest \
  tests/test_external_components_coverage.py \
  tests/test_external_tools_coverage.py tests/test_v0_4_physics.py -q --tb=short
/Users/romannuterman/miniforge3/envs/scipy/bin/python -m pytest tests/ -q --fast --tb=short
```

Expected: one-copy assertion and all existing Veros forcing/SST behavior pass.

- [ ] **Step 5: Commit**

```bash
git add vercor/setups/_external/veros_state.py \
  tests/test_external_components_coverage.py tests/test_external_tools_coverage.py
git commit -m "perf: update veros forcing with one state copy"
```

### Task 9: Update project records and run release-proportionate verification

**Files:**
- Modify: `PROGRESS.md`
- Modify if dependency order changed beyond Tasks 4–5 edits: `DEPENDENCIES.md`
- Update generated graph: `graphify-out/*`
- Verify: all modified source and tests

**Interfaces:**
- Produces: current project orientation, dependency order, graph, and one final verified implementation commit if formatting/documentation adjustments remain.

- [ ] **Step 1: Update `PROGRESS.md` with exact evidence**

Add a dated status entry listing completed simplifications, RED/GREEN focused counts, fast/full counts, coverage, formatting, lint, typing, compile, and graph-update results. Do not write expected counts before commands finish.

- [ ] **Step 2: Format and run static checks**

```bash
conda run -n scipy black vercor examples tests
conda run -n scipy flake8 . --count --max-line-length=120 --statistics
conda run -n scipy mypy vercor examples tests
/Users/romannuterman/miniforge3/envs/scipy/bin/python -m compileall -q vercor examples tests
git diff --check
```

Expected: Black completes, flake8 reports zero errors, mypy succeeds, compileall is silent, and whitespace check is clean.

- [ ] **Step 3: Run focused subsystem tests**

```bash
/Users/romannuterman/miniforge3/envs/scipy/bin/python -m pytest \
  tests/test_helpers_coverage.py \
  tests/test_component_base_coverage.py \
  tests/test_component_models_coverage.py \
  tests/test_fluxes_utilities.py \
  tests/test_bilinear_rectilinear_interpolator.py \
  tests/test_bilinear_rectilinear_regridder.py \
  tests/test_conservative_rectilinear_regridder.py \
  tests/test_period_averages.py \
  tests/test_v0_4_output_providers.py \
  tests/test_runtime_state.py \
  tests/test_runtime_facade_boundaries.py \
  \
  tests/test_external_components_coverage.py \
  tests/test_external_tools_coverage.py -q --tb=short
```

Expected: all focused tests pass.

- [ ] **Step 4: Run fast, full, and coverage gates**

```bash
/Users/romannuterman/miniforge3/envs/scipy/bin/python -m pytest \
  tests/ -q --fast -n4 --dist=loadscope --max-worker-restart=0 --durations=25 --tb=short
/Users/romannuterman/miniforge3/envs/scipy/bin/python -m pytest \
  tests/ -q -n4 --dist=loadscope --max-worker-restart=0 --durations=25 --tb=short
/Users/romannuterman/miniforge3/envs/scipy/bin/python -m pytest \
  --cov=vercor tests/ -q -n4 --dist=loadscope --max-worker-restart=0 \
  --durations=25 --tb=short
```

Expected: every selected test passes and branch coverage remains at least 90%.

- [ ] **Step 5: Update Graphify and review the final diff**

```bash
/Users/romannuterman/miniforge3/envs/scipy/bin/python -m graphify update .
git status --short
git diff --stat
git diff --check
```

Expected: graph update succeeds, only intentional project/graph files are dirty, and whitespace remains clean.

- [ ] **Step 6: Commit records and any generated graph update**

```bash
git add PROGRESS.md DEPENDENCIES.md graphify-out
git diff --cached --check
git commit -m "docs: record simplification verification"
```

- [ ] **Step 7: Verify final repository state**

```bash
git status --short
git log -10 --oneline
```

Expected: clean working tree and a sequence of small, test-passing simplification commits.
