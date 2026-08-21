# Dinosaur 1.3.6 and JCM 2.0.1 Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Migrate VerCOR's optional JCM integration from Dinosaur 1.3.5/JCM 1.1.1 to Dinosaur 1.3.6/JCM 2.0.1 while preserving public setup, coupling-field, output, dtype, and differentiation contracts.

**Architecture:** Keep the existing private JAXGCM adapter modules, but update their upstream boundary to JCM 2. Runtime state will separately own the latest gridpoint dynamics, physics diagnostics, native dycore state, and cross-step physics carry; JCM land and output adapters will translate JCM 2 `TimeSeries` and mapping diagnostics into existing VerCOR representations.

**Tech Stack:** Python 3.12+, JAX, tree-math, Dinosaur 1.3.6, JCM 2.0.1, pytest, Black, flake8, mypy, Flit.

**Spec:** `docs/superpowers/specs/2026-08-21-dinosaur-jcm-2-migration-design.md`

## Global Constraints

- The optional dependency pins are exactly `dinosaur==1.3.6` and `jcm==2.0.1`.
- Support JCM 2.0.1 directly; do not add JCM 1.x compatibility branches.
- Preserve all public `vercor.setups` signatures and lazy optional imports.
- Preserve existing VerCOR/JCM field names, orientation, units, flux signs, and dtype policy.
- Carry all evolving JCM state functionally in immutable runtime payloads.
- Preserve JIT, JVP, and reverse-mode differentiation.
- Do not add numerical correction factors.
- Use `/Users/romannuterman/miniforge3/envs/scipy/bin/python` for project commands.
- Keep successful test output concise and record durable outcomes in `PROGRESS.md`.
- Run `pytest tests/ -q --fast --tb=short` immediately before every commit.
- Treat Tasks 2-6 as one atomic adapter migration after JCM 2 is installed: their
  intermediate states are intentionally not commits because construction, runtime,
  land, output, real-model tests, and durable evidence change together under the
  new environment.

## File Structure

- `pyproject.toml`: exact optional dependency ownership.
- `vercor/setups/_external/jax_gcm_state.py`: JCM 2 construction, bootstrap,
  state bundle, and step closure.
- `vercor/setups/_external/jax_gcm_runtime.py`: immutable payload stepping and
  JCM 2 diagnostics extraction.
- `vercor/setups/_external/jax_gcm_output.py`: JCM 2 output translation and
  SPEEDY unit metadata.
- `vercor/setups/_external/jax_gcm.py`: lazy factory annotations.
- `vercor/setups/_data/jcm_land.py`: static and time-first forcing layout.
- `tests/test_distribution_boundaries.py`: dependency pin contract.
- `tests/test_api_boundaries.py`: removed/imported JCM API source contracts.
- `tests/test_external_components_coverage.py`: setup, runtime, and output units.
- `tests/test_component_models_coverage.py`: JCM land layout units.
- `tests/test_coupler_runtime.py`: immutable payload, JIT, AD, and real JCM tests.
- `DEPENDENCIES.md`: adapter dependency-order description.
- `PROGRESS.md`: bounded durable completion evidence.

---

### Task 1: Pin, Commit, and Install the New Optional Dependency Stack

**Files:**
- Modify: `tests/test_distribution_boundaries.py:164`
- Modify: `pyproject.toml:40`

**Interfaces:**
- Consumes: `[project.optional-dependencies].jcm`.
- Produces: the exact ordered dependency list
  `("dinosaur==1.3.6", "jcm==2.0.1")` for local and CI environments.

- [ ] **Step 1: Write the failing dependency contract**

Add after `extras` is loaded:

```python
assert extras["jcm"] == ["dinosaur==1.3.6", "jcm==2.0.1"]
```

- [ ] **Step 2: Run the contract and verify RED**

```bash
/Users/romannuterman/miniforge3/envs/scipy/bin/python -m pytest tests/test_distribution_boundaries.py::test_runtime_metadata_separates_test_and_development_dependencies -q -n0 --tb=short
```

Expected: FAIL showing the old exact pins.

- [ ] **Step 3: Update the exact pins**

```toml
jcm = ["dinosaur==1.3.6", "jcm==2.0.1"]
```

- [ ] **Step 4: Re-run the contract**

Run Step 2 again. Expected: PASS.

- [ ] **Step 5: Verify the old environment remains green before the pin commit**

```bash
/Users/romannuterman/miniforge3/envs/scipy/bin/python -m pytest tests/ -q --tb=short
/Users/romannuterman/miniforge3/envs/scipy/bin/python -m pytest tests/ -q --fast --tb=short
```

Expected: both suites PASS with the existing installed dependencies. These are
the last test runs before changing the environment itself.

- [ ] **Step 6: Commit the dependency unit**

```bash
git add -- pyproject.toml tests/test_distribution_boundaries.py
git diff --cached --check
git commit -m "build: update Dinosaur and JCM pins"
```

- [ ] **Step 7: Install and verify the project-owned JCM extra**

```bash
/Users/romannuterman/miniforge3/envs/scipy/bin/python -m pip install -e '.[jcm]'
/Users/romannuterman/miniforge3/envs/scipy/bin/python -c "from importlib.metadata import version; assert version('dinosaur') == '1.3.6'; assert version('jcm') == '2.0.1'"
```

Expected: both assertions pass. Do not commit again until Task 6 completes and
the coherent JCM 2 adapter passes every required verification gate.

---

### Task 2: Migrate JCM Construction, Bootstrap, and State Carry

**Files:**
- Modify: `tests/test_api_boundaries.py:1332`
- Modify: `tests/test_external_components_coverage.py:532-868`
- Modify: `vercor/setups/_external/jax_gcm_state.py:1-247`
- Modify: `vercor/setups/_external/jax_gcm.py:16-23`

**Interfaces:**
- Consumes: `Model.bootstrap_state()`, `Model.run_from_state_with_carry(...)`,
  `Model.dycore.to_physics_state(...)`, and `speedy_physics(...)`.
- Produces: `JCMState(dynamics, physics, dycore_state, physics_carry)` and
  `Callable[[JCMState, ForcingData], tuple[JCMState, ModelPredictions]]`.

- [ ] **Step 1: Add JCM 2 source-boundary tests**

Add to `tests/test_api_boundaries.py`:

```python
@pytest.mark.fast_always
def test_jax_gcm_adapter_uses_jcm_2_api_owners() -> None:
    state_source = Path(
        "vercor/setups/_external/jax_gcm_state.py"
    ).read_text(encoding="utf-8")
    factory_source = Path(
        "vercor/setups/_external/jax_gcm.py"
    ).read_text(encoding="utf-8")

    assert "from jcm.physics.speedy.speedy_terms import speedy_physics" in state_source
    assert "from jcm.terrain import TerrainData" in state_source
    assert "from jcm.forcing import ForcingData" in state_source
    assert "SpeedyPhysics" not in state_source
    assert "dynamics_state_to_physics_state" not in state_source
    assert "_prepare_initial_modal_state" not in state_source
    assert "from jcm.terrain import TerrainData as _TerrainData" in factory_source
```

Replace the old step-function unit with:

```python
def test_generate_step_function_threads_jcm_2_physics_carry() -> None:
    component = jax_gcm_state_module.JAXGCMSetupState.__new__(
        jax_gcm_state_module.JAXGCMSetupState
    )
    component.save_interval = timedelta(days=2)
    component.coupling_timestep = timedelta(hours=12)
    component._dtype_policy = DTypePolicy()
    calls: dict[str, Any] = {}

    class _FakeModel:
        def run_from_state_with_carry(
            self,
            initial_state: Any,
            forcing: Any,
            save_interval: float,
            total_time: float,
            output_averages: bool,
            initial_physics_state: Any,
        ) -> tuple[str, str, Any]:
            calls["run"] = (
                initial_state,
                initial_physics_state,
                save_interval,
                total_time,
                output_averages,
                forcing,
            )
            predictions = SimpleNamespace(
                dynamics={"wind": jnp.asarray([[1.0, 3.0], [5.0, 7.0]])},
                physics={"heat": jnp.asarray([[2.0, 4.0], [6.0, 8.0]])},
            )
            return "next-dycore", "next-carry", predictions

    component.model = _FakeModel()
    state = jax_gcm_state_module.JCMState(
        dynamics={},
        physics={},
        dycore_state="initial-dycore",
        physics_carry="initial-carry",
    )

    next_state, predictions = component._generate_step_function(jitted=False)(
        state, "forcing"
    )

    assert calls["run"] == (
        "initial-dycore", "initial-carry", 2.0, 0.5, False, "forcing"
    )
    assert next_state.dycore_state == "next-dycore"
    assert next_state.physics_carry == "next-carry"
    assert_allclose_compact(next_state.dynamics["wind"], np.asarray([3.0, 5.0]))
    assert_allclose_compact(next_state.physics["heat"], np.asarray([4.0, 6.0]))
    assert predictions.physics["heat"].shape == (2, 2)
```

Add:

```python
def test_bootstrap_jcm_state_captures_native_state_and_physics_carry() -> None:
    calls = {"bootstrap": 0}

    class _FakeDycore:
        def to_physics_state(self, state: Any) -> dict[str, Any]:
            return {"converted": state}

    class _FakeModel:
        def __init__(self) -> None:
            self.dycore = _FakeDycore()
            self._final_dycore_state = None
            self._final_physics_state = None

        def bootstrap_state(self) -> None:
            calls["bootstrap"] += 1
            self._final_dycore_state = {"vorticity": jnp.asarray(1.0)}
            self._final_physics_state = {"heating": jnp.asarray(2.0)}

    model = _FakeModel()
    state = jax_gcm_state_module._bootstrap_jcm_state(cast(Any, model))

    assert calls == {"bootstrap": 1}
    assert state.dynamics == {"converted": state.dycore_state}
    assert state.dycore_state is model._final_dycore_state
    assert state.physics is model._final_physics_state
    assert state.physics_carry is model._final_physics_state


@pytest.mark.parametrize(
    ("dycore_state", "physics_carry"),
    [(None, object()), (object(), None)],
)
def test_bootstrap_jcm_state_rejects_missing_final_state(
    dycore_state: Any,
    physics_carry: Any,
) -> None:
    model = SimpleNamespace(
        bootstrap_state=lambda: None,
        _final_dycore_state=dycore_state,
        _final_physics_state=physics_carry,
    )

    with pytest.raises(RuntimeError, match="did not initialize"):
        jax_gcm_state_module._bootstrap_jcm_state(cast(Any, model))
```

- [ ] **Step 2: Run the setup focus and verify RED**

```bash
/Users/romannuterman/miniforge3/envs/scipy/bin/python -m pytest tests/test_api_boundaries.py::test_jax_gcm_adapter_uses_jcm_2_api_owners tests/test_external_components_coverage.py::test_bootstrap_jcm_state_captures_native_state_and_physics_carry tests/test_external_components_coverage.py::test_bootstrap_jcm_state_rejects_missing_final_state tests/test_external_components_coverage.py::test_generate_step_function_threads_jcm_2_physics_carry tests/test_external_components_coverage.py::test_jax_gcm_constructor_builds_jax_backed_grid tests/test_external_components_coverage.py::test_jax_gcm_initialize_uses_provided_forcing_and_can_spin_up tests/test_external_components_coverage.py::test_jax_gcm_initialize_builds_default_forcing_when_missing -q -n0 --tb=short
```

Expected: import failures for removed JCM 1 APIs or old-state assertions.

- [ ] **Step 3: Replace JCM 1 imports and state fields**

Use:

```python
from jcm.forcing import ForcingData, default_forcing
from jcm.model import Model, ModelPredictions
from jcm.physics.speedy.params import Parameters
from jcm.physics.speedy.speedy_terms import speedy_physics
from jcm.physics_interface import PhysicsState
from jcm.terrain import TerrainData
```

Define:

```python
@tree_math.struct
@dataclass
class JCMState:
    """JCM gridpoint output, native dycore state, and physics carry."""

    dynamics: PhysicsState
    physics: Any
    dycore_state: Any
    physics_carry: Any
```

Update `jax_gcm.py` to type-check against `jcm.terrain.TerrainData`.

- [ ] **Step 4: Centralize bootstrap extraction**

Add:

```python
def _bootstrap_jcm_state(model: Model) -> JCMState:
    """Return the initialized JCM 2 dycore and physics state bundle."""

    model.bootstrap_state()
    dycore_state = model._final_dycore_state
    physics_carry = model._final_physics_state
    if dycore_state is None or physics_carry is None:
        raise RuntimeError("JCM bootstrap did not initialize dycore and physics state")
    return JCMState(
        dynamics=model.dycore.to_physics_state(dycore_state),
        physics=physics_carry,
        dycore_state=dycore_state,
        physics_carry=physics_carry,
    )
```

Construct physics with `speedy_physics(parameters=jcm_parameters)`. Replace
modal-state and `PhysicsData.zeros` setup with `_bootstrap_jcm_state(self.model)`.
Use `default_forcing(self.model.coords.horizontal)` without the removed
`lfluxland` forcing keyword.

- [ ] **Step 5: Implement carry-aware stepping**

Call:

```python
final_dycore_state, final_physics_carry, predictions = (
    self.model.run_from_state_with_carry(
        initial_state=state.dycore_state,
        initial_physics_state=state.physics_carry,
        save_interval=self.save_interval / timedelta(days=1),
        total_time=self.coupling_timestep / timedelta(days=1),
        output_averages=False,
        forcing=forcing,
    )
)
```

Return a new state with dtype-normalized means of `predictions.dynamics` and
`predictions.physics`, plus both final carry values.

- [ ] **Step 6: Update setup and spinup fakes**

Replace monkeypatches of `PhysicsData.zeros`,
`dynamics_state_to_physics_state`, and `_prepare_initial_modal_state` with fake
`bootstrap_state`, `dycore.to_physics_state`, and final-state attributes.
Assert spinup advances native state and physics carry and preserves dtype.

- [ ] **Step 7: Re-run the setup focus**

Run Step 2 again. Expected: PASS.

- [ ] **Step 8: Preserve the atomic migration checkpoint**

Run `git diff --check`, record the passing focused tests, and continue to Task
3 without committing. The installed JCM 2 environment requires the remaining
runtime, land, and output adapters to migrate before the repository-wide fast
suite can pass.

---

### Task 3: Adapt Runtime Field Mapping and Immutable Coupling Tests

**Files:**
- Modify: `vercor/setups/_external/jax_gcm_runtime.py:176-252`
- Modify: `tests/test_external_components_coverage.py:871-960`
- Modify: `tests/test_coupler_runtime.py:102-317`
- Modify: `tests/test_coupler_runtime.py:1656-1794`
- Modify: `tests/test_runtime_state.py:880-920`

**Interfaces:**
- Consumes: `JCMState.dynamics`, `JCMState.physics`, and JCM 2 mapping keys
  `_surface_flux` and `_shortwave_rad`.
- Produces: unchanged VerCOR atmosphere fields and an immutable payload with
  advanced dycore state and physics carry.

- [ ] **Step 1: Rewrite fake predictions as JCM 2 mappings**

Replace `_FakePhysicsPrediction` uses with:

```python
physics = {
    "_surface_flux": _FakeSurfaceFlux(
        shf=surface_flux[jnp.newaxis, ...],
        evap=(surface_flux * 0.1)[jnp.newaxis, ...],
        rlds=(surface_temperature + 3.0)[jnp.newaxis, ...],
    ),
    "_shortwave_rad": _FakeShortwaveRad(
        rsns=(surface_temperature + 4.0)[jnp.newaxis, ...],
    ),
}
```

Construct fake states with all four `JCMState` fields. Make the fake step
advance scalar marker leaves in `dycore_state` and `physics_carry` separately.

Add:

```python
def test_jax_gcm_runtime_rejects_missing_speedy_diagnostics() -> None:
    with pytest.raises(ComponentError, match="_shortwave_rad"):
        jax_gcm_runtime_module._required_speedy_diagnostics(
            {"_surface_flux": object()},
            component_name="ATM",
        )
```

- [ ] **Step 2: Run the runtime focus and verify RED**

```bash
/Users/romannuterman/miniforge3/envs/scipy/bin/python -m pytest tests/test_external_components_coverage.py::test_jax_gcm_step_maps_outputs_without_owning_output_cadence tests/test_external_components_coverage.py::test_jax_gcm_runtime_rejects_missing_speedy_diagnostics tests/test_coupler_runtime.py::test_jax_gcm_runs_inside_runtime_under_jit_and_grad tests/test_runtime_state.py -q -n0 --tb=short
```

Expected: old attribute access and three-field state fixtures fail.

- [ ] **Step 3: Add one validated diagnostics accessor**

Add to `jax_gcm_runtime.py`:

```python
def _required_speedy_diagnostics(
    physics: Mapping[str, Any],
    *,
    component_name: str,
) -> tuple[Any, Any]:
    """Return required SPEEDY diagnostics from a JCM 2 mapping."""

    required = ("_surface_flux", "_shortwave_rad")
    missing = tuple(name for name in required if name not in physics)
    if missing:
        names = ", ".join(missing)
        raise ComponentError(
            f"JAXGCM component '{component_name}' is missing JCM diagnostics: {names}"
        )
    return physics["_surface_flux"], physics["_shortwave_rad"]
```

- [ ] **Step 4: Map directly from the reduced state**

After `state._step_function(...)`, call:

```python
surface_flux, shortwave_rad = _required_speedy_diagnostics(
    jcm_state.physics,
    component_name=state.name,
)
```

Pass `surface_flux.shf`, `surface_flux.evap`, `surface_flux.rlds`,
`shortwave_rad.rsns`, and the arrays on `jcm_state.dynamics` to
`map_jcm_output_fields`. Remove runtime stack/unwrap/mean work because the step
closure already owns interval reduction.

- [ ] **Step 5: Assert immutable carry progression**

In the fake JIT test, compare explicit scalar marker leaves:

```python
assert float(atmosphere_state.payload.jcm_state.dycore_state["marker"]) != float(
    initial_payload.jcm_state.dycore_state["marker"]
)
assert float(atmosphere_state.payload.jcm_state.physics_carry["marker"]) != float(
    initial_payload.jcm_state.physics_carry["marker"]
)
```

Keep existing dtype, finite-field, forcing-template, and gradient assertions.

- [ ] **Step 6: Re-run the runtime focus**

Run Step 2 again. Expected: PASS.

- [ ] **Step 7: Preserve the atomic migration checkpoint**

Run `git diff --check`, record the passing runtime focus, and continue to Task
4 without committing.

---

### Task 4: Canonicalize JCM 2 Time-Series Land Forcing

**Files:**
- Modify: `vercor/setups/_data/jcm_land.py:1-72`
- Modify: `tests/test_component_models_coverage.py:1042-1104`
- Modify: `tests/test_coupler_runtime.py:1315-1368`

**Interfaces:**
- Consumes: `(longitude, latitude)` arrays or JCM 2 `TimeSeries.values` shaped
  `(time, longitude, latitude)`.
- Produces: `(latitude, longitude)` or `(time, latitude, longitude)` arrays for
  the existing daily transfer policy.

- [ ] **Step 1: Add time-first and invalid-shape tests**

Add:

```python
def test_jcm_land_canonicalizes_jcm_2_time_series_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        jcm_land_module,
        "create_lnd_mask_from_ocn",
        lambda **kwargs: (np.ones((2, 2)), np.zeros((2, 2))),
    )
    coords = SimpleNamespace(
        horizontal=SimpleNamespace(
            longitudes=np.deg2rad(np.asarray([0.0, 180.0])),
            latitudes=np.deg2rad(np.asarray([-45.0, 45.0])),
        )
    )
    values = np.arange(12.0).reshape(3, 2, 2)
    forcing = SimpleNamespace(
        stl_am=SimpleNamespace(values=values),
        soilw_am=SimpleNamespace(values=values + 20.0),
    )

    component = make_jcm_land(
        cast(Any, coords),
        cast(Any, forcing),
        make_test_grid(name="ocn"),
    )

    assert_allclose_compact(
        cast(Any, component)._data["land_surface_temperature"],
        values.transpose(0, 2, 1),
    )
```

Add:

```python
def test_jcm_land_rejects_invalid_jcm_forcing_rank() -> None:
    with pytest.raises(
        ValueError,
        match="JCM forcing field 'stl_am'.*shape \\(3,\\)",
    ):
        jcm_land_module._canonicalize_jcm_forcing_field(
            np.ones(3),
            field_name="stl_am",
        )
```

- [ ] **Step 2: Run the land focus and verify RED**

```bash
/Users/romannuterman/miniforge3/envs/scipy/bin/python -m pytest tests/test_component_models_coverage.py::test_jcm_land_constructor_converts_coords_and_preserves_data tests/test_component_models_coverage.py::test_jcm_land_canonicalizes_jcm_2_time_series_values tests/test_coupler_runtime.py::test_jcm_land_daily_forcing_replays_to_data_atmosphere_under_jit_and_grad -q -n0 --tb=short
```

Expected: the `TimeSeries` case fails in the generic array canonicalizer.

- [ ] **Step 3: Implement the JCM-specific canonicalizer**

```python
def _canonicalize_jcm_forcing_field(
    field: Any,
    *,
    field_name: str,
) -> jax.Array:
    """Return one static or time-first JCM surface field in VerCOR layout."""

    values = getattr(field, "values", field)
    array = as_jax_real_array(values)
    if array.ndim == 2:
        return array.T
    if array.ndim == 3:
        return array.transpose((0, 2, 1))
    raise ValueError(
        f"JCM forcing field '{field_name}' has shape {array.shape}; expected "
        "(longitude, latitude) or (time, longitude, latitude)"
    )
```

Import `jax`, remove the generic `canonicalize_surface_field` import, and call
the helper separately for `stl_am` and `soilw_am`.

- [ ] **Step 4: Run land and transfer tests**

Run Step 2, then:

```bash
/Users/romannuterman/miniforge3/envs/scipy/bin/python -m pytest tests/test_setup_lifecycle_helpers.py tests/test_component_models_coverage.py -q -n0 --tb=short
```

Expected: PASS.

- [ ] **Step 5: Preserve the atomic migration checkpoint**

Run `git diff --check`, record the passing land focus, and continue to Task 5
without committing.

---

### Task 5: Migrate Native JCM Output and Unit Metadata

**Files:**
- Modify: `vercor/setups/_external/jax_gcm_output.py:36-375`
- Modify: `tests/test_external_components_coverage.py:1071-1311`

**Interfaces:**
- Consumes: `JCMState.dynamics`, `JCMState.physics`,
  `ComposablePhysics.data_struct_to_dict`, and JCM's packaged unit tables.
- Produces: unchanged `OutputVariable`, `OutputFrame`, period, and snapshot
  contracts.

- [ ] **Step 1: Update output fixtures and add metadata fallback coverage**

Use:

```python
jcm_state = SimpleNamespace(
    dynamics={
        "temperature": np.arange(18.0).reshape(3, 2, 3),
        "u_wind": np.full((3, 2, 3), 4.0),
    },
    physics={},
)
```

Update assertions to reference `jcm_state.dynamics`. Add:

```python
def test_jax_gcm_unit_metadata_uses_packaged_speedy_fallback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    dynamics_root = tmp_path / "jcm"
    physics_root = tmp_path / "speedy"
    dynamics_root.mkdir()
    physics_root.mkdir()
    (dynamics_root / "dynamics_units_table.csv").write_text(
        "Variable,Units,Description\nunique_dynamics,K,dynamics field\n",
        encoding="utf-8",
    )
    (physics_root / "units_table.csv").write_text(
        "Variable,Units,Description\nunique_physics,W m-2,physics field\n",
        encoding="utf-8",
    )
    roots = {
        "jcm": dynamics_root,
        "jcm.physics.speedy": physics_root,
    }
    monkeypatch.setattr(
        jax_gcm_output_module.resources,
        "files",
        lambda package: roots[package],
    )

    metadata = jax_gcm_output_module.jax_gcm_unit_metadata(
        cast(Any, SimpleNamespace(UNITS_TABLE_CSV_PATH=None))
    )

    assert metadata["unique_dynamics"] == {
        "units": "K",
        "description": "dynamics field",
    }
    assert metadata["unique_physics"] == {
        "units": "W m-2",
        "description": "physics field",
    }
```

- [ ] **Step 2: Run output tests and verify RED**

```bash
/Users/romannuterman/miniforge3/envs/scipy/bin/python -m pytest tests/test_external_components_coverage.py::test_jax_gcm_write_output_persists_mean_dataset tests/test_external_components_coverage.py::test_jax_gcm_snapshot_output_uses_final_runtime_payload_not_runtime_data tests/test_external_components_coverage.py::test_jax_gcm_output_provider_samples_post_step_payload tests/test_external_components_coverage.py::test_jax_gcm_unit_metadata_uses_packaged_speedy_fallback -q -n0 --tb=short
```

Expected: state attribute and fallback metadata failures.

- [ ] **Step 3: Replace the default physics constructor**

```python
def _default_physics_module() -> _PhysicsModuleLike:
    from jcm.physics.speedy.speedy_terms import speedy_physics

    return cast(_PhysicsModuleLike, speedy_physics())
```

- [ ] **Step 4: Read the new state attributes**

```python
prediction = SimpleNamespace(
    dynamics=jax.tree_util.tree_map(
        _with_leading_time_dim,
        jcm_state.dynamics,
    ),
    physics=jax.tree_util.tree_map(
        _with_leading_time_dim,
        jcm_state.physics,
    ),
    times=as_jax_real_array([0.0]),
)
```

Keep dimension inference and direct NetCDF writing unchanged.

- [ ] **Step 5: Add the SPEEDY unit-table fallback**

When `UNITS_TABLE_CSV_PATH` is `None`, resolve:

```python
physics_resource = resources.files("jcm.physics.speedy").joinpath(
    "units_table.csv"
)
with resources.as_file(physics_resource) as physics_path:
    metadata.update(_read_units_table(physics_path))
```

Preserve the explicit-path behavior for custom physics modules.

- [ ] **Step 6: Run the complete JCM output focus**

```bash
/Users/romannuterman/miniforge3/envs/scipy/bin/python -m pytest tests/test_external_components_coverage.py -q -n0 --tb=short
```

Expected: PASS.

- [ ] **Step 7: Run the repository fast suite for the coherent adapter so far**

```bash
/Users/romannuterman/miniforge3/envs/scipy/bin/python -m pytest tests/ -q --fast --tb=short
git diff --check
```

Expected: PASS under Dinosaur 1.3.6/JCM 2.0.1.

- [ ] **Step 8: Preserve the final adapter checkpoint before real verification**

Do not commit. Continue to Task 6 so the real JCM regression, static checks,
full suite, coverage, and package build all verify this same implementation.

---

### Task 6: Verify Real JCM 2 and Update Durable Documentation

**Files:**
- Modify: `tests/test_coupler_runtime.py:1760-1794`
- Modify: `tests/test_external_components_coverage.py:720-797`
- Modify: `DEPENDENCIES.md:1-46`
- Modify: `PROGRESS.md:1-180`
- Modify if required to preserve the progress bound:
  `docs/progress-archive-2026-07-22.md`

**Interfaces:**
- Consumes: the installed JCM 2.0.1/Dinosaur 1.3.6 stack and Tasks 1-5.
- Produces: real optional-model evidence, current dependency-order
  documentation, and a bounded progress record.

- [ ] **Step 1: Replace the obsolete real payload assertion with a cross-step regression**

Replace `test_real_jax_gcm_initial_payload_seeds_speedy_coords` with:

```python
def test_real_jax_gcm_runtime_seeds_and_advances_jcm_2_carry(
    fast_mode: bool,
) -> None:
    if fast_mode:
        pytest.skip("Real JCM carry regression runs outside --fast")

    from vercor.setups import JAXGCMConfig, load_jcm_inputs, make_jax_gcm

    inputs = load_jcm_inputs()
    component = make_jax_gcm(
        inputs.coords,
        inputs.terrain,
        config=JAXGCMConfig(forcing_data=inputs.forcing, jitted=True),
    )
    coupler = Coupler(
        Clock(datetime(2000, 1, 1), 86400.0, 2),
        components=(component,),
        run_order=("ATM",),
        runtime=RuntimeOptions(topology=None),
    )
    initial_state = coupler.initial_state()
    setup_hook = component.spec.lifecycle.setup
    assert setup_hook is not None
    setup_state = cast(Any, setup_hook).__self__
    initial_payload = initial_state._component_state("ATM").payload
    assert initial_payload is not None
    initial_carry = initial_payload.jcm_state.physics_carry
    assert initial_payload.jcm_state.dycore_state is not None
    assert initial_carry is not None
    assert jax.tree_util.tree_structure(initial_carry) == jax.tree_util.tree_structure(
        setup_state.model._final_physics_state
    )
    initial_float_leaves = [
        np.asarray(leaf)
        for leaf in jax.tree_util.tree_leaves(initial_carry)
        if hasattr(leaf, "dtype") and jnp.issubdtype(leaf.dtype, jnp.inexact)
    ]
    assert all(np.all(np.isfinite(leaf)) for leaf in initial_float_leaves)

    final_state = run_scanned_coupler(coupler, initial_state)
    final_payload = final_state._component_state("ATM").payload
    assert final_payload is not None
    final_carry = final_payload.jcm_state.physics_carry
    final_float_leaves = [
        np.asarray(leaf)
        for leaf in jax.tree_util.tree_leaves(final_carry)
        if hasattr(leaf, "dtype") and jnp.issubdtype(leaf.dtype, jnp.inexact)
    ]
    assert jax.tree_util.tree_structure(final_carry) == jax.tree_util.tree_structure(
        initial_carry
    )
    assert len(final_float_leaves) == len(initial_float_leaves)
    assert any(
        not np.allclose(initial, final)
        for initial, final in zip(initial_float_leaves, final_float_leaves)
    )
    temperature = final_state._component_state("ATM").fields.get("temperature")
    assert np.all(np.isfinite(np.asarray(temperature)))
```

Retain the fake JIT/gradient and data-forcing JVP tests.

- [ ] **Step 2: Update direct JCM forcing construction**

Replace partial `ForcingData(**forcing_values)` calls with:

```python
forcing = jax_gcm_state_module.ForcingData.zeros((2, 3)).copy(
    stl_am=forcing_values["stl_am"],
    sea_surface_temperature=forcing_values["sea_surface_temperature"],
)
```

This retains JCM 2's additional forcing leaves.

- [ ] **Step 3: Run focused real JCM and AD verification**

```bash
/Users/romannuterman/miniforge3/envs/scipy/bin/python -m pytest tests/test_coupler_runtime.py::test_real_jax_gcm_runtime_seeds_and_advances_jcm_2_carry tests/test_external_components_coverage.py::test_jax_gcm_initialize_uses_provided_forcing_and_can_spin_up tests/test_external_components_coverage.py::test_jax_gcm_spinup_normalizes_loaded_forcing_to_runtime_dtype tests/test_setup_lifecycle_helpers.py::test_make_jcm_land_atmosphere_replaces_only_missing_forcing tests/test_coupler_runtime.py::test_jax_gcm_runs_inside_runtime_under_jit_and_grad tests/test_coupler_runtime.py::test_data_forcing_replays_into_jax_gcm_runtime_under_jit_grad_and_jvp -q -n0 --tb=short
```

Expected: PASS against the exact new dependency versions.

- [ ] **Step 4: Confirm the public smoke is bounded**

The new real regression in Step 1 is the bounded public smoke: it uses
`load_jcm_inputs`, `make_jax_gcm`, two coupling steps, no spinup, and
`RuntimeOptions(topology=None)`. Do not run the packaged 100-year example.

- [ ] **Step 5: Update dependency-order descriptions**

Keep `DEPENDENCIES.md` layer numbers unchanged. Update layers 2 and 14-18 to
name JCM 2 forcing/terrain loading, time-series land layout, mapping output,
and carry-aware runtime/state ownership. Do not reorder unrelated modules.

- [ ] **Step 6: Run static quality gates**

```bash
/Users/romannuterman/miniforge3/envs/scipy/bin/python -m black vercor tests
/Users/romannuterman/miniforge3/envs/scipy/bin/python -m flake8 . --count --max-line-length=120 --statistics
/Users/romannuterman/miniforge3/envs/scipy/bin/python -m mypy vercor tests
/Users/romannuterman/miniforge3/envs/scipy/bin/python -m compileall -q vercor tests
git diff --check
```

Expected: every command exits 0.

- [ ] **Step 7: Run fast, full, and coverage suites**

```bash
/Users/romannuterman/miniforge3/envs/scipy/bin/python -m pytest tests/ -q --fast --tb=short
/Users/romannuterman/miniforge3/envs/scipy/bin/python -m pytest tests/ -q --tb=short
/Users/romannuterman/miniforge3/envs/scipy/bin/python -m pytest --cov=vercor tests/ -q --tb=short
```

Expected: all tests pass and branch coverage remains at least 90%.

- [ ] **Step 8: Build and inspect distributions**

```bash
VERCOR_MIGRATION_DIST="$(mktemp -d /private/tmp/vercor-jcm2-dist.XXXXXX)"
/Users/romannuterman/miniforge3/envs/scipy/bin/python -m build --outdir "$VERCOR_MIGRATION_DIST"
/Users/romannuterman/miniforge3/envs/scipy/bin/python -m twine check "$VERCOR_MIGRATION_DIST"/*
/Users/romannuterman/miniforge3/envs/scipy/bin/python -m pytest tests/test_distribution_boundaries.py tests/test_setup_boundaries.py -q -n0 --tb=short
```

Expected: one wheel and one source distribution, successful Twine checks, and
passing distribution/setup boundaries.

- [ ] **Step 9: Record final evidence within 180 lines**

Add a top `PROGRESS.md` item naming exact versions, state/carry, diagnostics,
land-forcing, output, focused/static/fast/full/coverage/build results, and
remaining upstream warnings. If room is required, move the oldest complete
status item verbatim to `docs/progress-archive-2026-07-22.md`.

- [ ] **Step 10: Re-run progress and whitespace contracts**

```bash
/Users/romannuterman/miniforge3/envs/scipy/bin/python -m pytest tests/test_api_architecture_review.py::test_progress_is_bounded_active_memory tests/test_distribution_boundaries.py -q -n0 --tb=short
/Users/romannuterman/miniforge3/envs/scipy/bin/python -m pytest tests/ -q --fast --tb=short
git diff --check
```

Expected: PASS.

- [ ] **Step 11: Commit the verified atomic JCM 2 migration**

Inspect status and stage only changed migration files:

```bash
git add -- DEPENDENCIES.md PROGRESS.md vercor/setups/_data/jcm_land.py vercor/setups/_external/jax_gcm.py vercor/setups/_external/jax_gcm_output.py vercor/setups/_external/jax_gcm_runtime.py vercor/setups/_external/jax_gcm_state.py tests/test_api_boundaries.py tests/test_component_models_coverage.py tests/test_coupler_runtime.py tests/test_external_components_coverage.py tests/test_runtime_state.py
git diff --cached --check
git commit -m "refactor: migrate JCM integration to JCM 2"
```

If the archive changed, stage its exact path separately before committing.

---

### Task 7: Review, Publish, and Open the Draft Pull Request

**Files:**
- Inspect: every file changed from `origin/main`.
- Modify only when review identifies a concrete defect: its owning source,
  test, or documentation file from Tasks 1-6.

**Interfaces:**
- Consumes: a clean, fully verified feature branch.
- Produces: a pushed branch and one draft PR targeting `main`.

- [ ] **Step 1: Review the complete branch diff**

```bash
git status --short --branch
git diff --stat origin/main...HEAD
git diff --check origin/main...HEAD
git log --oneline origin/main..HEAD
```

Check every acceptance criterion in the design. Confirm there are no unrelated
files and no JCM 1 compatibility branches.

- [ ] **Step 2: Request code review**

Invoke `superpowers:requesting-code-review`. Address confirmed Critical and
Important findings, run their focused tests, then repeat the fast suite and
`git diff --check`.

- [ ] **Step 3: Run completion verification**

Invoke `superpowers:verification-before-completion`. If review changed source
or tests, refresh static, fast/full, coverage, and distribution evidence.

- [ ] **Step 4: Finish the development branch**

Invoke `superpowers:finishing-a-development-branch` and select the publish/PR
path requested by the user.

- [ ] **Step 5: Push the exact branch**

```bash
git push -u origin feat/jcm-2-api-migration
```

- [ ] **Step 6: Create one draft PR**

Use the GitHub capability to confirm no open PR already uses the head branch,
then create one draft PR:

- base: `main`;
- head: `feat/jcm-2-api-migration`;
- title: `Migrate VerCOR to Dinosaur 1.3.6 and JCM 2.0.1`;
- body sections: Summary, Upstream API Changes, Validation, Remaining Upstream
  Warnings.

- [ ] **Step 7: Inspect PR checks**

Report the PR URL and current CI state. Fix migration-owned failures that
finish during the session; otherwise leave the draft with complete local
verification evidence in its body.
