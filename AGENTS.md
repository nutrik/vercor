# Versatile Earth system COupleR (VerCOR) Development Guide

## CRITICAL: local environment rules

### Filesystem

- **NEVER** use `find /` or scan outside the project directory.
  The filesystem has millions of files and these command will hang forever.
- **Project root**: the current working directory (use `.` or relative paths)
- Use `grep` or `Grep` to search file contents — never `find` with broad paths.

## What is this?

VerCOR is a fully differentiable coupler in JAX for different Earth system models written in JAX. 

## Quick reference

- **Design document**: `DESIGN.md` (read this first)
- **Progress log**: `PROGRESS.md`

## Setup

```bash
# Activate and always use conda's virtual environment called scipy
conda activate scipy

# Run linters and formatters during development
flake8 . --count --exit-zero --max-line-length=120 --statistics
black vercor tests

# Do static type checking during development
mypy vercor tests

# Run Python tests
pytest tests/ -v -n4 --dist=loadscope --max-worker-restart=0 --durations=25 --tb=short
pytest tests/ -v -m "not slow" -v -n4 --dist=loadscope --max-worker-restart=0 --durations=25  # skip integration tests

# Do test coverage analysis
pytest --cov=vercor tests/ -v -n4 --dist=loadscope --max-worker-restart=0 --durations=25 --tb=short

# Commit changes to git only after full suite unit tests pass
git add .

# For example, after implementing module X and its unit tests:
git commit -m "Implement module X with tests"
```
---

## Orientation (read this first when starting a session)

When you start a new session, orient yourself:
1. Read `PROGRESS.md` to see what's done and what's next.
2. Run `pytest tests/ -v --fast 2>&1 | tail -20` to see current test status.
3. Pick the next failing test or unchecked item from `PROGRESS.md`.
4. When you finish a unit of work, update `PROGRESS.md` before stopping.

---

## Principles for autonomous development

### 1. Tests are everything

The test harness is the most important part of the project.
If the tests are wrong or incomplete, agents will solve the wrong problem.

**Rules:**
- Never merge or commit code that breaks existing passing tests.
- Every new module must have a corresponding test file BEFORE implementation.
  Write the test first, then make it pass.
- When you find a bug, add a test that reproduces it before fixing it.
- Tests must be nearly perfect. Invest heavily in the test harness: generate
  high-quality reference data at many parameter points, write clear
  verifiers, and watch for failure modes so you can add targeted tests.
- When a discrepancy is found, trace upstream through the pipeline to find the
  first module where things diverge. Fix there; downstream improves automatically.

### 2. Concise test output (context window hygiene)

LLMs have finite context windows. Every line of noisy test output displaces
useful information and degrades reasoning quality.

**Rules:**
- Tests print at most 5-10 lines on success, ~20 lines on failure.
- Use `pytest -q` by default. Never dump large arrays to stdout.
- Log verbose diagnostics to `test_logs/` files, not stdout.
- Pre-compute aggregate summary statistics. Print them, not raw data.
- When comparing arrays, print: max relative error, the index/value where
  it occurs, and the overall pass rate. Not the full arrays.
- Error messages should be greppable: put ERROR and the reason on one line
  so `grep ERROR logfile` works.

Good:
```
FAILED test_uv_matrix_assembly_SSTREAM.py::uvMatrixAssemblySSTREAM - max rel err 0.032% at z=1089.2
  Expected X value=8.27146287E-04, got X value=9.27244317E-04
  (23/25 quantities pass at <0.01%, 2 at <0.05%)
```

Bad:
```
FAILED - arrays not equal:
  [1.0183e-4, 1.0182e-4, 1.0181e-4, ...]  (500 more lines)
```

### 3. Fast tests to avoid time blindness

LLMs can't tell time and will happily spend hours running full test suites
instead of making progress.

**Rules:**
- Every test file has a `--fast` mode (via pytest fixture).
- `--fast` runs a deterministic ~10% subsample (e.g., `z_grid[::10]`).
- The subsample should be deterministic per-agent but cover different points
  across agents (use a hash of the agent ID or test name as seed).
- Default development cycle: run `--fast` after every change, full suite
  only before committing.

```python
@pytest.fixture
def fast_mode(request):
    return request.config.getoption("--fast", default=False)

def test_background_quantities(fast_mode, class_reference):
    z_grid = class_reference["z"]
    if fast_mode:
        z_grid = z_grid[::10]  # every 10th point
    ...
```

### 4. Keep PROGRESS.md current (agent orientation)

PROGRESS.md is the shared memory. Without it, agents waste time re-discovering what's done and what's broken.

**Rules:**
- Update `PROGRESS.md` after every meaningful unit of work.
- Check off completed items with dates.
- Note what worked, what didn't, what's blocked.
- **Record failed approaches** so they aren't re-attempted. If something does not work switch to alternative(s).
- Add new tasks discovered during implementation.
- When stuck, maintain a running doc of attempts in `PROGRESS.md`.

### 5. Prevent regressions (CI discipline)

Once the codebase grew, new features frequently broke existing functionality. 
Therefore, build a CI pipeline with strict enforcement and discipline.

**Rules:**
- Run `pytest tests/ -q --fast` before every commit.
- If anything regresses, fix it before committing. Never "fix it later."
- If a new feature requires changing behavior in an existing test, update the
  test explicitly (don't just delete or skip it).
- Track test pass rates over time in `PROGRESS.md` (e.g., "bilinear_rectilinear: 25/25,
  exchange: 18/20, couple: 142/150, etc.").

### 6. Structure work for parallelism

Parallelism is easy when there are many independent failing tests 
(each agent picks a different one), but hard when there's one giant failing task
(all agents hit the same bug and overwrite each other).

**How this applies to VerCOR:**
Easy to parallelize (many independent tasks); 
hard to parallelize (one giant task).

**Mitigation for the "one giant task" problem:** Break it into sub-tests.
Test individual equation and components separately:
- Test that interpolations are correct with dummy but physics based inputs.
- Test exchange of fields in isolation.
- Test clocks in isolation.
- Then combine. This way, multiple agents can work on different subsystems.

**Task claiming:** When working in parallel, note your task in `PROGRESS.md`
(e.g., "IN PROGRESS: diffusion.py (@agent-1)"). Check PROGRESS.md before
starting in order to avoid duplicate work.

### 7. Small, testable commits

**Rules:**
- Each git commit implements one thing (one function, one module, one bugfix).
- Each git commit passes all existing tests.
- Each git commit includes or updates tests for the new code.
- Avoid large git commits that change multiple modules at once.
- If a refactor touches many files, do it as a separate commit from features.

### 8. Document for the next session, not for users

Documentation is not a nicety; it's a critical coordination mechanism.

**Every module should have a docstring explaining:**
- What physics and numerics it implements (with equation references, papers etc.).
- What it takes as input and produces as output (types, shapes).
- Any non-obvious numerical choices (why this tolerance? why this grid size?).
- Known limitations or accuracy issues.

### 9. Specialized agent roles

Use specialized agents beyond just "write code": 
one for deduplication, one for performance, one for code quality review,
one for documentation.

**For VerCOR useful specializations:**
- **Implementer agents**: Write the module code to pass tests.
- **Test quality agent**: Reviews and improves the test harness. Adds edge
  cases, improves error messages, catches gaps in coverage.
- **Gradient validation agent**: Focused solely on testing AD correctness.
  Runs finite-difference checks for every module, every parameter.
- **Performance agent**: Profiles the code, identifies bottlenecks, optimizes
  JAX JIT compilation time, reduces memory usage.
- **Code quality agent**: Looks for duplicated code, inconsistent patterns,
  missing type hints, unclear variable names. Refactors.
- **Documentation agent**: Keeps `PROGRESS.md` and docstrings in sync with actual code.

---

## Architecture summary

The source code is a sequential pipeline of pure functions, each returning a frozen JAX PyTree.

Key principle:
**Physics related parameters/constants are JAX-traced** (for AD),
**Precision parameters are static** (control array shapes).
Never branch on Physics parameters.

---

## Coding conventions

- **Pure functions only with jax.jit decorator**. No mutable state, no global variables, no side effects.
- **Type hints** on all public functions using `jaxtyping.Float[Array, "..."]`.
- **Frozen dataclasses** for all result types, registered as JAX PyTrees.
- **No branching on traced values**. Use `jnp.where` instead of `if`. Use
  `jax.lax.cond` only when truly necessary (both branches must be same shape).
- **Units**: VerCOR natural SI units throughout (m/s, kg, s).
- **Naming**: Use snake_case for everything.
- **Docstrings**: Every public function and class must have a docstring explaining what it does, its inputs/outputs, and any non-obvious choices. Follow PEP 257 style.
- **Precision**: globaly define JAX precision policy (e.g., `jax.config.update("jax_enable_x64", True)`) and stick to it. Don't mix dtypes within the entire code. 

---

## Module dependency order

During implementation, create and update module dependency order as a numbered list in `DEPENDENCIES.md` file, such as:
1. constants.py (no deps)
2. parameters.py (no deps)
3. grid.py (1, 2)
etc.

---

## Critical rules to prevent physics bugs

These address the main risks unique to reimplementing a software project from NumPy to JAX. 

### Never add fudge factors

If a test fails with 0.2% error, there is a term that is wrong -- a sign
error, a missing factor, a wrong index. Find the actual bug. Do NOT multiply
by 1.002 to make the test pass. If you are tempted, it means you haven't
isolated the source of the discrepancy.

### Test at many parameter points, not just fiducial

A fudge factor or bug that cancels at fiducial coupler setup will show up when
parameters change. 

### Test intermediate quantities, not just final output

Generate and store intermediate outputs at every stage.

### Test gradients from the bottom up

Build confidence layer by layer. If the final output gradients are wrong,
test the gradients of the previous module's outputs with respect to its inputs. 
Then test the previous module's gradients, and so on.

If step N fails, the bug is between step N-1 and N. Also: forward mode
(`jax.jvp`) must match reverse mode (`jax.grad`). Disagreement pinpoints
a `custom_vjp` bug.

---

## Coding best practices 

Follow these principles to keep the source code maintainable, readable, and long-term successful.

### 1. Code Organization

- Many small files over few large files
- High cohesion, low coupling
- 200-400 lines typical, 800 max per file
- Organize by feature/domain, not by type

### 2. Code Style

- Use meaningful variable, function and class names (prioritizes readability)
- Follow PEP8 guidelines for Python
- Write docstrings for all public modules, functions, classes, and methods following PEP 257
- Functionality should only be added when deemed necessary, following YAGNI (You Aren't Gonna Need It) principle
- Follow DRY (Don't Repeat Yourself) principle
- Follow SOLID principles: Single Responsibility, Open/Closed, Liskov Substitution, Interface Segregation, and Dependency Inversion
- Proper error handling with try/except blocks and informative error messages

### 3. Testing

- TDD: Write tests first
- Aim for 80%+ minimum code coverage
- Use pytest for testing
- Test edge cases and error conditions, not just happy paths
- Use descriptive test names that clearly indicate what is being tested
- Organize tests in a separate directory (e.g., `tests/`) and mirror the structure of the main codebase
- Use fixtures for setup and teardown of test environments when necessary
