# liq-gp

General-purpose genetic programming engine for the LIQ Stack.

`liq-gp` provides typed program evolution with modern SOTA techniques (2024-2026):
joint structure and constant optimization, multi-objective Pareto selection,
semantic diversity management, and algebraic simplification.

It is a **foundation library** consumed by domain-specific libraries (e.g.,
`liq-evolution` for trading strategy synthesis). It has **no domain knowledge** --
all domain concerns are injected via typed primitive registries and fitness
evaluators provided by the consumer.

## Design Goals

- **Domain-agnostic** -- zero domain knowledge; all domain concerns injected via
  primitives and fitness evaluators
- **Programs as parametric models** -- evolved trees define functional form;
  constants within them are numerically optimized
- **Multi-objective by default** -- N objectives natively; single-objective is a
  special case
- **Deterministic reproducibility** -- same seed + config + data = identical
  evolution trajectory
- **Type-safe programs** -- the type system prevents construction of invalid trees
- **Evaluator-owned parallelism** -- liq-gp passes the full population to your
  evaluator; you choose the execution strategy (threads, processes, GPU, etc.)

## Installation

```bash
uv venv
uv pip install -e ".[dev]"
```

## Core Concepts

### Programs as Trees

Programs are immutable trees composed of four node types:

- **TerminalNode** -- reads a named array from the evaluation context (arity 0)
- **ConstantNode** -- holds a literal float value, broadcast to array length (arity 0)
- **FunctionNode** -- applies a primitive function to child outputs (arity >= 1)
- **ParameterizedNode** -- like FunctionNode, but with evolvable parameters (e.g., a
  lookback period)

### Evaluation Model

Programs evaluate bottom-up on a **context** -- a `dict[str, np.ndarray]` mapping
names to NumPy arrays. Each node type evaluates as follows:

- `TerminalNode("x")` -- returns `context["x"]` directly (the registered callable
  is a placeholder and is never invoked)
- `ConstantNode(3.14)` -- returns `np.full(n, 3.14)` broadcast to context length
- `FunctionNode(add, [child_a, child_b])` -- evaluates children, then calls
  `add(child_a_result, child_b_result)`
- `ParameterizedNode(sma, [child], params={"period": 10})` -- evaluates children,
  then calls `sma(child_result, period=10)`

All primitives operate on NumPy arrays and return NumPy arrays.

### Type System

Four built-in types enforce valid tree construction at the AST level:

| Type | Description |
|------|-------------|
| `Series` | Array of float values (one per observation) |
| `BoolSeries` | Boolean values encoded as float64 (1.0/0.0) |
| `Scalar` | Single float value |
| `Int` | Integer value (for periods, shifts) |

Custom types can be registered via `GPType.register_type()`.

### Evolution Loop

Each generation follows this pipeline:

1. **Evaluate** -- pass full population to your `FitnessEvaluator`
2. **Parsimony pressure** -- penalize bloat (lexicographic, Pareto, or linear)
3. **Statistics** -- compute generation stats, invoke callback
4. **Early stopping** -- check stall count against patience
5. **Elitism** -- preserve top individuals unchanged
6. **Selection** -- tournament or NSGA-II
7. **Variation** -- crossover and mutation operators
8. **Simplification** -- algebraic rewrite rules (optional)
9. **Constant optimization** -- Nelder-Mead on top-K programs (optional)
10. **Seed injection** -- replace worst individuals with seed-derived programs (optional)
11. **Semantic deduplication** -- remove output-equivalent programs

### Parallelism

liq-gp does **not** manage parallel workers internally. The engine calls
`evaluator.evaluate(programs, context)` with the complete population list.
Your evaluator chooses the execution strategy:

- **Sequential** -- evaluate in a loop (simplest, no overhead)
- **Threaded** -- `concurrent.futures.ThreadPoolExecutor` (good for NumPy-heavy work
  that releases the GIL)
- **Multiprocessing** -- joblib, Ray, or `ProcessPoolExecutor` (true CPU parallelism)
- **GPU** -- batch-evaluate on device

This design avoids the memory pitfalls of library-managed parallelism (CoW
shattering, recursive tree serialization overhead, worker memory leaks) and lets
you match the strategy to your workload.

## Quickstart

```python
import numpy as np
from liq.gp import (
    GPConfig, PrimitiveRegistry, Series,
    FitnessResult, evaluate, evolve,
)

# 1. Build a primitive registry
registry = PrimitiveRegistry()

# Terminals read from the evaluation context by name.
# The callable is a placeholder -- never invoked during evaluation.
registry.register("x", lambda: None, input_types=(), output_type=Series)

# Functions operate on child outputs (NumPy arrays in, NumPy array out).
registry.register("add", lambda a, b: a + b,
                  category="arithmetic",
                  input_types=(Series, Series), output_type=Series)
registry.register("mul", lambda a, b: a * b,
                  category="arithmetic",
                  input_types=(Series, Series), output_type=Series)
registry.register("neg", lambda a: -a,
                  category="arithmetic",
                  input_types=(Series,), output_type=Series)

# 2. Implement a fitness evaluator
class RegressionEvaluator:
    def __init__(self, target):
        self.target = target

    def evaluate(self, programs, context):
        results = []
        for prog in programs:
            try:
                output = evaluate(prog, context)
                mse = float(np.nanmean((output - self.target) ** 2))
                results.append(FitnessResult(objectives=(-mse,)))
            except Exception:
                results.append(FitnessResult(objectives=(-1e10,)))
        return results

# 3. Prepare data: y = x^2 + 2x + 1
x = np.linspace(-5, 5, 200)
target = x ** 2 + 2 * x + 1
context = {"x": x}

# 4. Configure and evolve
config = GPConfig(population_size=300, generations=40, seed=42)
result = evolve(registry=registry, config=config,
                evaluator=RegressionEvaluator(target), context=context)

# 5. Use the result
best = result.best_program
print(f"Best program size: {best.size} nodes")
prediction = evaluate(best, {"x": np.linspace(-10, 10, 50)})
```

See [`examples/symbolic_regression.py`](examples/symbolic_regression.py) for a
complete runnable version with callbacks, serialization, and test-set evaluation.

## Multi-Objective Evolution (NSGA-II)

```python
from liq.gp import FitnessConfig

# FitnessConfig must be embedded in GPConfig so that NSGA-II validation
# sees the correct number of objectives at construction time.
fitness_config = FitnessConfig(
    objectives=["accuracy", "simplicity"],
    objective_directions=["maximize", "maximize"],
)

config = GPConfig(
    population_size=500,
    generations=100,
    selection_mode="nsga2",
    parsimony_mode="pareto",
    fitness=fitness_config,
)

result = evolve(
    registry=registry,
    config=config,
    evaluator=my_evaluator,  # must return 2-objective FitnessResults
    context=context,
)

# Access the Pareto front
for program in result.pareto_front:
    print(f"  size={program.size}")
```

See [`examples/multi_objective.py`](examples/multi_objective.py) for a complete
example with NSGA-II, parameterized primitives, conditional logic (BoolSeries),
protected division, and early stopping.

## Population Seeding (Warm Start)

Seed the initial population with known programs -- from a previous run, manual
construction, or deserialized JSON:

```python
# Warm start from a previous run's best program
result1 = evolve(registry=registry, config=config1, evaluator=evaluator, context=context)

result2 = evolve(
    registry=registry,
    config=config2,
    evaluator=evaluator,
    context=context,
    seed_programs=[result1.best_program],  # or result1.pareto_front
)

# Hand-craft a seed program
from liq.gp import FunctionNode, TerminalNode, ConstantNode
manual_seed = FunctionNode(
    primitive=registry.get("add"),
    children=(TerminalNode(name="x", output_type=Series), ConstantNode(value=1.0)),
)
result3 = evolve(..., seed_programs=[manual_seed])
```

When fewer seeds than `population_size` are provided, seeds are placed directly and
remaining slots are filled by applying variation operators (crossover, mutation) to
the seeds.  When exactly `population_size` seeds are given, they are used as-is.

Use `validate_seed_programs()` to check seeds before evolution:

```python
from liq.gp import validate_seed_programs, EvolutionError

try:
    validate_seed_programs(seeds, config, registry=registry)
except EvolutionError as e:
    print(f"Invalid seeds: {e}")
```

See [`examples/warm_start.py`](examples/warm_start.py) for a complete example with
cold start, warm start, manual seeding, and serialize/deserialize round-trip.

## Periodic Seed Injection

During evolution, seed injection periodically replaces worst-fitness individuals
with new programs to maintain diversity and counteract stagnation:

```python
from liq.gp import GPConfig, SeedInjectionConfig

config = GPConfig(
    population_size=300,
    generations=100,
    seed_injection=SeedInjectionConfig(
        interval=10,         # inject every 10 generations
        count=5,             # replace 5 worst individuals per cycle
        method="variation",  # "direct", "variation", or "ramped"
    ),
)

result = evolve(
    registry=registry,
    config=config,
    evaluator=evaluator,
    context=context,
    seed_programs=[known_good_program],  # required for "direct" and "variation"
)

# Track injection events via GenerationStats
for stats in result.fitness_history:
    if stats.injected_count > 0:
        print(f"Gen {stats.generation}: injected {stats.injected_count} programs")
```

Three injection methods:

| Method | Description | Seeds Required? |
|--------|-------------|-----------------|
| `"direct"` | Re-inject seed programs as-is (round-robin cycling) | Yes |
| `"variation"` | Apply GP operators to seeds to create diverse offspring | Yes |
| `"ramped"` | Generate fresh random programs (ramped half-and-half) | No |

In NSGA-II mode, "worst" individuals are identified by Pareto rank and crowding
distance, consistent with the multi-objective selection semantics.

See [`examples/periodic_injection.py`](examples/periodic_injection.py) for a
complete comparison of all injection methods.

## Callbacks and Progress Tracking

Pass a callback to `evolve()` to receive `GenerationStats` each generation:

```python
from liq.gp import GenerationStats

def on_generation(stats: GenerationStats) -> None:
    print(
        f"Gen {stats.generation:3d}  "
        f"best={stats.best_fitness[0]:.6f}  "
        f"mean_size={stats.mean_program_size:.1f}  "
        f"unique={stats.unique_semantics_ratio:.0%}  "
        f"pareto={stats.pareto_front_size}"
    )

result = evolve(..., callback=on_generation)
```

## Serialization

Programs serialize to JSON-compatible dicts with schema versioning:

```python
from liq.gp import serialize, deserialize

data = serialize(program)                  # -> dict with schema_version
restored = deserialize(data, registry)     # requires registry for primitive lookup
```

## Configuration Reference

### GPConfig

| Parameter | Default | Description |
|-----------|---------|-------------|
| `population_size` | 300 | Population size (>= 10) |
| `max_depth` | 8 | Maximum tree depth (>= 2) |
| `max_size` | None | Optional maximum tree size |
| `generations` | 50 | Number of generations |
| `seed` | 42 | Random seed for reproducibility |
| `crossover_rate` | 0.7 | Subtree crossover probability |
| `subtree_mutation_rate` | 0.1 | Subtree mutation probability |
| `point_mutation_rate` | 0.1 | Point mutation probability |
| `parameter_mutation_rate` | 0.05 | Parameter mutation probability |
| `hoist_mutation_rate` | 0.05 | Hoist mutation probability |
| `tournament_size` | 5 | Tournament selection size |
| `elitism_count` | 5 | Number of elites preserved |
| `selection_mode` | "tournament" | "tournament" or "nsga2" |
| `parsimony_mode` | "lexicographic" | "lexicographic", "pareto", "linear", or "disabled" |
| `parsimony_coefficient` | 0.001 | Coefficient for linear parsimony |
| `constant_opt_enabled` | True | Enable constant optimization |
| `constant_opt_top_k` | 0.1 | Fraction of population to optimize |
| `constant_opt_max_iter` | 50 | Max Nelder-Mead iterations |
| `constant_opt_max_time_seconds` | 1.0 | Wall-clock time limit per program |
| `simplification_enabled` | True | Enable algebraic simplification |
| `semantic_ref_size` | 50 | Reference dataset size for fingerprinting |
| `semantic_precision` | 6 | Decimal places for fingerprint rounding |
| `early_stop_patience` | None | Generations without improvement before stopping |
| `early_stop_threshold` | 1e-6 | Minimum improvement to reset patience |
| `seed_injection` | None | Optional `SeedInjectionConfig` for periodic injection |

Operator rates must sum to 1.0. All rates must be non-negative.

### FitnessConfig

| Parameter | Default | Description |
|-----------|---------|-------------|
| `objectives` | ["fitness"] | Objective names |
| `objective_directions` | ["maximize"] | "maximize" or "minimize" per objective |
| `batch_size` | None | Optional mini-batch evaluation size |
| `full_eval_interval` | 10 | Full evaluation every N generations |

NSGA-II requires >= 2 objectives. Pareto parsimony requires NSGA-II selection.

## Public API

All public items are importable from `liq.gp`:

```python
from liq.gp import (
    # Configuration
    GPConfig, FitnessConfig, SeedInjectionConfig,
    # Type system
    Series, BoolSeries, Scalar, Int,
    # Data types
    ParamSpec, FitnessResult, GenerationStats, EvolutionResult,
    # Primitives
    PrimitiveRegistry, PrimitiveInfo,
    # AST nodes
    Program, TerminalNode, ConstantNode, FunctionNode, ParameterizedNode,
    # Functions
    evaluate, simplify, optimize_constants, serialize, deserialize,
    evolve, validate_seed_programs, initialize_seeded_population, inject_seeds,
    # Protocols
    FitnessEvaluator,
    # Errors
    GPError, PrimitiveError, TypeCheckError,
    EvaluationError, EvolutionError, SerializationError, ConfigurationError,
)
```

## Error Handling

All exceptions inherit from `GPError`:

| Exception | When Raised | Fatal? |
|-----------|-------------|--------|
| `PrimitiveError` | Unknown/duplicate primitive, arity mismatch | Yes |
| `TypeCheckError` | Type mismatch at node construction | Yes |
| `EvaluationError` | Failure during program evaluation | Yes |
| `EvolutionError` | Invalid seed programs, impossible tree construction | Yes |
| `SerializationError` | Schema mismatch, missing primitive, corrupt data | Yes |
| `ConfigurationError` | Invalid config values at construction | Yes |
| `ConstantOptError` | Failure during constant optimization | No -- logged, original constants kept |

## Ecosystem

```
liq-gp          (general-purpose GP engine, domain-agnostic)
    |
    v
liq-evolution   (trading strategy evolution)
    |
    +--> liq-ta       (indicator computation)
    +--> liq-runner   (rolling backtests)
    +--> liq-risk     (position sizing)
    +--> liq-sim      (execution simulation)
```

`liq-gp` owns: AST, evaluation, evolution loop, selection, operators,
simplification, constant optimization, semantic dedup, serialization.

`liq-evolution` owns: primitive registry population with indicators, fitness
evaluation via backtesting, strategy adaptation.

## Development

```bash
uv run pytest                         # run all tests (730+ tests, 95%+ coverage)
uv run pytest -m "not slow"           # skip performance tests
uv run ruff check src/ tests/         # lint
uv run ruff format src/ tests/        # format
uv run ty check src/                  # type check
```

## Operational Failure Mapping (Stage 6 hardening)

- `selection_mode="lexicase"` without valid per-individual
  `METADATA_KEY_SLICE_SCORES` fails fast with `ValueError`.
- `metadata["raw_objectives"]` shape and `slice_scores` size are validated in the
  selector before tournament-like selection is attempted.
- `parsimony_mode="disabled"` is required for some external evaluators that emit
  intentionally shaped `raw_objectives` used by lexicase key alignment.
- `QDArchive.sample` with `coverage_weight=0` selects by best first objective;
  with `coverage_weight=1` it prioritizes underfilled bins.
- Invalid `coverage_weight`, selection mode, or archive shape violations are
  surfaced at configuration/evolution boundaries before mutation begins.

## License

MIT
