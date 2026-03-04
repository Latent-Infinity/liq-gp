"""Warm-start example for liq-gp: seeding evolution from previous results.

Demonstrates three seeding patterns:
  1. Cold start: normal random initialization
  2. Warm start from best_program: resume from the previous best solution
  3. Warm start from pareto_front: resume from an entire Pareto front
  4. Manual seed: hand-craft a program and use it to bootstrap evolution

Usage:
    python examples/warm_start.py
"""

from __future__ import annotations

import numpy as np

from liq.gp import (
    ConstantNode,
    FitnessResult,
    FunctionNode,
    GenerationStats,
    GPConfig,
    PrimitiveRegistry,
    Program,
    Series,
    TerminalNode,
    deserialize,
    evaluate,
    evolve,
    serialize,
)

# ---------------------------------------------------------------------------
# 1. Build a primitive registry
# ---------------------------------------------------------------------------

registry = PrimitiveRegistry()

# Terminals read arrays from the evaluation context by name.
registry.register("x", lambda: None, input_types=(), output_type=Series)

# Functions operate on NumPy arrays.
registry.register(
    "add", lambda a, b: a + b,
    category="arithmetic", input_types=(Series, Series), output_type=Series,
)
registry.register(
    "mul", lambda a, b: a * b,
    category="arithmetic", input_types=(Series, Series), output_type=Series,
)
registry.register(
    "neg", lambda a: -a,
    category="arithmetic", input_types=(Series,), output_type=Series,
)


# ---------------------------------------------------------------------------
# 2. Define fitness evaluator
# ---------------------------------------------------------------------------
# Target: y = x^2 + 2x + 1 = (x + 1)^2


class RegressionEvaluator:
    """Single-objective evaluator: negative MSE against the target function."""

    def __init__(self, target: np.ndarray) -> None:
        self.target = target

    def evaluate(
        self, programs: list[Program], context: dict[str, np.ndarray],
    ) -> list[FitnessResult]:
        results: list[FitnessResult] = []
        for prog in programs:
            try:
                output = evaluate(prog, context)
                mse = float(np.nanmean((output - self.target) ** 2))
                results.append(FitnessResult(objectives=(-mse,)))
            except Exception:
                results.append(FitnessResult(objectives=(-1e10,)))
        return results


# ---------------------------------------------------------------------------
# 3. Prepare data
# ---------------------------------------------------------------------------

x_train = np.linspace(-5, 5, 200)
y_train = x_train**2 + 2 * x_train + 1
context = {"x": x_train}
evaluator = RegressionEvaluator(y_train)


def on_gen(stats: GenerationStats) -> None:
    """Per-generation callback for progress reporting."""
    print(
        f"  Gen {stats.generation:3d}  "
        f"best_fitness={stats.best_fitness[0]:.6f}  "
        f"best_size={stats.best_program_size}  "
        f"mean_size={stats.mean_program_size:.1f}"
    )


# ---------------------------------------------------------------------------
# 4. Stage 1: Cold start (random initialization)
# ---------------------------------------------------------------------------

print("=" * 60)
print("Stage 1: Cold start (random initialization)")
print("=" * 60)

config1 = GPConfig(
    population_size=200,
    generations=20,
    max_depth=6,
    seed=42,
    constant_opt_enabled=False,
)

result1 = evolve(registry, config1, evaluator, context, callback=on_gen)

best1 = result1.best_program
fit1 = result1.fitness_history[-1].best_fitness[0]
print(f"\nCold start result: fitness={fit1:.6f}, size={best1.size} nodes")

# ---------------------------------------------------------------------------
# 5. Warm start from best_program
# ---------------------------------------------------------------------------

print()
print("=" * 60)
print("Warm start from best_program")
print("=" * 60)

config2 = GPConfig(
    population_size=200,
    generations=20,
    max_depth=6,
    seed=99,  # Different RNG seed for variety
    constant_opt_enabled=False,
)

result2 = evolve(
    registry, config2, evaluator, context,
    seed_programs=[result1.best_program],
    callback=on_gen,
)

best2 = result2.best_program
fit2 = result2.fitness_history[-1].best_fitness[0]
print(f"\nWarm start (best) result: fitness={fit2:.6f}, size={best2.size} nodes")
print(f"Improvement over cold start: {fit2 - fit1:.6f}")

# ---------------------------------------------------------------------------
# 6. Stage 3: Warm start from Pareto front
# ---------------------------------------------------------------------------

print()
print("=" * 60)
print("Stage 3: Warm start from Pareto front")
print("=" * 60)
print(f"Seeding with {len(result1.pareto_front)} programs from Stage 1 Pareto front")

config3 = GPConfig(
    population_size=200,
    generations=20,
    max_depth=6,
    seed=77,
    constant_opt_enabled=False,
)

result3 = evolve(
    registry, config3, evaluator, context,
    seed_programs=result1.pareto_front,
    callback=on_gen,
)

best3 = result3.best_program
fit3 = result3.fitness_history[-1].best_fitness[0]
print(f"\nWarm start (Pareto) result: fitness={fit3:.6f}, size={best3.size} nodes")

# ---------------------------------------------------------------------------
# 7. Stage 4: Manual seed — inject a hand-crafted program
# ---------------------------------------------------------------------------

print()
print("=" * 60)
print("Stage 4: Manual seed (hand-crafted program)")
print("=" * 60)

# Build x + 1 manually: add(x, 1.0)
add_info = registry.get("add")
manual_seed = FunctionNode(
    primitive=add_info,
    children=(
        TerminalNode(name="x", output_type=Series),
        ConstantNode(value=1.0),
    ),
)
print(f"Manual seed: add(x, 1.0)  (size={manual_seed.size}, depth={manual_seed.depth})")

config4 = GPConfig(
    population_size=200,
    generations=20,
    max_depth=6,
    seed=55,
    constant_opt_enabled=False,
)

result4 = evolve(
    registry, config4, evaluator, context,
    seed_programs=[manual_seed],
    callback=on_gen,
)

best4 = result4.best_program
fit4 = result4.fitness_history[-1].best_fitness[0]
print(f"\nManual seed result: fitness={fit4:.6f}, size={best4.size} nodes")

# ---------------------------------------------------------------------------
# 8. Stage 5: Serialize, deserialize, and resume
# ---------------------------------------------------------------------------

print()
print("=" * 60)
print("Stage 5: Serialize -> Deserialize -> Resume")
print("=" * 60)

# Serialize the best program from Stage 1
data = serialize(result1.best_program)
print(f"Serialized best program to dict (schema_version={data['schema_version']})")

# Deserialize it back
restored = deserialize(data, registry)
print(f"Deserialized program: size={restored.size}")

# Use the deserialized program as a seed
config5 = GPConfig(
    population_size=200,
    generations=20,
    max_depth=6,
    seed=33,
    constant_opt_enabled=False,
)

result5 = evolve(
    registry, config5, evaluator, context,
    seed_programs=[restored],
    callback=on_gen,
)

best5 = result5.best_program
fit5 = result5.fitness_history[-1].best_fitness[0]
print(f"\nResume from deserialized: fitness={fit5:.6f}, size={best5.size} nodes")

# ---------------------------------------------------------------------------
# 9. Summary
# ---------------------------------------------------------------------------

print()
print("=" * 60)
print("Summary")
print("=" * 60)
print(f"  Stage 1 (cold start):       fitness={fit1:.6f}  size={best1.size}")
print(f"  Warm from best:             fitness={fit2:.6f}  size={best2.size}")
print(f"  Stage 3 (warm from Pareto): fitness={fit3:.6f}  size={best3.size}")
print(f"  Stage 4 (manual seed):      fitness={fit4:.6f}  size={best4.size}")
print(f"  Stage 5 (deserialized):     fitness={fit5:.6f}  size={best5.size}")
