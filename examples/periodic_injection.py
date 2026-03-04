"""Periodic seed injection example for liq-gp.

Demonstrates how to periodically re-inject seed programs during evolution
to maintain diversity and guide search toward known-good structures.

Three injection methods are compared:
  1. Baseline: no injection (standard evolution)
  2. Direct injection: re-inject the seed program as-is every N generations
  3. Variation injection: apply GP operators to seeds before injecting
  4. Ramped injection: inject fresh random programs (no seeds needed)

Usage:
    python examples/periodic_injection.py
"""

from __future__ import annotations

import numpy as np

from liq.gp import (
    FitnessResult,
    FunctionNode,
    GenerationStats,
    GPConfig,
    PrimitiveRegistry,
    Program,
    SeedInjectionConfig,
    Series,
    TerminalNode,
    evaluate,
    evolve,
)

# ---------------------------------------------------------------------------
# 1. Build a primitive registry
# ---------------------------------------------------------------------------

registry = PrimitiveRegistry()

registry.register("x", lambda: None, input_types=(), output_type=Series)
registry.register(
    "add",
    lambda a, b: a + b,
    category="arithmetic",
    input_types=(Series, Series),
    output_type=Series,
)
registry.register(
    "sub",
    lambda a, b: a - b,
    category="arithmetic",
    input_types=(Series, Series),
    output_type=Series,
)
registry.register(
    "mul",
    lambda a, b: a * b,
    category="arithmetic",
    input_types=(Series, Series),
    output_type=Series,
)
registry.register(
    "neg",
    lambda a: -a,
    category="arithmetic",
    input_types=(Series,),
    output_type=Series,
)


# ---------------------------------------------------------------------------
# 2. Define fitness evaluator
# ---------------------------------------------------------------------------
# Target: y = x^3 - 2x^2 + x  (a harder target than quadratic)


class RegressionEvaluator:
    """Single-objective evaluator: negative MSE against the target function."""

    def __init__(self, target: np.ndarray) -> None:
        self.target = target

    def evaluate(
        self,
        programs: list[Program],
        context: dict[str, np.ndarray],
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
# 3. Prepare data and hand-craft a seed program
# ---------------------------------------------------------------------------

x_train = np.linspace(-3, 3, 200)
y_train = x_train**3 - 2 * x_train**2 + x_train
context = {"x": x_train}
evaluator = RegressionEvaluator(y_train)

# Build a rough approximation as a seed: x * x  (captures the quadratic shape)
# This is intentionally imperfect — periodic injection will keep nudging search.
mul_info = registry.get("mul")
seed_program = FunctionNode(
    primitive=mul_info,
    children=(
        TerminalNode(name="x", output_type=Series),
        TerminalNode(name="x", output_type=Series),
    ),
)
print(f"Seed program: mul(x, x)  size={seed_program.size}  depth={seed_program.depth}")
print()


# ---------------------------------------------------------------------------
# 4. Generation callback that reports injection events
# ---------------------------------------------------------------------------


def make_callback() -> callable:
    """Create a callback that prints injection events and periodic progress."""

    def on_gen(stats: GenerationStats) -> None:
        parts = [
            f"  Gen {stats.generation:3d}",
            f"best_mse={-stats.best_fitness[0]:.4f}",
            f"size={stats.best_program_size}",
            f"unique={stats.unique_semantics_ratio:.0%}",
        ]
        if stats.injected_count > 0:
            parts.append(f"INJECTED={stats.injected_count}")
        # Print every 5 generations or when injection happens
        if stats.generation % 5 == 0 or stats.injected_count > 0:
            print("  ".join(parts))

    return on_gen


# ---------------------------------------------------------------------------
# 5. Shared config parameters
# ---------------------------------------------------------------------------

SHARED = {
    "population_size": 200,
    "generations": 30,
    "max_depth": 6,
    "crossover_rate": 0.7,
    "subtree_mutation_rate": 0.1,
    "point_mutation_rate": 0.1,
    "parameter_mutation_rate": 0.0,
    "hoist_mutation_rate": 0.1,
    "tournament_size": 5,
    "elitism_count": 3,
    "constant_opt_enabled": False,
}


# ---------------------------------------------------------------------------
# 6. Baseline: no injection
# ---------------------------------------------------------------------------

print("=" * 65)
print("Baseline: standard evolution (no periodic injection)")
print("=" * 65)

config_baseline = GPConfig(seed=42, **SHARED)

result_baseline = evolve(
    registry,
    config_baseline,
    evaluator,
    context,
    seed_programs=[seed_program],
    callback=make_callback(),
)
fit_baseline = result_baseline.fitness_history[-1].best_fitness[0]


# ---------------------------------------------------------------------------
# 7. Direct injection: re-inject seed as-is every 5 generations
# ---------------------------------------------------------------------------

print()
print("=" * 65)
print("Direct injection: re-inject seed every 5 generations (count=2)")
print("=" * 65)

config_direct = GPConfig(
    seed=42,
    seed_injection=SeedInjectionConfig(
        interval=5,
        count=2,
        method="direct",
    ),
    **SHARED,
)

result_direct = evolve(
    registry,
    config_direct,
    evaluator,
    context,
    seed_programs=[seed_program],
    callback=make_callback(),
)
fit_direct = result_direct.fitness_history[-1].best_fitness[0]


# ---------------------------------------------------------------------------
# 8. Variation injection: mutated seeds every 5 generations
# ---------------------------------------------------------------------------

print()
print("=" * 65)
print("Variation injection: mutated seeds every 5 gens (count=3)")
print("=" * 65)

config_variation = GPConfig(
    seed=42,
    seed_injection=SeedInjectionConfig(
        interval=5,
        count=3,
        method="variation",
    ),
    **SHARED,
)

result_variation = evolve(
    registry,
    config_variation,
    evaluator,
    context,
    seed_programs=[seed_program],
    callback=make_callback(),
)
fit_variation = result_variation.fitness_history[-1].best_fitness[0]


# ---------------------------------------------------------------------------
# 9. Ramped injection: fresh random programs every 10 generations
# ---------------------------------------------------------------------------

print()
print("=" * 65)
print("Ramped injection: fresh random programs every 10 gens (count=5)")
print("=" * 65)

config_ramped = GPConfig(
    seed=42,
    seed_injection=SeedInjectionConfig(
        interval=10,
        count=5,
        method="ramped",
    ),
    **SHARED,
)

# Ramped method doesn't need seed_programs for injection,
# but we still pass them for initialization seeding.
result_ramped = evolve(
    registry,
    config_ramped,
    evaluator,
    context,
    seed_programs=[seed_program],
    callback=make_callback(),
)
fit_ramped = result_ramped.fitness_history[-1].best_fitness[0]


# ---------------------------------------------------------------------------
# 10. Aggressive injection: every generation, multiple programs
# ---------------------------------------------------------------------------

print()
print("=" * 65)
print("Aggressive variation: inject 5 programs every generation")
print("=" * 65)

config_aggressive = GPConfig(
    seed=42,
    seed_injection=SeedInjectionConfig(
        interval=1,
        count=5,
        method="variation",
    ),
    **SHARED,
)

result_aggressive = evolve(
    registry,
    config_aggressive,
    evaluator,
    context,
    seed_programs=[seed_program],
    callback=make_callback(),
)
fit_aggressive = result_aggressive.fitness_history[-1].best_fitness[0]


# ---------------------------------------------------------------------------
# 11. Summary
# ---------------------------------------------------------------------------

print()
print("=" * 65)
print("Summary — final best MSE (lower is better)")
print("=" * 65)

results = [
    ("Baseline (no injection)", fit_baseline, result_baseline),
    ("Direct every 5 gens (count=2)", fit_direct, result_direct),
    ("Variation every 5 gens (count=3)", fit_variation, result_variation),
    ("Ramped every 10 gens (count=5)", fit_ramped, result_ramped),
    ("Aggressive variation (count=5, every gen)", fit_aggressive, result_aggressive),
]

for label, fit, res in results:
    total_injected = sum(s.injected_count for s in res.fitness_history)
    print(
        f"  {label:<42s}  "
        f"MSE={-fit:10.4f}  "
        f"size={res.best_program.size:3d}  "
        f"total_injected={total_injected}"
    )

print()
print("Note: Results depend on the random seed. Periodic injection helps")
print("maintain diversity and can prevent premature convergence, especially")
print("on harder problems or with longer runs.")
