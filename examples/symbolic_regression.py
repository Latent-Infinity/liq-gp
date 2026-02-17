"""Symbolic regression example for liq-gp.

Discovers the function y = x^2 + 2x + 1 from data using genetic programming.

Usage:
    python examples/symbolic_regression.py
"""

from __future__ import annotations

import numpy as np

from liq.gp import (
    FitnessResult,
    GenerationStats,
    GPConfig,
    PrimitiveRegistry,
    Program,
    Series,
    deserialize,
    evaluate,
    evolve,
    serialize,
)

# ---------------------------------------------------------------------------
# 1. Build a primitive registry
# ---------------------------------------------------------------------------
# The registry defines the building blocks the GP engine can use to construct
# programs. Each primitive has a name, input/output types, and a callable.


def build_registry() -> PrimitiveRegistry:
    """Create a registry with arithmetic primitives."""
    reg = PrimitiveRegistry()

    # Terminals (arity 0) -- read values from the evaluation context by name.
    # The callable is a placeholder (never invoked during evaluation);
    # TerminalNode evaluation does context[node.name] directly.
    reg.register("x", lambda: None, input_types=(), output_type=Series)

    # Functions (arity > 0) -- operate on child outputs.
    # Each callable receives NumPy arrays and returns a NumPy array.
    reg.register(
        "add",
        lambda a, b: a + b,
        category="arithmetic",
        input_types=(Series, Series),
        output_type=Series,
    )
    reg.register(
        "sub",
        lambda a, b: a - b,
        category="arithmetic",
        input_types=(Series, Series),
        output_type=Series,
    )
    reg.register(
        "mul",
        lambda a, b: a * b,
        category="arithmetic",
        input_types=(Series, Series),
        output_type=Series,
    )
    reg.register(
        "neg",
        lambda a: -a,
        category="arithmetic",
        input_types=(Series,),
        output_type=Series,
    )

    return reg


# ---------------------------------------------------------------------------
# 2. Implement a fitness evaluator
# ---------------------------------------------------------------------------
# The evaluator receives a batch of programs and an evaluation context
# (dict of NumPy arrays). It returns one FitnessResult per program.
#
# liq-gp calls this once per generation with the full population.
# If you need parallel evaluation, implement it inside your evaluator
# (e.g. using joblib, Ray, or multiprocessing).


class RegressionEvaluator:
    """Evaluate programs against a target function using negative MSE."""

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
                # Negative MSE because liq-gp maximizes by default
                results.append(FitnessResult(objectives=(-mse,)))
            except Exception:
                # Assign worst possible fitness to broken programs
                results.append(FitnessResult(objectives=(-1e10,)))
        return results


# ---------------------------------------------------------------------------
# 3. Generation callback (optional)
# ---------------------------------------------------------------------------


def on_generation(stats: GenerationStats) -> None:
    """Print progress every 5 generations."""
    if stats.generation % 5 == 0 or stats.generation == 0:
        neg_mse = stats.best_fitness[0]
        print(
            f"  Gen {stats.generation:3d}  "
            f"best_mse={-neg_mse:.6f}  "
            f"mean_size={stats.mean_program_size:.1f}  "
            f"unique={stats.unique_semantics_ratio:.0%}"
        )


# ---------------------------------------------------------------------------
# 4. Run evolution
# ---------------------------------------------------------------------------


def main() -> None:
    # --- Data: y = x^2 + 2x + 1 ---
    x_train = np.linspace(-5, 5, 200)
    y_train = x_train**2 + 2 * x_train + 1
    context = {"x": x_train}

    registry = build_registry()
    evaluator = RegressionEvaluator(y_train)

    config = GPConfig(
        population_size=300,
        max_depth=6,
        generations=40,
        seed=42,
        # Operator rates (must sum to 1.0)
        crossover_rate=0.7,
        subtree_mutation_rate=0.1,
        point_mutation_rate=0.1,
        parameter_mutation_rate=0.0,
        hoist_mutation_rate=0.1,
        # Selection
        tournament_size=5,
        elitism_count=5,
        # Enable built-in optimizations
        constant_opt_enabled=True,
        simplification_enabled=True,
        semantic_dedup_enabled=True,
    )

    print("Evolving programs to fit y = x^2 + 2x + 1 ...")
    print()

    result = evolve(
        registry=registry,
        config=config,
        evaluator=evaluator,
        context=context,
        callback=on_generation,
    )

    # --- Results ---
    best = result.best_program
    neg_mse = result.fitness_history[-1].best_fitness[0]

    print()
    print(f"Best program size: {best.size} nodes")
    print(f"Best MSE: {-neg_mse:.8f}")
    print(f"Generations run: {len(result.fitness_history)}")
    print(f"Pareto front size: {len(result.pareto_front)}")

    # --- Evaluate on unseen data ---
    x_test = np.linspace(-10, 10, 50)
    y_test = x_test**2 + 2 * x_test + 1
    test_context = {"x": x_test}

    prediction = evaluate(best, test_context)
    test_mse = float(np.mean((prediction - y_test) ** 2))
    print(f"Test MSE (extrapolation): {test_mse:.8f}")

    # --- Serialization round-trip ---
    data = serialize(best)
    restored = deserialize(data, registry)
    restored_output = evaluate(restored, test_context)
    assert np.allclose(prediction, restored_output), "Serialization round-trip failed!"
    print("Serialization round-trip: OK")


if __name__ == "__main__":
    main()
