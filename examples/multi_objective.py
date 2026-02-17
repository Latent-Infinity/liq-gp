"""Multi-objective symbolic regression with advanced features.

Discovers a piecewise function using NSGA-II selection, parameterized
primitives, conditional logic (BoolSeries), protected division, early
stopping, and batch evaluation.  Demonstrates the Pareto tradeoff between
accuracy and program simplicity.

Target function (piecewise):
    f(x, y) = 2*x + y   when x > 0
              x * y      otherwise

This requires a conditional branch -- it cannot be expressed as a single
smooth formula.  The GP must discover the ``if_then_else`` structure with a
``gt`` comparison and two distinct arithmetic branches.

Usage:
    python examples/multi_objective.py
"""

from __future__ import annotations

import numpy as np

from liq.gp import (
    BoolSeries,
    FitnessConfig,
    FitnessResult,
    GenerationStats,
    GPConfig,
    ParamSpec,
    PrimitiveRegistry,
    Program,
    Series,
    evaluate,
    evolve,
    serialize,
)

# ---------------------------------------------------------------------------
# 1. Primitive registry with advanced primitives
# ---------------------------------------------------------------------------


def build_registry() -> PrimitiveRegistry:
    """Build a rich registry with arithmetic, conditional, and parameterized ops."""
    reg = PrimitiveRegistry()

    # --- Terminals (arity 0) ---
    # The callable is a placeholder -- TerminalNode evaluation reads
    # context[name] directly.
    reg.register("x", lambda: None, input_types=(), output_type=Series)
    reg.register("y", lambda: None, input_types=(), output_type=Series)

    # --- Arithmetic functions ---
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

    # --- Protected division (safe against division by zero) ---
    def safe_div(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        with np.errstate(divide="ignore", invalid="ignore"):
            result = np.where(np.abs(b) > 1e-10, a / b, 0.0)
        return result

    reg.register(
        "div",
        safe_div,
        category="arithmetic",
        input_types=(Series, Series),
        output_type=Series,
    )

    # --- Boolean / comparison (produces BoolSeries) ---
    reg.register(
        "gt",
        lambda a, b: np.where(a > b, 1.0, 0.0),
        category="comparison",
        input_types=(Series, Series),
        output_type=BoolSeries,
    )

    # --- Conditional (if_then_else): takes a BoolSeries mask and two Series ---
    reg.register(
        "if_then_else",
        lambda cond, a, b: np.where(cond > 0.5, a, b),
        category="conditional",
        input_types=(BoolSeries, Series, Series),
        output_type=Series,
    )

    # --- Parameterized primitive: scale with an evolvable coefficient ---
    # The 'factor' parameter is optimized by the constant optimizer and
    # mutated by the parameter_mutation operator.
    reg.register(
        "scale",
        lambda a, *, factor=1.0: a * factor,
        category="parameterized",
        input_types=(Series,),
        output_type=Series,
        param_specs=[
            ParamSpec(
                name="factor",
                dtype=float,
                default=1.0,
                min_value=-10.0,
                max_value=10.0,
            ),
        ],
    )

    return reg


# ---------------------------------------------------------------------------
# 2. Multi-objective fitness evaluator
# ---------------------------------------------------------------------------


class MultiObjectiveEvaluator:
    """Evaluates two objectives: accuracy (negative MSE) and simplicity.

    NSGA-II uses both objectives to build a Pareto front, giving you a
    range of solutions from "most accurate" to "simplest".
    """

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
                # Objective 1: negative MSE (maximize = lower error)
                # Objective 2: negative size (maximize = smaller program)
                results.append(
                    FitnessResult(
                        objectives=(-mse, -float(prog.size)),
                        metadata={"mse": mse, "size": prog.size},
                    )
                )
            except Exception:
                results.append(
                    FitnessResult(objectives=(-1e10, -100.0)),
                )
        return results


# ---------------------------------------------------------------------------
# 3. Target function and data
# ---------------------------------------------------------------------------


def target_function(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Piecewise: 2*x + y when x > 0, else x*y."""
    return np.where(x > 0, 2.0 * x + y, x * y)


# ---------------------------------------------------------------------------
# 4. Callbacks
# ---------------------------------------------------------------------------


def on_generation(stats: GenerationStats) -> None:
    """Print progress every 10 generations."""
    if stats.generation % 10 == 0 or stats.generation == 0:
        neg_mse = stats.best_fitness[0]
        neg_size = stats.best_fitness[1]
        print(
            f"  Gen {stats.generation:3d}  "
            f"best_mse={-neg_mse:.6f}  "
            f"best_size={-neg_size:.0f}  "
            f"pareto_front={stats.pareto_front_size}  "
            f"unique={stats.unique_semantics_ratio:.0%}"
        )


# ---------------------------------------------------------------------------
# 5. Main
# ---------------------------------------------------------------------------


def main() -> None:
    # --- Generate training data ---
    rng = np.random.default_rng(0)
    x_train = rng.uniform(-5, 5, size=500)
    y_train = rng.uniform(-5, 5, size=500)
    z_train = target_function(x_train, y_train)
    context = {"x": x_train, "y": y_train}

    registry = build_registry()
    evaluator = MultiObjectiveEvaluator(z_train)

    # --- Configure with NSGA-II and advanced features ---
    # FitnessConfig must be set inside GPConfig so that NSGA-II validation
    # sees the correct number of objectives at construction time.
    fitness_config = FitnessConfig(
        objectives=["accuracy", "simplicity"],
        objective_directions=["maximize", "maximize"],
        # Optional: mini-batch evaluation can be enabled with batch_size
        # to speed up large-dataset evaluations.  We evaluate the full
        # dataset here for a stable fitness signal.
    )

    config = GPConfig(
        population_size=1000,
        max_depth=8,
        max_size=60,
        generations=100,
        seed=42,
        # Operator rates (must sum to 1.0)
        crossover_rate=0.55,
        subtree_mutation_rate=0.20,
        point_mutation_rate=0.10,
        parameter_mutation_rate=0.10,  # higher rate for parameterized primitives
        hoist_mutation_rate=0.05,
        # Multi-objective selection
        selection_mode="nsga2",
        parsimony_mode="pareto",  # size as explicit Pareto objective
        # Constant optimization on top 10% of population
        constant_opt_enabled=True,
        constant_opt_top_k=0.1,
        # Simplification and dedup
        simplification_enabled=True,
        semantic_dedup_enabled=True,
        # Early stopping: halt if no improvement for 20 generations
        early_stop_patience=20,
        early_stop_threshold=1e-6,
        # Embed the fitness config
        fitness=fitness_config,
    )

    print("Multi-objective evolution: accuracy vs simplicity")
    print("  Target: f(x,y) = 2*x + y if x>0, else x*y")
    print(
        f"  Population: {config.population_size}, Max generations: {config.generations}"
    )
    print(f"  Selection: NSGA-II, Early stop patience: {config.early_stop_patience}")
    print()

    result = evolve(
        registry=registry,
        config=config,
        evaluator=evaluator,
        context=context,
        callback=on_generation,
    )

    # --- Analyze the Pareto front ---
    print()
    print(f"Evolution completed in {len(result.fitness_history)} generations")
    print(f"Pareto front: {len(result.pareto_front)} programs")
    print()

    # Evaluate each Pareto-front program on the full training set
    print("Pareto front (accuracy vs simplicity):")
    print(f"  {'Size':>5}  {'Train MSE':>12}  {'Test MSE':>12}")
    print(f"  {'-----':>5}  {'----------':>12}  {'----------':>12}")

    # Generate test data (out of distribution)
    x_test = rng.uniform(-8, 8, size=300)
    y_test = rng.uniform(-8, 8, size=300)
    z_test = target_function(x_test, y_test)
    test_context = {"x": x_test, "y": y_test}

    seen_sizes: set[int] = set()
    for prog in sorted(result.pareto_front, key=lambda p: p.size):
        if prog.size in seen_sizes:
            continue
        seen_sizes.add(prog.size)
        try:
            train_pred = evaluate(prog, context)
            test_pred = evaluate(prog, test_context)
            train_mse = float(np.mean((train_pred - z_train) ** 2))
            test_mse = float(np.mean((test_pred - z_test) ** 2))
            print(f"  {prog.size:5d}  {train_mse:12.6f}  {test_mse:12.6f}")
        except Exception:
            print(f"  {prog.size:5d}  {'(eval error)':>12}  {'(eval error)':>12}")

    # --- Best program details ---
    best = result.best_program
    print()
    print(f"Best program: {best.size} nodes")

    # Serialization produces a compact JSON-compatible dict
    data = serialize(best)
    print(f"Serialized keys: {list(data.keys())}")
    print(f"Schema version: {data['schema_version']}")


if __name__ == "__main__":
    main()
