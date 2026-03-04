"""Verify that the example scripts run correctly."""

from __future__ import annotations

import numpy as np
import pytest

from liq.gp import (
    BoolSeries,
    FitnessConfig,
    FitnessResult,
    GPConfig,
    ParamSpec,
    PrimitiveRegistry,
    Program,
    Series,
    deserialize,
    evaluate,
    evolve,
    serialize,
)


def _build_registry() -> PrimitiveRegistry:
    """Same registry as the example."""
    reg = PrimitiveRegistry()
    reg.register("x", lambda: None, input_types=(), output_type=Series)
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


class _RegressionEvaluator:
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


class TestSymbolicRegressionExample:
    """Ensure the example pattern produces a good solution."""

    def test_finds_good_solution(self) -> None:
        x = np.linspace(-5, 5, 200)
        y = x**2 + 2 * x + 1
        context = {"x": x}

        registry = _build_registry()
        evaluator = _RegressionEvaluator(y)

        config = GPConfig(
            population_size=200,
            max_depth=6,
            generations=30,
            seed=42,
            crossover_rate=0.7,
            subtree_mutation_rate=0.1,
            point_mutation_rate=0.1,
            parameter_mutation_rate=0.0,
            hoist_mutation_rate=0.1,
            constant_opt_enabled=True,
            simplification_enabled=True,
        )

        result = evolve(
            registry=registry,
            config=config,
            evaluator=evaluator,
            context=context,
        )

        best = result.best_program
        neg_mse = result.fitness_history[-1].best_fitness[0]

        # Should find a very good fit (MSE < 0.01)
        assert -neg_mse < 0.01, f"MSE too high: {-neg_mse}"
        assert best.size > 0
        assert len(result.fitness_history) == 30

    def test_serialization_round_trip(self) -> None:
        x = np.linspace(-5, 5, 100)
        y = x**2 + 2 * x + 1
        context = {"x": x}

        registry = _build_registry()
        evaluator = _RegressionEvaluator(y)

        config = GPConfig(
            population_size=100,
            max_depth=5,
            generations=10,
            seed=42,
            crossover_rate=0.7,
            subtree_mutation_rate=0.1,
            point_mutation_rate=0.1,
            parameter_mutation_rate=0.0,
            hoist_mutation_rate=0.1,
        )

        result = evolve(
            registry=registry,
            config=config,
            evaluator=evaluator,
            context=context,
        )

        best = result.best_program
        original_output = evaluate(best, context)

        data = serialize(best)
        restored = deserialize(data, registry)
        restored_output = evaluate(restored, context)

        assert np.allclose(original_output, restored_output)

    def test_evaluate_on_new_data(self) -> None:
        x_train = np.linspace(-5, 5, 100)
        y_train = x_train**2 + 2 * x_train + 1

        registry = _build_registry()
        evaluator = _RegressionEvaluator(y_train)

        config = GPConfig(
            population_size=200,
            max_depth=6,
            generations=30,
            seed=42,
            crossover_rate=0.7,
            subtree_mutation_rate=0.1,
            point_mutation_rate=0.1,
            parameter_mutation_rate=0.0,
            hoist_mutation_rate=0.1,
            constant_opt_enabled=True,
            simplification_enabled=True,
        )

        result = evolve(
            registry=registry,
            config=config,
            evaluator=evaluator,
            context={"x": x_train},
        )

        # Evaluate on new data
        x_test = np.linspace(-10, 10, 50)
        y_test = x_test**2 + 2 * x_test + 1
        prediction = evaluate(result.best_program, {"x": x_test})
        test_mse = float(np.mean((prediction - y_test) ** 2))

        # Should generalize well since it's finding the exact polynomial
        assert test_mse < 1.0, f"Test MSE too high: {test_mse}"


# ---------------------------------------------------------------------------
# Multi-objective example
# ---------------------------------------------------------------------------


def _build_multi_registry() -> PrimitiveRegistry:
    """Same registry as the multi-objective example."""
    reg = PrimitiveRegistry()
    reg.register("x", lambda: None, input_types=(), output_type=Series)
    reg.register("y", lambda: None, input_types=(), output_type=Series)
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

    def safe_div(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(np.abs(b) > 1e-10, a / b, 0.0)

    reg.register(
        "div",
        safe_div,
        category="arithmetic",
        input_types=(Series, Series),
        output_type=Series,
    )
    reg.register(
        "gt",
        lambda a, b: np.where(a > b, 1.0, 0.0),
        category="comparison",
        input_types=(Series, Series),
        output_type=BoolSeries,
    )
    reg.register(
        "if_then_else",
        lambda cond, a, b: np.where(cond > 0.5, a, b),
        category="conditional",
        input_types=(BoolSeries, Series, Series),
        output_type=Series,
    )
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


class _MultiObjectiveEvaluator:
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
                results.append(FitnessResult(objectives=(-mse, -float(prog.size))))
            except Exception:
                results.append(FitnessResult(objectives=(-1e10, -100.0)))
        return results


@pytest.mark.slow
class TestMultiObjectiveExample:
    """Ensure the multi-objective example pattern works correctly."""

    def test_nsga2_finds_good_solution(self) -> None:
        """NSGA-II with pareto parsimony finds a low-MSE program."""
        rng = np.random.default_rng(0)
        x = rng.uniform(-5, 5, size=200)
        y = rng.uniform(-5, 5, size=200)
        target = np.where(x > 0, 2.0 * x + y, x * y)
        context = {"x": x, "y": y}

        registry = _build_multi_registry()
        evaluator = _MultiObjectiveEvaluator(target)

        fitness_config = FitnessConfig(
            objectives=["accuracy", "simplicity"],
            objective_directions=["maximize", "maximize"],
        )
        config = GPConfig(
            population_size=500,
            max_depth=8,
            max_size=60,
            generations=50,
            seed=42,
            crossover_rate=0.55,
            subtree_mutation_rate=0.20,
            point_mutation_rate=0.10,
            parameter_mutation_rate=0.10,
            hoist_mutation_rate=0.05,
            selection_mode="nsga2",
            parsimony_mode="pareto",
            constant_opt_enabled=True,
            simplification_enabled=True,
            fitness=fitness_config,
        )

        result = evolve(
            registry=registry,
            config=config,
            evaluator=evaluator,
            context=context,
        )

        best = result.best_program
        prediction = evaluate(best, context)
        mse = float(np.mean((prediction - target) ** 2))

        # Should find a good fit
        assert mse < 5.0, f"MSE too high: {mse}"
        assert best.size > 0

    def test_pareto_front_has_multiple_solutions(self) -> None:
        """NSGA-II should produce a Pareto front with diverse solutions."""
        rng = np.random.default_rng(0)
        x = rng.uniform(-5, 5, size=200)
        y = rng.uniform(-5, 5, size=200)
        target = np.where(x > 0, 2.0 * x + y, x * y)
        context = {"x": x, "y": y}

        registry = _build_multi_registry()
        evaluator = _MultiObjectiveEvaluator(target)

        fitness_config = FitnessConfig(
            objectives=["accuracy", "simplicity"],
            objective_directions=["maximize", "maximize"],
        )
        config = GPConfig(
            population_size=300,
            max_depth=7,
            generations=20,
            seed=42,
            crossover_rate=0.55,
            subtree_mutation_rate=0.20,
            point_mutation_rate=0.10,
            parameter_mutation_rate=0.10,
            hoist_mutation_rate=0.05,
            selection_mode="nsga2",
            parsimony_mode="pareto",
            fitness=fitness_config,
        )

        result = evolve(
            registry=registry,
            config=config,
            evaluator=evaluator,
            context=context,
        )

        # Should have multiple Pareto-front programs of varying sizes
        assert len(result.pareto_front) >= 2
        sizes = {p.size for p in result.pareto_front}
        assert len(sizes) >= 2, (
            "Pareto front should contain programs of different sizes"
        )
