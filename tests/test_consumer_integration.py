"""Consumer integration tests (requirements section 11).

Replicates the pattern where a consumer library (e.g. liq-evolution) builds
a PrimitiveRegistry with domain-specific primitives, provides a
FitnessEvaluator, configures GPConfig, and calls evolve().

All imports come from the public API ``liq.gp``.
"""

from __future__ import annotations

import numpy as np

from liq.gp import (
    EvolutionResult,
    FitnessConfig,
    FitnessResult,
    GenerationStats,
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

# ---------------------------------------------------------------------------
# Shared helpers — simulating what a consumer library would build
# ---------------------------------------------------------------------------


def _consumer_registry() -> PrimitiveRegistry:
    """Build a minimal registry like a consumer (e.g. liq-evolution) would."""
    reg = PrimitiveRegistry()
    # Terminals
    reg.register("x", lambda: None, input_types=(), output_type=Series)
    reg.register("y", lambda: None, input_types=(), output_type=Series)
    # Functions
    reg.register(
        "add",
        lambda a, b: a + b,
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


def _make_context(n: int = 100) -> dict[str, np.ndarray]:
    """Build an evaluation context with x and y columns."""
    rng = np.random.default_rng(42)
    x = rng.uniform(-2.0, 2.0, size=n)
    y = rng.uniform(-2.0, 2.0, size=n)
    return {"x": x, "y": y}


# ---------------------------------------------------------------------------
# Consumer-style evaluators
# ---------------------------------------------------------------------------


class SingleObjectiveEvaluator:
    """Computes negative MSE against a known target (single objective)."""

    def __init__(self, target_fn):
        self.target_fn = target_fn

    def evaluate(
        self,
        programs: list[Program],
        context: dict[str, np.ndarray],
    ) -> list[FitnessResult]:
        target = self.target_fn(context)
        results: list[FitnessResult] = []
        for prog in programs:
            try:
                output = evaluate(prog, context)
                mse = float(np.mean((output - target) ** 2))
                results.append(FitnessResult(objectives=(-mse,)))
            except Exception:
                results.append(FitnessResult(objectives=(-1e10,)))
        return results


class MultiObjectiveEvaluator:
    """Returns two objectives: -mse (maximize) and -program.size (maximize)."""

    def __init__(self, target_fn):
        self.target_fn = target_fn

    def evaluate(
        self,
        programs: list[Program],
        context: dict[str, np.ndarray],
    ) -> list[FitnessResult]:
        target = self.target_fn(context)
        results: list[FitnessResult] = []
        for prog in programs:
            try:
                output = evaluate(prog, context)
                mse = float(np.mean((output - target) ** 2))
                results.append(FitnessResult(objectives=(-mse, -float(prog.size))))
            except Exception:
                results.append(FitnessResult(objectives=(-1e10, -100.0)))
        return results


# ===========================================================================
# 1. Single-objective symbolic regression
# ===========================================================================


class TestConsumerSingleObjective:
    """Mimics a consumer doing single-objective symbolic regression.

    Target: y = x^2 + x
    """

    def test_returns_evolution_result(self) -> None:
        reg = _consumer_registry()
        config = GPConfig(
            population_size=50,
            max_depth=5,
            generations=10,
            seed=42,
            selection_mode="tournament",
            tournament_size=3,
            constant_opt_enabled=False,
            simplification_enabled=True,
            semantic_dedup_enabled=False,
            elitism_count=2,
        )
        context = _make_context()
        evaluator = SingleObjectiveEvaluator(lambda ctx: ctx["x"] ** 2 + ctx["x"])
        result = evolve(reg, config, evaluator, context)

        assert isinstance(result, EvolutionResult)

    def test_best_program_is_not_none(self) -> None:
        reg = _consumer_registry()
        config = GPConfig(
            population_size=50,
            max_depth=5,
            generations=10,
            seed=42,
            selection_mode="tournament",
            tournament_size=3,
            constant_opt_enabled=False,
            simplification_enabled=True,
            semantic_dedup_enabled=False,
            elitism_count=2,
        )
        context = _make_context()
        evaluator = SingleObjectiveEvaluator(lambda ctx: ctx["x"] ** 2 + ctx["x"])
        result = evolve(reg, config, evaluator, context)

        assert result.best_program is not None

    def test_fitness_improves_over_generations(self) -> None:
        reg = _consumer_registry()
        config = GPConfig(
            population_size=50,
            max_depth=5,
            generations=10,
            seed=42,
            selection_mode="tournament",
            tournament_size=3,
            constant_opt_enabled=False,
            simplification_enabled=True,
            semantic_dedup_enabled=False,
            elitism_count=2,
        )
        context = _make_context()
        evaluator = SingleObjectiveEvaluator(lambda ctx: ctx["x"] ** 2 + ctx["x"])
        result = evolve(reg, config, evaluator, context)

        first_gen_best = result.fitness_history[0].best_fitness[0]
        last_gen_best = result.fitness_history[-1].best_fitness[0]
        # Negative MSE, so higher (less negative) is better.
        # Best fitness should not degrade.
        assert last_gen_best >= first_gen_best

    def test_best_program_evaluable_correct_shape(self) -> None:
        reg = _consumer_registry()
        config = GPConfig(
            population_size=50,
            max_depth=5,
            generations=10,
            seed=42,
            selection_mode="tournament",
            tournament_size=3,
            constant_opt_enabled=False,
            simplification_enabled=True,
            semantic_dedup_enabled=False,
            elitism_count=2,
        )
        context = _make_context(n=80)
        evaluator = SingleObjectiveEvaluator(lambda ctx: ctx["x"] ** 2 + ctx["x"])
        result = evolve(reg, config, evaluator, context)

        output = evaluate(result.best_program, context)
        assert isinstance(output, np.ndarray)
        assert output.shape == (80,)
        assert output.dtype == np.float64


# ===========================================================================
# 2. Multi-objective (NSGA-II) consumer pattern
# ===========================================================================


class TestConsumerMultiObjective:
    """Mimics a consumer with NSGA-II multi-objective selection.

    Objectives: (-mse, -program_size), both maximized.
    """

    def test_pareto_front_has_entries(self) -> None:
        reg = _consumer_registry()
        config = GPConfig(
            population_size=30,
            max_depth=5,
            generations=8,
            seed=42,
            selection_mode="nsga2",
            constant_opt_enabled=False,
            simplification_enabled=False,
            semantic_dedup_enabled=False,
            fitness=FitnessConfig(
                objectives=["accuracy", "parsimony"],
                objective_directions=["maximize", "maximize"],
            ),
        )
        context = _make_context()
        evaluator = MultiObjectiveEvaluator(lambda ctx: ctx["x"] ** 2 + ctx["x"])
        result = evolve(reg, config, evaluator, context)

        assert isinstance(result, EvolutionResult)
        assert len(result.pareto_front) >= 1

    def test_stats_have_two_objective_tuples(self) -> None:
        reg = _consumer_registry()
        config = GPConfig(
            population_size=30,
            max_depth=5,
            generations=8,
            seed=42,
            selection_mode="nsga2",
            constant_opt_enabled=False,
            simplification_enabled=False,
            semantic_dedup_enabled=False,
            fitness=FitnessConfig(
                objectives=["accuracy", "parsimony"],
                objective_directions=["maximize", "maximize"],
            ),
        )
        context = _make_context()
        evaluator = MultiObjectiveEvaluator(lambda ctx: ctx["x"] ** 2 + ctx["x"])
        result = evolve(reg, config, evaluator, context)

        for stats in result.fitness_history:
            assert isinstance(stats, GenerationStats)
            assert len(stats.best_fitness) == 2
            assert len(stats.mean_fitness) == 2

    def test_section11_keyword_pattern_with_fitness_config(self) -> None:
        """Supports section-11 style call using a separate fitness_config kwarg."""
        registry = _consumer_registry()
        config = GPConfig(
            population_size=30,
            max_depth=5,
            generations=5,
            seed=42,
            selection_mode="nsga2",
            constant_opt_enabled=False,
            simplification_enabled=False,
            semantic_dedup_enabled=False,
            fitness=FitnessConfig(
                objectives=["placeholder_a", "placeholder_b"],
                objective_directions=["maximize", "maximize"],
            ),
        )
        fitness_config = FitnessConfig(
            objectives=["accuracy", "parsimony"],
            objective_directions=["maximize", "maximize"],
        )
        context = _make_context()
        evaluator = MultiObjectiveEvaluator(lambda ctx: ctx["x"] ** 2 + ctx["x"])

        result = evolve(
            config=config,
            fitness_config=fitness_config,
            registry=registry,
            evaluator=evaluator,
            context=context,
        )

        assert isinstance(result, EvolutionResult)
        assert result.config.fitness == fitness_config
        assert len(result.pareto_front) >= 1


# ===========================================================================
# 3. Consumer with parameterized primitive
# ===========================================================================


class TestConsumerWithParameterizedPrimitive:
    """Consumer registers a parameterized primitive ("scale") and runs evolution."""

    def test_parameterized_evolution_completes(self) -> None:
        reg = _consumer_registry()
        reg.register(
            "scale",
            lambda a, factor: a * factor,
            category="arithmetic",
            input_types=(Series,),
            output_type=Series,
            param_specs=[ParamSpec("factor", float, 1.0, 0.0, 10.0)],
        )

        config = GPConfig(
            population_size=30,
            max_depth=5,
            generations=5,
            seed=42,
            selection_mode="tournament",
            tournament_size=3,
            constant_opt_enabled=False,
            simplification_enabled=False,
            semantic_dedup_enabled=False,
            elitism_count=2,
        )
        context = _make_context()
        evaluator = SingleObjectiveEvaluator(lambda ctx: 3.0 * ctx["x"])
        result = evolve(reg, config, evaluator, context)

        assert isinstance(result, EvolutionResult)
        assert result.best_program is not None
        assert len(result.fitness_history) == 5

    def test_parameterized_best_program_evaluable(self) -> None:
        reg = _consumer_registry()
        reg.register(
            "scale",
            lambda a, factor: a * factor,
            category="arithmetic",
            input_types=(Series,),
            output_type=Series,
            param_specs=[ParamSpec("factor", float, 1.0, 0.0, 10.0)],
        )

        config = GPConfig(
            population_size=30,
            max_depth=5,
            generations=5,
            seed=42,
            selection_mode="tournament",
            tournament_size=3,
            constant_opt_enabled=False,
            simplification_enabled=False,
            semantic_dedup_enabled=False,
            elitism_count=2,
        )
        context = _make_context(n=50)
        evaluator = SingleObjectiveEvaluator(lambda ctx: 3.0 * ctx["x"])
        result = evolve(reg, config, evaluator, context)

        output = evaluate(result.best_program, context)
        assert isinstance(output, np.ndarray)
        assert output.shape == (50,)


# ===========================================================================
# 4. Consumer callback pattern
# ===========================================================================


class TestConsumerCallbackPattern:
    """Consumer uses a generation callback to collect per-generation stats."""

    def test_callback_called_each_generation(self) -> None:
        reg = _consumer_registry()
        config = GPConfig(
            population_size=20,
            max_depth=4,
            generations=6,
            seed=42,
            constant_opt_enabled=False,
            simplification_enabled=False,
            semantic_dedup_enabled=False,
        )
        context = _make_context()
        evaluator = SingleObjectiveEvaluator(lambda ctx: ctx["x"])

        collected: list[GenerationStats] = []
        evolve(
            reg,
            config,
            evaluator,
            context,
            callback=lambda s: collected.append(s),
        )

        assert len(collected) == config.generations

    def test_callback_stats_match_history(self) -> None:
        reg = _consumer_registry()
        config = GPConfig(
            population_size=20,
            max_depth=4,
            generations=6,
            seed=42,
            constant_opt_enabled=False,
            simplification_enabled=False,
            semantic_dedup_enabled=False,
        )
        context = _make_context()
        evaluator = SingleObjectiveEvaluator(lambda ctx: ctx["x"])

        collected: list[GenerationStats] = []
        result = evolve(
            reg,
            config,
            evaluator,
            context,
            callback=lambda s: collected.append(s),
        )

        assert collected == result.fitness_history
        for i, stats in enumerate(collected):
            assert stats.generation == i

    def test_callback_stats_have_valid_fields(self) -> None:
        reg = _consumer_registry()
        config = GPConfig(
            population_size=20,
            max_depth=4,
            generations=6,
            seed=42,
            constant_opt_enabled=False,
            simplification_enabled=False,
            semantic_dedup_enabled=False,
        )
        context = _make_context()
        evaluator = SingleObjectiveEvaluator(lambda ctx: ctx["x"])

        collected: list[GenerationStats] = []
        evolve(
            reg,
            config,
            evaluator,
            context,
            callback=lambda s: collected.append(s),
        )

        for stats in collected:
            assert isinstance(stats.best_fitness, tuple)
            assert len(stats.best_fitness) == 1
            assert stats.best_program_size >= 1
            assert stats.mean_program_size > 0
            assert stats.pareto_front_size >= 1


# ===========================================================================
# 5. Consumer serialization pattern
# ===========================================================================


class TestConsumerSerializationPattern:
    """Consumer serializes the best program from an evolution run."""

    def test_serialize_deserialize_round_trip(self) -> None:
        reg = _consumer_registry()
        config = GPConfig(
            population_size=30,
            max_depth=4,
            generations=5,
            seed=42,
            constant_opt_enabled=False,
            simplification_enabled=False,
            semantic_dedup_enabled=False,
        )
        context = _make_context()
        evaluator = SingleObjectiveEvaluator(lambda ctx: ctx["x"] ** 2 + ctx["x"])
        result = evolve(reg, config, evaluator, context)

        payload = serialize(result.best_program)
        restored = deserialize(payload, reg)

        assert restored == result.best_program

    def test_deserialized_evaluates_identically(self) -> None:
        reg = _consumer_registry()
        config = GPConfig(
            population_size=30,
            max_depth=4,
            generations=5,
            seed=42,
            constant_opt_enabled=False,
            simplification_enabled=False,
            semantic_dedup_enabled=False,
        )
        context = _make_context()
        evaluator = SingleObjectiveEvaluator(lambda ctx: ctx["x"] ** 2 + ctx["x"])
        result = evolve(reg, config, evaluator, context)

        payload = serialize(result.best_program)
        restored = deserialize(payload, reg)

        original_output = evaluate(result.best_program, context)
        restored_output = evaluate(restored, context)
        np.testing.assert_array_equal(original_output, restored_output)
