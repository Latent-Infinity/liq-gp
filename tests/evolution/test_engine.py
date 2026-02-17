"""Tests for the evolution engine loop (FR-5.5)."""

from __future__ import annotations

import numpy as np

from liq.gp.config import FitnessConfig, GPConfig
from liq.gp.primitives.registry import PrimitiveRegistry
from liq.gp.program.ast import Program
from liq.gp.program.eval import evaluate
from liq.gp.types import (
    EvolutionResult,
    FitnessResult,
    GenerationStats,
    Series,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_registry() -> PrimitiveRegistry:
    """Build a minimal registry for testing evolution."""
    reg = PrimitiveRegistry()
    reg.register("x", lambda: None, input_types=(), output_type=Series)
    reg.register(
        "add",
        lambda a, b: a + b,
        category="numeric",
        input_types=(Series, Series),
        output_type=Series,
    )
    reg.register(
        "mul",
        lambda a, b: a * b,
        category="numeric",
        input_types=(Series, Series),
        output_type=Series,
    )
    reg.register(
        "neg",
        lambda a: -a,
        category="numeric",
        input_types=(Series,),
        output_type=Series,
    )
    return reg


def _make_config(**overrides: object) -> GPConfig:
    """Build a GPConfig with small test defaults."""
    defaults: dict[str, object] = {
        "population_size": 20,
        "max_depth": 4,
        "generations": 5,
        "seed": 42,
        "constant_opt_enabled": False,
        "simplification_enabled": False,
        "semantic_dedup_enabled": False,
        "elitism_count": 2,
        "tournament_size": 3,
    }
    defaults.update(overrides)
    return GPConfig(**defaults)  # type: ignore[arg-type]


class SimpleFitnessEvaluator:
    """Evaluator that measures MSE fitness for y = 2*x."""

    def __init__(self, context: dict[str, np.ndarray]) -> None:
        self.context = context
        self.call_count = 0

    def evaluate(
        self,
        programs: list[Program],
        context: dict[str, np.ndarray],
    ) -> list[FitnessResult]:
        self.call_count += 1
        results: list[FitnessResult] = []
        target = 2.0 * context["x"]
        for prog in programs:
            try:
                output = evaluate(prog, context)
                mse = float(np.mean((output - target) ** 2))
                # Negate MSE so higher is better (maximize)
                results.append(FitnessResult(objectives=(-mse,)))
            except Exception:
                results.append(FitnessResult(objectives=(-1e10,)))
        return results


class MultiObjectiveEvaluator:
    """Two-objective evaluator used for NSGA-II coverage tests."""

    def __init__(self, context: dict[str, np.ndarray]) -> None:
        self.context = context

    def evaluate(
        self,
        programs: list[Program],
        context: dict[str, np.ndarray],
    ) -> list[FitnessResult]:
        target = 2.0 * context["x"]
        results: list[FitnessResult] = []
        for prog in programs:
            try:
                output = evaluate(prog, context)
                mse = float(np.mean((output - target) ** 2))
                results.append(
                    FitnessResult(objectives=(-mse, float(-len(prog.constants) - prog.size)))
                )
            except Exception:
                results.append(FitnessResult(objectives=(-1e10, -1e10)))
        return results


def _make_context(n: int = 50) -> dict[str, np.ndarray]:
    """Build a simple evaluation context."""
    rng = np.random.default_rng(0)
    return {"x": rng.uniform(-1.0, 1.0, size=n)}


def _make_low_diversity_registry() -> PrimitiveRegistry:
    """Registry constrained to produce many semantic duplicates."""
    reg = PrimitiveRegistry()
    reg.register("x", lambda: None, input_types=(), output_type=Series)
    reg.register(
        "neg",
        lambda a: -a,
        category="numeric",
        input_types=(Series,),
        output_type=Series,
    )
    return reg


# ===========================================================================
# Basic engine tests
# ===========================================================================


class TestEvolveBasic:
    """Basic tests for the evolve() function."""

    def test_returns_evolution_result(self) -> None:
        from liq.gp.evolution.engine import evolve

        reg = _make_registry()
        config = _make_config(generations=3)
        context = _make_context()
        evaluator = SimpleFitnessEvaluator(context)
        result = evolve(reg, config, evaluator, context)
        assert isinstance(result, EvolutionResult)

    def test_result_has_required_fields(self) -> None:
        from liq.gp.evolution.engine import evolve

        reg = _make_registry()
        config = _make_config(generations=3)
        context = _make_context()
        evaluator = SimpleFitnessEvaluator(context)
        result = evolve(reg, config, evaluator, context)
        assert result.best_program is not None
        assert isinstance(result.pareto_front, list)
        assert isinstance(result.fitness_history, list)
        assert result.config is config

    def test_fitness_history_length(self) -> None:
        from liq.gp.evolution.engine import evolve

        reg = _make_registry()
        config = _make_config(generations=5)
        context = _make_context()
        evaluator = SimpleFitnessEvaluator(context)
        result = evolve(reg, config, evaluator, context)
        assert len(result.fitness_history) == 5

    def test_generation_stats_fields(self) -> None:
        from liq.gp.evolution.engine import evolve

        reg = _make_registry()
        config = _make_config(generations=3)
        context = _make_context()
        evaluator = SimpleFitnessEvaluator(context)
        result = evolve(reg, config, evaluator, context)
        stats = result.fitness_history[0]
        assert isinstance(stats, GenerationStats)
        assert stats.generation == 0
        assert isinstance(stats.best_fitness, tuple)
        assert isinstance(stats.mean_fitness, tuple)
        assert isinstance(stats.best_program_size, int)
        assert isinstance(stats.mean_program_size, float)
        assert isinstance(stats.pareto_front_size, int)


# ===========================================================================
# Determinism
# ===========================================================================


class TestEvolveDeterminism:
    """evolve() is deterministic with fixed seed."""

    def test_same_seed_same_result(self) -> None:
        from liq.gp.evolution.engine import evolve

        reg = _make_registry()
        config = _make_config(generations=3, seed=123)
        context = _make_context()

        result1 = evolve(reg, config, SimpleFitnessEvaluator(context), context)
        result2 = evolve(reg, config, SimpleFitnessEvaluator(context), context)

        # Best program should be identical
        assert result1.best_program == result2.best_program
        # Fitness history should be identical
        assert len(result1.fitness_history) == len(result2.fitness_history)
        for s1, s2 in zip(
            result1.fitness_history, result2.fitness_history, strict=False
        ):
            assert s1.best_fitness == s2.best_fitness

    def test_different_seed_different_result(self) -> None:
        from liq.gp.evolution.engine import evolve

        reg = _make_registry()
        context = _make_context()

        result1 = evolve(
            reg,
            _make_config(generations=3, seed=1),
            SimpleFitnessEvaluator(context),
            context,
        )
        result2 = evolve(
            reg,
            _make_config(generations=3, seed=2),
            SimpleFitnessEvaluator(context),
            context,
        )
        # Different seeds take different evolutionary paths
        h1 = [s.mean_fitness for s in result1.fitness_history]
        h2 = [s.mean_fitness for s in result2.fitness_history]
        assert h1 != h2


# ===========================================================================
# Fitness improvement
# ===========================================================================


class TestFitnessImprovement:
    """Evolution should improve fitness over generations."""

    def test_fitness_improves_or_stays(self) -> None:
        from liq.gp.evolution.engine import evolve

        reg = _make_registry()
        config = _make_config(generations=10, population_size=30)
        context = _make_context()
        evaluator = SimpleFitnessEvaluator(context)
        result = evolve(reg, config, evaluator, context)
        # Best fitness should not degrade (elitism preserves best)
        first = result.fitness_history[0].best_fitness[0]
        last = result.fitness_history[-1].best_fitness[0]
        assert last >= first

    def test_nsga2_selection_runs(self) -> None:
        from liq.gp.evolution.engine import evolve

        reg = _make_registry()
        config = _make_config(
            generations=3,
            population_size=20,
            selection_mode="nsga2",
            fitness=FitnessConfig(
                objectives=["mse", "complexity"],
                objective_directions=["minimize", "maximize"],
            ),
        )
        context = _make_context()
        evaluator = MultiObjectiveEvaluator(context)

        result = evolve(
            reg,
            config,
            evaluator,
            context,
        )
        assert len(result.fitness_history) == 3
        assert isinstance(result.pareto_front, list)


# ===========================================================================
# Callback
# ===========================================================================


class TestGenerationCallback:
    """Generation callback receives GenerationStats each generation."""

    def test_callback_invoked_each_generation(self) -> None:
        from liq.gp.evolution.engine import evolve

        reg = _make_registry()
        config = _make_config(generations=5)
        context = _make_context()
        evaluator = SimpleFitnessEvaluator(context)

        stats_collected: list[GenerationStats] = []
        evolve(
            reg,
            config,
            evaluator,
            context,
            callback=lambda s: stats_collected.append(s),
        )
        assert len(stats_collected) == 5
        for i, s in enumerate(stats_collected):
            assert s.generation == i

    def test_callback_stats_match_history(self) -> None:
        from liq.gp.evolution.engine import evolve

        reg = _make_registry()
        config = _make_config(generations=3)
        context = _make_context()
        evaluator = SimpleFitnessEvaluator(context)

        stats_collected: list[GenerationStats] = []
        result = evolve(
            reg,
            config,
            evaluator,
            context,
            callback=lambda s: stats_collected.append(s),
        )
        assert stats_collected == result.fitness_history


class TestGenerationStats:
    """GenerationStats fields reflect actual generation state."""

    def test_unique_semantics_ratio_tracks_diversity(self) -> None:
        from liq.gp.evolution.engine import evolve

        reg = _make_low_diversity_registry()
        config = _make_config(
            population_size=30,
            max_depth=2,
            generations=3,
            semantic_dedup_enabled=True,
        )
        context = _make_context()
        evaluator = SimpleFitnessEvaluator(context)
        result = evolve(reg, config, evaluator, context)
        ratios = [s.unique_semantics_ratio for s in result.fitness_history]
        assert all(0.0 < r <= 1.0 for r in ratios)
        assert any(r < 1.0 for r in ratios)


# ===========================================================================
# Early stopping
# ===========================================================================


class TestEarlyStopping:
    """Early stopping terminates evolution when fitness stalls."""

    def test_early_stop_fewer_generations(self) -> None:
        from liq.gp.evolution.engine import evolve

        reg = _make_registry()
        # Very low threshold + patience=2: should stop quickly
        config = _make_config(
            generations=50,
            early_stop_patience=2,
            early_stop_threshold=1e10,  # absurdly high = always "stalled"
        )
        context = _make_context()
        evaluator = SimpleFitnessEvaluator(context)
        result = evolve(reg, config, evaluator, context)
        # Should have stopped before 50 generations
        assert len(result.fitness_history) < 50

    def test_no_early_stop_runs_all_generations(self) -> None:
        from liq.gp.evolution.engine import evolve

        reg = _make_registry()
        config = _make_config(generations=5, early_stop_patience=None)
        context = _make_context()
        evaluator = SimpleFitnessEvaluator(context)
        result = evolve(reg, config, evaluator, context)
        assert len(result.fitness_history) == 5

    def test_early_stop_respects_objective_direction(self) -> None:
        from liq.gp.evolution.engine import evolve

        reg = _make_registry()
        config = _make_config(
            generations=20,
            early_stop_patience=2,
            early_stop_threshold=1e-6,
            fitness={
                "objectives": ["loss"],
                "objective_directions": ["minimize"],
            },
        )
        context = _make_context()

        class WorseningLossEvaluator:
            def __init__(self) -> None:
                self.calls = 0

            def evaluate(
                self,
                programs: list[Program],
                context: dict[str, np.ndarray],
            ) -> list[FitnessResult]:
                self.calls += 1
                loss = float(self.calls)
                return [FitnessResult(objectives=(loss,)) for _ in programs]

        result = evolve(reg, config, WorseningLossEvaluator(), context)
        assert len(result.fitness_history) == 3


# ===========================================================================
# Pareto front tracking
# ===========================================================================


class TestParetoFrontTracking:
    """Pareto front is tracked and returned."""

    def test_tournament_mode_has_pareto_front(self) -> None:
        from liq.gp.evolution.engine import evolve

        reg = _make_registry()
        config = _make_config(generations=3)
        context = _make_context()
        evaluator = SimpleFitnessEvaluator(context)
        result = evolve(reg, config, evaluator, context)
        # In single-objective, pareto_front should have at least 1 entry
        assert len(result.pareto_front) >= 1


# ===========================================================================
# Evaluator integration
# ===========================================================================


class TestEvaluatorIntegration:
    """The consumer-provided FitnessEvaluator is properly invoked."""

    def test_evaluator_called(self) -> None:
        from liq.gp.evolution.engine import evolve

        reg = _make_registry()
        config = _make_config(generations=3)
        context = _make_context()
        evaluator = SimpleFitnessEvaluator(context)
        evolve(reg, config, evaluator, context)
        # Evaluator should have been called at least once per generation
        assert evaluator.call_count >= 3


# ===========================================================================
# Batch / mini-batch evaluation (FR-5.4.3)
# ===========================================================================


class ContextTrackingEvaluator:
    """Evaluator that records the context length seen on each call."""

    def __init__(self) -> None:
        self.observed_lengths: list[int] = []

    def evaluate(
        self,
        programs: list[Program],
        context: dict[str, np.ndarray],
    ) -> list[FitnessResult]:
        # Record the length of the first array in context
        first_key = next(k for k in context if not k.startswith("__"))
        self.observed_lengths.append(len(context[first_key]))
        return [FitnessResult(objectives=(0.0,)) for _ in programs]


class TestBatchEvaluation:
    """Batch/mini-batch evaluation (FR-5.4.3)."""

    def test_batch_size_subsets_context(self) -> None:
        """When batch_size is set, evaluation context is a subset."""
        from liq.gp.evolution.engine import evolve

        reg = _make_registry()
        context = _make_context(n=100)
        evaluator = ContextTrackingEvaluator()
        config = _make_config(
            generations=3,
            fitness={
                "objectives": ["fitness"],
                "objective_directions": ["maximize"],
                "batch_size": 20,
                "full_eval_interval": 10,
            },
        )
        evolve(reg, config, evaluator, context)
        # Most evaluations should use batch_size (20), not full (100)
        batch_evals = [L for L in evaluator.observed_lengths if L == 20]
        assert len(batch_evals) > 0

    def test_full_eval_at_interval(self) -> None:
        """Full evaluation happens every full_eval_interval generations."""
        from liq.gp.evolution.engine import evolve

        reg = _make_registry()
        context = _make_context(n=100)
        evaluator = ContextTrackingEvaluator()
        config = _make_config(
            generations=12,
            fitness={
                "objectives": ["fitness"],
                "objective_directions": ["maximize"],
                "batch_size": 20,
                "full_eval_interval": 5,
            },
        )
        evolve(reg, config, evaluator, context)
        # At least some evaluations should use full context (100)
        full_evals = [L for L in evaluator.observed_lengths if L == 100]
        assert len(full_evals) >= 1

    def test_final_evaluation_uses_full_context(self) -> None:
        """The final evaluation for result extraction always uses full context."""
        from liq.gp.evolution.engine import evolve

        reg = _make_registry()
        context = _make_context(n=100)
        evaluator = ContextTrackingEvaluator()
        config = _make_config(
            generations=3,
            fitness={
                "objectives": ["fitness"],
                "objective_directions": ["maximize"],
                "batch_size": 20,
                "full_eval_interval": 100,  # never triggers during 3 gens
            },
        )
        evolve(reg, config, evaluator, context)
        # Last evaluation (final) should use full context
        assert evaluator.observed_lengths[-1] == 100

    def test_no_batch_uses_full_context(self) -> None:
        """Without batch_size, all evaluations use the full context."""
        from liq.gp.evolution.engine import evolve

        reg = _make_registry()
        context = _make_context(n=100)
        evaluator = ContextTrackingEvaluator()
        config = _make_config(generations=3)
        evolve(reg, config, evaluator, context)
        assert all(L == 100 for L in evaluator.observed_lengths)

    def test_batch_deterministic_with_seed(self) -> None:
        """Batch selection is deterministic with same seed."""
        from liq.gp.evolution.engine import evolve

        reg = _make_registry()
        context = _make_context(n=100)

        config = _make_config(
            generations=3,
            seed=99,
            fitness={
                "objectives": ["fitness"],
                "objective_directions": ["maximize"],
                "batch_size": 20,
                "full_eval_interval": 10,
            },
        )
        eval1 = ContextTrackingEvaluator()
        eval2 = ContextTrackingEvaluator()
        evolve(reg, config, eval1, context)
        evolve(reg, config, eval2, context)
        assert eval1.observed_lengths == eval2.observed_lengths

    def test_batch_size_larger_than_data_uses_full(self) -> None:
        """If batch_size >= data length, full context is used."""
        from liq.gp.evolution.engine import evolve

        reg = _make_registry()
        context = _make_context(n=20)
        evaluator = ContextTrackingEvaluator()
        config = _make_config(
            generations=3,
            fitness={
                "objectives": ["fitness"],
                "objective_directions": ["maximize"],
                "batch_size": 50,  # larger than context size of 20
                "full_eval_interval": 10,
            },
        )
        evolve(reg, config, evaluator, context)
        assert all(L == 20 for L in evaluator.observed_lengths)
