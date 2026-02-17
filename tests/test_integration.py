"""Integration tests for the full GP evolution pipeline (FR-5.5, Task 2.19).

These tests exercise the end-to-end flow: initialization, evaluation,
selection, operators, simplification, constant optimization, semantic
dedup, early stopping, and serialization.
"""

from __future__ import annotations

import json

import numpy as np

from liq.gp.config import FitnessConfig, GPConfig
from liq.gp.evolution.engine import evolve
from liq.gp.primitives.registry import PrimitiveRegistry
from liq.gp.program.ast import Program
from liq.gp.program.eval import evaluate
from liq.gp.program.serialize import deserialize, serialize
from liq.gp.types import (
    EvolutionResult,
    FitnessResult,
    GenerationStats,
    Series,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _full_registry() -> PrimitiveRegistry:
    """Build a registry with arithmetic + constant terminals."""
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
        "sub",
        lambda a, b: a - b,
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


class MSEEvaluator:
    """Fitness evaluator measuring negative MSE against a target function."""

    def __init__(self, target_fn):
        self.target_fn = target_fn
        self.eval_count = 0

    def evaluate(
        self,
        programs: list[Program],
        context: dict[str, np.ndarray],
    ) -> list[FitnessResult]:
        self.eval_count += 1
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


def _make_context(n: int = 100) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(42)
    return {"x": rng.uniform(-2.0, 2.0, size=n)}


# ===========================================================================
# End-to-end regression target: y = 2*x
# ===========================================================================


class TestEndToEndSimple:
    """Full pipeline for a simple target y = 2*x."""

    def test_evolves_and_improves(self) -> None:
        reg = _full_registry()
        config = GPConfig(
            population_size=30,
            max_depth=4,
            generations=10,
            seed=42,
            constant_opt_enabled=False,
            simplification_enabled=True,
            semantic_dedup_enabled=False,
            elitism_count=2,
            tournament_size=3,
        )
        context = _make_context()
        evaluator = MSEEvaluator(lambda ctx: 2.0 * ctx["x"])
        result = evolve(reg, config, evaluator, context)

        assert isinstance(result, EvolutionResult)
        assert result.best_program is not None
        assert len(result.fitness_history) == 10
        # Fitness should improve (less negative)
        first = result.fitness_history[0].best_fitness[0]
        last = result.fitness_history[-1].best_fitness[0]
        assert last >= first
        # Best fitness should be close to 0 (perfect)
        assert last > -1.0

    def test_pareto_front_populated(self) -> None:
        reg = _full_registry()
        config = GPConfig(
            population_size=20,
            max_depth=4,
            generations=5,
            seed=7,
            constant_opt_enabled=False,
            simplification_enabled=False,
            semantic_dedup_enabled=False,
        )
        context = _make_context()
        evaluator = MSEEvaluator(lambda ctx: ctx["x"] ** 2)
        result = evolve(reg, config, evaluator, context)
        assert len(result.pareto_front) >= 1

    def test_callback_receives_all_generations(self) -> None:
        reg = _full_registry()
        config = GPConfig(
            population_size=20,
            max_depth=4,
            generations=5,
            seed=99,
            constant_opt_enabled=False,
            simplification_enabled=False,
            semantic_dedup_enabled=False,
        )
        context = _make_context()
        evaluator = MSEEvaluator(lambda ctx: ctx["x"])

        collected: list[GenerationStats] = []
        result = evolve(
            reg,
            config,
            evaluator,
            context,
            callback=lambda s: collected.append(s),
        )
        assert len(collected) == 5
        assert collected == result.fitness_history
        for i, s in enumerate(collected):
            assert s.generation == i
            assert s.pareto_front_size >= 1

    def test_deterministic_across_runs(self) -> None:
        reg = _full_registry()
        config = GPConfig(
            population_size=20,
            max_depth=4,
            generations=5,
            seed=77,
            constant_opt_enabled=False,
            simplification_enabled=False,
            semantic_dedup_enabled=False,
        )
        context = _make_context()

        r1 = evolve(reg, config, MSEEvaluator(lambda c: c["x"]), context)
        r2 = evolve(reg, config, MSEEvaluator(lambda c: c["x"]), context)
        assert r1.best_program == r2.best_program
        assert r1.fitness_history == r2.fitness_history


# ===========================================================================
# Simplification integration
# ===========================================================================


class TestSimplificationIntegration:
    """Simplification is applied during evolution when enabled."""

    def test_simplification_reduces_bloat(self) -> None:
        reg = _full_registry()

        config_no_simp = GPConfig(
            population_size=30,
            max_depth=5,
            generations=8,
            seed=42,
            simplification_enabled=False,
            constant_opt_enabled=False,
            semantic_dedup_enabled=False,
        )
        config_with_simp = GPConfig(
            population_size=30,
            max_depth=5,
            generations=8,
            seed=42,
            simplification_enabled=True,
            constant_opt_enabled=False,
            semantic_dedup_enabled=False,
        )
        context = _make_context()
        target = lambda c: c["x"] + c["x"]

        r_no = evolve(reg, config_no_simp, MSEEvaluator(target), context)
        r_yes = evolve(reg, config_with_simp, MSEEvaluator(target), context)

        # With simplification, mean program size should be <= without
        no_mean = r_no.fitness_history[-1].mean_program_size
        yes_mean = r_yes.fitness_history[-1].mean_program_size
        assert yes_mean <= no_mean + 2  # Allow small tolerance


# ===========================================================================
# Early stopping integration
# ===========================================================================


class TestEarlyStoppingIntegration:
    """Early stopping halts evolution before max generations."""

    def test_early_stop_triggers(self) -> None:
        reg = _full_registry()
        config = GPConfig(
            population_size=20,
            max_depth=4,
            generations=50,
            seed=42,
            early_stop_patience=2,
            early_stop_threshold=1e10,  # Absurdly high = always stalled
            constant_opt_enabled=False,
            simplification_enabled=False,
            semantic_dedup_enabled=False,
        )
        context = _make_context()
        evaluator = MSEEvaluator(lambda c: c["x"])
        result = evolve(reg, config, evaluator, context)
        assert len(result.fitness_history) < 50

    def test_no_early_stop_runs_full(self) -> None:
        reg = _full_registry()
        config = GPConfig(
            population_size=20,
            max_depth=4,
            generations=5,
            seed=42,
            early_stop_patience=None,
            constant_opt_enabled=False,
            simplification_enabled=False,
            semantic_dedup_enabled=False,
        )
        context = _make_context()
        evaluator = MSEEvaluator(lambda c: c["x"])
        result = evolve(reg, config, evaluator, context)
        assert len(result.fitness_history) == 5


# ===========================================================================
# Semantic dedup integration
# ===========================================================================


class TestSemanticDedupIntegration:
    """Semantic dedup replaces duplicates during evolution."""

    def test_dedup_enabled_runs_successfully(self) -> None:
        reg = _full_registry()
        config = GPConfig(
            population_size=20,
            max_depth=3,
            generations=5,
            seed=42,
            semantic_dedup_enabled=True,
            semantic_ref_size=10,
            constant_opt_enabled=False,
            simplification_enabled=False,
        )
        context = _make_context()
        evaluator = MSEEvaluator(lambda c: c["x"])
        result = evolve(reg, config, evaluator, context)
        assert isinstance(result, EvolutionResult)
        assert len(result.fitness_history) == 5


# ===========================================================================
# Serialization round-trip
# ===========================================================================


class TestSerializationIntegration:
    """Best program from evolution can be serialized and deserialized."""

    def test_best_program_round_trips(self) -> None:
        reg = _full_registry()
        config = GPConfig(
            population_size=20,
            max_depth=4,
            generations=5,
            seed=42,
            constant_opt_enabled=False,
            simplification_enabled=False,
            semantic_dedup_enabled=False,
        )
        context = _make_context()
        evaluator = MSEEvaluator(lambda c: 2.0 * c["x"])
        result = evolve(reg, config, evaluator, context)

        # Serialize -> deserialize
        d = serialize(result.best_program)
        json_str = json.dumps(d)
        loaded = deserialize(json.loads(json_str), reg)

        assert loaded == result.best_program

        # Evaluate both and compare
        output_orig = evaluate(result.best_program, context)
        output_loaded = evaluate(loaded, context)
        np.testing.assert_array_equal(output_orig, output_loaded)


# ===========================================================================
# Multi-objective (NSGA-II) integration
# ===========================================================================


class MultiObjectiveEvaluator:
    """Evaluator returning two objectives: accuracy + complexity proxy."""

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
                results.append(FitnessResult(objectives=(-mse, float(prog.size))))
            except Exception:
                results.append(FitnessResult(objectives=(-1e10, 100.0)))
        return results


class TestMultiObjectiveIntegration:
    """NSGA-II selection with multi-objective fitness."""

    def test_nsga2_evolves(self) -> None:
        reg = _full_registry()
        config = GPConfig(
            population_size=20,
            max_depth=4,
            generations=5,
            seed=42,
            selection_mode="nsga2",
            constant_opt_enabled=False,
            simplification_enabled=False,
            semantic_dedup_enabled=False,
            fitness=FitnessConfig(
                objectives=["accuracy", "complexity"],
                objective_directions=["maximize", "minimize"],
            ),
        )
        context = _make_context()
        evaluator = MultiObjectiveEvaluator(lambda c: c["x"])
        result = evolve(reg, config, evaluator, context)

        assert isinstance(result, EvolutionResult)
        assert len(result.pareto_front) >= 1
        assert len(result.fitness_history) == 5

        # Verify stats have 2 objectives
        for s in result.fitness_history:
            assert len(s.best_fitness) == 2
            assert len(s.mean_fitness) == 2


# ===========================================================================
# Constant optimization integration
# ===========================================================================


class TestConstantOptIntegration:
    """Constant optimization improves programs with tunable constants."""

    def test_constant_opt_runs(self) -> None:
        reg = _full_registry()
        config = GPConfig(
            population_size=20,
            max_depth=4,
            generations=3,
            seed=42,
            constant_opt_enabled=True,
            constant_opt_top_k=0.2,
            constant_opt_max_iter=5,
            constant_opt_max_time_seconds=1.0,
            simplification_enabled=False,
            semantic_dedup_enabled=False,
        )
        context = _make_context()
        evaluator = MSEEvaluator(lambda c: 3.5 * c["x"])
        result = evolve(reg, config, evaluator, context)
        assert isinstance(result, EvolutionResult)
        assert len(result.fitness_history) == 3


# ===========================================================================
# Config is passed through
# ===========================================================================


class TestConfigPassthrough:
    """EvolutionResult.config is the original config."""

    def test_config_preserved(self) -> None:
        reg = _full_registry()
        config = GPConfig(
            population_size=20,
            max_depth=4,
            generations=3,
            seed=42,
            constant_opt_enabled=False,
            simplification_enabled=False,
            semantic_dedup_enabled=False,
        )
        context = _make_context()
        evaluator = MSEEvaluator(lambda c: c["x"])
        result = evolve(reg, config, evaluator, context)
        assert result.config is config
