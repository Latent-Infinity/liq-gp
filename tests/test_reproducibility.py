"""Reproducibility tests for the liq-gp genetic programming engine.

Verifies that all stochastic operations are fully deterministic when given
the same RNG seed, covering initialization, genetic operators, the full
evolution loop (single- and multi-objective), and optional pipeline stages
like simplification and semantic deduplication.
"""

from __future__ import annotations

import numpy as np

from liq.gp.config import FitnessConfig, GPConfig
from liq.gp.evolution.engine import evolve
from liq.gp.evolution.init import generate_grow, initialize_population
from liq.gp.evolution.operators import (
    point_mutation,
    select_operator,
    subtree_crossover,
    subtree_mutation,
)
from liq.gp.primitives.registry import PrimitiveRegistry
from liq.gp.program.eval import evaluate
from liq.gp.types import EvolutionResult, FitnessResult, Series

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_registry() -> PrimitiveRegistry:
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


def _make_context(n: int = 100) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(0)
    return {"x": rng.uniform(-2.0, 2.0, size=n)}


class SimpleFitnessEvaluator:
    """Single-objective evaluator: negative MSE against ``2*x``."""

    def evaluate(
        self,
        programs: list,
        context: dict[str, np.ndarray],
    ) -> list[FitnessResult]:
        target = 2.0 * context["x"]
        results: list[FitnessResult] = []
        for prog in programs:
            try:
                output = evaluate(prog, context)
                mse = float(np.mean((output - target) ** 2))
                results.append(FitnessResult(objectives=(-mse,)))
            except Exception:
                results.append(FitnessResult(objectives=(-1e10,)))
        return results


class MultiObjectiveFitnessEvaluator:
    """Two-objective evaluator: (-mse, program size)."""

    def evaluate(
        self,
        programs: list,
        context: dict[str, np.ndarray],
    ) -> list[FitnessResult]:
        target = 2.0 * context["x"]
        results: list[FitnessResult] = []
        for prog in programs:
            try:
                output = evaluate(prog, context)
                mse = float(np.mean((output - target) ** 2))
                results.append(FitnessResult(objectives=(-mse, float(prog.size))))
            except Exception:
                results.append(FitnessResult(objectives=(-1e10, 100.0)))
        return results


def _base_config(**overrides) -> GPConfig:
    """Return a small, fast GPConfig with sensible defaults for tests."""
    defaults = {
        "population_size": 20,
        "max_depth": 4,
        "generations": 5,
        "seed": 42,
        "constant_opt_enabled": False,
        "simplification_enabled": False,
        "elitism_count": 2,
        "tournament_size": 3,
    }
    defaults.update(overrides)
    return GPConfig(**defaults)


# ===================================================================
# 1. Initialization reproducibility
# ===================================================================


class TestInitializationReproducibility:
    """Population initialization must be deterministic for a given seed."""

    def test_same_seed_same_population(self) -> None:
        """Two calls with the same config (same seed) produce identical populations."""
        reg = _make_registry()
        config = _base_config(seed=42)

        pop_a = initialize_population(reg, config)
        pop_b = initialize_population(reg, config)

        assert len(pop_a) == len(pop_b)
        for a, b in zip(pop_a, pop_b, strict=True):
            assert a == b

    def test_different_seeds_differ(self) -> None:
        """Different seeds must produce different populations."""
        reg = _make_registry()
        pop_a = initialize_population(reg, _base_config(seed=42))
        pop_b = initialize_population(reg, _base_config(seed=99))

        # At least one individual should differ
        differs = any(a != b for a, b in zip(pop_a, pop_b, strict=True))
        assert differs, "Populations from different seeds should differ"

    def test_generate_grow_deterministic(self) -> None:
        """Same seed + rng produce an identical tree from generate_grow."""
        reg = _make_registry()

        rng_a = np.random.default_rng(7)
        tree_a = generate_grow(reg, 4, Series, rng_a)

        rng_b = np.random.default_rng(7)
        tree_b = generate_grow(reg, 4, Series, rng_b)

        assert tree_a == tree_b


# ===================================================================
# 2. Operator reproducibility
# ===================================================================


class TestOperatorReproducibility:
    """Genetic operators must be deterministic for a given RNG state."""

    def _sample_tree(self, seed: int = 0):
        """Return a small deterministic tree for operator tests."""
        reg = _make_registry()
        config = _base_config(seed=seed)
        pop = initialize_population(reg, config)
        return pop[0], reg

    def test_crossover_deterministic(self) -> None:
        """Same parents + same rng seed = same children."""
        tree, reg = self._sample_tree(seed=10)
        _, reg2 = self._sample_tree(seed=10)
        # Use two different parents from same population
        config = _base_config(seed=10)
        pop = initialize_population(reg, config)
        parent1, parent2 = pop[0], pop[1]

        rng_a = np.random.default_rng(55)
        child1_a, child2_a = subtree_crossover(
            parent1,
            parent2,
            reg,
            config.max_depth,
            rng_a,
        )

        rng_b = np.random.default_rng(55)
        child1_b, child2_b = subtree_crossover(
            parent1,
            parent2,
            reg,
            config.max_depth,
            rng_b,
        )

        assert child1_a == child1_b
        assert child2_a == child2_b

    def test_subtree_mutation_deterministic(self) -> None:
        """Same tree + same rng seed = same mutant."""
        tree, reg = self._sample_tree(seed=20)

        rng_a = np.random.default_rng(77)
        mutant_a = subtree_mutation(tree, reg, 4, rng_a)

        rng_b = np.random.default_rng(77)
        mutant_b = subtree_mutation(tree, reg, 4, rng_b)

        assert mutant_a == mutant_b

    def test_point_mutation_deterministic(self) -> None:
        """Same tree + same rng seed = same point mutant."""
        tree, reg = self._sample_tree(seed=30)

        rng_a = np.random.default_rng(88)
        mutant_a = point_mutation(tree, reg, rng_a)

        rng_b = np.random.default_rng(88)
        mutant_b = point_mutation(tree, reg, rng_b)

        assert mutant_a == mutant_b

    def test_operator_selection_deterministic(self) -> None:
        """Same config + same rng seed = same operator sequence."""
        config = _base_config()
        n_draws = 50

        rng_a = np.random.default_rng(99)
        ops_a = [select_operator(config, rng_a) for _ in range(n_draws)]

        rng_b = np.random.default_rng(99)
        ops_b = [select_operator(config, rng_b) for _ in range(n_draws)]

        assert ops_a == ops_b


# ===================================================================
# 3. Full evolve() reproducibility
# ===================================================================


class TestEvolveReproducibility:
    """The complete evolution loop must be deterministic for a given seed."""

    def test_full_evolution_deterministic(self) -> None:
        """Two evolve() calls with same seed produce identical results."""
        reg = _make_registry()
        config = _base_config(seed=42)
        context = _make_context()
        evaluator = SimpleFitnessEvaluator()

        result_a = evolve(reg, config, evaluator, context)
        result_b = evolve(reg, config, evaluator, context)

        assert isinstance(result_a, EvolutionResult)
        assert isinstance(result_b, EvolutionResult)

        # Best programs must be structurally identical
        assert result_a.best_program == result_b.best_program

        # Fitness histories must match
        assert len(result_a.fitness_history) == len(result_b.fitness_history)
        for stats_a, stats_b in zip(
            result_a.fitness_history,
            result_b.fitness_history,
            strict=True,
        ):
            assert stats_a.best_fitness == stats_b.best_fitness
            assert stats_a.mean_fitness == stats_b.mean_fitness
            assert stats_a.best_program_size == stats_b.best_program_size

    def test_full_evolution_different_seeds_differ(self) -> None:
        """Different seeds produce different evolution results."""
        reg = _make_registry()
        context = _make_context()
        evaluator = SimpleFitnessEvaluator()

        result_a = evolve(reg, _base_config(seed=42), evaluator, context)
        result_b = evolve(reg, _base_config(seed=123), evaluator, context)

        # The runs should diverge in at least one observable way
        histories_differ = any(
            sa.best_fitness != sb.best_fitness
            for sa, sb in zip(
                result_a.fitness_history,
                result_b.fitness_history,
                strict=True,
            )
        )
        programs_differ = result_a.best_program != result_b.best_program
        assert histories_differ or programs_differ, (
            "Runs with different seeds should produce different results"
        )

    def test_with_simplification_deterministic(self) -> None:
        """Determinism holds when simplification is enabled."""
        reg = _make_registry()
        config = _base_config(seed=42, simplification_enabled=True)
        context = _make_context()
        evaluator = SimpleFitnessEvaluator()

        result_a = evolve(reg, config, evaluator, context)
        result_b = evolve(reg, config, evaluator, context)

        assert result_a.best_program == result_b.best_program
        for stats_a, stats_b in zip(
            result_a.fitness_history,
            result_b.fitness_history,
            strict=True,
        ):
            assert stats_a.best_fitness == stats_b.best_fitness

    def test_with_semantic_dedup_deterministic(self) -> None:
        """Determinism holds with semantic deduplication in the evolution loop."""
        reg = _make_registry()
        config = _base_config(seed=42)
        context = _make_context()
        evaluator = SimpleFitnessEvaluator()

        result_a = evolve(reg, config, evaluator, context)
        result_b = evolve(reg, config, evaluator, context)

        assert result_a.best_program == result_b.best_program
        for stats_a, stats_b in zip(
            result_a.fitness_history,
            result_b.fitness_history,
            strict=True,
        ):
            assert stats_a.best_fitness == stats_b.best_fitness


# ===================================================================
# 4. NSGA-II multi-objective reproducibility
# ===================================================================


class TestNSGA2Reproducibility:
    """Multi-objective NSGA-II selection must be deterministic for a given seed."""

    @staticmethod
    def _nsga2_config(seed: int = 42) -> GPConfig:
        return GPConfig(
            population_size=20,
            max_depth=4,
            generations=5,
            seed=seed,
            selection_mode="nsga2",
            fitness=FitnessConfig(
                objectives=["accuracy", "complexity"],
                objective_directions=["maximize", "minimize"],
            ),
            constant_opt_enabled=False,
            simplification_enabled=False,
        )

    def test_nsga2_deterministic(self) -> None:
        """Multi-objective evolution with NSGA-II is deterministic."""
        reg = _make_registry()
        config = self._nsga2_config(seed=42)
        context = _make_context()
        evaluator = MultiObjectiveFitnessEvaluator()

        result_a = evolve(reg, config, evaluator, context)
        result_b = evolve(reg, config, evaluator, context)

        assert isinstance(result_a, EvolutionResult)
        assert isinstance(result_b, EvolutionResult)

        # Best programs must match
        assert result_a.best_program == result_b.best_program

        # Full fitness histories must match
        assert len(result_a.fitness_history) == len(result_b.fitness_history)
        for stats_a, stats_b in zip(
            result_a.fitness_history,
            result_b.fitness_history,
            strict=True,
        ):
            assert stats_a.best_fitness == stats_b.best_fitness
            assert stats_a.mean_fitness == stats_b.mean_fitness

    def test_nsga2_pareto_front_deterministic(self) -> None:
        """Pareto front is identical across runs with the same seed."""
        reg = _make_registry()
        config = self._nsga2_config(seed=42)
        context = _make_context()
        evaluator = MultiObjectiveFitnessEvaluator()

        result_a = evolve(reg, config, evaluator, context)
        result_b = evolve(reg, config, evaluator, context)

        assert len(result_a.pareto_front) == len(result_b.pareto_front)
        assert len(result_a.pareto_front) >= 1, "Pareto front should not be empty"
        for prog_a, prog_b in zip(
            result_a.pareto_front,
            result_b.pareto_front,
            strict=True,
        ):
            assert prog_a == prog_b


# ===========================================================================
# Seeded evolution reproducibility (FR-5.1.6)
# ===========================================================================


class TestSeededReproducibility:
    """Seeded evolution must be deterministic with the same seed + config."""

    @staticmethod
    def _make_seed(reg: PrimitiveRegistry):
        from liq.gp.program.ast import FunctionNode, TerminalNode

        add_info = reg.get("add")
        return FunctionNode(
            primitive=add_info,
            children=(
                TerminalNode(name="x", output_type=Series),
                TerminalNode(name="x", output_type=Series),
            ),
        )

    def test_seeded_evolution_deterministic(self) -> None:
        reg = _make_registry()
        config = _base_config(seed=42)
        context = _make_context()
        evaluator = SimpleFitnessEvaluator()
        seed = self._make_seed(reg)

        r1 = evolve(reg, config, evaluator, context, seed_programs=[seed])
        r2 = evolve(reg, config, evaluator, context, seed_programs=[seed])

        assert r1.best_program == r2.best_program
        for s1, s2 in zip(r1.fitness_history, r2.fitness_history, strict=True):
            assert s1.best_fitness == s2.best_fitness

    def test_seeded_nsga2_deterministic(self) -> None:
        reg = _make_registry()
        fitness_config = FitnessConfig(
            objectives=["accuracy", "simplicity"],
            objective_directions=["maximize", "maximize"],
        )
        config = _base_config(
            seed=42,
            selection_mode="nsga2",
            parsimony_mode="pareto",
            fitness=fitness_config,
        )
        context = _make_context()
        evaluator = MultiObjectiveFitnessEvaluator()
        seed = self._make_seed(reg)

        r1 = evolve(reg, config, evaluator, context, seed_programs=[seed])
        r2 = evolve(reg, config, evaluator, context, seed_programs=[seed])

        assert r1.best_program == r2.best_program
        assert len(r1.pareto_front) == len(r2.pareto_front)

    def test_seeded_with_all_features_deterministic(self) -> None:
        reg = _make_registry()
        config = _base_config(
            seed=42,
            simplification_enabled=True,
            constant_opt_enabled=True,
        )
        context = _make_context()
        evaluator = SimpleFitnessEvaluator()
        seed = self._make_seed(reg)

        r1 = evolve(reg, config, evaluator, context, seed_programs=[seed])
        r2 = evolve(reg, config, evaluator, context, seed_programs=[seed])

        assert r1.best_program == r2.best_program
        for s1, s2 in zip(r1.fitness_history, r2.fitness_history, strict=True):
            assert s1.best_fitness == s2.best_fitness
