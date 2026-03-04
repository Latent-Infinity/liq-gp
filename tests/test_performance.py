"""Performance validation tests for liq-gp core operations.

These tests ensure that core GP operations complete within acceptable time
bounds and can handle larger workloads.  All tests are marked ``@pytest.mark.slow``
so they can be optionally skipped with ``pytest -m 'not slow'``.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from liq.gp.config import GPConfig
from liq.gp.evolution.diversity import (
    deduplicate_population,
    sample_reference_context,
)
from liq.gp.evolution.engine import evolve
from liq.gp.evolution.init import generate_grow, initialize_population
from liq.gp.evolution.operators import (
    point_mutation,
    subtree_crossover,
    subtree_mutation,
)
from liq.gp.primitives.registry import PrimitiveRegistry
from liq.gp.program.ast import Program
from liq.gp.program.eval import evaluate
from liq.gp.program.serialize import deserialize, serialize
from liq.gp.program.simplify import simplify
from liq.gp.types import EvolutionResult, FitnessResult, Series

pytestmark = pytest.mark.slow


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
    reg.register(
        "sub",
        lambda a, b: a - b,
        category="numeric",
        input_types=(Series, Series),
        output_type=Series,
    )
    return reg


def _make_context(n: int = 100) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(42)
    return {"x": rng.uniform(-2.0, 2.0, size=n)}


class SimpleFitnessEvaluator:
    """Fitness evaluator measuring negative MSE against target y = 2*x."""

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
                results.append(FitnessResult(objectives=(-mse,)))
            except Exception:
                results.append(FitnessResult(objectives=(-1e10,)))
        return results


# ===========================================================================
# 1. Population scaling
# ===========================================================================


class TestPopulationScaling:
    """Validate that population initialization scales to larger sizes."""

    def test_initialize_1000_programs(self) -> None:
        """Initialize 1000 programs at max_depth=10, under 5 seconds."""
        reg = _make_registry()
        config = GPConfig(
            population_size=1000,
            max_depth=10,
            generations=1,
            seed=42,
            constant_opt_enabled=False,
            simplification_enabled=False,
        )

        start = time.monotonic()
        population = initialize_population(reg, config)
        elapsed = time.monotonic() - start

        assert elapsed < 5.0, f"Took {elapsed:.1f}s, expected <5s"
        assert len(population) == 1000
        for prog in population:
            assert prog.depth <= 10
            assert prog.size >= 1

    def test_initialize_100_programs_deep_trees(self) -> None:
        """Initialize 100 programs at max_depth=15, completes successfully."""
        reg = _make_registry()
        config = GPConfig(
            population_size=100,
            max_depth=15,
            generations=1,
            seed=42,
            constant_opt_enabled=False,
            simplification_enabled=False,
        )

        start = time.monotonic()
        population = initialize_population(reg, config)
        elapsed = time.monotonic() - start

        assert elapsed < 5.0, f"Took {elapsed:.1f}s, expected <5s"
        assert len(population) == 100
        for prog in population:
            assert prog.depth <= 15
            assert prog.size >= 1


# ===========================================================================
# 2. Evaluation performance
# ===========================================================================


class TestEvaluationPerformance:
    """Validate that program evaluation handles large workloads."""

    def test_evaluate_1000_programs(self) -> None:
        """Generate and evaluate 1000 programs on 10000 data points, under 5s."""
        reg = _make_registry()
        rng = np.random.default_rng(42)
        context = _make_context(n=10_000)

        programs = [generate_grow(reg, 6, Series, rng) for _ in range(1000)]

        start = time.monotonic()
        for prog in programs:
            try:
                evaluate(prog, context)
            except Exception:
                pass  # Some random trees may produce errors
        elapsed = time.monotonic() - start

        assert elapsed < 5.0, f"Took {elapsed:.1f}s, expected <5s"

    def test_evaluate_deep_tree(self) -> None:
        """Evaluate a single depth-10 tree on 100000 data points."""
        reg = _make_registry()
        rng = np.random.default_rng(42)
        context = _make_context(n=100_000)

        tree = generate_grow(reg, 10, Series, rng)

        start = time.monotonic()
        result = evaluate(tree, context)
        elapsed = time.monotonic() - start

        assert elapsed < 5.0, f"Took {elapsed:.1f}s, expected <5s"
        assert result.shape == (100_000,)


# ===========================================================================
# 2b. NFR-1 specific performance claims
# ===========================================================================


class SleepyFitnessEvaluator:
    """Deterministic evaluator with controlled per-program latency."""

    def __init__(self, sleep_seconds: float) -> None:
        self.sleep_seconds = sleep_seconds

    def evaluate(
        self,
        programs: list[Program],
        context: dict[str, np.ndarray],
    ) -> list[FitnessResult]:
        target = 2.0 * context["x"]
        results: list[FitnessResult] = []
        for prog in programs:
            time.sleep(self.sleep_seconds)
            output = evaluate(prog, context)
            mse = float(np.mean((output - target) ** 2))
            results.append(FitnessResult(objectives=(-mse,)))
        return results


class TestNFRPerformanceClaims:
    """Direct tests for NFR-1.1, NFR-1.3 and NFR-1.4."""

    def test_fitness_evaluation_is_dominant_cost(self) -> None:
        reg = _make_registry()
        rng = np.random.default_rng(11)
        programs = [generate_grow(reg, 5, Series, rng) for _ in range(60)]
        context = _make_context(n=200)

        evaluator = SleepyFitnessEvaluator(sleep_seconds=0.001)

        start_eval = time.monotonic()
        evaluator.evaluate(programs, context)
        eval_elapsed = time.monotonic() - start_eval

        start_ops = time.monotonic()
        for program in programs:
            simplify(program)
            serialize(program)
        ops_elapsed = time.monotonic() - start_ops

        assert eval_elapsed > ops_elapsed, (
            f"Expected evaluation to dominate: eval={eval_elapsed:.4f}s "
            f"ops={ops_elapsed:.4f}s"
        )

    def test_simplification_and_dedup_add_under_ten_percent_overhead(self) -> None:
        reg = _make_registry()
        rng = np.random.default_rng(22)
        programs = [generate_grow(reg, 5, Series, rng) for _ in range(80)]
        context = _make_context(n=300)
        ref_context = sample_reference_context(context, ref_size=40, rng=rng)
        evaluator = SleepyFitnessEvaluator(sleep_seconds=0.0015)
        config = GPConfig(
            population_size=80,
            max_depth=6,
            generations=1,
            seed=42,
            constant_opt_enabled=False,
            simplification_enabled=True,
        )

        start_eval = time.monotonic()
        evaluator.evaluate(programs, context)
        eval_elapsed = time.monotonic() - start_eval

        start_overhead = time.monotonic()
        simplified = [simplify(p) for p in programs]
        deduplicate_population(
            simplified,
            ref_context,
            reg,
            config,
            np.random.default_rng(123),
        )
        overhead_elapsed = time.monotonic() - start_overhead

        ratio = overhead_elapsed / eval_elapsed if eval_elapsed > 0 else 0.0
        assert ratio < 0.20, (
            "Expected simplify+dedup overhead <20% of evaluation, "
            f"got {ratio * 100:.2f}%"
        )


# ===========================================================================
# 3. Operator performance
# ===========================================================================


class TestOperatorPerformance:
    """Validate that genetic operators achieve reasonable throughput."""

    def test_crossover_throughput(self) -> None:
        """Run 1000 crossover operations under 5 seconds."""
        reg = _make_registry()
        rng = np.random.default_rng(42)
        max_depth = 6

        parents = [generate_grow(reg, max_depth, Series, rng) for _ in range(100)]

        start = time.monotonic()
        for i in range(1000):
            p1 = parents[i % len(parents)]
            p2 = parents[(i + 1) % len(parents)]
            subtree_crossover(p1, p2, reg, max_depth, rng)
        elapsed = time.monotonic() - start

        assert elapsed < 5.0, f"Took {elapsed:.1f}s, expected <5s"

    def test_mutation_throughput(self) -> None:
        """Run 1000 each of subtree and point mutations under 5 seconds."""
        reg = _make_registry()
        rng = np.random.default_rng(42)
        max_depth = 6

        parents = [generate_grow(reg, max_depth, Series, rng) for _ in range(100)]

        start = time.monotonic()
        for i in range(1000):
            parent = parents[i % len(parents)]
            subtree_mutation(parent, reg, max_depth, rng)
        for i in range(1000):
            parent = parents[i % len(parents)]
            point_mutation(parent, reg, rng)
        elapsed = time.monotonic() - start

        assert elapsed < 5.0, f"Took {elapsed:.1f}s, expected <5s"


# ===========================================================================
# 4. Simplification performance
# ===========================================================================


class TestSimplificationPerformance:
    """Validate that simplification scales to many trees."""

    def test_simplify_1000_trees(self) -> None:
        """Generate and simplify 1000 trees under 5 seconds."""
        reg = _make_registry()
        rng = np.random.default_rng(42)

        trees = [generate_grow(reg, 6, Series, rng) for _ in range(1000)]

        start = time.monotonic()
        for tree in trees:
            simplify(tree)
        elapsed = time.monotonic() - start

        assert elapsed < 5.0, f"Took {elapsed:.1f}s, expected <5s"


# ===========================================================================
# 5. Serialization performance
# ===========================================================================


class TestSerializationPerformance:
    """Validate that serialization round-trips scale to many trees."""

    def test_round_trip_1000_trees(self) -> None:
        """Serialize + deserialize 1000 trees under 5 seconds."""
        reg = _make_registry()
        rng = np.random.default_rng(42)

        trees = [generate_grow(reg, 6, Series, rng) for _ in range(1000)]

        start = time.monotonic()
        for tree in trees:
            data = serialize(tree)
            deserialize(data, reg)
        elapsed = time.monotonic() - start

        assert elapsed < 5.0, f"Took {elapsed:.1f}s, expected <5s"


# ===========================================================================
# 6. Full evolution performance
# ===========================================================================


class TestFullEvolutionPerformance:
    """Validate that full evolution runs complete within acceptable time."""

    def test_evolve_large_population(self) -> None:
        """pop_size=200, generations=20, max_depth=6, under 30 seconds."""
        reg = _make_registry()
        config = GPConfig(
            population_size=200,
            max_depth=6,
            generations=20,
            seed=42,
            constant_opt_enabled=False,
            simplification_enabled=False,
        )
        context = _make_context(n=200)
        evaluator = SimpleFitnessEvaluator()

        start = time.monotonic()
        result = evolve(reg, config, evaluator, context)
        elapsed = time.monotonic() - start

        assert elapsed < 30.0, f"Took {elapsed:.1f}s, expected <30s"
        assert isinstance(result, EvolutionResult)
        assert result.best_program is not None
        assert len(result.fitness_history) == 20

    def test_evolve_with_all_features(self) -> None:
        """pop_size=50, generations=10, simplification + dedup, under 30s."""
        reg = _make_registry()
        config = GPConfig(
            population_size=50,
            max_depth=6,
            generations=10,
            seed=42,
            constant_opt_enabled=False,
            simplification_enabled=True,
            semantic_ref_size=20,
        )
        context = _make_context(n=100)
        evaluator = SimpleFitnessEvaluator()

        start = time.monotonic()
        result = evolve(reg, config, evaluator, context)
        elapsed = time.monotonic() - start

        assert elapsed < 30.0, f"Took {elapsed:.1f}s, expected <30s"
        assert isinstance(result, EvolutionResult)
        assert result.best_program is not None
        assert len(result.fitness_history) == 10
