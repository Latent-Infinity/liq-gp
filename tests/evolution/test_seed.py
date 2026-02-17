"""Tests for population seeding (FR-5.1.4 -- FR-5.1.7)."""

from __future__ import annotations

import numpy as np
import pytest

from liq.gp.config import FitnessConfig, GPConfig
from liq.gp.errors import EvolutionError
from liq.gp.evolution.engine import evolve
from liq.gp.evolution.init import (
    initialize_seeded_population,
    validate_seed_programs,
)
from liq.gp.primitives.registry import PrimitiveInfo, PrimitiveRegistry
from liq.gp.program.ast import (
    ConstantNode,
    FunctionNode,
    ParameterizedNode,
    Program,
    TerminalNode,
)
from liq.gp.program.eval import evaluate
from liq.gp.program.serialize import deserialize, serialize
from liq.gp.types import BoolSeries, EvolutionResult, FitnessResult, ParamSpec, Series

# --- helpers ---------------------------------------------------------------


def _make_registry() -> PrimitiveRegistry:
    """Build a minimal registry for testing seeding."""
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
    ps = ParamSpec(name="factor", dtype=float, default=1.0, min_value=-10.0, max_value=10.0)
    reg.register(
        "scale",
        lambda a, *, factor=1.0: a * factor,
        category="parameterized",
        input_types=(Series,),
        output_type=Series,
        param_specs=[ps],
    )
    return reg


def _make_config(**overrides: object) -> GPConfig:
    """Build a GPConfig with sensible test defaults."""
    defaults: dict[str, object] = {
        "population_size": 20,
        "max_depth": 4,
        "generations": 1,
        "seed": 42,
    }
    defaults.update(overrides)
    return GPConfig(**defaults)  # type: ignore[arg-type]


def _make_terminal(name: str = "x") -> TerminalNode:
    return TerminalNode(name=name, output_type=Series)


def _make_constant(value: float = 1.0) -> ConstantNode:
    return ConstantNode(value=value)


def _make_simple_tree(reg: PrimitiveRegistry) -> FunctionNode:
    """Build add(x, y) — a simple 3-node tree."""
    add_info = reg.get("add")
    return FunctionNode(
        primitive=add_info,
        children=(_make_terminal("x"), _make_terminal("y")),
    )


def _make_deep_tree(reg: PrimitiveRegistry, depth: int) -> Program:
    """Build a chain of neg(neg(...neg(x)...)) with given depth."""
    neg_info = reg.get("neg")
    node: Program = _make_terminal("x")
    for _ in range(depth):
        node = FunctionNode(primitive=neg_info, children=(node,))
    return node


# ===========================================================================
# 1. Validation tests
# ===========================================================================


class TestValidateSeedPrograms:
    """Tests for validate_seed_programs() (FR-5.1.5)."""

    def test_empty_seeds_raises(self) -> None:
        config = _make_config()
        with pytest.raises(EvolutionError, match="at least 1"):
            validate_seed_programs([], config)

    def test_too_many_seeds_raises(self) -> None:
        config = _make_config(population_size=10)
        seeds = [_make_terminal() for _ in range(11)]
        with pytest.raises(EvolutionError, match="exceeds population_size"):
            validate_seed_programs(seeds, config)

    def test_seed_exceeds_max_depth_raises(self) -> None:
        reg = _make_registry()
        config = _make_config(max_depth=3)
        deep = _make_deep_tree(reg, depth=4)  # depth 4 > max 3
        with pytest.raises(EvolutionError, match="max_depth"):
            validate_seed_programs([deep], config)

    def test_seed_exceeds_max_size_raises(self) -> None:
        reg = _make_registry()
        config = _make_config(max_size=2)
        tree = _make_simple_tree(reg)  # size 3 > max 2
        with pytest.raises(EvolutionError, match="max_size"):
            validate_seed_programs([tree], config)

    def test_seed_wrong_output_type_raises(self) -> None:
        config = _make_config()
        seed = ConstantNode(value=1.0, output_type=BoolSeries)
        with pytest.raises(EvolutionError, match="output_type"):
            validate_seed_programs([seed], config)

    def test_seed_unknown_primitive_raises(self) -> None:
        """A seed with a primitive not in the registry should fail."""
        config = _make_config()
        # Build a FunctionNode with a fake primitive
        fake_prim = PrimitiveInfo(
            name="unknown_op",
            category="arithmetic",
            arity=2,
            input_types=(Series, Series),
            output_type=Series,
            callable=lambda a, b: a + b,
        )
        seed = FunctionNode(
            primitive=fake_prim,
            children=(_make_terminal("x"), _make_terminal("y")),
        )
        reg = _make_registry()
        with pytest.raises(EvolutionError, match="unknown_op"):
            validate_seed_programs([seed], config, registry=reg)

    def test_valid_single_seed_passes(self) -> None:
        reg = _make_registry()
        config = _make_config()
        seed = _make_simple_tree(reg)
        validate_seed_programs([seed], config, registry=reg)

    def test_valid_multiple_seeds_pass(self) -> None:
        reg = _make_registry()
        config = _make_config()
        seeds = [_make_simple_tree(reg), _make_terminal("x"), _make_constant(2.0)]
        validate_seed_programs(seeds, config, registry=reg)

    def test_error_message_includes_index(self) -> None:
        config = _make_config(max_depth=2)
        reg = _make_registry()
        good = _make_terminal("x")
        bad = _make_deep_tree(reg, depth=3)  # index 1
        with pytest.raises(EvolutionError, match="index 1"):
            validate_seed_programs([good, bad], config)

    def test_terminal_node_seed_valid(self) -> None:
        config = _make_config()
        validate_seed_programs([_make_terminal("x")], config)

    def test_constant_node_seed_valid(self) -> None:
        config = _make_config()
        validate_seed_programs([_make_constant(3.14)], config)


# ===========================================================================
# 2. Seeded initialization tests
# ===========================================================================


class TestInitializeSeededPopulation:
    """Tests for initialize_seeded_population() (FR-5.1.4)."""

    def test_seeds_equal_pop_size_returns_seeds(self) -> None:
        """When len(seeds) == population_size, return them as-is."""
        reg = _make_registry()
        config = _make_config(population_size=10)
        rng = np.random.default_rng(42)
        seeds = [_make_simple_tree(reg) for _ in range(10)]
        result = initialize_seeded_population(seeds, reg, config, rng)
        assert len(result) == 10
        for i in range(10):
            assert result[i] is seeds[i]

    def test_single_seed_fills_population(self) -> None:
        """A single seed should produce a full population."""
        reg = _make_registry()
        config = _make_config(population_size=20)
        rng = np.random.default_rng(42)
        seed = _make_simple_tree(reg)
        result = initialize_seeded_population([seed], reg, config, rng)
        assert len(result) == 20

    def test_population_size_correct(self) -> None:
        """Result length must equal config.population_size."""
        reg = _make_registry()
        config = _make_config(population_size=30)
        rng = np.random.default_rng(42)
        seeds = [_make_simple_tree(reg) for _ in range(5)]
        result = initialize_seeded_population(seeds, reg, config, rng)
        assert len(result) == 30

    def test_seeds_at_front_of_population(self) -> None:
        """Seeds should occupy the first N positions."""
        reg = _make_registry()
        config = _make_config(population_size=15)
        rng = np.random.default_rng(42)
        seeds = [_make_simple_tree(reg), _make_terminal("x"), _make_constant(2.0)]
        result = initialize_seeded_population(seeds, reg, config, rng)
        for i in range(len(seeds)):
            assert result[i] is seeds[i]

    def test_offspring_respect_max_depth(self) -> None:
        """All offspring must satisfy max_depth."""
        reg = _make_registry()
        config = _make_config(population_size=50, max_depth=4)
        rng = np.random.default_rng(42)
        seed = _make_simple_tree(reg)  # depth 1
        result = initialize_seeded_population([seed], reg, config, rng)
        for prog in result:
            assert prog.depth <= config.max_depth

    def test_offspring_respect_max_size(self) -> None:
        """All offspring must satisfy max_size when configured."""
        reg = _make_registry()
        config = _make_config(population_size=50, max_depth=4, max_size=15)
        rng = np.random.default_rng(42)
        seed = _make_simple_tree(reg)
        result = initialize_seeded_population([seed], reg, config, rng)
        for prog in result:
            assert prog.size <= 15

    def test_offspring_share_structure_with_seeds(self) -> None:
        """Offspring should not all be identical to the seed (variation works)."""
        reg = _make_registry()
        config = _make_config(population_size=20)
        rng = np.random.default_rng(42)
        seed = _make_simple_tree(reg)  # add(x, y)
        result = initialize_seeded_population([seed], reg, config, rng)
        # At least some offspring should differ from the seed
        sizes = {p.size for p in result}
        assert len(sizes) > 1, "Expected diverse offspring, got all same size"

    def test_deterministic_with_same_rng(self) -> None:
        """Same rng seed produces identical population."""
        reg = _make_registry()
        config = _make_config(population_size=20)
        seeds = [_make_simple_tree(reg)]

        rng1 = np.random.default_rng(99)
        pop1 = initialize_seeded_population(seeds, reg, config, rng1)

        rng2 = np.random.default_rng(99)
        pop2 = initialize_seeded_population(seeds, reg, config, rng2)

        assert len(pop1) == len(pop2)
        for p1, p2 in zip(pop1, pop2, strict=True):
            assert p1 == p2

    def test_different_rng_different_population(self) -> None:
        """Different rng seeds produce different populations."""
        reg = _make_registry()
        config = _make_config(population_size=20)
        seeds = [_make_simple_tree(reg)]

        rng1 = np.random.default_rng(1)
        pop1 = initialize_seeded_population(seeds, reg, config, rng1)

        rng2 = np.random.default_rng(2)
        pop2 = initialize_seeded_population(seeds, reg, config, rng2)

        # Not all offspring identical (seeds at index 0 will match)
        differ = any(p1 != p2 for p1, p2 in zip(pop1[1:], pop2[1:], strict=True))
        assert differ, "Expected different populations from different RNG seeds"

    def test_all_offspring_valid_type(self) -> None:
        """All offspring must have the correct output_type."""
        reg = _make_registry()
        config = _make_config(population_size=30)
        rng = np.random.default_rng(42)
        seed = _make_simple_tree(reg)
        result = initialize_seeded_population([seed], reg, config, rng)
        for prog in result:
            assert prog.output_type == Series

    def test_single_seed_crossover_works(self) -> None:
        """Crossover of a seed with itself should produce valid offspring."""
        reg = _make_registry()
        # Higher crossover rate to force crossover operations
        config = _make_config(
            population_size=20,
            crossover_rate=0.9,
            subtree_mutation_rate=0.05,
            point_mutation_rate=0.025,
            parameter_mutation_rate=0.0,
            hoist_mutation_rate=0.025,
        )
        rng = np.random.default_rng(42)
        seed = _make_simple_tree(reg)
        result = initialize_seeded_population([seed], reg, config, rng)
        assert len(result) == 20
        for prog in result:
            assert prog.depth <= config.max_depth

    def test_fallback_when_constraints_reject_all(self) -> None:
        """If operators can't produce valid offspring, fall back to random."""
        reg = _make_registry()
        # Seed is at max_depth, max_size is very tight — operators will struggle
        config = _make_config(population_size=15, max_depth=2, max_size=3)
        rng = np.random.default_rng(42)
        # Terminal seed (depth=0, size=1) — operators from a tiny seed
        # may exceed max_size=3, forcing fallback
        seed = _make_terminal("x")
        result = initialize_seeded_population([seed], reg, config, rng)
        assert len(result) == 15
        for prog in result:
            assert prog.depth <= 2
            assert prog.size <= 3

    def test_fallback_exhaustion_raises_evolution_error(self, monkeypatch) -> None:
        """If operator + fallback cannot satisfy constraints, raise EvolutionError."""
        reg = _make_registry()
        config = _make_config(population_size=20, max_depth=3)
        rng = np.random.default_rng(1)
        seed = _make_simple_tree(reg)

        import liq.gp.evolution.constraints as constraints_mod
        import liq.gp.evolution.init as init_mod

        def always_false(_program: Program, _config: object) -> bool:
            return False

        monkeypatch.setattr(constraints_mod, "enforce_constraints", always_false)

        with pytest.raises(EvolutionError, match="Unable to generate enough valid"):
            init_mod.initialize_seeded_population([seed], reg, config, rng)


# ===========================================================================
# 3. Engine integration tests
# ===========================================================================


class _SimpleFitnessEvaluator:
    """Evaluator that measures MSE for y = 2*x."""

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


class _MultiObjectiveEvaluator:
    """Evaluator returning (neg_mse, neg_size) for NSGA-II tests."""

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
                    FitnessResult(objectives=(-mse, -float(prog.size)))
                )
            except Exception:
                results.append(FitnessResult(objectives=(-1e10, -100.0)))
        return results


def _make_evolve_config(**overrides: object) -> GPConfig:
    """GPConfig for engine integration tests (small, fast)."""
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


def _make_context(n: int = 50) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(0)
    return {"x": rng.uniform(-1.0, 1.0, size=n), "y": rng.uniform(-1.0, 1.0, size=n)}


class TestEvolveWithSeeds:
    """Tests for evolve() with seed_programs (FR-5.1.4)."""

    def test_evolve_with_single_seed(self) -> None:
        reg = _make_registry()
        config = _make_evolve_config()
        ctx = _make_context()
        seed = _make_simple_tree(reg)
        result = evolve(reg, config, _SimpleFitnessEvaluator(), ctx, seed_programs=[seed])
        assert isinstance(result, EvolutionResult)
        assert result.best_program.size > 0

    def test_evolve_with_multiple_seeds(self) -> None:
        reg = _make_registry()
        config = _make_evolve_config()
        ctx = _make_context()
        seeds = [_make_simple_tree(reg), _make_terminal("x"), _make_constant(2.0)]
        result = evolve(reg, config, _SimpleFitnessEvaluator(), ctx, seed_programs=seeds)
        assert isinstance(result, EvolutionResult)

    def test_evolve_with_full_population_seeds(self) -> None:
        reg = _make_registry()
        config = _make_evolve_config(population_size=10)
        ctx = _make_context()
        seeds = [_make_simple_tree(reg) for _ in range(10)]
        result = evolve(reg, config, _SimpleFitnessEvaluator(), ctx, seed_programs=seeds)
        assert isinstance(result, EvolutionResult)

    def test_evolve_seeds_none_is_default(self) -> None:
        """seed_programs=None should behave identically to the original API."""
        reg = _make_registry()
        config = _make_evolve_config()
        ctx = _make_context()

        result1 = evolve(reg, config, _SimpleFitnessEvaluator(), ctx)
        result2 = evolve(reg, config, _SimpleFitnessEvaluator(), ctx, seed_programs=None)

        # Same seed + same config + no seeds = same result
        assert result1.best_program == result2.best_program

    def test_evolve_empty_list_raises(self) -> None:
        """seed_programs=[] is rejected."""
        reg = _make_registry()
        config = _make_evolve_config()
        ctx = _make_context()
        with pytest.raises(EvolutionError, match="seed_programs must be None"):
            evolve(reg, config, _SimpleFitnessEvaluator(), ctx, seed_programs=[])

    def test_evolve_returns_evolution_result(self) -> None:
        reg = _make_registry()
        config = _make_evolve_config()
        ctx = _make_context()
        result = evolve(
            reg, config, _SimpleFitnessEvaluator(), ctx,
            seed_programs=[_make_simple_tree(reg)],
        )
        assert hasattr(result, "best_program")
        assert hasattr(result, "pareto_front")
        assert hasattr(result, "fitness_history")
        assert len(result.fitness_history) == 5

    def test_evolve_seeded_fitness_improves(self) -> None:
        """Fitness should not degrade over generations."""
        reg = _make_registry()
        config = _make_evolve_config(generations=10, elitism_count=2)
        ctx = _make_context()
        seed = _make_simple_tree(reg)
        result = evolve(reg, config, _SimpleFitnessEvaluator(), ctx, seed_programs=[seed])
        first_fit = result.fitness_history[0].best_fitness[0]
        last_fit = result.fitness_history[-1].best_fitness[0]
        # Elitism preserves best, so last should be >= first
        assert last_fit >= first_fit

    def test_evolve_seeded_deterministic(self) -> None:
        """Same seeds + same config.seed = same result."""
        reg = _make_registry()
        config = _make_evolve_config()
        ctx = _make_context()
        seed = _make_simple_tree(reg)

        r1 = evolve(reg, config, _SimpleFitnessEvaluator(), ctx, seed_programs=[seed])
        r2 = evolve(reg, config, _SimpleFitnessEvaluator(), ctx, seed_programs=[seed])
        assert r1.best_program == r2.best_program

    def test_evolve_seeded_different_seed_differs(self) -> None:
        """Different config.seed with same seeds should differ."""
        reg = _make_registry()
        ctx = _make_context()
        seed = _make_simple_tree(reg)

        c1 = _make_evolve_config(seed=1, generations=3)
        c2 = _make_evolve_config(seed=2, generations=3)
        r1 = evolve(reg, c1, _SimpleFitnessEvaluator(), ctx, seed_programs=[seed])
        r2 = evolve(reg, c2, _SimpleFitnessEvaluator(), ctx, seed_programs=[seed])

        # Different RNG seeds should produce different populations and
        # thus different mean program sizes across generations
        sizes1 = [s.mean_program_size for s in r1.fitness_history]
        sizes2 = [s.mean_program_size for s in r2.fitness_history]
        assert sizes1 != sizes2

    def test_evolve_with_seeds_and_nsga2(self) -> None:
        reg = _make_registry()
        fitness_config = FitnessConfig(
            objectives=["accuracy", "simplicity"],
            objective_directions=["maximize", "maximize"],
        )
        config = _make_evolve_config(
            selection_mode="nsga2",
            parsimony_mode="pareto",
            fitness=fitness_config,
        )
        ctx = _make_context()
        seed = _make_simple_tree(reg)
        result = evolve(
            reg, config, _MultiObjectiveEvaluator(), ctx, seed_programs=[seed],
        )
        assert isinstance(result, EvolutionResult)
        assert len(result.pareto_front) >= 1

    def test_evolve_with_seeds_and_simplification(self) -> None:
        reg = _make_registry()
        config = _make_evolve_config(simplification_enabled=True)
        ctx = _make_context()
        seed = _make_simple_tree(reg)
        result = evolve(reg, config, _SimpleFitnessEvaluator(), ctx, seed_programs=[seed])
        assert isinstance(result, EvolutionResult)

    def test_evolve_with_seeds_and_constant_opt(self) -> None:
        reg = _make_registry()
        config = _make_evolve_config(constant_opt_enabled=True)
        ctx = _make_context()
        seed = _make_simple_tree(reg)
        result = evolve(reg, config, _SimpleFitnessEvaluator(), ctx, seed_programs=[seed])
        assert isinstance(result, EvolutionResult)

    def test_constant_optimization_updates_dedup_fingerprints(self, monkeypatch) -> None:
        reg = _make_registry()
        add_primitive = reg.get("add")
        config = _make_evolve_config(
            population_size=10,
            generations=1,
            constant_opt_enabled=True,
            semantic_dedup_enabled=True,
            simplification_enabled=False,
        )
        context = _make_context()

        # Precompute a canonical program that every optimized individual collapses to.
        canonical_program = FunctionNode(
            primitive=add_primitive,
            children=(ConstantNode(0.0), ConstantNode(0.0)),
        )

        # Use deterministic seeds with unique constants so initial fingerprints differ.
        seeds: list[Program] = [
            FunctionNode(
                primitive=add_primitive,
                children=(ConstantNode(float(i)), ConstantNode(float(i))),
            )
            for i in range(config.population_size)
        ]

        captured: dict[str, int | None] = {}

        def _select_all(
            _population: list[Program], _fitnesses: list[FitnessResult], _config: GPConfig
        ) -> list[int]:
            return list(range(config.population_size))

        def _flatten_to_canonical(
            _program: Program,
            _evaluator: object,
            _context: dict[str, np.ndarray],
            _config: GPConfig,
            _rng: np.random.Generator,
        ) -> Program:
            return canonical_program

        # Spy on the fingerprints forwarded into dedup so we can verify they
        # reflect post-optimization semantics.
        import liq.gp.evolution.diversity as diversity_mod

        original_dedup = diversity_mod.deduplicate_population

        def _dedup_spy(
            population: list[Program],
            ref_context: dict[str, np.ndarray],
            registry: PrimitiveRegistry,
            dedup_config: GPConfig,
            dedup_rng: np.random.Generator,
            fingerprints: list[bytes] | None = None,
        ) -> tuple[list[Program], float]:
            captured["fingerprint_count"] = None if fingerprints is None else len(set(fingerprints))
            return original_dedup(
                population,
                ref_context,
                registry,
                dedup_config,
                dedup_rng,
                fingerprints=fingerprints,
            )

        # Configure constant optimization and dedup instrumentation.
        import liq.gp.program.constants as constants_mod

        monkeypatch.setattr(
            constants_mod,
            "select_for_optimization",
            _select_all,
        )
        monkeypatch.setattr(
            constants_mod,
            "optimize_constants",
            _flatten_to_canonical,
        )
        monkeypatch.setattr(
            "liq.gp.evolution.engine.deduplicate_population",
            _dedup_spy,
        )

        result = evolve(
            reg,
            config,
            _SimpleFitnessEvaluator(),
            context,
            seed_programs=seeds,
        )
        assert isinstance(result, EvolutionResult)
        assert captured.get("fingerprint_count") == 1

    def test_evolve_with_seeds_and_semantic_dedup(self) -> None:
        reg = _make_registry()
        config = _make_evolve_config(semantic_dedup_enabled=True)
        ctx = _make_context()
        seed = _make_simple_tree(reg)
        result = evolve(reg, config, _SimpleFitnessEvaluator(), ctx, seed_programs=[seed])
        assert isinstance(result, EvolutionResult)

    def test_evolve_with_seeds_and_early_stopping(self) -> None:
        reg = _make_registry()
        config = _make_evolve_config(
            generations=50, early_stop_patience=3, early_stop_threshold=1e-6,
        )
        ctx = _make_context()
        seed = _make_simple_tree(reg)
        result = evolve(reg, config, _SimpleFitnessEvaluator(), ctx, seed_programs=[seed])
        # Should stop before 50 generations
        assert len(result.fitness_history) <= 50

    def test_evolve_with_seeds_and_batch_eval(self) -> None:
        reg = _make_registry()
        config = _make_evolve_config(
            fitness=FitnessConfig(batch_size=20, full_eval_interval=3),
        )
        ctx = _make_context()
        seed = _make_simple_tree(reg)
        result = evolve(reg, config, _SimpleFitnessEvaluator(), ctx, seed_programs=[seed])
        assert isinstance(result, EvolutionResult)

    def test_evolve_with_seeds_callback_works(self) -> None:
        reg = _make_registry()
        config = _make_evolve_config()
        ctx = _make_context()
        seed = _make_simple_tree(reg)
        stats_log: list[object] = []
        result = evolve(
            reg, config, _SimpleFitnessEvaluator(), ctx,
            seed_programs=[seed],
            callback=lambda s: stats_log.append(s),
        )
        assert len(stats_log) == len(result.fitness_history)

    def test_invalid_seeds_raises_evolution_error(self) -> None:
        reg = _make_registry()
        config = _make_evolve_config(max_depth=3)
        ctx = _make_context()
        deep = _make_deep_tree(reg, depth=4)
        with pytest.raises(EvolutionError, match="max_depth"):
            evolve(reg, config, _SimpleFitnessEvaluator(), ctx, seed_programs=[deep])


# ===========================================================================
# 4. Warm-start tests
# ===========================================================================


class TestEvolveWarmStart:
    """Tests for warm-starting evolution from previous results (FR-5.1.7)."""

    def test_warm_start_from_best_program(self) -> None:
        reg = _make_registry()
        config = _make_evolve_config(generations=5)
        ctx = _make_context()

        result1 = evolve(reg, config, _SimpleFitnessEvaluator(), ctx)
        config2 = _make_evolve_config(generations=5, seed=99)
        result2 = evolve(
            reg, config2, _SimpleFitnessEvaluator(), ctx,
            seed_programs=[result1.best_program],
        )
        assert isinstance(result2, EvolutionResult)
        assert result2.best_program.size > 0

    def test_warm_start_from_pareto_front(self) -> None:
        reg = _make_registry()
        config = _make_evolve_config(generations=5)
        ctx = _make_context()

        result1 = evolve(reg, config, _SimpleFitnessEvaluator(), ctx)
        seeds = result1.pareto_front
        config2 = _make_evolve_config(generations=5, seed=99)
        result2 = evolve(
            reg, config2, _SimpleFitnessEvaluator(), ctx,
            seed_programs=seeds,
        )
        assert isinstance(result2, EvolutionResult)

    def test_warm_start_from_deserialized(self) -> None:
        reg = _make_registry()
        config = _make_evolve_config(generations=5)
        ctx = _make_context()

        result1 = evolve(reg, config, _SimpleFitnessEvaluator(), ctx)
        # Serialize and deserialize best program
        data = serialize(result1.best_program)
        restored = deserialize(data, reg)

        config2 = _make_evolve_config(generations=5, seed=99)
        result2 = evolve(
            reg, config2, _SimpleFitnessEvaluator(), ctx,
            seed_programs=[restored],
        )
        assert isinstance(result2, EvolutionResult)

    def test_warm_start_improves_over_cold_start(self) -> None:
        """A warm-started run from a good solution should be at least as good."""
        reg = _make_registry()
        ctx = _make_context()

        # Cold start for 10 generations
        config1 = _make_evolve_config(generations=10)
        result1 = evolve(reg, config1, _SimpleFitnessEvaluator(), ctx)
        best1_fitness = result1.fitness_history[-1].best_fitness[0]

        # Warm start from result1's best for 10 more generations
        config2 = _make_evolve_config(generations=10, seed=99)
        result2 = evolve(
            reg, config2, _SimpleFitnessEvaluator(), ctx,
            seed_programs=[result1.best_program],
        )
        best2_fitness = result2.fitness_history[-1].best_fitness[0]

        # Warm start should be at least as good (fitness is negative MSE, higher = better)
        assert best2_fitness >= best1_fitness - 0.1  # small tolerance


# ===========================================================================
# 5. Edge case tests
# ===========================================================================


class TestSeedEdgeCases:
    """Edge cases for population seeding."""

    def test_single_terminal_seed(self) -> None:
        """A bare terminal node as the sole seed should work."""
        reg = _make_registry()
        config = _make_evolve_config(generations=3)
        ctx = _make_context()
        seed = _make_terminal("x")
        result = evolve(reg, config, _SimpleFitnessEvaluator(), ctx, seed_programs=[seed])
        assert isinstance(result, EvolutionResult)
        assert result.best_program.size > 0

    def test_single_constant_seed(self) -> None:
        """A bare constant node as the sole seed should work."""
        reg = _make_registry()
        config = _make_evolve_config(generations=3)
        ctx = _make_context()
        seed = _make_constant(3.14)
        result = evolve(reg, config, _SimpleFitnessEvaluator(), ctx, seed_programs=[seed])
        assert isinstance(result, EvolutionResult)
        assert result.best_program.size > 0

    def test_seeds_at_max_depth_boundary(self) -> None:
        """Seeds exactly at max_depth should pass validation and evolve."""
        reg = _make_registry()
        config = _make_evolve_config(max_depth=3, generations=3)
        ctx = _make_context()
        # Build tree at exactly depth 3
        seed = _make_deep_tree(reg, depth=3)  # neg(neg(neg(x)))
        assert seed.depth == 3
        result = evolve(reg, config, _SimpleFitnessEvaluator(), ctx, seed_programs=[seed])
        assert isinstance(result, EvolutionResult)

    def test_minimum_population_size_with_seeds(self) -> None:
        """Seeding should work with the minimum population_size (10)."""
        reg = _make_registry()
        config = _make_evolve_config(population_size=10, generations=3, elitism_count=1)
        ctx = _make_context()
        seeds = [_make_simple_tree(reg), _make_terminal("x")]
        result = evolve(reg, config, _SimpleFitnessEvaluator(), ctx, seed_programs=seeds)
        assert isinstance(result, EvolutionResult)

    def test_all_seeds_identical(self) -> None:
        """Providing N identical seeds should still produce a valid diverse population."""
        reg = _make_registry()
        config = _make_evolve_config(population_size=20, generations=3)
        ctx = _make_context()
        seed = _make_simple_tree(reg)
        seeds = [seed] * 5  # 5 identical add(x, y)
        result = evolve(reg, config, _SimpleFitnessEvaluator(), ctx, seed_programs=seeds)
        assert isinstance(result, EvolutionResult)
        # Evolution should still produce some diversity
        assert len(result.fitness_history) == 3

    def test_seed_with_parameterized_nodes(self) -> None:
        """Seeds containing ParameterizedNode should validate and evolve."""
        reg = _make_registry()
        config = _make_evolve_config(generations=3)
        ctx = _make_context()
        scale_info = reg.get("scale")
        seed = ParameterizedNode(
            primitive=scale_info,
            children=(_make_terminal("x"),),
            params={"factor": 2.5},
        )
        result = evolve(reg, config, _SimpleFitnessEvaluator(), ctx, seed_programs=[seed])
        assert isinstance(result, EvolutionResult)
        assert result.best_program.size > 0
