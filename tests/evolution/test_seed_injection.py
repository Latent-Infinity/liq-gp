"""Tests for periodic seed injection during evolution."""

from __future__ import annotations

import numpy as np
import pytest

from liq.gp.config import FitnessConfig, GPConfig, SeedInjectionConfig
from liq.gp.errors import EvolutionError
from liq.gp.primitives.registry import PrimitiveRegistry
from liq.gp.program.ast import (
    ConstantNode,
    FunctionNode,
    Program,
    TerminalNode,
)
from liq.gp.program.eval import evaluate
from liq.gp.types import FitnessResult, ParamSpec, Series

# --- helpers ---------------------------------------------------------------


def _make_registry() -> PrimitiveRegistry:
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
    ps = ParamSpec(
        name="factor", dtype=float, default=1.0, min_value=-10.0, max_value=10.0
    )
    reg.register(
        "scale",
        lambda a, *, factor=1.0: a * factor,
        category="parameterized",
        input_types=(Series,),
        output_type=Series,
        param_specs=[ps],
    )
    return reg


def _make_terminal(name: str = "x") -> TerminalNode:
    return TerminalNode(name=name, output_type=Series)


def _make_constant(value: float = 1.0) -> ConstantNode:
    return ConstantNode(value=value)


def _make_simple_tree(reg: PrimitiveRegistry) -> FunctionNode:
    add_info = reg.get("add")
    return FunctionNode(
        primitive=add_info,
        children=(_make_terminal("x"), _make_terminal("y")),
    )


def _make_deep_tree(reg: PrimitiveRegistry, depth: int) -> Program:
    neg_info = reg.get("neg")
    node: Program = _make_terminal("x")
    for _ in range(depth):
        node = FunctionNode(primitive=neg_info, children=(node,))
    return node


def _make_config(**overrides: object) -> GPConfig:
    defaults: dict[str, object] = {
        "population_size": 20,
        "max_depth": 4,
        "generations": 1,
        "seed": 42,
        "elitism_count": 2,
    }
    defaults.update(overrides)
    return GPConfig(**defaults)  # type: ignore[arg-type]


def _make_population(reg: PrimitiveRegistry, size: int) -> list[Program]:
    """Build a population of varied programs."""
    programs: list[Program] = []
    for i in range(size):
        if i % 3 == 0:
            programs.append(_make_simple_tree(reg))
        elif i % 3 == 1:
            programs.append(_make_terminal("x"))
        else:
            programs.append(_make_constant(float(i)))
    return programs


def _make_fitnesses(size: int, *, direction: str = "maximize") -> list[FitnessResult]:
    """Build fitnesses with distinct values so worst is identifiable."""
    return [FitnessResult(objectives=(float(i),)) for i in range(size)]


def _make_context(n: int = 50) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(0)
    return {
        "x": rng.uniform(-1.0, 1.0, size=n),
        "y": rng.uniform(-1.0, 1.0, size=n),
    }


# ===========================================================================
# 1. Direct injection tests
# ===========================================================================


class TestInjectSeedsDirect:
    """Tests for inject_seeds with method='direct'."""

    def test_injects_correct_count(self) -> None:
        from liq.gp.evolution.injection import inject_seeds

        reg = _make_registry()
        config = _make_config(
            seed_injection=SeedInjectionConfig(interval=1, count=2, method="direct"),
        )
        pop = _make_population(reg, 20)
        fitnesses = _make_fitnesses(20)
        seeds = [_make_simple_tree(reg), _make_terminal("y")]
        rng = np.random.default_rng(42)

        result, count = inject_seeds(
            pop, fitnesses, seeds, config, reg, rng, generation=1
        )
        assert len(result) == 20
        assert count == 2

    def test_replaces_worst_fitness_individuals(self) -> None:
        from liq.gp.evolution.injection import inject_seeds

        reg = _make_registry()
        config = _make_config(
            seed_injection=SeedInjectionConfig(interval=1, count=2, method="direct"),
        )
        pop = _make_population(reg, 20)
        # Fitness 0..19 ascending — worst are indices 0 and 1 (lowest)
        fitnesses = _make_fitnesses(20)
        seed_a = _make_terminal("y")
        seed_b = _make_constant(99.0)
        seeds = [seed_a, seed_b]
        rng = np.random.default_rng(42)

        result, _ = inject_seeds(pop, fitnesses, seeds, config, reg, rng, generation=1)
        # The two worst (indices 0, 1) should be replaced
        assert result[0] is seed_a
        assert result[1] is seed_b

    def test_preserves_population_size(self) -> None:
        from liq.gp.evolution.injection import inject_seeds

        reg = _make_registry()
        config = _make_config(
            seed_injection=SeedInjectionConfig(interval=1, count=3, method="direct"),
        )
        pop = _make_population(reg, 20)
        fitnesses = _make_fitnesses(20)
        seeds = [_make_terminal("x"), _make_terminal("y"), _make_constant(1.0)]
        rng = np.random.default_rng(42)

        result, _ = inject_seeds(pop, fitnesses, seeds, config, reg, rng, generation=1)
        assert len(result) == len(pop)

    def test_cycles_through_seeds(self) -> None:
        """When count > len(seeds), cycles through seeds round-robin."""
        from liq.gp.evolution.injection import inject_seeds

        reg = _make_registry()
        config = _make_config(
            seed_injection=SeedInjectionConfig(interval=1, count=4, method="direct"),
        )
        pop = _make_population(reg, 20)
        fitnesses = _make_fitnesses(20)
        seed_a = _make_terminal("x")
        seed_b = _make_terminal("y")
        seeds = [seed_a, seed_b]
        rng = np.random.default_rng(42)

        result, _ = inject_seeds(pop, fitnesses, seeds, config, reg, rng, generation=1)
        # 4 injected, cycling through 2 seeds: a, b, a, b
        worst_indices = [0, 1, 2, 3]  # lowest fitness
        assert result[worst_indices[0]] is seed_a
        assert result[worst_indices[1]] is seed_b
        assert result[worst_indices[2]] is seed_a
        assert result[worst_indices[3]] is seed_b

    def test_no_injection_at_generation_zero(self) -> None:
        from liq.gp.evolution.injection import inject_seeds

        reg = _make_registry()
        config = _make_config(
            seed_injection=SeedInjectionConfig(interval=1, count=2, method="direct"),
        )
        pop = _make_population(reg, 20)
        original = list(pop)
        fitnesses = _make_fitnesses(20)
        seeds = [_make_terminal("y")]
        rng = np.random.default_rng(42)

        result, _ = inject_seeds(pop, fitnesses, seeds, config, reg, rng, generation=0)
        assert result == original

    def test_no_injection_off_interval(self) -> None:
        from liq.gp.evolution.injection import inject_seeds

        reg = _make_registry()
        config = _make_config(
            seed_injection=SeedInjectionConfig(interval=5, count=2, method="direct"),
        )
        pop = _make_population(reg, 20)
        original = list(pop)
        fitnesses = _make_fitnesses(20)
        seeds = [_make_terminal("y")]
        rng = np.random.default_rng(42)

        # Generation 3 is not a multiple of 5
        result, _ = inject_seeds(pop, fitnesses, seeds, config, reg, rng, generation=3)
        assert result == original

    def test_injection_at_interval(self) -> None:
        from liq.gp.evolution.injection import inject_seeds

        reg = _make_registry()
        config = _make_config(
            seed_injection=SeedInjectionConfig(interval=5, count=1, method="direct"),
        )
        pop = _make_population(reg, 20)
        fitnesses = _make_fitnesses(20)
        seeds = [_make_terminal("y")]
        rng = np.random.default_rng(42)

        # Generation 5 is a multiple of 5 and > 0
        result, _ = inject_seeds(pop, fitnesses, seeds, config, reg, rng, generation=5)
        assert result[0] is seeds[0]  # worst replaced

    def test_no_injection_when_config_is_none(self) -> None:
        from liq.gp.evolution.injection import inject_seeds

        reg = _make_registry()
        config = _make_config()  # seed_injection=None
        pop = _make_population(reg, 20)
        original = list(pop)
        fitnesses = _make_fitnesses(20)
        rng = np.random.default_rng(42)

        result, _ = inject_seeds(pop, fitnesses, None, config, reg, rng, generation=1)
        assert result == original


# ===========================================================================
# 2. Variation injection tests
# ===========================================================================


class TestInjectSeedsVariation:
    """Tests for inject_seeds with method='variation'."""

    def test_produces_correct_count(self) -> None:
        from liq.gp.evolution.injection import inject_seeds

        reg = _make_registry()
        config = _make_config(
            seed_injection=SeedInjectionConfig(interval=1, count=3, method="variation"),
        )
        pop = _make_population(reg, 20)
        fitnesses = _make_fitnesses(20)
        seeds = [_make_simple_tree(reg)]
        rng = np.random.default_rng(42)

        result, _ = inject_seeds(pop, fitnesses, seeds, config, reg, rng, generation=1)
        assert len(result) == 20

    def test_offspring_respect_max_depth(self) -> None:
        from liq.gp.evolution.injection import inject_seeds

        reg = _make_registry()
        config = _make_config(
            max_depth=3,
            seed_injection=SeedInjectionConfig(interval=1, count=5, method="variation"),
        )
        pop = _make_population(reg, 20)
        fitnesses = _make_fitnesses(20)
        seeds = [_make_simple_tree(reg)]
        rng = np.random.default_rng(42)

        result, _ = inject_seeds(pop, fitnesses, seeds, config, reg, rng, generation=1)
        for prog in result:
            assert prog.depth <= config.max_depth

    def test_offspring_respect_max_size(self) -> None:
        from liq.gp.evolution.injection import inject_seeds

        reg = _make_registry()
        config = _make_config(
            max_depth=4,
            max_size=10,
            seed_injection=SeedInjectionConfig(interval=1, count=5, method="variation"),
        )
        pop = _make_population(reg, 20)
        fitnesses = _make_fitnesses(20)
        seeds = [_make_simple_tree(reg)]
        rng = np.random.default_rng(42)

        result, _ = inject_seeds(pop, fitnesses, seeds, config, reg, rng, generation=1)
        for prog in result:
            assert prog.size <= 10

    def test_replaces_worst_individuals(self) -> None:
        from liq.gp.evolution.injection import inject_seeds

        reg = _make_registry()
        config = _make_config(
            seed_injection=SeedInjectionConfig(interval=1, count=2, method="variation"),
        )
        pop = _make_population(reg, 20)
        fitnesses = _make_fitnesses(20)
        seeds = [_make_simple_tree(reg)]
        rng = np.random.default_rng(42)

        original_worst = [pop[0], pop[1]]
        result, _ = inject_seeds(pop, fitnesses, seeds, config, reg, rng, generation=1)
        # The worst-fitness slots should have been replaced
        replaced = (
            result[0] is not original_worst[0] or result[1] is not original_worst[1]
        )
        assert replaced

    def test_deterministic_with_same_rng(self) -> None:
        from liq.gp.evolution.injection import inject_seeds

        reg = _make_registry()
        config = _make_config(
            seed_injection=SeedInjectionConfig(interval=1, count=3, method="variation"),
        )
        pop = _make_population(reg, 20)
        fitnesses = _make_fitnesses(20)
        seeds = [_make_simple_tree(reg)]

        rng1 = np.random.default_rng(42)
        result1, _ = inject_seeds(
            list(pop), fitnesses, seeds, config, reg, rng1, generation=1
        )

        rng2 = np.random.default_rng(42)
        result2, _ = inject_seeds(
            list(pop), fitnesses, seeds, config, reg, rng2, generation=1
        )

        for p1, p2 in zip(result1, result2, strict=True):
            assert p1 == p2


# ===========================================================================
# 3. Ramped injection tests
# ===========================================================================


class TestInjectSeedsRamped:
    """Tests for inject_seeds with method='ramped'."""

    def test_produces_correct_count(self) -> None:
        from liq.gp.evolution.injection import inject_seeds

        reg = _make_registry()
        config = _make_config(
            seed_injection=SeedInjectionConfig(interval=1, count=3, method="ramped"),
        )
        pop = _make_population(reg, 20)
        fitnesses = _make_fitnesses(20)
        rng = np.random.default_rng(42)

        result, _ = inject_seeds(pop, fitnesses, None, config, reg, rng, generation=1)
        assert len(result) == 20

    def test_no_seeds_required(self) -> None:
        """Ramped method works without seed programs."""
        from liq.gp.evolution.injection import inject_seeds

        reg = _make_registry()
        config = _make_config(
            seed_injection=SeedInjectionConfig(interval=1, count=2, method="ramped"),
        )
        pop = _make_population(reg, 20)
        fitnesses = _make_fitnesses(20)
        rng = np.random.default_rng(42)

        result, _ = inject_seeds(pop, fitnesses, None, config, reg, rng, generation=1)
        assert len(result) == 20

    def test_offspring_respect_max_depth(self) -> None:
        from liq.gp.evolution.injection import inject_seeds

        reg = _make_registry()
        config = _make_config(
            max_depth=3,
            seed_injection=SeedInjectionConfig(interval=1, count=5, method="ramped"),
        )
        pop = _make_population(reg, 20)
        fitnesses = _make_fitnesses(20)
        rng = np.random.default_rng(42)

        result, _ = inject_seeds(pop, fitnesses, None, config, reg, rng, generation=1)
        for prog in result:
            assert prog.depth <= config.max_depth

    def test_replaces_worst_individuals(self) -> None:
        from liq.gp.evolution.injection import inject_seeds

        reg = _make_registry()
        config = _make_config(
            seed_injection=SeedInjectionConfig(interval=1, count=2, method="ramped"),
        )
        pop = _make_population(reg, 20)
        fitnesses = _make_fitnesses(20)
        rng = np.random.default_rng(42)

        original_worst = [pop[0], pop[1]]
        result, _ = inject_seeds(pop, fitnesses, None, config, reg, rng, generation=1)
        # Worst slots should have been replaced
        replaced = (
            result[0] is not original_worst[0] or result[1] is not original_worst[1]
        )
        assert replaced

    def test_uses_mixed_full_and_grow(self) -> None:
        """Ramped injection should produce diverse tree structures."""
        from liq.gp.evolution.injection import inject_seeds

        reg = _make_registry()
        config = _make_config(
            seed_injection=SeedInjectionConfig(interval=1, count=10, method="ramped"),
        )
        pop = _make_population(reg, 20)
        fitnesses = _make_fitnesses(20)
        rng = np.random.default_rng(42)

        result, _ = inject_seeds(pop, fitnesses, None, config, reg, rng, generation=1)
        # Injected programs should have varied depths
        injected = result[:10]  # worst 10 were replaced
        depths = {p.depth for p in injected}
        assert len(depths) > 1, "Expected varied depths from ramped injection"


# ===========================================================================
# 4. Edge cases
# ===========================================================================


class TestInjectSeedsEdgeCases:
    """Edge cases for inject_seeds."""

    def test_count_equals_replaceable_slots(self) -> None:
        """Replacing all non-elite slots should work."""
        from liq.gp.evolution.injection import inject_seeds

        reg = _make_registry()
        # pop=20, elitism=2, so max replaceable is 18
        config = _make_config(
            population_size=20,
            elitism_count=2,
            seed_injection=SeedInjectionConfig(interval=1, count=18, method="ramped"),
        )
        pop = _make_population(reg, 20)
        fitnesses = _make_fitnesses(20)
        rng = np.random.default_rng(42)

        result, _ = inject_seeds(pop, fitnesses, None, config, reg, rng, generation=1)
        assert len(result) == 20

    def test_single_seed_count_greater_than_one(self) -> None:
        """Single seed with count > 1 should cycle it."""
        from liq.gp.evolution.injection import inject_seeds

        reg = _make_registry()
        config = _make_config(
            seed_injection=SeedInjectionConfig(interval=1, count=3, method="direct"),
        )
        pop = _make_population(reg, 20)
        fitnesses = _make_fitnesses(20)
        seed = _make_terminal("y")
        rng = np.random.default_rng(42)

        result, _ = inject_seeds(pop, fitnesses, [seed], config, reg, rng, generation=1)
        # All 3 worst slots get the same seed (cycling)
        assert result[0] is seed
        assert result[1] is seed
        assert result[2] is seed

    def test_seeds_at_constraint_boundary(self) -> None:
        """Seeds exactly at max_depth should work in variation mode."""
        from liq.gp.evolution.injection import inject_seeds

        reg = _make_registry()
        config = _make_config(
            max_depth=3,
            seed_injection=SeedInjectionConfig(interval=1, count=3, method="variation"),
        )
        seed = _make_deep_tree(reg, depth=3)
        assert seed.depth == 3
        pop = _make_population(reg, 20)
        fitnesses = _make_fitnesses(20)
        rng = np.random.default_rng(42)

        result, _ = inject_seeds(pop, fitnesses, [seed], config, reg, rng, generation=1)
        assert len(result) == 20
        for prog in result:
            assert prog.depth <= config.max_depth

    def test_generation_offset_cycles_seeds_across_injections(self) -> None:
        """Different generations should cycle through seeds differently."""
        from liq.gp.evolution.injection import inject_seeds

        reg = _make_registry()
        config = _make_config(
            seed_injection=SeedInjectionConfig(interval=1, count=1, method="direct"),
        )
        pop = _make_population(reg, 20)
        fitnesses = _make_fitnesses(20)
        seed_a = _make_terminal("x")
        seed_b = _make_terminal("y")
        seeds = [seed_a, seed_b]

        rng1 = np.random.default_rng(42)
        result1, _ = inject_seeds(
            list(pop), fitnesses, seeds, config, reg, rng1, generation=1
        )

        rng2 = np.random.default_rng(42)
        result2, _ = inject_seeds(
            list(pop), fitnesses, seeds, config, reg, rng2, generation=2
        )

        # Generation 1 and 2 should inject different seeds from the list
        assert result1[0] is not result2[0]

    def test_minimize_direction_replaces_highest(self) -> None:
        """With minimize direction, highest-value individuals are worst."""
        from liq.gp.evolution.injection import inject_seeds

        reg = _make_registry()
        config = _make_config(
            seed_injection=SeedInjectionConfig(interval=1, count=2, method="direct"),
            fitness=FitnessConfig(
                objectives=["loss"],
                objective_directions=["minimize"],
            ),
        )
        pop = _make_population(reg, 20)
        # For minimize, fitness 19, 18 are worst (highest values)
        fitnesses = _make_fitnesses(20)
        seed = _make_terminal("y")
        rng = np.random.default_rng(42)

        result, _ = inject_seeds(pop, fitnesses, [seed], config, reg, rng, generation=1)
        # Indices 19 and 18 (highest fitness = worst for minimize) should be replaced
        assert result[19] is seed
        assert result[18] is seed


# ===========================================================================
# 4b. Count enforcement tests
# ===========================================================================


class TestGenerateVariationCountEnforcement:
    """_generate_variation must produce exactly count offspring or raise."""

    def test_variation_raises_when_count_unreachable(self) -> None:
        """When constraints reject everything, raise EvolutionError."""
        from unittest.mock import patch

        from liq.gp.evolution.injection import _generate_variation

        reg = _make_registry()
        config = _make_config(max_depth=2, max_size=3)
        seeds = [_make_simple_tree(reg)]
        rng = np.random.default_rng(42)

        with (
            patch(
                "liq.gp.evolution.constraints.enforce_constraints",
                return_value=False,
            ),
            pytest.raises(EvolutionError, match="Could not generate"),
        ):
            _generate_variation(seeds, 5, config, reg, rng)

    def test_variation_succeeds_with_fallback(self) -> None:
        """Normal call produces exactly count offspring."""
        from liq.gp.evolution.injection import _generate_variation

        reg = _make_registry()
        config = _make_config(max_depth=4)
        seeds = [_make_simple_tree(reg)]
        rng = np.random.default_rng(42)

        result = _generate_variation(seeds, 3, config, reg, rng)
        assert len(result) == 3


# ===========================================================================
# 4c. NSGA-II worst-individual selection tests
# ===========================================================================


class TestFindWorstIndicesNSGA2:
    """_find_worst_indices uses NSGA-II ranking when ranks/crowding provided."""

    def test_worst_by_rank_and_crowding(self) -> None:
        """With NSGA-II data, worst = highest rank, then lowest crowding."""
        from liq.gp.evolution.injection import _find_worst_indices

        # 6 individuals:
        #   idx 0,1: rank 0 (best front), crowding inf
        #   idx 2,3: rank 1, crowding 3.0 and 1.0
        #   idx 4,5: rank 2 (worst front), crowding 2.0 and 0.5
        # Worst 2: idx 5 (rank=2, cd=0.5), then idx 4 (rank=2, cd=2.0)
        config = _make_config()
        fitnesses = [FitnessResult(objectives=(float(i),)) for i in range(6)]
        ranks = [0, 0, 1, 1, 2, 2]
        crowding = [float("inf"), float("inf"), 3.0, 1.0, 2.0, 0.5]

        worst = _find_worst_indices(
            fitnesses, 2, config, ranks=ranks, crowding=crowding
        )
        assert len(worst) == 2
        assert worst[0] == 5  # rank 2, cd 0.5 (worst)
        assert worst[1] == 4  # rank 2, cd 2.0

    def test_worst_four_spans_two_fronts(self) -> None:
        """Selecting 4 worst should span the two worst fronts."""
        from liq.gp.evolution.injection import _find_worst_indices

        config = _make_config()
        fitnesses = [FitnessResult(objectives=(float(i),)) for i in range(6)]
        ranks = [0, 0, 1, 1, 2, 2]
        crowding = [float("inf"), float("inf"), 3.0, 1.0, 2.0, 0.5]

        worst = _find_worst_indices(
            fitnesses, 4, config, ranks=ranks, crowding=crowding
        )
        assert len(worst) == 4
        # All from rank 2 first (sorted by crowding asc), then rank 1
        assert worst[0] == 5  # rank 2, cd 0.5
        assert worst[1] == 4  # rank 2, cd 2.0
        assert worst[2] == 3  # rank 1, cd 1.0
        assert worst[3] == 2  # rank 1, cd 3.0

    def test_without_ranks_falls_back_to_first_objective(self) -> None:
        """Without ranks/crowding, uses first-objective sorting."""
        from liq.gp.evolution.injection import _find_worst_indices

        config = _make_config()
        fitnesses = _make_fitnesses(10)
        worst = _find_worst_indices(fitnesses, 3, config)
        # maximize direction: lowest values are worst = indices 0, 1, 2
        assert worst == [0, 1, 2]

    def test_inject_seeds_passes_ranks_through(self) -> None:
        """inject_seeds threads ranks/crowding to _find_worst_indices."""
        from liq.gp.evolution.injection import inject_seeds

        reg = _make_registry()
        config = _make_config(
            seed_injection=SeedInjectionConfig(interval=1, count=1, method="direct"),
        )
        pop = _make_population(reg, 20)
        fitnesses = [FitnessResult(objectives=(float(i),)) for i in range(20)]
        # Each individual in its own front; rank 19 is worst
        ranks = list(range(20))
        crowding_vals = [1.0] * 20
        seed = _make_terminal("y")
        rng = np.random.default_rng(42)

        result, count = inject_seeds(
            pop,
            fitnesses,
            [seed],
            config,
            reg,
            rng,
            generation=1,
            ranks=ranks,
            crowding=crowding_vals,
        )
        assert count == 1
        assert result[19] is seed  # rank 19 is worst


# ===========================================================================
# 5. Evolution engine integration tests
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
                results.append(FitnessResult(objectives=(-mse, -float(prog.size))))
            except Exception:
                results.append(FitnessResult(objectives=(-1e10, -100.0)))
        return results


def _make_evolve_config(**overrides: object) -> GPConfig:
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


class TestEvolveWithSeedInjection:
    """Integration tests for evolve() with periodic seed injection."""

    def test_evolve_with_direct_injection(self) -> None:
        from liq.gp.evolution.engine import evolve

        reg = _make_registry()
        config = _make_evolve_config(
            seed_injection=SeedInjectionConfig(interval=2, count=2, method="direct"),
        )
        ctx = _make_context()
        seed = _make_simple_tree(reg)
        result = evolve(
            reg,
            config,
            _SimpleFitnessEvaluator(),
            ctx,
            seed_programs=[seed],
        )
        assert len(result.fitness_history) == 5
        assert result.best_program.size > 0

    def test_evolve_with_variation_injection(self) -> None:
        from liq.gp.evolution.engine import evolve

        reg = _make_registry()
        config = _make_evolve_config(
            seed_injection=SeedInjectionConfig(interval=1, count=3, method="variation"),
        )
        ctx = _make_context()
        seed = _make_simple_tree(reg)
        result = evolve(
            reg,
            config,
            _SimpleFitnessEvaluator(),
            ctx,
            seed_programs=[seed],
        )
        assert len(result.fitness_history) == 5

    def test_evolve_with_ramped_injection_no_seeds(self) -> None:
        """Ramped injection should work without seed_programs."""
        from liq.gp.evolution.engine import evolve

        reg = _make_registry()
        config = _make_evolve_config(
            seed_injection=SeedInjectionConfig(interval=2, count=2, method="ramped"),
        )
        ctx = _make_context()
        result = evolve(reg, config, _SimpleFitnessEvaluator(), ctx)
        assert len(result.fitness_history) == 5

    def test_population_size_preserved(self) -> None:
        """Population size should remain constant with injection enabled."""
        from liq.gp.evolution.engine import evolve

        reg = _make_registry()
        config = _make_evolve_config(
            seed_injection=SeedInjectionConfig(interval=1, count=5, method="ramped"),
        )
        ctx = _make_context()
        # Callback to verify population size indirectly via stats
        stats_log: list[object] = []
        evolve(
            reg,
            config,
            _SimpleFitnessEvaluator(),
            ctx,
            callback=lambda s: stats_log.append(s),
        )
        assert len(stats_log) == 5

    def test_injection_deterministic(self) -> None:
        """Same config.seed → same result with injection."""
        from liq.gp.evolution.engine import evolve

        reg = _make_registry()
        config = _make_evolve_config(
            seed_injection=SeedInjectionConfig(interval=2, count=2, method="direct"),
        )
        ctx = _make_context()
        seed = _make_simple_tree(reg)

        r1 = evolve(reg, config, _SimpleFitnessEvaluator(), ctx, seed_programs=[seed])
        r2 = evolve(reg, config, _SimpleFitnessEvaluator(), ctx, seed_programs=[seed])
        assert r1.best_program == r2.best_program

    def test_injection_different_seed_differs(self) -> None:
        """Different config.seed → different result with injection."""
        from liq.gp.evolution.engine import evolve

        reg = _make_registry()
        ctx = _make_context()
        seed = _make_simple_tree(reg)

        c1 = _make_evolve_config(
            seed=1,
            seed_injection=SeedInjectionConfig(interval=1, count=2, method="direct"),
        )
        c2 = _make_evolve_config(
            seed=2,
            seed_injection=SeedInjectionConfig(interval=1, count=2, method="direct"),
        )
        r1 = evolve(reg, c1, _SimpleFitnessEvaluator(), ctx, seed_programs=[seed])
        r2 = evolve(reg, c2, _SimpleFitnessEvaluator(), ctx, seed_programs=[seed])
        sizes1 = [s.mean_program_size for s in r1.fitness_history]
        sizes2 = [s.mean_program_size for s in r2.fitness_history]
        assert sizes1 != sizes2

    def test_injected_count_in_stats(self) -> None:
        """GenerationStats should report injected_count."""
        from liq.gp.evolution.engine import evolve

        reg = _make_registry()
        config = _make_evolve_config(
            generations=6,
            seed_injection=SeedInjectionConfig(interval=2, count=3, method="ramped"),
        )
        ctx = _make_context()
        result = evolve(reg, config, _SimpleFitnessEvaluator(), ctx)
        # Generations 0,1,2,3,4,5 — injection at gen 2, 4 (interval=2, skip gen 0)
        for stats in result.fitness_history:
            if stats.generation > 0 and stats.generation % 2 == 0:
                assert stats.injected_count == 3
            else:
                assert stats.injected_count == 0

    def test_no_injection_stats_default_zero(self) -> None:
        """Without seed_injection, injected_count is always 0."""
        from liq.gp.evolution.engine import evolve

        reg = _make_registry()
        config = _make_evolve_config()
        ctx = _make_context()
        result = evolve(reg, config, _SimpleFitnessEvaluator(), ctx)
        for stats in result.fitness_history:
            assert stats.injected_count == 0


class TestEvolveInjectionErrors:
    """Error cases for evolve() with seed injection."""

    def test_direct_without_seeds_raises(self) -> None:
        from liq.gp.evolution.engine import evolve

        reg = _make_registry()
        config = _make_evolve_config(
            seed_injection=SeedInjectionConfig(interval=1, count=1, method="direct"),
        )
        ctx = _make_context()
        with pytest.raises(EvolutionError, match="requires seed_programs"):
            evolve(reg, config, _SimpleFitnessEvaluator(), ctx)

    def test_variation_without_seeds_raises(self) -> None:
        from liq.gp.evolution.engine import evolve

        reg = _make_registry()
        config = _make_evolve_config(
            seed_injection=SeedInjectionConfig(interval=1, count=1, method="variation"),
        )
        ctx = _make_context()
        with pytest.raises(EvolutionError, match="requires seed_programs"):
            evolve(reg, config, _SimpleFitnessEvaluator(), ctx)


class TestEvolveInjectionFeatureInteractions:
    """Test injection interacts correctly with other features."""

    def test_injection_with_nsga2(self) -> None:
        from liq.gp.evolution.engine import evolve

        reg = _make_registry()
        fitness_config = FitnessConfig(
            objectives=["accuracy", "simplicity"],
            objective_directions=["maximize", "maximize"],
        )
        config = _make_evolve_config(
            selection_mode="nsga2",
            parsimony_mode="pareto",
            fitness=fitness_config,
            seed_injection=SeedInjectionConfig(interval=2, count=2, method="ramped"),
        )
        ctx = _make_context()
        result = evolve(reg, config, _MultiObjectiveEvaluator(), ctx)
        assert len(result.pareto_front) >= 1

    def test_injection_nsga2_uses_pareto_ranking(self) -> None:
        """With NSGA-II, injection replaces based on rank/crowding, not first objective."""
        from liq.gp.evolution.engine import evolve

        reg = _make_registry()
        fitness_config = FitnessConfig(
            objectives=["accuracy", "simplicity"],
            objective_directions=["maximize", "maximize"],
        )
        config = _make_evolve_config(
            selection_mode="nsga2",
            parsimony_mode="pareto",
            fitness=fitness_config,
            seed_injection=SeedInjectionConfig(interval=1, count=2, method="ramped"),
        )
        ctx = _make_context()
        result = evolve(reg, config, _MultiObjectiveEvaluator(), ctx)
        # Injection events should have occurred (every gen except 0)
        injected = [s.injected_count for s in result.fitness_history]
        assert any(c > 0 for c in injected)

    def test_injection_with_semantic_dedup(self) -> None:
        from liq.gp.evolution.engine import evolve

        reg = _make_registry()
        config = _make_evolve_config(
            semantic_dedup_enabled=True,
            seed_injection=SeedInjectionConfig(interval=2, count=2, method="ramped"),
        )
        ctx = _make_context()
        result = evolve(reg, config, _SimpleFitnessEvaluator(), ctx)
        assert len(result.fitness_history) == 5

    def test_injection_with_early_stopping(self) -> None:
        from liq.gp.evolution.engine import evolve

        reg = _make_registry()
        config = _make_evolve_config(
            generations=50,
            early_stop_patience=3,
            early_stop_threshold=1e-6,
            seed_injection=SeedInjectionConfig(interval=5, count=2, method="ramped"),
        )
        ctx = _make_context()
        seed = _make_simple_tree(reg)
        result = evolve(
            reg,
            config,
            _SimpleFitnessEvaluator(),
            ctx,
            seed_programs=[seed],
        )
        assert len(result.fitness_history) <= 50

    def test_injection_with_constant_opt(self) -> None:
        from liq.gp.evolution.engine import evolve

        reg = _make_registry()
        config = _make_evolve_config(
            constant_opt_enabled=True,
            seed_injection=SeedInjectionConfig(interval=2, count=2, method="variation"),
        )
        ctx = _make_context()
        seed = _make_simple_tree(reg)
        result = evolve(
            reg,
            config,
            _SimpleFitnessEvaluator(),
            ctx,
            seed_programs=[seed],
        )
        assert len(result.fitness_history) == 5

    def test_injection_fires_on_early_stop_generation(self) -> None:
        """Injection should run even on the generation that triggers early stop."""
        from liq.gp.evolution.engine import evolve

        reg = _make_registry()
        config = _make_evolve_config(
            generations=50,
            early_stop_patience=1,
            early_stop_threshold=1e10,  # always stalled
            seed_injection=SeedInjectionConfig(interval=1, count=2, method="ramped"),
        )
        ctx = _make_context()
        result = evolve(reg, config, _SimpleFitnessEvaluator(), ctx)
        # Early stop should have triggered
        assert len(result.fitness_history) < 50
        # Injection should have occurred on at least one generation
        injected = [s.injected_count for s in result.fitness_history]
        assert any(c > 0 for c in injected), (
            "Injection was never triggered -- early stop suppressed it"
        )


# ===========================================================================
# 7. Issue 1: assert -> EvolutionError guards
# ===========================================================================


class TestSeedGuardsWithoutAssert:
    """Assert-based guards must be explicit EvolutionError for -O safety."""

    def test_generate_direct_raises_on_none_seeds(self) -> None:
        from liq.gp.evolution.injection import _generate_direct

        with pytest.raises(EvolutionError, match="seeds.*required"):
            _generate_direct(None, 1, 1)

    def test_generate_direct_raises_on_empty_seeds(self) -> None:
        from liq.gp.evolution.injection import _generate_direct

        with pytest.raises(EvolutionError, match="seeds.*required"):
            _generate_direct([], 1, 1)

    def test_generate_variation_raises_on_none_seeds(self) -> None:
        from liq.gp.evolution.injection import _generate_variation

        reg = _make_registry()
        config = _make_config()
        rng = np.random.default_rng(42)
        with pytest.raises(EvolutionError, match="seeds.*required"):
            _generate_variation(None, 1, config, reg, rng)

    def test_generate_variation_raises_on_empty_seeds(self) -> None:
        from liq.gp.evolution.injection import _generate_variation

        reg = _make_registry()
        config = _make_config()
        rng = np.random.default_rng(42)
        with pytest.raises(EvolutionError, match="seeds.*required"):
            _generate_variation([], 1, config, reg, rng)


# ===========================================================================
# 8. Issue 3: Elite protection during injection
# ===========================================================================


class TestEliteProtection:
    """Elites must never be replaced during seed injection."""

    def test_find_worst_excludes_elite_indices(self) -> None:
        """All-tied fitnesses: without protection any index could be chosen."""
        from liq.gp.evolution.injection import _find_worst_indices

        # All fitnesses identical — without protection, elites can be picked
        fitnesses = [FitnessResult(objectives=(1.0,)) for _ in range(10)]
        config = _make_config(population_size=10, elitism_count=2)
        elite_indices = {0, 1}
        result = _find_worst_indices(fitnesses, 3, config, elite_indices=elite_indices)
        assert len(result) == 3
        for idx in result:
            assert idx not in elite_indices, f"elite index {idx} was selected as worst"

    def test_find_worst_nsga2_excludes_elite_indices(self) -> None:
        """NSGA-II path: elites excluded even with identical ranks/crowding."""
        from liq.gp.evolution.injection import _find_worst_indices

        n = 10
        fitnesses = [FitnessResult(objectives=(1.0,)) for _ in range(n)]
        # All same rank and crowding — any index could be picked
        ranks = [1] * n
        crowding = [0.5] * n
        config = _make_config(population_size=n, elitism_count=2)
        elite_indices = {0, 1}
        result = _find_worst_indices(
            fitnesses,
            3,
            config,
            ranks=ranks,
            crowding=crowding,
            elite_indices=elite_indices,
        )
        assert len(result) == 3
        for idx in result:
            assert idx not in elite_indices, f"elite index {idx} was selected as worst"

    def test_inject_seeds_passes_elite_indices_through(self) -> None:
        """inject_seeds with all-tied fitnesses preserves elite programs."""
        from liq.gp.evolution.injection import inject_seeds

        reg = _make_registry()
        pop = _make_population(reg, 10)
        elite_0 = pop[0]
        elite_1 = pop[1]
        # All fitnesses identical
        fitnesses = [FitnessResult(objectives=(1.0,)) for _ in range(10)]
        seeds = [_make_terminal("y")]
        config = _make_config(
            population_size=10,
            elitism_count=2,
            seed_injection=SeedInjectionConfig(
                method="direct",
                count=3,
                interval=1,
            ),
        )
        rng = np.random.default_rng(42)
        result, count = inject_seeds(
            pop,
            fitnesses,
            seeds,
            config,
            reg,
            rng,
            1,
            elite_indices={0, 1},
        )
        assert count == 3
        assert result[0] is elite_0
        assert result[1] is elite_1

    def test_elite_indices_none_is_backward_compatible(self) -> None:
        """Passing elite_indices=None matches no-arg behavior."""
        from liq.gp.evolution.injection import _find_worst_indices

        fitnesses = [FitnessResult(objectives=(float(i),)) for i in range(10)]
        config = _make_config(population_size=10, elitism_count=2)
        result_none = _find_worst_indices(fitnesses, 3, config, elite_indices=None)
        result_default = _find_worst_indices(fitnesses, 3, config)
        assert result_none == result_default


# ===========================================================================
# 9. Issue 2: Event-based seed cycling
# ===========================================================================


class TestInjectionEventCycling:
    """Injection event counter ensures all seeds get used with intervals > 1."""

    def test_direct_cycles_all_seeds_with_interval(self) -> None:
        """interval=3, count=1, 3 seeds: events 0,1,2 pick seeds 0,1,2."""
        from liq.gp.evolution.injection import _generate_direct

        seeds = [_make_terminal("x"), _make_terminal("y"), _make_constant(1.0)]
        for event in range(3):
            result = _generate_direct(seeds, 1, generation=1, injection_event=event)
            assert len(result) == 1
            assert result[0] is seeds[event], f"event {event} should pick seed {event}"

    def test_generate_direct_uses_injection_event(self) -> None:
        """injection_event overrides generation-based offset."""
        from liq.gp.evolution.injection import _generate_direct

        seeds = [_make_terminal("x"), _make_terminal("y")]
        # generation=100 would give a big offset, but injection_event=1
        # should produce seed[1]
        result = _generate_direct(seeds, 1, generation=100, injection_event=1)
        assert result[0] is seeds[1]

    def test_generate_direct_falls_back_when_event_none(self) -> None:
        """injection_event=None preserves generation-based offset behavior."""
        from liq.gp.evolution.injection import _generate_direct

        seeds = [_make_terminal("x"), _make_terminal("y"), _make_constant(1.0)]
        # generation=2, count=1 → offset = (2-1)*1 = 1 → seeds[1]
        result = _generate_direct(seeds, 1, generation=2, injection_event=None)
        assert result[0] is seeds[1]
        # Same as no kwarg
        result_default = _generate_direct(seeds, 1, generation=2)
        assert result_default[0] is seeds[1]


# ===========================================================================
# 10. Issue 5: Output type threading
# ===========================================================================


class TestOutputTypeThreading:
    """output_type should be threadable, not hardcoded to Series."""

    def test_generate_ramped_uses_provided_output_type(self) -> None:
        from liq.gp.evolution.injection import _generate_ramped

        reg = _make_registry()
        config = _make_config(
            seed_injection=SeedInjectionConfig(method="ramped", count=2, interval=1)
        )
        rng = np.random.default_rng(42)
        result = _generate_ramped(2, config, reg, rng, output_type=Series)
        assert len(result) == 2
        for p in result:
            assert p.output_type is Series

    def test_generate_ramped_defaults_to_series_when_none(self) -> None:
        from liq.gp.evolution.injection import _generate_ramped

        reg = _make_registry()
        config = _make_config(
            seed_injection=SeedInjectionConfig(method="ramped", count=2, interval=1)
        )
        rng = np.random.default_rng(42)
        result = _generate_ramped(2, config, reg, rng, output_type=None)
        assert len(result) == 2
        for p in result:
            assert p.output_type is Series

    def test_inject_seeds_threads_output_type_to_ramped(self) -> None:
        from liq.gp.evolution.injection import inject_seeds

        reg = _make_registry()
        pop = _make_population(reg, 10)
        fitnesses = _make_fitnesses(10)
        config = _make_config(
            population_size=10,
            elitism_count=2,
            seed_injection=SeedInjectionConfig(method="ramped", count=2, interval=1),
        )
        rng = np.random.default_rng(42)
        result, count = inject_seeds(
            pop,
            fitnesses,
            None,
            config,
            reg,
            rng,
            1,
            output_type=Series,
        )
        assert count == 2

    def test_variation_fallback_uses_output_type_param(self) -> None:
        from liq.gp.evolution.injection import _generate_variation

        reg = _make_registry()
        seeds = [_make_terminal("x")]
        config = _make_config(
            max_depth=2,
            max_size=3,
            seed_injection=SeedInjectionConfig(
                method="variation",
                count=2,
                interval=1,
            ),
        )
        rng = np.random.default_rng(42)
        result = _generate_variation(seeds, 2, config, reg, rng, output_type=Series)
        assert len(result) == 2


# ===========================================================================
# 11. Issue 4: Unconditional re-evaluation before injection
# ===========================================================================


class TestConstOptFitnessConsistency:
    """Injection must use post-const-opt fitnesses, not stale pre-opt ones."""

    def test_injection_reevaluates_after_const_opt(self) -> None:
        """With const-opt + injection both enabled, evolution completes
        successfully and all generations run including injection events.

        This covers the code path where fitnesses are unconditionally
        re-evaluated between constant optimization and seed injection,
        ensuring injection replaces individuals based on post-optimization
        fitness rather than stale pre-optimization values.
        """
        from liq.gp.evolution.engine import evolve

        reg = _make_registry()
        context = _make_context(50)
        eval_fn = evaluate  # avoid shadowing in inner class

        class SimpleEvaluator:
            def evaluate(
                self,
                programs: list[Program],
                ctx: dict[str, np.ndarray],
            ) -> list[FitnessResult]:
                results: list[FitnessResult] = []
                for p in programs:
                    try:
                        raw = eval_fn(p, ctx)
                        if isinstance(raw, np.ndarray):
                            results.append(
                                FitnessResult(objectives=(float(np.mean(raw)),))
                            )
                        else:
                            results.append(FitnessResult(objectives=(float(raw),)))
                    except Exception:
                        results.append(FitnessResult(objectives=(-1e10,)))
                return results

        config = _make_config(
            population_size=10,
            generations=4,
            elitism_count=2,
            constant_opt_enabled=True,
            constant_opt_method="scipy",
            constant_opt_budget=5,
            seed_injection=SeedInjectionConfig(
                method="ramped",
                count=2,
                interval=1,
            ),
        )
        evo_result = evolve(
            config=config,
            registry=reg,
            evaluator=SimpleEvaluator(),
            context=context,
        )
        assert evo_result.best_program is not None
        assert len(evo_result.fitness_history) == 4
        # Injection fires on generations 1, 2, 3 (not gen 0)
        total_injected = sum(s.injected_count for s in evo_result.fitness_history)
        assert total_injected > 0, "No seeds injected despite injection being enabled"


# ===========================================================================
# 12. Issue 6: Stats timing documentation contract
# ===========================================================================


class TestStatsInjectionTimingContract:
    """Stats fitness metrics describe pre-injection population; injected_count
    is additive metadata grafted on afterwards."""

    def test_stats_fitness_reflects_pre_injection_population(self) -> None:
        """Stats have finite fitness/size values and non-negative injected_count."""
        from liq.gp.evolution.engine import evolve

        reg = _make_registry()
        context = _make_context(50)
        eval_fn = evaluate  # avoid shadowing

        class SimpleEval:
            def evaluate(
                self,
                programs: list[Program],
                ctx: dict[str, np.ndarray],
            ) -> list[FitnessResult]:
                results: list[FitnessResult] = []
                for p in programs:
                    try:
                        raw = eval_fn(p, ctx)
                        if isinstance(raw, np.ndarray):
                            results.append(
                                FitnessResult(objectives=(float(np.mean(raw)),))
                            )
                        else:
                            results.append(FitnessResult(objectives=(float(raw),)))
                    except Exception:
                        results.append(FitnessResult(objectives=(-1e10,)))
                return results

        config = _make_config(
            population_size=10,
            generations=3,
            elitism_count=2,
            seed_injection=SeedInjectionConfig(
                method="direct",
                count=2,
                interval=1,
            ),
        )
        seeds = [_make_terminal("x"), _make_terminal("y")]
        evo_result = evolve(
            config=config,
            registry=reg,
            evaluator=SimpleEval(),
            context=context,
            seed_programs=seeds,
        )
        for stats in evo_result.fitness_history:
            assert np.isfinite(stats.best_fitness[0])
            assert np.isfinite(stats.mean_fitness[0])
            assert stats.mean_program_size > 0
            assert stats.injected_count >= 0
