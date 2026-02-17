"""Tests for selection operators (FR-5.3)."""

from __future__ import annotations

import math

import numpy as np

from liq.gp.config import FitnessConfig, GPConfig
from liq.gp.evolution.selection import (
    compute_nsga2_rankings,
    crowding_distance,
    get_elites,
    non_dominated_sort,
    nsga2_select,
    select,
    tournament_select,
)
from liq.gp.program.ast import Program, TerminalNode
from liq.gp.types import FitnessResult, Series

# --- helpers ---------------------------------------------------------------


def _make_programs(n: int) -> list[Program]:
    """Create *n* distinct dummy programs."""
    return [TerminalNode(name=f"p{i}", output_type=Series) for i in range(n)]


def _make_fitness(objectives: tuple[float, ...]) -> FitnessResult:
    return FitnessResult(objectives=objectives)


def _tournament_config(**overrides) -> GPConfig:
    """Return a tournament-mode config with sensible defaults for tests."""
    defaults = {
        "population_size": 10,
        "tournament_size": 3,
        "elitism_count": 2,
        "selection_mode": "tournament",
        "fitness": FitnessConfig(
            objectives=["fitness"],
            objective_directions=["maximize"],
        ),
    }
    defaults.update(overrides)
    return GPConfig(**defaults)


def _nsga2_config(**overrides) -> GPConfig:
    """Return an NSGA-II config with 2 objectives."""
    defaults = {
        "population_size": 10,
        "tournament_size": 3,
        "elitism_count": 2,
        "selection_mode": "nsga2",
        "fitness": FitnessConfig(
            objectives=["accuracy", "complexity"],
            objective_directions=["maximize", "minimize"],
        ),
    }
    defaults.update(overrides)
    return GPConfig(**defaults)


# ===========================================================================
# Non-dominated sorting
# ===========================================================================


class TestNonDominatedSort:
    """non_dominated_sort partitions indices into Pareto fronts."""

    def test_single_front_no_domination(self) -> None:
        """All individuals on the Pareto front when no one dominates another."""
        fitnesses = [
            _make_fitness((1.0, 3.0)),
            _make_fitness((3.0, 1.0)),
            _make_fitness((2.0, 2.0)),
        ]
        fronts = non_dominated_sort(fitnesses, ["maximize", "maximize"])
        assert len(fronts) == 1
        assert set(fronts[0]) == {0, 1, 2}

    def test_two_fronts_clear_domination(self) -> None:
        """A dominated individual is placed in front 1."""
        fitnesses = [
            _make_fitness((3.0, 3.0)),  # dominates idx=2
            _make_fitness((1.0, 4.0)),  # non-dominated
            _make_fitness((1.0, 1.0)),  # dominated by 0 and 1
        ]
        fronts = non_dominated_sort(fitnesses, ["maximize", "maximize"])
        assert len(fronts) == 2
        assert set(fronts[0]) == {0, 1}
        assert fronts[1] == [2]

    def test_minimize_direction(self) -> None:
        """Correctly handles 'minimize' directions."""
        fitnesses = [
            _make_fitness((1.0, 1.0)),  # best on both (minimize)
            _make_fitness((2.0, 2.0)),  # dominated
            _make_fitness((1.0, 3.0)),  # non-dominated (ties on obj0)
        ]
        fronts = non_dominated_sort(fitnesses, ["minimize", "minimize"])
        assert len(fronts) == 2
        # idx=0 dominates idx=1 (1<2 on both); idx=2 doesn't dominate idx=1 (3>2 on obj1)
        # idx=0 and idx=2: 0 has (1,1) vs (1,3) => 0 dominates 2 on obj1
        # Actually: 0 dominates 2? obj0: 1<=1 (tie), obj1: 1<3 => yes, 0 dominates 2.
        # idx=2 vs idx=1: obj0: 1<2 (better), obj1: 3>2 (worse) => neither dominates.
        # So front 0 = {0}, front 1 = {1, 2}? No:
        # 0 dominates 1 (1<2, 1<2) and 0 dominates 2 (1<=1, 1<3).
        # 2 vs 1: obj0: 1<2 (better), obj1: 3>2 (worse) => neither dominates.
        # So front 0 = {0}, front 1 = {1, 2}.
        assert fronts[0] == [0]
        assert set(fronts[1]) == {1, 2}

    def test_empty_population(self) -> None:
        """Empty input yields empty output."""
        assert non_dominated_sort([], ["maximize"]) == []

    def test_single_individual(self) -> None:
        """Single individual is its own front."""
        fitnesses = [_make_fitness((5.0,))]
        fronts = non_dominated_sort(fitnesses, ["maximize"])
        assert fronts == [[0]]

    def test_three_fronts(self) -> None:
        """Three clearly layered fronts."""
        fitnesses = [
            _make_fitness((5.0, 5.0)),  # front 0
            _make_fitness((3.0, 3.0)),  # front 1 (dominated by 0)
            _make_fitness((1.0, 1.0)),  # front 2 (dominated by 0 and 1)
        ]
        fronts = non_dominated_sort(fitnesses, ["maximize", "maximize"])
        assert len(fronts) == 3
        assert fronts[0] == [0]
        assert fronts[1] == [1]
        assert fronts[2] == [2]

    def test_mixed_directions(self) -> None:
        """Maximize first objective, minimize second."""
        fitnesses = [
            _make_fitness((3.0, 1.0)),  # high first, low second = ideal
            _make_fitness((1.0, 3.0)),  # low first, high second = bad on both
            _make_fitness((2.0, 2.0)),  # middle
        ]
        fronts = non_dominated_sort(fitnesses, ["maximize", "minimize"])
        # idx=0 vs idx=1: obj0: 3>1(better), obj1: 1<3(better) => 0 dominates 1
        # idx=0 vs idx=2: obj0: 3>2(better), obj1: 1<2(better) => 0 dominates 2
        # idx=2 vs idx=1: obj0: 2>1(better), obj1: 2<3(better) => 2 dominates 1
        # Front 0: {0}, Front 1: {2}, Front 2: {1}
        assert len(fronts) == 3
        assert fronts[0] == [0]
        assert fronts[1] == [2]
        assert fronts[2] == [1]


# ===========================================================================
# Crowding distance
# ===========================================================================


class TestCrowdingDistance:
    """crowding_distance assigns diversity scores within a front."""

    def test_boundary_individuals_get_infinity(self) -> None:
        """The extreme individuals on each objective get infinite distance."""
        fitnesses = [
            _make_fitness((1.0, 5.0)),
            _make_fitness((3.0, 3.0)),
            _make_fitness((5.0, 1.0)),
        ]
        front = [0, 1, 2]
        dists = crowding_distance(fitnesses, front, ["maximize", "maximize"])
        # Individuals 0 and 2 are boundaries on both objectives
        assert dists[0] == math.inf
        assert dists[2] == math.inf
        # Middle individual gets a finite distance
        assert math.isfinite(dists[1])

    def test_two_individuals_get_infinity(self) -> None:
        """With only 2 individuals, both get infinite distance."""
        fitnesses = [
            _make_fitness((1.0, 5.0)),
            _make_fitness((5.0, 1.0)),
        ]
        front = [0, 1]
        dists = crowding_distance(fitnesses, front, ["maximize", "maximize"])
        assert dists[0] == math.inf
        assert dists[1] == math.inf

    def test_single_individual_gets_infinity(self) -> None:
        """Single individual in a front gets infinite distance."""
        fitnesses = [_make_fitness((3.0, 3.0))]
        front = [0]
        dists = crowding_distance(fitnesses, front, ["maximize", "maximize"])
        assert dists[0] == math.inf

    def test_middle_individual_distance_value(self) -> None:
        """Verify the crowding distance value for a middle individual."""
        # Three individuals evenly spread: distance should be 2.0
        # (1.0 per objective for the normalized gap)
        fitnesses = [
            _make_fitness((0.0, 4.0)),
            _make_fitness((2.0, 2.0)),
            _make_fitness((4.0, 0.0)),
        ]
        front = [0, 1, 2]
        dists = crowding_distance(fitnesses, front, ["maximize", "maximize"])
        # For obj0: sorted order is [0,1,2], range=4, middle gap = (4-0)/4 = 1.0
        # For obj1: sorted order is [2,1,0], range=4, middle gap = (4-0)/4 = 1.0
        # Total = 2.0
        assert math.isclose(dists[1], 2.0, abs_tol=1e-9)

    def test_uses_subset_of_population(self) -> None:
        """Front can be a subset of total population indices."""
        fitnesses = [
            _make_fitness((0.0, 0.0)),  # idx 0 - not in front
            _make_fitness((1.0, 5.0)),  # idx 1
            _make_fitness((3.0, 3.0)),  # idx 2
            _make_fitness((5.0, 1.0)),  # idx 3
        ]
        front = [1, 2, 3]
        dists = crowding_distance(fitnesses, front, ["maximize", "maximize"])
        assert len(dists) == 3
        assert dists[0] == math.inf  # idx 1 is boundary
        assert dists[2] == math.inf  # idx 3 is boundary

    def test_constant_objective_skipped(self) -> None:
        """When one objective has zero range, it contributes zero distance."""
        fitnesses = [
            _make_fitness((1.0, 5.0)),
            _make_fitness((2.0, 5.0)),  # same obj1 as others
            _make_fitness((3.0, 5.0)),
        ]
        front = [0, 1, 2]
        dists = crowding_distance(fitnesses, front, ["maximize", "maximize"])
        # Boundaries still get infinity
        assert dists[0] == math.inf
        assert dists[2] == math.inf
        # Middle individual: obj0 contributes 1.0 (full range), obj1 contributes 0
        assert math.isclose(dists[1], 1.0, abs_tol=1e-9)


# ===========================================================================
# Tournament selection
# ===========================================================================


class TestTournamentSelect:
    """tournament_select picks the best of k random individuals."""

    def test_returns_correct_count(self) -> None:
        """Returns population_size - elitism_count parents."""
        pop = _make_programs(10)
        fit = [_make_fitness((float(i),)) for i in range(10)]
        config = _tournament_config()
        rng = np.random.default_rng(42)
        selected = tournament_select(pop, fit, config, rng)
        assert len(selected) == config.population_size - config.elitism_count

    def test_favours_best_fitness(self) -> None:
        """Over many selections, the best individual appears most often."""
        pop = _make_programs(10)
        # Fitness is index value: higher = better (maximize)
        fit = [_make_fitness((float(i),)) for i in range(10)]
        config = _tournament_config(population_size=100, elitism_count=0)
        rng = np.random.default_rng(42)
        selected = tournament_select(pop, fit, config, rng)
        # Count how often the best individual (p9) is selected
        best = pop[9]
        count_best = sum(1 for p in selected if p == best)
        # With tournament_size=3 and maximize, best should appear frequently
        assert count_best > 10  # out of 100

    def test_minimize_direction(self) -> None:
        """Tournament respects minimize direction."""
        pop = _make_programs(10)
        fit = [_make_fitness((float(i),)) for i in range(10)]
        config = _tournament_config(
            population_size=100,
            elitism_count=0,
            fitness=FitnessConfig(
                objectives=["loss"],
                objective_directions=["minimize"],
            ),
        )
        rng = np.random.default_rng(42)
        selected = tournament_select(pop, fit, config, rng)
        # p0 (fitness 0.0) should be most common when minimizing
        worst_for_max = pop[0]
        count = sum(1 for p in selected if p == worst_for_max)
        assert count > 10

    def test_deterministic_with_seed(self) -> None:
        """Same seed produces identical selection."""
        pop = _make_programs(10)
        fit = [_make_fitness((float(i),)) for i in range(10)]
        config = _tournament_config()
        s1 = tournament_select(pop, fit, config, np.random.default_rng(99))
        s2 = tournament_select(pop, fit, config, np.random.default_rng(99))
        assert s1 == s2

    def test_different_seed_different_result(self) -> None:
        """Different seeds produce different selections (with high probability)."""
        pop = _make_programs(10)
        fit = [_make_fitness((float(i),)) for i in range(10)]
        config = _tournament_config()
        s1 = tournament_select(pop, fit, config, np.random.default_rng(1))
        s2 = tournament_select(pop, fit, config, np.random.default_rng(2))
        # Not guaranteed to differ, but overwhelmingly likely with 8 picks
        assert s1 != s2

    def test_lexicographic_tie_break_uses_secondary_objective(self) -> None:
        """At equal primary fitness, secondary objective breaks ties."""
        pop = _make_programs(10)
        fit = [_make_fitness((1.0, -5.0)) for _ in range(10)]
        fit[0] = _make_fitness((1.0, -1.0))
        config = _tournament_config(
            population_size=20,
            elitism_count=0,
            tournament_size=10,
        )
        selected = tournament_select(pop, fit, config, np.random.default_rng(42))
        assert all(p == pop[0] for p in selected)


# ===========================================================================
# NSGA-II selection
# ===========================================================================


class TestNSGA2Select:
    """nsga2_select uses non-dominated sorting + crowding distance."""

    def test_returns_correct_count(self) -> None:
        pop = _make_programs(10)
        fit = [_make_fitness((float(i), float(10 - i))) for i in range(10)]
        config = _nsga2_config()
        rng = np.random.default_rng(42)
        selected = nsga2_select(pop, fit, config, rng)
        assert len(selected) == config.population_size - config.elitism_count

    def test_prefers_front0_individuals(self) -> None:
        """Individuals on Pareto front 0 should be selected more often."""
        pop = _make_programs(10)
        # Directions: maximize obj0, minimize obj1.
        # Front 0: trade-off between high obj0 and low obj1.
        fit = [
            _make_fitness((10.0, 5.0)),  # best obj0, moderate obj1
            _make_fitness((2.0, 0.1)),  # low obj0, best obj1
            _make_fitness((6.0, 1.0)),  # middle trade-off
            # Dominated by at least one front-0 member (lower obj0 AND higher obj1)
            _make_fitness((1.0, 2.0)),  # dominated by p2 (6>1, 1<2)
            _make_fitness((0.8, 3.0)),
            _make_fitness((0.5, 4.0)),
            _make_fitness((0.3, 6.0)),
            _make_fitness((0.2, 7.0)),
            _make_fitness((0.1, 8.0)),
            _make_fitness((0.01, 9.0)),
        ]
        config = _nsga2_config(population_size=100, elitism_count=0)
        rng = np.random.default_rng(42)
        selected = nsga2_select(pop, fit, config, rng)
        # Count front-0 individuals in selection
        front0_names = {"p0", "p1", "p2"}
        count = sum(
            1
            for p in selected
            if isinstance(p, TerminalNode) and p.name in front0_names
        )
        # Front-0 should dominate selections
        assert count > 50

    def test_deterministic_with_seed(self) -> None:
        pop = _make_programs(10)
        fit = [_make_fitness((float(i), float(10 - i))) for i in range(10)]
        config = _nsga2_config()
        s1 = nsga2_select(pop, fit, config, np.random.default_rng(42))
        s2 = nsga2_select(pop, fit, config, np.random.default_rng(42))
        assert s1 == s2

    def test_select_with_provided_ranking(self) -> None:
        pop = _make_programs(12)
        fit = [_make_fitness((float(i), float(20 - i))) for i in range(12)]
        config = _nsga2_config(population_size=12, elitism_count=4)
        fronts, ranks, crowding = compute_nsga2_rankings(fit, config)
        expected = nsga2_select(pop, fit, config, np.random.default_rng(42))
        observed = nsga2_select(
            pop,
            fit,
            config,
            np.random.default_rng(42),
            fronts=fronts,
            ranks=ranks,
            crowding=crowding,
        )
        assert observed == expected


# ===========================================================================
# Elitism
# ===========================================================================


class TestGetElites:
    """get_elites preserves top individuals."""

    def test_tournament_elites_top_n(self) -> None:
        """Single-objective elitism returns the top N individuals."""
        pop = _make_programs(10)
        fit = [_make_fitness((float(i),)) for i in range(10)]
        config = _tournament_config(elitism_count=3)
        elites = get_elites(pop, fit, config)
        assert len(elites) == 3
        # Best are p9, p8, p7 (maximize)
        elite_names = {p.name for p in elites if isinstance(p, TerminalNode)}
        assert elite_names == {"p9", "p8", "p7"}

    def test_tournament_elites_minimize(self) -> None:
        """Minimize direction picks smallest values."""
        pop = _make_programs(10)
        fit = [_make_fitness((float(i),)) for i in range(10)]
        config = _tournament_config(
            elitism_count=2,
            fitness=FitnessConfig(
                objectives=["loss"],
                objective_directions=["minimize"],
            ),
        )
        elites = get_elites(pop, fit, config)
        elite_names = {p.name for p in elites if isinstance(p, TerminalNode)}
        assert elite_names == {"p0", "p1"}

    def test_tournament_elites_lexicographic_tie_break(self) -> None:
        """Elites use secondary objective when primary objective ties."""
        pop = _make_programs(10)
        fit = [_make_fitness((1.0, -10.0)) for _ in range(10)]
        fit[3] = _make_fitness((1.0, -1.0))
        fit[7] = _make_fitness((1.0, -2.0))
        config = _tournament_config(elitism_count=2)
        elites = get_elites(pop, fit, config)
        elite_names = {p.name for p in elites if isinstance(p, TerminalNode)}
        assert elite_names == {"p3", "p7"}

    def test_nsga2_elites_first_front(self) -> None:
        """Multi-objective elitism returns the first Pareto front."""
        pop = _make_programs(10)
        # Directions: maximize obj0, minimize obj1.
        # Front 0: trade-off between high obj0 and low obj1.
        fit = [
            _make_fitness((10.0, 5.0)),  # best obj0, moderate obj1 (front 0)
            _make_fitness((2.0, 0.1)),  # low obj0, best obj1 (front 0)
            _make_fitness((6.0, 1.0)),  # middle trade-off (front 0)
            # Dominated by at least one front-0 member
            _make_fitness((1.0, 2.0)),
            _make_fitness((0.8, 3.0)),
            _make_fitness((0.5, 4.0)),
            _make_fitness((0.3, 6.0)),
            _make_fitness((0.2, 7.0)),
            _make_fitness((0.1, 8.0)),
            _make_fitness((0.01, 9.0)),
        ]
        config = _nsga2_config(elitism_count=5)
        elites = get_elites(pop, fit, config)
        # First front has 3 members, plus 2 from front 1 by crowding distance
        assert len(elites) == 5
        elite_names = {p.name for p in elites if isinstance(p, TerminalNode)}
        # Front 0 members must all be present
        assert {"p0", "p1", "p2"}.issubset(elite_names)

    def test_nsga2_elites_respects_count(self) -> None:
        """When first front > elitism_count, truncates by crowding distance."""
        pop = _make_programs(10)
        # All on one Pareto front (trade-off curve) under [maximize, minimize]:
        # higher i => better obj0 but worse obj1. All are non-dominated.
        fit = [_make_fitness((float(i), float(i))) for i in range(10)]
        config = _nsga2_config(elitism_count=3)
        elites = get_elites(pop, fit, config)
        assert len(elites) == 3
        # Boundary individuals (extreme on each objective) should be preferred
        # due to infinite crowding distance.
        elite_names = {p.name for p in elites if isinstance(p, TerminalNode)}
        # p0 (0,0) = best on obj1 (minimize), p9 (9,9) = best on obj0 (maximize)
        assert "p0" in elite_names
        assert "p9" in elite_names

    def test_nsga2_elites_consider_extended_objective_directions(self) -> None:
        """When an extra objective is present, it still influences Pareto ranking."""
        pop = _make_programs(10)
        fit = [_make_fitness((0.0, 0.0, -5.0)) for _ in range(10)]
        fit[1] = _make_fitness((0.0, 0.0, -1.0))
        config = _nsga2_config(elitism_count=1)
        elites = get_elites(pop, fit, config)
        assert len(elites) == 1
        assert elites[0] == pop[1]

    def test_zero_elitism(self) -> None:
        """elitism_count=0 returns empty list."""
        pop = _make_programs(10)
        fit = [_make_fitness((float(i),)) for i in range(10)]
        config = _tournament_config(elitism_count=0)
        assert get_elites(pop, fit, config) == []

    def test_nsga2_elites_with_provided_ranking(self) -> None:
        pop = _make_programs(12)
        fit = [_make_fitness((float(i), float(20 - i))) for i in range(12)]
        config = _nsga2_config(population_size=12, elitism_count=4)
        fronts, ranks, crowding = compute_nsga2_rankings(fit, config)
        expected = get_elites(pop, fit, config)
        observed = get_elites(
            pop,
            fit,
            config,
            fronts=fronts,
            ranks=ranks,
            crowding=crowding,
        )
        assert observed == expected


# ===========================================================================
# Dispatcher
# ===========================================================================


class TestSelectDispatcher:
    """select() routes to the correct implementation based on config."""

    def test_dispatches_to_tournament(self) -> None:
        pop = _make_programs(10)
        fit = [_make_fitness((float(i),)) for i in range(10)]
        config = _tournament_config()
        rng = np.random.default_rng(42)
        result_dispatch = select(pop, fit, config, rng)
        result_direct = tournament_select(pop, fit, config, np.random.default_rng(42))
        assert result_dispatch == result_direct

    def test_dispatches_to_nsga2(self) -> None:
        pop = _make_programs(10)
        fit = [_make_fitness((float(i), float(10 - i))) for i in range(10)]
        config = _nsga2_config()
        rng = np.random.default_rng(42)
        result_dispatch = select(pop, fit, config, rng)
        result_direct = nsga2_select(pop, fit, config, np.random.default_rng(42))
        assert result_dispatch == result_direct


# ===========================================================================
# Determinism
# ===========================================================================


class TestDeterminism:
    """All selection functions are deterministic with the same seed."""

    def test_non_dominated_sort_is_deterministic(self) -> None:
        """Sorting is fully deterministic (no randomness involved)."""
        fitnesses = [_make_fitness((float(i), float(10 - i))) for i in range(20)]
        f1 = non_dominated_sort(fitnesses, ["maximize", "maximize"])
        f2 = non_dominated_sort(fitnesses, ["maximize", "maximize"])
        assert f1 == f2

    def test_crowding_distance_is_deterministic(self) -> None:
        fitnesses = [_make_fitness((float(i), float(10 - i))) for i in range(5)]
        front = list(range(5))
        d1 = crowding_distance(fitnesses, front, ["maximize", "maximize"])
        d2 = crowding_distance(fitnesses, front, ["maximize", "maximize"])
        assert d1 == d2

    def test_get_elites_is_deterministic(self) -> None:
        pop = _make_programs(10)
        fit = [_make_fitness((float(i),)) for i in range(10)]
        config = _tournament_config()
        e1 = get_elites(pop, fit, config)
        e2 = get_elites(pop, fit, config)
        assert e1 == e2
