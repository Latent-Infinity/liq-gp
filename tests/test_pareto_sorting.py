"""Unit tests for non-dominated sorting, crowding distance, and elite indices."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from liq.gp.config import FitnessConfig, GPConfig
from liq.gp.evolution.selection import (
    crowding_distance,
    get_elite_indices,
    get_elites,
    non_dominated_sort,
)
from liq.gp.program.ast import ConstantNode
from liq.gp.types import FitnessResult


def _dominates(a, b, signs) -> bool:
    ea = [s * v for s, v in zip(signs, a, strict=True)]
    eb = [s * v for s, v in zip(signs, b, strict=True)]
    return all(x >= y for x, y in zip(ea, eb, strict=True)) and any(
        x > y for x, y in zip(ea, eb, strict=True)
    )


def _reference_fronts(objs, directions) -> list[list[int]]:
    """Brute-force front peeling used as an oracle."""
    signs = [1.0 if d == "maximize" else -1.0 for d in directions]
    remaining = list(range(len(objs)))
    fronts: list[list[int]] = []
    while remaining:
        front = [
            i
            for i in remaining
            if not any(_dominates(objs[j], objs[i], signs) for j in remaining if j != i)
        ]
        fronts.append(front)
        remaining = [i for i in remaining if i not in front]
    return fronts


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("n_obj", [1, 2, 3])
def test_matches_bruteforce_reference(seed: int, n_obj: int) -> None:
    rng = np.random.default_rng(seed)
    n = 40
    # Coarse grid so that ties and exact duplicates occur.
    objs = [tuple(float(v) for v in rng.integers(0, 4, size=n_obj)) for _ in range(n)]
    directions = [
        "maximize" if bit else "minimize" for bit in rng.integers(0, 2, size=n_obj)
    ]
    fitnesses = [FitnessResult(objectives=o) for o in objs]

    fronts = non_dominated_sort(fitnesses, directions)
    expected = _reference_fronts(objs, directions)

    assert [sorted(f) for f in fronts] == [sorted(f) for f in expected]
    assert sorted(itertools.chain.from_iterable(fronts)) == list(range(n))


def test_empty_population() -> None:
    assert non_dominated_sort([], ["maximize"]) == []


def test_minimize_direction_is_honoured() -> None:
    fitnesses = [FitnessResult(objectives=(v,)) for v in (3.0, 1.0, 2.0)]
    assert non_dominated_sort(fitnesses, ["minimize"]) == [[1], [2], [0]]
    assert non_dominated_sort(fitnesses, ["maximize"]) == [[0], [2], [1]]


def test_nan_objectives_do_not_crash_and_never_dominate() -> None:
    fitnesses = [
        FitnessResult(objectives=(float("nan"), 1.0)),
        FitnessResult(objectives=(1.0, 1.0)),
    ]
    fronts = non_dominated_sort(fitnesses, ["maximize", "maximize"])
    assert sorted(itertools.chain.from_iterable(fronts)) == [0, 1]


def test_crowding_distance_boundaries_are_infinite() -> None:
    fitnesses = [
        FitnessResult(objectives=(0.0, 3.0)),
        FitnessResult(objectives=(1.0, 2.0)),
        FitnessResult(objectives=(2.0, 1.0)),
        FitnessResult(objectives=(3.0, 0.0)),
    ]
    distances = crowding_distance(fitnesses, [0, 1, 2, 3], ["maximize", "maximize"])
    assert distances[0] == np.inf
    assert distances[3] == np.inf
    assert distances[1] == pytest.approx(distances[2])
    assert 0.0 < distances[1] < np.inf


class TestEliteIndices:
    def _population(self, n: int):
        return [ConstantNode(float(i)) for i in range(n)]

    def test_tournament_mode_returns_top_by_first_objective(self) -> None:
        config = GPConfig(
            population_size=10, elitism_count=3, constant_opt_enabled=False
        )
        pop = self._population(10)
        fitnesses = [FitnessResult(objectives=(float(i % 5),)) for i in range(10)]
        indices = get_elite_indices(pop, fitnesses, config)
        assert len(indices) == 3
        assert sorted(fitnesses[i].objectives[0] for i in indices) == [3.0, 4.0, 4.0]
        assert get_elites(pop, fitnesses, config) == [pop[i] for i in indices]

    def test_nsga2_mode_prefers_first_front(self) -> None:
        config = GPConfig(
            population_size=10,
            elitism_count=2,
            selection_mode="nsga2",
            constant_opt_enabled=False,
            fitness=FitnessConfig(
                objectives=["a", "b"], objective_directions=["maximize", "maximize"]
            ),
        )
        pop = self._population(4)
        fitnesses = [
            FitnessResult(objectives=(1.0, 1.0)),
            FitnessResult(objectives=(0.0, 3.0)),
            FitnessResult(objectives=(3.0, 0.0)),
            FitnessResult(objectives=(0.5, 0.5)),
        ]
        indices = get_elite_indices(pop, fitnesses, config)
        assert set(indices) <= {0, 1, 2}
        assert len(indices) == 2

    def test_zero_elitism(self) -> None:
        config = GPConfig(
            population_size=10, elitism_count=0, constant_opt_enabled=False
        )
        pop = self._population(10)
        fitnesses = [FitnessResult(objectives=(1.0,)) for _ in pop]
        assert get_elite_indices(pop, fitnesses, config) == []
