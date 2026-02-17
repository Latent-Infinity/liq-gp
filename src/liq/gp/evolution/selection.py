"""Selection operators for GP evolution (FR-5.3).

Provides: tournament selection, NSGA-II selection (non-dominated sorting +
crowding distance), elitism, and a unified dispatcher.
"""

from __future__ import annotations

import math
from functools import cmp_to_key
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from numpy.random import Generator

    from liq.gp.config import GPConfig
    from liq.gp.program.ast import Program
    from liq.gp.types import FitnessResult


def _effective_directions(
    fitnesses: list[FitnessResult],
    base_directions: list[str],
) -> list[str]:
    """Return objective directions aligned with objective tuple length."""
    if not fitnesses:
        return list(base_directions)
    n_objectives = len(fitnesses[0].objectives)
    directions = list(base_directions)
    if n_objectives > len(directions):
        directions.extend(["maximize"] * (n_objectives - len(directions)))
    return directions


def _compare_objectives_lexicographic(
    a: tuple[float, ...],
    b: tuple[float, ...],
    directions: list[str],
) -> int:
    """Lexicographic comparison using objective directions.

    Returns:
        1 if ``a`` is better, -1 if ``b`` is better, 0 if tied.
    """
    for i, direction in enumerate(directions):
        av = a[i]
        bv = b[i]
        if av == bv:
            continue
        if direction == "maximize":
            return 1 if av > bv else -1
        return 1 if av < bv else -1
    return 0


# ---------------------------------------------------------------------------
# Non-dominated sorting (NSGA-II front decomposition)
# ---------------------------------------------------------------------------


def non_dominated_sort(
    fitnesses: list[FitnessResult],
    directions: list[str],
) -> list[list[int]]:
    """Partition indices into successive Pareto fronts.

    Parameters
    ----------
    fitnesses:
        One ``FitnessResult`` per individual.
    directions:
        ``"maximize"`` or ``"minimize"`` for each objective.

    Returns
    -------
    list[list[int]]
        Each inner list contains the indices belonging to one front,
        ordered from the best (front 0) to the worst.
    """
    n = len(fitnesses)
    if n == 0:
        return []

    # Convert to "the higher the better" by negating minimised objectives.
    signs = [1.0 if d == "maximize" else -1.0 for d in directions]
    objs = [
        tuple(s * v for s, v in zip(signs, f.objectives, strict=True))
        for f in fitnesses
    ]

    # domination_count[i] = how many individuals dominate i
    domination_count = [0] * n
    # dominated_set[i] = set of individuals i dominates
    dominated_set: list[list[int]] = [[] for _ in range(n)]

    for i in range(n):
        for j in range(i + 1, n):
            i_dom_j = _dominates(objs[i], objs[j])
            j_dom_i = _dominates(objs[j], objs[i])
            if i_dom_j:
                dominated_set[i].append(j)
                domination_count[j] += 1
            elif j_dom_i:
                dominated_set[j].append(i)
                domination_count[i] += 1

    fronts: list[list[int]] = []
    current_front = [i for i in range(n) if domination_count[i] == 0]

    while current_front:
        fronts.append(current_front)
        next_front: list[int] = []
        for i in current_front:
            for j in dominated_set[i]:
                domination_count[j] -= 1
                if domination_count[j] == 0:
                    next_front.append(j)
        current_front = next_front

    return fronts


def _dominates(a: tuple[float, ...], b: tuple[float, ...]) -> bool:
    """Return True if *a* Pareto-dominates *b* (all >= and at least one >)."""
    dominated = False
    for ai, bi in zip(a, b, strict=True):
        if ai < bi:
            return False
        if ai > bi:
            dominated = True
    return dominated


# ---------------------------------------------------------------------------
# Crowding distance
# ---------------------------------------------------------------------------


def crowding_distance(
    fitnesses: list[FitnessResult],
    front: list[int],
    directions: list[str],
) -> list[float]:
    """Compute crowding distance for each individual in *front*.

    Parameters
    ----------
    fitnesses:
        Full population fitness list (indexed by population index).
    front:
        Indices of individuals in this front.
    directions:
        ``"maximize"`` or ``"minimize"`` for each objective.

    Returns
    -------
    list[float]
        Crowding distance for each position in *front* (same order).
    """
    k = len(front)
    if k <= 2:
        return [math.inf] * k

    n_obj = len(directions)
    distances = [0.0] * k

    # Map front position -> population index
    for m in range(n_obj):
        sign = 1.0 if directions[m] == "maximize" else -1.0
        # Sort front positions by objective m value (ascending effective value)
        sorted_positions = sorted(
            range(k),
            key=lambda p, _m=m, _s=sign: _s * fitnesses[front[p]].objectives[_m],
        )

        # Boundary individuals get infinite distance
        distances[sorted_positions[0]] = math.inf
        distances[sorted_positions[-1]] = math.inf

        # Range of this objective across the front
        f_min = sign * fitnesses[front[sorted_positions[0]]].objectives[m]
        f_max = sign * fitnesses[front[sorted_positions[-1]]].objectives[m]
        obj_range = f_max - f_min
        if obj_range == 0.0:
            continue

        for i in range(1, k - 1):
            prev_val = sign * fitnesses[front[sorted_positions[i - 1]]].objectives[m]
            next_val = sign * fitnesses[front[sorted_positions[i + 1]]].objectives[m]
            distances[sorted_positions[i]] += (next_val - prev_val) / obj_range

    return distances


# ---------------------------------------------------------------------------
# Tournament selection (single-objective)
# ---------------------------------------------------------------------------


def tournament_select(
    population: list[Program],
    fitnesses: list[FitnessResult],
    config: GPConfig,
    rng: Generator,
) -> list[Program]:
    """Select parents via tournament selection.

    For each slot (population_size - elitism_count), pick ``tournament_size``
    random individuals and keep the one with the best first-objective fitness
    (respecting the configured direction).

    Parameters
    ----------
    population:
        Current population of programs.
    fitnesses:
        Parallel list of fitness results.
    config:
        GP configuration (uses tournament_size, population_size, elitism_count,
        fitness.objective_directions).
    rng:
        Numpy random generator for reproducibility.

    Returns
    -------
    list[Program]
        Selected parents (length = population_size - elitism_count).
    """
    n_select = config.population_size - config.elitism_count
    directions = _effective_directions(
        fitnesses, list(config.fitness.objective_directions)
    )

    selected: list[Program] = []
    pop_size = len(population)

    for _ in range(n_select):
        candidates = rng.choice(pop_size, size=config.tournament_size, replace=False)
        best_idx = candidates[0]
        for idx in candidates[1:]:
            if (
                _compare_objectives_lexicographic(
                    fitnesses[idx].objectives,
                    fitnesses[best_idx].objectives,
                    directions,
                )
                > 0
            ):
                best_idx = idx
        selected.append(population[best_idx])

    return selected


# ---------------------------------------------------------------------------
# NSGA-II selection (multi-objective)
# ---------------------------------------------------------------------------


def nsga2_select(
    population: list[Program],
    fitnesses: list[FitnessResult],
    config: GPConfig,
    rng: Generator,
) -> list[Program]:
    """Select parents via NSGA-II selection.

    Individuals are ranked by non-dominated sorting, then by crowding distance
    within each front. Binary tournament selection is then applied using this
    composite ranking.

    Parameters
    ----------
    population:
        Current population of programs.
    fitnesses:
        Parallel list of fitness results.
    config:
        GP configuration.
    rng:
        Numpy random generator for reproducibility.

    Returns
    -------
    list[Program]
        Selected parents (length = population_size - elitism_count).
    """
    directions = _effective_directions(
        fitnesses, list(config.fitness.objective_directions)
    )
    fronts = non_dominated_sort(fitnesses, directions)

    # Assign rank and crowding distance to every individual
    rank = [0] * len(population)
    cd = [0.0] * len(population)

    for front_rank, front in enumerate(fronts):
        dists = crowding_distance(fitnesses, front, directions)
        for pos, idx in enumerate(front):
            rank[idx] = front_rank
            cd[idx] = dists[pos]

    n_select = config.population_size - config.elitism_count
    pop_size = len(population)
    selected: list[Program] = []

    for _ in range(n_select):
        # Binary tournament using NSGA-II crowded comparison
        i, j = rng.choice(pop_size, size=2, replace=False)
        winner = _crowded_compare(i, j, rank, cd)
        selected.append(population[winner])

    return selected


def _crowded_compare(i: int, j: int, rank: list[int], cd: list[float]) -> int:
    """Return the index of the better individual under NSGA-II ordering."""
    if rank[i] < rank[j]:
        return i
    if rank[j] < rank[i]:
        return j
    # Same rank: prefer higher crowding distance
    if cd[i] >= cd[j]:
        return i
    return j


# ---------------------------------------------------------------------------
# Elitism
# ---------------------------------------------------------------------------


def get_elites(
    population: list[Program],
    fitnesses: list[FitnessResult],
    config: GPConfig,
) -> list[Program]:
    """Return elite individuals to carry forward unchanged.

    For single-objective (tournament mode): top ``elitism_count`` by first
    objective.

    For multi-objective (NSGA-II mode): the entire first Pareto front (up to
    ``elitism_count``). If the first front is smaller than ``elitism_count``,
    fill from subsequent fronts sorted by crowding distance.

    Parameters
    ----------
    population:
        Current population.
    fitnesses:
        Parallel fitness results.
    config:
        GP configuration.

    Returns
    -------
    list[Program]
        Elite programs (length <= elitism_count, or <= first-front size
        for NSGA-II).
    """
    if config.elitism_count == 0:
        return []

    if config.selection_mode == "tournament":
        return _tournament_elites(population, fitnesses, config)
    else:
        return _nsga2_elites(population, fitnesses, config)


def _tournament_elites(
    population: list[Program],
    fitnesses: list[FitnessResult],
    config: GPConfig,
) -> list[Program]:
    """Top-N elites by first objective for single-objective mode."""
    directions = _effective_directions(
        fitnesses, list(config.fitness.objective_directions)
    )

    def _compare_indices(i: int, j: int) -> int:
        result = _compare_objectives_lexicographic(
            fitnesses[i].objectives,
            fitnesses[j].objectives,
            directions,
        )
        if result > 0:
            return -1
        if result < 0:
            return 1
        return 0

    indices = sorted(range(len(population)), key=cmp_to_key(_compare_indices))
    return [population[i] for i in indices[: config.elitism_count]]


def _nsga2_elites(
    population: list[Program],
    fitnesses: list[FitnessResult],
    config: GPConfig,
) -> list[Program]:
    """First Pareto front elites for multi-objective mode."""
    directions = _effective_directions(
        fitnesses, list(config.fitness.objective_directions)
    )
    fronts = non_dominated_sort(fitnesses, directions)

    elites: list[Program] = []
    for front in fronts:
        if len(elites) >= config.elitism_count:
            break
        remaining = config.elitism_count - len(elites)
        if len(front) <= remaining:
            elites.extend(population[i] for i in front)
        else:
            # Sort by crowding distance (descending) and take the top
            dists = crowding_distance(fitnesses, front, directions)
            paired = sorted(
                zip(front, dists, strict=True), key=lambda x: x[1], reverse=True
            )
            elites.extend(population[idx] for idx, _ in paired[:remaining])

    return elites


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


def select(
    population: list[Program],
    fitnesses: list[FitnessResult],
    config: GPConfig,
    rng: Generator,
) -> list[Program]:
    """Select parents using the configured selection mode.

    Parameters
    ----------
    population:
        Current population of programs.
    fitnesses:
        Parallel list of fitness results.
    config:
        GP configuration (``config.selection_mode`` determines the algorithm).
    rng:
        Numpy random generator for reproducibility.

    Returns
    -------
    list[Program]
        Selected parents.
    """
    if config.selection_mode == "tournament":
        return tournament_select(population, fitnesses, config, rng)
    elif config.selection_mode == "nsga2":
        return nsga2_select(population, fitnesses, config, rng)
    else:
        msg = f"Unknown selection mode: {config.selection_mode!r}"
        raise ValueError(msg)
