"""Selection operators for GP evolution (FR-5.3).

Provides tournament, NSGA-II, and (ε-)lexicase selection, plus elitism and
the shared selection dispatcher.
"""

from __future__ import annotations

import math
import statistics
import numbers
from collections import defaultdict
from functools import cmp_to_key
from typing import TYPE_CHECKING, Literal, cast

ObjectiveDirection = Literal["maximize", "minimize"]


if TYPE_CHECKING:
    from numpy.random import Generator

    from liq.gp.config import GPConfig
    from liq.gp.program.ast import Program
    from liq.gp.types import FitnessResult


def _effective_directions(
    fitnesses: list[FitnessResult],
    base_directions: list[ObjectiveDirection],
) -> list[ObjectiveDirection]:
    """Return objective directions aligned with objective tuple length."""
    if not fitnesses:
        return list(base_directions)
    n_objectives = len(fitnesses[0].objectives)
    directions: list[ObjectiveDirection] = list(base_directions)
    if n_objectives > len(directions):
        pad = ["maximize"] * (n_objectives - len(directions))
        directions.extend(cast(list[ObjectiveDirection], pad))
    return directions


def _compare_objectives_lexicographic(
    a: tuple[float, ...],
    b: tuple[float, ...],
    directions: list[ObjectiveDirection],
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


def _safe_float(value: object, *, penalty: float) -> float:
    """Convert metadata values to finite float scores."""
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        return penalty
    value_float = float(value)
    if not math.isfinite(value_float):
        return penalty
    return value_float


def _extract_slice_scores(
    fitness: FitnessResult,
    nan_penalty: float,
) -> dict[str, float]:
    """Extract per-slice scores from metadata.

    Non-dict metadata or invalid per-slice values are converted to
    ``nan_penalty`` so lexicase treats them as very poor outcomes.
    """
    raw = fitness.metadata.get("slice_scores")
    if not isinstance(raw, dict):
        return {}

    scores: dict[str, float] = {}
    for case_id, value in raw.items():
        if not isinstance(case_id, str):
            continue
        scores[case_id] = _safe_float(value, penalty=nan_penalty)
    return scores


def _extract_raw_objectives(fitness: FitnessResult) -> tuple[float, ...] | None:
    """Extract raw objectives metadata when present."""
    raw = fitness.metadata.get("raw_objectives")
    if raw is None:
        return None
    if not isinstance(raw, tuple):
        msg = (
            "metadata['raw_objectives'] must be a tuple when provided "
            f"(got {type(raw).__name__})"
        )
        raise ValueError(msg)
    return raw


def _extract_slice_weights(
    fitness: FitnessResult,
) -> dict[str, float]:
    """Extract optional per-slice weights from metadata."""
    raw = fitness.metadata.get("slice_weights")
    if not isinstance(raw, dict):
        return {}

    weights: dict[str, float] = {}
    for case_id, value in raw.items():
        if not isinstance(case_id, str):
            continue
        if not isinstance(value, numbers.Real) or isinstance(value, bool):
            continue
        value_float = float(value)
        if not math.isfinite(value_float):
            continue
        weights[case_id] = max(0.0, value_float)
    return weights


def _collect_case_weights(
    case_ids: list[str],
    fitnesses: list[FitnessResult],
) -> dict[str, float]:
    """Aggregate per-slice weights across the population."""
    total_weights = defaultdict(float)
    counts = defaultdict(int)

    case_set = set(case_ids)
    for fitness in fitnesses:
        weight_map = _extract_slice_weights(fitness)
        for case_id, value in weight_map.items():
            if case_id not in case_set:
                continue
            total_weights[case_id] += value
            counts[case_id] += 1

    weights: dict[str, float] = {}
    for case_id in case_ids:
        if counts[case_id] > 0:
            weights[case_id] = total_weights[case_id] / counts[case_id]
    return weights


def _align_slice_scores(
    fitnesses: list[FitnessResult],
    nan_penalty: float,
) -> tuple[list[str], list[list[float]]]:
    """Align all per-individual slice scores to the union key set."""
    score_maps: list[dict[str, float]] = []
    case_ids: set[str] = set()

    for fitness in fitnesses:
        score_map = _extract_slice_scores(fitness, nan_penalty=nan_penalty)
        score_maps.append(score_map)
        case_ids.update(score_map.keys())

    if not case_ids:
        return [], []

    ordered_case_ids = sorted(case_ids)
    case_index = {case_id: pos for pos, case_id in enumerate(ordered_case_ids)}
    aligned: list[list[float]] = []
    for score_map in score_maps:
        row = [nan_penalty] * len(ordered_case_ids)
        for case_id, score in score_map.items():
            row[case_index[case_id]] = score
        aligned.append(row)
    return ordered_case_ids, aligned


def _case_epsilon(values: list[float], strategy: Literal["mad", "percentile", "zero"], q: float) -> float:
    """Compute epsilon for one case column."""
    if not values:
        return 0.0
    if strategy == "zero":
        return 0.0
    if strategy == "mad":
        center = statistics.median(values)
        return statistics.median(abs(value - center) for value in values)

    q = min(max(q, 0.0), 100.0)
    if q == 0.0:
        return min(values)
    if q == 100.0:
        return max(values)

    sorted_values = sorted(values)
    rank = (len(sorted_values) - 1) * (q / 100.0)
    lower = int(math.floor(rank))
    upper = int(math.ceil(rank))
    if lower == upper:
        return sorted_values[lower]
    return sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * (
        rank - lower
    )


def _downsample_cases(
    cases: list[str],
    config: GPConfig,
    rng: Generator,
    case_weights: dict[str, float] | None = None,
) -> list[str]:
    """Return policy-specific subset of cases for one selection event."""
    if config.lexicase_downsample_policy == "none" or not cases:
        return cases

    shuffled = cases.copy()
    rng.shuffle(shuffled)
    if config.lexicase_downsample_cases is None:
        return shuffled

    target = min(
        len(shuffled),
        max(config.lexicase_downsample_min_cases, config.lexicase_downsample_cases),
    )
    if target <= 0:
        return []
    if target >= len(shuffled):
        return shuffled
    if config.lexicase_downsample_policy == "random":
        return shuffled[:target]
    if config.lexicase_downsample_policy == "informed":
        if not case_weights:
            return shuffled[:target]
        return sorted(
            shuffled,
            key=lambda case_id: (-case_weights.get(case_id, 0.0), case_id),
        )[:target]

    return shuffled[:target]


def _prepare_lexicase_case_scores(
    fitnesses: list[FitnessResult],
    config: GPConfig,
) -> tuple[list[str], list[float], list[list[float]]]:
    """Prepare case IDs, per-case epsilons, and per-individual aligned rows."""
    for fitness in fitnesses:
        raw_objectives = _extract_raw_objectives(fitness)
        if raw_objectives is None:
            continue

        raw_len = len(raw_objectives)
        score_map = _extract_slice_scores(fitness, nan_penalty=config.lexicase_nan_penalty)
        if score_map and len(score_map) != raw_len:
            msg = (
                "metadata['slice_scores'] and metadata['raw_objectives'] "
                "dimensions are incompatible"
            )
            raise ValueError(msg)

    case_ids, aligned_rows = _align_slice_scores(
        fitnesses=fitnesses,
        nan_penalty=config.lexicase_nan_penalty,
    )
    if not case_ids:
        msg = (
            "selection_mode='lexicase' requires metadata['slice_scores'] with at "
            "least one case key on at least one individual"
        )
        raise ValueError(msg)

    strategy = (
        "zero"
        if config.selection_mode == "lexicase"
        else config.lexicase_epsilon_strategy
    )
    epsilons = [
        _case_epsilon(
            [row[index] for row in aligned_rows],
            strategy=strategy,
            q=config.lexicase_epsilon_percentile,
        )
        for index in range(len(case_ids))
    ]
    return case_ids, epsilons, aligned_rows


def _select_with_lexicase(
    aligned_rows: list[list[float]],
    case_ids: list[str],
    epsilons: list[float],
    config: GPConfig,
    case_weights: dict[str, float],
    rng: Generator,
) -> int:
    """Return one winner index by lexicase using prepared case rows."""
    candidate_indices = list(range(len(aligned_rows)))
    if not candidate_indices:
        raise ValueError("Cannot perform lexicase selection on empty population")

    cases = case_ids.copy()
    rng.shuffle(cases)
    cases = _downsample_cases(
        cases=cases,
        config=config,
        rng=rng,
        case_weights=case_weights,
    )
    case_position = {case_id: index for index, case_id in enumerate(case_ids)}
    for case in cases:
        column = case_position[case]
        selected_column = [aligned_rows[index][column] for index in candidate_indices]
        best = min(selected_column)
        threshold = best + epsilons[column]
        candidate_indices = [
            index
            for index in candidate_indices
            if aligned_rows[index][column] <= threshold
        ]
        if len(candidate_indices) == 1:
            return candidate_indices[0]

    return int(rng.choice(candidate_indices))


def lexicase_select(
    population: list[Program],
    fitnesses: list[FitnessResult],
    config: GPConfig,
    rng: Generator,
) -> list[Program]:
    """Select parents with lexicase or ε-lexicase."""
    n_select = config.population_size - config.elitism_count
    case_ids, epsilons, aligned_rows = _prepare_lexicase_case_scores(
        fitnesses=fitnesses,
        config=config,
    )
    case_weights = _collect_case_weights(case_ids=case_ids, fitnesses=fitnesses)

    selected: list[Program] = []
    for _ in range(n_select):
        winner = _select_with_lexicase(
            aligned_rows=aligned_rows,
            case_ids=case_ids,
            epsilons=epsilons,
            config=config,
            case_weights=case_weights,
            rng=rng,
        )
        selected.append(population[winner])

    return selected


# ---------------------------------------------------------------------------
# Non-dominated sorting (NSGA-II front decomposition)
# ---------------------------------------------------------------------------


def non_dominated_sort(
    fitnesses: list[FitnessResult],
    directions: list[ObjectiveDirection],
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
    directions: list[ObjectiveDirection],
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
# NSGA-II shared ranking
# ---------------------------------------------------------------------------


def compute_nsga2_rankings(
    fitnesses: list[FitnessResult],
    config: GPConfig,
) -> tuple[list[list[int]], list[int], list[float]]:
    """Compute NSGA-II fronts, ranks, and crowding distances once."""
    directions = _effective_directions(
        fitnesses, list(config.fitness.objective_directions)
    )
    fronts = non_dominated_sort(fitnesses, directions)

    ranks = [0] * len(fitnesses)
    crowding = [0.0] * len(fitnesses)
    for front_rank, front in enumerate(fronts):
        distances = crowding_distance(fitnesses, front, directions)
        for pos, idx in enumerate(front):
            ranks[idx] = front_rank
            crowding[idx] = distances[pos]

    return fronts, ranks, crowding


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
    fronts: list[list[int]] | None = None,
    ranks: list[int] | None = None,
    crowding: list[float] | None = None,
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
    if fronts is None or ranks is None or crowding is None:
        fronts, ranks, crowding = compute_nsga2_rankings(fitnesses, config)
    elif len(ranks) != len(population) or len(crowding) != len(population):
        msg = "ranks and crowding must have length equal to population size"
        raise ValueError(msg)
    elif len(fronts) == 0:
        ranks = [0] * len(population)
        crowding = [0.0] * len(population)

    n_select = config.population_size - config.elitism_count
    pop_size = len(population)
    selected: list[Program] = []

    for _ in range(n_select):
        # Binary tournament using NSGA-II crowded comparison
        i, j = rng.choice(pop_size, size=2, replace=False)
        winner = _crowded_compare(i, j, ranks, crowding)
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
    fronts: list[list[int]] | None = None,
    ranks: list[int] | None = None,
    crowding: list[float] | None = None,
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

    if config.selection_mode in {"tournament", "lexicase", "lexicase_eps"}:
        return _tournament_elites(population, fitnesses, config)
    return _nsga2_elites(
        population,
        fitnesses,
        config,
        fronts=fronts,
        ranks=ranks,
        crowding=crowding,
    )


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
    fronts: list[list[int]] | None = None,
    ranks: list[int] | None = None,
    crowding: list[float] | None = None,
) -> list[Program]:
    """First Pareto front elites for multi-objective mode."""
    if fronts is None or ranks is None or crowding is None:
        fronts, ranks, crowding = compute_nsga2_rankings(fitnesses, config)
    elif len(ranks) != len(population) or len(crowding) != len(population):
        msg = "ranks and crowding must have length equal to population size"
        raise ValueError(msg)
    elif len(fronts) == 0:
        ranks = [0] * len(population)
        crowding = [0.0] * len(population)

    elites: list[Program] = []
    for front in fronts:
        if len(elites) >= config.elitism_count:
            break
        remaining = config.elitism_count - len(elites)
        if len(front) <= remaining:
            elites.extend(population[i] for i in front)
        else:
            # Sort by crowding distance (descending) and take the top
            paired = sorted(
                zip(front, (crowding[idx] for idx in front), strict=True),
                key=lambda x: x[1],
                reverse=True,
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
    fronts: list[list[int]] | None = None,
    ranks: list[int] | None = None,
    crowding: list[float] | None = None,
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
    if config.selection_mode == "nsga2":
        return nsga2_select(
            population,
            fitnesses,
            config,
            rng,
            fronts=fronts,
            ranks=ranks,
            crowding=crowding,
        )
    if config.selection_mode in {"lexicase", "lexicase_eps"}:
        return lexicase_select(
            population=population,
            fitnesses=fitnesses,
            config=config,
            rng=rng,
        )
    else:
        msg = f"Unknown selection mode: {config.selection_mode!r}"
        raise ValueError(msg)
