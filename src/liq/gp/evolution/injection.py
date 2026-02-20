"""Periodic seed injection during evolution.

Provides :func:`inject_seeds` which replaces worst-fitness individuals
with new programs generated from seed programs or random initialization,
according to the :class:`~liq.gp.config.SeedInjectionConfig`.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from liq.gp.config import GPConfig
from liq.gp.errors import EvolutionError
from liq.gp.primitives.registry import PrimitiveRegistry
from liq.gp.program.ast import Program
from liq.gp.types import FitnessResult, GPType, Series


def inject_seeds(
    population: list[Program],
    fitnesses: list[FitnessResult],
    seeds: Sequence[Program] | None,
    config: GPConfig,
    registry: PrimitiveRegistry,
    rng: np.random.Generator,
    generation: int,
    *,
    ranks: list[int] | None = None,
    crowding: list[float] | None = None,
    elite_indices: set[int] | None = None,
    injection_event: int | None = None,
    output_type: GPType | None = None,
) -> tuple[list[Program], int]:
    """Inject seed programs into the population, replacing worst individuals.

    Does nothing and returns the population unchanged when:
    - ``config.seed_injection`` is ``None``
    - ``generation`` is 0 (first generation, population just initialized)
    - ``generation`` is not a multiple of the configured interval

    Args:
        population: Current population.
        fitnesses: Fitness results aligned with *population*.
        seeds: Seed programs (required for ``"direct"`` and ``"variation"``
            methods; may be ``None`` for ``"ramped"``).
        config: GP configuration including ``seed_injection``.
        registry: Primitive registry for tree generation.
        rng: Random number generator.
        generation: Current generation index (0-based).
        ranks: Optional NSGA-II Pareto-front ranks (one per individual).
        crowding: Optional NSGA-II crowding distances (one per individual).
        elite_indices: Indices of elite individuals to protect from replacement.
        injection_event: Injection event counter (how many times injection
            has actually fired).  Used for round-robin seed cycling in
            ``"direct"`` mode.
        output_type: Output type for generated programs.  When ``None``,
            defaults to ``Series``.

    Returns:
        A tuple of (modified_population, injected_count).  When no injection
        occurs, returns (original_population, 0).
    """
    inj = config.seed_injection
    if inj is None:
        return population, 0
    if generation == 0:
        return population, 0
    if generation % inj.interval != 0:
        return population, 0

    # Find indices of worst individuals to replace
    worst_indices = _find_worst_indices(
        fitnesses,
        inj.count,
        config,
        ranks=ranks,
        crowding=crowding,
        elite_indices=elite_indices,
    )

    # Generate replacement programs
    if inj.method == "direct":
        replacements = _generate_direct(
            seeds, inj.count, generation, injection_event=injection_event
        )
    elif inj.method == "variation":
        replacements = _generate_variation(
            seeds, inj.count, config, registry, rng, output_type=output_type
        )
    else:  # ramped
        replacements = _generate_ramped(
            inj.count, config, registry, rng, output_type=output_type
        )

    # Replace worst with new programs
    result = list(population)
    for idx, replacement in zip(worst_indices, replacements, strict=True):
        result[idx] = replacement

    return result, len(replacements)


def _find_worst_indices(
    fitnesses: list[FitnessResult],
    count: int,
    config: GPConfig,
    *,
    ranks: list[int] | None = None,
    crowding: list[float] | None = None,
    elite_indices: set[int] | None = None,
) -> list[int]:
    """Return indices of the *count* worst individuals.

    When *ranks* and *crowding* are provided (NSGA-II mode), worst is
    defined as highest rank first, then lowest crowding distance within
    the same rank.  Otherwise falls back to first-objective sorting.

    Indices in *elite_indices* are excluded from consideration.
    """
    if ranks is not None and crowding is not None:
        # NSGA-II: worst = highest rank, then lowest crowding distance
        indices = [
            i
            for i in range(len(fitnesses))
            if elite_indices is None or i not in elite_indices
        ]
        indices.sort(key=lambda i: (-ranks[i], crowding[i]))
        return indices[:count]

    # Single-objective fallback: sort by first objective
    direction = config.fitness.objective_directions[0]
    # For maximize, worst = lowest value; for minimize, worst = highest value
    reverse = direction == "minimize"
    indexed = [
        (i, f.objectives[0])
        for i, f in enumerate(fitnesses)
        if elite_indices is None or i not in elite_indices
    ]
    indexed.sort(key=lambda x: x[1], reverse=reverse)
    return [i for i, _ in indexed[:count]]


def _generate_direct(
    seeds: Sequence[Program] | None,
    count: int,
    generation: int,
    *,
    injection_event: int | None = None,
) -> list[Program]:
    """Pick *count* seeds by cycling round-robin, offset by injection event.

    When *injection_event* is provided, the offset is based on the event
    counter (number of times injection has actually fired) rather than the
    generation number.  This ensures all seeds are used even when
    ``interval > 1``.
    """
    if not seeds:
        raise EvolutionError(
            "seeds are required for 'direct' injection method but were None or empty"
        )
    result: list[Program] = []
    if injection_event is not None:
        offset = injection_event * count
    else:
        offset = (generation - 1) * count  # legacy fallback
    for i in range(count):
        result.append(seeds[(offset + i) % len(seeds)])
    return result


def _generate_variation(
    seeds: Sequence[Program] | None,
    count: int,
    config: GPConfig,
    registry: PrimitiveRegistry,
    rng: np.random.Generator,
    *,
    output_type: GPType | None = None,
) -> list[Program]:
    """Generate *count* offspring by applying variation operators to seeds."""
    from liq.gp.evolution.constraints import enforce_constraints
    from liq.gp.evolution.init import generate_grow
    from liq.gp.evolution.operators import (
        hoist_mutation,
        parameter_mutation,
        point_mutation,
        select_operator,
        subtree_crossover,
        subtree_mutation,
    )

    if not seeds:
        raise EvolutionError(
            "seeds are required for 'variation' injection method but were None or empty"
        )

    offspring: list[Program] = []
    pi = 0
    max_attempts = count * 10

    for _ in range(max_attempts):
        if len(offspring) >= count:
            break

        op = select_operator(config, rng)

        if op == "crossover":
            p1 = seeds[pi % len(seeds)]
            p2 = seeds[(pi + 1) % len(seeds)]
            pi += 2
            child1, child2 = subtree_crossover(
                p1,
                p2,
                registry,
                config.max_depth,
                rng,
                max_attempts=config.max_crossover_attempts,
            )
            for child in (child1, child2):
                if enforce_constraints(child, config) and len(offspring) < count:
                    offspring.append(child)
        elif op == "subtree_mutation":
            parent = seeds[pi % len(seeds)]
            pi += 1
            child = subtree_mutation(parent, registry, config.max_depth, rng)
            if enforce_constraints(child, config):
                offspring.append(child)
        elif op == "point_mutation":
            parent = seeds[pi % len(seeds)]
            pi += 1
            child = point_mutation(parent, registry, rng)
            if enforce_constraints(child, config):
                offspring.append(child)
        elif op == "parameter_mutation":
            parent = seeds[pi % len(seeds)]
            pi += 1
            child = parameter_mutation(parent, rng)
            if enforce_constraints(child, config):
                offspring.append(child)
        elif op == "hoist_mutation":
            parent = seeds[pi % len(seeds)]
            pi += 1
            child = hoist_mutation(parent, rng)
            if enforce_constraints(child, config):
                offspring.append(child)
        else:
            parent = seeds[pi % len(seeds)]
            pi += 1
            offspring.append(parent)

    # Fallback: generate random trees if operators couldn't produce enough
    if len(offspring) < count:
        fallback_type = (
            output_type
            if output_type is not None
            else (seeds[0].output_type if seeds else Series)
        )
        for _ in range(count * 10):
            if len(offspring) >= count:
                break
            depth = int(rng.integers(1, config.max_depth + 1))
            tree = generate_grow(registry, depth, fallback_type, rng)
            if enforce_constraints(tree, config):
                offspring.append(tree)

    if len(offspring) < count:
        raise EvolutionError(
            f"Could not generate {count} offspring via variation: "
            f"only {len(offspring)} passed constraints after "
            f"{max_attempts} operator attempts and {count * 10} fallback attempts. "
            f"Check max_depth={config.max_depth} / max_size={config.max_size} constraints."
        )

    return offspring[:count]


def _generate_ramped(
    count: int,
    config: GPConfig,
    registry: PrimitiveRegistry,
    rng: np.random.Generator,
    *,
    output_type: GPType | None = None,
) -> list[Program]:
    """Generate *count* random programs using ramped half-and-half."""
    from liq.gp.evolution.init import generate_full, generate_grow

    if output_type is None:
        output_type = Series
    programs: list[Program] = []
    min_depth = min(1, config.max_depth)
    depth_range = list(range(min_depth, config.max_depth + 1))
    if not depth_range:
        depth_range = [config.max_depth]

    for i in range(count):
        depth = depth_range[i % len(depth_range)]
        if i % 2 == 0:
            programs.append(generate_full(registry, depth, output_type, rng))
        else:
            programs.append(generate_grow(registry, depth, output_type, rng))

    return programs
