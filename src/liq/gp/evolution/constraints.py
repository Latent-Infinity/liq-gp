"""Constraint enforcement and parsimony pressure for GP evolution (FR-5.6).

Functions
---------
- enforce_constraints: check whether a program satisfies depth/size limits
- apply_parsimony: modify fitnesses according to parsimony mode
- filter_population: remove constraint-violating individuals
"""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING

from liq.gp.types import FitnessResult

if TYPE_CHECKING:
    from liq.gp.config import GPConfig
    from liq.gp.program.ast import Program


def enforce_constraints(program: Program, config: GPConfig) -> bool:
    """Return True if *program* satisfies all configured constraints.

    Checks:
    - ``program.depth <= config.max_depth``
    - ``program.size <= config.max_size`` (only when ``max_size`` is not None)
    """
    if program.depth > config.max_depth:
        return False
    return not (config.max_size is not None and program.size > config.max_size)


def apply_parsimony(
    fitnesses: list[FitnessResult],
    population: list[Program],
    config: GPConfig,
) -> list[FitnessResult]:
    """Modify *fitnesses* according to ``config.parsimony_mode``.

    Modes
    -----
    ``"disabled"``
        Return fitnesses unchanged.
    ``"lexicographic"``
        At equal primary fitness, prefer smaller programs. This is achieved by
        appending a ``-program.size`` objective (to be maximised, i.e. smaller
        size => larger value).
    ``"pareto"``
        Append ``-program.size`` as a separate minimisation objective for
        NSGA-II multi-objective selection.
    ``"linear"``
        Subtract ``parsimony_coefficient * program.size`` from the first
        objective.
    ``"size_diversity"``
        Add a size-density objective rewarding under-represented tree sizes.
    """
    mode = config.parsimony_mode

    if mode == "disabled":
        return list(fitnesses)

    if mode == "size_diversity":
        size_counts = Counter(prog.size for prog in population)
        size_scores = {size: 1.0 / count for size, count in size_counts.items()}
        return [
            FitnessResult(
                objectives=fr.objectives + (size_scores[prog.size],),
                metadata={**fr.metadata, "raw_objectives": fr.objectives},
            )
            for fr, prog in zip(fitnesses, population, strict=True)
        ]

    if mode == "lexicographic":
        return [
            FitnessResult(
                objectives=fr.objectives + (-prog.size,),
                metadata={**fr.metadata, "raw_objectives": fr.objectives},
            )
            for fr, prog in zip(fitnesses, population, strict=True)
        ]

    if mode == "pareto":
        return [
            FitnessResult(
                objectives=fr.objectives + (-prog.size,),
                metadata={**fr.metadata, "raw_objectives": fr.objectives},
            )
            for fr, prog in zip(fitnesses, population, strict=True)
        ]

    if mode == "linear":
        coeff = config.parsimony_coefficient
        return [
            FitnessResult(
                objectives=(
                    (fr.objectives[0] - coeff * prog.size,) + fr.objectives[1:]
                ),
                metadata={
                    **fr.metadata,
                    "raw_objectives": fr.objectives,
                },
            )
            for fr, prog in zip(fitnesses, population, strict=True)
        ]

    # Should never reach here due to Literal type, but be defensive.
    msg = f"Unknown parsimony_mode: {mode!r}"
    raise ValueError(msg)


def filter_population(
    population: list[Program],
    fitnesses: list[FitnessResult],
    config: GPConfig,
) -> tuple[list[Program], list[FitnessResult]]:
    """Remove individuals that violate constraints.

    Returns a ``(programs, fitnesses)`` tuple containing only the individuals
    whose programs pass :func:`enforce_constraints`.
    """
    filtered_programs: list[Program] = []
    filtered_fitnesses: list[FitnessResult] = []

    for prog, fit in zip(population, fitnesses, strict=True):
        if enforce_constraints(prog, config):
            filtered_programs.append(prog)
            filtered_fitnesses.append(fit)

    return filtered_programs, filtered_fitnesses
