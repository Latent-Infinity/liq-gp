"""Semantic deduplication for GP populations (FR-8).

Provides fingerprinting based on program behaviour over a reference dataset,
and duplicate replacement to maintain population diversity.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np

from liq.gp.program.eval import evaluate

if TYPE_CHECKING:
    from liq.gp.config import GPConfig
    from liq.gp.primitives.registry import PrimitiveRegistry
    from liq.gp.program.ast import Program


def compute_fingerprint(
    program: Program,
    ref_context: dict[str, np.ndarray],
    precision: int,
) -> bytes:
    """Evaluate a program on the reference dataset and return a semantic fingerprint.

    The fingerprint is the byte representation of the rounded output array.
    NaN values are replaced with 0.0 before rounding so that programs producing
    NaN at the same positions are considered identical (FR-8.1).

    Args:
        program: The program AST to fingerprint.
        ref_context: The reference evaluation context (sampled subset).
        precision: Number of decimal places for rounding.

    Returns:
        A ``bytes`` object representing the semantic fingerprint.
    """
    output = evaluate(program, ref_context)
    # Replace NaN with 0.0 for deterministic fingerprinting
    output = np.where(np.isnan(output), 0.0, output)
    rounded = np.round(output, decimals=precision)
    return rounded.tobytes()


def sample_reference_context(
    context: dict[str, np.ndarray],
    ref_size: int,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """Deterministically sample a reference context from the full context.

    Uniformly samples ``ref_size`` row indices (without replacement) and
    extracts the corresponding rows from every array in the context (FR-8.2).

    If ``ref_size`` is greater than or equal to the number of rows in the
    context, the full context is returned unchanged.

    Args:
        context: Full evaluation context mapping names to arrays.
        ref_size: Number of reference data points to sample.
        rng: NumPy random generator for deterministic sampling.

    Returns:
        A new context dict with arrays of length ``min(ref_size, n_rows)``.
    """
    # Determine the number of rows from the first array
    n_rows = 0
    for arr in context.values():
        n_rows = len(arr)
        break

    if ref_size >= n_rows:
        # Return a copy of the full context
        return {key: arr.copy() for key, arr in context.items()}

    indices = rng.choice(n_rows, size=ref_size, replace=False)
    return {key: arr[indices] for key, arr in context.items()}


def deduplicate_population(
    population: list[Program],
    ref_context: dict[str, np.ndarray],
    registry: PrimitiveRegistry,
    config: GPConfig,
    rng: np.random.Generator,
    fingerprints: Sequence[bytes] | None = None,
) -> tuple[list[Program], float]:
    """Replace semantically duplicate programs with new random individuals.

    Computes a fingerprint for each program. The first occurrence of each
    unique fingerprint is kept; subsequent duplicates are replaced with
    freshly generated random programs (via ``generate_grow``) when dedup
    is enabled (FR-8.4).

    The ``unique_semantics_ratio`` is always computed as the number of
    unique fingerprints divided by the population size, regardless of
    whether dedup is enabled (FR-8.5).

    Args:
        population: Current list of programs.
        ref_context: Reference evaluation context for fingerprinting.
        registry: Primitive registry (used for generating replacements).
        config: GP configuration (dedup settings, max_depth).
        rng: NumPy random generator.
        fingerprints: Optional pre-computed semantic fingerprints in population order.

    Returns:
        A tuple of (new_population, unique_semantics_ratio).
    """
    from liq.gp.evolution.init import generate_grow

    precision = config.semantic_precision
    pop_size = len(population)
    if fingerprints is None:
        fingerprints = [
            compute_fingerprint(prog, ref_context, precision) for prog in population
        ]
    elif len(fingerprints) != pop_size:
        msg = (
            "fingerprints must be provided for every individual in "
            "population and in population order"
        )
        raise ValueError(msg)

    # Count unique fingerprints for the ratio metric
    unique_count = len(set(fingerprints))
    unique_ratio = unique_count / pop_size if pop_size > 0 else 1.0

    if not config.semantic_dedup_enabled:
        return list(population), unique_ratio

    # Replace duplicates: keep first occurrence, replace the rest
    seen: set[bytes] = set()
    new_population: list[Program] = []

    for prog, fp in zip(population, fingerprints, strict=True):
        if fp not in seen:
            seen.add(fp)
            new_population.append(prog)
        else:
            replacement = generate_grow(
                registry, config.max_depth, prog.output_type, rng
            )
            new_population.append(replacement)

    return new_population, unique_ratio
