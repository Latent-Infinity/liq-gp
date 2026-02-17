"""Constant optimization for GP programs (FR-6).

Extracts ConstantNode values from a program tree, optimizes them via
scipy.optimize.minimize (Nelder-Mead), and writes optimized values back
into a new AST (immutability preserved).
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.optimize import minimize

from liq.gp.program.ast import (
    ConstantNode,
    FunctionNode,
    ParameterizedNode,
    Program,
    TerminalNode,
)

if TYPE_CHECKING:
    from liq.gp.config import GPConfig
    from liq.gp.types import FitnessResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_constants(program: Program) -> list[float]:
    """Collect all ConstantNode values in depth-first left-to-right order.

    Args:
        program: The AST to extract constants from.

    Returns:
        A list of float values, one per ConstantNode in traversal order.
    """
    return [c.value for c in program.constants]


def inject_constants(program: Program, values: list[float]) -> Program:
    """Create a new AST with updated constant values (FR-6, immutability).

    The constants are consumed in the same depth-first left-to-right order
    as ``extract_constants`` produces them.

    Args:
        program: The original AST (not mutated).
        values: New constant values in traversal order.

    Returns:
        A new Program tree with the updated constants.

    Raises:
        ValueError: If ``len(values)`` does not match the number of
            constants in the tree.
    """
    expected = len(program.constants)
    if len(values) != expected:
        msg = (
            f"Constant count mismatch: tree has {expected} constants "
            f"but got {len(values)} values"
        )
        raise ValueError(msg)

    index = [0]  # mutable counter for recursive consumption
    return _inject_recursive(program, values, index)


def optimize_constants(
    program: Program,
    evaluator: Any,
    context: dict[str, np.ndarray],
    config: GPConfig,
    rng: np.random.Generator,
) -> Program:
    """Optimize ConstantNode values via Nelder-Mead (FR-6).

    Uses ``scipy.optimize.minimize`` with ``method='Nelder-Mead'``.
    Bounded by ``config.constant_opt_max_iter`` and
    ``config.constant_opt_max_time_seconds`` (whichever comes first).

    If the program has no constants, it is returned unchanged (no-op).
    Any exception during optimization is caught, logged, and the original
    program is returned (ConstantOptError is non-fatal).

    Args:
        program: The AST whose constants are to be optimized.
        evaluator: Object with an ``evaluate(programs, context)`` method
            returning a list of FitnessResult.
        context: Evaluation context (mapping of names to arrays).
        config: GP configuration (provides iteration/time limits).
        rng: NumPy random generator for deterministic initial perturbation.

    Returns:
        A new Program with optimized constants, or the original if
        optimization fails or has no constants to optimize.
    """
    current_constants = extract_constants(program)
    if not current_constants:
        return program

    try:
        optimized = _run_optimization(
            program, evaluator, context, config, rng, current_constants
        )
        return optimized
    except Exception:
        logger.warning(
            "Constant optimization failed; keeping original constants",
            exc_info=True,
        )
        return program


def select_for_optimization(
    population: list[Program],
    fitnesses: list[FitnessResult],
    config: GPConfig,
) -> list[int]:
    """Return sorted indices of the top-K programs eligible for constant optimization.

    Programs with zero constants are excluded. The top-K fraction
    (``config.constant_opt_top_k``) of the remaining programs is
    selected, with at least 1 selected if any eligible programs exist.

    Selection is based on the first objective (primary fitness), with
    higher values considered better.

    Args:
        population: The full population of programs.
        fitnesses: Corresponding fitness results.
        config: GP configuration (provides ``constant_opt_top_k``).

    Returns:
        Sorted list of indices into ``population``.
    """
    # Filter to programs that have at least one constant
    eligible: list[tuple[int, float]] = []
    for i, (prog, fit) in enumerate(zip(population, fitnesses, strict=True)):
        if extract_constants(prog):
            eligible.append((i, fit.objectives[0]))

    if not eligible:
        return []

    maximize = config.fitness.objective_directions[0] == "maximize"
    eligible.sort(key=lambda x: x[1], reverse=maximize)

    # Select top-K fraction, at least 1
    k = max(1, int(len(eligible) * config.constant_opt_top_k))
    selected_indices = [idx for idx, _fit in eligible[:k]]

    # Return sorted ascending
    selected_indices.sort()
    return selected_indices


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _inject_recursive(
    node: Program,
    values: list[float],
    index: list[int],
) -> Program:
    """Recursively rebuild the tree, substituting constant values."""
    if isinstance(node, TerminalNode):
        return node

    if isinstance(node, ConstantNode):
        new_value = values[index[0]]
        index[0] += 1
        return ConstantNode(value=new_value, output_type=node.output_type)

    if isinstance(node, FunctionNode):
        new_children = tuple(_inject_recursive(c, values, index) for c in node.children)
        return FunctionNode(primitive=node.primitive, children=new_children)

    if isinstance(node, ParameterizedNode):
        new_children = tuple(_inject_recursive(c, values, index) for c in node.children)
        return ParameterizedNode(
            primitive=node.primitive,
            children=new_children,
            params=dict(node.params),
        )

    msg = f"Unknown node type: {type(node)}"  # pragma: no cover
    raise TypeError(msg)  # pragma: no cover


class _TimeLimitReached(Exception):
    """Internal signal that the wall-clock time limit was exceeded."""


def _run_optimization(
    program: Program,
    evaluator: Any,
    context: dict[str, np.ndarray],
    config: GPConfig,
    rng: np.random.Generator,
    initial_constants: list[float],
) -> Program:
    """Execute scipy Nelder-Mead optimization with time + iter limits."""
    x0 = np.array(initial_constants, dtype=np.float64)

    # Add small deterministic perturbation to break symmetry
    perturbation = rng.uniform(-0.01, 0.01, size=len(x0))
    x0 = x0 + perturbation

    start_time = time.monotonic()
    max_time = config.constant_opt_max_time_seconds
    maximize = config.fitness.objective_directions[0] == "maximize"

    # Track the best result seen so far
    best_loss = [float("inf")]
    best_x = [x0.copy()]

    def time_callback(_xk: np.ndarray) -> bool:
        """Return True to stop optimization when time limit exceeded."""
        elapsed = time.monotonic() - start_time
        return elapsed >= max_time

    def objective(params: np.ndarray) -> float:
        """Negative primary fitness (we minimize, so negate the objective)."""
        # Check time limit inside objective too, since callback only
        # fires between iterations, not between function evaluations.
        elapsed = time.monotonic() - start_time
        if elapsed >= max_time:
            raise _TimeLimitReached

        candidate = inject_constants(program, params.tolist())
        results = evaluator.evaluate([candidate], context)
        fitness_value = results[0].objectives[0]

        loss = -fitness_value if maximize else fitness_value

        # Handle NaN by returning a large penalty
        if np.isnan(loss):
            return 1e18

        # Track best
        if loss < best_loss[0]:
            best_loss[0] = loss
            best_x[0] = params.copy()

        return loss

    try:
        result = minimize(
            objective,
            x0,
            method="Nelder-Mead",
            callback=time_callback,
            options={
                "maxiter": config.constant_opt_max_iter,
                "xatol": 1e-8,
                "fatol": 1e-8,
                "adaptive": True,
            },
        )

        # Use the best constants found (either from result or tracked best)
        if result.success or np.isfinite(result.fun):
            optimized_x = result.x.tolist()
        else:
            optimized_x = best_x[0].tolist()
    except _TimeLimitReached:
        # Time limit was hit inside objective; use best found so far
        optimized_x = best_x[0].tolist()

    return inject_constants(program, optimized_x)
