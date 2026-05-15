"""Constant optimization for GP programs (FR-6).

Extracts ConstantNode values from a program tree, optimizes them via
scipy.optimize.minimize, and writes optimized values back into a new
AST (immutability preserved).
"""

from __future__ import annotations

import logging
import math
import time
from typing import TYPE_CHECKING, Any, Literal

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

ConstantRole = Literal[
    "gate_threshold",
    "gate_slope",
    "expert_weight",
    "risk_scale",
    "other",
]

_ROLE_PRIORITY: tuple[ConstantRole, ...] = (
    "gate_threshold",
    "gate_slope",
    "expert_weight",
    "risk_scale",
    "other",
)

_GATE_PARAM_ROLE_MAP: dict[str, dict[int, ConstantRole]] = {
    "smooth_gate": {1: "gate_threshold", 2: "gate_slope"},
    "softmax_gate": {2: "gate_slope"},
    "hysteresis_gate": {1: "gate_threshold", 2: "gate_slope"},
    "cooldown_gate": {1: "gate_threshold", 2: "gate_slope"},
    "logistic_gate": {1: "gate_threshold", 2: "gate_slope"},
}


def infer_constant_roles(program: Program) -> list[ConstantRole]:
    """Infer per-constant roles for role-aware optimization.

    The role tags are returned in the same order as ``extract_constants``. Known
    tags are:

    - ``gate_threshold``
    - ``gate_slope``
    - ``expert_weight``
    - ``risk_scale``
    - ``other`` (default fallback)
    """
    from liq.gp.evolution.operators import _collect_regime_block_roles

    regime_roles = _collect_regime_block_roles(program)
    roles: list[ConstantRole] = []

    def _walk(
        node: Program,
        parent: Program | None = None,
        child_index: int | None = None,
    ) -> None:
        if isinstance(node, ConstantNode):
            roles.append(_infer_constant_role(node, parent, child_index, regime_roles))
            return
        if isinstance(node, (FunctionNode, ParameterizedNode)):
            for index, child in enumerate(node.children):
                _walk(child, node, index)

    _walk(program)
    return roles


def infer_program_constant_role(program: Program) -> ConstantRole:
    """Return the dominant role of all constants in ``program``."""
    roles = infer_constant_roles(program)
    if not roles:
        return "other"
    for role in _ROLE_PRIORITY:
        if role in roles:
            return role
    return "other"


def _infer_constant_role(
    node: ConstantNode,
    parent: Program | None,
    child_index: int | None,
    regime_roles: dict[int, str],
) -> ConstantRole:
    if parent is None or child_index is None:
        return "other"

    if isinstance(parent, (FunctionNode, ParameterizedNode)):
        explicit = _GATE_PARAM_ROLE_MAP.get(parent.primitive.name, {}).get(
            child_index,
        )
        if explicit is not None:
            return explicit

        if regime_roles.get(id(parent)) in {"risk", "risk_scale"}:
            return "risk_scale"

        if parent.primitive.name == "mul" and len(parent.children) == 2:
            sibling = parent.children[1 - child_index]
            sibling_role = regime_roles.get(id(sibling))

            if sibling_role == "risk":
                return "risk_scale"

            if sibling_role is not None and sibling_role.startswith("expert"):
                return "expert_weight"

    if isinstance(parent, (FunctionNode, ParameterizedNode)) and regime_roles.get(
        id(node)
    ) in {"risk", "risk_scale"}:
        return "risk_scale"

    if id(node) in regime_roles:
        mapped_role = regime_roles[id(node)]
        if mapped_role == "risk":
            return "risk_scale"
        if mapped_role.startswith("expert"):
            return "expert_weight"
        if mapped_role == "gate":
            return "gate_threshold"
        if mapped_role == "detector":
            return "gate_threshold"

    if (
        isinstance(node, ConstantNode)
        and isinstance(parent, (FunctionNode, ParameterizedNode))
        and parent.primitive.name
        in {
            "gt",
            "ge",
            "lt",
            "le",
        }
    ):
        return "gate_threshold"

    return "other"


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
        return _run_optimization(
            program,
            evaluator,
            context,
            config,
            rng,
            current_constants,
        )
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
    rng: np.random.Generator | None = None,
    max_evals: int | None = None,
    generation: int | None = None,
) -> list[int]:
    """Return sorted indices of programs eligible for constant optimization.

    Programs with zero constants are excluded.

    Args:
        population: The full population of programs.
        fitnesses: Corresponding fitness results.
        config: GP configuration.
        rng: Optional RNG for stochastic modes. A new generator is created
            when omitted.
        max_evals: Optional hard cap on how many candidates can be selected.

    Returns:
        Sorted list of indices into ``population``.
    """
    if rng is None:
        rng = np.random.default_rng()

    if max_evals is None:
        max_evals = config.constant_opt_max_evals

    if max_evals is not None and max_evals <= 0:
        return []

    # Filter to programs that have at least one constant
    eligible: list[tuple[int, float, ConstantRole]] = []
    for i, (prog, fit) in enumerate(zip(population, fitnesses, strict=True)):
        if extract_constants(prog):
            role = infer_program_constant_role(prog)
            if _is_role_allowed_this_generation(
                role,
                generation,
                config,
            ):
                eligible.append((i, fit.objectives[0], role))

    if not eligible:
        return []

    maximize = config.fitness.objective_directions[0] == "maximize"
    eligible.sort(key=lambda x: x[1], reverse=maximize)

    if config.constant_opt_mode == "top_k":
        # Select top-K fraction, at least 1.
        k = max(1, int(len(eligible) * config.constant_opt_top_k))
        if max_evals is not None:
            k = min(k, max_evals)
        selected_indices = [idx for idx, _fit, _role in eligible[:k]]
    else:
        # Rank-proportional sampling (probabilistic mode).
        # Best individuals receive higher selection probability.
        n_eligible = len(eligible)
        if n_eligible == 0:
            return []
        budget = (
            max_evals
            if max_evals is not None
            else max(1, math.ceil(n_eligible * config.constant_opt_top_k))
        )
        budget = min(budget, n_eligible)
        if budget <= 0:
            return []

        # Linearly decreasing rank weights: best gets weight n_eligible, next n_eligible-1, etc.
        ranks = n_eligible - np.arange(n_eligible, dtype=float)
        probabilities = ranks / np.sum(ranks)

        sampled = rng.choice(
            n_eligible,
            size=budget,
            replace=False,
            p=probabilities,
        )
        selected_indices: list[int] = []
        for sampled_position in sampled:
            selected_indices.append(eligible[int(sampled_position)][0])

    # Return sorted ascending
    selected_indices.sort()
    return selected_indices


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _is_role_allowed_this_generation(
    role: ConstantRole,
    generation: int | None,
    config: GPConfig,
) -> bool:
    if generation is None:
        return True

    schedule = config.constant_opt_role_schedule
    interval = {
        "gate_threshold": schedule.gate_eval_interval,
        "gate_slope": schedule.gate_eval_interval,
        "expert_weight": schedule.expert_eval_interval,
        "risk_scale": schedule.risk_eval_interval,
        "other": schedule.other_eval_interval,
    }[role]

    return generation % interval == 0


def _constant_bounds_for_role(
    role: ConstantRole,
    config: GPConfig,
) -> tuple[float | None, float | None]:
    bounds_cfg = config.constant_opt_role_bounds
    if role == "gate_threshold":
        return bounds_cfg.gate_threshold
    if role == "gate_slope":
        return bounds_cfg.gate_slope
    if role == "expert_weight":
        return bounds_cfg.expert_weight
    if role == "risk_scale":
        return bounds_cfg.risk_scale
    return (None, None)


def _collect_constant_bounds(
    roles: list[ConstantRole],
    config: GPConfig,
) -> list[tuple[float | None, float | None]]:
    return [_constant_bounds_for_role(role, config) for role in roles]


def _coerce_initial_constants_with_bounds(
    x0: np.ndarray,
    bounds: list[tuple[float | None, float | None]],
) -> np.ndarray:
    bounded = x0.astype(np.float64, copy=True)
    for index, (lower, upper) in enumerate(bounds):
        if lower is not None:
            bounded[index] = max(lower, bounded[index])
        if upper is not None:
            bounded[index] = min(upper, bounded[index])
    return bounded


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
    """Run scipy optimization with deterministic time/iteration limits."""
    roles = infer_constant_roles(program)
    if len(roles) != len(initial_constants):
        raise ValueError(
            f"Constant role mismatch: got {len(initial_constants)} "
            f"constants but inferred {len(roles)} roles"
        )

    x0 = np.array(initial_constants, dtype=np.float64)
    bounds = _collect_constant_bounds(roles, config)
    x0 = _coerce_initial_constants_with_bounds(x0, bounds)

    # Add small deterministic perturbation to break symmetry
    perturbation = rng.uniform(-0.01, 0.01, size=len(x0))
    x0 = x0 + perturbation
    x0 = _coerce_initial_constants_with_bounds(x0, bounds)

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

    # FR-6.2 suggests L-BFGS-B with Nelder-Mead fallback.
    # We attempt L-BFGS-B first for faster convergence on smooth objectives,
    # then fall back to Nelder-Mead if L-BFGS-B fails.
    try:
        optimized_x = _minimize_constants(
            objective,
            x0,
            time_callback,
            max_iter=config.constant_opt_max_iter,
            method="L-BFGS-B",
            bounds=bounds,
            best_x=best_x,
        )
    except _TimeLimitReached:
        optimized_x = best_x[0].tolist()
    except Exception:
        try:
            optimized_x = _minimize_constants(
                objective,
                x0,
                time_callback,
                max_iter=config.constant_opt_max_iter,
                method="Nelder-Mead",
                bounds=None,
                best_x=best_x,
            )
        except _TimeLimitReached:
            optimized_x = best_x[0].tolist()

    # Use the best constants found (either optimizer output or tracked best)
    if optimized_x is None:
        optimized_x = best_x[0].tolist()

    return inject_constants(program, optimized_x)


def _minimize_constants(
    objective: Any,
    x0: np.ndarray,
    time_callback: Any,
    *,
    max_iter: int,
    method: str,
    bounds: list[tuple[float | None, float | None]] | None,
    best_x: list[np.ndarray],
) -> list[float]:
    """Run one scipy optimization pass and return candidate constants.

    Returns the best constants seen during the pass, or raises on failure.
    """
    minimize_kwargs = {
        "method": method,
        "options": {
            "maxiter": max_iter,
        },
    }

    if method == "L-BFGS-B":
        minimize_kwargs["options"]["ftol"] = 1e-8
    elif method == "Nelder-Mead":
        minimize_kwargs["options"].update(
            {
                "xatol": 1e-8,
                "fatol": 1e-8,
                "adaptive": True,
            }
        )

    if method == "L-BFGS-B" and bounds is not None:
        minimize_kwargs["bounds"] = bounds

    result = minimize(
        objective,
        x0,
        callback=time_callback,
        **minimize_kwargs,
    )

    # Return tracked best when optimizer does not report success.
    if result.success or np.isfinite(result.fun):
        return result.x.tolist()

    return best_x[0].tolist()
