"""Evaluation engine for GP programs (FR-4).

Recursively evaluates an AST over an evaluation context.
Supports an optional subtree cache for shared subtrees (FR-4.7).
"""

from __future__ import annotations

import numpy as np

from liq.gp.program.ast import (
    ConstantNode,
    FunctionNode,
    ParameterizedNode,
    Program,
    TerminalNode,
)

# Type alias for evaluation context (FR-4.2)
EvaluationContext = dict[str, np.ndarray]


def evaluate(
    program: Program,
    context: EvaluationContext,
    *,
    use_cache: bool = False,
) -> np.ndarray:
    """Evaluate a program AST on the given context (FR-4.1).

    Args:
        program: The AST to evaluate.
        context: Mapping of names to arrays.
        use_cache: If True, cache subtree results keyed by
            ``(structural_hash, context_id)`` (FR-4.7).

    Returns:
        np.ndarray[float64] with the same length as the context arrays.
    """
    cache: dict[tuple[int, int], np.ndarray] | None = {} if use_cache else None
    context_id = id(context)
    return _eval_node(program, context, cache, context_id)


def _get_context_length(context: EvaluationContext) -> int:
    """Get the length of context arrays for broadcasting constants."""
    for arr in context.values():
        return len(arr)
    return 0  # pragma: no cover


def _eval_node(
    node: Program,
    context: EvaluationContext,
    cache: dict[tuple[int, int], np.ndarray] | None,
    context_id: int,
) -> np.ndarray:
    """Recursively evaluate a single node."""
    # Cache lookup by (structural hash, context identity) (FR-4.7)
    cache_key = (hash(node), context_id) if cache is not None else None
    if cache_key is not None and cache_key in cache:
        return cache[cache_key]

    result: np.ndarray

    if isinstance(node, TerminalNode):
        result = context[node.name]

    elif isinstance(node, ConstantNode):
        n = _get_context_length(context)
        result = np.full(n, node.value, dtype=np.float64)

    elif isinstance(node, FunctionNode):
        child_results = tuple(
            _eval_node(c, context, cache, context_id) for c in node.children
        )
        result = node.primitive.callable(*child_results)

    elif isinstance(node, ParameterizedNode):
        child_results = tuple(
            _eval_node(c, context, cache, context_id) for c in node.children
        )
        result = node.primitive.callable(*child_results, **node.params)

    else:
        msg = f"Unknown node type: {type(node)}"
        raise TypeError(msg)  # pragma: no cover

    # Ensure float64 output (FR-4.3)
    if not isinstance(result, np.ndarray):
        n = _get_context_length(context)
        result = np.full(n, float(result), dtype=np.float64)
    elif result.dtype != np.float64:
        result = result.astype(np.float64)

    if cache_key is not None:
        cache[cache_key] = result

    return result
