"""Genetic operators for GP evolution (FR-5.2).

Provides: subtree crossover, subtree mutation, point mutation,
parameter mutation, hoist mutation, and operator selection.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from liq.gp.program.ast import (
    ConstantNode,
    FunctionNode,
    ParameterizedNode,
    TerminalNode,
)

if TYPE_CHECKING:
    from liq.gp.config import GPConfig
    from liq.gp.primitives.registry import PrimitiveRegistry
    from liq.gp.program.ast import Program


# ---------------------------------------------------------------------------
# Tree utilities
# ---------------------------------------------------------------------------


def _collect_nodes_with_depth(
    program: Program,
    current_depth: int = 0,
) -> list[tuple[Program, int]]:
    """Collect all (node, depth) pairs in pre-order."""
    result: list[tuple[Program, int]] = [(program, current_depth)]
    if isinstance(program, (FunctionNode, ParameterizedNode)):
        for child in program.children:
            result.extend(_collect_nodes_with_depth(child, current_depth + 1))
    return result


def _collect_subtrees_by_type(
    program: Program,
) -> list[tuple[Program, int]]:
    """Collect (node, depth_in_tree) for all nodes."""
    return _collect_nodes_with_depth(program)


def _replace_at_node(
    tree: Program,
    target: Program,
    replacement: Program,
) -> Program:
    """Replace the first occurrence of target (by identity) with replacement."""
    if tree is target:
        return replacement
    if isinstance(tree, (TerminalNode, ConstantNode)):
        return tree
    if isinstance(tree, FunctionNode):
        new_children = tuple(
            _replace_at_node(c, target, replacement) for c in tree.children
        )
        if new_children == tree.children:
            return tree
        return FunctionNode(primitive=tree.primitive, children=new_children)
    if isinstance(tree, ParameterizedNode):
        new_children = tuple(
            _replace_at_node(c, target, replacement) for c in tree.children
        )
        if new_children == tree.children:
            return tree
        return ParameterizedNode(
            primitive=tree.primitive,
            children=new_children,
            params=tree.params,
        )
    return tree  # pragma: no cover


# ---------------------------------------------------------------------------
# Subtree crossover (FR-5.2.1)
# ---------------------------------------------------------------------------

_MAX_CROSSOVER_ATTEMPTS = 20


def subtree_crossover(
    parent1: Program,
    parent2: Program,
    _registry: PrimitiveRegistry,
    max_depth: int,
    rng: np.random.Generator,
    *,
    max_attempts: int = _MAX_CROSSOVER_ATTEMPTS,
) -> tuple[Program, Program]:
    """Swap compatible-type subtrees between two parents (FR-5.2.1).

    Returns copies of the parents if no valid swap point is found.
    """
    nodes1 = _collect_subtrees_by_type(parent1)
    nodes2 = _collect_subtrees_by_type(parent2)

    # Filter to internal nodes (at least one function node) for swap points
    # But also allow swapping at any node
    for _ in range(max(1, max_attempts)):
        n1, d1 = nodes1[int(rng.integers(len(nodes1)))]
        n2, d2 = nodes2[int(rng.integers(len(nodes2)))]

        # Type compatibility
        if n1.output_type != n2.output_type:
            continue

        # Check depth constraints after swap
        child1 = _replace_at_node(parent1, n1, n2)
        child2 = _replace_at_node(parent2, n2, n1)

        if child1.depth <= max_depth and child2.depth <= max_depth:
            return child1, child2

    # No valid swap found, return copies (identity preserving)
    return parent1, parent2


# ---------------------------------------------------------------------------
# Subtree mutation (FR-5.2.2)
# ---------------------------------------------------------------------------


def subtree_mutation(
    tree: Program,
    registry: PrimitiveRegistry,
    max_depth: int,
    rng: np.random.Generator,
) -> Program:
    """Replace a random subtree with a new random one (FR-5.2.2)."""
    from liq.gp.evolution.init import generate_grow

    nodes = _collect_nodes_with_depth(tree)
    target, target_depth = nodes[int(rng.integers(len(nodes)))]

    # Max depth available for the new subtree
    remaining_depth = max(0, max_depth - target_depth)
    replacement = generate_grow(
        registry,
        remaining_depth,
        target.output_type,
        rng,
    )
    result = _replace_at_node(tree, target, replacement)

    # Ensure depth constraint (defensive)
    if result.depth > max_depth:
        return tree
    return result


# ---------------------------------------------------------------------------
# Point mutation (FR-5.2.3)
# ---------------------------------------------------------------------------


def point_mutation(
    tree: Program,
    registry: PrimitiveRegistry,
    rng: np.random.Generator,
) -> Program:
    """Replace a single node with a same-arity, same-type primitive (FR-5.2.3)."""
    nodes = _collect_nodes_with_depth(tree)
    target, _ = nodes[int(rng.integers(len(nodes)))]

    if isinstance(target, (TerminalNode, ConstantNode)):
        # Replace terminal with a different terminal of the same type
        from liq.gp.evolution.init import _sample_terminal

        replacement = _sample_terminal(registry, target.output_type, rng)
        return _replace_at_node(tree, target, replacement)

    if isinstance(target, (FunctionNode, ParameterizedNode)):
        # Find functions with same arity and compatible input/output types
        functions = registry.functions(output_type=target.output_type)
        compatible = [
            f
            for f in functions
            if f.arity == target.primitive.arity
            and f.input_types == target.primitive.input_types
            and f.name != target.primitive.name
        ]
        if not compatible:
            return tree  # No compatible replacement found

        new_prim = compatible[int(rng.integers(len(compatible)))]
        if new_prim.param_specs:
            from liq.gp.evolution.init import _sample_params

            params = _sample_params(new_prim, rng)
            replacement = ParameterizedNode(
                primitive=new_prim,
                children=target.children,
                params=params,
            )
        else:
            replacement = FunctionNode(
                primitive=new_prim,
                children=target.children,
            )
        return _replace_at_node(tree, target, replacement)

    return tree  # pragma: no cover


# ---------------------------------------------------------------------------
# Parameter mutation (FR-5.2.4)
# ---------------------------------------------------------------------------


def _mutate_params_in_node(
    node: ParameterizedNode,
    rng: np.random.Generator,
) -> ParameterizedNode:
    """Add Gaussian noise to parameters, clamp and round as needed."""
    new_params: dict[str, int | float] = {}
    for ps in node.primitive.param_specs:
        old_val = node.params[ps.name]
        # Gaussian noise scaled to ~10% of the range
        noise_scale = (ps.max_value - ps.min_value) * 0.1
        noise = float(rng.normal(0, noise_scale))
        new_val = float(old_val) + noise
        # Clamp to range
        new_val = max(float(ps.min_value), min(float(ps.max_value), new_val))
        if ps.dtype is int:
            new_params[ps.name] = int(round(new_val))
        else:
            new_params[ps.name] = new_val
    return ParameterizedNode(
        primitive=node.primitive,
        children=node.children,
        params=new_params,
    )


def parameter_mutation(
    tree: Program,
    rng: np.random.Generator,
) -> Program:
    """Mutate parameters of a ParameterizedNode in the tree (FR-5.2.4)."""
    # Collect all parameterized nodes
    all_nodes = _collect_nodes_with_depth(tree)
    param_nodes = [(n, d) for n, d in all_nodes if isinstance(n, ParameterizedNode)]
    if not param_nodes:
        return tree  # No parameters to mutate

    target, _ = param_nodes[int(rng.integers(len(param_nodes)))]
    assert isinstance(target, ParameterizedNode)
    mutated = _mutate_params_in_node(target, rng)
    return _replace_at_node(tree, target, mutated)


# ---------------------------------------------------------------------------
# Hoist mutation (FR-5.2.5)
# ---------------------------------------------------------------------------


def hoist_mutation(
    tree: Program,
    rng: np.random.Generator,
) -> Program:
    """Replace the tree with one of its subtrees (FR-5.2.5)."""
    all_nodes = _collect_nodes_with_depth(tree)
    # Filter to subtrees with matching output type
    compatible = [(n, d) for n, d in all_nodes if n.output_type == tree.output_type]
    if len(compatible) <= 1:
        return tree  # Only root matches, nothing to hoist

    # Pick a non-root compatible subtree
    non_root = [(n, d) for n, d in compatible if n is not tree]
    if not non_root:
        return tree
    chosen, _ = non_root[int(rng.integers(len(non_root)))]
    return chosen


# ---------------------------------------------------------------------------
# Operator selection (FR-5.2.8)
# ---------------------------------------------------------------------------


def select_operator(
    config: GPConfig,
    rng: np.random.Generator,
) -> str:
    """Select a genetic operator according to configured rates."""
    r = float(rng.random())
    cumulative = 0.0
    operators = [
        ("crossover", config.crossover_rate),
        ("subtree_mutation", config.subtree_mutation_rate),
        ("point_mutation", config.point_mutation_rate),
        ("parameter_mutation", config.parameter_mutation_rate),
        ("hoist_mutation", config.hoist_mutation_rate),
    ]
    for name, rate in operators:
        cumulative += rate
        if r < cumulative:
            return name
    return operators[-1][0]  # pragma: no cover  (floating point edge)
