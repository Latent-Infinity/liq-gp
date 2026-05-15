"""Genetic operators for GP evolution (FR-5.2).

Provides: subtree crossover, subtree mutation, point mutation,
parameter mutation, hoist mutation, and operator selection.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, TypeGuard

import numpy as np

from liq.gp.program.ast import (
    ConstantNode,
    FunctionNode,
    ParameterizedNode,
    TerminalNode,
)
from liq.gp.types import BoolSeries, Series

if TYPE_CHECKING:
    from liq.gp.config import GPConfig
    from liq.gp.primitives.registry import PrimitiveRegistry
    from liq.gp.program.ast import Program


LOGGER = logging.getLogger(__name__)


@dataclass
class BlockConstraintTelemetry:
    """Simple telemetry counters for block-constrained operators.

    Keys have the form ``"<operation>:<role>"`` where role may be one of
    ``risk``, ``gate``, ``detector``, ``expert:0`` etc.
    """

    attempted: dict[str, int] = field(default_factory=dict)
    blocked: dict[str, int] = field(default_factory=dict)
    accepted: dict[str, int] = field(default_factory=dict)

    def _inc(self, bucket: dict[str, int], key: str) -> None:
        bucket[key] = bucket.get(key, 0) + 1

    def record_attempt(self, operation: str, role: str, *, blocked: bool) -> None:
        key = f"{operation}:{role}"
        self._inc(self.attempted, key)
        if blocked:
            self._inc(self.blocked, key)
            return
        self._inc(self.accepted, key)


@dataclass(frozen=True)
class RegimeMotif:
    """Frequent module signature observed in elite regime programs."""

    role: str
    signature: str
    frequency: int
    exemplar: Program


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


def _is_named_primitive(
    node: Program,
    name: str,
) -> TypeGuard[FunctionNode | ParameterizedNode]:
    return (
        isinstance(node, (FunctionNode, ParameterizedNode))
        and node.primitive.name == name
    )


def _assign_role(
    role_map: dict[int, str],
    node: Program,
    role: str,
) -> None:
    """Assign a role to ``node`` and all descendants."""
    role_map[id(node)] = role
    if isinstance(node, (FunctionNode, ParameterizedNode)):
        for child in node.children:
            _assign_role(role_map, child, role)


def _assign_expert_role(
    role_map: dict[int, str],
    node: Program,
    role: str,
) -> None:
    """Assign roles for weighted expert terms and their descendants."""
    if not isinstance(node, (FunctionNode, ParameterizedNode)):
        role_map[id(node)] = role
        return

    # Weighted experts are typically ``mul(expert, weight)``.
    if _is_named_primitive(node, "mul") and len(node.children) == 2:
        expert_child: Program | None = None
        for child in node.children:
            if isinstance(child, ConstantNode):
                continue
            if _is_named_primitive(child, "if_then_else"):
                # Defensive: do not mistake nested control-flow for expert payload.
                expert_child = child
            else:
                expert_child = child
        if expert_child is None:
            expert_child = node.children[0]
        _assign_role(role_map, expert_child, role)
        return

    role_map[id(node)] = role
    for child in node.children:
        _assign_role(role_map, child, role)


def _collect_expert_terms(
    node: Program,
    role_map: dict[int, str],
    expert_terms: list[Program],
) -> None:
    """Collect weighted expert terms in deterministic left-to-right fold order."""
    if _is_named_primitive(node, "add") and node.primitive.arity == 3:
        # Defensive fallback: non-binary add is not expected; keep as a term.
        expert_terms.append(node)
        return

    if _is_named_primitive(node, "add") and node.primitive.arity == 2:
        role_map[id(node)] = "expert_fold"
        for child in node.children:
            _collect_expert_terms(child, role_map, expert_terms)
        return

    expert_terms.append(node)


def _collect_regime_block_roles(program: Program) -> dict[int, str]:
    """Infer regime block roles from a compiled-regime AST.

    Returns ``{id(node): role}`` for trees that match the canonical layout from
    ``types_regime.compile_regime_model_to_program``.
    """
    role_map: dict[int, str] = {}

    def looks_like_gate_root(
        node: Program,
    ) -> TypeGuard[FunctionNode | ParameterizedNode]:
        return (
            _is_named_primitive(node, "if_then_else")
            and node.primitive.arity == 3
            and node.primitive.output_type is Series
            and node.children[0].output_type is BoolSeries
            and node.children[1].output_type is Series
            and node.children[2].output_type is Series
        )

    gate_root: FunctionNode | ParameterizedNode | None = None

    # Optional risk wrapper: mul(risk, active_gate_tree)
    if _is_named_primitive(program, "mul") and program.primitive.arity == 2:
        left, right = program.children
        if looks_like_gate_root(left):
            gate_root = left
            _assign_role(role_map, right, "risk")
        elif looks_like_gate_root(right):
            gate_root = right
            _assign_role(role_map, left, "risk")
        else:
            return role_map
    elif looks_like_gate_root(program):
        gate_root = program
    else:
        return role_map

    # gate_root: if_then_else(gate, if_then_else(detector,...), 0)
    assert gate_root is not None
    if len(gate_root.children) != 3:
        return role_map

    gate_condition = gate_root.children[0]
    detector_branch = gate_root.children[1]
    if gate_condition.output_type is not BoolSeries:
        return role_map

    _assign_role(role_map, gate_condition, "gate")
    _assign_role(role_map, gate_root.children[2], "gate")
    _assign_role(role_map, gate_root, "gate")

    if not looks_like_gate_root(detector_branch):
        return role_map

    # detector_root: if_then_else(detector, experts_fold, 0)
    if len(detector_branch.children) != 3:
        return role_map

    detector_expr = detector_branch.children[0]
    experts_root = detector_branch.children[1]
    if detector_expr.output_type is not BoolSeries:
        return role_map

    _assign_role(role_map, detector_branch, "detector")
    _assign_role(role_map, detector_expr, "detector")
    _assign_role(role_map, detector_branch.children[2], "detector")

    expert_terms: list[Program] = []
    _collect_expert_terms(experts_root, role_map, expert_terms)
    if not expert_terms:
        return role_map

    for index, term in enumerate(expert_terms):
        role = f"expert:{index}"
        if _is_named_primitive(term, "if_then_else"):
            _assign_role(role_map, term, role)
        else:
            _assign_expert_role(role_map, term, role)

    return role_map


def _classify_crossover_roles(
    role_map1: dict[int, str],
    role_map2: dict[int, str],
    node1: Program,
    node2: Program,
) -> tuple[str | None, str | None]:
    return role_map1.get(id(node1)), role_map2.get(id(node2))


def _is_block_compatible(role1: str | None, role2: str | None) -> bool:
    if role1 is None or role2 is None:
        return False
    return role1 == role2


def _record_block_result(
    telemetry: BlockConstraintTelemetry | None,
    operation: str,
    role: str,
    *,
    blocked: bool,
) -> None:
    if telemetry is None:
        return
    telemetry.record_attempt(operation, role, blocked=blocked)


def _is_module_role(role: str) -> bool:
    return role in {"gate", "detector", "risk"} or role.startswith("expert:")


def _module_signature(program: Program) -> str:
    from liq.gp.program.serialize import serialize

    payload = serialize(program)
    return json.dumps(payload["program"], sort_keys=True, separators=(",", ":"))


def _collect_role_boundary_modules(program: Program) -> dict[str, list[Program]]:
    role_map = _collect_regime_block_roles(program)
    if not role_map:
        return {}

    modules: dict[str, list[Program]] = {}

    def visit(node: Program, parent_role: str | None) -> None:
        role = role_map.get(id(node))
        current_role = role if role is not None else parent_role
        if role is not None and role != parent_role and _is_module_role(role):
            modules.setdefault(role, []).append(node)

        if isinstance(node, (FunctionNode, ParameterizedNode)):
            for child in node.children:
                visit(child, current_role)

    visit(program, None)
    return modules


def extract_regime_modules(program: Program) -> dict[str, list[Program]]:
    """Extract module-boundary subtrees keyed by regime role."""
    return _collect_role_boundary_modules(program)


def mine_regime_motifs(
    elites: list[Program],
    *,
    min_frequency: int = 2,
) -> list[RegimeMotif]:
    """Mine frequent role-bounded motifs from elite programs."""
    if min_frequency < 1:
        raise ValueError("min_frequency must be >= 1")

    counts: Counter[tuple[str, str]] = Counter()
    exemplars: dict[tuple[str, str], Program] = {}
    for program in elites:
        role_modules = extract_regime_modules(program)
        for role, modules in role_modules.items():
            for module in modules:
                signature = _module_signature(module)
                key = (role, signature)
                counts[key] += 1
                exemplars.setdefault(key, module)

    motifs = [
        RegimeMotif(
            role=role,
            signature=signature,
            frequency=count,
            exemplar=exemplars[(role, signature)],
        )
        for (role, signature), count in counts.items()
        if count >= min_frequency
    ]
    motifs.sort(key=lambda motif: (-motif.frequency, motif.role, motif.signature))
    return motifs


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
    block_constraint_telemetry: BlockConstraintTelemetry | None = None,
    enforce_block_constraints: bool = True,
) -> tuple[Program, Program]:
    """Swap compatible-type subtrees between two parents (FR-5.2.1).

    Returns copies of the parents if no valid swap point is found.
    """
    nodes1 = _collect_subtrees_by_type(parent1)
    nodes2 = _collect_subtrees_by_type(parent2)

    role_map1: dict[int, str] = {}
    role_map2: dict[int, str] = {}
    if enforce_block_constraints:
        role_map1 = _collect_regime_block_roles(parent1)
        role_map2 = _collect_regime_block_roles(parent2)
    enforce_roles = bool(role_map1 and role_map2)

    # Filter to internal nodes (at least one function node) for swap points
    # But also allow swapping at any node
    for _ in range(max(1, max_attempts)):
        n1, d1 = nodes1[int(rng.integers(len(nodes1)))]
        n2, d2 = nodes2[int(rng.integers(len(nodes2)))]
        role1: str | None = None

        if enforce_roles:
            role1, role2 = _classify_crossover_roles(role_map1, role_map2, n1, n2)
            if not _is_block_compatible(role1, role2):
                for role in {role for role in (role1, role2) if role is not None}:
                    _record_block_result(
                        block_constraint_telemetry,
                        "crossover",
                        role,
                        blocked=True,
                    )
                    LOGGER.debug(
                        "crossover blocked by block constraints: %s vs %s",
                        role1,
                        role2,
                    )
                continue

        # Type compatibility
        if n1.output_type != n2.output_type:
            continue

        # Check depth constraints after swap
        child1 = _replace_at_node(parent1, n1, n2)
        child2 = _replace_at_node(parent2, n2, n1)

        if child1.depth <= max_depth and child2.depth <= max_depth:
            if enforce_roles and role1 is not None:
                _record_block_result(
                    block_constraint_telemetry,
                    "crossover",
                    role1,
                    blocked=False,
                )
            return child1, child2

        if enforce_roles and role1 is not None:
            _record_block_result(
                block_constraint_telemetry,
                "crossover",
                role1,
                blocked=True,
            )

    # No valid swap found, return copies (identity preserving)
    return parent1, parent2


def module_preserving_crossover(
    parent1: Program,
    parent2: Program,
    _registry: PrimitiveRegistry,
    max_depth: int,
    rng: np.random.Generator,
    *,
    max_attempts: int = _MAX_CROSSOVER_ATTEMPTS,
    block_constraint_telemetry: BlockConstraintTelemetry | None = None,
) -> tuple[Program, Program]:
    """Swap same-role module boundaries without breaking regime decomposition."""
    modules1 = extract_regime_modules(parent1)
    modules2 = extract_regime_modules(parent2)
    if not modules1 or not modules2:
        return parent1, parent2

    shared_roles = sorted(set(modules1).intersection(modules2))
    if not shared_roles:
        return parent1, parent2

    for _ in range(max(1, max_attempts)):
        role = shared_roles[int(rng.integers(len(shared_roles)))]
        module1 = modules1[role][int(rng.integers(len(modules1[role])))]
        module2 = modules2[role][int(rng.integers(len(modules2[role])))]

        if module1.output_type != module2.output_type:
            _record_block_result(
                block_constraint_telemetry,
                "module_crossover",
                role,
                blocked=True,
            )
            continue

        child1 = _replace_at_node(parent1, module1, module2)
        child2 = _replace_at_node(parent2, module2, module1)
        if child1.depth <= max_depth and child2.depth <= max_depth:
            _record_block_result(
                block_constraint_telemetry,
                "module_crossover",
                role,
                blocked=False,
            )
            return child1, child2

        _record_block_result(
            block_constraint_telemetry,
            "module_crossover",
            role,
            blocked=True,
        )

    return parent1, parent2


# ---------------------------------------------------------------------------
# Subtree mutation (FR-5.2.2)
# ---------------------------------------------------------------------------


def subtree_mutation(
    tree: Program,
    registry: PrimitiveRegistry,
    max_depth: int,
    rng: np.random.Generator,
    *,
    max_attempts: int = 20,
    block_constraint_telemetry: BlockConstraintTelemetry | None = None,
    enforce_block_constraints: bool = True,
) -> Program:
    """Replace a random subtree with a new random one (FR-5.2.2)."""
    from liq.gp.evolution.init import generate_grow

    role_map: dict[int, str] = {}
    if enforce_block_constraints:
        role_map = _collect_regime_block_roles(tree)
    enforce_roles = bool(role_map)

    nodes = _collect_nodes_with_depth(tree)
    for _ in range(max(1, max_attempts)):
        target, target_depth = nodes[int(rng.integers(len(nodes)))]
        role = role_map.get(id(target)) if role_map else None
        role_label = role or "unconstrained"

        if enforce_roles and role is None:
            _record_block_result(
                block_constraint_telemetry,
                "subtree_mutation",
                role_label,
                blocked=True,
            )
            continue

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
            _record_block_result(
                block_constraint_telemetry,
                "subtree_mutation",
                role_label,
                blocked=True,
            )
            continue

        _record_block_result(
            block_constraint_telemetry,
            "subtree_mutation",
            role_label,
            blocked=False,
        )
        return result

    return tree


# ---------------------------------------------------------------------------
# Point mutation (FR-5.2.3)
# ---------------------------------------------------------------------------


def point_mutation(
    tree: Program,
    registry: PrimitiveRegistry,
    rng: np.random.Generator,
    *,
    max_attempts: int = 20,
    block_constraint_telemetry: BlockConstraintTelemetry | None = None,
    enforce_block_constraints: bool = True,
) -> Program:
    """Replace a single node with a same-arity, same-type primitive (FR-5.2.3)."""
    nodes = _collect_nodes_with_depth(tree)

    role_map: dict[int, str] = {}
    if enforce_block_constraints:
        role_map = _collect_regime_block_roles(tree)
    enforce_roles = bool(role_map)

    for _ in range(max(1, max_attempts)):
        target, _ = nodes[int(rng.integers(len(nodes)))]
        role = role_map.get(id(target)) if role_map else None
        role_label = role or "unconstrained"

        if enforce_roles and role is None:
            _record_block_result(
                block_constraint_telemetry,
                "point_mutation",
                role_label,
                blocked=True,
            )
            continue

        if isinstance(target, (TerminalNode, ConstantNode)):
            # Replace terminal with a different terminal of the same type
            from liq.gp.evolution.init import _sample_terminal

            replacement = _sample_terminal(registry, target.output_type, rng)
            _record_block_result(
                block_constraint_telemetry,
                "point_mutation",
                role_label,
                blocked=False,
            )
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
                _record_block_result(
                    block_constraint_telemetry,
                    "point_mutation",
                    role_label,
                    blocked=True,
                )
                continue

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
            _record_block_result(
                block_constraint_telemetry,
                "point_mutation",
                role_label,
                blocked=False,
            )
            return _replace_at_node(tree, target, replacement)

        _record_block_result(
            block_constraint_telemetry,
            "point_mutation",
            role_label,
            blocked=False,
        )

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
        if ps.value_is_discrete():
            assert ps.allowed_values is not None
            values = ps.allowed_values
            value_to_index = {value: idx for idx, value in enumerate(values)}
            current_index = value_to_index.get(old_val)
            if current_index is None:
                # Fallback for malformed trees: select nearest neighbor.
                nearest_index = min(
                    range(len(values)),
                    key=lambda idx: abs(float(values[idx]) - float(old_val)),
                )
                current_index = nearest_index

            # Distance-weighted neighbor selection by inverse distance.
            neighbor_indices = [
                index for index in range(len(values)) if index != current_index
            ]
            if not neighbor_indices:
                # Single-value domain: no-op (deterministic self-reference).
                new_params[ps.name] = values[current_index]
                continue

            distances = [abs(index - current_index) for index in neighbor_indices]
            weights = np.array([1.0 / distance for distance in distances], dtype=float)
            probabilities = weights / weights.sum()
            choice = int(rng.choice(len(neighbor_indices), p=probabilities))
            next_index = neighbor_indices[choice]
            new_params[ps.name] = values[next_index]
            continue

        # Gaussian noise scaled to ~10% of the range
        if ps.min_value is None or ps.max_value is None:
            raise ValueError(f"parameter {ps.name!r} requires min_value and max_value")
        min_value = float(ps.min_value)
        max_value = float(ps.max_value)
        noise_scale = (max_value - min_value) * 0.1
        noise = float(rng.normal(0, noise_scale))
        new_val = float(old_val) + noise
        # Clamp to range
        new_val = max(min_value, min(max_value, new_val))
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
