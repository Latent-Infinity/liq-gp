"""Deterministic algebraic simplification for GP programs (FR-7).

Performs a single bottom-up pass applying rewrite rules.  All rules are
deterministic and preserve semantic equivalence (FR-7.3).  The pass is
idempotent: ``simplify(simplify(p)) == simplify(p)`` (FR-7.4).

The rules registry is extensible (FR-7.5) and individual rules can be
enabled or disabled (FR-7.6).
"""

from __future__ import annotations

from collections.abc import Callable

from liq.gp.program.ast import (
    ConstantNode,
    FunctionNode,
    ParameterizedNode,
    Program,
    TerminalNode,
)

# A simplification rule takes a node and returns a replacement node or None
# (meaning no change).
SimplificationRule = Callable[[Program], Program | None]


class SimplificationRegistry:
    """Holds named simplification rules with enable/disable support (FR-7.5, FR-7.6)."""

    def __init__(self) -> None:
        self._rules: list[tuple[str, SimplificationRule]] = []
        self._disabled: set[str] = set()

    @property
    def rules(self) -> list[tuple[str, SimplificationRule]]:
        """Return a copy of the list of (name, rule) tuples."""
        return list(self._rules)

    def add_rule(
        self,
        rule: SimplificationRule,
        name: str | None = None,
    ) -> None:
        """Register a simplification rule.

        Args:
            rule: A callable ``(Program) -> Program | None``.
            name: Optional name for enable/disable support.  If *None*,
                  an auto-generated name is used.
        """
        if name is None:
            name = f"_anon_{len(self._rules)}"
        self._rules.append((name, rule))

    def disable_rule(self, name: str) -> None:
        """Disable the rule with the given *name*.

        Raises:
            KeyError: If no rule with that name exists.
        """
        if not any(n == name for n, _ in self._rules):
            raise KeyError(name)
        self._disabled.add(name)

    def enable_rule(self, name: str) -> None:
        """Re-enable a previously disabled rule.

        Raises:
            KeyError: If no rule with that name exists.
        """
        if not any(n == name for n, _ in self._rules):
            raise KeyError(name)
        self._disabled.discard(name)

    def active_rules(self) -> list[SimplificationRule]:
        """Return only the currently enabled rules."""
        return [rule for name, rule in self._rules if name not in self._disabled]


# ---------------------------------------------------------------------------
# Built-in rewrite rules (FR-7.2)
# ---------------------------------------------------------------------------


def _is_constant(node: Program, value: float) -> bool:
    """Return True if *node* is a ConstantNode with the given *value*."""
    return isinstance(node, ConstantNode) and node.value == value


def _is_zero(node: Program) -> bool:
    """Return True if *node* is a zero constant."""
    return _is_constant(node, 0.0)


def _is_one(node: Program) -> bool:
    """Return True if *node* is a one constant."""
    return _is_constant(node, 1.0)


def _identity_elimination(node: Program) -> Program | None:
    """x + 0 -> x, 0 + x -> x, x * 1 -> x, 1 * x -> x."""
    if not isinstance(node, FunctionNode):
        return None

    name = node.primitive.name
    children = node.children

    if name == "add" and len(children) == 2:
        if _is_constant(children[1], 0.0):
            return children[0]
        if _is_constant(children[0], 0.0):
            return children[1]

    if name == "mul" and len(children) == 2:
        if _is_constant(children[1], 1.0):
            return children[0]
        if _is_constant(children[0], 1.0):
            return children[1]

    return None


def _zero_annihilation(node: Program) -> Program | None:
    """x * 0 -> 0, 0 * x -> 0."""
    if not isinstance(node, FunctionNode):
        return None

    if (
        node.primitive.name == "mul"
        and len(node.children) == 2
        and (_is_constant(node.children[0], 0.0) or _is_constant(node.children[1], 0.0))
    ):
        return ConstantNode(value=0.0)

    return None


def _self_cancellation(node: Program) -> Program | None:
    """x - x -> 0, x / x -> 1 (when subtrees are structurally equal)."""
    if not isinstance(node, FunctionNode):
        return None

    name = node.primitive.name
    children = node.children

    if len(children) != 2:
        return None

    if children[0] == children[1]:
        if name == "sub":
            return ConstantNode(value=0.0)
        if name == "div":
            return ConstantNode(value=1.0)

    return None


def _dead_branch_elimination(node: Program) -> Program | None:
    """if(True, a, b) -> a, if(False, a, b) -> b.

    The branch is selected purely from constant first-argument predicates.
    """
    if not isinstance(node, FunctionNode):
        return None

    if node.primitive.name not in ("if", "where"):
        return None

    children = node.children
    if len(children) != 3:
        return None

    if _is_one(children[0]):
        return children[1]
    if _is_zero(children[0]):
        return children[2]
    return None


def _double_negation(node: Program) -> Program | None:
    """neg(neg(x)) -> x."""
    if not isinstance(node, FunctionNode):
        return None
    if node.primitive.name != "neg" or len(node.children) != 1:
        return None

    inner = node.children[0]
    if (
        isinstance(inner, FunctionNode)
        and inner.primitive.name == "neg"
        and len(inner.children) == 1
    ):
        return inner.children[0]

    return None


def _double_not(node: Program) -> Program | None:
    """not(not(x)) -> x."""
    if not isinstance(node, FunctionNode):
        return None
    if node.primitive.name != "not" or len(node.children) != 1:
        return None

    inner = node.children[0]
    if (
        isinstance(inner, FunctionNode)
        and inner.primitive.name == "not"
        and len(inner.children) == 1
    ):
        return inner.children[0]

    return None


def _abs_idempotence(node: Program) -> Program | None:
    """abs(abs(x)) -> abs(x)."""
    if not isinstance(node, FunctionNode):
        return None
    if node.primitive.name != "abs" or len(node.children) != 1:
        return None

    inner = node.children[0]
    if isinstance(inner, FunctionNode) and inner.primitive.name == "abs":
        return inner

    return None


def _constant_folding(node: Program) -> Program | None:
    """Operations on ConstantNodes produce a ConstantNode.

    Handles both unary (e.g. neg(3) -> -3) and binary (e.g. 3 + 5 -> 8)
    operations.  If the callable raises (e.g. division by zero), the node
    is left unchanged.

    The resulting ConstantNode preserves the original node's output_type
    so that parent nodes expecting a specific type (e.g. BoolSeries) are
    not broken by the simplification.
    """
    if not isinstance(node, FunctionNode):
        return None

    if not all(isinstance(c, ConstantNode) for c in node.children):
        return None

    args = [c.value for c in node.children]  # type: ignore[union-attr]
    try:
        result = node.primitive.callable(*args)
    except Exception:  # noqa: BLE001 – guard against div-by-zero etc.
        return None

    return ConstantNode(value=float(result), output_type=node.primitive.output_type)


# ---------------------------------------------------------------------------
# Default registry
# ---------------------------------------------------------------------------


def default_rules() -> SimplificationRegistry:
    """Return a new :class:`SimplificationRegistry` populated with the
    built-in FR-7.2 rules.

    A fresh instance is returned on every call so that callers can
    extend or disable rules without affecting other users.
    """
    registry = SimplificationRegistry()
    # Order matters: zero annihilation before identity so that ``mul(0, 1)``
    # yields 0 rather than entering the identity path.  Self-cancellation,
    # dead-branch elimination, and boolean/absolute idempotence are applied
    # before constant folding.
    registry.add_rule(_zero_annihilation, name="zero_annihilation")
    registry.add_rule(_identity_elimination, name="identity_elimination")
    registry.add_rule(_self_cancellation, name="self_cancellation")
    registry.add_rule(_dead_branch_elimination, name="dead_branch_elimination")
    registry.add_rule(_double_negation, name="double_negation")
    registry.add_rule(_double_not, name="double_not")
    registry.add_rule(_abs_idempotence, name="abs_idempotence")
    registry.add_rule(_constant_folding, name="constant_folding")
    return registry


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def simplify(
    program: Program,
    registry: SimplificationRegistry | None = None,
) -> Program:
    """Simplify *program* in a single deterministic bottom-up pass (FR-7).

    Children are simplified first, then the rules in *registry* are
    applied to each node.  The first rule that returns a non-``None``
    result wins; its output replaces the node.

    Args:
        program: The root node of the program tree.
        registry: Rules to apply.  If *None*, :func:`default_rules` is
                  used.

    Returns:
        A simplified (possibly identical) program tree.
    """
    if registry is None:
        registry = default_rules()

    return _simplify_node(program, registry)


def _simplify_node(
    node: Program,
    registry: SimplificationRegistry,
) -> Program:
    """Recursively simplify *node* bottom-up."""
    # Leaf nodes: nothing to recurse into
    if isinstance(node, (TerminalNode, ConstantNode)):
        return _apply_rules(node, registry)

    # FunctionNode: simplify children, rebuild, then apply rules
    if isinstance(node, FunctionNode):
        new_children = tuple(_simplify_node(child, registry) for child in node.children)
        if new_children != node.children:
            rebuilt = FunctionNode(primitive=node.primitive, children=new_children)
        else:
            rebuilt = node
        return _apply_rules(rebuilt, registry)

    # ParameterizedNode: simplify children, rebuild, then apply rules
    if isinstance(node, ParameterizedNode):
        new_children = tuple(_simplify_node(child, registry) for child in node.children)
        if new_children != node.children:
            rebuilt = ParameterizedNode(
                primitive=node.primitive,
                children=new_children,
                params=node.params,
            )
        else:
            rebuilt = node
        return _apply_rules(rebuilt, registry)

    return node  # pragma: no cover – unreachable for valid Program types


def _apply_rules(
    node: Program,
    registry: SimplificationRegistry,
) -> Program:
    """Apply the first matching rule to *node*."""
    for rule in registry.active_rules():
        result = rule(node)
        if result is not None:
            return result
    return node
