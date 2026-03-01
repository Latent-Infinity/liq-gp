"""Tests for GP program simplification (FR-7)."""

from __future__ import annotations

import operator

import numpy as np
import pytest

from liq.gp.primitives.registry import PrimitiveInfo
from liq.gp.program.ast import (
    ConstantNode,
    FunctionNode,
    ParameterizedNode,
    TerminalNode,
)
from liq.gp.program.simplify import (
    SimplificationRegistry,
    default_rules,
    simplify,
)
from liq.gp.types import BoolSeries, ParamSpec, Series

# --- helpers ---------------------------------------------------------------


def _make_add_info() -> PrimitiveInfo:
    return PrimitiveInfo(
        name="add",
        category="numeric",
        arity=2,
        input_types=(Series, Series),
        output_type=Series,
        callable=operator.add,
    )


def _make_sub_info() -> PrimitiveInfo:
    return PrimitiveInfo(
        name="sub",
        category="numeric",
        arity=2,
        input_types=(Series, Series),
        output_type=Series,
        callable=operator.sub,
    )


def _make_mul_info() -> PrimitiveInfo:
    return PrimitiveInfo(
        name="mul",
        category="numeric",
        arity=2,
        input_types=(Series, Series),
        output_type=Series,
        callable=operator.mul,
    )


def _make_div_info() -> PrimitiveInfo:
    return PrimitiveInfo(
        name="div",
        category="numeric",
        arity=2,
        input_types=(Series, Series),
        output_type=Series,
        callable=operator.truediv,
    )


def _make_neg_info() -> PrimitiveInfo:
    return PrimitiveInfo(
        name="neg",
        category="numeric",
        arity=1,
        input_types=(Series,),
        output_type=Series,
        callable=operator.neg,
    )


def _make_if_info() -> PrimitiveInfo:
    return PrimitiveInfo(
        name="if",
        category="logical",
        arity=3,
        input_types=(BoolSeries, Series, Series),
        output_type=Series,
        callable=lambda cond, a, b: np.where(cond > 0.5, a, b),  # pragma: no cover
    )


def _make_not_info() -> PrimitiveInfo:
    return PrimitiveInfo(
        name="not",
        category="logical",
        arity=1,
        input_types=(BoolSeries,),
        output_type=BoolSeries,
        callable=lambda x: 1.0 - x,  # pragma: no cover
    )


def _make_abs_info() -> PrimitiveInfo:
    return PrimitiveInfo(
        name="abs",
        category="math",
        arity=1,
        input_types=(Series,),
        output_type=Series,
        callable=abs,  # pragma: no cover
    )


def _make_highest_info() -> PrimitiveInfo:
    ps = ParamSpec(name="period", dtype=int, default=20, min_value=2, max_value=200)
    return PrimitiveInfo(
        name="highest",
        category="indicator",
        arity=1,
        input_types=(Series,),
        output_type=Series,
        callable=lambda a, period=20: a,
        param_specs=[ps],
    )


def _close() -> TerminalNode:
    return TerminalNode(name="close", output_type=Series)


def _const(value: float) -> ConstantNode:
    return ConstantNode(value=value)


def _const_bool(value: float) -> ConstantNode:
    return ConstantNode(value=value, output_type=BoolSeries)


# --- Identity elimination: x + 0 -> x, 0 + x -> x -------------------------


class TestAddIdentity:
    """FR-7.2: x + 0 -> x and 0 + x -> x."""

    def test_x_plus_zero(self) -> None:
        add_info = _make_add_info()
        x = _close()
        tree = FunctionNode(primitive=add_info, children=(x, _const(0.0)))
        result = simplify(tree)
        assert result == x

    def test_zero_plus_x(self) -> None:
        add_info = _make_add_info()
        x = _close()
        tree = FunctionNode(primitive=add_info, children=(_const(0.0), x))
        result = simplify(tree)
        assert result == x

    def test_x_plus_nonzero_unchanged(self) -> None:
        add_info = _make_add_info()
        x = _close()
        tree = FunctionNode(primitive=add_info, children=(x, _const(5.0)))
        result = simplify(tree)
        assert result == tree


# --- Identity elimination: x * 1 -> x, 1 * x -> x -------------------------


class TestMulIdentity:
    """FR-7.2: x * 1 -> x and 1 * x -> x."""

    def test_x_times_one(self) -> None:
        mul_info = _make_mul_info()
        x = _close()
        tree = FunctionNode(primitive=mul_info, children=(x, _const(1.0)))
        result = simplify(tree)
        assert result == x

    def test_one_times_x(self) -> None:
        mul_info = _make_mul_info()
        x = _close()
        tree = FunctionNode(primitive=mul_info, children=(_const(1.0), x))
        result = simplify(tree)
        assert result == x


# --- Zero annihilation: x * 0 -> 0, 0 * x -> 0 ----------------------------


class TestMulZeroAnnihilation:
    """FR-7.2: x * 0 -> 0 and 0 * x -> 0."""

    def test_x_times_zero(self) -> None:
        mul_info = _make_mul_info()
        x = _close()
        tree = FunctionNode(primitive=mul_info, children=(x, _const(0.0)))
        result = simplify(tree)
        assert isinstance(result, ConstantNode)
        assert result.value == 0.0

    def test_zero_times_x(self) -> None:
        mul_info = _make_mul_info()
        x = _close()
        tree = FunctionNode(primitive=mul_info, children=(_const(0.0), x))
        result = simplify(tree)
        assert isinstance(result, ConstantNode)
        assert result.value == 0.0


# --- Self-cancellation: x - x -> 0, x / x -> 1 ----------------------------


class TestSelfCancellation:
    """FR-7.2: x - x -> 0, x / x -> 1 when structurally equal."""

    def test_x_minus_x(self) -> None:
        sub_info = _make_sub_info()
        x = _close()
        tree = FunctionNode(primitive=sub_info, children=(x, x))
        result = simplify(tree)
        assert isinstance(result, ConstantNode)
        assert result.value == 0.0

    def test_x_div_x(self) -> None:
        div_info = _make_div_info()
        x = _close()
        tree = FunctionNode(primitive=div_info, children=(x, x))
        result = simplify(tree)
        assert isinstance(result, ConstantNode)
        assert result.value == 1.0

    def test_different_subtrees_not_cancelled(self) -> None:
        sub_info = _make_sub_info()
        x = _close()
        y = TerminalNode(name="volume", output_type=Series)
        tree = FunctionNode(primitive=sub_info, children=(x, y))
        result = simplify(tree)
        assert result == tree

    def test_complex_equal_subtrees_sub(self) -> None:
        """x - x should work even with nested identical subtrees."""
        add_info = _make_add_info()
        sub_info = _make_sub_info()
        x = _close()
        c = _const(2.0)
        subtree = FunctionNode(primitive=add_info, children=(x, c))
        tree = FunctionNode(primitive=sub_info, children=(subtree, subtree))
        result = simplify(tree)
        assert isinstance(result, ConstantNode)
        assert result.value == 0.0


# --- Double negation: neg(neg(x)) -> x ------------------------------------


class TestDoubleNegation:
    """FR-7.2: neg(neg(x)) -> x."""

    def test_double_neg(self) -> None:
        neg_info = _make_neg_info()
        x = _close()
        inner = FunctionNode(primitive=neg_info, children=(x,))
        outer = FunctionNode(primitive=neg_info, children=(inner,))
        result = simplify(outer)
        assert result == x

    def test_single_neg_unchanged(self) -> None:
        neg_info = _make_neg_info()
        x = _close()
        tree = FunctionNode(primitive=neg_info, children=(x,))
        result = simplify(tree)
        assert result == tree


class TestAdditionalSimplificationRules:
    """FR-7.2: additional rewrite rules now required by requirements."""

    def test_if_true_branch(self) -> None:
        if_info = _make_if_info()
        condition = _const_bool(1.0)
        tree = FunctionNode(
            primitive=if_info,
            children=(condition, _close(), _close()),
        )
        result = simplify(tree)
        assert result == _close()

    def test_if_false_branch(self) -> None:
        if_info = _make_if_info()
        x = _close()
        tree = FunctionNode(
            primitive=if_info,
            children=(_const_bool(0.0), x, _const(7.0)),
        )
        result = simplify(tree)
        assert isinstance(result, ConstantNode)
        assert result.value == 7.0

    def test_not_not_cancels(self) -> None:
        not_info = _make_not_info()
        x = TerminalNode(name="flag", output_type=BoolSeries)
        inner = FunctionNode(primitive=not_info, children=(x,))
        outer = FunctionNode(primitive=not_info, children=(inner,))
        result = simplify(outer)
        assert result == x

    def test_abs_idempotent(self) -> None:
        abs_info = _make_abs_info()
        x = _close()
        tree = FunctionNode(
            primitive=abs_info,
            children=(FunctionNode(primitive=abs_info, children=(x,)),),
        )
        result = simplify(tree)
        assert isinstance(result, FunctionNode)
        assert result.primitive.name == "abs"
        assert result.children == (x,)


# --- Constant folding -------------------------------------------------------


class TestConstantFolding:
    """FR-7.2: operations on two ConstantNodes produce a ConstantNode."""

    def test_add_constants(self) -> None:
        add_info = _make_add_info()
        tree = FunctionNode(primitive=add_info, children=(_const(3.0), _const(5.0)))
        result = simplify(tree)
        assert isinstance(result, ConstantNode)
        assert result.value == 8.0

    def test_sub_constants(self) -> None:
        sub_info = _make_sub_info()
        tree = FunctionNode(primitive=sub_info, children=(_const(10.0), _const(4.0)))
        result = simplify(tree)
        assert isinstance(result, ConstantNode)
        assert result.value == 6.0

    def test_mul_constants(self) -> None:
        mul_info = _make_mul_info()
        tree = FunctionNode(primitive=mul_info, children=(_const(3.0), _const(7.0)))
        result = simplify(tree)
        assert isinstance(result, ConstantNode)
        assert result.value == 21.0

    def test_div_constants(self) -> None:
        div_info = _make_div_info()
        tree = FunctionNode(primitive=div_info, children=(_const(10.0), _const(2.0)))
        result = simplify(tree)
        assert isinstance(result, ConstantNode)
        assert result.value == 5.0

    def test_neg_constant(self) -> None:
        neg_info = _make_neg_info()
        tree = FunctionNode(primitive=neg_info, children=(_const(3.0),))
        result = simplify(tree)
        assert isinstance(result, ConstantNode)
        assert result.value == -3.0

    def test_div_by_zero_constant_not_folded(self) -> None:
        """Division by zero should NOT be constant-folded (would raise)."""
        div_info = _make_div_info()
        tree = FunctionNode(primitive=div_info, children=(_const(1.0), _const(0.0)))
        # x / x rule does not apply (children differ), and constant folding
        # should be guarded against exceptions.
        result = simplify(tree)
        # Should remain unchanged because constant folding catches the exception
        assert result == tree

    def test_constant_folding_preserves_output_type(self) -> None:
        """Constant folding must preserve the original node's output type."""
        from liq.gp.types import BoolSeries

        gt_info = PrimitiveInfo(
            name="gt",
            category="comparison",
            arity=2,
            input_types=(Series, Series),
            output_type=BoolSeries,
            callable=lambda a, b: 1.0 if a > b else 0.0,
        )
        tree = FunctionNode(
            primitive=gt_info,
            children=(_const(3.0), _const(1.0)),
        )
        result = simplify(tree)
        assert isinstance(result, ConstantNode)
        assert result.value == 1.0
        assert result.output_type == BoolSeries


# --- Idempotency (FR-7.4) --------------------------------------------------


class TestIdempotency:
    """FR-7.4: simplify(simplify(p)) == simplify(p)."""

    def test_idempotent_add_zero(self) -> None:
        add_info = _make_add_info()
        x = _close()
        tree = FunctionNode(primitive=add_info, children=(x, _const(0.0)))
        once = simplify(tree)
        twice = simplify(once)
        assert once == twice

    def test_idempotent_nested(self) -> None:
        add_info = _make_add_info()
        neg_info = _make_neg_info()
        x = _close()
        # neg(neg(x + 0))
        inner = FunctionNode(primitive=add_info, children=(x, _const(0.0)))
        neg1 = FunctionNode(primitive=neg_info, children=(inner,))
        neg2 = FunctionNode(primitive=neg_info, children=(neg1,))
        once = simplify(neg2)
        twice = simplify(once)
        assert once == twice
        # Should simplify to just x
        assert once == x

    def test_idempotent_constant_folding(self) -> None:
        add_info = _make_add_info()
        tree = FunctionNode(primitive=add_info, children=(_const(2.0), _const(3.0)))
        once = simplify(tree)
        twice = simplify(once)
        assert once == twice

    def test_idempotent_terminal_passthrough(self) -> None:
        x = _close()
        once = simplify(x)
        twice = simplify(once)
        assert once == twice == x

    def test_idempotent_constant_passthrough(self) -> None:
        c = _const(42.0)
        once = simplify(c)
        twice = simplify(once)
        assert once == twice == c


# --- Bottom-up pass behavior ------------------------------------------------


class TestBottomUpPass:
    """The simplifier should work bottom-up, simplifying children first."""

    def test_nested_add_zero_then_identity(self) -> None:
        """add(add(x, 0), 0) -> x in a single pass (bottom-up)."""
        add_info = _make_add_info()
        x = _close()
        inner = FunctionNode(primitive=add_info, children=(x, _const(0.0)))
        outer = FunctionNode(primitive=add_info, children=(inner, _const(0.0)))
        result = simplify(outer)
        assert result == x

    def test_nested_constant_fold_then_identity(self) -> None:
        """add(x, add(2, 3)) -> add(x, 5) via constant folding."""
        add_info = _make_add_info()
        x = _close()
        const_sum = FunctionNode(
            primitive=add_info, children=(_const(2.0), _const(3.0))
        )
        tree = FunctionNode(primitive=add_info, children=(x, const_sum))
        result = simplify(tree)
        expected = FunctionNode(primitive=add_info, children=(x, _const(5.0)))
        assert result == expected


# --- Extensibility (FR-7.5) ------------------------------------------------


class TestExtensibility:
    """FR-7.5: Rules registry is extensible."""

    def test_add_custom_rule(self) -> None:
        """Register a custom rule that replaces add(x, x) with mul(x, 2)."""
        mul_info = _make_mul_info()
        add_info = _make_add_info()

        def double_rule(node: object) -> object | None:
            if (
                isinstance(node, FunctionNode)
                and node.primitive.name == "add"
                and len(node.children) == 2
                and node.children[0] == node.children[1]
            ):
                return FunctionNode(
                    primitive=mul_info,
                    children=(node.children[0], _const(2.0)),
                )
            return None

        registry = default_rules()
        registry.add_rule(double_rule)

        x = _close()
        tree = FunctionNode(primitive=add_info, children=(x, x))
        result = simplify(tree, registry=registry)
        expected = FunctionNode(primitive=mul_info, children=(x, _const(2.0)))
        assert result == expected

    def test_default_rules_returns_new_instance(self) -> None:
        """Each call to default_rules() should return an independent registry."""
        r1 = default_rules()
        r2 = default_rules()
        r1.add_rule(lambda n: None)
        assert len(r1.rules) != len(r2.rules)

    def test_registry_rules_property(self) -> None:
        reg = SimplificationRegistry()
        assert reg.rules == []
        reg.add_rule(lambda n: None)
        assert len(reg.rules) == 1


# --- Configurable enable/disable (FR-7.6) ----------------------------------


class TestConfigurableEnableDisable:
    """FR-7.6: Rules can be individually enabled/disabled."""

    def test_disable_rule_by_name(self) -> None:
        """Disabling identity_elimination should stop x + 0 -> x."""
        add_info = _make_add_info()
        x = _close()
        tree = FunctionNode(primitive=add_info, children=(x, _const(0.0)))

        registry = default_rules()
        registry.disable_rule("identity_elimination")
        # With identity elimination disabled, the tree should NOT simplify
        # (constant folding doesn't apply because x is a terminal, not constant).
        result = simplify(tree, registry=registry)
        assert result == tree

    def test_enable_rule_after_disable(self) -> None:
        add_info = _make_add_info()
        x = _close()
        tree = FunctionNode(primitive=add_info, children=(x, _const(0.0)))

        registry = default_rules()
        registry.disable_rule("identity_elimination")
        registry.enable_rule("identity_elimination")
        result = simplify(tree, registry=registry)
        assert result == x

    def test_disable_nonexistent_rule_raises(self) -> None:
        registry = default_rules()
        with pytest.raises(KeyError):
            registry.disable_rule("nonexistent_rule")

    def test_enable_nonexistent_rule_raises(self) -> None:
        registry = default_rules()
        with pytest.raises(KeyError):
            registry.enable_rule("nonexistent_rule")


# --- Edge cases -------------------------------------------------------------


class TestEdgeCases:
    """Edge cases and passthrough behavior."""

    def test_terminal_passthrough(self) -> None:
        x = _close()
        assert simplify(x) == x

    def test_constant_passthrough(self) -> None:
        c = _const(42.0)
        assert simplify(c) == c

    def test_parameterized_node_children_simplified(self) -> None:
        """Children of ParameterizedNode should be simplified too."""
        add_info = _make_add_info()
        highest_info = _make_highest_info()
        x = _close()
        # highest(add(x, 0)) should become highest(x)
        child = FunctionNode(primitive=add_info, children=(x, _const(0.0)))
        tree = ParameterizedNode(
            primitive=highest_info,
            children=(child,),
            params={"period": 20},
        )
        result = simplify(tree)
        expected = ParameterizedNode(
            primitive=highest_info,
            children=(x,),
            params={"period": 20},
        )
        assert result == expected

    def test_simplify_with_empty_registry(self) -> None:
        """An empty registry should leave the tree unchanged."""
        add_info = _make_add_info()
        x = _close()
        tree = FunctionNode(primitive=add_info, children=(x, _const(0.0)))
        empty_reg = SimplificationRegistry()
        result = simplify(tree, registry=empty_reg)
        assert result == tree

    def test_simplify_preserves_output_type(self) -> None:
        """Simplified constant nodes should preserve Series type."""
        add_info = _make_add_info()
        tree = FunctionNode(primitive=add_info, children=(_const(1.0), _const(2.0)))
        result = simplify(tree)
        assert isinstance(result, ConstantNode)
        assert result.output_type == Series

    def test_mul_zero_wins_over_identity(self) -> None:
        """mul(0, 1) should produce 0 (zero annihilation before identity)."""
        mul_info = _make_mul_info()
        tree = FunctionNode(primitive=mul_info, children=(_const(0.0), _const(1.0)))
        result = simplify(tree)
        assert isinstance(result, ConstantNode)
        assert result.value == 0.0

    def test_no_default_registry_required(self) -> None:
        """simplify() with no registry argument should use default rules."""
        add_info = _make_add_info()
        x = _close()
        tree = FunctionNode(primitive=add_info, children=(x, _const(0.0)))
        result = simplify(tree)
        assert result == x

    def test_deeply_nested_simplification(self) -> None:
        """Multiple layers of simplifiable patterns resolve in one pass."""
        add_info = _make_add_info()
        neg_info = _make_neg_info()
        mul_info = _make_mul_info()
        x = _close()
        # mul(neg(neg(x)), add(1, 0)) -> mul(x, 1) -> x
        double_neg = FunctionNode(
            primitive=neg_info,
            children=(FunctionNode(primitive=neg_info, children=(x,)),),
        )
        add_one_zero = FunctionNode(
            primitive=add_info, children=(_const(1.0), _const(0.0))
        )
        tree = FunctionNode(primitive=mul_info, children=(double_neg, add_one_zero))
        result = simplify(tree)
        # double_neg -> x, add_one_zero -> const(1), then mul(x, 1) -> x
        assert result == x
