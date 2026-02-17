"""Golden tests: hand-built ASTs with known expected evaluation results.

Every test in this module uses a manually constructed AST and asserts
exact numerical correctness via ``np.testing.assert_array_almost_equal``.
"""

from __future__ import annotations

import numpy as np

from liq.gp.primitives.registry import PrimitiveInfo
from liq.gp.program.ast import (
    ConstantNode,
    FunctionNode,
    ParameterizedNode,
    Program,
    TerminalNode,
)
from liq.gp.program.eval import evaluate
from liq.gp.program.simplify import simplify
from liq.gp.types import BoolSeries, ParamSpec, Series

# ---------------------------------------------------------------------------
# Helper factories
# ---------------------------------------------------------------------------


def _x() -> TerminalNode:
    return TerminalNode("x", Series)


def _const(v: float) -> ConstantNode:
    return ConstantNode(v, Series)


def _add_info() -> PrimitiveInfo:
    return PrimitiveInfo(
        name="add",
        category="numeric",
        arity=2,
        input_types=(Series, Series),
        output_type=Series,
        callable=lambda a, b: a + b,
    )


def _mul_info() -> PrimitiveInfo:
    return PrimitiveInfo(
        name="mul",
        category="numeric",
        arity=2,
        input_types=(Series, Series),
        output_type=Series,
        callable=lambda a, b: a * b,
    )


def _neg_info() -> PrimitiveInfo:
    return PrimitiveInfo(
        name="neg",
        category="numeric",
        arity=1,
        input_types=(Series,),
        output_type=Series,
        callable=lambda a: -a,
    )


def _sub_info() -> PrimitiveInfo:
    return PrimitiveInfo(
        name="sub",
        category="numeric",
        arity=2,
        input_types=(Series, Series),
        output_type=Series,
        callable=lambda a, b: a - b,
    )


def _scale_info() -> PrimitiveInfo:
    return PrimitiveInfo(
        name="scale",
        category="numeric",
        arity=1,
        input_types=(Series,),
        output_type=Series,
        callable=lambda a, *, factor=1.0: a * factor,
        param_specs=[
            ParamSpec(
                name="factor",
                dtype=float,
                default=1.0,
                min_value=-10.0,
                max_value=10.0,
            )
        ],
    )


def _gt_info() -> PrimitiveInfo:
    return PrimitiveInfo(
        name="gt",
        category="comparison",
        arity=2,
        input_types=(Series, Series),
        output_type=BoolSeries,
        callable=lambda a, b: (a > b).astype(np.float64),
    )


def _if_else_info() -> PrimitiveInfo:
    return PrimitiveInfo(
        name="if_else",
        category="control",
        arity=3,
        input_types=(BoolSeries, Series, Series),
        output_type=Series,
        callable=lambda cond, a, b: np.where(cond > 0.5, a, b),
    )


def _add(a: Program, b: Program) -> FunctionNode:
    return FunctionNode(_add_info(), (a, b))


def _mul(a: Program, b: Program) -> FunctionNode:
    return FunctionNode(_mul_info(), (a, b))


def _neg(a: Program) -> FunctionNode:
    return FunctionNode(_neg_info(), (a,))


def _sub(a: Program, b: Program) -> FunctionNode:
    return FunctionNode(_sub_info(), (a, b))


CTX: dict[str, np.ndarray] = {"x": np.array([1.0, 2.0, 3.0])}


def _scale(a: Program, *, factor: float) -> ParameterizedNode:
    return ParameterizedNode(
        _scale_info(),
        (a,),
        {"factor": factor},
    )


def _gt(a: Program, b: Program) -> FunctionNode:
    return FunctionNode(_gt_info(), (a, b))


def _if_else(cond: Program, when_true: Program, when_false: Program) -> FunctionNode:
    return FunctionNode(_if_else_info(), (cond, when_true, when_false))


# ---------------------------------------------------------------------------
# TestGoldenEvaluation
# ---------------------------------------------------------------------------


class TestGoldenEvaluation:
    """Hand-built AST trees with exact expected evaluation results."""

    # 1. terminal x
    def test_terminal_x(self) -> None:
        result = evaluate(_x(), CTX)
        np.testing.assert_array_almost_equal(result, [1.0, 2.0, 3.0])

    # 2. constant broadcasts to context length
    def test_constant(self) -> None:
        ctx = {"x": np.array([1.0, 2.0])}
        result = evaluate(_const(3.14), ctx)
        np.testing.assert_array_almost_equal(result, [3.14, 3.14])

    # 3. add(x, x) -> 2x
    def test_add_x_x(self) -> None:
        tree = _add(_x(), _x())
        result = evaluate(tree, CTX)
        np.testing.assert_array_almost_equal(result, [2.0, 4.0, 6.0])

    # 4. mul(x, x) -> x^2
    def test_mul_x_x(self) -> None:
        tree = _mul(_x(), _x())
        result = evaluate(tree, CTX)
        np.testing.assert_array_almost_equal(result, [1.0, 4.0, 9.0])

    # 5. neg(x) -> -x
    def test_neg_x(self) -> None:
        tree = _neg(_x())
        result = evaluate(tree, CTX)
        np.testing.assert_array_almost_equal(result, [-1.0, -2.0, -3.0])

    # 6. add(x, 1.0) -> x + 1
    def test_add_x_constant(self) -> None:
        tree = _add(_x(), _const(1.0))
        result = evaluate(tree, CTX)
        np.testing.assert_array_almost_equal(result, [2.0, 3.0, 4.0])

    # 7. add(mul(x, x), x) -> x^2 + x
    def test_nested_add_mul(self) -> None:
        tree = _add(_mul(_x(), _x()), _x())
        result = evaluate(tree, CTX)
        np.testing.assert_array_almost_equal(result, [2.0, 6.0, 12.0])

    # 8. neg(add(mul(x, x), neg(x))) -> -(x^2 - x) = x - x^2
    def test_deep_nesting(self) -> None:
        tree = _neg(_add(_mul(_x(), _x()), _neg(_x())))
        result = evaluate(tree, CTX)
        # x - x^2: [1-1, 2-4, 3-9] = [0, -2, -6]
        np.testing.assert_array_almost_equal(result, [0.0, -2.0, -6.0])

    # 9. sub(x, x) -> 0
    def test_sub_x_x(self) -> None:
        tree = _sub(_x(), _x())
        result = evaluate(tree, CTX)
        np.testing.assert_array_almost_equal(result, [0.0, 0.0, 0.0])

    # 10. add(const(2.0), const(3.0)) -> 5.0 broadcast
    def test_constant_tree(self) -> None:
        tree = _add(_const(2.0), _const(3.0))
        result = evaluate(tree, CTX)
        np.testing.assert_array_almost_equal(result, [5.0, 5.0, 5.0])

    # 11. mul(add(x, 1.0), neg(x)) -> (x+1)*(-x) = -x^2 - x
    def test_complex_expression(self) -> None:
        tree = _mul(_add(_x(), _const(1.0)), _neg(_x()))
        result = evaluate(tree, CTX)
        # (1+1)*(-1), (2+1)*(-2), (3+1)*(-3) = -2, -6, -12
        np.testing.assert_array_almost_equal(result, [-2.0, -6.0, -12.0])

    # 12. ConstantNode(0.0) -> all zeros
    def test_zero_constant(self) -> None:
        result = evaluate(_const(0.0), CTX)
        np.testing.assert_array_almost_equal(result, [0.0, 0.0, 0.0])

    # 13. add(x, x) with negative inputs
    def test_negative_inputs(self) -> None:
        ctx = {"x": np.array([-3.0, -2.0, -1.0])}
        tree = _add(_x(), _x())
        result = evaluate(tree, ctx)
        np.testing.assert_array_almost_equal(result, [-6.0, -4.0, -2.0])

    # 14. mul(x, x) with large values
    def test_large_values(self) -> None:
        ctx = {"x": np.array([1000.0, 2000.0])}
        tree = _mul(_x(), _x())
        result = evaluate(tree, ctx)
        np.testing.assert_array_almost_equal(result, [1e6, 4e6])

    # 15. parameterized node: scale(x, factor=2.5)
    def test_parameterized_scale(self) -> None:
        tree = _scale(_x(), factor=2.5)
        result = evaluate(tree, CTX)
        np.testing.assert_array_almost_equal(result, [2.5, 5.0, 7.5])

    # 16. mixed types: gt(x, 2.0) -> BoolSeries (1.0/0.0)
    def test_mixed_type_bool_series_output(self) -> None:
        tree = _gt(_x(), _const(2.0))
        result = evaluate(tree, CTX)
        np.testing.assert_array_almost_equal(result, [0.0, 0.0, 1.0])

    # 17. mixed types: if_else(gt(x,2), x, -x)
    def test_mixed_type_if_else(self) -> None:
        cond = _gt(_x(), _const(2.0))
        tree = _if_else(cond, _x(), _neg(_x()))
        result = evaluate(tree, CTX)
        np.testing.assert_array_almost_equal(result, [-1.0, -2.0, 3.0])


# ---------------------------------------------------------------------------
# TestGoldenTreeProperties
# ---------------------------------------------------------------------------


class TestGoldenTreeProperties:
    """Structural properties of hand-built trees.

    Note: In liq.gp the depth of leaf nodes (TerminalNode, ConstantNode) is 0,
    and FunctionNode.depth = 1 + max(children.depth).
    """

    # 1. TerminalNode has depth=0, size=1
    def test_terminal_depth_and_size(self) -> None:
        node = _x()
        assert node.depth == 0
        assert node.size == 1

    # 2. ConstantNode has depth=0, size=1
    def test_constant_depth_and_size(self) -> None:
        node = _const(42.0)
        assert node.depth == 0
        assert node.size == 1

    # 3. add(x, x) has depth=1, size=3
    def test_function_depth(self) -> None:
        tree = _add(_x(), _x())
        assert tree.depth == 1
        assert tree.size == 3

    # 4. add(mul(x, x), x) has depth=2, size=5
    def test_nested_depth(self) -> None:
        tree = _add(_mul(_x(), _x()), _x())
        assert tree.depth == 2
        assert tree.size == 5

    # 5. Four levels deep: neg(add(mul(x, x), neg(x)))
    #    neg -> add -> mul -> x  (depth 3)
    #    neg -> add -> neg -> x  (depth 3)
    #    max chain is 3
    def test_deep_tree_depth(self) -> None:
        tree = _neg(_add(_mul(_x(), _x()), _neg(_x())))
        # neg(depth=1+add.depth)
        # add(depth=1+max(mul.depth, neg.depth))
        # mul(depth=1+max(x.depth, x.depth)) = 1+0 = 1
        # neg(depth=1+x.depth) = 1+0 = 1
        # add.depth = 1 + max(1, 1) = 2
        # outer neg.depth = 1 + 2 = 3
        assert tree.depth == 3
        # size: neg(1) + add(1) + mul(1) + x(1) + x(1) + neg(1) + x(1) = 7
        assert tree.size == 7


# ---------------------------------------------------------------------------
# TestGoldenSimplification
# ---------------------------------------------------------------------------


class TestGoldenSimplification:
    """Deterministic algebraic simplification of known expressions."""

    # 1. add(x, 0.0) simplifies to x
    def test_add_zero(self) -> None:
        tree = _add(_x(), _const(0.0))
        result = simplify(tree)
        assert isinstance(result, TerminalNode)
        assert result.name == "x"

    # 2. mul(x, 1.0) simplifies to x
    def test_mul_one(self) -> None:
        tree = _mul(_x(), _const(1.0))
        result = simplify(tree)
        assert isinstance(result, TerminalNode)
        assert result.name == "x"

    # 3. mul(x, 0.0) simplifies to ConstantNode(0.0)
    def test_mul_zero(self) -> None:
        tree = _mul(_x(), _const(0.0))
        result = simplify(tree)
        assert isinstance(result, ConstantNode)
        assert result.value == 0.0

    # 4. neg(neg(x)) simplifies to x
    def test_neg_neg(self) -> None:
        tree = _neg(_neg(_x()))
        result = simplify(tree)
        assert isinstance(result, TerminalNode)
        assert result.name == "x"

    # 5. Simplification preserves semantics for each of the above cases
    def test_simplification_preserves_semantics_add_zero(self) -> None:
        tree = _add(_x(), _const(0.0))
        before = evaluate(tree, CTX)
        after = evaluate(simplify(tree), CTX)
        np.testing.assert_array_almost_equal(before, after)

    def test_simplification_preserves_semantics_mul_one(self) -> None:
        tree = _mul(_x(), _const(1.0))
        before = evaluate(tree, CTX)
        after = evaluate(simplify(tree), CTX)
        np.testing.assert_array_almost_equal(before, after)

    def test_simplification_preserves_semantics_mul_zero(self) -> None:
        tree = _mul(_x(), _const(0.0))
        before = evaluate(tree, CTX)
        after = evaluate(simplify(tree), CTX)
        np.testing.assert_array_almost_equal(before, after)

    def test_simplification_preserves_semantics_neg_neg(self) -> None:
        tree = _neg(_neg(_x()))
        before = evaluate(tree, CTX)
        after = evaluate(simplify(tree), CTX)
        np.testing.assert_array_almost_equal(before, after)
