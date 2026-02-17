"""Tests for GP AST nodes (FR-2)."""

from __future__ import annotations

import pytest

from liq.gp.errors import TypeCheckError
from liq.gp.primitives.registry import PrimitiveInfo
from liq.gp.program.ast import (
    ConstantNode,
    FunctionNode,
    ParameterizedNode,
    Program,
    TerminalNode,
)
from liq.gp.types import BoolSeries, Int, ParamSpec, Series

# --- helpers ---------------------------------------------------------------


def _make_add_info() -> PrimitiveInfo:
    return PrimitiveInfo(
        name="add",
        category="numeric",
        arity=2,
        input_types=(Series, Series),
        output_type=Series,
        callable=lambda a, b: a + b,
    )


def _make_neg_info() -> PrimitiveInfo:
    return PrimitiveInfo(
        name="neg",
        category="numeric",
        arity=1,
        input_types=(Series,),
        output_type=Series,
        callable=lambda a: -a,
    )


def _make_gt_info() -> PrimitiveInfo:
    return PrimitiveInfo(
        name="gt",
        category="comparison",
        arity=2,
        input_types=(Series, Series),
        output_type=BoolSeries,
        callable=lambda a, b: (a > b).astype(float),
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


# --- TerminalNode -----------------------------------------------------------


class TestTerminalNode:
    """TerminalNode wraps a zero-arity primitive (FR-2.2)."""

    def test_creation(self) -> None:
        node = TerminalNode(name="close", output_type=Series)
        assert node.name == "close"
        assert node.output_type is Series

    def test_depth_is_zero(self) -> None:
        node = TerminalNode(name="close", output_type=Series)
        assert node.depth == 0

    def test_size_is_one(self) -> None:
        node = TerminalNode(name="close", output_type=Series)
        assert node.size == 1

    def test_constants_empty(self) -> None:
        node = TerminalNode(name="close", output_type=Series)
        assert node.constants == []

    def test_parameters_empty(self) -> None:
        node = TerminalNode(name="close", output_type=Series)
        assert node.parameters == {}

    def test_frozen(self) -> None:
        node = TerminalNode(name="close", output_type=Series)
        with pytest.raises(AttributeError):
            node.name = "volume"  # type: ignore[misc]


# --- ConstantNode -----------------------------------------------------------


class TestConstantNode:
    """ConstantNode stores a literal float value (FR-2.2)."""

    def test_creation(self) -> None:
        node = ConstantNode(value=3.14)
        assert node.value == 3.14
        assert node.output_type is Series  # constants are Series by default

    def test_creation_with_type(self) -> None:
        node = ConstantNode(value=5.0, output_type=Int)
        assert node.output_type is Int

    def test_depth_is_zero(self) -> None:
        node = ConstantNode(value=1.0)
        assert node.depth == 0

    def test_size_is_one(self) -> None:
        node = ConstantNode(value=1.0)
        assert node.size == 1

    def test_constants_returns_self(self) -> None:
        node = ConstantNode(value=2.5)
        assert node.constants == [node]

    def test_parameters_empty(self) -> None:
        node = ConstantNode(value=1.0)
        assert node.parameters == {}

    def test_frozen(self) -> None:
        node = ConstantNode(value=1.0)
        with pytest.raises(AttributeError):
            node.value = 2.0  # type: ignore[misc]


# --- FunctionNode -----------------------------------------------------------


class TestFunctionNode:
    """FunctionNode wraps a function primitive with typed children (FR-2.2)."""

    def test_creation(self) -> None:
        close = TerminalNode(name="close", output_type=Series)
        volume = TerminalNode(name="volume", output_type=Series)
        add_info = _make_add_info()
        node = FunctionNode(primitive=add_info, children=(close, volume))
        assert node.primitive is add_info
        assert node.output_type is Series
        assert len(node.children) == 2

    def test_depth(self) -> None:
        close = TerminalNode(name="close", output_type=Series)
        volume = TerminalNode(name="volume", output_type=Series)
        add_info = _make_add_info()
        node = FunctionNode(primitive=add_info, children=(close, volume))
        # depth of function node = 1 + max child depth
        assert node.depth == 1

    def test_nested_depth(self) -> None:
        close = TerminalNode(name="close", output_type=Series)
        const = ConstantNode(value=1.0)
        add_info = _make_add_info()
        neg_info = _make_neg_info()
        inner = FunctionNode(primitive=add_info, children=(close, const))
        outer = FunctionNode(primitive=neg_info, children=(inner,))
        assert outer.depth == 2

    def test_size(self) -> None:
        close = TerminalNode(name="close", output_type=Series)
        volume = TerminalNode(name="volume", output_type=Series)
        add_info = _make_add_info()
        node = FunctionNode(primitive=add_info, children=(close, volume))
        assert node.size == 3  # add + close + volume

    def test_constants_from_children(self) -> None:
        close = TerminalNode(name="close", output_type=Series)
        const = ConstantNode(value=2.0)
        add_info = _make_add_info()
        node = FunctionNode(primitive=add_info, children=(close, const))
        assert len(node.constants) == 1
        assert node.constants[0] is const

    def test_parameters_empty(self) -> None:
        close = TerminalNode(name="close", output_type=Series)
        volume = TerminalNode(name="volume", output_type=Series)
        add_info = _make_add_info()
        node = FunctionNode(primitive=add_info, children=(close, volume))
        assert node.parameters == {}

    def test_frozen(self) -> None:
        close = TerminalNode(name="close", output_type=Series)
        volume = TerminalNode(name="volume", output_type=Series)
        add_info = _make_add_info()
        node = FunctionNode(primitive=add_info, children=(close, volume))
        with pytest.raises(AttributeError):
            node.children = ()  # type: ignore[misc]


# --- ParameterizedNode ------------------------------------------------------


class TestParameterizedNode:
    """ParameterizedNode: function with children AND evolvable params (FR-2.2)."""

    def test_creation(self) -> None:
        close = TerminalNode(name="close", output_type=Series)
        info = _make_highest_info()
        node = ParameterizedNode(
            primitive=info,
            children=(close,),
            params={"period": 20},
        )
        assert node.output_type is Series
        assert node.params == {"period": 20}

    def test_depth(self) -> None:
        close = TerminalNode(name="close", output_type=Series)
        info = _make_highest_info()
        node = ParameterizedNode(
            primitive=info,
            children=(close,),
            params={"period": 20},
        )
        assert node.depth == 1

    def test_size(self) -> None:
        close = TerminalNode(name="close", output_type=Series)
        info = _make_highest_info()
        node = ParameterizedNode(
            primitive=info,
            children=(close,),
            params={"period": 20},
        )
        assert node.size == 2  # highest + close

    def test_parameters(self) -> None:
        close = TerminalNode(name="close", output_type=Series)
        info = _make_highest_info()
        node = ParameterizedNode(
            primitive=info,
            children=(close,),
            params={"period": 30},
        )
        assert node.parameters == {"period": 30}

    def test_constants_from_children(self) -> None:
        const = ConstantNode(value=1.0)
        info = _make_highest_info()
        node = ParameterizedNode(
            primitive=info,
            children=(const,),
            params={"period": 20},
        )
        assert len(node.constants) == 1

    def test_frozen(self) -> None:
        close = TerminalNode(name="close", output_type=Series)
        info = _make_highest_info()
        node = ParameterizedNode(
            primitive=info,
            children=(close,),
            params={"period": 20},
        )
        with pytest.raises(AttributeError):
            node.params = {}  # type: ignore[misc]


# --- Type checking at construction (FR-1.4) ---------------------------------


class TestTypeChecking:
    """TypeCheckError on mismatched child types."""

    def test_wrong_child_type_raises(self) -> None:
        """add expects (Series, Series) but we give (Series, BoolSeries)."""
        close = TerminalNode(name="close", output_type=Series)
        flag = TerminalNode(name="flag", output_type=BoolSeries)
        add_info = _make_add_info()
        with pytest.raises(TypeCheckError, match="type mismatch"):
            FunctionNode(primitive=add_info, children=(close, flag))

    def test_wrong_arity_raises(self) -> None:
        """add expects 2 children but we give 1."""
        close = TerminalNode(name="close", output_type=Series)
        add_info = _make_add_info()
        with pytest.raises(TypeCheckError, match="arity"):
            FunctionNode(primitive=add_info, children=(close,))

    def test_correct_types_pass(self) -> None:
        close = TerminalNode(name="close", output_type=Series)
        volume = TerminalNode(name="volume", output_type=Series)
        add_info = _make_add_info()
        node = FunctionNode(primitive=add_info, children=(close, volume))
        assert node.output_type is Series

    def test_parameterized_type_check(self) -> None:
        flag = TerminalNode(name="flag", output_type=BoolSeries)
        info = _make_highest_info()  # expects (Series,)
        with pytest.raises(TypeCheckError, match="type mismatch"):
            ParameterizedNode(
                primitive=info,
                children=(flag,),
                params={"period": 20},
            )


# --- Structural equality and hashing (FR-2.5) ------------------------------


class TestEqualityAndHashing:
    """Structural equality comparison and hashing."""

    def test_terminal_equality(self) -> None:
        a = TerminalNode(name="close", output_type=Series)
        b = TerminalNode(name="close", output_type=Series)
        assert a == b

    def test_terminal_inequality(self) -> None:
        a = TerminalNode(name="close", output_type=Series)
        b = TerminalNode(name="volume", output_type=Series)
        assert a != b

    def test_constant_equality(self) -> None:
        a = ConstantNode(value=3.14)
        b = ConstantNode(value=3.14)
        assert a == b

    def test_constant_inequality(self) -> None:
        a = ConstantNode(value=1.0)
        b = ConstantNode(value=2.0)
        assert a != b

    def test_function_equality(self) -> None:
        close = TerminalNode(name="close", output_type=Series)
        volume = TerminalNode(name="volume", output_type=Series)
        add_info = _make_add_info()
        a = FunctionNode(primitive=add_info, children=(close, volume))
        b = FunctionNode(primitive=add_info, children=(close, volume))
        assert a == b

    def test_function_inequality_different_children(self) -> None:
        close = TerminalNode(name="close", output_type=Series)
        volume = TerminalNode(name="volume", output_type=Series)
        const = ConstantNode(value=1.0)
        add_info = _make_add_info()
        a = FunctionNode(primitive=add_info, children=(close, volume))
        b = FunctionNode(primitive=add_info, children=(close, const))
        assert a != b

    def test_parameterized_equality(self) -> None:
        close = TerminalNode(name="close", output_type=Series)
        info = _make_highest_info()
        a = ParameterizedNode(primitive=info, children=(close,), params={"period": 20})
        b = ParameterizedNode(primitive=info, children=(close,), params={"period": 20})
        assert a == b

    def test_parameterized_inequality_different_params(self) -> None:
        close = TerminalNode(name="close", output_type=Series)
        info = _make_highest_info()
        a = ParameterizedNode(primitive=info, children=(close,), params={"period": 20})
        b = ParameterizedNode(primitive=info, children=(close,), params={"period": 30})
        assert a != b

    def test_hashable_in_set(self) -> None:
        close = TerminalNode(name="close", output_type=Series)
        volume = TerminalNode(name="volume", output_type=Series)
        const = ConstantNode(value=1.0)
        nodes = {close, volume, const}
        assert len(nodes) == 3

    def test_equal_nodes_same_hash(self) -> None:
        a = TerminalNode(name="close", output_type=Series)
        b = TerminalNode(name="close", output_type=Series)
        assert hash(a) == hash(b)

    def test_function_node_hashable(self) -> None:
        close = TerminalNode(name="close", output_type=Series)
        volume = TerminalNode(name="volume", output_type=Series)
        add_info = _make_add_info()
        a = FunctionNode(primitive=add_info, children=(close, volume))
        b = FunctionNode(primitive=add_info, children=(close, volume))
        assert hash(a) == hash(b)
        assert len({a, b}) == 1

    def test_parameterized_node_hashable(self) -> None:
        close = TerminalNode(name="close", output_type=Series)
        info = _make_highest_info()
        a = ParameterizedNode(primitive=info, children=(close,), params={"period": 20})
        b = ParameterizedNode(primitive=info, children=(close,), params={"period": 20})
        assert hash(a) == hash(b)
        assert len({a, b}) == 1

    def test_not_equal_to_non_node(self) -> None:
        a = TerminalNode(name="close", output_type=Series)
        assert a != "close"

    def test_function_not_equal_to_non_function(self) -> None:
        close = TerminalNode(name="close", output_type=Series)
        volume = TerminalNode(name="volume", output_type=Series)
        add_info = _make_add_info()
        node = FunctionNode(primitive=add_info, children=(close, volume))
        assert node != close

    def test_parameterized_not_equal_to_non_parameterized(self) -> None:
        close = TerminalNode(name="close", output_type=Series)
        info = _make_highest_info()
        node = ParameterizedNode(
            primitive=info, children=(close,), params={"period": 20}
        )
        assert node != close

    def test_constant_not_equal_to_non_constant(self) -> None:
        const = ConstantNode(value=1.0)
        term = TerminalNode(name="close", output_type=Series)
        assert const != term


# --- Program union type ----------------------------------------------------


class TestProgramType:
    """Program is a union of all node kinds."""

    def test_terminal_is_program(self) -> None:
        node = TerminalNode(name="close", output_type=Series)
        assert isinstance(node, TerminalNode)
        # Program is a type alias, so isinstance doesn't work directly,
        # but we verify the union includes all types
        from typing import get_args

        args = get_args(Program)
        assert TerminalNode in args
        assert ConstantNode in args
        assert FunctionNode in args
        assert ParameterizedNode in args
