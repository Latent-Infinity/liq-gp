"""Tests for the evaluation engine (FR-4)."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from liq.gp.primitives.registry import PrimitiveInfo
from liq.gp.program.ast import (
    ConstantNode,
    FunctionNode,
    ParameterizedNode,
    TerminalNode,
)
from liq.gp.program.eval import evaluate
from liq.gp.types import ParamSpec, Series

# --- helpers ---------------------------------------------------------------


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


def _highest_info() -> PrimitiveInfo:
    ps = ParamSpec(name="period", dtype=int, default=20, min_value=2, max_value=200)
    return PrimitiveInfo(
        name="highest",
        category="indicator",
        arity=1,
        input_types=(Series,),
        output_type=Series,
        callable=lambda a, *, period: np.maximum.accumulate(a[-period:]),
        param_specs=[ps],
    )


def _shift_info() -> PrimitiveInfo:
    ps = ParamSpec(name="n", dtype=int, default=1, min_value=1, max_value=50)
    return PrimitiveInfo(
        name="shift",
        category="transform",
        arity=1,
        input_types=(Series,),
        output_type=Series,
        callable=lambda a, *, n: np.roll(a, n),
        param_specs=[ps],
    )


def _make_context(n: int = 5) -> dict[str, np.ndarray]:
    return {
        "close": np.array([1.0, 2.0, 3.0, 4.0, 5.0][:n], dtype=np.float64),
        "volume": np.array([100.0, 200.0, 300.0, 400.0, 500.0][:n], dtype=np.float64),
    }


# --- TerminalNode evaluation -----------------------------------------------


class TestTerminalEvaluation:
    """TerminalNode reads from context by name."""

    def test_reads_close(self) -> None:
        ctx = _make_context()
        node = TerminalNode(name="close", output_type=Series)
        result = evaluate(node, ctx)
        np.testing.assert_array_equal(result, ctx["close"])

    def test_reads_volume(self) -> None:
        ctx = _make_context()
        node = TerminalNode(name="volume", output_type=Series)
        result = evaluate(node, ctx)
        np.testing.assert_array_equal(result, ctx["volume"])

    def test_unknown_terminal_raises(self) -> None:
        ctx = _make_context()
        node = TerminalNode(name="missing", output_type=Series)
        with pytest.raises(KeyError):
            evaluate(node, ctx)


# --- ConstantNode evaluation ------------------------------------------------


class TestConstantEvaluation:
    """ConstantNode broadcasts scalar to array."""

    def test_broadcasts_to_array(self) -> None:
        ctx = _make_context()
        node = ConstantNode(value=3.14)
        result = evaluate(node, ctx)
        expected = np.full(5, 3.14, dtype=np.float64)
        np.testing.assert_array_equal(result, expected)

    def test_output_dtype_float64(self) -> None:
        ctx = _make_context()
        node = ConstantNode(value=1.0)
        result = evaluate(node, ctx)
        assert result.dtype == np.float64

    def test_output_length_matches_context(self) -> None:
        ctx = _make_context(3)
        node = ConstantNode(value=2.0)
        result = evaluate(node, ctx)
        assert len(result) == 3


# --- FunctionNode evaluation ------------------------------------------------


class TestFunctionEvaluation:
    """FunctionNode applies callable to children outputs."""

    def test_add(self) -> None:
        ctx = _make_context()
        close = TerminalNode(name="close", output_type=Series)
        volume = TerminalNode(name="volume", output_type=Series)
        node = FunctionNode(primitive=_add_info(), children=(close, volume))
        result = evaluate(node, ctx)
        expected = ctx["close"] + ctx["volume"]
        np.testing.assert_array_equal(result, expected)

    def test_mul(self) -> None:
        ctx = _make_context()
        close = TerminalNode(name="close", output_type=Series)
        const = ConstantNode(value=2.0)
        node = FunctionNode(primitive=_mul_info(), children=(close, const))
        result = evaluate(node, ctx)
        expected = ctx["close"] * 2.0
        np.testing.assert_array_equal(result, expected)

    def test_nested(self) -> None:
        """neg(add(close, volume))"""
        ctx = _make_context()
        close = TerminalNode(name="close", output_type=Series)
        volume = TerminalNode(name="volume", output_type=Series)
        add_node = FunctionNode(primitive=_add_info(), children=(close, volume))
        neg_node = FunctionNode(primitive=_neg_info(), children=(add_node,))
        result = evaluate(neg_node, ctx)
        expected = -(ctx["close"] + ctx["volume"])
        np.testing.assert_array_equal(result, expected)


# --- ParameterizedNode evaluation -------------------------------------------


class TestParameterizedEvaluation:
    """ParameterizedNode passes params to callable."""

    def test_shift(self) -> None:
        ctx = _make_context()
        close = TerminalNode(name="close", output_type=Series)
        info = _shift_info()
        node = ParameterizedNode(
            primitive=info,
            children=(close,),
            params={"n": 1},
        )
        result = evaluate(node, ctx)
        expected = np.roll(ctx["close"], 1)
        np.testing.assert_array_equal(result, expected)


# --- Golden tests (FR-4.3) -------------------------------------------------


class TestGoldenTests:
    """Hand-built ASTs produce expected output vectors."""

    def test_linear_combination(self) -> None:
        """2 * close + 1.0"""
        ctx = _make_context()
        close = TerminalNode(name="close", output_type=Series)
        two = ConstantNode(value=2.0)
        one = ConstantNode(value=1.0)
        scaled = FunctionNode(primitive=_mul_info(), children=(two, close))
        result_node = FunctionNode(primitive=_add_info(), children=(scaled, one))
        result = evaluate(result_node, ctx)
        expected = 2.0 * ctx["close"] + 1.0
        np.testing.assert_array_almost_equal(result, expected)


# --- Output properties (FR-4.3) --------------------------------------------


class TestOutputProperties:
    """Output dtype and length."""

    def test_output_dtype_float64(self) -> None:
        ctx = _make_context()
        close = TerminalNode(name="close", output_type=Series)
        result = evaluate(close, ctx)
        assert result.dtype == np.float64

    def test_output_length_equals_input(self) -> None:
        ctx = _make_context(3)
        close = TerminalNode(name="close", output_type=Series)
        volume = TerminalNode(name="volume", output_type=Series)
        node = FunctionNode(primitive=_add_info(), children=(close, volume))
        result = evaluate(node, ctx)
        assert len(result) == 3


# --- Determinism (FR-4.4) --------------------------------------------------


class TestDeterminism:
    """Same program + context = identical output."""

    def test_deterministic(self) -> None:
        ctx = _make_context()
        close = TerminalNode(name="close", output_type=Series)
        const = ConstantNode(value=2.0)
        node = FunctionNode(primitive=_mul_info(), children=(close, const))
        r1 = evaluate(node, ctx)
        r2 = evaluate(node, ctx)
        np.testing.assert_array_equal(r1, r2)


# --- Immutability (FR-4.5) -------------------------------------------------


class TestImmutability:
    """Context and input arrays not mutated."""

    def test_context_not_mutated(self) -> None:
        ctx = _make_context()
        close_before = ctx["close"].copy()
        volume_before = ctx["volume"].copy()
        close = TerminalNode(name="close", output_type=Series)
        volume = TerminalNode(name="volume", output_type=Series)
        node = FunctionNode(primitive=_add_info(), children=(close, volume))
        evaluate(node, ctx)
        np.testing.assert_array_equal(ctx["close"], close_before)
        np.testing.assert_array_equal(ctx["volume"], volume_before)


# --- NaN propagation (FR-4.6) ----------------------------------------------


class TestNaNPropagation:
    """NaN inputs produce NaN outputs."""

    def test_scalar_return_coerced_to_array(self) -> None:
        """Primitive returning a scalar gets broadcast to an array."""
        scalar_info = PrimitiveInfo(
            name="sum_reduce",
            category="numeric",
            arity=1,
            input_types=(Series,),
            output_type=Series,
            callable=lambda a: float(np.sum(a)),  # returns scalar
        )
        ctx = _make_context()
        close = TerminalNode(name="close", output_type=Series)
        node = FunctionNode(primitive=scalar_info, children=(close,))
        result = evaluate(node, ctx)
        assert isinstance(result, np.ndarray)
        assert len(result) == 5
        assert result.dtype == np.float64

    def test_non_float64_coerced(self) -> None:
        """Primitive returning int32 array gets coerced to float64."""
        int_info = PrimitiveInfo(
            name="as_int",
            category="numeric",
            arity=1,
            input_types=(Series,),
            output_type=Series,
            callable=lambda a: a.astype(np.int32),  # returns int32
        )
        ctx = _make_context()
        close = TerminalNode(name="close", output_type=Series)
        node = FunctionNode(primitive=int_info, children=(close,))
        result = evaluate(node, ctx)
        assert result.dtype == np.float64

    def test_nan_in_terminal(self) -> None:
        ctx = {"close": np.array([1.0, np.nan, 3.0], dtype=np.float64)}
        close = TerminalNode(name="close", output_type=Series)
        const = ConstantNode(value=2.0)
        node = FunctionNode(primitive=_add_info(), children=(close, const))
        result = evaluate(node, ctx)
        assert np.isnan(result[1])
        assert not np.isnan(result[0])
        assert not np.isnan(result[2])


# --- Vectorized evaluation (NFR-1.2) ---------------------------------------


class TestVectorizedEvaluation:
    """Primitive callables operate on full arrays, not per-observation loops."""

    def test_add_vectorized(self) -> None:
        """Verify add operates on arrays, not element-by-element."""
        call_count = 0
        original_add = np.add

        def counting_add(a: np.ndarray, b: np.ndarray) -> np.ndarray:
            nonlocal call_count
            call_count += 1
            return original_add(a, b)

        info = PrimitiveInfo(
            name="add_counting",
            category="numeric",
            arity=2,
            input_types=(Series, Series),
            output_type=Series,
            callable=counting_add,
        )
        ctx = _make_context()
        close = TerminalNode(name="close", output_type=Series)
        volume = TerminalNode(name="volume", output_type=Series)
        node = FunctionNode(primitive=info, children=(close, volume))
        result = evaluate(node, ctx)
        # callable should be called exactly once (vectorized), not once per element
        assert call_count == 1
        np.testing.assert_array_equal(result, ctx["close"] + ctx["volume"])


# --- Evaluation cache (FR-4.7) ---------------------------------------------


class TestEvaluationCache:
    """Optional cache for shared subtrees."""

    def test_cache_avoids_redundant_computation(self) -> None:
        """Same subtree object used twice should only be evaluated once."""
        eval_count = 0

        def counting_neg(a: np.ndarray) -> np.ndarray:
            nonlocal eval_count
            eval_count += 1
            return -a

        neg_info = PrimitiveInfo(
            name="counting_neg",
            category="numeric",
            arity=1,
            input_types=(Series,),
            output_type=Series,
            callable=counting_neg,
        )
        ctx = _make_context()
        close = TerminalNode(name="close", output_type=Series)
        neg_close = FunctionNode(primitive=neg_info, children=(close,))
        # Use same subtree twice: add(neg(close), neg(close))
        node = FunctionNode(primitive=_add_info(), children=(neg_close, neg_close))
        result = evaluate(node, ctx, use_cache=True)
        # With caching, counting_neg should be called only once
        assert eval_count == 1
        expected = -ctx["close"] + -ctx["close"]
        np.testing.assert_array_equal(result, expected)

    def test_cache_uses_structural_hash(self) -> None:
        """Structurally identical but distinct objects share cache entries."""
        eval_count = 0

        def counting_neg(a: np.ndarray) -> np.ndarray:
            nonlocal eval_count
            eval_count += 1
            return -a

        neg_info = PrimitiveInfo(
            name="counting_neg",
            category="numeric",
            arity=1,
            input_types=(Series,),
            output_type=Series,
            callable=counting_neg,
        )
        ctx = _make_context()
        close_a = TerminalNode(name="close", output_type=Series)
        close_b = TerminalNode(name="close", output_type=Series)
        assert close_a is not close_b  # different objects
        assert close_a == close_b  # structurally equal
        neg_a = FunctionNode(primitive=neg_info, children=(close_a,))
        neg_b = FunctionNode(primitive=neg_info, children=(close_b,))
        assert neg_a is not neg_b  # different objects
        # add(neg(close), neg(close)) with distinct subtree objects
        node = FunctionNode(primitive=_add_info(), children=(neg_a, neg_b))
        result = evaluate(node, ctx, use_cache=True)
        # Structural hash matches, so counting_neg called only once
        assert eval_count == 1
        expected = -ctx["close"] + -ctx["close"]
        np.testing.assert_array_equal(result, expected)

    def test_cache_discriminates_by_context(self) -> None:
        """Different context dicts produce separate cache entries."""
        ctx1 = {"close": np.array([1.0, 2.0, 3.0], dtype=np.float64)}
        ctx2 = {"close": np.array([10.0, 20.0, 30.0], dtype=np.float64)}
        close = TerminalNode(name="close", output_type=Series)
        const = ConstantNode(value=1.0)
        node = FunctionNode(primitive=_add_info(), children=(close, const))
        r1 = evaluate(node, ctx1, use_cache=True)
        r2 = evaluate(node, ctx2, use_cache=True)
        # Results must differ because contexts differ
        assert not np.array_equal(r1, r2)
        np.testing.assert_array_equal(r1, ctx1["close"] + 1.0)
        np.testing.assert_array_equal(r2, ctx2["close"] + 1.0)

    def test_cache_disabled_by_default(self) -> None:
        """Without use_cache=True, no caching occurs."""
        eval_count = 0

        def counting_neg(a: np.ndarray) -> np.ndarray:
            nonlocal eval_count
            eval_count += 1
            return -a

        neg_info = PrimitiveInfo(
            name="counting_neg",
            category="numeric",
            arity=1,
            input_types=(Series,),
            output_type=Series,
            callable=counting_neg,
        )
        ctx = _make_context()
        close = TerminalNode(name="close", output_type=Series)
        neg_close = FunctionNode(primitive=neg_info, children=(close,))
        node = FunctionNode(primitive=_add_info(), children=(neg_close, neg_close))
        result = evaluate(node, ctx)  # use_cache defaults to False
        # Without caching, counting_neg called twice (once per child)
        assert eval_count == 2
        expected = -ctx["close"] + -ctx["close"]
        np.testing.assert_array_equal(result, expected)


# --- Property tests (hypothesis) -------------------------------------------


class TestPropertyBased:
    """Property-based tests for evaluation invariants."""

    @given(
        data=arrays(
            dtype=np.float64,
            shape=st.integers(min_value=1, max_value=100),
            elements=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False),
        )
    )
    @settings(max_examples=50)
    def test_output_length_equals_input(self, data: np.ndarray) -> None:
        ctx = {"x": data}
        node = TerminalNode(name="x", output_type=Series)
        result = evaluate(node, ctx)
        assert len(result) == len(data)

    @given(
        data=arrays(
            dtype=np.float64,
            shape=st.integers(min_value=1, max_value=100),
            elements=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False),
        )
    )
    @settings(max_examples=50)
    def test_determinism(self, data: np.ndarray) -> None:
        ctx = {"x": data}
        node = TerminalNode(name="x", output_type=Series)
        const = ConstantNode(value=2.0)
        tree = FunctionNode(primitive=_mul_info(), children=(node, const))
        r1 = evaluate(tree, ctx)
        r2 = evaluate(tree, ctx)
        np.testing.assert_array_equal(r1, r2)

    @given(
        data=arrays(
            dtype=np.float64,
            shape=st.integers(min_value=1, max_value=100),
            elements=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False),
        )
    )
    @settings(max_examples=50)
    def test_context_immutability(self, data: np.ndarray) -> None:
        original = data.copy()
        ctx = {"x": data}
        node = TerminalNode(name="x", output_type=Series)
        const = ConstantNode(value=3.0)
        tree = FunctionNode(primitive=_add_info(), children=(node, const))
        evaluate(tree, ctx)
        np.testing.assert_array_equal(ctx["x"], original)
