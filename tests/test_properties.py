"""Property-based tests for liq-gp invariants using Hypothesis.

Tests GP operator invariants (generation, crossover, mutation, evaluation,
simplification, serialization) under randomly generated inputs.
"""

from __future__ import annotations

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from liq.gp.evolution.init import generate_full, generate_grow
from liq.gp.evolution.operators import (
    point_mutation,
    subtree_crossover,
    subtree_mutation,
)
from liq.gp.primitives.registry import PrimitiveRegistry
from liq.gp.program.ast import (
    ConstantNode,
    FunctionNode,
    ParameterizedNode,
    Program,
    TerminalNode,
)
from liq.gp.program.eval import evaluate
from liq.gp.program.serialize import deserialize, serialize
from liq.gp.program.simplify import simplify
from liq.gp.types import Series

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_registry() -> PrimitiveRegistry:
    """Build a minimal registry with terminals and numeric functions."""
    reg = PrimitiveRegistry()
    reg.register("x", lambda: None, input_types=(), output_type=Series)
    reg.register(
        "add",
        lambda a, b: a + b,
        category="numeric",
        input_types=(Series, Series),
        output_type=Series,
    )
    reg.register(
        "mul",
        lambda a, b: a * b,
        category="numeric",
        input_types=(Series, Series),
        output_type=Series,
    )
    reg.register(
        "neg",
        lambda a: -a,
        category="numeric",
        input_types=(Series,),
        output_type=Series,
    )
    return reg


def _random_tree(max_depth: int, seed: int) -> Program:
    """Generate a random tree using grow initialisation."""
    reg = _make_registry()
    rng = np.random.default_rng(seed)
    return generate_grow(reg, max_depth, Series, rng)


def _make_context(length: int = 10) -> dict[str, np.ndarray]:
    """Build a simple evaluation context with known data."""
    rng = np.random.default_rng(0)
    return {"x": rng.standard_normal(length)}


# ---------------------------------------------------------------------------
# 1. Tree generation invariants
# ---------------------------------------------------------------------------


class TestTreeGenerationInvariants:
    """Property tests for generate_grow and generate_full."""

    @settings(max_examples=50)
    @given(
        max_depth=st.integers(min_value=1, max_value=6),
        seed=st.integers(min_value=0, max_value=10000),
    )
    def test_grow_depth_never_exceeds_max(self, max_depth: int, seed: int) -> None:
        """generate_grow must produce trees with depth <= max_depth."""
        reg = _make_registry()
        rng = np.random.default_rng(seed)
        tree = generate_grow(reg, max_depth, Series, rng)
        assert tree.depth <= max_depth

    @settings(max_examples=50)
    @given(
        max_depth=st.integers(min_value=1, max_value=6),
        seed=st.integers(min_value=0, max_value=10000),
    )
    def test_full_depth_never_exceeds_max(self, max_depth: int, seed: int) -> None:
        """generate_full must produce trees with depth <= max_depth."""
        reg = _make_registry()
        rng = np.random.default_rng(seed)
        tree = generate_full(reg, max_depth, Series, rng)
        assert tree.depth <= max_depth

    @settings(max_examples=50)
    @given(
        max_depth=st.integers(min_value=1, max_value=6),
        seed=st.integers(min_value=0, max_value=10000),
    )
    def test_grow_size_at_least_one(self, max_depth: int, seed: int) -> None:
        """Every generated tree has size >= 1."""
        reg = _make_registry()
        rng = np.random.default_rng(seed)
        tree = generate_grow(reg, max_depth, Series, rng)
        assert tree.size >= 1

    @settings(max_examples=50)
    @given(
        max_depth=st.integers(min_value=1, max_value=6),
        seed=st.integers(min_value=0, max_value=10000),
    )
    def test_full_size_at_least_one(self, max_depth: int, seed: int) -> None:
        """Every generated tree has size >= 1."""
        reg = _make_registry()
        rng = np.random.default_rng(seed)
        tree = generate_full(reg, max_depth, Series, rng)
        assert tree.size >= 1

    @settings(max_examples=50)
    @given(
        max_depth=st.integers(min_value=1, max_value=6),
        seed=st.integers(min_value=0, max_value=10000),
    )
    def test_grow_output_type_matches_request(self, max_depth: int, seed: int) -> None:
        """generate_grow must produce a tree whose output_type equals the requested type."""
        reg = _make_registry()
        rng = np.random.default_rng(seed)
        tree = generate_grow(reg, max_depth, Series, rng)
        assert tree.output_type == Series

    @settings(max_examples=50)
    @given(
        max_depth=st.integers(min_value=1, max_value=6),
        seed=st.integers(min_value=0, max_value=10000),
    )
    def test_full_output_type_matches_request(self, max_depth: int, seed: int) -> None:
        """generate_full must produce a tree whose output_type equals the requested type."""
        reg = _make_registry()
        rng = np.random.default_rng(seed)
        tree = generate_full(reg, max_depth, Series, rng)
        assert tree.output_type == Series

    @settings(max_examples=50)
    @given(
        max_depth=st.integers(min_value=1, max_value=6),
        seed=st.integers(min_value=0, max_value=10000),
    )
    def test_generated_tree_is_valid_program_type(
        self, max_depth: int, seed: int
    ) -> None:
        """Every node produced is one of the valid Program union members."""
        tree = _random_tree(max_depth, seed)
        assert isinstance(
            tree, (TerminalNode, ConstantNode, FunctionNode, ParameterizedNode)
        )


# ---------------------------------------------------------------------------
# 2. Crossover invariants
# ---------------------------------------------------------------------------


class TestCrossoverInvariants:
    """Property tests for subtree_crossover."""

    @settings(max_examples=50)
    @given(
        max_depth=st.integers(min_value=2, max_value=5),
        seed1=st.integers(min_value=0, max_value=10000),
        seed2=st.integers(min_value=0, max_value=10000),
        xo_seed=st.integers(min_value=0, max_value=10000),
    )
    def test_returns_exactly_two_programs(
        self, max_depth: int, seed1: int, seed2: int, xo_seed: int
    ) -> None:
        """subtree_crossover always returns a 2-tuple of programs."""
        reg = _make_registry()
        p1 = _random_tree(max_depth, seed1)
        p2 = _random_tree(max_depth, seed2)
        rng = np.random.default_rng(xo_seed)
        result = subtree_crossover(p1, p2, reg, max_depth, rng)
        assert isinstance(result, tuple)
        assert len(result) == 2
        c1, c2 = result
        assert isinstance(
            c1, (TerminalNode, ConstantNode, FunctionNode, ParameterizedNode)
        )
        assert isinstance(
            c2, (TerminalNode, ConstantNode, FunctionNode, ParameterizedNode)
        )

    @settings(max_examples=50)
    @given(
        max_depth=st.integers(min_value=2, max_value=5),
        seed1=st.integers(min_value=0, max_value=10000),
        seed2=st.integers(min_value=0, max_value=10000),
        xo_seed=st.integers(min_value=0, max_value=10000),
    )
    def test_output_types_preserved(
        self, max_depth: int, seed1: int, seed2: int, xo_seed: int
    ) -> None:
        """Both children preserve the output_type of their respective parent."""
        reg = _make_registry()
        p1 = _random_tree(max_depth, seed1)
        p2 = _random_tree(max_depth, seed2)
        rng = np.random.default_rng(xo_seed)
        c1, c2 = subtree_crossover(p1, p2, reg, max_depth, rng)
        assert c1.output_type == p1.output_type
        assert c2.output_type == p2.output_type

    @settings(max_examples=50)
    @given(
        max_depth=st.integers(min_value=2, max_value=5),
        seed1=st.integers(min_value=0, max_value=10000),
        seed2=st.integers(min_value=0, max_value=10000),
        xo_seed=st.integers(min_value=0, max_value=10000),
    )
    def test_depth_constraints_respected(
        self, max_depth: int, seed1: int, seed2: int, xo_seed: int
    ) -> None:
        """Children depth <= max_depth, or at most the parent depth if crossover falls back."""
        reg = _make_registry()
        p1 = _random_tree(max_depth, seed1)
        p2 = _random_tree(max_depth, seed2)
        rng = np.random.default_rng(xo_seed)
        c1, c2 = subtree_crossover(p1, p2, reg, max_depth, rng)
        # Children are either within max_depth (successful swap)
        # or equal to parent depth (fallback returned originals)
        assert c1.depth <= max(max_depth, p1.depth)
        assert c2.depth <= max(max_depth, p2.depth)


# ---------------------------------------------------------------------------
# 3. Mutation invariants
# ---------------------------------------------------------------------------


class TestMutationInvariants:
    """Property tests for subtree_mutation and point_mutation."""

    @settings(max_examples=50)
    @given(
        max_depth=st.integers(min_value=2, max_value=5),
        seed=st.integers(min_value=0, max_value=10000),
        mut_seed=st.integers(min_value=0, max_value=10000),
    )
    def test_subtree_mutation_preserves_output_type(
        self, max_depth: int, seed: int, mut_seed: int
    ) -> None:
        """subtree_mutation preserves root output_type."""
        reg = _make_registry()
        tree = _random_tree(max_depth, seed)
        rng = np.random.default_rng(mut_seed)
        mutant = subtree_mutation(tree, reg, max_depth, rng)
        assert mutant.output_type == tree.output_type

    @settings(max_examples=50)
    @given(
        max_depth=st.integers(min_value=2, max_value=5),
        seed=st.integers(min_value=0, max_value=10000),
        mut_seed=st.integers(min_value=0, max_value=10000),
    )
    def test_subtree_mutation_returns_valid_program(
        self, max_depth: int, seed: int, mut_seed: int
    ) -> None:
        """subtree_mutation always returns a valid Program (non-None)."""
        reg = _make_registry()
        tree = _random_tree(max_depth, seed)
        rng = np.random.default_rng(mut_seed)
        mutant = subtree_mutation(tree, reg, max_depth, rng)
        assert mutant is not None
        assert isinstance(
            mutant, (TerminalNode, ConstantNode, FunctionNode, ParameterizedNode)
        )

    @settings(max_examples=50)
    @given(
        max_depth=st.integers(min_value=2, max_value=5),
        seed=st.integers(min_value=0, max_value=10000),
        mut_seed=st.integers(min_value=0, max_value=10000),
    )
    def test_subtree_mutation_depth_within_bounds(
        self, max_depth: int, seed: int, mut_seed: int
    ) -> None:
        """subtree_mutation result depth <= max_depth."""
        reg = _make_registry()
        tree = _random_tree(max_depth, seed)
        rng = np.random.default_rng(mut_seed)
        mutant = subtree_mutation(tree, reg, max_depth, rng)
        assert mutant.depth <= max_depth

    @settings(max_examples=50)
    @given(
        max_depth=st.integers(min_value=2, max_value=5),
        seed=st.integers(min_value=0, max_value=10000),
        mut_seed=st.integers(min_value=0, max_value=10000),
    )
    def test_point_mutation_preserves_output_type(
        self, max_depth: int, seed: int, mut_seed: int
    ) -> None:
        """point_mutation preserves root output_type."""
        reg = _make_registry()
        tree = _random_tree(max_depth, seed)
        rng = np.random.default_rng(mut_seed)
        mutant = point_mutation(tree, reg, rng)
        assert mutant.output_type == tree.output_type

    @settings(max_examples=50)
    @given(
        max_depth=st.integers(min_value=2, max_value=5),
        seed=st.integers(min_value=0, max_value=10000),
        mut_seed=st.integers(min_value=0, max_value=10000),
    )
    def test_point_mutation_returns_valid_program(
        self, max_depth: int, seed: int, mut_seed: int
    ) -> None:
        """point_mutation always returns a valid Program (non-None)."""
        reg = _make_registry()
        tree = _random_tree(max_depth, seed)
        rng = np.random.default_rng(mut_seed)
        mutant = point_mutation(tree, reg, rng)
        assert mutant is not None
        assert isinstance(
            mutant, (TerminalNode, ConstantNode, FunctionNode, ParameterizedNode)
        )


# ---------------------------------------------------------------------------
# 4. Evaluation determinism
# ---------------------------------------------------------------------------


class TestEvaluationDeterminism:
    """Property tests for evaluate."""

    @settings(max_examples=50)
    @given(
        max_depth=st.integers(min_value=1, max_value=4),
        seed=st.integers(min_value=0, max_value=10000),
    )
    def test_same_program_same_context_same_output(
        self, max_depth: int, seed: int
    ) -> None:
        """evaluate is deterministic: same program + same context = same output."""
        tree = _random_tree(max_depth, seed)
        ctx = _make_context(length=20)
        result1 = evaluate(tree, ctx)
        result2 = evaluate(tree, ctx)
        np.testing.assert_array_equal(result1, result2)

    @settings(max_examples=50)
    @given(
        const_val=st.floats(
            min_value=-100, max_value=100, allow_nan=False, allow_infinity=False
        ),
    )
    def test_constant_node_evaluation_deterministic(self, const_val: float) -> None:
        """Evaluating a ConstantNode always produces the same array."""
        node = ConstantNode(value=const_val)
        ctx = _make_context(length=10)
        result1 = evaluate(node, ctx)
        result2 = evaluate(node, ctx)
        np.testing.assert_array_equal(result1, result2)
        # All values in the array equal the constant
        np.testing.assert_array_equal(result1, np.full(10, const_val, dtype=np.float64))

    @settings(max_examples=50)
    @given(
        max_depth=st.integers(min_value=1, max_value=4),
        seed=st.integers(min_value=0, max_value=10000),
    )
    def test_evaluate_returns_ndarray(self, max_depth: int, seed: int) -> None:
        """evaluate always returns a numpy ndarray."""
        tree = _random_tree(max_depth, seed)
        ctx = _make_context(length=15)
        result = evaluate(tree, ctx)
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float64

    @settings(max_examples=50)
    @given(
        max_depth=st.integers(min_value=1, max_value=4),
        seed=st.integers(min_value=0, max_value=10000),
    )
    def test_evaluate_output_length_matches_context(
        self, max_depth: int, seed: int
    ) -> None:
        """evaluate output array has the same length as the context arrays."""
        tree = _random_tree(max_depth, seed)
        ctx = _make_context(length=25)
        result = evaluate(tree, ctx)
        assert len(result) == 25

    @settings(max_examples=50)
    @given(
        max_depth=st.integers(min_value=1, max_value=4),
        seed=st.integers(min_value=0, max_value=10000),
    )
    def test_cached_evaluation_matches_uncached(
        self, max_depth: int, seed: int
    ) -> None:
        """evaluate with use_cache=True matches evaluate with use_cache=False."""
        tree = _random_tree(max_depth, seed)
        ctx = _make_context(length=20)
        uncached = evaluate(tree, ctx, use_cache=False)
        cached = evaluate(tree, ctx, use_cache=True)
        np.testing.assert_array_equal(uncached, cached)


# ---------------------------------------------------------------------------
# 5. Simplification invariants
# ---------------------------------------------------------------------------


class TestSimplificationInvariants:
    """Property tests for simplify."""

    @settings(max_examples=50)
    @given(
        max_depth=st.integers(min_value=1, max_value=5),
        seed=st.integers(min_value=0, max_value=10000),
    )
    def test_simplification_preserves_output_type(
        self, max_depth: int, seed: int
    ) -> None:
        """simplify must preserve the output_type of the root node."""
        tree = _random_tree(max_depth, seed)
        simplified = simplify(tree)
        assert simplified.output_type == tree.output_type

    @settings(max_examples=50)
    @given(
        max_depth=st.integers(min_value=1, max_value=5),
        seed=st.integers(min_value=0, max_value=10000),
    )
    def test_simplification_is_idempotent(self, max_depth: int, seed: int) -> None:
        """simplify(simplify(p)) == simplify(p) (FR-7.4)."""
        tree = _random_tree(max_depth, seed)
        once = simplify(tree)
        twice = simplify(once)
        assert once == twice

    @settings(max_examples=50)
    @given(
        max_depth=st.integers(min_value=1, max_value=4),
        seed=st.integers(min_value=0, max_value=10000),
    )
    def test_simplification_preserves_semantics(
        self, max_depth: int, seed: int
    ) -> None:
        """evaluate(tree) == evaluate(simplify(tree)) on the same context."""
        tree = _random_tree(max_depth, seed)
        ctx = _make_context(length=20)
        original_result = evaluate(tree, ctx)
        simplified_result = evaluate(simplify(tree), ctx)
        np.testing.assert_array_equal(original_result, simplified_result)

    @settings(max_examples=50)
    @given(
        max_depth=st.integers(min_value=1, max_value=5),
        seed=st.integers(min_value=0, max_value=10000),
    )
    def test_simplification_returns_valid_program(
        self, max_depth: int, seed: int
    ) -> None:
        """simplify returns a valid Program node type."""
        tree = _random_tree(max_depth, seed)
        simplified = simplify(tree)
        assert isinstance(
            simplified,
            (TerminalNode, ConstantNode, FunctionNode, ParameterizedNode),
        )

    @settings(max_examples=50)
    @given(
        max_depth=st.integers(min_value=1, max_value=5),
        seed=st.integers(min_value=0, max_value=10000),
    )
    def test_simplification_does_not_increase_size(
        self, max_depth: int, seed: int
    ) -> None:
        """simplify never makes the tree larger."""
        tree = _random_tree(max_depth, seed)
        simplified = simplify(tree)
        assert simplified.size <= tree.size


# ---------------------------------------------------------------------------
# 6. Serialization round-trip
# ---------------------------------------------------------------------------


class TestSerializationRoundTrip:
    """Property tests for serialize / deserialize."""

    @settings(max_examples=50)
    @given(
        max_depth=st.integers(min_value=1, max_value=5),
        seed=st.integers(min_value=0, max_value=10000),
    )
    def test_round_trip_preserves_equality(self, max_depth: int, seed: int) -> None:
        """deserialize(serialize(p)) == p for random trees."""
        reg = _make_registry()
        tree = _random_tree(max_depth, seed)
        data = serialize(tree)
        restored = deserialize(data, reg)
        assert restored == tree

    @settings(max_examples=50)
    @given(
        max_depth=st.integers(min_value=1, max_value=5),
        seed=st.integers(min_value=0, max_value=10000),
    )
    def test_serialized_form_is_dict_with_schema_version(
        self, max_depth: int, seed: int
    ) -> None:
        """serialize returns a dict containing a 'schema_version' key."""
        tree = _random_tree(max_depth, seed)
        data = serialize(tree)
        assert isinstance(data, dict)
        assert "schema_version" in data

    @settings(max_examples=50)
    @given(
        max_depth=st.integers(min_value=1, max_value=5),
        seed=st.integers(min_value=0, max_value=10000),
    )
    def test_serialized_form_has_program_key(self, max_depth: int, seed: int) -> None:
        """serialize returns a dict containing a 'program' key."""
        tree = _random_tree(max_depth, seed)
        data = serialize(tree)
        assert "program" in data
        assert isinstance(data["program"], dict)

    @settings(max_examples=50)
    @given(
        max_depth=st.integers(min_value=1, max_value=5),
        seed=st.integers(min_value=0, max_value=10000),
    )
    def test_round_trip_preserves_output_type(self, max_depth: int, seed: int) -> None:
        """Round-trip preserves the output_type of the program."""
        reg = _make_registry()
        tree = _random_tree(max_depth, seed)
        data = serialize(tree)
        restored = deserialize(data, reg)
        assert restored.output_type == tree.output_type

    @settings(max_examples=50)
    @given(
        max_depth=st.integers(min_value=1, max_value=5),
        seed=st.integers(min_value=0, max_value=10000),
    )
    def test_round_trip_preserves_depth_and_size(
        self, max_depth: int, seed: int
    ) -> None:
        """Round-trip preserves structural properties (depth, size)."""
        reg = _make_registry()
        tree = _random_tree(max_depth, seed)
        data = serialize(tree)
        restored = deserialize(data, reg)
        assert restored.depth == tree.depth
        assert restored.size == tree.size

    @settings(max_examples=50)
    @given(
        max_depth=st.integers(min_value=1, max_value=4),
        seed=st.integers(min_value=0, max_value=10000),
    )
    def test_round_trip_preserves_evaluation_semantics(
        self, max_depth: int, seed: int
    ) -> None:
        """evaluate(deserialize(serialize(p))) == evaluate(p)."""
        reg = _make_registry()
        tree = _random_tree(max_depth, seed)
        ctx = _make_context(length=20)
        original_result = evaluate(tree, ctx)
        restored = deserialize(serialize(tree), reg)
        restored_result = evaluate(restored, ctx)
        np.testing.assert_array_equal(original_result, restored_result)
