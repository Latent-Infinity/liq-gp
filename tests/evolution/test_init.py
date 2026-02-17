"""Tests for population initialization (FR-5.1)."""

from __future__ import annotations

import numpy as np
import pytest

from liq.gp.config import GPConfig
from liq.gp.primitives.registry import PrimitiveRegistry
from liq.gp.program.ast import (
    ConstantNode,
    FunctionNode,
    ParameterizedNode,
    Program,
    TerminalNode,
)
from liq.gp.types import ParamSpec, Series

# --- helpers ---------------------------------------------------------------


def _make_registry() -> PrimitiveRegistry:
    """Build a minimal registry for testing initialization."""
    reg = PrimitiveRegistry()

    # Terminals
    reg.register("close", lambda: None, input_types=(), output_type=Series)
    reg.register("volume", lambda: None, input_types=(), output_type=Series)

    # Functions
    reg.register(
        "add",
        lambda a, b: a + b,
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
    reg.register(
        "mul",
        lambda a, b: a * b,
        category="numeric",
        input_types=(Series, Series),
        output_type=Series,
    )

    # Parameterized function
    ps = ParamSpec(name="period", dtype=int, default=20, min_value=2, max_value=200)
    reg.register(
        "highest",
        lambda a, *, period=20: a,
        category="indicator",
        input_types=(Series,),
        output_type=Series,
        param_specs=[ps],
    )

    return reg


def _make_config(**overrides: object) -> GPConfig:
    """Build a GPConfig with sensible test defaults."""
    defaults: dict[str, object] = {
        "population_size": 20,
        "max_depth": 4,
        "generations": 1,
        "seed": 42,
    }
    defaults.update(overrides)
    return GPConfig(**defaults)  # type: ignore[arg-type]


def _collect_nodes(program: Program) -> list[Program]:
    """Collect all nodes in a tree (pre-order)."""
    nodes: list[Program] = [program]
    if isinstance(program, (FunctionNode, ParameterizedNode)):
        for child in program.children:
            nodes.extend(_collect_nodes(child))
    return nodes


def _verify_type_validity(program: Program, registry: PrimitiveRegistry) -> None:
    """Verify that all type signatures in a tree are valid."""
    if isinstance(program, (TerminalNode, ConstantNode)):
        assert program.output_type is not None
    elif isinstance(program, (FunctionNode, ParameterizedNode)):
        # Type checking is enforced at construction, so if the node exists
        # it must be valid. Just verify children recursively.
        for child in program.children:
            _verify_type_validity(child, registry)


# --- Full tree generation ---------------------------------------------------


class TestFullTree:
    """'Full' trees have all branches at max depth (FR-5.1.1)."""

    def test_full_tree_reaches_target_depth(self) -> None:
        from liq.gp.evolution.init import generate_full

        reg = _make_registry()
        rng = np.random.default_rng(42)
        tree = generate_full(reg, max_depth=3, output_type=Series, rng=rng)
        assert tree.depth == 3

    def test_full_tree_all_leaves_at_max_depth(self) -> None:
        """Every path from root to a leaf should have length = max_depth."""
        from liq.gp.evolution.init import generate_full

        reg = _make_registry()
        rng = np.random.default_rng(42)
        tree = generate_full(reg, max_depth=3, output_type=Series, rng=rng)

        def check_leaf_depths(node: Program, current_depth: int, target: int) -> None:
            if isinstance(node, (TerminalNode, ConstantNode)):
                assert current_depth == target, (
                    f"Leaf at depth {current_depth}, expected {target}"
                )
            elif isinstance(node, (FunctionNode, ParameterizedNode)):
                for child in node.children:
                    check_leaf_depths(child, current_depth + 1, target)

        check_leaf_depths(tree, 0, 3)

    def test_full_tree_depth_zero_is_terminal(self) -> None:
        from liq.gp.evolution.init import generate_full

        reg = _make_registry()
        rng = np.random.default_rng(42)
        tree = generate_full(reg, max_depth=0, output_type=Series, rng=rng)
        assert isinstance(tree, (TerminalNode, ConstantNode))
        assert tree.depth == 0

    def test_full_tree_type_valid(self) -> None:
        from liq.gp.evolution.init import generate_full

        reg = _make_registry()
        rng = np.random.default_rng(42)
        tree = generate_full(reg, max_depth=3, output_type=Series, rng=rng)
        _verify_type_validity(tree, reg)
        assert tree.output_type is Series


# --- Grow tree generation ---------------------------------------------------


class TestGrowTree:
    """'Grow' trees have branches terminating at random depths (FR-5.1.1)."""

    def test_grow_tree_respects_max_depth(self) -> None:
        from liq.gp.evolution.init import generate_grow

        reg = _make_registry()
        rng = np.random.default_rng(42)
        for _ in range(20):
            tree = generate_grow(reg, max_depth=3, output_type=Series, rng=rng)
            assert tree.depth <= 3

    def test_grow_tree_can_be_shorter_than_max(self) -> None:
        """Over many samples, at least one grow tree should be shorter."""
        from liq.gp.evolution.init import generate_grow

        reg = _make_registry()
        rng = np.random.default_rng(42)
        depths = set()
        for _ in range(50):
            tree = generate_grow(reg, max_depth=4, output_type=Series, rng=rng)
            depths.add(tree.depth)
        # With multiple function arities and terminals, we expect variety
        assert len(depths) > 1

    def test_grow_tree_depth_zero_is_terminal(self) -> None:
        from liq.gp.evolution.init import generate_grow

        reg = _make_registry()
        rng = np.random.default_rng(42)
        tree = generate_grow(reg, max_depth=0, output_type=Series, rng=rng)
        assert isinstance(tree, (TerminalNode, ConstantNode))

    def test_grow_tree_type_valid(self) -> None:
        from liq.gp.evolution.init import generate_grow

        reg = _make_registry()
        rng = np.random.default_rng(42)
        tree = generate_grow(reg, max_depth=3, output_type=Series, rng=rng)
        _verify_type_validity(tree, reg)
        assert tree.output_type is Series


# --- Ramped half-and-half ---------------------------------------------------


class TestRampedHalfAndHalf:
    """Ramped half-and-half initialization (FR-5.1.1)."""

    def test_population_size_matches_config(self) -> None:
        from liq.gp.evolution.init import initialize_population

        reg = _make_registry()
        config = _make_config(population_size=20)
        pop = initialize_population(reg, config)
        assert len(pop) == 20

    def test_population_size_odd(self) -> None:
        from liq.gp.evolution.init import initialize_population

        reg = _make_registry()
        config = _make_config(population_size=15)
        pop = initialize_population(reg, config)
        assert len(pop) == 15

    def test_all_trees_respect_max_depth(self) -> None:
        from liq.gp.evolution.init import initialize_population

        reg = _make_registry()
        config = _make_config(population_size=30, max_depth=4)
        pop = initialize_population(reg, config)
        for tree in pop:
            assert tree.depth <= 4

    def test_all_trees_type_valid(self) -> None:
        from liq.gp.evolution.init import initialize_population

        reg = _make_registry()
        config = _make_config(population_size=20, max_depth=4)
        pop = initialize_population(reg, config)
        for tree in pop:
            _verify_type_validity(tree, reg)
            assert tree.output_type is Series

    def test_depth_variety(self) -> None:
        """Ramped initialization should produce trees at various depths."""
        from liq.gp.evolution.init import initialize_population

        reg = _make_registry()
        config = _make_config(population_size=40, max_depth=5)
        pop = initialize_population(reg, config)
        depths = {tree.depth for tree in pop}
        # With ramped init across depths 1..5, we expect multiple depths
        assert len(depths) >= 2


# --- Parameterized node sampling -------------------------------------------


class TestParameterizedNodeSampling:
    """Parameterized nodes have params sampled from range (FR-5.1.2)."""

    def test_params_within_range(self) -> None:
        from liq.gp.evolution.init import initialize_population

        reg = _make_registry()
        config = _make_config(population_size=50, max_depth=4)
        pop = initialize_population(reg, config)

        for tree in pop:
            for node in _collect_nodes(tree):
                if isinstance(node, ParameterizedNode):
                    for ps in node.primitive.param_specs:
                        val = node.params[ps.name]
                        assert ps.min_value <= val <= ps.max_value, (
                            f"{ps.name}={val} outside [{ps.min_value}, {ps.max_value}]"
                        )

    def test_int_params_are_integers(self) -> None:
        from liq.gp.evolution.init import initialize_population

        reg = _make_registry()
        config = _make_config(population_size=50, max_depth=4)
        pop = initialize_population(reg, config)

        for tree in pop:
            for node in _collect_nodes(tree):
                if isinstance(node, ParameterizedNode):
                    for ps in node.primitive.param_specs:
                        if ps.dtype is int:
                            val = node.params[ps.name]
                            assert isinstance(val, int), (
                                f"{ps.name}={val} should be int"
                            )


# --- Determinism -----------------------------------------------------------


class TestDeterminism:
    """Initialization is deterministic with fixed seed (FR-5.1.3)."""

    def test_same_seed_same_population(self) -> None:
        from liq.gp.evolution.init import initialize_population

        reg = _make_registry()
        config = _make_config(population_size=20, seed=123)
        pop1 = initialize_population(reg, config)
        pop2 = initialize_population(reg, config)
        assert len(pop1) == len(pop2)
        for t1, t2 in zip(pop1, pop2, strict=True):
            assert t1 == t2

    def test_different_seed_different_population(self) -> None:
        from liq.gp.evolution.init import initialize_population

        reg = _make_registry()
        pop1 = initialize_population(reg, _make_config(seed=1))
        pop2 = initialize_population(reg, _make_config(seed=2))
        # At least some trees should differ
        differences = sum(1 for t1, t2 in zip(pop1, pop2, strict=True) if t1 != t2)
        assert differences > 0


# --- Output type -----------------------------------------------------------


class TestOutputType:
    """Generated trees have the correct output type."""

    def test_default_output_type_is_series(self) -> None:
        from liq.gp.evolution.init import initialize_population

        reg = _make_registry()
        config = _make_config(population_size=10)
        pop = initialize_population(reg, config)
        for tree in pop:
            assert tree.output_type is Series


# --- Edge cases ------------------------------------------------------------


class TestEdgeCases:
    """Edge cases for initialization."""

    def test_no_functions_for_type_raises(self) -> None:
        """Full tree at depth > 0 with no functions raises PrimitiveError."""
        from liq.gp.errors import PrimitiveError
        from liq.gp.evolution.init import generate_full

        reg = PrimitiveRegistry()
        reg.register("close", lambda: None, input_types=(), output_type=Series)
        # No functions registered, so full tree at depth > 0 should fail
        rng = np.random.default_rng(42)
        with pytest.raises(PrimitiveError, match="No functions"):
            generate_full(reg, max_depth=2, output_type=Series, rng=rng)

    def test_grow_no_functions_returns_terminal(self) -> None:
        """Grow tree with no functions available returns a terminal."""
        from liq.gp.evolution.init import generate_grow

        reg = PrimitiveRegistry()
        reg.register("close", lambda: None, input_types=(), output_type=Series)
        rng = np.random.default_rng(42)
        tree = generate_grow(reg, max_depth=3, output_type=Series, rng=rng)
        assert isinstance(tree, (TerminalNode, ConstantNode))

    def test_no_terminals_uses_constant(self) -> None:
        """When no terminals exist for a type, falls back to a constant."""
        from liq.gp.evolution.init import _sample_terminal

        reg = PrimitiveRegistry()
        # No terminals at all
        rng = np.random.default_rng(42)
        node = _sample_terminal(reg, Series, rng)
        assert isinstance(node, ConstantNode)

    def test_float_param_spec_sampling(self) -> None:
        """Float params are sampled as floats within range."""
        from liq.gp.evolution.init import initialize_population

        reg = PrimitiveRegistry()
        reg.register("x", lambda: None, input_types=(), output_type=Series)
        ps = ParamSpec(
            name="alpha", dtype=float, default=0.5, min_value=0.0, max_value=1.0
        )
        reg.register(
            "smooth",
            lambda a, *, alpha=0.5: a,
            category="transform",
            input_types=(Series,),
            output_type=Series,
            param_specs=[ps],
        )
        config = _make_config(population_size=30, max_depth=3)
        pop = initialize_population(reg, config)

        found_param = False
        for tree in pop:
            for node in _collect_nodes(tree):
                if isinstance(node, ParameterizedNode) and "alpha" in node.params:
                    found_param = True
                    val = node.params["alpha"]
                    assert isinstance(val, float)
                    assert 0.0 <= val <= 1.0
        assert found_param, "Expected at least one ParameterizedNode with 'alpha'"

    def test_max_depth_2(self) -> None:
        """Minimum valid max_depth=2 produces valid trees."""
        from liq.gp.evolution.init import initialize_population

        reg = _make_registry()
        config = _make_config(population_size=10, max_depth=2)
        pop = initialize_population(reg, config)
        assert len(pop) == 10
        for tree in pop:
            assert tree.depth <= 2
