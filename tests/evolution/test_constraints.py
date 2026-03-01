"""Tests for constraint enforcement and parsimony pressure (FR-5.6)."""

from __future__ import annotations

import pytest

from liq.gp.config import GPConfig
from liq.gp.evolution.constraints import (
    apply_parsimony,
    enforce_constraints,
    filter_population,
)
from liq.gp.primitives.registry import PrimitiveInfo
from liq.gp.program.ast import (
    FunctionNode,
    Program,
    TerminalNode,
)
from liq.gp.types import FitnessResult, Series

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _unary_info() -> PrimitiveInfo:
    """Create a unary Series -> Series primitive info."""
    return PrimitiveInfo(
        name="neg",
        category="numeric",
        arity=1,
        input_types=(Series,),
        output_type=Series,
        callable=lambda a: -a,
    )


def _binary_info() -> PrimitiveInfo:
    """Create a binary (Series, Series) -> Series primitive info."""
    return PrimitiveInfo(
        name="add",
        category="numeric",
        arity=2,
        input_types=(Series, Series),
        output_type=Series,
        callable=lambda a, b: a + b,
    )


def _leaf() -> TerminalNode:
    """Single terminal node (depth=0, size=1)."""
    return TerminalNode(name="x", output_type=Series)


def _depth1_tree() -> FunctionNode:
    """neg(x) -- depth=1, size=2."""
    return FunctionNode(primitive=_unary_info(), children=(_leaf(),))


def _depth2_tree() -> FunctionNode:
    """add(neg(x), x) -- depth=2, size=4."""
    return FunctionNode(
        primitive=_binary_info(),
        children=(_depth1_tree(), _leaf()),
    )


def _deep_chain(depth: int) -> Program:
    """Build a linear chain of unary nodes: neg(neg(...neg(x)...)).

    The resulting tree has the given *depth* and ``size = depth + 1``.
    """
    node: Program = _leaf()
    info = _unary_info()
    for _ in range(depth):
        node = FunctionNode(primitive=info, children=(node,))
    return node


def _wide_tree(width: int) -> FunctionNode:
    """Build a binary tree that adds *width* leaves together.

    Returns a balanced binary tree of add nodes. Total size grows linearly
    with *width*: ``2*width - 1`` nodes.
    """
    leaves: list[Program] = [_leaf() for _ in range(width)]
    info = _binary_info()
    while len(leaves) > 1:
        new_layer: list[Program] = []
        for i in range(0, len(leaves) - 1, 2):
            new_layer.append(
                FunctionNode(primitive=info, children=(leaves[i], leaves[i + 1]))
            )
        if len(leaves) % 2 == 1:
            new_layer.append(leaves[-1])
        leaves = new_layer
    assert isinstance(leaves[0], FunctionNode)
    return leaves[0]


def _default_config(**overrides: object) -> GPConfig:
    """Create a GPConfig with sensible defaults, applying overrides."""
    defaults: dict[str, object] = {
        "max_depth": 8,
        "parsimony_mode": "disabled",
    }
    defaults.update(overrides)
    return GPConfig(**defaults)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# enforce_constraints
# ---------------------------------------------------------------------------


class TestEnforceConstraintsDepth:
    """Max depth enforcement."""

    def test_terminal_always_passes(self) -> None:
        """A terminal (depth 0) passes any max_depth >= 2."""
        config = _default_config(max_depth=2)
        assert enforce_constraints(_leaf(), config) is True

    def test_tree_at_max_depth_passes(self) -> None:
        """Tree whose depth equals max_depth is accepted."""
        tree = _deep_chain(4)
        assert tree.depth == 4
        config = _default_config(max_depth=4)
        assert enforce_constraints(tree, config) is True

    def test_tree_exceeding_max_depth_rejected(self) -> None:
        """Tree whose depth exceeds max_depth is rejected."""
        tree = _deep_chain(5)
        assert tree.depth == 5
        config = _default_config(max_depth=4)
        assert enforce_constraints(tree, config) is False

    def test_tree_below_max_depth_passes(self) -> None:
        """Tree whose depth is well under max_depth is accepted."""
        tree = _depth1_tree()
        assert tree.depth == 1
        config = _default_config(max_depth=8)
        assert enforce_constraints(tree, config) is True


class TestEnforceConstraintsSize:
    """Optional max size enforcement."""

    def test_no_max_size_always_passes(self) -> None:
        """When max_size is None, any size passes."""
        tree = _deep_chain(7)
        config = _default_config(max_depth=8, max_size=None)
        assert enforce_constraints(tree, config) is True

    def test_tree_at_max_size_passes(self) -> None:
        """Tree whose size equals max_size is accepted."""
        tree = _depth2_tree()
        assert tree.size == 4
        config = _default_config(max_size=4)
        assert enforce_constraints(tree, config) is True

    def test_tree_exceeding_max_size_rejected(self) -> None:
        """Tree whose size exceeds max_size is rejected."""
        tree = _depth2_tree()
        assert tree.size == 4
        config = _default_config(max_size=3)
        assert enforce_constraints(tree, config) is False

    def test_depth_and_size_both_checked(self) -> None:
        """A tree can fail on depth even if size is fine, and vice-versa."""
        tree = _deep_chain(5)  # depth=5, size=6
        # Passes depth, fails size
        config_a = _default_config(max_depth=8, max_size=3)
        assert enforce_constraints(tree, config_a) is False
        # Fails depth, passes size
        config_b = _default_config(max_depth=3, max_size=100)
        assert enforce_constraints(tree, config_b) is False


# ---------------------------------------------------------------------------
# apply_parsimony -- disabled mode
# ---------------------------------------------------------------------------


class TestParsimonyDisabled:
    """parsimony_mode='disabled' leaves fitnesses unchanged."""

    def test_disabled_returns_same_values(self) -> None:
        pop = [_leaf(), _depth1_tree()]
        fits = [
            FitnessResult(objectives=(1.0,)),
            FitnessResult(objectives=(2.0,)),
        ]
        config = _default_config(parsimony_mode="disabled")
        result = apply_parsimony(fits, pop, config)
        assert len(result) == 2
        assert result[0].objectives == (1.0,)
        assert result[1].objectives == (2.0,)

    def test_disabled_does_not_add_objectives(self) -> None:
        pop = [_leaf()]
        fits = [FitnessResult(objectives=(5.0,))]
        config = _default_config(parsimony_mode="disabled")
        result = apply_parsimony(fits, pop, config)
        assert len(result[0].objectives) == 1


# ---------------------------------------------------------------------------
# apply_parsimony -- lexicographic mode
# ---------------------------------------------------------------------------


class TestParsimonyLexicographic:
    """parsimony_mode='lexicographic' appends -size as tie-breaker."""

    def test_appends_negative_size_objective(self) -> None:
        leaf = _leaf()  # size=1
        tree = _depth2_tree()  # size=4
        pop = [leaf, tree]
        fits = [
            FitnessResult(objectives=(1.0,)),
            FitnessResult(objectives=(1.0,)),
        ]
        config = _default_config(parsimony_mode="lexicographic")
        result = apply_parsimony(fits, pop, config)
        # Leaf should have -1 appended, tree should have -4 appended
        assert result[0].objectives == (1.0, -1)
        assert result[1].objectives == (1.0, -4)

    def test_preserves_raw_objectives_in_metadata(self) -> None:
        pop = [_leaf()]
        fits = [FitnessResult(objectives=(3.0,))]
        config = _default_config(parsimony_mode="lexicographic")
        result = apply_parsimony(fits, pop, config)
        assert result[0].metadata["raw_objectives"] == (3.0,)

    def test_smaller_program_preferred_at_equal_fitness(self) -> None:
        """At equal primary fitness, the program with smaller size gets a
        higher (less negative) secondary objective, so it's preferred."""
        small = _leaf()  # size=1
        large = _depth2_tree()  # size=4
        pop = [small, large]
        fits = [
            FitnessResult(objectives=(1.0,)),
            FitnessResult(objectives=(1.0,)),
        ]
        config = _default_config(parsimony_mode="lexicographic")
        result = apply_parsimony(fits, pop, config)
        # -1 > -4, so small is preferred as a secondary objective
        assert result[0].objectives[1] > result[1].objectives[1]


# ---------------------------------------------------------------------------
# apply_parsimony -- pareto mode
# ---------------------------------------------------------------------------


class TestParsimonyPareto:
    """parsimony_mode='pareto' appends -size for NSGA-II multi-objective."""

    def test_appends_size_objective(self) -> None:
        tree = _depth2_tree()  # size=4
        pop = [tree]
        fits = [FitnessResult(objectives=(1.0, 0.5))]
        config = _default_config(
            parsimony_mode="pareto",
            selection_mode="nsga2",
            fitness={
                "objectives": ["f1", "f2"],
                "objective_directions": ["maximize", "minimize"],
            },
        )
        result = apply_parsimony(fits, pop, config)
        assert result[0].objectives == (1.0, 0.5, -4)

    def test_preserves_raw_objectives_in_metadata(self) -> None:
        pop = [_leaf()]
        fits = [FitnessResult(objectives=(2.0, 1.0))]
        config = _default_config(
            parsimony_mode="pareto",
            selection_mode="nsga2",
            fitness={
                "objectives": ["f1", "f2"],
                "objective_directions": ["maximize", "minimize"],
            },
        )
        result = apply_parsimony(fits, pop, config)
        assert result[0].metadata["raw_objectives"] == (2.0, 1.0)


# ---------------------------------------------------------------------------
# apply_parsimony -- linear mode
# ---------------------------------------------------------------------------


class TestParsimonyLinear:
    """parsimony_mode='linear' penalises fitness proportionally to size."""

    def test_subtracts_penalty_from_first_objective(self) -> None:
        tree = _depth2_tree()  # size=4
        pop = [tree]
        fits = [FitnessResult(objectives=(10.0,))]
        config = _default_config(
            parsimony_mode="linear",
            parsimony_coefficient=0.5,
        )
        result = apply_parsimony(fits, pop, config)
        # 10.0 - 0.5 * 4 = 8.0
        assert result[0].objectives[0] == pytest.approx(8.0)

    def test_preserves_subsequent_objectives(self) -> None:
        pop = [_leaf()]  # size=1
        fits = [FitnessResult(objectives=(5.0, 3.0))]
        config = _default_config(
            parsimony_mode="linear",
            parsimony_coefficient=1.0,
        )
        result = apply_parsimony(fits, pop, config)
        # First: 5.0 - 1.0*1 = 4.0; second unchanged at 3.0
        assert result[0].objectives == pytest.approx((4.0, 3.0))

    def test_zero_coefficient_no_change(self) -> None:
        pop = [_depth2_tree()]
        fits = [FitnessResult(objectives=(7.0,))]
        config = _default_config(
            parsimony_mode="linear",
            parsimony_coefficient=0.0,
        )
        result = apply_parsimony(fits, pop, config)
        assert result[0].objectives[0] == pytest.approx(7.0)

    def test_preserves_raw_objectives_in_metadata(self) -> None:
        pop = [_leaf()]
        fits = [FitnessResult(objectives=(5.0,))]
        config = _default_config(
            parsimony_mode="linear",
            parsimony_coefficient=0.1,
        )
        result = apply_parsimony(fits, pop, config)
        assert result[0].metadata["raw_objectives"] == (5.0,)

    def test_larger_tree_penalised_more(self) -> None:
        small = _leaf()  # size=1
        large = _depth2_tree()  # size=4
        pop = [small, large]
        fits = [
            FitnessResult(objectives=(10.0,)),
            FitnessResult(objectives=(10.0,)),
        ]
        config = _default_config(
            parsimony_mode="linear",
            parsimony_coefficient=0.5,
        )
        result = apply_parsimony(fits, pop, config)
        # small: 10.0 - 0.5*1 = 9.5 ; large: 10.0 - 0.5*4 = 8.0
        assert result[0].objectives[0] > result[1].objectives[0]


class TestParsimonySizeDiversity:
    """parsimony_mode='size_diversity' rewards rare tree sizes."""

    def test_rarer_size_gets_higher_score(self) -> None:
        small = _leaf()  # size=1 (rare)
        medium_a = _depth2_tree()  # size=4
        medium_b = _depth2_tree()  # size=4
        pop = [small, medium_a, medium_b]
        fits = [
            FitnessResult(objectives=(1.0,)),
            FitnessResult(objectives=(1.0,)),
            FitnessResult(objectives=(1.0,)),
        ]
        config = _default_config(
            parsimony_mode="size_diversity",
        )
        result = apply_parsimony(fits, pop, config)
        # small size appears once, medium size appears twice
        assert result[0].objectives == (1.0, 1.0)
        assert result[1].objectives == pytest.approx((1.0, 0.5))
        assert result[2].objectives == pytest.approx((1.0, 0.5))
        # smaller size class is now favored
        assert result[0].objectives[1] > result[1].objectives[1]

    def test_uniform_frequency_keeps_objective_balanced(self) -> None:
        small = _leaf()  # size=1
        medium = _depth1_tree()  # size=2
        large = _depth2_tree()  # size=4
        pop = [small, medium, large]
        fits = [
            FitnessResult(objectives=(1.0,)),
            FitnessResult(objectives=(1.0,)),
            FitnessResult(objectives=(1.0,)),
        ]
        config = _default_config(
            parsimony_mode="size_diversity",
        )
        result = apply_parsimony(fits, pop, config)
        assert result[0].objectives[1] == pytest.approx(1.0)
        assert result[1].objectives[1] == pytest.approx(1.0)
        assert result[2].objectives[1] == pytest.approx(1.0)

    def test_size_diversity_keeps_primary_objective_first(self) -> None:
        small = _leaf()
        medium = _depth2_tree()
        big = _wide_tree(4)  # larger size than medium
        pop = [small, medium, big]
        fits = [
            FitnessResult(objectives=(0.0,)),
            FitnessResult(objectives=(10.0,)),
            FitnessResult(objectives=(20.0,)),
        ]
        config = _default_config(
            parsimony_mode="size_diversity",
        )
        result = apply_parsimony(fits, pop, config)
        # Larger primary fitness should remain larger before parsimony reward.
        assert result[2].objectives[0] > result[1].objectives[0] > result[0].objectives[0]

    def test_size_diversity_is_compatible_with_nsga2(self) -> None:
        """size_diversity mode appends bonus objective without breaking NSGA-II fitness shape."""
        small = _leaf()
        medium = _depth2_tree()
        big = _wide_tree(4)
        pop = [small, medium, big]
        fits = [
            FitnessResult(objectives=(1.0, 5.0)),
            FitnessResult(objectives=(2.0, 4.0)),
            FitnessResult(objectives=(3.0, 3.0)),
        ]
        config = _default_config(
            parsimony_mode="size_diversity",
            selection_mode="nsga2",
            fitness={
                "objectives": ["f1", "f2"],
                "objective_directions": ["maximize", "minimize"],
            },
        )
        result = apply_parsimony(fits, pop, config)
        assert len(result) == 3
        assert all(len(fr.objectives) == 3 for fr in result)
        assert all(fr.objectives[:2] == fit.objectives for fr, fit in zip(result, fits, strict=True))
        assert all(
            fr.metadata["raw_objectives"] == fit.objectives
            for fr, fit in zip(result, fits, strict=True)
        )


# ---------------------------------------------------------------------------
# filter_population
# ---------------------------------------------------------------------------


class TestFilterPopulation:
    """Remove constraint-violating individuals."""

    def test_all_valid_unchanged(self) -> None:
        pop = [_leaf(), _depth1_tree()]
        fits = [
            FitnessResult(objectives=(1.0,)),
            FitnessResult(objectives=(2.0,)),
        ]
        config = _default_config(max_depth=8)
        progs, frs = filter_population(pop, fits, config)
        assert len(progs) == 2
        assert len(frs) == 2

    def test_removes_violators(self) -> None:
        ok = _leaf()
        too_deep = _deep_chain(6)
        pop = [ok, too_deep]
        fits = [
            FitnessResult(objectives=(1.0,)),
            FitnessResult(objectives=(2.0,)),
        ]
        config = _default_config(max_depth=4)
        progs, frs = filter_population(pop, fits, config)
        assert len(progs) == 1
        assert progs[0] is ok
        assert frs[0].objectives == (1.0,)

    def test_all_violators_returns_empty(self) -> None:
        pop = [_deep_chain(5), _deep_chain(6)]
        fits = [
            FitnessResult(objectives=(1.0,)),
            FitnessResult(objectives=(2.0,)),
        ]
        config = _default_config(max_depth=3)
        progs, frs = filter_population(pop, fits, config)
        assert progs == []
        assert frs == []

    def test_filters_by_size(self) -> None:
        small = _leaf()  # size=1
        big = _depth2_tree()  # size=4
        pop = [small, big]
        fits = [
            FitnessResult(objectives=(1.0,)),
            FitnessResult(objectives=(2.0,)),
        ]
        config = _default_config(max_depth=8, max_size=2)
        progs, frs = filter_population(pop, fits, config)
        assert len(progs) == 1
        assert progs[0] is small

    def test_preserves_order(self) -> None:
        """Filtered results maintain original ordering."""
        p1, p2, p3 = _leaf(), _deep_chain(3), _leaf()
        f1 = FitnessResult(objectives=(1.0,))
        f2 = FitnessResult(objectives=(2.0,))
        f3 = FitnessResult(objectives=(3.0,))
        config = _default_config(max_depth=2)
        progs, frs = filter_population([p1, p2, p3], [f1, f2, f3], config)
        assert len(progs) == 2
        assert progs[0] is p1
        assert progs[1] is p3
        assert frs[0].objectives == (1.0,)
        assert frs[1].objectives == (3.0,)
