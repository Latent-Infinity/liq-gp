"""Tests for genetic operators (FR-5.2)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from liq.gp.primitives.registry import PrimitiveInfo, PrimitiveRegistry
from liq.gp.program.ast import (
    ConstantNode,
    FunctionNode,
    ParameterizedNode,
    Program,
    TerminalNode,
)
from liq.gp.types import BoolSeries, ParamSpec, Series

# --- helpers ---------------------------------------------------------------


@dataclass(frozen=True)
class _RegimeBlock:
    program: Program


@dataclass(frozen=True)
class _RegimeModel:
    detector: _RegimeBlock
    gate: _RegimeBlock
    experts: tuple[_RegimeBlock, ...]
    risk: _RegimeBlock | None = None
    weights: tuple[float, ...] | list[float] | None = None


def _make_regime_registry() -> PrimitiveRegistry:
    reg = PrimitiveRegistry()
    reg.register(
        "if_then_else",
        lambda cond, on_true, on_false: np.where(cond > 0.5, on_true, on_false),
        category="control",
        input_types=(BoolSeries, Series, Series),
        output_type=Series,
    )
    reg.register(
        "mul",
        lambda a, b: a * b,
        category="math",
        input_types=(Series, Series),
        output_type=Series,
    )
    reg.register(
        "add",
        lambda a, b: a + b,
        category="math",
        input_types=(Series, Series),
        output_type=Series,
    )
    return reg


def _make_regime_model(
    registry: PrimitiveRegistry,
    *,
    gate_name: str = "gate",
    detector_name: str = "detector",
    expert_names: tuple[str, ...] = ("e1", "e2"),
    include_risk: bool = False,
) -> Program:
    from liq.gp import compile_regime_model_to_program

    detector = _RegimeBlock(TerminalNode(name=detector_name, output_type=BoolSeries))
    gate = _RegimeBlock(TerminalNode(name=gate_name, output_type=BoolSeries))
    experts = tuple(
        _RegimeBlock(TerminalNode(name=name, output_type=Series))
        for name in expert_names
    )
    risk = (
        _RegimeBlock(TerminalNode(name="risk", output_type=Series))
        if include_risk
        else None
    )

    model = _RegimeModel(
        detector=detector,
        gate=gate,
        experts=experts,
        risk=risk,
    )
    return compile_regime_model_to_program(model, registry)


def _find_node(program: Program, name: str) -> Program:
    for node in _collect_nodes(program):
        if isinstance(node, TerminalNode) and node.name == name:
            return node
    return terminal_not_found(name)  # pragma: no cover


def terminal_not_found(name: str) -> Program:
    msg = f"No terminal named {name!r} in program"
    raise AssertionError(msg)


class _ScriptedRNG:
    def __init__(self, values: tuple[int, ...] | list[int]) -> None:
        self._values = list(values)
        self._position = 0

    def _next_int(self, low: int, high: int | None = None) -> int:
        if self._position >= len(self._values):
            self._position += 1
            return 0
        value = int(self._values[self._position])
        self._position += 1
        raw = value % (high - low if high is not None else low)
        if high is None:
            return raw
        return raw + low

    def integers(self, low: int, high: int | None = None, *args, **kwargs) -> int:
        return self._next_int(low, high)

    def random(self, *args, **kwargs) -> float:
        return 0.5

    def uniform(self, low: float, high: float, *args, **kwargs) -> float:
        return (low + high) / 2.0


def _make_registry() -> PrimitiveRegistry:
    reg = PrimitiveRegistry()
    reg.register("close", lambda: None, input_types=(), output_type=Series)
    reg.register("volume", lambda: None, input_types=(), output_type=Series)
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


def _make_tree() -> FunctionNode:
    """add(neg(close), mul(volume, 2.0))"""
    close = TerminalNode(name="close", output_type=Series)
    volume = TerminalNode(name="volume", output_type=Series)
    const = ConstantNode(value=2.0)
    neg_info = PrimitiveInfo(
        name="neg",
        category="numeric",
        arity=1,
        input_types=(Series,),
        output_type=Series,
        callable=lambda a: -a,
    )
    mul_info = PrimitiveInfo(
        name="mul",
        category="numeric",
        arity=2,
        input_types=(Series, Series),
        output_type=Series,
        callable=lambda a, b: a * b,
    )
    add_info = PrimitiveInfo(
        name="add",
        category="numeric",
        arity=2,
        input_types=(Series, Series),
        output_type=Series,
        callable=lambda a, b: a + b,
    )
    neg_close = FunctionNode(primitive=neg_info, children=(close,))
    mul_vol = FunctionNode(primitive=mul_info, children=(volume, const))
    return FunctionNode(primitive=add_info, children=(neg_close, mul_vol))


def _make_param_tree() -> ParameterizedNode:
    """highest(close, period=20)"""
    close = TerminalNode(name="close", output_type=Series)
    ps = ParamSpec(name="period", dtype=int, default=20, min_value=2, max_value=200)
    info = PrimitiveInfo(
        name="highest",
        category="indicator",
        arity=1,
        input_types=(Series,),
        output_type=Series,
        callable=lambda a, *, period=20: a,
        param_specs=[ps],
    )
    return ParameterizedNode(primitive=info, children=(close,), params={"period": 20})


def _make_discrete_param_tree() -> ParameterizedNode:
    """highest(close, period=21) with discrete grid."""
    close = TerminalNode(name="close", output_type=Series)
    ps = ParamSpec(
        name="period",
        dtype=int,
        default=21,
        allowed_values=[8, 13, 21, 34],
    )
    info = PrimitiveInfo(
        name="highest",
        category="indicator",
        arity=1,
        input_types=(Series,),
        output_type=Series,
        callable=lambda a, *, period=21: a,
        param_specs=[ps],
    )
    return ParameterizedNode(primitive=info, children=(close,), params={"period": 21})


def _collect_nodes(program: Program) -> list[Program]:
    nodes: list[Program] = [program]
    if isinstance(program, (FunctionNode, ParameterizedNode)):
        for child in program.children:
            nodes.extend(_collect_nodes(child))
    return nodes


# --- Subtree crossover (FR-5.2.1) ------------------------------------------


class TestSubtreeCrossover:
    """Subtree crossover swaps compatible-type subtrees."""

    def test_produces_new_trees(self) -> None:
        from liq.gp.evolution.operators import subtree_crossover

        reg = _make_registry()
        p1 = _make_tree()
        p2 = _make_tree()
        rng = np.random.default_rng(42)
        c1, c2 = subtree_crossover(p1, p2, reg, max_depth=6, rng=rng)
        assert c1 is not p1
        assert c2 is not p2

    def test_respects_max_depth(self) -> None:
        from liq.gp.evolution.operators import subtree_crossover

        reg = _make_registry()
        p1 = _make_tree()
        p2 = _make_tree()
        rng = np.random.default_rng(42)
        for _ in range(20):
            c1, c2 = subtree_crossover(p1, p2, reg, max_depth=4, rng=rng)
            assert c1.depth <= 4
            assert c2.depth <= 4

    def test_preserves_output_type(self) -> None:
        from liq.gp.evolution.operators import subtree_crossover

        reg = _make_registry()
        p1 = _make_tree()
        p2 = _make_tree()
        rng = np.random.default_rng(42)
        c1, c2 = subtree_crossover(p1, p2, reg, max_depth=6, rng=rng)
        assert c1.output_type is Series
        assert c2.output_type is Series

    def test_returns_parents_when_no_valid_swap(self) -> None:
        """When no compatible swap point exists, returns copies of parents."""
        from liq.gp.evolution.operators import subtree_crossover

        reg = _make_registry()
        # Two terminals with incompatible types -- no valid swap
        p1 = TerminalNode(name="close", output_type=Series)
        p2 = TerminalNode(name="flag", output_type=BoolSeries)
        rng = np.random.default_rng(42)
        c1, c2 = subtree_crossover(p1, p2, reg, max_depth=6, rng=rng)
        # Should return the original trees unchanged (types don't match)
        assert c1 == p1
        assert c2 == p2

    def test_deterministic(self) -> None:
        from liq.gp.evolution.operators import subtree_crossover

        reg = _make_registry()
        p1, p2 = _make_tree(), _make_tree()
        c1a, c2a = subtree_crossover(
            p1, p2, reg, max_depth=6, rng=np.random.default_rng(99)
        )
        c1b, c2b = subtree_crossover(
            p1, p2, reg, max_depth=6, rng=np.random.default_rng(99)
        )
        assert c1a == c1b
        assert c2a == c2b

    def test_respects_max_attempts_override(self) -> None:
        from liq.gp.evolution.operators import subtree_crossover

        reg = _make_registry()
        p1 = TerminalNode(name="close", output_type=Series)
        p2 = TerminalNode(name="flag", output_type=BoolSeries)
        c1, c2 = subtree_crossover(
            p1,
            p2,
            reg,
            max_depth=6,
            rng=np.random.default_rng(42),
            max_attempts=1,
        )
        assert c1 == p1
        assert c2 == p2


# --- Subtree mutation (FR-5.2.2) -------------------------------------------


class TestSubtreeMutation:
    """Subtree mutation replaces a subtree with a new random one."""

    def test_produces_new_tree(self) -> None:
        from liq.gp.evolution.operators import subtree_mutation

        reg = _make_registry()
        tree = _make_tree()
        rng = np.random.default_rng(42)
        mutant = subtree_mutation(tree, reg, max_depth=6, rng=rng)
        assert mutant is not tree

    def test_respects_max_depth(self) -> None:
        from liq.gp.evolution.operators import subtree_mutation

        reg = _make_registry()
        tree = _make_tree()
        rng = np.random.default_rng(42)
        for _ in range(20):
            mutant = subtree_mutation(tree, reg, max_depth=4, rng=rng)
            assert mutant.depth <= 4

    def test_preserves_output_type(self) -> None:
        from liq.gp.evolution.operators import subtree_mutation

        reg = _make_registry()
        tree = _make_tree()
        rng = np.random.default_rng(42)
        mutant = subtree_mutation(tree, reg, max_depth=6, rng=rng)
        assert mutant.output_type is Series

    def test_deterministic(self) -> None:
        from liq.gp.evolution.operators import subtree_mutation

        reg = _make_registry()
        tree = _make_tree()
        m1 = subtree_mutation(tree, reg, max_depth=6, rng=np.random.default_rng(42))
        m2 = subtree_mutation(tree, reg, max_depth=6, rng=np.random.default_rng(42))
        assert m1 == m2


# --- Point mutation (FR-5.2.3) ---------------------------------------------


class TestPointMutation:
    """Point mutation replaces a node with a same-arity, same-type primitive."""

    def test_produces_new_tree(self) -> None:
        from liq.gp.evolution.operators import point_mutation

        reg = _make_registry()
        tree = _make_tree()
        rng = np.random.default_rng(42)
        mutant = point_mutation(tree, reg, rng=rng)
        assert mutant is not tree

    def test_preserves_output_type(self) -> None:
        from liq.gp.evolution.operators import point_mutation

        reg = _make_registry()
        tree = _make_tree()
        rng = np.random.default_rng(42)
        mutant = point_mutation(tree, reg, rng=rng)
        assert mutant.output_type is Series

    def test_preserves_depth(self) -> None:
        """Point mutation doesn't change tree structure."""
        from liq.gp.evolution.operators import point_mutation

        reg = _make_registry()
        tree = _make_tree()
        rng = np.random.default_rng(42)
        mutant = point_mutation(tree, reg, rng=rng)
        assert mutant.depth == tree.depth

    def test_deterministic(self) -> None:
        from liq.gp.evolution.operators import point_mutation

        reg = _make_registry()
        tree = _make_tree()
        m1 = point_mutation(tree, reg, rng=np.random.default_rng(42))
        m2 = point_mutation(tree, reg, rng=np.random.default_rng(42))
        assert m1 == m2


# --- Parameter mutation (FR-5.2.4) -----------------------------------------


class TestParameterMutation:
    """Parameter mutation adds Gaussian noise to parameters."""

    def test_mutates_parameters(self) -> None:
        from liq.gp.evolution.operators import parameter_mutation

        tree = _make_param_tree()
        rng = np.random.default_rng(42)
        mutant = parameter_mutation(tree, rng=rng)
        # Should be a new tree
        assert mutant is not tree

    def test_params_stay_in_range(self) -> None:
        from liq.gp.evolution.operators import parameter_mutation

        tree = _make_param_tree()
        rng = np.random.default_rng(42)
        for _ in range(50):
            mutant = parameter_mutation(tree, rng=rng)
            for node in _collect_nodes(mutant):
                if isinstance(node, ParameterizedNode):
                    for ps in node.primitive.param_specs:
                        val = node.params[ps.name]
                        assert ps.min_value <= val <= ps.max_value

    def test_int_params_rounded(self) -> None:
        from liq.gp.evolution.operators import parameter_mutation

        tree = _make_param_tree()
        rng = np.random.default_rng(42)
        for _ in range(20):
            mutant = parameter_mutation(tree, rng=rng)
            for node in _collect_nodes(mutant):
                if isinstance(node, ParameterizedNode):
                    for ps in node.primitive.param_specs:
                        if ps.dtype is int:
                            assert isinstance(node.params[ps.name], int)

    def test_no_params_returns_copy(self) -> None:
        """Tree with no parameterized nodes returned unchanged."""
        from liq.gp.evolution.operators import parameter_mutation

        tree = _make_tree()  # no ParameterizedNode
        rng = np.random.default_rng(42)
        mutant = parameter_mutation(tree, rng=rng)
        assert mutant == tree

    def test_deterministic(self) -> None:
        from liq.gp.evolution.operators import parameter_mutation

        tree = _make_param_tree()
        m1 = parameter_mutation(tree, rng=np.random.default_rng(42))
        m2 = parameter_mutation(tree, rng=np.random.default_rng(42))
        assert m1 == m2

    def test_discrete_mutation_selects_allowed_neighbor(self) -> None:
        from liq.gp.evolution.operators import parameter_mutation

        tree = _make_discrete_param_tree()
        rng = np.random.default_rng(42)
        mutant = parameter_mutation(tree, rng=rng)
        for node in _collect_nodes(mutant):
            if isinstance(node, ParameterizedNode):
                ps = node.primitive.param_specs[0]
                assert ps.value_is_discrete()
                assert node.params[ps.name] in ps.allowed_values


# --- Hoist mutation (FR-5.2.5) ---------------------------------------------


class TestHoistMutation:
    """Hoist mutation replaces the tree with one of its subtrees."""

    def test_result_is_subtree(self) -> None:
        from liq.gp.evolution.operators import hoist_mutation

        tree = _make_tree()
        rng = np.random.default_rng(42)
        mutant = hoist_mutation(tree, rng=rng)
        # Mutant must be smaller or equal in size
        assert mutant.size <= tree.size

    def test_preserves_output_type(self) -> None:
        from liq.gp.evolution.operators import hoist_mutation

        tree = _make_tree()
        rng = np.random.default_rng(42)
        mutant = hoist_mutation(tree, rng=rng)
        assert mutant.output_type is Series

    def test_reduces_depth(self) -> None:
        """Over many hoists, at least some should be shallower."""
        from liq.gp.evolution.operators import hoist_mutation

        tree = _make_tree()
        rng = np.random.default_rng(42)
        depths = set()
        for _ in range(30):
            mutant = hoist_mutation(tree, rng=rng)
            depths.add(mutant.depth)
        assert min(depths) < tree.depth

    def test_terminal_unchanged(self) -> None:
        """Hoisting a terminal returns the terminal."""
        from liq.gp.evolution.operators import hoist_mutation

        tree = TerminalNode(name="close", output_type=Series)
        rng = np.random.default_rng(42)
        mutant = hoist_mutation(tree, rng=rng)
        assert mutant == tree

    def test_deterministic(self) -> None:
        from liq.gp.evolution.operators import hoist_mutation

        tree = _make_tree()
        m1 = hoist_mutation(tree, rng=np.random.default_rng(42))
        m2 = hoist_mutation(tree, rng=np.random.default_rng(42))
        assert m1 == m2


# --- Immutability (FR-5.2.6) -----------------------------------------------


class TestImmutability:
    """All operators produce new trees; parents are never modified."""

    def test_crossover_does_not_modify_parents(self) -> None:
        from liq.gp.evolution.operators import subtree_crossover

        reg = _make_registry()
        p1 = _make_tree()
        p2 = _make_tree()
        p1_hash = hash(p1)
        p2_hash = hash(p2)
        rng = np.random.default_rng(42)
        subtree_crossover(p1, p2, reg, max_depth=6, rng=rng)
        assert hash(p1) == p1_hash
        assert hash(p2) == p2_hash

    def test_subtree_mutation_does_not_modify_parent(self) -> None:
        from liq.gp.evolution.operators import subtree_mutation

        reg = _make_registry()
        tree = _make_tree()
        tree_hash = hash(tree)
        rng = np.random.default_rng(42)
        subtree_mutation(tree, reg, max_depth=6, rng=rng)
        assert hash(tree) == tree_hash


# --- Operator selection (FR-5.2.8) -----------------------------------------


class TestOperatorSelection:
    """select_operator picks operators according to configured rates."""

    def test_select_operator_distribution(self) -> None:
        from liq.gp.config import GPConfig
        from liq.gp.evolution.operators import select_operator

        config = GPConfig()
        rng = np.random.default_rng(42)
        counts: dict[str, int] = {}
        for _ in range(1000):
            op = select_operator(config, rng)
            counts[op] = counts.get(op, 0) + 1
        # crossover_rate=0.7 should dominate
        assert counts.get("crossover", 0) > 500
        # All operators should appear at least once with 1000 samples
        assert "crossover" in counts
        assert "subtree_mutation" in counts
        assert "point_mutation" in counts


class TestRegimeBlockConstrainedOperators:
    """Block constraints protect regime boundaries during variation."""

    def test_crossover_blocks_block_mismatches(self) -> None:
        from liq.gp.evolution.operators import (
            BlockConstraintTelemetry,
            _collect_subtrees_by_type,
            subtree_crossover,
        )

        registry = _make_regime_registry()
        parent1 = _make_regime_model(
            registry,
            include_risk=True,
            gate_name="gate_a",
            detector_name="det_a",
            expert_names=("a1", "a2"),
        )
        parent2 = _make_regime_model(
            registry,
            include_risk=True,
            gate_name="gate_b",
            detector_name="det_b",
            expert_names=("b1", "b2"),
        )

        nodes1 = [node for node, _ in _collect_subtrees_by_type(parent1)]
        nodes2 = [node for node, _ in _collect_subtrees_by_type(parent2)]
        risk_idx = nodes1.index(_find_node(parent1, "risk"))
        gate_idx = nodes2.index(_find_node(parent2, "gate_b"))
        rng = _ScriptedRNG([risk_idx, gate_idx])
        telemetry = BlockConstraintTelemetry()

        child1, child2 = subtree_crossover(
            parent1,
            parent2,
            registry,
            max_depth=12,
            rng=rng,
            max_attempts=1,
            block_constraint_telemetry=telemetry,
        )

        assert child1 == parent1
        assert child2 == parent2
        assert telemetry.blocked["crossover:risk"] == 1
        assert telemetry.blocked["crossover:gate"] == 1
        assert telemetry.attempted["crossover:risk"] == 1
        assert telemetry.attempted["crossover:gate"] == 1

    def test_crossover_only_accepts_matching_roles(self) -> None:
        from liq.gp.evolution.operators import (
            BlockConstraintTelemetry,
            _collect_subtrees_by_type,
            subtree_crossover,
        )

        registry = _make_regime_registry()
        parent1 = _make_regime_model(
            registry,
            include_risk=True,
            gate_name="gate_a",
            detector_name="det_a",
            expert_names=("a1", "a2"),
        )
        parent2 = _make_regime_model(
            registry,
            include_risk=True,
            gate_name="gate_b",
            detector_name="det_b",
            expert_names=("b1", "b2"),
        )

        nodes1 = [node for node, _ in _collect_subtrees_by_type(parent1)]
        nodes2 = [node for node, _ in _collect_subtrees_by_type(parent2)]
        gate_idx_1 = nodes1.index(_find_node(parent1, "gate_a"))
        gate_idx_2 = nodes2.index(_find_node(parent2, "gate_b"))
        rng = _ScriptedRNG([gate_idx_1, gate_idx_2])
        telemetry = BlockConstraintTelemetry()

        child1, child2 = subtree_crossover(
            parent1,
            parent2,
            registry,
            max_depth=12,
            rng=rng,
            max_attempts=1,
            block_constraint_telemetry=telemetry,
        )

        assert child1 != parent1
        assert child2 != parent2
        assert telemetry.accepted["crossover:gate"] == 1
        assert telemetry.blocked == {}

    def test_subtree_mutation_preserves_regime_roles(self) -> None:
        from liq.gp.evolution.operators import (
            BlockConstraintTelemetry,
            _collect_subtrees_by_type,
            subtree_mutation,
        )

        registry = _make_regime_registry()
        parent = _make_regime_model(
            registry,
            include_risk=True,
            gate_name="gate_a",
            detector_name="det_a",
            expert_names=("a1", "a2"),
        )
        nodes = [node for node, _ in _collect_subtrees_by_type(parent)]
        det_idx = nodes.index(_find_node(parent, "det_a"))
        rng = _ScriptedRNG([det_idx, 0])
        telemetry = BlockConstraintTelemetry()

        mutant = subtree_mutation(
            parent,
            registry,
            max_depth=12,
            rng=rng,
            max_attempts=1,
            block_constraint_telemetry=telemetry,
        )

        assert mutant != parent
        assert telemetry.accepted["subtree_mutation:detector"] == 1
        assert telemetry.blocked == {}

    def test_point_mutation_preserves_block_context(self) -> None:
        from liq.gp.evolution.operators import (
            BlockConstraintTelemetry,
            _collect_subtrees_by_type,
            point_mutation,
        )

        registry = _make_regime_registry()
        parent = _make_regime_model(
            registry,
            include_risk=True,
            gate_name="gate_a",
            detector_name="det_a",
            expert_names=("a1", "a2"),
        )
        nodes = [node for node, _ in _collect_subtrees_by_type(parent)]
        expert_idx = nodes.index(_find_node(parent, "a1"))
        rng = _ScriptedRNG([expert_idx, 0])
        telemetry = BlockConstraintTelemetry()

        mutant = point_mutation(
            parent,
            registry,
            rng,
            max_attempts=1,
            block_constraint_telemetry=telemetry,
        )

        assert mutant != parent
        assert telemetry.accepted["point_mutation:expert:0"] == 1
        assert telemetry.blocked == {}
