"""Tests for constant optimization (FR-6)."""

from __future__ import annotations

import time

import numpy as np
import pytest

from liq.gp.config import FitnessConfig, GPConfig
from liq.gp.primitives.registry import PrimitiveInfo
from liq.gp.program.ast import (
    ConstantNode,
    FunctionNode,
    ParameterizedNode,
    Program,
    TerminalNode,
)
from liq.gp.program.constants import (
    extract_constants,
    inject_constants,
    optimize_constants,
    select_for_optimization,
)
from liq.gp.types import FitnessResult, ParamSpec, Series

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mul_info() -> PrimitiveInfo:
    return PrimitiveInfo(
        name="mul",
        category="numeric",
        arity=2,
        input_types=(Series, Series),
        output_type=Series,
        callable=lambda a, b: a * b,
    )


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


def _make_linear_program() -> FunctionNode:
    """add(mul(a, x), b) where a and b are constants."""
    x = TerminalNode(name="x", output_type=Series)
    a = ConstantNode(value=0.5)  # should optimize to ~2.0
    b = ConstantNode(value=0.0)  # should optimize to ~1.0
    mul_info = _make_mul_info()
    add_info = _make_add_info()
    mul_node = FunctionNode(primitive=mul_info, children=(a, x))
    return FunctionNode(primitive=add_info, children=(mul_node, b))


def _make_context() -> dict[str, np.ndarray]:
    """Context with x and target y = 2*x + 1."""
    rng = np.random.default_rng(42)
    x = rng.uniform(-5, 5, size=100)
    return {"x": x, "target": 2.0 * x + 1.0}


def _mse_evaluator(program: Program, context: dict[str, np.ndarray]) -> FitnessResult:
    """Evaluate fitness as negative MSE (higher is better)."""
    from liq.gp.program.eval import evaluate

    pred = evaluate(program, context)
    target = context["target"]
    mse = float(np.mean((pred - target) ** 2))
    return FitnessResult(objectives=(-mse,))


class SimpleFitnessEvaluator:
    """A simple fitness evaluator that wraps a single-program function."""

    def evaluate(
        self,
        programs: list[Program],
        context: dict[str, np.ndarray],
    ) -> list[FitnessResult]:
        return [_mse_evaluator(p, context) for p in programs]


# ---------------------------------------------------------------------------
# Tests: extract_constants
# ---------------------------------------------------------------------------


class TestExtractConstants:
    """FR-6.1: Extract ConstantNode values from a program tree."""

    def test_extract_from_linear_program(self) -> None:
        """Linear program add(mul(a, x), b) should yield [a, b]."""
        program = _make_linear_program()
        values = extract_constants(program)
        assert values == [0.5, 0.0]

    def test_extract_from_terminal(self) -> None:
        """A terminal node has no constants."""
        terminal = TerminalNode(name="x", output_type=Series)
        values = extract_constants(terminal)
        assert values == []

    def test_extract_from_constant_node(self) -> None:
        """A single constant node yields its value."""
        c = ConstantNode(value=3.14)
        values = extract_constants(c)
        assert values == [3.14]

    def test_extract_preserves_order(self) -> None:
        """Constants are extracted in depth-first left-to-right order."""
        add_info = _make_add_info()
        c1 = ConstantNode(value=1.0)
        c2 = ConstantNode(value=2.0)
        c3 = ConstantNode(value=3.0)
        # add(add(c1, c2), c3) -> constants should be [1, 2, 3]
        inner = FunctionNode(primitive=add_info, children=(c1, c2))
        outer = FunctionNode(primitive=add_info, children=(inner, c3))
        values = extract_constants(outer)
        assert values == [1.0, 2.0, 3.0]


# ---------------------------------------------------------------------------
# Tests: inject_constants
# ---------------------------------------------------------------------------


class TestInjectConstants:
    """FR-6.2: Inject constants into a new program tree (immutability)."""

    def test_inject_roundtrip(self) -> None:
        """extract then inject with same values produces equal tree."""
        program = _make_linear_program()
        values = extract_constants(program)
        new_program = inject_constants(program, values)
        assert new_program == program

    def test_inject_different_values(self) -> None:
        """Injecting different values produces a tree with new constants."""
        program = _make_linear_program()
        new_program = inject_constants(program, [2.0, 1.0])
        new_values = extract_constants(new_program)
        assert new_values == [2.0, 1.0]

    def test_inject_preserves_structure(self) -> None:
        """The injected tree has the same structure (function names, terminals)."""
        program = _make_linear_program()
        new_program = inject_constants(program, [10.0, 20.0])
        # Structure should still be add(mul(const, x), const)
        assert isinstance(new_program, FunctionNode)
        assert new_program.primitive.name == "add"
        inner = new_program.children[0]
        assert isinstance(inner, FunctionNode)
        assert inner.primitive.name == "mul"

    def test_inject_returns_new_tree(self) -> None:
        """The returned tree should be a new object (immutability)."""
        program = _make_linear_program()
        new_program = inject_constants(program, [2.0, 1.0])
        assert new_program is not program

    def test_inject_terminal_unchanged(self) -> None:
        """Injecting into a terminal returns the same terminal."""
        terminal = TerminalNode(name="x", output_type=Series)
        result = inject_constants(terminal, [])
        assert result == terminal

    def test_inject_parameterized_node(self) -> None:
        """inject_constants works through ParameterizedNode children."""
        ps = ParamSpec(name="period", dtype=int, default=20, min_value=2, max_value=200)
        highest_info = PrimitiveInfo(
            name="highest",
            category="indicator",
            arity=1,
            input_types=(Series,),
            output_type=Series,
            callable=lambda a, period=20: a,
            param_specs=[ps],
        )
        c = ConstantNode(value=5.0)
        tree = ParameterizedNode(
            primitive=highest_info, children=(c,), params={"period": 20}
        )
        new_tree = inject_constants(tree, [10.0])
        assert isinstance(new_tree, ParameterizedNode)
        child = new_tree.children[0]
        assert isinstance(child, ConstantNode)
        assert child.value == 10.0


# ---------------------------------------------------------------------------
# Tests: optimize_constants
# ---------------------------------------------------------------------------


class TestOptimizeConstants:
    """FR-6.3: Optimize constants via scipy."""

    def test_optimization_improves_fitness(self) -> None:
        """Optimization should bring constants closer to a=2, b=1."""
        program = _make_linear_program()
        context = _make_context()
        evaluator = SimpleFitnessEvaluator()
        config = GPConfig(constant_opt_enabled=True, constant_opt_max_iter=200)
        rng = np.random.default_rng(42)

        result = optimize_constants(program, evaluator, context, config, rng)
        optimized_values = extract_constants(result)

        # Should be close to a=2.0, b=1.0
        assert abs(optimized_values[0] - 2.0) < 0.1
        assert abs(optimized_values[1] - 1.0) < 0.1

    def test_zero_constant_program_unchanged(self) -> None:
        """A program with no constants should be returned unchanged."""
        # Just a terminal
        terminal = TerminalNode(name="x", output_type=Series)
        context = _make_context()
        evaluator = SimpleFitnessEvaluator()
        config = GPConfig(constant_opt_enabled=True)
        rng = np.random.default_rng(42)

        result = optimize_constants(terminal, evaluator, context, config, rng)
        assert result is terminal

    def test_optimization_returns_new_tree(self) -> None:
        """Optimized program should be a new tree object (immutability)."""
        program = _make_linear_program()
        context = _make_context()
        evaluator = SimpleFitnessEvaluator()
        config = GPConfig(constant_opt_enabled=True, constant_opt_max_iter=10)
        rng = np.random.default_rng(42)

        result = optimize_constants(program, evaluator, context, config, rng)
        assert result is not program

    def test_deterministic_with_seed(self) -> None:
        """Same seed should produce the same optimized result."""
        program = _make_linear_program()
        context = _make_context()
        evaluator = SimpleFitnessEvaluator()
        config = GPConfig(constant_opt_enabled=True, constant_opt_max_iter=50)

        rng1 = np.random.default_rng(123)
        result1 = optimize_constants(program, evaluator, context, config, rng1)

        rng2 = np.random.default_rng(123)
        result2 = optimize_constants(program, evaluator, context, config, rng2)

        values1 = extract_constants(result1)
        values2 = extract_constants(result2)
        assert values1 == values2

    def test_max_iter_respected(self) -> None:
        """Optimization should not exceed max_iter function evaluations."""
        program = _make_linear_program()
        context = _make_context()

        call_count = 0
        original_evaluator = SimpleFitnessEvaluator()

        class CountingEvaluator:
            def evaluate(
                self,
                programs: list[Program],
                context: dict[str, np.ndarray],
            ) -> list[FitnessResult]:
                nonlocal call_count
                call_count += len(programs)
                return original_evaluator.evaluate(programs, context)

        evaluator = CountingEvaluator()
        config = GPConfig(
            constant_opt_enabled=True,
            constant_opt_max_iter=5,
            constant_opt_max_time_seconds=60.0,
        )
        rng = np.random.default_rng(42)

        optimize_constants(program, evaluator, context, config, rng)
        # Nelder-Mead does multiple evaluations per iteration, but maxiter
        # limits total iterations. The call count should be bounded.
        # With maxiter=5 and 2 parameters, Nelder-Mead does at most
        # roughly (n+1) + 5*2 = 13 evaluations, but we just check it's bounded.
        assert call_count < 100

    def test_max_time_respected(self) -> None:
        """Optimization should stop when time limit is exceeded."""
        program = _make_linear_program()
        context = _make_context()

        call_count = 0

        class SlowEvaluator:
            def evaluate(
                self,
                programs: list[Program],
                context: dict[str, np.ndarray],
            ) -> list[FitnessResult]:
                nonlocal call_count
                call_count += 1
                time.sleep(0.02)
                return [FitnessResult(objectives=(-1.0,)) for _ in programs]

        evaluator = SlowEvaluator()
        config = GPConfig(
            constant_opt_enabled=True,
            constant_opt_max_iter=10000,
            constant_opt_max_time_seconds=0.1,
        )
        rng = np.random.default_rng(42)

        start = time.monotonic()
        optimize_constants(program, evaluator, context, config, rng)
        elapsed = time.monotonic() - start

        # Should finish reasonably close to the time limit.
        # The time check in the objective fires before each evaluation,
        # so we expect at most ~ceil(0.1/0.02) + 1 = 6 calls before
        # the time limit is detected, plus overhead.
        assert elapsed < 1.0
        # Verify that the optimizer did not run 10000 iterations
        assert call_count < 50

    def test_constant_opt_error_nonfatal(self) -> None:
        """ConstantOptError is non-fatal: original program returned."""
        program = _make_linear_program()
        context = _make_context()

        class FailingEvaluator:
            def evaluate(
                self,
                programs: list[Program],
                context: dict[str, np.ndarray],
            ) -> list[FitnessResult]:
                raise RuntimeError("Evaluation exploded")

        evaluator = FailingEvaluator()
        config = GPConfig(constant_opt_enabled=True)
        rng = np.random.default_rng(42)

        # Should not raise, should return original program
        result = optimize_constants(program, evaluator, context, config, rng)
        assert result == program

    def test_single_constant_program(self) -> None:
        """Optimization on a single-constant program should work."""
        # neg(const) -- optimize const so neg(const) ~= target
        # target = -3.0 constant, so const should approach 3.0
        neg_info = _make_neg_info()
        c = ConstantNode(value=0.0)
        program = FunctionNode(primitive=neg_info, children=(c,))
        x = np.linspace(-1, 1, 50)
        context = {"x": x, "target": np.full(50, -3.0)}
        evaluator = SimpleFitnessEvaluator()
        config = GPConfig(constant_opt_enabled=True, constant_opt_max_iter=200)
        rng = np.random.default_rng(42)

        result = optimize_constants(program, evaluator, context, config, rng)
        values = extract_constants(result)
        assert abs(values[0] - 3.0) < 0.2

    def test_minimize_primary_objective_direction(self) -> None:
        """Optimization respects primary objective direction='minimize'."""
        program = ConstantNode(value=0.0)
        context = {"x": np.linspace(-1.0, 1.0, 20), "target": np.full(20, 2.0)}

        class MinimizeMSEEvaluator:
            def evaluate(
                self,
                programs: list[Program],
                context: dict[str, np.ndarray],
            ) -> list[FitnessResult]:
                from liq.gp.program.eval import evaluate

                target = context["target"]
                results: list[FitnessResult] = []
                for prog in programs:
                    pred = evaluate(prog, context)
                    mse = float(np.mean((pred - target) ** 2))
                    results.append(FitnessResult(objectives=(mse,)))
                return results

        evaluator = MinimizeMSEEvaluator()
        config = GPConfig(
            constant_opt_enabled=True,
            constant_opt_max_iter=80,
            fitness=FitnessConfig(
                objectives=["loss"],
                objective_directions=["minimize"],
            ),
        )
        rng = np.random.default_rng(7)

        baseline = evaluator.evaluate([program], context)[0].objectives[0]
        optimized = optimize_constants(program, evaluator, context, config, rng)
        improved = evaluator.evaluate([optimized], context)[0].objectives[0]
        assert improved <= baseline


# ---------------------------------------------------------------------------
# Tests: select_for_optimization
# ---------------------------------------------------------------------------


class TestSelectForOptimization:
    """FR-6.4: Select top-K fraction of population for optimization."""

    def test_basic_selection(self) -> None:
        """Top 10% of 100 programs -> 10 indices."""
        population = [_make_linear_program() for _ in range(100)]
        # Fitness values: 0..99, first objective higher is better
        fitnesses = [FitnessResult(objectives=(float(i),)) for i in range(100)]
        config = GPConfig(constant_opt_top_k=0.1)

        indices = select_for_optimization(population, fitnesses, config)
        assert len(indices) == 10
        # Should be the top 10 indices (90..99)
        assert set(indices) == set(range(90, 100))

    def test_all_selected_when_top_k_is_one(self) -> None:
        """top_k=1.0 should select all."""
        population = [_make_linear_program() for _ in range(20)]
        fitnesses = [FitnessResult(objectives=(float(i),)) for i in range(20)]
        config = GPConfig(constant_opt_top_k=1.0)

        indices = select_for_optimization(population, fitnesses, config)
        assert len(indices) == 20

    def test_at_least_one_selected(self) -> None:
        """Even with tiny population and fraction, at least 1 is selected."""
        population = [_make_linear_program() for _ in range(10)]
        fitnesses = [FitnessResult(objectives=(float(i),)) for i in range(10)]
        config = GPConfig(constant_opt_top_k=0.05)

        indices = select_for_optimization(population, fitnesses, config)
        assert len(indices) >= 1

    def test_zero_constant_programs_excluded(self) -> None:
        """Programs with no constants should be excluded from selection."""
        terminal = TerminalNode(name="x", output_type=Series)
        linear = _make_linear_program()
        population: list[Program] = [terminal, linear]
        fitnesses = [
            FitnessResult(objectives=(10.0,)),  # terminal, no constants
            FitnessResult(objectives=(5.0,)),  # linear, has constants
        ]
        config = GPConfig(constant_opt_top_k=1.0)

        indices = select_for_optimization(population, fitnesses, config)
        # Only the linear program should be selected
        assert indices == [1]

    def test_selection_returns_sorted_indices(self) -> None:
        """Returned indices should be in ascending order."""
        population = [_make_linear_program() for _ in range(50)]
        fitnesses = [FitnessResult(objectives=(float(i),)) for i in range(50)]
        config = GPConfig(constant_opt_top_k=0.2)

        indices = select_for_optimization(population, fitnesses, config)
        assert indices == sorted(indices)

    def test_no_eligible_programs_returns_empty(self) -> None:
        """If all programs have zero constants, return empty list."""
        terminal = TerminalNode(name="x", output_type=Series)
        population: list[Program] = [terminal, terminal]
        fitnesses = [
            FitnessResult(objectives=(1.0,)),
            FitnessResult(objectives=(2.0,)),
        ]
        config = GPConfig(constant_opt_top_k=1.0)

        indices = select_for_optimization(population, fitnesses, config)
        assert indices == []

    def test_selection_respects_minimize_direction(self) -> None:
        """For minimize direction, lower objective values are selected first."""
        population = [_make_linear_program() for _ in range(5)]
        fitnesses = [
            FitnessResult(objectives=(5.0,)),
            FitnessResult(objectives=(4.0,)),
            FitnessResult(objectives=(3.0,)),
            FitnessResult(objectives=(2.0,)),
            FitnessResult(objectives=(1.0,)),
        ]
        config = GPConfig(
            constant_opt_top_k=0.4,
            fitness=FitnessConfig(
                objectives=["loss"],
                objective_directions=["minimize"],
            ),
        )

        indices = select_for_optimization(population, fitnesses, config)
        assert indices == [3, 4]


# ---------------------------------------------------------------------------
# Tests: edge cases and integration
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Additional edge cases for constant optimization."""

    def test_inject_mismatched_length_raises(self) -> None:
        """Injecting wrong number of constants should raise ValueError."""
        program = _make_linear_program()
        with pytest.raises(ValueError, match="mismatch"):
            inject_constants(program, [1.0])  # needs 2, got 1

    def test_extract_inject_roundtrip_preserves_types(self) -> None:
        """After extract + inject, output_type should be preserved."""
        program = _make_linear_program()
        values = extract_constants(program)
        new_program = inject_constants(program, values)
        assert new_program.output_type == program.output_type

    def test_optimizer_with_nan_fitness_nonfatal(self) -> None:
        """If evaluator returns NaN fitness, optimization is non-fatal."""
        program = _make_linear_program()
        context = _make_context()

        class NanEvaluator:
            def evaluate(
                self,
                programs: list[Program],
                context: dict[str, np.ndarray],
            ) -> list[FitnessResult]:
                return [FitnessResult(objectives=(float("nan"),)) for _ in programs]

        evaluator = NanEvaluator()
        config = GPConfig(constant_opt_enabled=True, constant_opt_max_iter=5)
        rng = np.random.default_rng(42)

        # Should not crash -- returns original or partially optimized
        result = optimize_constants(program, evaluator, context, config, rng)
        assert isinstance(result, FunctionNode)
