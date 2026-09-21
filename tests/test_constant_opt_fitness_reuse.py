"""The optimizer already scored the constants it returns; the engine must
reuse that fitness rather than paying for a duplicate evaluation.
"""

from __future__ import annotations

import numpy as np

from liq.gp.config import GPConfig
from liq.gp.evolution.engine import evolve
from liq.gp.primitives.registry import PrimitiveRegistry
from liq.gp.program.ast import ConstantNode, FunctionNode, Program, TerminalNode
from liq.gp.program.constants import (
    extract_constants,
    optimize_constants,
    optimize_constants_with_fitness,
)
from liq.gp.program.eval import evaluate
from liq.gp.types import FitnessResult, Series


def _registry() -> PrimitiveRegistry:
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
    return reg


class RecordingEvaluator:
    """Scores ``(x + c)`` against ``x + 3`` and counts every program scored."""

    def __init__(self) -> None:
        self.calls = 0
        self.programs_scored = 0

    def evaluate(
        self, programs: list[Program], context: dict[str, np.ndarray]
    ) -> list[FitnessResult]:
        self.calls += 1
        self.programs_scored += len(programs)
        target = context["x"] + 3.0
        out = []
        for program in programs:
            pred = evaluate(program, context)
            mse = float(np.mean((pred - target) ** 2))
            if not np.isfinite(mse):
                mse = 1e12
            out.append(
                FitnessResult(objectives=(-mse,), metadata={"tag": "from-optimizer"})
            )
        return out


def _program(reg: PrimitiveRegistry, value: float) -> Program:
    return FunctionNode(
        reg.get("add"), (TerminalNode("x", Series), ConstantNode(value))
    )


def _context() -> dict[str, np.ndarray]:
    return {"x": np.linspace(-1.0, 1.0, 32)}


class TestOptimizeConstantsWithFitness:
    def test_returns_fitness_matching_the_returned_program(self) -> None:
        reg = _registry()
        ctx = _context()
        config = GPConfig(constant_opt_max_iter=60)
        evaluator = RecordingEvaluator()

        optimized, fitness = optimize_constants_with_fitness(
            _program(reg, 0.0), evaluator, ctx, config, np.random.default_rng(0)
        )
        assert fitness is not None
        # Constants improved, and the reported fitness is exactly what a fresh
        # evaluation of the returned program produces.
        assert abs(extract_constants(optimized)[0] - 3.0) < 0.1
        fresh = evaluator.evaluate([optimized], ctx)[0]
        assert fitness.objectives == fresh.objectives

    def test_no_constants_reports_no_fitness(self) -> None:
        program = TerminalNode("x", Series)
        optimized, fitness = optimize_constants_with_fitness(
            program,
            RecordingEvaluator(),
            _context(),
            GPConfig(),
            np.random.default_rng(0),
        )
        assert optimized is program
        assert fitness is None

    def test_unchanged_program_reports_no_fitness(self) -> None:
        reg = _registry()
        ctx = _context()
        # Already optimal: optimisation cannot beat the baseline.
        program = _program(reg, 3.0)
        optimized, fitness = optimize_constants_with_fitness(
            program,
            RecordingEvaluator(),
            ctx,
            GPConfig(constant_opt_max_iter=5),
            np.random.default_rng(0),
        )
        assert optimized is program
        assert fitness is None

    def test_plain_wrapper_still_returns_a_program(self) -> None:
        reg = _registry()
        result = optimize_constants(
            _program(reg, 0.0),
            RecordingEvaluator(),
            _context(),
            GPConfig(constant_opt_max_iter=30),
            np.random.default_rng(0),
        )
        assert isinstance(result, FunctionNode)


class TestEngineSkipsRedundantRefresh:
    def test_no_refresh_batch_after_constant_optimization(self) -> None:
        """Every optimized program's fitness comes from the optimizer itself."""
        reg = _registry()
        ctx = _context()
        evaluator = RecordingEvaluator()
        config = GPConfig(
            population_size=20,
            generations=4,
            constant_opt_enabled=True,
            constant_opt_top_k=0.5,
            constant_opt_max_iter=10,
            seed=5,
        )

        batch_sizes: list[int] = []
        original = evaluator.evaluate

        def spy(programs, context):
            batch_sizes.append(len(programs))
            return original(programs, context)

        evaluator.evaluate = spy  # type: ignore[method-assign]
        evolve(reg, config, evaluator, ctx)

        # Population passes are size 20; constant-optimisation probes are
        # size 1.  A refresh batch would be an intermediate size.
        assert set(batch_sizes) <= {1, 20}
        assert batch_sizes.count(20) == config.generations
