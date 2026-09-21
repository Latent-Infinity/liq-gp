"""Constant optimisation must never hand back a worse program than it received."""

from __future__ import annotations

import numpy as np

from liq.gp.config import FitnessConfig, GPConfig
from liq.gp.primitives.registry import PrimitiveRegistry
from liq.gp.program.ast import ConstantNode, FunctionNode, Program, TerminalNode
from liq.gp.program.constants import extract_constants, optimize_constants
from liq.gp.program.eval import evaluate
from liq.gp.types import FitnessResult, Series


def _registry() -> PrimitiveRegistry:
    reg = PrimitiveRegistry()
    reg.register("x", lambda: None, input_types=(), output_type=Series)
    reg.register(
        "gt",
        lambda a, b: (a > b).astype(float),
        input_types=(Series, Series),
        output_type=Series,
    )
    return reg


class StepEvaluator:
    """Piecewise-constant objective: gradient-free and hostile to perturbation.

    Fitness is the fraction of rows where ``x > threshold`` agrees with the
    label ``x > 0.5``.  Any threshold in ``(0.4, 0.5]`` is optimal; a small
    positive perturbation past ``0.5`` immediately loses one row.
    """

    def __init__(self) -> None:
        self.calls = 0

    def evaluate(
        self,
        programs: list[Program],
        context: dict[str, np.ndarray],
    ) -> list[FitnessResult]:
        self.calls += len(programs)
        label = (context["x"] > 0.5).astype(float)
        out: list[FitnessResult] = []
        for program in programs:
            pred = evaluate(program, context)
            out.append(FitnessResult(objectives=(float(np.mean(pred == label)),)))
        return out


def _fitness(program: Program, evaluator: StepEvaluator, ctx) -> float:
    return evaluator.evaluate([program], ctx)[0].objectives[0]


def test_optimized_program_is_never_worse_than_original() -> None:
    reg = _registry()
    # Grid that has a sample exactly at 0.5 so that thresholds > 0.5 mislabel it.
    ctx = {"x": np.linspace(0.0, 1.0, 21)}
    program = FunctionNode(
        reg.get("gt"), (TerminalNode("x", Series), ConstantNode(0.5))
    )
    config = GPConfig(
        constant_opt_max_iter=20,
        fitness=FitnessConfig(objectives=["acc"], objective_directions=["maximize"]),
    )
    evaluator = StepEvaluator()
    before = _fitness(program, evaluator, ctx)

    worst_seen = before
    for seed in range(25):
        optimized = optimize_constants(
            program, evaluator, ctx, config, np.random.default_rng(seed)
        )
        worst_seen = min(worst_seen, _fitness(optimized, evaluator, ctx))
    assert worst_seen >= before


def test_optimization_still_improves_when_possible() -> None:
    reg = _registry()
    ctx = {"x": np.linspace(0.0, 1.0, 101)}
    # Start far from the optimum; a smooth-ish objective gives the optimiser
    # room to move the threshold toward 0.5.
    program = FunctionNode(
        reg.get("gt"), (TerminalNode("x", Series), ConstantNode(0.0))
    )
    config = GPConfig(constant_opt_max_iter=200)

    class SmoothEvaluator:
        def evaluate(self, programs, context):
            results = []
            for p in programs:
                threshold = extract_constants(p)[0]
                results.append(FitnessResult(objectives=(-((threshold - 0.5) ** 2),)))
            return results

    optimized = optimize_constants(
        program, SmoothEvaluator(), ctx, config, np.random.default_rng(0)
    )
    assert abs(extract_constants(optimized)[0] - 0.5) < 0.05
