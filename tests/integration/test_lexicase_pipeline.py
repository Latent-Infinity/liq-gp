"""Integration smoke tests for lexicase evolution pipeline."""

from __future__ import annotations

from typing import Any

import numpy as np

from liq.gp.config import FitnessConfig, GPConfig
from liq.gp.evolution.engine import evolve
from liq.gp.primitives.registry import PrimitiveRegistry
from liq.gp.types import FitnessResult, Series


class _ToyLexicaseEvaluator:
    def __init__(self) -> None:
        self.calls = 0

    def evaluate(
        self,
        programs: list[Any],
        context: dict[str, np.ndarray],  # noqa: ARG002
    ) -> list[FitnessResult]:
        del context
        self.calls += len(programs)
        results: list[FitnessResult] = []
        for program in programs:
            size = float(program.size)
            case_1 = size + 0.1
            case_2 = 10.0 / (size + 1.0)
            results.append(
                FitnessResult(
                    objectives=(size,),
                    metadata={
                        "slice_scores": {
                            "time_window:split_0:train:cagr": case_1,
                            "event:liquidity:high": case_2,
                        },
                        "raw_objectives": (case_1, case_2),
                    },
                )
            )
        return results


def _make_registry() -> PrimitiveRegistry:
    reg = PrimitiveRegistry()
    reg.register("x", lambda: None, output_type=Series)
    reg.register(
        "add",
        lambda a, b: a + b,
        category="arithmetic",
        input_types=(Series, Series),
        output_type=Series,
    )
    reg.register(
        "neg",
        lambda x: -x,
        category="arithmetic",
        input_types=(Series,),
        output_type=Series,
    )
    reg.register(
        "sub",
        lambda a, b: a - b,
        category="arithmetic",
        input_types=(Series, Series),
        output_type=Series,
    )
    reg.register(
        "mul",
        lambda a, b: a * b,
        category="arithmetic",
        input_types=(Series, Series),
        output_type=Series,
    )
    return reg


def _make_lexicase_config(seed: int) -> GPConfig:
    return GPConfig(
        population_size=24,
        max_depth=4,
        generations=3,
        seed=seed,
        tournament_size=3,
        elitism_count=2,
        selection_mode="lexicase",
        lexicase_downsample_policy="random",
        lexicase_downsample_cases=2,
        parsimony_mode="disabled",
        constant_opt_enabled=False,
        simplification_enabled=False,
        fitness=FitnessConfig(
            objectives=["fitness"], objective_directions=["maximize"]
        ),
    )


class _Builder:
    """Minimal helper to prevent constant recomputation."""

    registry = _make_registry()


def test_lexicase_evolution_smoke_and_deterministic_replay() -> None:
    registry = _Builder.registry
    config = _make_lexicase_config(seed=314)
    context = {"x": np.linspace(-1.0, 1.0, 120)}

    evaluator_1 = _ToyLexicaseEvaluator()
    result_1 = evolve(
        registry=registry, config=config, evaluator=evaluator_1, context=context
    )

    evaluator_2 = _ToyLexicaseEvaluator()
    result_2 = evolve(
        registry=registry, config=config, evaluator=evaluator_2, context=context
    )

    assert result_1.best_program == result_2.best_program
    assert result_1.fitness_history == result_2.fitness_history
    assert len(result_1.pareto_front) >= 1
    assert evaluator_1.calls > 0
    assert evaluator_2.calls > 0
