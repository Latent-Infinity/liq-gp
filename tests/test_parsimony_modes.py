"""Parsimony pressure must penalise size regardless of objective direction."""

from __future__ import annotations

import pytest

from liq.gp.config import FitnessConfig, GPConfig
from liq.gp.evolution.constraints import apply_parsimony
from liq.gp.primitives.registry import PrimitiveRegistry
from liq.gp.program.ast import FunctionNode, Program, TerminalNode
from liq.gp.types import FitnessResult, Series


def _programs() -> list[Program]:
    reg = PrimitiveRegistry()
    reg.register("x", lambda: None, input_types=(), output_type=Series)
    reg.register(
        "add",
        lambda a, b: a + b,
        input_types=(Series, Series),
        output_type=Series,
    )
    small: Program = TerminalNode("x", Series)
    big: Program = FunctionNode(reg.get("add"), (small, small))
    return [small, big]


@pytest.mark.parametrize("direction", ["maximize", "minimize"])
def test_linear_parsimony_penalises_larger_programs(direction: str) -> None:
    config = GPConfig(
        parsimony_mode="linear",
        parsimony_coefficient=0.5,
        constant_opt_enabled=False,
        fitness=FitnessConfig(objectives=["f"], objective_directions=[direction]),
    )
    small, big = _programs()
    # Identical raw fitness: the smaller program must come out strictly better.
    raw = [FitnessResult(objectives=(1.0,)), FitnessResult(objectives=(1.0,))]

    adjusted = apply_parsimony(raw, [small, big], config)
    small_value, big_value = (fr.objectives[0] for fr in adjusted)

    if direction == "maximize":
        assert small_value > big_value
        assert small_value == pytest.approx(1.0 - 0.5 * small.size)
    else:
        assert small_value < big_value
        assert small_value == pytest.approx(1.0 + 0.5 * small.size)

    for fr in adjusted:
        assert fr.metadata["raw_objectives"] == (1.0,)


def test_linear_parsimony_leaves_secondary_objectives_untouched() -> None:
    config = GPConfig(
        parsimony_mode="linear",
        parsimony_coefficient=0.1,
        constant_opt_enabled=False,
        fitness=FitnessConfig(
            objectives=["f", "g"], objective_directions=["minimize", "maximize"]
        ),
    )
    small, _ = _programs()
    adjusted = apply_parsimony([FitnessResult(objectives=(2.0, 7.0))], [small], config)
    assert adjusted[0].objectives[1] == 7.0
