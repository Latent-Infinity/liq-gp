"""Configurable search-strategy options that survived ablation:
grow terminal probability and constant-optimisation budgeting.

Every option defaults to the historical behaviour; these tests pin both the
default and the alternative.
"""

from __future__ import annotations

import numpy as np
import pytest

from liq.gp.config import GPConfig
from liq.gp.errors import ConfigurationError
from liq.gp.evolution.init import generate_grow
from liq.gp.primitives.registry import PrimitiveRegistry
from liq.gp.program.ast import (
    FunctionNode,
    ParameterizedNode,
    Program,
    TerminalNode,
)
from liq.gp.types import Series


def _registry() -> PrimitiveRegistry:
    reg = PrimitiveRegistry()
    reg.register("x", lambda: None, input_types=(), output_type=Series)
    reg.register("y", lambda: None, input_types=(), output_type=Series)
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


def _is_internal(node: Program) -> bool:
    return isinstance(node, (FunctionNode, ParameterizedNode))


def _nodes(tree: Program) -> list[Program]:
    out = [tree]
    if _is_internal(tree):
        for child in tree.children:  # type: ignore[union-attr]
            out.extend(_nodes(child))
    return out


def _deep_tree(reg: PrimitiveRegistry, depth: int) -> Program:
    """Left-spine tree of the requested depth with leaves at every level."""
    node: Program = TerminalNode("x", Series)
    for _ in range(depth):
        node = FunctionNode(reg.get("add"), (node, TerminalNode("y", Series)))
    return node


# ---------------------------------------------------------------------------
# Configuration validation
# ---------------------------------------------------------------------------


class TestConfigDefaultsAndValidation:
    def test_defaults_preserve_historical_behaviour(self) -> None:
        config = GPConfig()
        assert config.grow_terminal_probability == 0.3
        assert config.constant_opt_interval == 1
        assert config.constant_opt_max_program_evals is None

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"grow_terminal_probability": -0.1}, "grow_terminal_probability"),
            ({"grow_terminal_probability": 1.5}, "grow_terminal_probability"),
            ({"constant_opt_interval": 0}, "constant_opt_interval"),
            ({"constant_opt_max_program_evals": 0}, "constant_opt_max_program_evals"),
        ],
    )
    def test_invalid_values_rejected(self, kwargs: dict, message: str) -> None:
        with pytest.raises(ConfigurationError, match=message):
            GPConfig(**kwargs)

    def test_proportional_grow_probability_accepted(self) -> None:
        assert (
            GPConfig(grow_terminal_probability=None).grow_terminal_probability is None
        )


# ---------------------------------------------------------------------------
# Grow terminal probability
# ---------------------------------------------------------------------------


class TestGrowTerminalProbability:
    def _mean_size(self, probability: float | None, seed: int) -> float:
        reg = _registry()
        rng = np.random.default_rng(seed)
        sizes = [
            generate_grow(reg, 5, Series, rng, terminal_probability=probability).size
            for _ in range(300)
        ]
        return float(np.mean(sizes))

    def test_higher_terminal_probability_yields_smaller_trees(self) -> None:
        assert self._mean_size(0.7, seed=1) < self._mean_size(0.1, seed=1)

    def test_proportional_mode_uses_registry_composition(self) -> None:
        # 2 terminals, 3 functions -> P(terminal) = 0.4, between the extremes.
        proportional = self._mean_size(None, seed=2)
        assert (
            self._mean_size(0.7, seed=2) < proportional < self._mean_size(0.1, seed=2)
        )

    def test_default_matches_explicit_historical_value(self) -> None:
        reg = _registry()
        explicit = generate_grow(
            reg, 5, Series, np.random.default_rng(11), terminal_probability=0.3
        )
        default = generate_grow(reg, 5, Series, np.random.default_rng(11))
        assert explicit == default
