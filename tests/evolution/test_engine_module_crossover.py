"""Stage-16 engine coverage for module-preserving crossover mode."""

from __future__ import annotations

import numpy as np

from liq.gp.config import GPConfig
from liq.gp.evolution.engine import evolve
from liq.gp.primitives.registry import PrimitiveRegistry
from liq.gp.types import FitnessResult, Series


class _ConstantFitnessEvaluator:
    def evaluate(self, programs, _context):
        return [FitnessResult(objectives=(1.0,), metadata={}) for _ in programs]


def _registry() -> PrimitiveRegistry:
    registry = PrimitiveRegistry()
    registry.register("close", lambda: None, input_types=(), output_type=Series)
    registry.register("volume", lambda: None, input_types=(), output_type=Series)
    registry.register(
        "add",
        lambda a, b: a + b,
        input_types=(Series, Series),
        output_type=Series,
    )
    return registry


def test_engine_uses_module_preserving_crossover_mode(monkeypatch) -> None:
    import liq.gp.evolution.engine as engine_module

    calls = {"count": 0}
    real = engine_module.module_preserving_crossover

    def _wrapped(*args, **kwargs):
        calls["count"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(engine_module, "module_preserving_crossover", _wrapped)

    config = GPConfig(
        population_size=10,
        max_depth=3,
        generations=2,
        tournament_size=2,
        elitism_count=1,
        crossover_rate=1.0,
        subtree_mutation_rate=0.0,
        point_mutation_rate=0.0,
        parameter_mutation_rate=0.0,
        hoist_mutation_rate=0.0,
        crossover_mode="module_preserving",
    )
    context = {
        "close": np.ones(32),
        "volume": np.ones(32),
    }

    evolve(_registry(), config, _ConstantFitnessEvaluator(), context)
    assert calls["count"] > 0
