"""Tests for deterministic objective-vector adapter boundary."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from liq.gp.config import GPConfig
from liq.gp.evolution.engine import evolve
from liq.gp.evolution.objective_adapter import (
    OBJECTIVE_ADAPTER_VERSION,
    ObjectiveVectorAdapter,
)
from liq.gp.primitives.registry import PrimitiveRegistry
from liq.gp.types import FitnessResult, Series


class _VectorEvaluator:
    def __init__(self, results: list[FitnessResult]) -> None:
        self._results = results

    def evaluate(
        self,
        programs: list[Any],
        context: dict[str, np.ndarray],
    ) -> list[FitnessResult]:
        del context
        return self._results[: len(programs)]


class _FitnessOnlyVectorEvaluator:
    def __init__(self, results: list[FitnessResult]) -> None:
        self._results = results

    def evaluate_fitness(
        self,
        programs: list[Any],
        context: dict[str, np.ndarray],
    ) -> list[FitnessResult]:
        del context
        return self._results[: len(programs)]


def _small_registry() -> PrimitiveRegistry:
    reg = PrimitiveRegistry()
    reg.register(
        name="close",
        callable=lambda close: close,
        category="source",
        input_types=(),
        output_type=Series,
        arity=0,
    )
    reg.register(
        name="identity",
        callable=lambda x: x,
        category="math",
        input_types=(Series,),
        output_type=Series,
        arity=1,
    )
    return reg


def _small_config() -> GPConfig:
    return GPConfig(
        population_size=10,
        max_depth=3,
        generations=1,
        seed=7,
        tournament_size=2,
        elitism_count=1,
        selection_mode="tournament",
        constant_opt_enabled=False,
        simplification_enabled=False,
    )


class TestObjectiveVectorAdapterContract:
    def test_vector_shape_and_versioned_metadata(self) -> None:
        adapter = ObjectiveVectorAdapter(
            _VectorEvaluator([FitnessResult(objectives=(0.10, 0.20, 0.30), metadata={})]),
            expected_objective_count=3,
            objective_directions=("maximize", "minimize", "minimize"),
        )
        [result] = adapter.evaluate(["p0"], {"close": np.array([1.0])})
        payload = result.metadata["objective_adapter"]
        assert payload["version"] == OBJECTIVE_ADAPTER_VERSION
        assert payload["expected_objective_count"] == 3
        assert payload["raw_objectives"] == (0.10, 0.20, 0.30)

    def test_invalid_shape_routes_to_penalty(self) -> None:
        adapter = ObjectiveVectorAdapter(
            _VectorEvaluator([FitnessResult(objectives=(0.10, 0.20), metadata={})]),
            expected_objective_count=3,
            objective_directions=("maximize", "minimize", "minimize"),
        )
        [result] = adapter.evaluate(["p0"], {"close": np.array([1.0])})
        assert result.objectives[0] < 0
        assert result.metadata["reason_code"] == "invalid_objective_shape"

    def test_non_finite_objective_routes_to_penalty(self) -> None:
        adapter = ObjectiveVectorAdapter(
            _VectorEvaluator(
                [FitnessResult(objectives=(0.10, float("nan"), 0.30), metadata={})]
            ),
            expected_objective_count=3,
            objective_directions=("maximize", "minimize", "minimize"),
        )
        [result] = adapter.evaluate(["p0"], {"close": np.array([1.0])})
        assert result.objectives[0] < 0
        assert result.metadata["reason_code"] == "non_finite_objective"

    def test_reduction_is_deterministic(self) -> None:
        results = [
            FitnessResult(objectives=(0.11, 0.04, 0.20), metadata={}),
            FitnessResult(objectives=(0.12, 0.05, 0.18), metadata={}),
        ]
        adapter = ObjectiveVectorAdapter(
            _VectorEvaluator(results),
            expected_objective_count=3,
            objective_directions=("maximize", "minimize", "minimize"),
            reduction_weights=(0.5, 0.3, 0.2),
        )
        first = adapter.evaluate(["p0", "p1"], {"close": np.array([1.0])})
        second = adapter.evaluate(["p0", "p1"], {"close": np.array([1.0])})
        assert [x.objectives for x in first] == [x.objectives for x in second]

    def test_evaluate_fitness_alias_path(self) -> None:
        adapter = ObjectiveVectorAdapter(
            _FitnessOnlyVectorEvaluator(
                [FitnessResult(objectives=(0.10, 0.02, 0.15), metadata={})]
            ),
            expected_objective_count=3,
            objective_directions=("maximize", "minimize", "minimize"),
        )
        [result] = adapter.evaluate_fitness(["p0"], {"close": np.array([1.0])})
        assert result.metadata["reason_code"] == "ok"

    def test_callable_evaluator_path(self) -> None:
        def _callable(
            programs: list[Any], context: dict[str, np.ndarray]
        ) -> list[FitnessResult]:
            del context
            return [FitnessResult(objectives=(0.10, 0.02, 0.15), metadata={}) for _ in programs]

        adapter = ObjectiveVectorAdapter(
            _callable,
            expected_objective_count=3,
            objective_directions=("maximize", "minimize", "minimize"),
        )
        [result] = adapter.evaluate(["p0"], {"close": np.array([1.0])})
        assert result.metadata["reason_code"] == "ok"

    def test_mismatched_result_count_raises(self) -> None:
        adapter = ObjectiveVectorAdapter(
            _VectorEvaluator([]),
            expected_objective_count=3,
            objective_directions=("maximize", "minimize", "minimize"),
        )
        with pytest.raises(ValueError, match="mismatched result count"):
            adapter.evaluate(["p0"], {"close": np.array([1.0])})

    def test_invalid_wrapped_evaluator_type_raises(self) -> None:
        adapter = ObjectiveVectorAdapter(
            object(),
            expected_objective_count=3,
            objective_directions=("maximize", "minimize", "minimize"),
        )
        with pytest.raises(TypeError, match="wrapped evaluator must provide"):
            adapter.evaluate(["p0"], {"close": np.array([1.0])})


class TestObjectiveVectorAdapterEvolutionIntegration:
    def test_engine_accepts_adapter_reduced_scores(self) -> None:
        reg = _small_registry()
        config = _small_config()
        context = {"close": np.linspace(1.0, 2.0, 16)}

        vector_eval = _VectorEvaluator(
            [FitnessResult(objectives=(0.10, 0.02, 0.15), metadata={})] * 10
        )
        adapter = ObjectiveVectorAdapter(
            vector_eval,
            expected_objective_count=3,
            objective_directions=("maximize", "minimize", "minimize"),
        )

        result = evolve(reg, config, adapter, context)
        assert result.best_program is not None

    def test_init_validation_errors(self) -> None:
        with pytest.raises(ValueError, match="expected_objective_count"):
            ObjectiveVectorAdapter(
                _VectorEvaluator([]),
                expected_objective_count=0,
                objective_directions=(),
            )
        with pytest.raises(ValueError, match="objective_directions must match"):
            ObjectiveVectorAdapter(
                _VectorEvaluator([]),
                expected_objective_count=2,
                objective_directions=("maximize",),
            )
        with pytest.raises(ValueError, match="objective_directions must contain only"):
            ObjectiveVectorAdapter(
                _VectorEvaluator([]),
                expected_objective_count=1,
                objective_directions=("invalid",),
            )
        with pytest.raises(ValueError, match="reduction_weights must match"):
            ObjectiveVectorAdapter(
                _VectorEvaluator([]),
                expected_objective_count=2,
                objective_directions=("maximize", "minimize"),
                reduction_weights=(1.0,),
            )
        with pytest.raises(ValueError, match="reduction_weights must be >="):
            ObjectiveVectorAdapter(
                _VectorEvaluator([]),
                expected_objective_count=1,
                objective_directions=("maximize",),
                reduction_weights=(-0.1,),
            )
