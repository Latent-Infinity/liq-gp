"""Numeric-contract tests for the objective-vector adapter reduction."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from liq.gp.evolution.objective_adapter import ObjectiveVectorAdapter
from liq.gp.types import FitnessResult

_CONTEXT: dict[str, np.ndarray] = {"close": np.array([1.0])}


def _evaluator(results: list[FitnessResult]) -> Any:
    def _call(
        programs: list[Any],
        context: dict[str, np.ndarray],
    ) -> list[FitnessResult]:
        del context
        return results[: len(programs)]

    return _call


class TestObjectiveVectorAdapterReduction:
    """The scalar reduction is a signed, weighted dot product."""

    def test_weighted_reduction_value(self) -> None:
        adapter = ObjectiveVectorAdapter(
            _evaluator([FitnessResult(objectives=(0.11, 0.04, 0.20), metadata={})]),
            expected_objective_count=3,
            objective_directions=("maximize", "minimize", "minimize"),
            reduction_weights=(0.5, 0.3, 0.2),
        )
        [result] = adapter.evaluate(["p0"], _CONTEXT)
        expected = 0.5 * 0.11 - 0.3 * 0.04 - 0.2 * 0.20
        assert result.objectives == pytest.approx((expected,))
        payload = result.metadata["objective_adapter"]
        assert payload["reduced_score"] == pytest.approx(expected)
        assert payload["reduction_weights"] == (0.5, 0.3, 0.2)

    def test_default_weights_are_unit_weights(self) -> None:
        adapter = ObjectiveVectorAdapter(
            _evaluator([FitnessResult(objectives=(2.0, 0.5), metadata={})]),
            expected_objective_count=2,
            objective_directions=("maximize", "minimize"),
        )
        [result] = adapter.evaluate(["p0"], _CONTEXT)
        assert result.objectives == pytest.approx((1.5,))
        assert result.metadata["objective_adapter"]["reduction_weights"] == (1.0, 1.0)

    def test_zero_weight_dimension_is_ignored(self) -> None:
        adapter = ObjectiveVectorAdapter(
            _evaluator([FitnessResult(objectives=(3.0, 99.0), metadata={})]),
            expected_objective_count=2,
            objective_directions=("maximize", "minimize"),
            reduction_weights=(1.0, 0.0),
        )
        [result] = adapter.evaluate(["p0"], _CONTEXT)
        assert result.objectives == pytest.approx((3.0,))

    def test_existing_metadata_is_preserved(self) -> None:
        adapter = ObjectiveVectorAdapter(
            _evaluator(
                [FitnessResult(objectives=(1.0,), metadata={"trace_id": "abc"})]
            ),
            expected_objective_count=1,
            objective_directions=("maximize",),
        )
        [result] = adapter.evaluate(["p0"], _CONTEXT)
        assert result.metadata["trace_id"] == "abc"
        assert result.metadata["reason_code"] == "ok"


class TestObjectiveVectorAdapterPenalty:
    """Invalid vectors collapse to the configured penalty value."""

    def test_custom_penalty_used_for_shape_mismatch(self) -> None:
        adapter = ObjectiveVectorAdapter(
            _evaluator([FitnessResult(objectives=(1.0, 2.0), metadata={})]),
            expected_objective_count=3,
            objective_directions=("maximize", "maximize", "maximize"),
            invalid_penalty=-5.0,
        )
        [result] = adapter.evaluate(["p0"], _CONTEXT)
        assert result.objectives == (-5.0,)
        payload = result.metadata["objective_adapter"]
        assert payload["actual_objective_count"] == 2
        assert payload["expected_objective_count"] == 3

    def test_custom_penalty_used_for_infinite_objective(self) -> None:
        adapter = ObjectiveVectorAdapter(
            _evaluator([FitnessResult(objectives=(1.0, float("inf")), metadata={})]),
            expected_objective_count=2,
            objective_directions=("maximize", "minimize"),
            invalid_penalty=-7.5,
        )
        [result] = adapter.evaluate(["p0"], _CONTEXT)
        assert result.objectives == (-7.5,)
        assert result.metadata["reason_code"] == "non_finite_objective"
        assert result.metadata["objective_adapter"]["raw_objectives"] == (
            1.0,
            float("inf"),
        )
