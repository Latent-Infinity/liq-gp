"""Deterministic objective-vector adapter for GP fitness boundaries."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from liq.gp.types import FitnessResult

OBJECTIVE_ADAPTER_VERSION = "1.0"


class ObjectiveVectorAdapter:
    """Adapt multi-objective vectors into scalar GP fitness values.

    This boundary is intended for consumers that produce ordered objective
    vectors (for example Stage-B trading evaluators) but need deterministic
    scalar fitness values for single-objective GP workflows.
    """

    def __init__(
        self,
        evaluator: Any,
        *,
        expected_objective_count: int,
        objective_directions: Sequence[str],
        reduction_weights: Sequence[float] | None = None,
        invalid_penalty: float = -1e12,
    ) -> None:
        if expected_objective_count < 1:
            raise ValueError("expected_objective_count must be >= 1")
        if len(objective_directions) != expected_objective_count:
            raise ValueError(
                "objective_directions must match expected_objective_count"
            )
        if any(direction not in {"maximize", "minimize"} for direction in objective_directions):
            raise ValueError(
                "objective_directions must contain only 'maximize'/'minimize'"
            )
        if reduction_weights is None:
            reduction_weights = tuple([1.0] * expected_objective_count)
        if len(reduction_weights) != expected_objective_count:
            raise ValueError(
                "reduction_weights must match expected_objective_count"
            )
        if any(weight < 0.0 for weight in reduction_weights):
            raise ValueError("reduction_weights must be >= 0")
        self._evaluator = evaluator
        self._expected_count = expected_objective_count
        self._objective_directions = tuple(objective_directions)
        self._weights = tuple(float(weight) for weight in reduction_weights)
        self._invalid_penalty = float(invalid_penalty)

    def evaluate(
        self,
        programs: list[Any],
        context: Mapping[str, Any],
    ) -> list[FitnessResult]:
        raw = self._evaluate_inner(programs, context)
        if len(raw) != len(programs):
            raise ValueError(
                "wrapped evaluator returned mismatched result count"
            )
        return [self._adapt_result(result) for result in raw]

    def evaluate_fitness(
        self,
        programs: list[Any],
        context: Mapping[str, Any],
    ) -> list[FitnessResult]:
        return self.evaluate(programs, context)

    def _evaluate_inner(
        self,
        programs: list[Any],
        context: Mapping[str, Any],
    ) -> list[FitnessResult]:
        if hasattr(self._evaluator, "evaluate") and callable(self._evaluator.evaluate):
            return self._evaluator.evaluate(programs, context)
        if hasattr(self._evaluator, "evaluate_fitness") and callable(
            self._evaluator.evaluate_fitness
        ):
            return self._evaluator.evaluate_fitness(programs, context)
        if callable(self._evaluator):
            return self._evaluator(programs, context)
        raise TypeError(
            "wrapped evaluator must provide evaluate(), evaluate_fitness(), or be callable"
        )

    def _adapt_result(self, result: FitnessResult) -> FitnessResult:
        objectives = tuple(float(value) for value in result.objectives)
        metadata = dict(result.metadata)

        if len(objectives) != self._expected_count:
            metadata["objective_adapter"] = {
                "version": OBJECTIVE_ADAPTER_VERSION,
                "reason_code": "invalid_objective_shape",
                "expected_objective_count": self._expected_count,
                "actual_objective_count": len(objectives),
                "objective_directions": self._objective_directions,
                "reduction_weights": self._weights,
                "raw_objectives": objectives,
            }
            metadata["reason_code"] = "invalid_objective_shape"
            return FitnessResult(objectives=(self._invalid_penalty,), metadata=metadata)

        arr = np.asarray(objectives, dtype=np.float64)
        if not np.all(np.isfinite(arr)):
            metadata["objective_adapter"] = {
                "version": OBJECTIVE_ADAPTER_VERSION,
                "reason_code": "non_finite_objective",
                "expected_objective_count": self._expected_count,
                "objective_directions": self._objective_directions,
                "reduction_weights": self._weights,
                "raw_objectives": objectives,
            }
            metadata["reason_code"] = "non_finite_objective"
            return FitnessResult(objectives=(self._invalid_penalty,), metadata=metadata)

        signed = np.asarray(
            [
                value if direction == "maximize" else -value
                for value, direction in zip(
                    arr, self._objective_directions, strict=True
                )
            ],
            dtype=np.float64,
        )
        score = float(np.dot(np.asarray(self._weights, dtype=np.float64), signed))
        metadata["objective_adapter"] = {
            "version": OBJECTIVE_ADAPTER_VERSION,
            "reason_code": "ok",
            "expected_objective_count": self._expected_count,
            "objective_directions": self._objective_directions,
            "reduction_weights": self._weights,
            "raw_objectives": objectives,
            "reduced_score": score,
        }
        metadata["reason_code"] = "ok"
        return FitnessResult(objectives=(score,), metadata=metadata)
