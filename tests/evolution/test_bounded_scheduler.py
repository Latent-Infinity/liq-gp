"""Stage-5 bounded scheduler tests for GP evolution."""

from __future__ import annotations

import logging
import time

import numpy as np
import pytest

from liq.gp.config import GPConfig, SchedulerConfig
from liq.gp.errors import EvolutionError
from liq.gp.evolution.engine import evolve
from liq.gp.primitives.registry import PrimitiveRegistry
from liq.gp.program.ast import Program
from liq.gp.types import FitnessResult, Series


def _make_registry() -> PrimitiveRegistry:
    reg = PrimitiveRegistry()
    reg.register("x", lambda: None, input_types=(), output_type=Series)
    reg.register(
        "add",
        lambda a, b: a + b,
        category="numeric",
        input_types=(Series, Series),
        output_type=Series,
    )
    return reg


def _base_config(*, scheduler: SchedulerConfig, seed: int = 42) -> GPConfig:
    return GPConfig(
        population_size=20,
        max_depth=4,
        generations=1,
        seed=seed,
        constant_opt_enabled=False,
        simplification_enabled=False,
        elitism_count=2,
        tournament_size=3,
        scheduler=scheduler,
    )


def _context(n: int = 64) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(0)
    return {"x": rng.uniform(-1.0, 1.0, size=n)}


class RecordingEvaluator:
    def __init__(self) -> None:
        self.call_count = 0
        self.max_batch_size = 0

    def evaluate(
        self,
        programs: list[Program],
        context: dict[str, np.ndarray],
    ) -> list[FitnessResult]:
        del context
        self.call_count += 1
        self.max_batch_size = max(self.max_batch_size, len(programs))
        return [FitnessResult(objectives=(1.0,)) for _ in programs]


class SlowEvaluator:
    def __init__(self, delay_seconds: float = 0.02) -> None:
        self.delay_seconds = delay_seconds

    def evaluate(
        self,
        programs: list[Program],
        context: dict[str, np.ndarray],
    ) -> list[FitnessResult]:
        del context
        time.sleep(self.delay_seconds)
        return [FitnessResult(objectives=(1.0,)) for _ in programs]


class TestBoundedSchedulerContracts:
    def test_bounded_in_flight_and_batch_caps(self) -> None:
        scheduler = SchedulerConfig(
            enabled=True,
            max_in_flight=2,
            queue_capacity=16,
            eval_batch_size=3,
            eval_timeout_seconds=2.0,
            memory_budget_mb=2048,
            max_cpu_workers=2,
            safe_fallback_mode="fail",
        )
        evaluator = RecordingEvaluator()
        result = evolve(
            _make_registry(), _base_config(scheduler=scheduler), evaluator, _context()
        )
        metrics = result.fitness_history[0].scheduler_metrics
        assert metrics["mode"] == "bounded"
        assert metrics["peak_in_flight"] <= 2
        assert metrics["queue_capacity"] == 16
        assert metrics["saturation_reason_code"] == "ok"
        assert evaluator.max_batch_size <= 3

    def test_queue_saturation_raises_with_fail_mode(self) -> None:
        scheduler = SchedulerConfig(
            enabled=True,
            max_in_flight=2,
            queue_capacity=2,
            eval_batch_size=1,
            eval_timeout_seconds=1.0,
            memory_budget_mb=2048,
            max_cpu_workers=2,
            safe_fallback_mode="fail",
        )
        with pytest.raises(EvolutionError, match="scheduler_queue_saturated"):
            evolve(
                _make_registry(),
                _base_config(scheduler=scheduler),
                RecordingEvaluator(),
                _context(),
            )

    def test_timeout_saturation_falls_back_safely(self, caplog) -> None:
        scheduler = SchedulerConfig(
            enabled=True,
            max_in_flight=2,
            queue_capacity=16,
            eval_batch_size=2,
            eval_timeout_seconds=1e-6,
            memory_budget_mb=2048,
            max_cpu_workers=2,
            safe_fallback_mode="sequential",
        )
        with caplog.at_level(logging.WARNING, logger="liq.gp.evolution.engine"):
            result = evolve(
                _make_registry(),
                _base_config(scheduler=scheduler),
                SlowEvaluator(delay_seconds=0.01),
                _context(),
            )
        messages = [record.getMessage() for record in caplog.records]
        assert any(
            "bounded_evaluator_saturated reason=scheduler_timeout" in message
            for message in messages
        )
        assert any(
            "bounded_evaluator_saturation_fallback reason=scheduler_timeout" in message
            for message in messages
        )
        metrics = result.fitness_history[0].scheduler_metrics
        assert metrics["mode"] == "sequential_fallback"
        assert metrics["saturation_reason_code"] == "scheduler_timeout"

    def test_timeout_saturation_raises_when_fail_mode(self, caplog) -> None:
        scheduler = SchedulerConfig(
            enabled=True,
            max_in_flight=2,
            queue_capacity=16,
            eval_batch_size=2,
            eval_timeout_seconds=1e-6,
            memory_budget_mb=2048,
            max_cpu_workers=2,
            safe_fallback_mode="fail",
        )
        with (
            caplog.at_level(logging.WARNING, logger="liq.gp.evolution.engine"),
            pytest.raises(EvolutionError, match="scheduler_timeout"),
        ):
            evolve(
                _make_registry(),
                _base_config(scheduler=scheduler),
                SlowEvaluator(delay_seconds=0.01),
                _context(),
            )
        messages = [record.getMessage() for record in caplog.records]
        assert any(
            "bounded_evaluator_saturated reason=scheduler_timeout" in message
            for message in messages
        )
        assert any(
            "bounded_evaluator_failed reason=scheduler_timeout" in message
            for message in messages
        )
        assert any("stage=bounded_scheduler" in message for message in messages)

    def test_memory_budget_saturation_uses_safe_fallback(self) -> None:
        scheduler = SchedulerConfig(
            enabled=True,
            max_in_flight=5,
            queue_capacity=32,
            eval_batch_size=2,
            eval_timeout_seconds=1.0,
            memory_budget_mb=128,
            max_cpu_workers=2,
            safe_fallback_mode="sequential",
        )
        heavy = {
            "x": np.ones(2_000_000, dtype=np.float64),
            "y": np.ones(2_000_000, dtype=np.float64),
        }
        result = evolve(
            _make_registry(),
            _base_config(scheduler=scheduler),
            RecordingEvaluator(),
            heavy,
        )
        metrics = result.fitness_history[0].scheduler_metrics
        assert metrics["mode"] == "sequential_fallback"
        assert metrics["saturation_reason_code"] == "scheduler_memory_saturated"
