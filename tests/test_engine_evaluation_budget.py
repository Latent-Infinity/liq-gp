"""Evaluation-budget and pipeline-consistency tests for the evolution loop.

The evaluator is the dominant cost in GP.  These tests pin down how many
times, and on which populations, the engine calls ``evaluator.evaluate`` so
that redundant full-population passes cannot silently creep back in.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from liq.gp.config import GPConfig, SeedInjectionConfig
from liq.gp.evolution import engine as engine_module
from liq.gp.evolution.diversity import compute_fingerprint
from liq.gp.evolution.engine import evolve
from liq.gp.primitives.registry import PrimitiveRegistry
from liq.gp.program.ast import ConstantNode, FunctionNode, Program, TerminalNode
from liq.gp.program.eval import evaluate
from liq.gp.types import FitnessResult, Series

POP = 40
GENS = 6


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
    reg.register(
        "neg",
        lambda a: -a,
        category="numeric",
        input_types=(Series,),
        output_type=Series,
    )
    return reg


@dataclass
class CountingEvaluator:
    """Negative-MSE evaluator recording the size of every batch it receives."""

    batch_sizes: list[int] = field(default_factory=list)

    def evaluate(
        self,
        programs: list[Program],
        context: dict[str, np.ndarray],
    ) -> list[FitnessResult]:
        self.batch_sizes.append(len(programs))
        x = context["x"]
        target = x * x + 1.0
        results = []
        for program in programs:
            out = evaluate(program, context)
            mse = float(np.mean((out - target) ** 2))
            if not np.isfinite(mse):
                mse = 1e12
            results.append(FitnessResult(objectives=(-mse,)))
        return results

    @property
    def full_population_calls(self) -> int:
        return sum(1 for size in self.batch_sizes if size == POP)

    @property
    def programs_evaluated(self) -> int:
        return sum(self.batch_sizes)


def _context() -> dict[str, np.ndarray]:
    return {"x": np.linspace(-2.0, 2.0, 64)}


def _seed() -> Program:
    return FunctionNode(
        primitive=_registry().get("add"),
        children=(TerminalNode("x", Series), ConstantNode(1.0)),
    )


class TestFullPopulationPasses:
    """Exactly one full-population evaluation per generation, and no extra final pass."""

    def test_baseline_without_optional_stages(self) -> None:
        evaluator = CountingEvaluator()
        config = GPConfig(
            population_size=POP,
            generations=GENS,
            constant_opt_enabled=False,
            seed=3,
        )
        evolve(_registry(), config, evaluator, _context())
        assert evaluator.full_population_calls == GENS
        assert evaluator.programs_evaluated == POP * GENS

    def test_constant_optimization_adds_no_full_pass(self) -> None:
        evaluator = CountingEvaluator()
        config = GPConfig(
            population_size=POP,
            generations=GENS,
            constant_opt_enabled=True,
            constant_opt_top_k=0.1,
            seed=3,
        )
        evolve(_registry(), config, evaluator, _context())
        assert evaluator.full_population_calls == GENS

    def test_seed_injection_adds_no_full_pass(self) -> None:
        evaluator = CountingEvaluator()
        config = GPConfig(
            population_size=POP,
            generations=GENS,
            constant_opt_enabled=False,
            seed_injection=SeedInjectionConfig(interval=2, count=3, method="ramped"),
            seed=3,
        )
        evolve(_registry(), config, evaluator, _context())
        assert evaluator.full_population_calls == GENS
        # Injection fires at generations 2 and 4; each adds exactly one
        # subset evaluation of the injected programs.
        assert evaluator.batch_sizes.count(3) == 2

    def test_all_optional_stages_together(self) -> None:
        evaluator = CountingEvaluator()
        config = GPConfig(
            population_size=POP,
            generations=GENS,
            constant_opt_enabled=True,
            seed_injection=SeedInjectionConfig(interval=2, count=3, method="direct"),
            seed=3,
        )
        evolve(_registry(), config, evaluator, _context(), seed_programs=[_seed()])
        assert evaluator.full_population_calls == GENS


class TestInjectionReportedInStats:
    def test_injected_count_recorded_on_injection_generations(self) -> None:
        evaluator = CountingEvaluator()
        config = GPConfig(
            population_size=POP,
            generations=GENS,
            constant_opt_enabled=False,
            seed_injection=SeedInjectionConfig(interval=2, count=3, method="ramped"),
            seed=3,
        )
        result = evolve(_registry(), config, evaluator, _context())
        injected = [s.injected_count for s in result.fitness_history]
        assert injected == [0, 0, 3, 0, 3, 0]


class TestDedupReceivesMatchingFingerprints:
    """Fingerprints handed to dedup must describe the population being deduped."""

    def test_fingerprints_match_population(self, monkeypatch) -> None:
        captured: list[tuple[list[Program], dict, list[bytes]]] = []
        original = engine_module.deduplicate_population

        def spy(population, ref_context, registry, config, rng, fingerprints=None):
            captured.append((list(population), ref_context, list(fingerprints or [])))
            return original(
                population,
                ref_context,
                registry,
                config,
                rng,
                fingerprints=fingerprints,
            )

        monkeypatch.setattr(engine_module, "deduplicate_population", spy)
        config = GPConfig(
            population_size=POP,
            generations=3,
            constant_opt_enabled=False,
            seed=11,
        )
        evolve(_registry(), config, CountingEvaluator(), _context())

        assert captured, "dedup was never invoked"
        for population, ref_context, fingerprints in captured:
            # Either dedup fingerprints the population itself, or the engine
            # hands over fingerprints computed for exactly this population.
            if not fingerprints:
                continue
            expected = [
                compute_fingerprint(p, ref_context, config.semantic_precision)
                for p in population
            ]
            assert fingerprints == expected
