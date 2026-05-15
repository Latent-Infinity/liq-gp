"""Stage-16 tests for motif mining and module-preserving recombination."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from liq.gp import (
    BoolSeries,
    PrimitiveRegistry,
    Program,
    Series,
    TerminalNode,
    compile_regime_model_to_program,
)
from liq.gp.evolution.operators import (
    BlockConstraintTelemetry,
    extract_regime_modules,
    mine_regime_motifs,
    module_preserving_crossover,
)


@dataclass(frozen=True)
class _RegimeBlock:
    program: Program


@dataclass(frozen=True)
class _RegimeModel:
    detector: _RegimeBlock
    gate: _RegimeBlock
    experts: tuple[_RegimeBlock, ...]
    risk: _RegimeBlock | None = None
    weights: tuple[float, ...] | None = None


class _ScriptedRNG:
    def __init__(self, values: tuple[int, ...] | list[int]) -> None:
        self._values = list(values)
        self._position = 0

    def integers(self, low: int, high: int | None = None, *args, **kwargs) -> int:
        if self._position >= len(self._values):
            value = 0
        else:
            value = int(self._values[self._position])
        self._position += 1
        if high is None:
            return value % low
        return low + (value % (high - low))

    def random(self, *args, **kwargs) -> float:
        return 0.5


def _make_registry() -> PrimitiveRegistry:
    registry = PrimitiveRegistry()
    for name in ("det_a", "det_b", "gate_a", "gate_b"):
        registry.register(name, lambda: None, input_types=(), output_type=BoolSeries)
    for name in ("e1", "e2", "f1", "f2", "risk_a", "risk_b"):
        registry.register(name, lambda: None, input_types=(), output_type=Series)

    registry.register(
        "if_then_else",
        lambda cond, on_true, on_false: np.where(cond > 0.5, on_true, on_false),
        input_types=(BoolSeries, Series, Series),
        output_type=Series,
    )
    registry.register(
        "mul",
        lambda a, b: a * b,
        input_types=(Series, Series),
        output_type=Series,
    )
    registry.register(
        "add",
        lambda a, b: a + b,
        input_types=(Series, Series),
        output_type=Series,
    )
    return registry


def _regime_program(
    registry: PrimitiveRegistry,
    *,
    detector: str,
    gate: str,
    experts: tuple[str, str],
    risk: str,
) -> Program:
    model = _RegimeModel(
        detector=_RegimeBlock(TerminalNode(name=detector, output_type=BoolSeries)),
        gate=_RegimeBlock(TerminalNode(name=gate, output_type=BoolSeries)),
        experts=tuple(
            _RegimeBlock(TerminalNode(name=name, output_type=Series))
            for name in experts
        ),
        risk=_RegimeBlock(TerminalNode(name=risk, output_type=Series)),
    )
    return compile_regime_model_to_program(model, registry)


def _terminal_names(program: Program) -> set[str]:
    names: set[str] = set()
    if isinstance(program, TerminalNode):
        names.add(program.name)
    children = getattr(program, "children", ())
    for child in children:
        names.update(_terminal_names(child))
    return names


class TestModuleExtractionAndMotifs:
    def test_extract_regime_modules_respects_block_boundaries(self) -> None:
        registry = _make_registry()
        program = _regime_program(
            registry,
            detector="det_a",
            gate="gate_a",
            experts=("e1", "e2"),
            risk="risk_a",
        )

        modules = extract_regime_modules(program)

        assert {"gate", "detector", "risk"}.issubset(set(modules))
        assert {"expert:0", "expert:1"}.issubset(set(modules))
        assert all(
            role.startswith("expert:") or role in {"gate", "detector", "risk"}
            for role in modules
        )

    def test_mine_regime_motifs_discovers_frequent_modules(self) -> None:
        registry = _make_registry()
        elite_a = _regime_program(
            registry,
            detector="det_a",
            gate="gate_a",
            experts=("e1", "e2"),
            risk="risk_a",
        )
        elite_b = _regime_program(
            registry,
            detector="det_b",
            gate="gate_b",
            experts=("f1", "f2"),
            risk="risk_b",
        )

        motifs = mine_regime_motifs([elite_a, elite_a, elite_b], min_frequency=2)

        assert motifs
        assert any(motif.role == "risk" and motif.frequency >= 2 for motif in motifs)


class TestModulePreservingCrossover:
    def test_module_preserving_crossover_swaps_modules_with_telemetry(self) -> None:
        registry = _make_registry()
        parent1 = _regime_program(
            registry,
            detector="det_a",
            gate="gate_a",
            experts=("e1", "e2"),
            risk="risk_a",
        )
        parent2 = _regime_program(
            registry,
            detector="det_b",
            gate="gate_b",
            experts=("f1", "f2"),
            risk="risk_b",
        )
        telemetry = BlockConstraintTelemetry()
        # sorted roles: detector, expert:0, expert:1, gate, risk -> choose risk
        rng = _ScriptedRNG([99, 0, 0])

        child1, child2 = module_preserving_crossover(
            parent1,
            parent2,
            registry,
            max_depth=16,
            rng=rng,
            max_attempts=1,
            block_constraint_telemetry=telemetry,
        )

        assert child1 != parent1
        assert child2 != parent2
        assert "risk_b" in _terminal_names(child1)
        assert "risk_a" in _terminal_names(child2)
        assert telemetry.accepted["module_crossover:risk"] == 1

    def test_module_preserving_crossover_falls_back_for_non_regime_programs(
        self,
    ) -> None:
        registry = _make_registry()
        parent1 = TerminalNode(name="e1", output_type=Series)
        parent2 = TerminalNode(name="f1", output_type=Series)
        telemetry = BlockConstraintTelemetry()

        child1, child2 = module_preserving_crossover(
            parent1,
            parent2,
            registry,
            max_depth=4,
            rng=np.random.default_rng(5),
            max_attempts=3,
            block_constraint_telemetry=telemetry,
        )

        assert child1 == parent1
        assert child2 == parent2
        assert telemetry.accepted == {}
