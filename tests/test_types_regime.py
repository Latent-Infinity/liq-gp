"""Compiler contract tests for typed regime models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from liq.gp import (
    BoolSeries,
    ConstantNode,
    FunctionNode,
    PrimitiveRegistry,
    Program,
    RegimeModelContractError,
    RegimePrimitiveContractError,
    Series,
    TerminalNode,
    compile_regime_model_to_program,
)
from liq.gp.program.eval import evaluate


@dataclass(frozen=True)
class _RegimeBlock:
    program: Program


@dataclass(frozen=True)
class _Model:
    detector: _RegimeBlock
    gate: _RegimeBlock
    experts: tuple[_RegimeBlock, ...]
    risk: _RegimeBlock | None = None
    weights: tuple[float, ...] | list[float] | None = None


def _terminal(name: str, output_type) -> TerminalNode:
    return TerminalNode(name=name, output_type=output_type)


def _block(name: str, output_type) -> _RegimeBlock:
    return _RegimeBlock(program=_terminal(name, output_type))


def _model(
    *,
    include_risk: bool = False,
    detector: _RegimeBlock | None = None,
    gate: _RegimeBlock | None = None,
    expert_names: tuple[str, ...] = ("e1", "e2"),
    weights: tuple[float, ...] | list[float] | None = None,
) -> _Model:
    risk_block = _block("risk", Series) if include_risk else None
    return _Model(
        detector=detector or _block("det", BoolSeries),
        gate=gate or _block("gate", BoolSeries),
        experts=tuple(_block(name, Series) for name in expert_names),
        risk=risk_block,
        weights=weights,
    )


def _build_registry(
    *,
    include_if_then_else: bool = True,
    mul_inputs: int = 2,
    add_inputs: int = 2,
    if_arity: int = 3,
) -> PrimitiveRegistry:
    reg = PrimitiveRegistry()
    reg.register("det", lambda: None, input_types=(), output_type=BoolSeries)
    reg.register("gate", lambda: None, input_types=(), output_type=BoolSeries)
    for name in ("e1", "e2", "e3", "risk"):
        reg.register(name, lambda: None, input_types=(), output_type=Series)

    reg.register(
        "mul",
        lambda *values: np.prod(np.stack(values), axis=0),
        input_types=tuple([Series] * mul_inputs),
        output_type=Series,
    )
    reg.register(
        "add",
        lambda *values: np.sum(np.stack(values), axis=0),
        input_types=tuple([Series] * add_inputs),
        output_type=Series,
    )

    if include_if_then_else:
        reg.register(
            "if_then_else",
            lambda cond, on_true, on_false: np.where(cond > 0.5, on_true, on_false),
            input_types=tuple([BoolSeries, Series, Series][:if_arity]),
            output_type=Series,
        )
    return reg


class TestRegimeCompilerContracts:
    """Typed contract coverage for regime compilation."""

    def test_legacy_program_passthrough(self) -> None:
        registry = _build_registry()
        source = ConstantNode(1.0, output_type=Series)
        assert compile_regime_model_to_program(source, registry) is source

    def test_valid_model_compiles_with_deterministic_expert_order(self) -> None:
        registry = _build_registry()
        program = compile_regime_model_to_program(
            _model(
                include_risk=True,
                expert_names=("e1", "e2", "e3"),
                weights=(2.0, 3.0, 4.0),
            ),
            registry,
        )

        assert isinstance(program, FunctionNode)
        assert program.primitive.name == "mul"
        assert isinstance(program.children[0], TerminalNode)
        assert program.children[0].name == "risk"

        active = program.children[1]
        assert isinstance(active, FunctionNode)
        assert active.primitive.name == "if_then_else"
        assert isinstance(active.children[0], TerminalNode)
        assert active.children[0].name == "gate"

        detected = active.children[1]
        assert isinstance(detected, FunctionNode)
        assert detected.primitive.name == "if_then_else"
        blend = detected.children[1]
        assert isinstance(blend, FunctionNode)
        assert blend.primitive.name == "add"

        first_add = blend.children[0]
        second = blend.children[1]
        assert isinstance(first_add, FunctionNode)
        assert first_add.primitive.name == "add"

        first_weighted = first_add.children[0]
        second_weighted = first_add.children[1]
        third_weighted = second

        assert isinstance(first_weighted, FunctionNode)
        assert isinstance(second_weighted, FunctionNode)
        assert isinstance(third_weighted, FunctionNode)
        assert first_weighted.primitive.name == "mul"
        assert second_weighted.primitive.name == "mul"
        assert third_weighted.primitive.name == "mul"
        assert first_weighted.children[0].name == "e1"
        assert first_weighted.children[1].value == 2.0
        assert second_weighted.children[0].name == "e2"
        assert second_weighted.children[1].value == 3.0
        assert third_weighted.children[0].name == "e3"
        assert third_weighted.children[1].value == 4.0

        context = {
            "det": np.ones(4),
            "gate": np.ones(4),
            "e1": np.ones(4),
            "e2": np.full(4, 2.0),
            "e3": np.full(4, 3.0),
            "risk": np.full(4, 0.5),
        }
        result = evaluate(program, context)
        assert np.allclose(result, np.full(4, 10.0))

    def test_invalid_model_object_is_rejected(self) -> None:
        registry = _build_registry()
        with pytest.raises(
            RegimeModelContractError, match="protocol-compatible object"
        ):
            compile_regime_model_to_program({"bad": "model"}, registry)

    def test_missing_required_primitive_fails(self) -> None:
        registry = _build_registry(include_if_then_else=False)
        with pytest.raises(RegimePrimitiveContractError, match="not registered"):
            compile_regime_model_to_program(_model(), registry)

    def test_wrong_primitive_arity_fails(self) -> None:
        registry = _build_registry(add_inputs=3)
        with pytest.raises(RegimePrimitiveContractError, match="arity"):
            compile_regime_model_to_program(_model(), registry)

    def test_output_type_mismatches_raise_model_contract_errors(self) -> None:
        registry = _build_registry()
        bad_detector = _model(expert_names=("e1",), detector=_block("det", Series))
        with pytest.raises(
            RegimeModelContractError, match="detector program must output BoolSeries"
        ):
            compile_regime_model_to_program(bad_detector, registry)

    def test_weight_length_mismatch_raises(self) -> None:
        registry = _build_registry()
        with pytest.raises(RegimeModelContractError, match="must match expert count"):
            compile_regime_model_to_program(
                _model(expert_names=("e1", "e2"), weights=(1.0,)),
                registry,
            )
