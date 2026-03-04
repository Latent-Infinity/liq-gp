"""Compiler bridge from typed regime models to typed GP AST programs."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from liq.gp.errors import PrimitiveError
from liq.gp.primitives.registry import PrimitiveInfo, PrimitiveRegistry
from liq.gp.program.ast import (
    ConstantNode,
    FunctionNode,
    ParameterizedNode,
    Program,
    TerminalNode,
)
from liq.gp.types import BoolSeries, GPType, Series


class RegimeCompilerError(ValueError):
    """Base class for regime compiler contract failures."""


class RegimeModelContractError(RegimeCompilerError):
    """Input model does not satisfy the regime contract."""


class RegimePrimitiveContractError(RegimeCompilerError):
    """Configured primitive or arity contract is invalid for compile."""


@runtime_checkable
class RegimeBlockLike(Protocol):
    """Structural protocol for compiler-visible regime blocks."""

    program: Program


@runtime_checkable
class RegimeWeightsLike(Protocol):
    """Structural protocol for weighted expert collections."""

    values: Sequence[float]


@runtime_checkable
class RegimeModelLike(Protocol):
    """Structural protocol for typed regime model inputs."""

    detector: RegimeBlockLike
    gate: RegimeBlockLike
    experts: Sequence[RegimeBlockLike]
    risk: RegimeBlockLike | None
    weights: RegimeWeightsLike | Sequence[float] | None


def _is_program(candidate: object) -> bool:
    return isinstance(
        candidate,
        (TerminalNode, ConstantNode, FunctionNode, ParameterizedNode),
    )


def _require_primitive(
    registry: PrimitiveRegistry,
    name: str,
    *,
    expected_arity: int,
    expected_output_type: GPType,
) -> PrimitiveInfo:
    try:
        primitive = registry.get(name)
    except PrimitiveError as exc:
        raise RegimePrimitiveContractError(
            f"Required primitive {name!r} is not registered"
        ) from exc

    if primitive.arity != expected_arity:
        raise RegimePrimitiveContractError(
            f"Primitive {name!r} must have arity={expected_arity}, "
            f"got {primitive.arity}"
        )

    if primitive.output_type is not expected_output_type:
        raise RegimePrimitiveContractError(
            f"Primitive {name!r} output_type mismatch: "
            f"expected {expected_output_type.name}, got {primitive.output_type.name}"
        )

    return primitive


def _extract_weights(model: RegimeModelLike) -> tuple[float, ...]:
    raw = model.weights
    if raw is None:
        return ()
    if isinstance(raw, RegimeWeightsLike):
        raw_weights = raw.values
    else:
        raw_weights = raw
    return tuple(float(item) for item in raw_weights)


def _expect_block_program(model: RegimeModelLike, role: str) -> Program:
    block = getattr(model, role)

    if _is_program(block):
        return block  # legacy passthrough shortcut not expected here

    if not isinstance(block, RegimeBlockLike):
        raise RegimeModelContractError(f"Regime model missing or invalid {role} block")

    program = getattr(block, "program")
    if not isinstance(program, (TerminalNode, ConstantNode, FunctionNode, ParameterizedNode)):
        raise RegimeModelContractError(
            f"Regime {role} block program must be a GP Program node"
        )
    return program


def _expect_expert_programs(model: RegimeModelLike) -> list[Program]:
    experts = getattr(model, "experts")
    if not isinstance(experts, Sequence):
        raise RegimeModelContractError("Regime model experts must be a sequence")
    if not experts:
        raise RegimeModelContractError("Regime model must contain at least one expert")

    programs: list[Program] = []
    for expert in experts:
        if not isinstance(expert, RegimeBlockLike):
            raise RegimeModelContractError("Each expert must be a regime block")
        if not isinstance(
            expert.program,
            (TerminalNode, ConstantNode, FunctionNode, ParameterizedNode),
        ):
            raise RegimeModelContractError("Expert block program must be a GP Program node")
        programs.append(expert.program)
    return programs


def _expect_optional_risk_program(model: RegimeModelLike) -> Program | None:
    risk = getattr(model, "risk")
    if risk is None:
        return None
    if _is_program(risk):
        return risk
    if not isinstance(risk, RegimeBlockLike):
        raise RegimeModelContractError("Regime risk block must be a valid regime block")
    if not isinstance(risk.program, (TerminalNode, ConstantNode, FunctionNode, ParameterizedNode)):
        raise RegimeModelContractError("Risk block program must be a GP Program node")
    return risk.program


def _require_output_type(program: Program, expected: GPType, role: str) -> None:
    if program.output_type is not expected:
        raise RegimeModelContractError(
            f"{role} program must output {expected.name}, got {program.output_type.name}"
        )


def _mul(registry: PrimitiveRegistry, a: Program, b: Program) -> FunctionNode:
    mul_primitive = _require_primitive(registry, "mul", expected_arity=2, expected_output_type=Series)
    return FunctionNode(primitive=mul_primitive, children=(a, b))


def _add(registry: PrimitiveRegistry, a: Program, b: Program) -> FunctionNode:
    add_primitive = _require_primitive(
        registry,
        "add",
        expected_arity=2,
        expected_output_type=Series,
    )
    return FunctionNode(primitive=add_primitive, children=(a, b))


def _if_then_else(
    registry: PrimitiveRegistry,
    condition: Program,
    on_true: Program,
    on_false: Program,
) -> FunctionNode:
    ite_primitive = _require_primitive(
        registry,
        "if_then_else",
        expected_arity=3,
        expected_output_type=Series,
    )
    return FunctionNode(primitive=ite_primitive, children=(condition, on_true, on_false))


def compile_regime_model_to_program(
    model: Program | RegimeModelLike,
    registry: PrimitiveRegistry,
) -> Program:
    """Compile a typed regime model into a typed GP AST.

    Legacy path: if ``model`` is already a GP program, it is returned unchanged.
    """
    if _is_program(model):
        return model

    if not isinstance(model, RegimeModelLike):
        raise RegimeModelContractError(
            "Regime model must be a Program or protocol-compatible object"
        )

    detector = _expect_block_program(model, "detector")
    gate = _expect_block_program(model, "gate")
    expert_programs = _expect_expert_programs(model)
    risk = _expect_optional_risk_program(model)
    weights = _extract_weights(model)

    if not isinstance(expert_programs, Sequence):
        raise RegimeModelContractError("Regime experts must be a sequence")
    if weights and len(weights) != len(expert_programs):
        raise RegimeModelContractError(
            "Regime weights length must match expert count: "
            f"{len(weights)} != {len(expert_programs)}"
        )

    _require_output_type(detector, BoolSeries, "detector")
    _require_output_type(gate, BoolSeries, "gate")
    for expert in expert_programs:
        _require_output_type(expert, Series, "expert")
    if risk is not None:
        _require_output_type(risk, Series, "risk")

    normalized_weights = weights
    if not normalized_weights:
        normalized_weights = tuple(1.0 for _ in expert_programs)

    experts_fold: Program | None = None
    for expert_program, weight in zip(expert_programs, normalized_weights):
        weighted = _mul(registry, expert_program, ConstantNode(weight, output_type=Series))
        if experts_fold is None:
            experts_fold = weighted
        else:
            experts_fold = _add(registry, experts_fold, weighted)

    if experts_fold is None:
        raise RegimeModelContractError("No experts were provided")

    detected = _if_then_else(registry, detector, experts_fold, ConstantNode(0.0, output_type=Series))
    active = _if_then_else(
        registry,
        gate,
        detected,
        ConstantNode(0.0, output_type=Series),
    )
    if risk is None:
        return active
    return _mul(registry, risk, active)


if TYPE_CHECKING:
    __all__ = [
        "compile_regime_model_to_program",
        "RegimeCompilerError",
        "RegimeModelContractError",
        "RegimePrimitiveContractError",
        "RegimeModelLike",
        "RegimeBlockLike",
        "RegimeWeightsLike",
    ]
else:
    __all__ = [
        "compile_regime_model_to_program",
        "RegimeCompilerError",
        "RegimeModelContractError",
        "RegimePrimitiveContractError",
    ]
