"""Serialization and deserialization of GP program trees (FR-9).

Converts Program trees to/from JSON-serializable dicts, enabling persistence,
transport, and reproducibility of evolved programs.
"""

from __future__ import annotations

from typing import Any

from liq.gp.errors import SerializationError
from liq.gp.primitives.registry import PrimitiveRegistry
from liq.gp.program.ast import (
    ConstantNode,
    FunctionNode,
    ParameterizedNode,
    Program,
    TerminalNode,
)
from liq.gp.types import GPType

SCHEMA_VERSION = "1.0.0"


def _validate_schema_version(data: dict[str, Any]) -> None:
    """Validate payload schema version for forward-compatible reads."""
    version = data.get("schema_version")
    if version is None:
        raise SerializationError("Missing schema_version in serialized payload")
    if version != SCHEMA_VERSION:
        msg = f"Unsupported schema_version {version!r}; expected {SCHEMA_VERSION!r}"
        raise SerializationError(msg)


# ---------------------------------------------------------------------------
# Program serialization
# ---------------------------------------------------------------------------


def _serialize_node(node: Program) -> dict[str, Any]:
    """Convert a single AST node to a JSON-serializable dict (recursive)."""
    if isinstance(node, TerminalNode):
        return {
            "type": "terminal",
            "name": node.name,
            "output_type": node.output_type.name,
        }

    if isinstance(node, ConstantNode):
        return {
            "type": "constant",
            "value": float(node.value),
            "output_type": node.output_type.name,
        }

    if isinstance(node, FunctionNode):
        return {
            "type": "function",
            "primitive": node.primitive.name,
            "children": [_serialize_node(c) for c in node.children],
        }

    if isinstance(node, ParameterizedNode):
        return {
            "type": "parameterized",
            "primitive": node.primitive.name,
            "children": [_serialize_node(c) for c in node.children],
            "params": {
                k: int(v) if isinstance(v, int) else float(v)
                for k, v in node.params.items()
            },
        }

    msg = f"Unknown node type: {type(node).__name__}"
    raise SerializationError(msg)


def serialize(program: Program) -> dict[str, Any]:
    """Serialize a Program tree to a JSON-serializable dict (FR-9).

    Returns:
        A dict with ``schema_version`` and ``program`` keys.
    """
    return {
        "schema_version": SCHEMA_VERSION,
        "program": _serialize_node(program),
    }


# ---------------------------------------------------------------------------
# Program deserialization
# ---------------------------------------------------------------------------


def _resolve_gp_type(type_name: str) -> GPType:
    """Look up a GPType by name, raising SerializationError on failure."""
    try:
        return GPType.get(type_name)
    except KeyError:
        msg = f"Unknown GPType {type_name!r} during deserialization"
        raise SerializationError(msg) from None


def _resolve_primitive(name: str, registry: PrimitiveRegistry) -> Any:
    """Look up a primitive by name, raising SerializationError on failure."""
    try:
        return registry.get(name)
    except Exception:
        msg = f"Primitive {name!r} not found in registry during deserialization"
        raise SerializationError(msg) from None


def _deserialize_node(data: dict[str, Any], registry: PrimitiveRegistry) -> Program:
    """Reconstruct a single AST node from a dict (recursive)."""
    node_type = data.get("type")

    if node_type == "terminal":
        output_type = _resolve_gp_type(data["output_type"])
        return TerminalNode(name=data["name"], output_type=output_type)

    if node_type == "constant":
        output_type = _resolve_gp_type(data["output_type"])
        return ConstantNode(value=float(data["value"]), output_type=output_type)

    if node_type == "function":
        primitive = _resolve_primitive(data["primitive"], registry)
        children = tuple(_deserialize_node(c, registry) for c in data["children"])
        return FunctionNode(primitive=primitive, children=children)

    if node_type == "parameterized":
        primitive = _resolve_primitive(data["primitive"], registry)
        children = tuple(_deserialize_node(c, registry) for c in data["children"])
        # Coerce param values to match the primitive's param_spec dtypes
        raw_params = dict(data.get("params", {}))
        params = _coerce_params(raw_params, primitive)
        return ParameterizedNode(primitive=primitive, children=children, params=params)

    msg = f"Unknown node type {node_type!r} during deserialization"
    raise SerializationError(msg)


def _coerce_params(
    raw_params: dict[str, Any], primitive: Any
) -> dict[str, int | float]:
    """Coerce parameter values to the dtype declared in param_specs."""
    coerced: dict[str, int | float] = {}
    spec_map = {ps.name: ps for ps in primitive.param_specs}
    for key, value in raw_params.items():
        spec = spec_map.get(key)
        if spec is not None and spec.dtype is int:
            coerced[key] = int(value)
        else:
            coerced[key] = float(value) if not isinstance(value, int) else value
    return coerced


def deserialize(data: dict[str, Any], registry: PrimitiveRegistry) -> Program:
    """Deserialize a dict back into a Program tree (FR-9).

    Args:
        data: Dict produced by :func:`serialize`.
        registry: A :class:`PrimitiveRegistry` used to resolve primitive names
                  (FR-9.4).

    Returns:
        The reconstructed Program.

    Raises:
        SerializationError: If a primitive name is not in the registry,
            a GPType is unknown, or the node type is unrecognised.
    """
    _validate_schema_version(data)
    return _deserialize_node(data["program"], registry)


# ---------------------------------------------------------------------------
# EvolutionResult serialization (FR-9.6)
# ---------------------------------------------------------------------------


def _serialize_generation_stats(stats: Any) -> dict[str, Any]:
    """Serialize a GenerationStats to a plain dict."""
    return {
        "generation": int(stats.generation),
        "best_fitness": list(stats.best_fitness),
        "mean_fitness": list(stats.mean_fitness),
        "best_program_size": int(stats.best_program_size),
        "mean_program_size": float(stats.mean_program_size),
        "unique_semantics_ratio": float(stats.unique_semantics_ratio),
        "pareto_front_size": int(stats.pareto_front_size),
    }


def serialize_result(result: Any) -> dict[str, Any]:
    """Serialize an EvolutionResult to a JSON-serializable dict (FR-9.6).

    Args:
        result: An :class:`EvolutionResult` instance.

    Returns:
        A dict with ``schema_version``, ``best_program``, ``pareto_front``,
        ``fitness_history``, and ``config`` keys.
    """
    return {
        "schema_version": SCHEMA_VERSION,
        "best_program": _serialize_node(result.best_program),
        "pareto_front": [_serialize_node(p) for p in result.pareto_front],
        "fitness_history": [
            _serialize_generation_stats(s) for s in result.fitness_history
        ],
        "config": result.config.model_dump(mode="json"),
    }


def _deserialize_generation_stats(data: dict[str, Any]) -> Any:
    """Deserialize a GenerationStats from a dict."""
    from liq.gp.types import GenerationStats

    return GenerationStats(
        generation=int(data["generation"]),
        best_fitness=tuple(data["best_fitness"]),
        mean_fitness=tuple(data["mean_fitness"]),
        best_program_size=int(data["best_program_size"]),
        mean_program_size=float(data["mean_program_size"]),
        unique_semantics_ratio=float(data["unique_semantics_ratio"]),
        pareto_front_size=int(data["pareto_front_size"]),
    )


def deserialize_result(data: dict[str, Any], registry: PrimitiveRegistry) -> Any:
    """Deserialize a dict back into an EvolutionResult (FR-9.6).

    Args:
        data: Dict produced by :func:`serialize_result`.
        registry: A :class:`PrimitiveRegistry` for resolving primitive names.

    Returns:
        The reconstructed EvolutionResult.

    Raises:
        SerializationError: On missing primitives or corrupt data.
    """
    from liq.gp.config import GPConfig
    from liq.gp.types import EvolutionResult

    _validate_schema_version(data)

    best_program = _deserialize_node(data["best_program"], registry)
    pareto_front = [_deserialize_node(p, registry) for p in data["pareto_front"]]
    fitness_history = [
        _deserialize_generation_stats(s) for s in data["fitness_history"]
    ]
    config = GPConfig(**data["config"])

    return EvolutionResult(
        best_program=best_program,
        pareto_front=pareto_front,
        fitness_history=fitness_history,
        config=config,
    )
