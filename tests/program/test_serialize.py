"""Tests for program serialization/deserialization (FR-9)."""

from __future__ import annotations

import json

import pytest

from liq.gp.config import GPConfig
from liq.gp.errors import SerializationError
from liq.gp.primitives.registry import PrimitiveInfo, PrimitiveRegistry
from liq.gp.program.ast import (
    ConstantNode,
    FunctionNode,
    ParameterizedNode,
    TerminalNode,
)
from liq.gp.program.serialize import (
    deserialize,
    deserialize_result,
    serialize,
    serialize_result,
)
from liq.gp.types import (
    BoolSeries,
    EvolutionResult,
    GenerationStats,
    Int,
    ParamSpec,
    Series,
)

# --- helpers ---------------------------------------------------------------


def _make_registry() -> PrimitiveRegistry:
    """Build a small registry for round-trip tests."""
    reg = PrimitiveRegistry()
    reg.register(
        "close", lambda: None, input_types=(), output_type=Series, category="data"
    )
    reg.register(
        "volume", lambda: None, input_types=(), output_type=Series, category="data"
    )
    reg.register(
        "add",
        lambda a, b: a + b,
        input_types=(Series, Series),
        output_type=Series,
        category="numeric",
    )
    reg.register(
        "neg",
        lambda a: -a,
        input_types=(Series,),
        output_type=Series,
        category="numeric",
    )
    reg.register(
        "gt",
        lambda a, b: (a > b).astype(float),
        input_types=(Series, Series),
        output_type=BoolSeries,
        category="comparison",
    )
    ps = ParamSpec(name="period", dtype=int, default=20, min_value=2, max_value=200)
    reg.register(
        "highest",
        lambda a, period=20: a,
        input_types=(Series,),
        output_type=Series,
        category="indicator",
        param_specs=[ps],
    )
    return reg


def _make_add_info(reg: PrimitiveRegistry) -> PrimitiveInfo:
    return reg.get("add")


def _make_neg_info(reg: PrimitiveRegistry) -> PrimitiveInfo:
    return reg.get("neg")


def _make_gt_info(reg: PrimitiveRegistry) -> PrimitiveInfo:
    return reg.get("gt")


def _make_highest_info(reg: PrimitiveRegistry) -> PrimitiveInfo:
    return reg.get("highest")


# --- TestSerializeTerminalNode -----------------------------------------------


class TestSerializeTerminalNode:
    """Serialize/deserialize TerminalNode."""

    def test_serialize_terminal(self) -> None:
        node = TerminalNode(name="close", output_type=Series)
        data = serialize(node)
        program_data = data["program"]
        assert program_data["type"] == "terminal"
        assert program_data["name"] == "close"
        assert program_data["output_type"] == "Series"

    def test_round_trip_terminal(self) -> None:
        reg = _make_registry()
        node = TerminalNode(name="close", output_type=Series)
        data = serialize(node)
        restored = deserialize(data, reg)
        assert restored == node

    def test_round_trip_terminal_bool_series(self) -> None:
        """Terminal with BoolSeries output type."""
        reg = _make_registry()
        reg.register(
            "flag",
            lambda: None,
            input_types=(),
            output_type=BoolSeries,
            category="data",
        )
        node = TerminalNode(name="flag", output_type=BoolSeries)
        data = serialize(node)
        restored = deserialize(data, reg)
        assert restored == node
        assert restored.output_type == BoolSeries


# --- TestSerializeConstantNode -----------------------------------------------


class TestSerializeConstantNode:
    """Serialize/deserialize ConstantNode."""

    def test_serialize_constant(self) -> None:
        node = ConstantNode(value=2.5)
        data = serialize(node)
        program_data = data["program"]
        assert program_data["type"] == "constant"
        assert program_data["value"] == 2.5
        assert program_data["output_type"] == "Series"

    def test_round_trip_constant(self) -> None:
        reg = _make_registry()
        node = ConstantNode(value=3.14)
        data = serialize(node)
        restored = deserialize(data, reg)
        assert restored == node

    def test_round_trip_constant_with_int_type(self) -> None:
        """Constant with non-default output type."""
        reg = _make_registry()
        node = ConstantNode(value=5.0, output_type=Int)
        data = serialize(node)
        restored = deserialize(data, reg)
        assert restored == node
        assert restored.output_type == Int


# --- TestSerializeFunctionNode -----------------------------------------------


class TestSerializeFunctionNode:
    """Serialize/deserialize FunctionNode."""

    def test_serialize_function(self) -> None:
        reg = _make_registry()
        close = TerminalNode(name="close", output_type=Series)
        volume = TerminalNode(name="volume", output_type=Series)
        add_info = _make_add_info(reg)
        node = FunctionNode(primitive=add_info, children=(close, volume))
        data = serialize(node)
        program_data = data["program"]
        assert program_data["type"] == "function"
        assert program_data["primitive"] == "add"
        assert len(program_data["children"]) == 2
        assert program_data["children"][0]["type"] == "terminal"
        assert program_data["children"][1]["type"] == "terminal"

    def test_round_trip_function(self) -> None:
        reg = _make_registry()
        close = TerminalNode(name="close", output_type=Series)
        volume = TerminalNode(name="volume", output_type=Series)
        add_info = _make_add_info(reg)
        node = FunctionNode(primitive=add_info, children=(close, volume))
        data = serialize(node)
        restored = deserialize(data, reg)
        assert restored == node


# --- TestSerializeParameterizedNode ------------------------------------------


class TestSerializeParameterizedNode:
    """Serialize/deserialize ParameterizedNode."""

    def test_serialize_parameterized(self) -> None:
        reg = _make_registry()
        close = TerminalNode(name="close", output_type=Series)
        highest_info = _make_highest_info(reg)
        node = ParameterizedNode(
            primitive=highest_info,
            children=(close,),
            params={"period": 20},
        )
        data = serialize(node)
        program_data = data["program"]
        assert program_data["type"] == "parameterized"
        assert program_data["primitive"] == "highest"
        assert program_data["params"] == {"period": 20}
        assert len(program_data["children"]) == 1

    def test_round_trip_parameterized(self) -> None:
        reg = _make_registry()
        close = TerminalNode(name="close", output_type=Series)
        highest_info = _make_highest_info(reg)
        node = ParameterizedNode(
            primitive=highest_info,
            children=(close,),
            params={"period": 30},
        )
        data = serialize(node)
        restored = deserialize(data, reg)
        assert restored == node
        assert restored.parameters == {"period": 30}


# --- TestSchemaVersion -------------------------------------------------------


class TestSchemaVersion:
    """Schema version is present in serialized output (FR-9.2)."""

    def test_schema_version_present(self) -> None:
        node = TerminalNode(name="close", output_type=Series)
        data = serialize(node)
        assert "schema_version" in data
        assert data["schema_version"] == "1.0.0"

    def test_program_key_present(self) -> None:
        node = TerminalNode(name="close", output_type=Series)
        data = serialize(node)
        assert "program" in data

    def test_deserialize_rejects_unknown_schema_version(self) -> None:
        reg = _make_registry()
        data = {
            "schema_version": "9.9.9",
            "program": {"type": "terminal", "name": "close", "output_type": "Series"},
        }
        with pytest.raises(SerializationError, match="Unsupported schema_version"):
            deserialize(data, reg)

    def test_deserialize_rejects_missing_schema_version(self) -> None:
        reg = _make_registry()
        data = {
            "program": {"type": "terminal", "name": "close", "output_type": "Series"},
        }
        with pytest.raises(SerializationError, match="schema_version"):
            deserialize(data, reg)


# --- TestMissingPrimitive ----------------------------------------------------


class TestMissingPrimitive:
    """Deserialization raises SerializationError for missing primitives (FR-9.4)."""

    def test_missing_function_primitive(self) -> None:
        data = {
            "schema_version": "1.0.0",
            "program": {
                "type": "function",
                "primitive": "nonexistent_func",
                "children": [
                    {"type": "terminal", "name": "close", "output_type": "Series"},
                ],
            },
        }
        reg = _make_registry()
        with pytest.raises(SerializationError, match="nonexistent_func"):
            deserialize(data, reg)

    def test_missing_parameterized_primitive(self) -> None:
        data = {
            "schema_version": "1.0.0",
            "program": {
                "type": "parameterized",
                "primitive": "unknown_indicator",
                "children": [
                    {"type": "terminal", "name": "close", "output_type": "Series"},
                ],
                "params": {"period": 10},
            },
        }
        reg = _make_registry()
        with pytest.raises(SerializationError, match="unknown_indicator"):
            deserialize(data, reg)


# --- TestComplexNestedTree ---------------------------------------------------


class TestComplexNestedTree:
    """Complex nested trees round-trip correctly (FR-9.5)."""

    def test_deeply_nested_tree(self) -> None:
        """neg(add(highest(close, period=20), 3.14)) round-trips."""
        reg = _make_registry()
        close = TerminalNode(name="close", output_type=Series)
        highest_info = _make_highest_info(reg)
        add_info = _make_add_info(reg)
        neg_info = _make_neg_info(reg)

        highest_node = ParameterizedNode(
            primitive=highest_info,
            children=(close,),
            params={"period": 20},
        )
        const = ConstantNode(value=3.14)
        add_node = FunctionNode(
            primitive=add_info,
            children=(highest_node, const),
        )
        tree = FunctionNode(primitive=neg_info, children=(add_node,))

        data = serialize(tree)
        restored = deserialize(data, reg)
        assert restored == tree

    def test_tree_with_multiple_constants(self) -> None:
        """Tree with several constant nodes round-trips."""
        reg = _make_registry()
        c1 = ConstantNode(value=1.0)
        c2 = ConstantNode(value=2.0)
        add_info = _make_add_info(reg)
        tree = FunctionNode(primitive=add_info, children=(c1, c2))

        data = serialize(tree)
        restored = deserialize(data, reg)
        assert restored == tree

    def test_tree_with_bool_output(self) -> None:
        """gt(close, volume) produces BoolSeries."""
        reg = _make_registry()
        close = TerminalNode(name="close", output_type=Series)
        volume = TerminalNode(name="volume", output_type=Series)
        gt_info = _make_gt_info(reg)
        tree = FunctionNode(primitive=gt_info, children=(close, volume))

        data = serialize(tree)
        restored = deserialize(data, reg)
        assert restored == tree
        assert restored.output_type == BoolSeries


# --- TestDeterministicOutput -------------------------------------------------


class TestDeterministicOutput:
    """Serialization output is deterministic."""

    def test_serialize_same_node_twice(self) -> None:
        node = TerminalNode(name="close", output_type=Series)
        d1 = serialize(node)
        d2 = serialize(node)
        assert d1 == d2

    def test_serialized_is_json_compatible(self) -> None:
        """Serialized dict is JSON-encodable (no numpy types, etc.)."""
        reg = _make_registry()
        close = TerminalNode(name="close", output_type=Series)
        highest_info = _make_highest_info(reg)
        tree = ParameterizedNode(
            primitive=highest_info,
            children=(close,),
            params={"period": 20},
        )
        data = serialize(tree)
        # Should not raise
        json_str = json.dumps(data)
        assert isinstance(json_str, str)
        # And can be parsed back to identical dict
        assert json.loads(json_str) == data


# --- TestEvolutionResult -----------------------------------------------------


class TestEvolutionResult:
    """EvolutionResult serialization (FR-9.6)."""

    def _make_evolution_result(self, reg: PrimitiveRegistry) -> EvolutionResult:
        close = TerminalNode(name="close", output_type=Series)
        volume = TerminalNode(name="volume", output_type=Series)
        add_info = _make_add_info(reg)
        best = FunctionNode(primitive=add_info, children=(close, volume))
        pareto_front = [best, close]
        stats = GenerationStats(
            generation=0,
            best_fitness=(0.95,),
            mean_fitness=(0.5,),
            best_program_size=3,
            mean_program_size=5.0,
            unique_semantics_ratio=0.8,
            pareto_front_size=2,
        )
        config = GPConfig()
        return EvolutionResult(
            best_program=best,
            pareto_front=pareto_front,
            fitness_history=[stats],
            config=config,
        )

    def test_serialize_result_schema_version(self) -> None:
        reg = _make_registry()
        result = self._make_evolution_result(reg)
        data = serialize_result(result)
        assert data["schema_version"] == "1.0.0"

    def test_serialize_result_has_required_keys(self) -> None:
        reg = _make_registry()
        result = self._make_evolution_result(reg)
        data = serialize_result(result)
        assert "best_program" in data
        assert "pareto_front" in data
        assert "fitness_history" in data
        assert "config" in data

    def test_round_trip_evolution_result(self) -> None:
        reg = _make_registry()
        result = self._make_evolution_result(reg)
        data = serialize_result(result)
        restored = deserialize_result(data, reg)
        assert restored.best_program == result.best_program
        assert len(restored.pareto_front) == len(result.pareto_front)
        for orig, rest in zip(result.pareto_front, restored.pareto_front, strict=True):
            assert orig == rest
        assert len(restored.fitness_history) == len(result.fitness_history)
        assert restored.fitness_history[0].generation == 0
        assert restored.fitness_history[0].best_fitness == (0.95,)
        assert restored.config == result.config

    def test_deserialize_result_rejects_unknown_schema(self) -> None:
        reg = _make_registry()
        result = self._make_evolution_result(reg)
        data = serialize_result(result)
        data["schema_version"] = "2.0.0"
        with pytest.raises(SerializationError, match="Unsupported schema_version"):
            deserialize_result(data, reg)

    def test_evolution_result_json_compatible(self) -> None:
        """Serialized EvolutionResult is JSON-encodable."""
        reg = _make_registry()
        result = self._make_evolution_result(reg)
        data = serialize_result(result)
        json_str = json.dumps(data)
        assert isinstance(json_str, str)
        assert json.loads(json_str) == data


# --- TestInvalidNodeType -----------------------------------------------------


class TestInvalidNodeType:
    """Deserialization rejects unknown node types."""

    def test_unknown_node_type_raises(self) -> None:
        data = {
            "schema_version": "1.0.0",
            "program": {
                "type": "unknown_type",
                "name": "foo",
                "output_type": "Series",
            },
        }
        reg = _make_registry()
        with pytest.raises(SerializationError, match="unknown_type"):
            deserialize(data, reg)

    def test_unknown_gp_type_raises(self) -> None:
        """Deserializing a terminal with an unregistered GPType raises."""
        data = {
            "schema_version": "1.0.0",
            "program": {
                "type": "terminal",
                "name": "close",
                "output_type": "UnknownType",
            },
        }
        reg = _make_registry()
        with pytest.raises(SerializationError, match="UnknownType"):
            deserialize(data, reg)
