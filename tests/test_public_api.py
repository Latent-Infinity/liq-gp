"""Tests verifying the public API surface matches requirements section 10."""

from __future__ import annotations

import liq.gp

# ---------------------------------------------------------------------------
# Section 10: All listed items importable from liq.gp
# ---------------------------------------------------------------------------


class TestPublicAPIImports:
    """Every item listed in requirements section 10 is importable from liq.gp."""

    # -- Configuration --

    def test_gpconfig(self) -> None:
        from liq.gp import GPConfig

        assert GPConfig is not None

    def test_fitness_config(self) -> None:
        from liq.gp import FitnessConfig

        assert FitnessConfig is not None

    # -- Types --

    def test_series(self) -> None:
        from liq.gp import Series

        assert Series is not None

    def test_bool_series(self) -> None:
        from liq.gp import BoolSeries

        assert BoolSeries is not None

    def test_scalar(self) -> None:
        from liq.gp import Scalar

        assert Scalar is not None

    def test_int_type(self) -> None:
        from liq.gp import Int

        assert Int is not None

    def test_fitness_result(self) -> None:
        from liq.gp import FitnessResult

        assert FitnessResult is not None

    def test_evolution_result(self) -> None:
        from liq.gp import EvolutionResult

        assert EvolutionResult is not None

    def test_generation_stats(self) -> None:
        from liq.gp import GenerationStats

        assert GenerationStats is not None

    # -- Primitives --

    def test_primitive_registry(self) -> None:
        from liq.gp import PrimitiveRegistry

        assert PrimitiveRegistry is not None

    def test_primitive_info(self) -> None:
        from liq.gp import PrimitiveInfo

        assert PrimitiveInfo is not None

    def test_param_spec(self) -> None:
        from liq.gp import ParamSpec

        assert ParamSpec is not None

    # -- Program / AST --

    def test_program(self) -> None:
        from liq.gp import Program

        assert Program is not None

    def test_terminal_node(self) -> None:
        from liq.gp import TerminalNode

        assert TerminalNode is not None

    def test_function_node(self) -> None:
        from liq.gp import FunctionNode

        assert FunctionNode is not None

    def test_parameterized_node(self) -> None:
        from liq.gp import ParameterizedNode

        assert ParameterizedNode is not None

    def test_constant_node(self) -> None:
        from liq.gp import ConstantNode

        assert ConstantNode is not None

    def test_evaluate(self) -> None:
        from liq.gp import evaluate

        assert callable(evaluate)

    def test_simplify(self) -> None:
        from liq.gp import simplify

        assert callable(simplify)

    def test_optimize_constants(self) -> None:
        from liq.gp import optimize_constants

        assert callable(optimize_constants)

    def test_serialize(self) -> None:
        from liq.gp import serialize

        assert callable(serialize)

    def test_deserialize(self) -> None:
        from liq.gp import deserialize

        assert callable(deserialize)

    def test_serialize_result(self) -> None:
        from liq.gp import serialize_result

        assert callable(serialize_result)

    def test_deserialize_result(self) -> None:
        from liq.gp import deserialize_result

        assert callable(deserialize_result)

    # -- Evolution --

    def test_evolve(self) -> None:
        from liq.gp import evolve

        assert callable(evolve)

    def test_fitness_evaluator(self) -> None:
        from liq.gp import FitnessEvaluator

        assert FitnessEvaluator is not None

    def test_generation_callback(self) -> None:
        from liq.gp import GenerationCallback

        assert GenerationCallback is not None

    def test_validate_seed_programs(self) -> None:
        from liq.gp import validate_seed_programs

        assert callable(validate_seed_programs)

    def test_initialize_seeded_population(self) -> None:
        from liq.gp import initialize_seeded_population

        assert callable(initialize_seeded_population)

    # -- Errors --

    def test_gp_error(self) -> None:
        from liq.gp import GPError

        assert issubclass(GPError, Exception)

    def test_primitive_error(self) -> None:
        from liq.gp import PrimitiveError

        assert issubclass(PrimitiveError, Exception)

    def test_type_check_error(self) -> None:
        from liq.gp import TypeCheckError

        assert issubclass(TypeCheckError, Exception)

    def test_evaluation_error(self) -> None:
        from liq.gp import EvaluationError

        assert issubclass(EvaluationError, Exception)

    def test_serialization_error(self) -> None:
        from liq.gp import SerializationError

        assert issubclass(SerializationError, Exception)

    def test_configuration_error(self) -> None:
        from liq.gp import ConfigurationError

        assert issubclass(ConfigurationError, Exception)

    def test_evolution_error(self) -> None:
        from liq.gp import EvolutionError

        assert issubclass(EvolutionError, Exception)


# ---------------------------------------------------------------------------
# __all__ completeness
# ---------------------------------------------------------------------------


class TestAllExports:
    """__all__ contains exactly the public API items from section 10."""

    REQUIRED_EXPORTS = {
        # Configuration
        "GPConfig",
        "FitnessConfig",
        "SeedInjectionConfig",
        # Types
        "Series",
        "BoolSeries",
        "Scalar",
        "Int",
        "FitnessResult",
        "EvolutionResult",
        "GenerationStats",
        # Primitives
        "PrimitiveRegistry",
        "PrimitiveInfo",
        "ParamSpec",
        # Program / AST
        "Program",
        "TerminalNode",
        "FunctionNode",
        "ParameterizedNode",
        "ConstantNode",
        "evaluate",
        "simplify",
        "optimize_constants",
        "serialize",
        "deserialize",
        "serialize_result",
        "deserialize_result",
        # Evolution
        "evolve",
        "validate_seed_programs",
        "initialize_seeded_population",
        "inject_seeds",
        "FitnessEvaluator",
        "GenerationCallback",
        # Errors
        "GPError",
        "PrimitiveError",
        "TypeCheckError",
        "EvaluationError",
        "EvolutionError",
        "SerializationError",
        "ConfigurationError",
    }

    def test_all_contains_required(self) -> None:
        all_set = set(liq.gp.__all__)
        missing = self.REQUIRED_EXPORTS - all_set
        assert not missing, f"Missing from __all__: {missing}"
        unexpected = all_set - self.REQUIRED_EXPORTS
        assert not unexpected, f"Unexpected extra exports in __all__: {unexpected}"

    def test_all_items_importable(self) -> None:
        for name in liq.gp.__all__:
            assert hasattr(liq.gp, name), f"{name} in __all__ but not importable"

    def test_version_defined(self) -> None:
        assert hasattr(liq.gp, "__version__")
        assert isinstance(liq.gp.__version__, str)
