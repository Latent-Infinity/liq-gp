"""Tests verifying all liq.gp subpackages are importable."""

from __future__ import annotations


class TestSubpackageImports:
    """All subpackages must be importable without error."""

    def test_import_root(self) -> None:
        import liq.gp

        assert hasattr(liq.gp, "__version__")

    def test_import_config(self) -> None:
        from liq.gp import config  # noqa: F401

    def test_import_errors(self) -> None:
        from liq.gp import errors  # noqa: F401

    def test_import_types(self) -> None:
        from liq.gp import types  # noqa: F401

    def test_import_protocols(self) -> None:
        from liq.gp import protocols  # noqa: F401

    def test_import_primitives(self) -> None:
        from liq.gp import primitives  # noqa: F401
        from liq.gp.primitives import registry  # noqa: F401

    def test_import_program(self) -> None:
        from liq.gp import program  # noqa: F401
        from liq.gp.program import (
            ast,  # noqa: F401
            eval,  # noqa: F401
        )

    def test_import_evolution(self) -> None:
        from liq.gp import evolution  # noqa: F401


class TestPublicAPIReexports:
    """All public API items from __all__ must be importable from liq.gp."""

    def test_all_exports_importable(self) -> None:
        import liq.gp

        for name in liq.gp.__all__:
            assert hasattr(liq.gp, name), f"{name} not importable from liq.gp"

    def test_config_exports(self) -> None:
        from liq.gp import FitnessConfig, GPConfig

        assert GPConfig is not None
        assert FitnessConfig is not None

    def test_type_exports(self) -> None:
        from liq.gp import (
            BoolSeries,
            EvolutionResult,
            FitnessResult,
            GenerationStats,
            Int,
            ParamSpec,
            Scalar,
            Series,
        )

        assert all(
            t is not None
            for t in [
                Series,
                BoolSeries,
                Scalar,
                Int,
                ParamSpec,
                FitnessResult,
                GenerationStats,
                EvolutionResult,
            ]
        )

    def test_error_exports(self) -> None:
        from liq.gp import (
            ConfigurationError,
            EvaluationError,
            GPError,
            PrimitiveError,
            SerializationError,
            TypeCheckError,
        )

        assert all(
            issubclass(e, GPError)
            for e in [
                PrimitiveError,
                TypeCheckError,
                EvaluationError,
                SerializationError,
                ConfigurationError,
            ]
        )

    def test_primitive_exports(self) -> None:
        from liq.gp import PrimitiveInfo, PrimitiveRegistry

        assert PrimitiveInfo is not None
        assert PrimitiveRegistry is not None

    def test_ast_exports(self) -> None:
        from liq.gp import (
            ConstantNode,
            FunctionNode,
            ParameterizedNode,
            Program,
            TerminalNode,
        )

        assert all(
            t is not None
            for t in [
                Program,
                TerminalNode,
                ConstantNode,
                FunctionNode,
                ParameterizedNode,
            ]
        )

    def test_protocol_exports(self) -> None:
        from liq.gp import FitnessEvaluator

        assert FitnessEvaluator is not None
