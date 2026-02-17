"""Tests for liq.gp.errors module."""

from __future__ import annotations

import pytest

from liq.gp.errors import (
    ConfigurationError,
    ConstantOptError,
    EvaluationError,
    EvolutionError,
    GPError,
    PrimitiveError,
    SerializationError,
    SimplificationError,
    TypeCheckError,
)


class TestExceptionHierarchy:
    """All exceptions derive from GPError."""

    @pytest.mark.parametrize(
        "exc_class",
        [
            PrimitiveError,
            TypeCheckError,
            EvaluationError,
            EvolutionError,
            SimplificationError,
            ConstantOptError,
            SerializationError,
            ConfigurationError,
        ],
    )
    def test_subclass_of_gp_error(self, exc_class: type[GPError]) -> None:
        assert issubclass(exc_class, GPError)
        assert issubclass(exc_class, Exception)

    @pytest.mark.parametrize(
        "exc_class",
        [
            PrimitiveError,
            TypeCheckError,
            EvaluationError,
            EvolutionError,
            SimplificationError,
            ConstantOptError,
            SerializationError,
            ConfigurationError,
        ],
    )
    def test_instantiable_with_message(self, exc_class: type[GPError]) -> None:
        exc = exc_class("test message")
        assert str(exc) == "test message"

    @pytest.mark.parametrize(
        "exc_class",
        [
            PrimitiveError,
            TypeCheckError,
            EvaluationError,
            EvolutionError,
            SimplificationError,
            ConstantOptError,
            SerializationError,
            ConfigurationError,
        ],
    )
    def test_raisable_and_catchable_as_gp_error(self, exc_class: type[GPError]) -> None:
        with pytest.raises(GPError):
            raise exc_class("boom")

    def test_gp_error_is_exception(self) -> None:
        assert issubclass(GPError, Exception)

    def test_type_check_error_is_not_builtin_type_error(self) -> None:
        """TypeCheckError is distinct from Python's built-in TypeError."""
        assert not issubclass(TypeCheckError, TypeError)
