"""Tests for liq.gp.types module."""

from __future__ import annotations

import pytest

from liq.gp.types import (
    BoolSeries,
    EvolutionResult,
    FitnessResult,
    GenerationStats,
    GPType,
    Int,
    ParamSpec,
    Scalar,
    Series,
)


class TestGPTypeBuiltins:
    """Built-in types are available and correct."""

    def test_series_exists(self) -> None:
        assert Series.name == "Series"
        assert isinstance(Series, GPType)

    def test_bool_series_exists(self) -> None:
        assert BoolSeries.name == "BoolSeries"
        assert isinstance(BoolSeries, GPType)

    def test_scalar_exists(self) -> None:
        assert Scalar.name == "Scalar"
        assert isinstance(Scalar, GPType)

    def test_int_exists(self) -> None:
        assert Int.name == "Int"
        assert isinstance(Int, GPType)

    def test_all_four_in_registry(self) -> None:
        all_types = GPType.all_types()
        assert set(all_types.keys()) == {"Series", "BoolSeries", "Scalar", "Int"}

    def test_get_by_name(self) -> None:
        assert GPType.get("Series") is Series
        assert GPType.get("BoolSeries") is BoolSeries
        assert GPType.get("Scalar") is Scalar
        assert GPType.get("Int") is Int

    def test_get_unknown_raises(self) -> None:
        with pytest.raises(KeyError):
            GPType.get("Unknown")


class TestGPTypeEquality:
    """Type equality and hashing."""

    def test_same_name_equal(self) -> None:
        a = GPType("X", "test")
        b = GPType("X", "different desc")
        assert a == b

    def test_different_name_not_equal(self) -> None:
        a = GPType("X", "test")
        b = GPType("Y", "test")
        assert a != b

    def test_hashable(self) -> None:
        types_set = {Series, BoolSeries, Scalar, Int}
        assert len(types_set) == 4

    def test_repr(self) -> None:
        assert repr(Series) == "GPType('Series')"

    def test_description(self) -> None:
        assert "float" in Series.description.lower()

    def test_not_equal_to_non_gptype(self) -> None:
        assert Series != "Series"
        assert Series.__eq__("Series") is NotImplemented


class TestGPTypeExtensibility:
    """Consumers can register additional types (FR-1.2)."""

    def test_register_new_type(self) -> None:
        custom = GPType.register_type("CustomSeries", "A custom type")
        assert isinstance(custom, GPType)
        assert custom.name == "CustomSeries"
        assert GPType.get("CustomSeries") is custom

    def test_register_duplicate_raises(self) -> None:
        GPType.register_type("Dup", "first")
        with pytest.raises(ValueError, match="already registered"):
            GPType.register_type("Dup", "second")

    def test_registered_type_usable_in_sets(self) -> None:
        custom = GPType.register_type("SetTest", "test")
        types_set = {Series, custom}
        assert custom in types_set

    def test_reset_removes_custom_types(self) -> None:
        GPType.register_type("Temp", "temporary")
        assert "Temp" in GPType.all_types()
        GPType._reset_registry()
        assert "Temp" not in GPType.all_types()
        # Built-ins remain
        assert "Series" in GPType.all_types()


class TestParamSpec:
    """ParamSpec validation (FR-3.3)."""

    def test_valid_int_param(self) -> None:
        p = ParamSpec(name="period", dtype=int, default=20, min_value=2, max_value=200)
        assert p.name == "period"
        assert p.dtype is int
        assert p.default == 20

    def test_valid_float_param(self) -> None:
        p = ParamSpec(
            name="alpha", dtype=float, default=0.5, min_value=0.0, max_value=1.0
        )
        assert p.dtype is float

    def test_invalid_dtype_raises(self) -> None:
        with pytest.raises(TypeError, match="dtype must be int or float"):
            ParamSpec(name="bad", dtype=str, default="x", min_value=0, max_value=1)

    def test_min_greater_than_max_raises(self) -> None:
        with pytest.raises(ValueError, match="min_value"):
            ParamSpec(name="bad", dtype=int, default=5, min_value=10, max_value=5)

    def test_frozen(self) -> None:
        p = ParamSpec(name="x", dtype=int, default=1, min_value=0, max_value=10)
        with pytest.raises(AttributeError):
            p.name = "y"  # type: ignore[misc]

    def test_discrete_allowed_values(self) -> None:
        p = ParamSpec(
            name="period",
            dtype=int,
            default=13,
            allowed_values=[34, 2, 13, 3, 55, 21],
        )
        assert p.value_is_discrete()
        assert p.allowed_values == [2, 3, 13, 21, 34, 55]

    def test_discrete_defaults_sorted(self) -> None:
        p = ParamSpec(
            name="alpha",
            dtype=float,
            default=0.5,
            allowed_values=[1.0, 0.2, 0.2, 0.5, 1.0],
        )
        assert p.allowed_values == [0.2, 0.5, 1.0]

    def test_discrete_default_must_be_in_values(self) -> None:
        with pytest.raises(
            ValueError, match="default \\(13\\) must be in allowed_values"
        ):
            ParamSpec(
                name="period",
                dtype=int,
                default=13,
                allowed_values=[2, 3, 5],
            )

    def test_discrete_empty_values_rejected(self) -> None:
        with pytest.raises(
            ValueError, match="allowed_values must be a non-empty sequence"
        ):
            ParamSpec(name="period", dtype=int, default=2, allowed_values=[])

    def test_discrete_min_max_optional(self) -> None:
        p = ParamSpec(name="period", dtype=int, default=13, allowed_values=[13, 21, 34])
        assert p.min_value is None
        assert p.max_value is None


class TestFitnessResult:
    """FitnessResult immutability and fields (FR-5.4.2)."""

    def test_single_objective(self) -> None:
        r = FitnessResult(objectives=(0.95,))
        assert r.objectives == (0.95,)
        assert r.metadata == {}

    def test_multi_objective(self) -> None:
        r = FitnessResult(objectives=(0.95, 12.3), metadata={"sharpe": 1.5})
        assert len(r.objectives) == 2
        assert r.metadata["sharpe"] == 1.5

    def test_frozen(self) -> None:
        r = FitnessResult(objectives=(1.0,))
        with pytest.raises(AttributeError):
            r.objectives = (2.0,)  # type: ignore[misc]


class TestGenerationStats:
    """GenerationStats fields (FR-5.5.3)."""

    def test_all_fields(self) -> None:
        stats = GenerationStats(
            generation=5,
            best_fitness=(0.95, 1.2),
            mean_fitness=(0.80, 2.1),
            best_program_size=12,
            mean_program_size=18.5,
            unique_semantics_ratio=0.85,
            pareto_front_size=10,
        )
        assert stats.generation == 5
        assert stats.best_fitness == (0.95, 1.2)
        assert stats.mean_fitness == (0.80, 2.1)
        assert stats.best_program_size == 12
        assert stats.mean_program_size == 18.5
        assert stats.unique_semantics_ratio == 0.85
        assert stats.pareto_front_size == 10
        assert stats.scheduler_metrics == {}

    def test_frozen(self) -> None:
        stats = GenerationStats(
            generation=0,
            best_fitness=(0.0,),
            mean_fitness=(0.0,),
            best_program_size=1,
            mean_program_size=1.0,
            unique_semantics_ratio=1.0,
            pareto_front_size=0,
        )
        with pytest.raises(AttributeError):
            stats.generation = 1  # type: ignore[misc]

    def test_scheduler_metrics_payload(self) -> None:
        stats = GenerationStats(
            generation=1,
            best_fitness=(1.0,),
            mean_fitness=(0.5,),
            best_program_size=2,
            mean_program_size=2.5,
            unique_semantics_ratio=1.0,
            pareto_front_size=1,
            scheduler_metrics={"mode": "bounded", "peak_in_flight": 2},
        )
        assert stats.scheduler_metrics["mode"] == "bounded"


class TestEvolutionResult:
    """EvolutionResult fields (FR-5.5.2)."""

    def test_all_fields(self) -> None:
        stats = GenerationStats(
            generation=0,
            best_fitness=(1.0,),
            mean_fitness=(0.5,),
            best_program_size=5,
            mean_program_size=10.0,
            unique_semantics_ratio=0.9,
            pareto_front_size=1,
        )
        result = EvolutionResult(
            best_program="mock_program",
            pareto_front=["mock_program"],
            fitness_history=[stats],
            config="mock_config",
        )
        assert result.best_program == "mock_program"
        assert len(result.pareto_front) == 1
        assert len(result.fitness_history) == 1
        assert result.config == "mock_config"

    def test_frozen(self) -> None:
        result = EvolutionResult(
            best_program=None,
            pareto_front=[],
            fitness_history=[],
            config=None,
        )
        with pytest.raises(AttributeError):
            result.best_program = "new"  # type: ignore[misc]
