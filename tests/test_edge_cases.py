"""Tests for error conditions and edge cases in the GP engine."""

from __future__ import annotations

import numpy as np
import pytest

from liq.gp import (
    ConfigurationError,
    ConstantNode,
    EvolutionResult,
    FitnessResult,
    FunctionNode,
    GPConfig,
    PrimitiveError,
    PrimitiveRegistry,
    SerializationError,
    Series,
    TerminalNode,
    deserialize,
    evaluate,
    evolve,
)
from liq.gp.evolution.init import initialize_population
from liq.gp.program.ast import Program
from liq.gp.program.constants import (
    extract_constants,
    inject_constants,
    optimize_constants,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _minimal_registry() -> PrimitiveRegistry:
    """Build a minimal registry with terminals and functions."""
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
        "neg",
        lambda a: -a,
        category="numeric",
        input_types=(Series,),
        output_type=Series,
    )
    return reg


def _minimal_config(**overrides: object) -> GPConfig:
    """Build a GPConfig with small test defaults."""
    defaults: dict[str, object] = {
        "population_size": 20,
        "max_depth": 4,
        "generations": 3,
        "seed": 42,
        "constant_opt_enabled": False,
        "simplification_enabled": False,
        "elitism_count": 2,
        "tournament_size": 3,
    }
    defaults.update(overrides)
    return GPConfig(**defaults)  # type: ignore[arg-type]


def _make_context(n: int = 50) -> dict[str, np.ndarray]:
    """Build a simple evaluation context."""
    rng = np.random.default_rng(0)
    return {"x": rng.uniform(-1.0, 1.0, size=n)}


class SimpleFitnessEvaluator:
    """Evaluator that measures MSE fitness for y = 2*x."""

    def evaluate(
        self,
        programs: list[Program],
        context: dict[str, np.ndarray],
    ) -> list[FitnessResult]:
        target = 2.0 * context["x"]
        results: list[FitnessResult] = []
        for prog in programs:
            try:
                output = evaluate(prog, context)
                mse = float(np.mean((output - target) ** 2))
                results.append(FitnessResult(objectives=(-mse,)))
            except Exception:
                results.append(FitnessResult(objectives=(-1e10,)))
        return results


# ===========================================================================
# 1. TestEmptyRegistry
# ===========================================================================


class TestEmptyRegistry:
    """An empty or terminal-only registry should fail gracefully."""

    def test_empty_registry_raises_on_population_init(self) -> None:
        """An empty registry (no primitives) should raise PrimitiveError
        when attempting to initialize a population, because generate_full
        cannot find any function primitives."""
        reg = PrimitiveRegistry()
        config = _minimal_config()
        with pytest.raises(PrimitiveError):
            initialize_population(reg, config)

    def test_terminals_only_registry_raises_on_full_generation(self) -> None:
        """A registry with only terminals (no functions) should raise
        PrimitiveError when generate_full tries to pick a function for
        depth > 0.  With max_depth >= 2, at least some individuals in
        initialize_population use generate_full at depth >= 1."""
        reg = PrimitiveRegistry()
        reg.register("x", lambda: None, input_types=(), output_type=Series)
        config = _minimal_config(max_depth=2)
        # generate_full at depth >= 1 calls _sample_function which raises
        with pytest.raises(PrimitiveError, match="No functions available"):
            initialize_population(reg, config)


# ===========================================================================
# 2. TestInvalidConfigurations
# ===========================================================================


class TestInvalidConfigurations:
    """GPConfig validates constraints and raises ConfigurationError."""

    def test_population_size_too_small(self) -> None:
        """population_size < 10 should raise ConfigurationError."""
        with pytest.raises(ConfigurationError, match="population_size"):
            GPConfig(population_size=5)

    def test_max_depth_too_small(self) -> None:
        """max_depth < 2 should raise ConfigurationError."""
        with pytest.raises(ConfigurationError, match="max_depth"):
            GPConfig(max_depth=1)

    def test_tournament_size_exceeds_population(self) -> None:
        """tournament_size > population_size should raise ConfigurationError."""
        with pytest.raises(ConfigurationError, match="tournament_size"):
            GPConfig(population_size=10, tournament_size=20)

    def test_operator_rates_not_summing_to_one(self) -> None:
        """Operator rates that do not sum to 1.0 should raise ConfigurationError."""
        with pytest.raises(ConfigurationError, match="sum to 1.0"):
            GPConfig(
                crossover_rate=0.5,
                subtree_mutation_rate=0.1,
                point_mutation_rate=0.1,
                parameter_mutation_rate=0.05,
                hoist_mutation_rate=0.05,
            )


class TestContextValidation:
    """Input-context validation errors in evolve() are explicit and early."""

    def test_evolve_rejects_empty_context(self) -> None:
        reg = _minimal_registry()
        config = _minimal_config()

        class StubEvaluator:
            def evaluate(
                self,
                programs: list[Program],
                context: dict[str, np.ndarray],
            ) -> list[FitnessResult]:
                return [FitnessResult(objectives=(0.0,)) for _ in programs]

        with pytest.raises(ValueError, match="at least one array"):
            evolve(reg, config, StubEvaluator(), {})

    def test_evolve_rejects_non_ndarray_context_entry(self) -> None:
        reg = _minimal_registry()
        config = _minimal_config()
        context = {"x": [1, 2, 3]}

        class StubEvaluator:
            def evaluate(
                self,
                programs: list[Program],
                context: dict[str, np.ndarray],
            ) -> list[FitnessResult]:
                return [FitnessResult(objectives=(0.0,)) for _ in programs]

        with pytest.raises(TypeError, match="must be a numpy.ndarray"):
            evolve(reg, config, StubEvaluator(), context)

    def test_evolve_rejects_non_1d_context(self) -> None:
        reg = _minimal_registry()
        config = _minimal_config()
        context = {"x": np.array([[1, 2], [3, 4]])}

        class StubEvaluator:
            def evaluate(
                self,
                programs: list[Program],
                context: dict[str, np.ndarray],
            ) -> list[FitnessResult]:
                return [FitnessResult(objectives=(0.0,)) for _ in programs]

        with pytest.raises(ValueError, match="must be 1D"):
            evolve(reg, config, StubEvaluator(), context)

    def test_evolve_rejects_mismatched_context_lengths(self) -> None:
        reg = _minimal_registry()
        config = _minimal_config()
        context = {
            "x": np.array([1, 2, 3], dtype=np.float64),
            "y": np.array([1, 2], dtype=np.float64),
        }

        class StubEvaluator:
            def evaluate(
                self,
                programs: list[Program],
                context: dict[str, np.ndarray],
            ) -> list[FitnessResult]:
                return [FitnessResult(objectives=(0.0,)) for _ in programs]

        with pytest.raises(ValueError, match="must all have the same length"):
            evolve(reg, config, StubEvaluator(), context)


# ===========================================================================
# 3. TestNaNHandling
# ===========================================================================


class TestNaNHandling:
    """NaN fitness and evaluation values should not crash the engine."""

    def test_nan_fitness_evaluator_does_not_crash(self) -> None:
        """An evaluator returning NaN fitness should not crash evolve()."""

        class NaNEvaluator:
            def evaluate(
                self,
                programs: list[Program],
                context: dict[str, np.ndarray],
            ) -> list[FitnessResult]:
                return [FitnessResult(objectives=(float("nan"),)) for _ in programs]

        reg = _minimal_registry()
        config = _minimal_config(generations=2)
        context = _make_context()
        # Should complete without raising
        result = evolve(reg, config, NaNEvaluator(), context)
        assert isinstance(result, EvolutionResult)
        assert result.best_program is not None

    def test_nan_evaluation_result(self) -> None:
        """A program that evaluates to all NaN should return a NaN array
        without crashing."""
        reg = _minimal_registry()
        # Register a function that produces NaN
        reg.register(
            "nan_fn",
            lambda a: np.full_like(a, float("nan")),
            category="numeric",
            input_types=(Series,),
            output_type=Series,
        )
        nan_info = reg.get("nan_fn")
        x_terminal = TerminalNode(name="x", output_type=Series)
        nan_node = FunctionNode(primitive=nan_info, children=(x_terminal,))

        context = _make_context()
        result = evaluate(nan_node, context)
        assert isinstance(result, np.ndarray)
        assert np.all(np.isnan(result))


# ===========================================================================
# 4. TestConstantOptErrorNonFatal
# ===========================================================================


class TestConstantOptErrorNonFatal:
    """optimize_constants returns the original program on failure."""

    def test_broken_evaluator_returns_original(self) -> None:
        """When the evaluator returns NaN, optimize_constants should return
        the original program (or a program with the same structure)."""
        reg = _minimal_registry()
        add_info = reg.get("add")
        c1 = ConstantNode(value=1.5)
        c2 = ConstantNode(value=2.5)
        program = FunctionNode(primitive=add_info, children=(c1, c2))

        class NaNEvaluator:
            def evaluate(
                self,
                programs: list[Program],
                context: dict[str, np.ndarray],
            ) -> list[FitnessResult]:
                return [FitnessResult(objectives=(float("nan"),)) for _ in programs]

        context = _make_context()
        config = _minimal_config(
            constant_opt_max_iter=2,
            constant_opt_max_time_seconds=0.5,
        )
        rng = np.random.default_rng(42)

        result = optimize_constants(program, NaNEvaluator(), context, config, rng)
        # The result should be a valid program (not crash)
        assert isinstance(result, FunctionNode)
        # Should have the same number of constants
        assert len(extract_constants(result)) == 2

    def test_exception_raising_evaluator_returns_original(self) -> None:
        """When the evaluator raises an exception, optimize_constants should
        catch it and return the original program."""
        reg = _minimal_registry()
        add_info = reg.get("add")
        c1 = ConstantNode(value=3.0)
        c2 = ConstantNode(value=4.0)
        program = FunctionNode(primitive=add_info, children=(c1, c2))

        class CrashingEvaluator:
            def evaluate(
                self,
                programs: list[Program],
                context: dict[str, np.ndarray],
            ) -> list[FitnessResult]:
                raise RuntimeError("Evaluator exploded")

        context = _make_context()
        config = _minimal_config(
            constant_opt_max_iter=2,
            constant_opt_max_time_seconds=0.5,
        )
        rng = np.random.default_rng(42)

        result = optimize_constants(
            program,
            CrashingEvaluator(),
            context,
            config,
            rng,
        )
        # Should return original program unchanged
        assert result is program

    def test_program_without_constants_returned_unchanged(self) -> None:
        """optimize_constants on a program with no constants is a no-op."""
        x_terminal = TerminalNode(name="x", output_type=Series)

        context = _make_context()
        config = _minimal_config()
        rng = np.random.default_rng(42)

        result = optimize_constants(
            x_terminal,
            SimpleFitnessEvaluator(),
            context,
            config,
            rng,
        )
        assert result is x_terminal

    def test_extract_inject_roundtrip(self) -> None:
        """extract_constants + inject_constants is a value-preserving roundtrip."""
        reg = _minimal_registry()
        add_info = reg.get("add")
        c1 = ConstantNode(value=1.0)
        c2 = ConstantNode(value=2.0)
        program = FunctionNode(primitive=add_info, children=(c1, c2))

        constants = extract_constants(program)
        assert constants == [1.0, 2.0]

        restored = inject_constants(program, constants)
        assert extract_constants(restored) == constants

    def test_inject_constants_mismatched_count_raises(self) -> None:
        """inject_constants raises ValueError when count does not match."""
        reg = _minimal_registry()
        add_info = reg.get("add")
        c1 = ConstantNode(value=1.0)
        c2 = ConstantNode(value=2.0)
        program = FunctionNode(primitive=add_info, children=(c1, c2))

        with pytest.raises(ValueError, match="mismatch"):
            inject_constants(program, [1.0])  # only 1 value for 2 constants


# ===========================================================================
# 5. TestFitnessEvaluatorExceptions
# ===========================================================================


class TestFitnessEvaluatorExceptions:
    """Exceptions from the fitness evaluator should propagate through evolve()."""

    def test_evaluator_exception_propagates(self) -> None:
        """If the fitness evaluator raises, evolve() should not swallow it."""

        class ExplodingEvaluator:
            def evaluate(
                self,
                programs: list[Program],
                context: dict[str, np.ndarray],
            ) -> list[FitnessResult]:
                raise ValueError("Evaluator intentionally failed")

        reg = _minimal_registry()
        config = _minimal_config(generations=2)
        context = _make_context()

        with pytest.raises(ValueError, match="Evaluator intentionally failed"):
            evolve(reg, config, ExplodingEvaluator(), context)

    def test_evaluator_runtime_error_propagates(self) -> None:
        """RuntimeError from the evaluator should propagate."""

        class RuntimeErrorEvaluator:
            def evaluate(
                self,
                programs: list[Program],
                context: dict[str, np.ndarray],
            ) -> list[FitnessResult]:
                raise RuntimeError("Runtime failure in evaluator")

        reg = _minimal_registry()
        config = _minimal_config(generations=1)
        context = _make_context()

        with pytest.raises(RuntimeError, match="Runtime failure"):
            evolve(reg, config, RuntimeErrorEvaluator(), context)


# ===========================================================================
# 6. TestMinimalViableEvolution
# ===========================================================================


class TestMinimalViableEvolution:
    """Smallest possible valid config should produce a valid result."""

    def test_minimal_config_evolves(self) -> None:
        """population_size=10, max_depth=2, generations=1 should succeed."""
        reg = _minimal_registry()
        config = GPConfig(
            population_size=10,
            max_depth=2,
            generations=1,
            seed=42,
            tournament_size=2,
            elitism_count=1,
            constant_opt_enabled=False,
            simplification_enabled=False,
        )
        context = _make_context()
        evaluator = SimpleFitnessEvaluator()
        result = evolve(reg, config, evaluator, context)

        assert isinstance(result, EvolutionResult)
        assert result.best_program is not None
        assert len(result.fitness_history) == 1
        assert result.config is config

    def test_minimal_config_best_program_evaluates(self) -> None:
        """The best program from minimal evolution should evaluate."""
        reg = _minimal_registry()
        config = GPConfig(
            population_size=10,
            max_depth=2,
            generations=1,
            seed=42,
            tournament_size=2,
            elitism_count=1,
            constant_opt_enabled=False,
            simplification_enabled=False,
        )
        context = _make_context()
        evaluator = SimpleFitnessEvaluator()
        result = evolve(reg, config, evaluator, context)

        output = evaluate(result.best_program, context)
        assert isinstance(output, np.ndarray)
        assert len(output) == len(context["x"])


# ===========================================================================
# 7. TestSerializationErrors
# ===========================================================================


class TestSerializationErrors:
    """Deserialization rejects invalid or incomplete data."""

    def test_missing_primitive_raises_serialization_error(self) -> None:
        """Deserializing a program that references a primitive not in the
        registry should raise SerializationError."""
        reg = PrimitiveRegistry()
        reg.register("x", lambda: None, input_types=(), output_type=Series)
        # No "add" registered -- deserialization should fail
        data = {
            "schema_version": "1.0.0",
            "program": {
                "type": "function",
                "primitive": "add",
                "children": [
                    {"type": "terminal", "name": "x", "output_type": "Series"},
                    {"type": "terminal", "name": "x", "output_type": "Series"},
                ],
            },
        }
        with pytest.raises(SerializationError, match="add"):
            deserialize(data, reg)

    def test_invalid_node_type_raises_serialization_error(self) -> None:
        """Deserializing data with an unknown node type should raise
        SerializationError."""
        reg = PrimitiveRegistry()
        data = {
            "schema_version": "1.0.0",
            "program": {
                "type": "alien_node",
                "name": "x",
                "output_type": "Series",
            },
        }
        with pytest.raises(SerializationError, match="alien_node"):
            deserialize(data, reg)

    def test_missing_schema_version_raises(self) -> None:
        """Deserializing data without schema_version should raise
        SerializationError."""
        reg = PrimitiveRegistry()
        data = {
            "program": {
                "type": "terminal",
                "name": "x",
                "output_type": "Series",
            },
        }
        with pytest.raises(SerializationError, match="schema_version"):
            deserialize(data, reg)

    def test_wrong_schema_version_raises(self) -> None:
        """Deserializing data with an unsupported schema_version should raise
        SerializationError."""
        reg = PrimitiveRegistry()
        data = {
            "schema_version": "99.0.0",
            "program": {
                "type": "terminal",
                "name": "x",
                "output_type": "Series",
            },
        }
        with pytest.raises(SerializationError, match="Unsupported"):
            deserialize(data, reg)

    def test_unknown_gp_type_raises(self) -> None:
        """Deserializing a terminal with an unknown GPType should raise
        SerializationError."""
        reg = PrimitiveRegistry()
        data = {
            "schema_version": "1.0.0",
            "program": {
                "type": "terminal",
                "name": "x",
                "output_type": "NoSuchType",
            },
        }
        with pytest.raises(SerializationError, match="NoSuchType"):
            deserialize(data, reg)
