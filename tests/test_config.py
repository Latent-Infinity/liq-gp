"""Tests for liq.gp.config module."""

from __future__ import annotations

import pytest

from liq.gp.config import FitnessConfig, GPConfig
from liq.gp.errors import ConfigurationError


class TestGPConfigDefaults:
    """GPConfig default values match requirements section 6."""

    def test_defaults(self) -> None:
        cfg = GPConfig()
        assert cfg.population_size == 300
        assert cfg.max_depth == 8
        assert cfg.max_size is None
        assert cfg.generations == 50
        assert cfg.seed == 42
        assert cfg.crossover_rate == 0.7
        assert cfg.subtree_mutation_rate == 0.1
        assert cfg.point_mutation_rate == 0.1
        assert cfg.parameter_mutation_rate == 0.05
        assert cfg.hoist_mutation_rate == 0.05
        assert cfg.max_crossover_attempts == 20
        assert cfg.tournament_size == 5
        assert cfg.elitism_count == 5
        assert cfg.selection_mode == "tournament"
        assert cfg.parsimony_mode == "lexicographic"
        assert cfg.parsimony_coefficient == 0.001
        assert cfg.constant_opt_enabled is True
        assert cfg.constant_opt_top_k == 0.1
        assert cfg.constant_opt_max_iter == 50
        assert cfg.constant_opt_max_time_seconds == 1.0
        assert cfg.simplification_enabled is True
        assert cfg.semantic_dedup_enabled is True
        assert cfg.semantic_ref_size == 50
        assert cfg.semantic_precision == 6
        assert cfg.early_stop_patience is None
        assert cfg.early_stop_threshold == 1e-6

    def test_frozen(self) -> None:
        cfg = GPConfig()
        with pytest.raises(Exception):  # noqa: B017, PT011
            cfg.population_size = 100  # type: ignore[misc]


class TestFitnessConfigDefaults:
    """FitnessConfig default values match requirements section 6."""

    def test_defaults(self) -> None:
        cfg = FitnessConfig()
        assert cfg.objectives == ["fitness"]
        assert cfg.objective_directions == ["maximize"]
        assert cfg.batch_size is None
        assert cfg.full_eval_interval == 10

    def test_frozen(self) -> None:
        cfg = FitnessConfig()
        with pytest.raises(Exception):  # noqa: B017, PT011
            cfg.objectives = ["x"]  # type: ignore[misc]


class TestGPConfigValidation:
    """GPConfig validation rules from requirements section 6."""

    def test_population_size_too_small(self) -> None:
        with pytest.raises(ConfigurationError, match="population_size"):
            GPConfig(population_size=5)

    def test_max_depth_too_small(self) -> None:
        with pytest.raises(ConfigurationError, match="max_depth"):
            GPConfig(max_depth=1)

    def test_operator_rates_must_sum_to_one(self) -> None:
        with pytest.raises(ConfigurationError, match="rates must sum"):
            GPConfig(crossover_rate=0.5)  # sum != 1.0

    def test_operator_rates_exact_sum(self) -> None:
        # Should not raise when sum is exactly 1.0
        cfg = GPConfig(
            crossover_rate=0.5,
            subtree_mutation_rate=0.2,
            point_mutation_rate=0.15,
            parameter_mutation_rate=0.1,
            hoist_mutation_rate=0.05,
        )
        assert cfg.crossover_rate == 0.5

    def test_negative_rate_rejected(self) -> None:
        with pytest.raises(ConfigurationError, match="non-negative"):
            GPConfig(
                crossover_rate=-0.1,
                subtree_mutation_rate=0.5,
                point_mutation_rate=0.3,
                parameter_mutation_rate=0.2,
                hoist_mutation_rate=0.1,
            )

    def test_tournament_size_too_small(self) -> None:
        with pytest.raises(ConfigurationError, match="tournament_size"):
            GPConfig(tournament_size=1)

    def test_tournament_size_exceeds_population(self) -> None:
        with pytest.raises(ConfigurationError, match="tournament_size"):
            GPConfig(population_size=10, tournament_size=20)

    def test_max_crossover_attempts_must_be_positive(self) -> None:
        with pytest.raises(ConfigurationError, match="max_crossover_attempts"):
            GPConfig(max_crossover_attempts=0)

    def test_elitism_negative(self) -> None:
        with pytest.raises(ConfigurationError, match="elitism_count"):
            GPConfig(elitism_count=-1)

    def test_elitism_equals_population(self) -> None:
        with pytest.raises(ConfigurationError, match="elitism_count"):
            GPConfig(population_size=10, elitism_count=10)

    def test_constant_opt_top_k_zero(self) -> None:
        with pytest.raises(ConfigurationError, match="constant_opt_top_k"):
            GPConfig(constant_opt_top_k=0.0)

    def test_constant_opt_top_k_above_one(self) -> None:
        with pytest.raises(ConfigurationError, match="constant_opt_top_k"):
            GPConfig(constant_opt_top_k=1.1)

    def test_constant_opt_top_k_one_valid(self) -> None:
        cfg = GPConfig(constant_opt_top_k=1.0)
        assert cfg.constant_opt_top_k == 1.0

    def test_constant_opt_max_time_zero(self) -> None:
        with pytest.raises(ConfigurationError, match="constant_opt_max_time"):
            GPConfig(constant_opt_max_time_seconds=0.0)

    def test_nsga2_requires_two_objectives(self) -> None:
        with pytest.raises(ConfigurationError, match="NSGA-II"):
            GPConfig(
                selection_mode="nsga2",
                fitness=FitnessConfig(
                    objectives=["fitness"],
                    objective_directions=["maximize"],
                ),
            )

    def test_nsga2_with_two_objectives_valid(self) -> None:
        cfg = GPConfig(
            selection_mode="nsga2",
            fitness=FitnessConfig(
                objectives=["fitness", "complexity"],
                objective_directions=["maximize", "minimize"],
            ),
        )
        assert cfg.selection_mode == "nsga2"

    def test_pareto_parsimony_requires_nsga2(self) -> None:
        with pytest.raises(ConfigurationError, match="Pareto parsimony"):
            GPConfig(
                parsimony_mode="pareto",
                selection_mode="tournament",
            )

    def test_max_size_zero_rejected(self) -> None:
        with pytest.raises(ConfigurationError, match="max_size"):
            GPConfig(max_size=0)


class TestFitnessConfigValidation:
    """FitnessConfig validation rules."""

    def test_mismatched_objective_lengths(self) -> None:
        with pytest.raises(ConfigurationError, match="len"):
            FitnessConfig(
                objectives=["a", "b"],
                objective_directions=["maximize"],
            )

    def test_batch_size_zero(self) -> None:
        with pytest.raises(ConfigurationError, match="batch_size"):
            FitnessConfig(batch_size=0)

    def test_full_eval_interval_zero(self) -> None:
        with pytest.raises(ConfigurationError, match="full_eval_interval"):
            FitnessConfig(full_eval_interval=0)

    def test_valid_multi_objective(self) -> None:
        cfg = FitnessConfig(
            objectives=["sharpe", "complexity"],
            objective_directions=["maximize", "minimize"],
        )
        assert len(cfg.objectives) == 2
