"""Tests for liq.gp.config module."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from liq.gp.config import FitnessConfig, GPConfig, SchedulerConfig, SeedInjectionConfig
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
        assert cfg.constant_opt_mode == "top_k"
        assert cfg.constant_opt_max_evals is None
        assert cfg.constant_opt_max_iter == 50
        assert cfg.constant_opt_max_time_seconds == 1.0
        assert cfg.constant_opt_role_schedule.gate_eval_interval == 1
        assert cfg.constant_opt_role_schedule.expert_eval_interval == 1
        assert cfg.constant_opt_role_schedule.risk_eval_interval == 1
        assert cfg.constant_opt_role_schedule.other_eval_interval == 1
        assert cfg.constant_opt_role_bounds.gate_threshold == (0.0, 1.0)
        assert cfg.constant_opt_role_bounds.gate_slope == (0.1, 20.0)
        assert cfg.constant_opt_role_bounds.expert_weight == (0.0, 10.0)
        assert cfg.constant_opt_role_bounds.risk_scale == (0.0, 3.0)
        assert cfg.constant_opt_role_bounds.enable_unbounded_for_unknown is True
        assert cfg.simplification_enabled is True
        assert cfg.semantic_ref_size == 50
        assert cfg.semantic_precision == 6
        assert cfg.early_stop_patience is None
        assert cfg.early_stop_threshold == 1e-6
        assert cfg.lexicase_downsample_policy == "none"
        assert cfg.lexicase_downsample_cases is None
        assert cfg.lexicase_downsample_min_cases == 1
        assert cfg.lexicase_epsilon_strategy == "mad"
        assert cfg.lexicase_epsilon_percentile == 50.0
        assert cfg.lexicase_nan_penalty == 1e6
        assert cfg.scheduler.enabled is False
        assert cfg.scheduler.max_in_flight == 4
        assert cfg.scheduler.queue_capacity == 16
        assert cfg.scheduler.eval_batch_size == 32
        assert cfg.scheduler.eval_timeout_seconds == 30.0
        assert cfg.scheduler.memory_budget_mb == 2048
        assert cfg.scheduler.max_cpu_workers == 1
        assert cfg.scheduler.safe_fallback_mode == "sequential"

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

    def test_invalid_selection_mode_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            GPConfig(selection_mode="not-a-mode")

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

    def test_constant_opt_mode_invalid(self) -> None:
        with pytest.raises(ValidationError, match="constant_opt_mode"):
            GPConfig(constant_opt_mode="invalid")

    def test_constant_opt_max_evals_must_be_positive(self) -> None:
        with pytest.raises(ConfigurationError, match="constant_opt_max_evals"):
            GPConfig(constant_opt_max_evals=0)

    def test_constant_opt_role_schedule_intervals_validated(self) -> None:
        with pytest.raises(ConfigurationError, match="constant_opt_gate_interval"):
            GPConfig(constant_opt_role_schedule={"gate_eval_interval": 0})
        with pytest.raises(ConfigurationError, match="constant_opt_expert_interval"):
            GPConfig(
                constant_opt_role_schedule={"gate_eval_interval": 1, "expert_eval_interval": 0}
            )
        with pytest.raises(ConfigurationError, match="constant_opt_risk_interval"):
            GPConfig(
                constant_opt_role_schedule={"gate_eval_interval": 1, "risk_eval_interval": 0}
            )
        with pytest.raises(ConfigurationError, match="constant_opt_other_interval"):
            GPConfig(
                constant_opt_role_schedule={"gate_eval_interval": 1, "other_eval_interval": 0}
            )

    def test_constant_opt_role_bounds_enable_unbounded_flag(self) -> None:
        cfg = GPConfig(
            constant_opt_role_bounds={
                "enable_unbounded_for_unknown": False,
                "gate_threshold": (0.0, 1.0),
                "gate_slope": (0.1, 3.0),
                "expert_weight": (0.0, 10.0),
                "risk_scale": (0.0, 3.0),
            },
        )
        assert cfg.constant_opt_role_bounds.enable_unbounded_for_unknown is False

    def test_constant_opt_role_bounds_order_validated(self) -> None:
        with pytest.raises(ConfigurationError, match="low < high"):
            GPConfig(
                constant_opt_role_bounds={"gate_threshold": (1.0, 0.0)},
            )

    def test_constant_opt_max_evals_valid(self) -> None:
        cfg = GPConfig(constant_opt_max_evals=5)
        assert cfg.constant_opt_max_evals == 5

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

    def test_size_diversity_parsimony_mode_accepted(self) -> None:
        cfg = GPConfig(parsimony_mode="size_diversity")
        assert cfg.parsimony_mode == "size_diversity"

    def test_selection_mode_lexicase(self) -> None:
        cfg = GPConfig(
            selection_mode="lexicase",
            lexicase_downsample_policy="none",
        )
        assert cfg.selection_mode == "lexicase"

    def test_selection_mode_lexicase_informed(self) -> None:
        cfg = GPConfig(
            selection_mode="lexicase",
            lexicase_downsample_policy="informed",
            lexicase_downsample_cases=2,
        )
        assert cfg.lexicase_downsample_policy == "informed"

    def test_selection_mode_lexicase_eps(self) -> None:
        cfg = GPConfig(
            selection_mode="lexicase_eps",
            lexicase_epsilon_strategy="percentile",
            lexicase_epsilon_percentile=25.0,
        )
        assert cfg.selection_mode == "lexicase_eps"

    def test_lexicase_invalid_downsample_cases(self) -> None:
        with pytest.raises(ConfigurationError, match="lexicase_downsample_cases"):
            GPConfig(
                lexicase_downsample_policy="random",
                lexicase_downsample_cases=0,
            )
        with pytest.raises(ConfigurationError, match="lexicase_downsample_cases"):
            GPConfig(
                lexicase_downsample_policy="informed",
                lexicase_downsample_cases=0,
            )

    def test_lexicase_invalid_epsilon_percentile(self) -> None:
        with pytest.raises(ConfigurationError, match="lexicase_epsilon_percentile"):
            GPConfig(lexicase_epsilon_percentile=-1.0)

    def test_lexicase_invalid_epsilon_percentile_upper_bound(self) -> None:
        with pytest.raises(ConfigurationError, match="lexicase_epsilon_percentile"):
            GPConfig(lexicase_epsilon_percentile=101.0)

    def test_lexicase_invalid_min_cases(self) -> None:
        with pytest.raises(ConfigurationError, match="lexicase_downsample_min_cases"):
            GPConfig(lexicase_downsample_min_cases=0)

    def test_max_size_zero_rejected(self) -> None:
        with pytest.raises(ConfigurationError, match="max_size"):
            GPConfig(max_size=0)

    def test_scheduler_validation(self) -> None:
        with pytest.raises(ConfigurationError, match="scheduler.max_in_flight"):
            SchedulerConfig(max_in_flight=0)
        with pytest.raises(ConfigurationError, match="scheduler.queue_capacity"):
            SchedulerConfig(max_in_flight=2, queue_capacity=1)
        with pytest.raises(ConfigurationError, match="scheduler.eval_batch_size"):
            SchedulerConfig(eval_batch_size=0)
        with pytest.raises(ConfigurationError, match="scheduler.eval_timeout_seconds"):
            SchedulerConfig(eval_timeout_seconds=0.0)
        with pytest.raises(ConfigurationError, match="scheduler.memory_budget_mb"):
            SchedulerConfig(memory_budget_mb=64)
        with pytest.raises(ConfigurationError, match="scheduler.max_cpu_workers"):
            SchedulerConfig(max_cpu_workers=0)


class TestSeedInjectionConfigValidation:
    """SeedInjectionConfig validation rules."""

    def test_valid_defaults(self) -> None:
        cfg = SeedInjectionConfig(interval=5)
        assert cfg.interval == 5
        assert cfg.count == 1
        assert cfg.method == "variation"
        assert cfg.method == "variation"  # default method confirmed above

    def test_frozen(self) -> None:
        cfg = SeedInjectionConfig(interval=5)
        with pytest.raises(Exception):  # noqa: B017, PT011
            cfg.interval = 10  # type: ignore[misc]

    def test_all_methods_valid(self) -> None:
        for method in ("direct", "variation", "ramped"):
            cfg = SeedInjectionConfig(interval=1, method=method)
            assert cfg.method == method

    def test_interval_zero_rejected(self) -> None:
        with pytest.raises(ConfigurationError, match="interval"):
            SeedInjectionConfig(interval=0)

    def test_interval_negative_rejected(self) -> None:
        with pytest.raises(ConfigurationError, match="interval"):
            SeedInjectionConfig(interval=-1)

    def test_count_zero_rejected(self) -> None:
        with pytest.raises(ConfigurationError, match="count"):
            SeedInjectionConfig(interval=5, count=0)

    def test_count_negative_rejected(self) -> None:
        with pytest.raises(ConfigurationError, match="count"):
            SeedInjectionConfig(interval=5, count=-1)

    def test_count_large_valid_standalone(self) -> None:
        # SeedInjectionConfig alone doesn't know population_size
        cfg = SeedInjectionConfig(interval=1, count=100)
        assert cfg.count == 100


class TestGPConfigSeedInjection:
    """GPConfig cross-field validation for seed_injection."""

    def test_default_is_none(self) -> None:
        cfg = GPConfig()
        assert cfg.seed_injection is None

    def test_valid_seed_injection(self) -> None:
        cfg = GPConfig(
            seed_injection=SeedInjectionConfig(interval=5, count=3),
        )
        assert cfg.seed_injection is not None
        assert cfg.seed_injection.interval == 5
        assert cfg.seed_injection.count == 3

    def test_count_exceeds_replaceable_slots(self) -> None:
        # population_size=10, elitism_count=5 → max_replaceable=5
        with pytest.raises(ConfigurationError, match="seed_injection.count"):
            GPConfig(
                population_size=10,
                elitism_count=5,
                seed_injection=SeedInjectionConfig(interval=1, count=6),
            )

    def test_count_equals_replaceable_slots(self) -> None:
        # Boundary: count == population_size - elitism_count should be valid
        cfg = GPConfig(
            population_size=10,
            elitism_count=5,
            seed_injection=SeedInjectionConfig(interval=1, count=5),
        )
        assert cfg.seed_injection is not None
        assert cfg.seed_injection.count == 5

    def test_seed_injection_with_zero_elitism(self) -> None:
        cfg = GPConfig(
            population_size=10,
            elitism_count=0,
            seed_injection=SeedInjectionConfig(interval=10, count=10),
        )
        assert cfg.seed_injection is not None
        assert cfg.seed_injection.count == 10

    def test_ramped_method_valid(self) -> None:
        cfg = GPConfig(
            seed_injection=SeedInjectionConfig(interval=5, method="ramped"),
        )
        assert cfg.seed_injection is not None
        assert cfg.seed_injection.method == "ramped"


class TestSeedInjectionConfigValidation:
    """SeedInjectionConfig validation rules."""

    def test_valid_defaults(self) -> None:
        cfg = SeedInjectionConfig(interval=5)
        assert cfg.interval == 5
        assert cfg.count == 1
        assert cfg.method == "variation"
        assert cfg.method == "variation"  # default method confirmed above

    def test_frozen(self) -> None:
        cfg = SeedInjectionConfig(interval=5)
        with pytest.raises(Exception):  # noqa: B017, PT011
            cfg.interval = 10  # type: ignore[misc]

    def test_all_methods_valid(self) -> None:
        for method in ("direct", "variation", "ramped"):
            cfg = SeedInjectionConfig(interval=1, method=method)
            assert cfg.method == method

    def test_interval_zero_rejected(self) -> None:
        with pytest.raises(ConfigurationError, match="interval"):
            SeedInjectionConfig(interval=0)

    def test_interval_negative_rejected(self) -> None:
        with pytest.raises(ConfigurationError, match="interval"):
            SeedInjectionConfig(interval=-1)

    def test_count_zero_rejected(self) -> None:
        with pytest.raises(ConfigurationError, match="count"):
            SeedInjectionConfig(interval=5, count=0)

    def test_count_negative_rejected(self) -> None:
        with pytest.raises(ConfigurationError, match="count"):
            SeedInjectionConfig(interval=5, count=-1)

    def test_count_large_valid_standalone(self) -> None:
        # SeedInjectionConfig alone doesn't know population_size
        cfg = SeedInjectionConfig(interval=1, count=100)
        assert cfg.count == 100


class TestGPConfigSeedInjection:
    """GPConfig cross-field validation for seed_injection."""

    def test_default_is_none(self) -> None:
        cfg = GPConfig()
        assert cfg.seed_injection is None

    def test_valid_seed_injection(self) -> None:
        cfg = GPConfig(
            seed_injection=SeedInjectionConfig(interval=5, count=3),
        )
        assert cfg.seed_injection is not None
        assert cfg.seed_injection.interval == 5
        assert cfg.seed_injection.count == 3

    def test_count_exceeds_replaceable_slots(self) -> None:
        # population_size=10, elitism_count=5 → max_replaceable=5
        with pytest.raises(ConfigurationError, match="seed_injection.count"):
            GPConfig(
                population_size=10,
                elitism_count=5,
                seed_injection=SeedInjectionConfig(interval=1, count=6),
            )

    def test_count_equals_replaceable_slots(self) -> None:
        # Boundary: count == population_size - elitism_count should be valid
        cfg = GPConfig(
            population_size=10,
            elitism_count=5,
            seed_injection=SeedInjectionConfig(interval=1, count=5),
        )
        assert cfg.seed_injection is not None
        assert cfg.seed_injection.count == 5

    def test_seed_injection_with_zero_elitism(self) -> None:
        cfg = GPConfig(
            population_size=10,
            elitism_count=0,
            seed_injection=SeedInjectionConfig(interval=10, count=10),
        )
        assert cfg.seed_injection is not None
        assert cfg.seed_injection.count == 10

    def test_ramped_method_valid(self) -> None:
        cfg = GPConfig(
            seed_injection=SeedInjectionConfig(interval=5, method="ramped"),
        )
        assert cfg.seed_injection is not None
        assert cfg.seed_injection.method == "ramped"


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
