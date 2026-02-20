"""Configuration models for liq-gp (Pydantic v2).

Provides :class:`FitnessConfig` and :class:`GPConfig` -- frozen Pydantic
models that validate all parameters at construction time (fail-fast).
Invalid values raise :class:`~liq.gp.errors.ConfigurationError`.
"""

from __future__ import annotations

import math
from typing import Literal, Self

from pydantic import BaseModel, model_validator

from liq.gp.errors import ConfigurationError


class FitnessConfig(BaseModel, frozen=True):
    """Configuration for fitness evaluation.

    Attributes:
        objectives: Names for each objective (e.g. ``["profit", "simplicity"]``).
        objective_directions: ``"maximize"`` or ``"minimize"`` per objective.
            Must have the same length as *objectives*.
        batch_size: Optional mini-batch size for evaluation.  When set, only
            a random subset of the data is used per generation, with a full
            evaluation every *full_eval_interval* generations.
        full_eval_interval: How often (in generations) to evaluate on the full
            dataset when *batch_size* is set.
    """

    objectives: list[str] = ["fitness"]
    objective_directions: list[Literal["maximize", "minimize"]] = ["maximize"]
    batch_size: int | None = None
    full_eval_interval: int = 10

    @model_validator(mode="after")
    def _validate_objectives(self) -> Self:
        if len(self.objectives) != len(self.objective_directions):
            raise ConfigurationError(
                f"len(objectives)={len(self.objectives)} must equal "
                f"len(objective_directions)={len(self.objective_directions)}"
            )
        if self.batch_size is not None and self.batch_size < 1:
            raise ConfigurationError("batch_size must be >= 1 when set")
        if self.full_eval_interval < 1:
            raise ConfigurationError("full_eval_interval must be >= 1")
        return self


class SeedInjectionConfig(BaseModel, frozen=True):
    """Configuration for periodic seed injection during evolution.

    Controls when and how seed programs are re-injected into the population
    to maintain diversity or guide search toward known-good structures.

    Attributes:
        interval: Inject every *interval* generations (must be >= 1).
        count: Number of programs to inject per cycle (>= 1).
        method: Injection strategy:

            - ``"direct"`` — inject seed programs as-is (cycles through them).
            - ``"variation"`` — apply variation operators to seeds to produce
              offspring (like seeded initialization).
            - ``"ramped"`` — generate fresh random programs via ramped
              half-and-half (no seed programs required).
    Replacement always targets the worst-fitness individuals in the population.
    In NSGA-II mode, "worst" is determined by Pareto rank and crowding distance.
    """

    interval: int
    count: int = 1
    method: Literal["direct", "variation", "ramped"] = "variation"

    @model_validator(mode="after")
    def _validate(self) -> Self:
        if self.interval < 1:
            raise ConfigurationError("seed_injection.interval must be >= 1")
        if self.count < 1:
            raise ConfigurationError("seed_injection.count must be >= 1")
        return self


class GPConfig(BaseModel, frozen=True):
    """Main configuration for the GP engine.

    All parameters have sensible defaults.  Operator rates
    (``crossover_rate`` through ``hoist_mutation_rate``) must sum to 1.0.
    Invalid values raise :class:`~liq.gp.errors.ConfigurationError` at
    construction time.

    See the README configuration reference for a full description of
    each parameter.
    """

    population_size: int = 300
    max_depth: int = 8
    max_size: int | None = None
    generations: int = 50
    seed: int = 42

    # Operator rates (must sum to 1.0)
    crossover_rate: float = 0.7
    subtree_mutation_rate: float = 0.1
    point_mutation_rate: float = 0.1
    parameter_mutation_rate: float = 0.05
    hoist_mutation_rate: float = 0.05
    max_crossover_attempts: int = 20

    # Selection
    tournament_size: int = 5
    elitism_count: int = 5
    selection_mode: Literal["tournament", "nsga2"] = "tournament"

    # Parsimony
    parsimony_mode: Literal["lexicographic", "pareto", "linear", "disabled"] = (
        "lexicographic"
    )
    parsimony_coefficient: float = 0.001

    # Constant optimization
    constant_opt_enabled: bool = True
    constant_opt_top_k: float = 0.1
    constant_opt_max_iter: int = 50
    constant_opt_max_time_seconds: float = 1.0

    # Simplification
    simplification_enabled: bool = True

    # Semantic dedup
    semantic_dedup_enabled: bool = True
    semantic_ref_size: int = 50
    semantic_precision: int = 6

    # Early stopping
    early_stop_patience: int | None = None
    early_stop_threshold: float = 1e-6

    # Seed injection
    seed_injection: SeedInjectionConfig | None = None

    # Fitness config
    fitness: FitnessConfig = FitnessConfig()

    @model_validator(mode="after")
    def _validate_all(self) -> Self:
        # Population
        if self.population_size < 10:
            raise ConfigurationError("population_size must be >= 10")

        # Depth
        if self.max_depth < 2:
            raise ConfigurationError("max_depth must be >= 2")

        # Max size
        if self.max_size is not None and self.max_size < 1:
            raise ConfigurationError("max_size must be >= 1 when set")

        # Operator rates sum to 1.0
        rate_sum = (
            self.crossover_rate
            + self.subtree_mutation_rate
            + self.point_mutation_rate
            + self.parameter_mutation_rate
            + self.hoist_mutation_rate
        )
        if not math.isclose(rate_sum, 1.0, abs_tol=1e-9):
            raise ConfigurationError(f"Operator rates must sum to 1.0, got {rate_sum}")

        # Non-negative rates
        for name in (
            "crossover_rate",
            "subtree_mutation_rate",
            "point_mutation_rate",
            "parameter_mutation_rate",
            "hoist_mutation_rate",
            "parsimony_coefficient",
            "early_stop_threshold",
        ):
            if getattr(self, name) < 0:
                raise ConfigurationError(f"{name} must be non-negative")

        # Tournament size
        if self.tournament_size < 2:
            raise ConfigurationError("tournament_size must be >= 2")
        if self.tournament_size > self.population_size:
            raise ConfigurationError("tournament_size must be <= population_size")

        if self.max_crossover_attempts < 1:
            raise ConfigurationError("max_crossover_attempts must be >= 1")

        # Elitism
        if self.elitism_count < 0:
            raise ConfigurationError("elitism_count must be >= 0")
        if self.elitism_count >= self.population_size:
            raise ConfigurationError("elitism_count must be < population_size")

        # Constant opt
        if self.constant_opt_top_k <= 0.0 or self.constant_opt_top_k > 1.0:
            raise ConfigurationError("constant_opt_top_k must be in (0.0, 1.0]")
        if self.constant_opt_max_time_seconds <= 0:
            raise ConfigurationError("constant_opt_max_time_seconds must be > 0")

        # NSGA-II requires >= 2 objectives
        if self.selection_mode == "nsga2" and len(self.fitness.objectives) < 2:
            raise ConfigurationError("NSGA-II selection requires >= 2 objectives")

        # Pareto parsimony requires NSGA-II
        if self.parsimony_mode == "pareto" and self.selection_mode != "nsga2":
            raise ConfigurationError("Pareto parsimony requires selection_mode='nsga2'")

        # Seed injection
        if self.seed_injection is not None:
            max_replaceable = self.population_size - self.elitism_count
            if self.seed_injection.count > max_replaceable:
                raise ConfigurationError(
                    f"seed_injection.count ({self.seed_injection.count}) must be "
                    f"<= population_size - elitism_count ({max_replaceable})"
                )

        return self
