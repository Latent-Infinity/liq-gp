"""Type system and core data types for liq-gp.

Defines the GP type system (:class:`GPType` with built-in types
:data:`Series`, :data:`BoolSeries`, :data:`Scalar`, :data:`Int`) and
the core data types used throughout the library: :class:`ParamSpec`,
:class:`FitnessResult`, :class:`GenerationStats`, :class:`EvolutionResult`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
import math
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from numpy.random import Generator
    from liq.gp.config import GPConfig
    from liq.gp.program.ast import Program


class GPType:
    """Base class for GP type objects.

    The four built-in types (Series, BoolSeries, Scalar, Int) are pre-registered
    instances. Consumers can register additional types via ``GPType.register_type()``.
    """

    _registry: dict[str, GPType] = {}

    def __init__(self, name: str, description: str) -> None:
        self._name = name
        self._description = description

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    def __repr__(self) -> str:
        return f"GPType({self._name!r})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, GPType):
            return NotImplemented
        return self._name == other._name

    def __hash__(self) -> int:
        return hash(self._name)

    @classmethod
    def register_type(cls, name: str, description: str = "") -> GPType:
        """Register a new GP type.

        Args:
            name: Unique type name.
            description: Human-readable description.

        Returns:
            A new GPType instance usable in primitive signatures.

        Raises:
            ValueError: If a type with that name is already registered.
        """
        if name in cls._registry:
            msg = f"GP type {name!r} is already registered"
            raise ValueError(msg)
        new_type = cls(name, description)
        cls._registry[name] = new_type
        return new_type

    @classmethod
    def get(cls, name: str) -> GPType:
        """Look up a registered type by name.

        Raises:
            KeyError: If no type with that name exists.
        """
        return cls._registry[name]

    @classmethod
    def all_types(cls) -> dict[str, GPType]:
        """Return a copy of all registered types."""
        return dict(cls._registry)

    @classmethod
    def _reset_registry(cls) -> None:
        """Reset to built-in types only. For testing."""
        cls._registry = {
            k: v
            for k, v in cls._registry.items()
            if k in ("Series", "BoolSeries", "Scalar", "Int")
        }


# Built-in types (FR-1.1)
Series = GPType("Series", "Array of float values (one per observation)")
BoolSeries = GPType(
    "BoolSeries", "Array of boolean values encoded as float64 (1.0/0.0)"
)
Scalar = GPType("Scalar", "Single float value")
Int = GPType("Int", "Integer value (for periods, shifts)")

# Register built-ins
GPType._registry["Series"] = Series
GPType._registry["BoolSeries"] = BoolSeries
GPType._registry["Scalar"] = Scalar
GPType._registry["Int"] = Int


@dataclass(frozen=True)
class ParamSpec:
    """Specification for an evolvable parameter on a ParameterizedNode (FR-3.3).

    Attributes:
        name: Parameter name (e.g. ``"period"``).
        dtype: ``int`` or ``float``.
        default: Default value used during initialisation.
        min_value: Lower bound (inclusive) for sampling and clamping.
        max_value: Upper bound (inclusive) for sampling and clamping.
    """

    name: str
    dtype: type  # int or float
    default: int | float
    min_value: int | float | None = None
    max_value: int | float | None = None
    allowed_values: list[int | float] | None = None

    def __post_init__(self) -> None:
        if self.dtype not in (int, float):
            msg = f"dtype must be int or float, got {self.dtype}"
            raise TypeError(msg)
        if self.allowed_values is not None:
            cleaned = self._normalize_allowed_values(self.allowed_values)
            object.__setattr__(self, "allowed_values", cleaned)
            if self.dtype is int and any(type(value) is not int for value in cleaned):
                raise TypeError("all allowed_values must be int when dtype is int")
            if self.dtype is float and any(
                not isinstance(value, (int, float)) or isinstance(value, bool)
                for value in cleaned
            ):
                raise TypeError(
                    "allowed_values must be int or float values when dtype is float"
                )
            if self.dtype is float and not all(isinstance(v, (int, float)) for v in cleaned):
                raise TypeError("allowed_values must be numeric when dtype is float")
            if self.default not in cleaned:
                msg = f"default ({self.default}) must be in allowed_values"
                raise ValueError(msg)
            return

        if self.min_value is None or self.max_value is None:
            raise ValueError("min_value and max_value are required when allowed_values is None")
        if self.dtype is int and not isinstance(self.min_value, int):
            raise TypeError("min_value must be int when dtype is int")
        if self.dtype is int and isinstance(self.min_value, bool):
            raise TypeError("min_value must be int, not bool")
        if self.dtype is float and isinstance(self.min_value, bool):
            raise TypeError("min_value must be int or float, not bool")
        if self.dtype is int and not isinstance(self.max_value, int):
            raise TypeError("max_value must be int when dtype is int")
        if self.dtype is int and isinstance(self.max_value, bool):
            raise TypeError("max_value must be int, not bool")
        if self.dtype is float and isinstance(self.max_value, bool):
            raise TypeError("max_value must be int or float, not bool")
        if not isinstance(self.min_value, (int, float)):
            raise TypeError("min_value must be int or float")
        if not isinstance(self.max_value, (int, float)):
            raise TypeError("max_value must be int or float")
        if self.min_value > self.max_value:
            msg = (
                f"min_value ({self.min_value}) must be <= max_value ({self.max_value})"
            )
            raise ValueError(msg)
        if not math.isfinite(self.min_value) or not math.isfinite(self.max_value):
            raise ValueError("min_value and max_value must be finite")
        if self.dtype is int and not (isinstance(self.default, int) and not isinstance(self.default, bool)):
            raise TypeError("default must be int when dtype is int")
        if self.dtype is float and (not isinstance(self.default, (int, float)) or isinstance(self.default, bool)):
            raise TypeError("default must be int or float when dtype is float")
        if not self.min_value <= self.default <= self.max_value:
            msg = (
                f"default ({self.default}) must be within [{self.min_value}, {self.max_value}]"
            )
            raise ValueError(msg)

    @staticmethod
    def _normalize_allowed_values(values: list[int | float]) -> list[int | float]:
        """Normalize allowed values:

        - remove duplicates
        - sort ascending
        - enforce finiteness
        """
        if not values:
            raise ValueError("allowed_values must be a non-empty sequence")
        cleaned = sorted(set(values))
        if any(not isinstance(value, (int, float)) for value in cleaned):
            raise TypeError("allowed_values must contain only int/float values")
        if any(isinstance(value, bool) for value in cleaned):
            raise TypeError("allowed_values must not contain bool values")
        if any(not math.isfinite(float(value)) for value in cleaned):
            raise ValueError("allowed_values must be finite")
        return cleaned

    def value_is_discrete(self) -> bool:
        return self.allowed_values is not None


@dataclass(frozen=True)
class FitnessResult:
    """Result of fitness evaluation for a single program (FR-5.4.2).

    Attributes:
        objectives: Tuple of objective values.  For single-objective problems
            this is a one-element tuple, e.g. ``(0.95,)``.
        metadata: Optional dict for consumer-specific data (not used by the
            engine itself).
    """

    objectives: tuple[float, ...]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SelectionContext:
    """Precomputed selection metadata for extension hooks.

    Contains enough NSGA-II state for hooks that want to avoid recomputing Pareto
    rankings inside the parent-selection stage.
    """

    fronts: list[list[int]] | None = None
    ranks: list[int] | None = None
    crowding: list[float] | None = None


ParentSourceFn = Callable[
    ["list[Program]", "list[FitnessResult]", "GPConfig", "Generator", int, "SelectionContext"],
    "list[Program]",
]


@dataclass(frozen=True)
class GenerationStats:
    """Per-generation statistics reported to callbacks (FR-5.5.3).

    Fitness and size metrics describe the **evaluated** population at the
    start of the generation, before seed injection and semantic deduplication.
    These post-evaluation steps modify the population for the *next*
    generation but do not retroactively change the statistics.

    The ``injected_count`` field records how many programs were injected
    *after* statistics were computed — it is additive metadata, not
    reflected in the fitness or size fields.

    Attributes:
        generation: Zero-based generation index.
        best_fitness: Objective values of the best individual.
        mean_fitness: Mean objective values across the population.
        best_program_size: Node count of the best individual.
        mean_program_size: Mean node count across the population.
        unique_semantics_ratio: Fraction of semantically unique individuals
            (0.0-1.0); computed via semantic fingerprinting.
        pareto_front_size: Number of individuals on the first Pareto front
            (meaningful in multi-objective mode).
        injected_count: Number of programs injected this generation via
            periodic seed injection (0 when injection is disabled or not
            triggered).
        scheduler_metrics: Optional bounded-scheduler metrics for this
            generation's primary evaluation pass.
    """

    generation: int
    best_fitness: tuple[float, ...]
    mean_fitness: tuple[float, ...]
    best_program_size: int
    mean_program_size: float
    unique_semantics_ratio: float
    pareto_front_size: int
    injected_count: int = 0
    scheduler_metrics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvolutionResult:
    """Final result of an evolution run (FR-5.5.2).

    Attributes:
        best_program: The highest-fitness program found (by first objective).
        pareto_front: All non-dominated programs from the final generation.
            In single-objective mode this contains only ``best_program``.
        fitness_history: List of :class:`GenerationStats`, one per generation.
        config: The :class:`~liq.gp.GPConfig` used for this run (including
            any ``fitness_config`` override).
    """

    best_program: Any  # Program (defined in ast module)
    pareto_front: list[Any]  # list[Program]
    fitness_history: list[GenerationStats]
    config: Any  # GPConfig (defined in config module)
