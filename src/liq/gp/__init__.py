"""
liq-gp: General-purpose genetic programming engine for the LIQ Stack.

This package provides typed program evolution with modern SOTA techniques:
joint structure and constant optimization, multi-objective Pareto selection,
semantic diversity management, and algebraic simplification.
"""

from liq.gp.config import FitnessConfig, GPConfig, SeedInjectionConfig
from liq.gp.errors import (
    ConfigurationError,
    EvaluationError,
    EvolutionError,
    GPError,
    PrimitiveError,
    SerializationError,
    TypeCheckError,
)
from liq.gp.evolution.engine import evolve
from liq.gp.evolution.init import (
    initialize_seeded_population,
    validate_seed_programs,
)
from liq.gp.evolution.injection import inject_seeds
from liq.gp.primitives.registry import PrimitiveInfo, PrimitiveRegistry
from liq.gp.program.ast import (
    ConstantNode,
    FunctionNode,
    ParameterizedNode,
    Program,
    TerminalNode,
)
from liq.gp.program.constants import optimize_constants
from liq.gp.program.eval import evaluate
from liq.gp.program.serialize import (
    deserialize,
    deserialize_result,
    serialize,
    serialize_result,
)
from liq.gp.program.simplify import simplify
from liq.gp.protocols import FitnessEvaluator, GenerationCallback
from liq.gp.types import (
    BoolSeries,
    EvolutionResult,
    FitnessResult,
    GenerationStats,
    Int,
    ParamSpec,
    Scalar,
    Series,
)

__all__ = [
    # Configuration
    "GPConfig",
    "FitnessConfig",
    "SeedInjectionConfig",
    # Types
    "Series",
    "BoolSeries",
    "Scalar",
    "Int",
    "ParamSpec",
    "FitnessResult",
    "GenerationStats",
    "EvolutionResult",
    # Primitives
    "PrimitiveRegistry",
    "PrimitiveInfo",
    # Program / AST
    "Program",
    "TerminalNode",
    "ConstantNode",
    "FunctionNode",
    "ParameterizedNode",
    # Program functions
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
    # Protocols
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
]
