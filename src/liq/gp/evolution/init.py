"""Population initialization for GP (FR-5.1).

Supports ramped half-and-half initialization: half "full" trees (all branches
at max depth), half "grow" trees (branches terminate at random depths).

Also supports seeded initialization (FR-5.1.4): the caller provides 1 to
``population_size`` known programs.  These are placed directly in the
population and remaining slots are filled by applying variation operators
to the seeds.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from liq.gp.program.ast import (
    ConstantNode,
    FunctionNode,
    ParameterizedNode,
    TerminalNode,
)

if TYPE_CHECKING:
    from liq.gp.config import GPConfig
    from liq.gp.primitives.registry import PrimitiveInfo, PrimitiveRegistry
    from liq.gp.program.ast import Program
    from liq.gp.types import GPType


def _sample_terminal(
    registry: PrimitiveRegistry,
    output_type: GPType,
    rng: np.random.Generator,
) -> TerminalNode | ConstantNode:
    """Pick a random terminal or constant node for the given output type."""
    terminals = registry.terminals(output_type=output_type)
    # Include an ERC (ephemeral random constant) option for Series type
    from liq.gp.types import Series

    use_constant = len(terminals) == 0 or (output_type is Series and rng.random() < 0.2)
    if use_constant and output_type is Series:
        value = float(rng.uniform(-1.0, 1.0))
        return ConstantNode(value=value)
    if len(terminals) == 0:
        # Fallback to constant for any numeric type
        value = float(rng.uniform(-1.0, 1.0))
        return ConstantNode(value=value, output_type=output_type)
    choice = terminals[int(rng.integers(len(terminals)))]
    return TerminalNode(name=choice.name, output_type=choice.output_type)


def _sample_function(
    registry: PrimitiveRegistry,
    output_type: GPType,
    rng: np.random.Generator,
) -> PrimitiveInfo:
    """Pick a random function primitive for the given output type."""
    functions = registry.functions(output_type=output_type)
    if len(functions) == 0:
        msg = f"No functions available with output_type={output_type}"
        from liq.gp.errors import PrimitiveError

        raise PrimitiveError(msg)
    return functions[int(rng.integers(len(functions)))]


def _sample_params(
    primitive: PrimitiveInfo,
    rng: np.random.Generator,
) -> dict[str, int | float]:
    """Sample parameter values uniformly from their ranges."""
    params: dict[str, int | float] = {}
    for ps in primitive.param_specs:
        if ps.dtype is int:
            val = int(rng.integers(int(ps.min_value), int(ps.max_value) + 1))
            params[ps.name] = val
        else:
            val = float(rng.uniform(ps.min_value, ps.max_value))
            params[ps.name] = val
    return params


def _make_function_node(
    primitive: PrimitiveInfo,
    children: tuple[Program, ...],
    rng: np.random.Generator,
) -> FunctionNode | ParameterizedNode:
    """Create a FunctionNode or ParameterizedNode depending on param_specs."""
    if primitive.param_specs:
        params = _sample_params(primitive, rng)
        return ParameterizedNode(
            primitive=primitive,
            children=children,
            params=params,
        )
    return FunctionNode(primitive=primitive, children=children)


def generate_full(
    registry: PrimitiveRegistry,
    max_depth: int,
    output_type: GPType,
    rng: np.random.Generator,
) -> Program:
    """Generate a 'full' tree where all branches reach max_depth (FR-5.1.1).

    At depth < max_depth, always pick a function node.
    At depth == max_depth, always pick a terminal.
    """
    if max_depth == 0:
        return _sample_terminal(registry, output_type, rng)

    prim = _sample_function(registry, output_type, rng)
    children = tuple(
        generate_full(registry, max_depth - 1, child_type, rng)
        for child_type in prim.input_types
    )
    return _make_function_node(prim, children, rng)


def generate_grow(
    registry: PrimitiveRegistry,
    max_depth: int,
    output_type: GPType,
    rng: np.random.Generator,
) -> Program:
    """Generate a 'grow' tree where branches terminate at random depths (FR-5.1.1).

    At any depth < max_depth, randomly choose between terminal and function.
    At depth == max_depth, always pick a terminal.
    """
    if max_depth == 0:
        return _sample_terminal(registry, output_type, rng)

    functions = registry.functions(output_type=output_type)
    if len(functions) == 0:
        # No functions available, must use terminal
        return _sample_terminal(registry, output_type, rng)

    # Probability of choosing a terminal at this depth
    # (higher probability at shallow depths gives variety)
    if rng.random() < 0.3:
        return _sample_terminal(registry, output_type, rng)

    prim = functions[int(rng.integers(len(functions)))]
    children = tuple(
        generate_grow(registry, max_depth - 1, child_type, rng)
        for child_type in prim.input_types
    )
    return _make_function_node(prim, children, rng)


def initialize_population(
    registry: PrimitiveRegistry,
    config: GPConfig,
    *,
    output_type: GPType | None = None,
) -> list[Program]:
    """Initialize a population using ramped half-and-half (FR-5.1).

    Half the population uses "full" trees, half uses "grow" trees.
    Depths are ramped from 1 to max_depth for diversity.

    Args:
        registry: Primitive registry to draw from.
        config: GP configuration (population_size, max_depth, seed).
        output_type: Output type for root nodes (default: Series).

    Returns:
        List of programs of length ``config.population_size``.
    """
    if output_type is None:
        from liq.gp.types import Series

        output_type = Series

    rng = np.random.default_rng(config.seed)
    population: list[Program] = []
    pop_size = config.population_size
    max_depth = config.max_depth

    # Ramp depths from 1 to max_depth (or 0 to max_depth if max_depth < 2)
    min_init_depth = min(1, max_depth)
    depth_range = list(range(min_init_depth, max_depth + 1))
    if len(depth_range) == 0:
        depth_range = [max_depth]

    for i in range(pop_size):
        depth = depth_range[i % len(depth_range)]
        if i % 2 == 0:
            tree = generate_full(registry, depth, output_type, rng)
        else:
            tree = generate_grow(registry, depth, output_type, rng)
        population.append(tree)

    return population


# --- Seed validation -------------------------------------------------------


def _collect_primitives(program: Program) -> list[str]:
    """Collect all primitive names referenced by function/parameterized nodes."""
    names: list[str] = []
    if isinstance(program, (FunctionNode, ParameterizedNode)):
        names.append(program.primitive.name)
        for child in program.children:
            names.extend(_collect_primitives(child))
    return names


def validate_seed_programs(
    seeds: list[Program],
    config: GPConfig,
    *,
    output_type: GPType | None = None,
    registry: PrimitiveRegistry | None = None,
) -> None:
    """Validate seed programs before use in seeded initialization (FR-5.1.5).

    Raises:
        EvolutionError: If any seed fails validation.  The message includes
            the seed index and the specific violation.
    """
    from liq.gp.errors import EvolutionError
    from liq.gp.types import Series

    if output_type is None:
        output_type = Series

    if len(seeds) == 0:
        msg = "seed_programs must contain at least 1 program"
        raise EvolutionError(msg)

    if len(seeds) > config.population_size:
        msg = (
            f"Number of seed programs ({len(seeds)}) "
            f"exceeds population_size ({config.population_size})"
        )
        raise EvolutionError(msg)

    for i, seed in enumerate(seeds):
        # Output type check
        if seed.output_type != output_type:
            msg = (
                f"Seed program at index {i} has output_type "
                f"{seed.output_type}, expected {output_type}"
            )
            raise EvolutionError(msg)

        # Depth / size constraints
        if seed.depth > config.max_depth:
            msg = (
                f"Seed program at index {i} exceeds max_depth "
                f"(got {seed.depth}, limit {config.max_depth})"
            )
            raise EvolutionError(msg)

        if config.max_size is not None and seed.size > config.max_size:
            msg = (
                f"Seed program at index {i} exceeds max_size "
                f"(got {seed.size}, limit {config.max_size})"
            )
            raise EvolutionError(msg)

        # Registry compatibility: all primitives must exist
        if registry is not None:
            from liq.gp.errors import PrimitiveError

            for prim_name in _collect_primitives(seed):
                try:
                    registry.get(prim_name)
                except PrimitiveError:
                    msg = (
                        f"Seed program at index {i} references primitive "
                        f"'{prim_name}' not found in registry"
                    )
                    raise EvolutionError(msg) from None


# --- Seeded population initialization -------------------------------------


def initialize_seeded_population(
    seeds: list[Program],
    registry: PrimitiveRegistry,
    config: GPConfig,
    rng: np.random.Generator,
) -> list[Program]:
    """Build an initial population from seed programs (FR-5.1.4).

    Seeds are placed at the front of the population.  Remaining slots are
    filled by applying variation operators (crossover, mutation) to the seeds,
    using the configured operator rates.

    If all operator-derived offspring violate constraints, the function falls
    back to ``generate_grow()`` to fill remaining slots.

    Args:
        seeds: 1 to ``config.population_size`` validated seed programs.
        registry: Primitive registry for operator use.
        config: GP configuration.
        rng: Random number generator (from the engine's master stream).

    Returns:
        Population of length ``config.population_size``.
    """
    from liq.gp.evolution.constraints import enforce_constraints
    from liq.gp.evolution.operators import (
        hoist_mutation,
        parameter_mutation,
        point_mutation,
        select_operator,
        subtree_crossover,
        subtree_mutation,
    )
    from liq.gp.types import Series

    pop_size = config.population_size

    # Full population of seeds: return as-is
    if len(seeds) >= pop_size:
        return list(seeds[:pop_size])

    population = list(seeds)
    remaining = pop_size - len(seeds)

    offspring: list[Program] = []
    pi = 0  # parent index (cycles through seeds)
    max_attempts = remaining * 10  # safety valve
    attempts = 0

    while len(offspring) < remaining and attempts < max_attempts:
        attempts += 1
        op = select_operator(config, rng)

        if op == "crossover":
            p1 = seeds[pi % len(seeds)]
            p2 = seeds[(pi + 1) % len(seeds)]
            pi += 2
            child1, child2 = subtree_crossover(
                p1, p2, registry, config.max_depth, rng,
                max_attempts=config.max_crossover_attempts,
            )
            for child in (child1, child2):
                if enforce_constraints(child, config) and len(offspring) < remaining:
                    offspring.append(child)
        elif op == "subtree_mutation":
            parent = seeds[pi % len(seeds)]
            pi += 1
            child = subtree_mutation(parent, registry, config.max_depth, rng)
            if enforce_constraints(child, config):
                offspring.append(child)
        elif op == "point_mutation":
            parent = seeds[pi % len(seeds)]
            pi += 1
            child = point_mutation(parent, registry, rng)
            if enforce_constraints(child, config):
                offspring.append(child)
        elif op == "parameter_mutation":
            parent = seeds[pi % len(seeds)]
            pi += 1
            child = parameter_mutation(parent, rng)
            if enforce_constraints(child, config):
                offspring.append(child)
        elif op == "hoist_mutation":
            parent = seeds[pi % len(seeds)]
            pi += 1
            child = hoist_mutation(parent, rng)
            if enforce_constraints(child, config):
                offspring.append(child)
        else:
            # Reproduction: copy parent
            parent = seeds[pi % len(seeds)]
            pi += 1
            offspring.append(parent)

    # Fallback: fill remaining with random grow trees if operators
    # couldn't produce enough valid offspring
    if len(offspring) < remaining:
        output_type = seeds[0].output_type if seeds else Series
        while len(offspring) < remaining:
            depth = int(rng.integers(1, config.max_depth + 1))
            tree = generate_grow(registry, depth, output_type, rng)
            if enforce_constraints(tree, config):
                offspring.append(tree)

    return population + offspring[:remaining]
