"""Main evolution loop for the GP engine (FR-5.5).

Orchestrates population initialization, evaluation, selection, genetic
operators, constraints, simplification, constant optimization, semantic
deduplication, and statistics tracking.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import replace
from typing import Literal, cast

import numpy as np

from liq.gp.config import FitnessConfig, GPConfig, SchedulerConfig
from liq.gp.errors import EvolutionError
from liq.gp.evolution.constraints import apply_parsimony, enforce_constraints
from liq.gp.evolution.diversity import (
    deduplicate_population,
    sample_reference_context,
)
from liq.gp.evolution.init import initialize_population
from liq.gp.evolution.operators import (
    hoist_mutation,
    module_preserving_crossover,
    parameter_mutation,
    point_mutation,
    select_operator,
    subtree_crossover,
    subtree_mutation,
)
from liq.gp.evolution.selection import (
    compute_nsga2_rankings,
    get_elite_indices,
    get_elites,
    non_dominated_sort,
    select,
)
from liq.gp.primitives.registry import PrimitiveRegistry
from liq.gp.program.ast import Program
from liq.gp.program.simplify import simplify
from liq.gp.protocols import FitnessEvaluator
from liq.gp.types import (
    EvolutionResult,
    FitnessResult,
    GenerationStats,
    ParentSourceFn,
    SelectionContext,
)

logger = logging.getLogger(__name__)


def evolve(
    registry: PrimitiveRegistry,
    config: GPConfig,
    evaluator: FitnessEvaluator,
    context: dict[str, np.ndarray],
    *,
    fitness_config: FitnessConfig | None = None,
    callback: Callable[[GenerationStats], None] | None = None,
    parent_source: ParentSourceFn | None = None,
    seed_programs: Sequence[Program] | None = None,
) -> EvolutionResult:
    """Run the full evolution loop (FR-5.5).

    Args:
        registry: Primitive registry for tree construction.
        config: GP configuration controlling all parameters.
        evaluator: Object satisfying the :class:`~liq.gp.FitnessEvaluator`
            protocol (``evaluate(programs, context) -> list[FitnessResult]``).
        context: Evaluation context mapping names to arrays.
        fitness_config: Optional override for ``config.fitness``. This supports
            the consumer integration pattern where fitness objectives are
            configured separately from the base ``GPConfig``.
        callback: Optional per-generation callback receiving
            :class:`GenerationStats`.
        parent_source: Optional hook to source parents each generation.
            When provided, replaces :func:`select`.
        seed_programs: Optional list of 1 to ``population_size`` programs
            to seed the initial population (FR-5.1.4).  Seeds are placed
            directly; remaining slots are filled by applying variation
            operators to the seeds.  ``None`` uses random
            initialization.

    Returns:
        An :class:`EvolutionResult` with the best program, Pareto front,
        fitness history, and config.
    """
    if fitness_config is not None:
        config = config.model_copy(update={"fitness": fitness_config})

    rng = np.random.default_rng(config.seed)
    _validate_context(context)

    # --- Initialize population ---
    if seed_programs is not None:
        if len(seed_programs) == 0:
            msg = "seed_programs must be None or contain at least 1 program"
            raise EvolutionError(msg)
        from liq.gp.evolution.init import (
            initialize_seeded_population,
            validate_seed_programs,
        )

        validate_seed_programs(seed_programs, config, registry=registry)
        init_rng = rng.spawn(1)[0]
        population = initialize_seeded_population(
            seed_programs,
            registry,
            config,
            init_rng,
        )
    else:
        population = initialize_population(registry, config)

    # --- Validate seed injection requirements ---
    if (
        config.seed_injection is not None
        and config.seed_injection.method in ("direct", "variation")
        and seed_programs is None
    ):
        raise EvolutionError(
            f"seed_injection method '{config.seed_injection.method}' "
            "requires seed_programs to be provided"
        )

    # Reference context for semantic dedup (sampled once at start)
    ref_context = sample_reference_context(
        context,
        config.semantic_ref_size,
        rng,
    )

    fitness_history: list[GenerationStats] = []
    best_stall_count = 0
    prev_best: tuple[float, ...] | None = None
    injection_event_counter = 0
    final_fitnesses: list[FitnessResult] | None = None

    for gen in range(config.generations):
        # --- Determine evaluation context (batch vs full) ---
        batch_size = config.fitness.batch_size
        if batch_size is None or gen % config.fitness.full_eval_interval == 0:
            eval_context = context
        else:
            eval_context = _batch_context(context, batch_size, rng)

        # --- Evaluate (the single full-population pass of this generation) ---
        raw_fitnesses, scheduler_metrics = _evaluate_population(
            population,
            evaluator,
            eval_context,
            config,
        )

        # --- Constant optimization on the evaluated population ---
        if config.constant_opt_enabled:
            population, raw_fitnesses = _optimize_constants_stage(
                population,
                raw_fitnesses,
                evaluator,
                eval_context,
                config,
                rng,
                gen,
            )

        # --- Apply parsimony pressure + NSGA-II rankings ---
        fitnesses = apply_parsimony(raw_fitnesses, population, config)
        fronts, ranks, crowding = _rankings(fitnesses, config)

        # Stall detection measures *search* progress, so it reads the
        # population before injection: re-injecting a known seed is not a
        # discovery and must not reset the patience counter.
        search_best = _best_objectives(fitnesses, config)

        # --- Seed injection: replace worst non-elites, evaluate only them ---
        injected_count = 0
        if config.seed_injection is not None:
            population, raw_fitnesses, injected_count = _inject_seeds_stage(
                population,
                raw_fitnesses,
                fitnesses,
                evaluator,
                eval_context,
                config,
                registry,
                rng,
                gen,
                seed_programs,
                injection_event=injection_event_counter,
                ranks=ranks,
                crowding=crowding,
                protected=set(
                    get_elite_indices(
                        population,
                        fitnesses,
                        config,
                        fronts=fronts,
                        ranks=ranks,
                        crowding=crowding,
                    )
                ),
            )
            if injected_count > 0:
                injection_event_counter += 1
                fitnesses = apply_parsimony(raw_fitnesses, population, config)
                fronts, ranks, crowding = _rankings(fitnesses, config)

        selection_context = SelectionContext(
            fronts=fronts, ranks=ranks, crowding=crowding
        )

        # --- Compute generation statistics ---
        pareto_front_size = len(fronts[0]) if fronts else None
        stats = _compute_stats(
            gen,
            population,
            fitnesses,
            config,
            pareto_front_size=pareto_front_size,
        )
        stats = replace(
            stats,
            unique_semantics_ratio=_compute_unique_semantics_ratio(
                population, ref_context, config
            ),
            scheduler_metrics=scheduler_metrics,
            injected_count=injected_count,
        )

        # --- Early stopping (deferred break) ---
        should_stop = False
        if config.early_stop_patience is not None:
            if prev_best is not None:
                improvement = _primary_improvement(
                    previous=prev_best[0],
                    current=search_best[0],
                    direction=config.fitness.objective_directions[0],
                )
                if improvement < config.early_stop_threshold:
                    best_stall_count += 1
                else:
                    best_stall_count = 0
            prev_best = search_best
            if best_stall_count >= config.early_stop_patience:
                should_stop = True

        # --- Generation reporting ---
        fitness_history.append(stats)
        if callback is not None:
            callback(stats)

        # --- Stop before breeding a population nobody will evaluate ---
        if should_stop or gen == config.generations - 1:
            final_fitnesses = fitnesses if eval_context is context else None
            break

        # --- Elitism ---
        elites = get_elites(
            population,
            fitnesses,
            config,
            fronts=fronts,
            ranks=ranks,
            crowding=crowding,
        )

        # --- Selection ---
        target_size = config.population_size - len(elites)
        if parent_source is None:
            parents = select(
                population,
                fitnesses,
                config,
                rng,
                fronts=fronts,
                ranks=ranks,
                crowding=crowding,
            )
        else:
            parents = parent_source(
                population,
                fitnesses,
                config,
                rng,
                target_size,
                selection_context,
            )
            if len(parents) != target_size:
                msg = "parent_source must return exactly target_size parents"
                raise ValueError(msg)

        # --- Variation (crossover + mutation) ---
        offspring = _breed(parents, target_size, registry, config, rng)

        # --- Simplification ---
        if config.simplification_enabled:
            offspring = [simplify(p) for p in offspring]

        # --- Combine elites + offspring, then semantic deduplication ---
        population, _ = deduplicate_population(
            list(elites) + offspring,
            ref_context,
            registry,
            config,
            rng,
        )

    # --- Final evaluation only when the loop left no full-context fitnesses ---
    if final_fitnesses is None:
        raw_fitnesses, _ = _evaluate_population(population, evaluator, context, config)
        final_fitnesses = apply_parsimony(raw_fitnesses, population, config)

    # Find best program (by first objective, respecting direction)
    best_idx = _find_best(final_fitnesses, config)

    # Compute Pareto front
    fronts, _, _ = _rankings(final_fitnesses, config)
    pareto_front = [population[i] for i in fronts[0]] if fronts else []

    return EvolutionResult(
        best_program=population[best_idx],
        pareto_front=pareto_front,
        fitness_history=fitness_history,
        config=config,
    )


def _rankings(
    fitnesses: list[FitnessResult],
    config: GPConfig,
) -> tuple[list[list[int]], list[int] | None, list[float] | None]:
    """Return Pareto fronts, plus NSGA-II ranks/crowding when that mode is on."""
    if config.selection_mode == "nsga2":
        return compute_nsga2_rankings(fitnesses, config)
    directions = _effective_directions(
        config, len(fitnesses[0].objectives) if fitnesses else 0
    )
    return non_dominated_sort(fitnesses, directions), None, None


def _optimize_constants_stage(
    population: list[Program],
    fitnesses: list[FitnessResult],
    evaluator: FitnessEvaluator,
    context: dict[str, np.ndarray],
    config: GPConfig,
    rng: np.random.Generator,
    generation: int,
) -> tuple[list[Program], list[FitnessResult]]:
    """Optimize constants of the selected programs and refresh their fitness.

    The optimizer already evaluated the constants it returns, so its fitness
    is reused directly; only programs it changed without reporting a fitness
    need a follow-up evaluation.
    """
    from liq.gp.program.constants import (
        optimize_constants_with_fitness,
        select_for_optimization,
    )

    if generation % config.constant_opt_interval != 0:
        return population, fitnesses

    indices = select_for_optimization(
        population,
        fitnesses,
        config,
        rng=rng,
        max_evals=config.constant_opt_max_evals,
        generation=generation,
    )
    population = list(population)
    fitnesses = list(fitnesses)
    unpriced: list[int] = []
    for idx in indices:
        optimized, optimized_fitness = optimize_constants_with_fitness(
            population[idx], evaluator, context, config, rng
        )
        if optimized is population[idx]:
            continue
        population[idx] = optimized
        if optimized_fitness is None:
            unpriced.append(idx)
        else:
            fitnesses[idx] = optimized_fitness

    if unpriced:
        fitnesses = _refresh_fitnesses(
            population, fitnesses, unpriced, evaluator, context, config
        )
    return population, fitnesses


def _inject_seeds_stage(
    population: list[Program],
    raw_fitnesses: list[FitnessResult],
    fitnesses: list[FitnessResult],
    evaluator: FitnessEvaluator,
    context: dict[str, np.ndarray],
    config: GPConfig,
    registry: PrimitiveRegistry,
    rng: np.random.Generator,
    generation: int,
    seed_programs: Sequence[Program] | None,
    *,
    injection_event: int,
    ranks: list[int] | None,
    crowding: list[float] | None,
    protected: set[int],
) -> tuple[list[Program], list[FitnessResult], int]:
    """Run periodic seed injection and evaluate only the injected programs."""
    from liq.gp.evolution.injection import inject_seeds

    new_population, injected_count = inject_seeds(
        population,
        fitnesses,
        seed_programs,
        config,
        registry,
        rng,
        generation,
        ranks=ranks,
        crowding=crowding,
        elite_indices=protected,
        injection_event=injection_event,
        output_type=population[0].output_type if population else None,
    )
    if injected_count == 0:
        return population, raw_fitnesses, 0

    changed = [
        idx
        for idx, (before, after) in enumerate(
            zip(population, new_population, strict=True)
        )
        if before is not after
    ]
    if changed:
        raw_fitnesses = _refresh_fitnesses(
            new_population, raw_fitnesses, changed, evaluator, context, config
        )
    return new_population, raw_fitnesses, injected_count


def _refresh_fitnesses(
    population: list[Program],
    fitnesses: list[FitnessResult],
    indices: list[int],
    evaluator: FitnessEvaluator,
    context: dict[str, np.ndarray],
    config: GPConfig,
) -> list[FitnessResult]:
    """Re-evaluate ``indices`` of ``population`` in one batch, returning a new list."""
    subset, _ = _evaluate_population(
        [population[i] for i in indices], evaluator, context, config
    )
    refreshed = list(fitnesses)
    for idx, result in zip(indices, subset, strict=True):
        refreshed[idx] = result
    return refreshed


def _breed(
    parents: list[Program],
    target_size: int,
    registry: PrimitiveRegistry,
    config: GPConfig,
    rng: np.random.Generator,
) -> list[Program]:
    """Produce ``target_size`` constraint-satisfying offspring from ``parents``."""
    offspring: list[Program] = []
    pi = 0  # parent index

    def next_parent() -> Program:
        nonlocal pi
        parent = parents[pi % len(parents)]
        pi += 1
        return parent

    while len(offspring) < target_size:
        op = select_operator(config, rng)

        if op == "crossover":
            p1 = next_parent()
            p2 = next_parent()
            if config.crossover_mode == "module_preserving":
                children = module_preserving_crossover(
                    p1,
                    p2,
                    registry,
                    config.max_depth,
                    rng,
                    max_attempts=config.max_crossover_attempts,
                )
            else:
                children = subtree_crossover(
                    p1,
                    p2,
                    registry,
                    config.max_depth,
                    rng,
                    max_attempts=config.max_crossover_attempts,
                )
        elif op == "subtree_mutation":
            children = (
                subtree_mutation(
                    next_parent(),
                    registry,
                    config.max_depth,
                    rng,
                    terminal_probability=config.grow_terminal_probability,
                ),
            )
        elif op == "point_mutation":
            children = (point_mutation(next_parent(), registry, rng),)
        elif op == "parameter_mutation":
            children = (parameter_mutation(next_parent(), rng),)
        elif op == "hoist_mutation":
            children = (hoist_mutation(next_parent(), rng),)
        else:  # pragma: no cover - select_operator only emits the names above
            children = (next_parent(),)

        for child in children:
            if enforce_constraints(child, config):
                offspring.append(child)

    return offspring[:target_size]


def _compute_stats(
    generation: int,
    population: list[Program],
    fitnesses: list[FitnessResult],
    config: GPConfig,
    *,
    pareto_front_size: int | None = None,
) -> GenerationStats:
    """Compute per-generation statistics."""
    n_objectives = len(config.fitness.objectives)

    # Best fitness: for each objective, find the best value
    best_fitness = _best_objectives(fitnesses, config)

    # Mean fitness
    mean_fitness = tuple(
        float(np.mean([f.objectives[i] for f in fitnesses]))
        for i in range(n_objectives)
    )

    # Program sizes
    sizes = [p.size for p in population]
    best_idx = _find_best(fitnesses, config)
    best_program_size = sizes[best_idx]
    mean_program_size = float(np.mean(sizes))

    if pareto_front_size is None:
        directions = _effective_directions(
            config, len(fitnesses[0].objectives) if fitnesses else 0
        )
        fronts = non_dominated_sort(fitnesses, directions)
        pareto_front_size = len(fronts[0]) if fronts else 0

    return GenerationStats(
        generation=generation,
        best_fitness=best_fitness,
        mean_fitness=mean_fitness,
        best_program_size=best_program_size,
        mean_program_size=mean_program_size,
        unique_semantics_ratio=1.0,
        pareto_front_size=pareto_front_size,
    )


def _find_best(
    fitnesses: list[FitnessResult],
    config: GPConfig,
) -> int:
    """Return the best individual index using lexicographic objective ordering."""
    directions = _effective_directions(
        config, len(fitnesses[0].objectives) if fitnesses else 0
    )
    best_idx = 0
    for i in range(1, len(fitnesses)):
        if _is_better(
            fitnesses[i].objectives,
            fitnesses[best_idx].objectives,
            directions,
        ):
            best_idx = i
    return best_idx


def _best_objectives(
    fitnesses: list[FitnessResult],
    config: GPConfig,
) -> tuple[float, ...]:
    """Return the best value for each objective."""
    n = len(config.fitness.objectives)
    result: list[float] = []
    for i in range(n):
        vals = [f.objectives[i] for f in fitnesses]
        direction = config.fitness.objective_directions[i]
        if direction == "maximize":
            result.append(float(max(vals)))
        else:
            result.append(float(min(vals)))
    return tuple(result)


ObjectiveDirection = Literal["maximize", "minimize"]


def _effective_directions(
    config: GPConfig,
    objective_count: int,
) -> list[ObjectiveDirection]:
    """Return objective directions aligned with fitness objective length."""
    directions: list[ObjectiveDirection] = list(config.fitness.objective_directions)
    if objective_count > len(directions):
        pad = ["maximize"] * (objective_count - len(directions))
        directions.extend(cast(list[ObjectiveDirection], pad))
    return directions


def _is_better(
    a: tuple[float, ...],
    b: tuple[float, ...],
    directions: list[ObjectiveDirection],
) -> bool:
    """Return True if objective tuple ``a`` is lexicographically better than ``b``."""
    for i, direction in enumerate(directions):
        if a[i] == b[i]:
            continue
        if direction == "maximize":
            return a[i] > b[i]
        return a[i] < b[i]
    return False


def _primary_improvement(
    *,
    previous: float,
    current: float,
    direction: ObjectiveDirection,
) -> float:
    """Return signed improvement for the primary objective."""
    if direction == "maximize":
        return current - previous
    return previous - current


def _compute_unique_semantics_ratio(
    population: list[Program],
    ref_context: dict[str, np.ndarray],
    config: GPConfig,
) -> float:
    """Compute semantic uniqueness ratio without mutating the population."""
    fingerprints = _compute_semantic_fingerprints(
        population,
        ref_context,
        config.semantic_precision,
    )
    return _compute_unique_semantics_ratio_from_fingerprints(fingerprints)


def _compute_semantic_fingerprints(
    population: list[Program],
    ref_context: dict[str, np.ndarray],
    precision: int,
) -> list[bytes]:
    """Compute semantic fingerprints for each program in population order."""
    from liq.gp.evolution.diversity import compute_fingerprint

    return [
        compute_fingerprint(program, ref_context, precision) for program in population
    ]


def _compute_unique_semantics_ratio_from_fingerprints(
    fingerprints: list[bytes],
) -> float:
    """Compute semantic uniqueness ratio from pre-computed fingerprints."""
    if not fingerprints:
        return 1.0
    return len(set(fingerprints)) / len(fingerprints)


def _batch_context(
    context: dict[str, np.ndarray],
    batch_size: int,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """Return a random subset of the context arrays.

    If ``batch_size`` >= the context length, the original context is returned
    unchanged.
    """
    # Determine context length from first array
    first_key = next(iter(context))
    n = len(context[first_key])
    if batch_size >= n:
        return context

    indices = rng.choice(n, size=batch_size, replace=False)
    indices.sort()
    return {k: v[indices] for k, v in context.items()}


def _validate_context(context: dict[str, np.ndarray]) -> None:
    """Validate evaluation context shape and compatibility."""
    if not context:
        msg = "evaluation context must contain at least one array"
        raise ValueError(msg)

    lengths: set[int] = set()
    for key, values in context.items():
        if not isinstance(values, np.ndarray):
            msg = f"context['{key}'] must be a numpy.ndarray"
            raise TypeError(msg)
        if values.ndim != 1:
            msg = f"context['{key}'] must be 1D, got ndim={values.ndim}"
            raise ValueError(msg)
        lengths.add(len(values))

    if len(lengths) != 1:
        lengths_str = ", ".join(str(length) for length in sorted(lengths))
        msg = f"context arrays must all have the same length, got [{lengths_str}]"
        raise ValueError(msg)


def _evaluate_population(
    population: list[Program],
    evaluator: FitnessEvaluator,
    context: dict[str, np.ndarray],
    config: GPConfig,
) -> tuple[list[FitnessResult], dict[str, object]]:
    """Evaluate the full population via the evaluator.

    Parallelism is the evaluator's responsibility: liq-gp always passes
    the complete population list and context, letting the consuming library
    choose the execution strategy (threads, processes, GPU, etc.).
    """
    scheduler = config.scheduler
    if not scheduler.enabled:
        return (
            evaluator.evaluate(population, context),
            {
                "mode": "direct",
                "enabled": False,
                "peak_in_flight": 1,
                "submitted_chunks": 1,
                "completed_chunks": 1,
                "saturation_reason_code": "ok",
            },
        )
    return _evaluate_population_bounded(population, evaluator, context, scheduler)


def _evaluate_population_bounded(
    population: list[Program],
    evaluator: FitnessEvaluator,
    context: dict[str, np.ndarray],
    scheduler: SchedulerConfig,
) -> tuple[list[FitnessResult], dict[str, object]]:
    start = time.perf_counter()
    chunks = [
        (idx, population[idx : idx + scheduler.eval_batch_size])
        for idx in range(0, len(population), scheduler.eval_batch_size)
    ]
    context_bytes = int(sum(values.nbytes for values in context.values()))
    memory_budget_bytes = int(scheduler.memory_budget_mb * 1024 * 1024)
    estimated_inflight_bytes = context_bytes * max(
        1, min(scheduler.max_in_flight, len(chunks))
    )

    def _sequential(reason_code: str) -> tuple[list[FitnessResult], dict[str, object]]:
        elapsed = float(time.perf_counter() - start)
        logger.warning(
            "bounded_evaluator_saturation_fallback reason=%s stage=%s elapsed_seconds=%.6f",
            reason_code,
            "sequential_fallback",
            elapsed,
        )
        return (
            evaluator.evaluate(population, context),
            {
                "mode": "sequential_fallback",
                "enabled": True,
                "max_in_flight": scheduler.max_in_flight,
                "queue_capacity": scheduler.queue_capacity,
                "eval_batch_size": scheduler.eval_batch_size,
                "timeout_seconds": scheduler.eval_timeout_seconds,
                "memory_budget_mb": scheduler.memory_budget_mb,
                "peak_in_flight": 1,
                "submitted_chunks": 1,
                "completed_chunks": 1,
                "saturation_reason_code": reason_code,
                "elapsed_seconds": elapsed,
            },
        )

    def _saturation(reason_code: str) -> tuple[list[FitnessResult], dict[str, object]]:
        elapsed = float(time.perf_counter() - start)
        logger.warning(
            "bounded_evaluator_saturated reason=%s stage=%s elapsed_seconds=%.6f"
            " queue_capacity=%s max_in_flight=%s eval_batch_size=%s"
            " submitted_chunks=%s completed_chunks=%s context_mb=%s",
            reason_code,
            "bounded_scheduler",
            elapsed,
            scheduler.queue_capacity,
            scheduler.max_in_flight,
            scheduler.eval_batch_size,
            len(chunks),
            0,
            float(context_bytes / (1024 * 1024)),
        )
        if scheduler.safe_fallback_mode == "sequential":
            return _sequential(reason_code)
        logger.error(
            "bounded_evaluator_failed reason=%s stage=%s elapsed_seconds=%.6f",
            reason_code,
            "bounded_scheduler",
            elapsed,
        )
        raise EvolutionError(reason_code)

    if len(chunks) > scheduler.queue_capacity:
        return _saturation("scheduler_queue_saturated")

    if estimated_inflight_bytes > memory_budget_bytes:
        return _saturation("scheduler_memory_saturated")

    results: list[FitnessResult | None] = [None] * len(population)
    submitted_chunks = 0
    completed_chunks = 0
    peak_in_flight = 0

    with ThreadPoolExecutor(max_workers=scheduler.max_cpu_workers) as pool:
        pending: dict[Future[list[FitnessResult]], tuple[int, int]] = {}
        cursor = 0
        while cursor < len(chunks) or pending:
            while cursor < len(chunks) and len(pending) < scheduler.max_in_flight:
                start_idx, batch = chunks[cursor]
                fut = pool.submit(evaluator.evaluate, batch, context)
                pending[fut] = (start_idx, len(batch))
                submitted_chunks += 1
                cursor += 1
                if len(pending) > peak_in_flight:
                    peak_in_flight = len(pending)

            if not pending:
                continue

            done, _ = wait(
                list(pending.keys()),
                timeout=scheduler.eval_timeout_seconds,
                return_when=FIRST_COMPLETED,
            )
            if not done:
                for future in pending:
                    future.cancel()
                return _saturation("scheduler_timeout")

            for future in done:
                start_idx, batch_len = pending.pop(future)
                try:
                    batch_result = future.result()
                except Exception:
                    for pending_future in pending:
                        pending_future.cancel()
                    return _saturation("scheduler_worker_error")

                if len(batch_result) != batch_len:
                    for pending_future in pending:
                        pending_future.cancel()
                    return _saturation("scheduler_result_size_mismatch")

                for offset, result in enumerate(batch_result):
                    results[start_idx + offset] = result
                completed_chunks += 1

    if any(item is None for item in results):
        return _saturation("scheduler_incomplete_results")

    elapsed = float(time.perf_counter() - start)
    finalized = cast(list[FitnessResult], results)
    return finalized, {
        "mode": "bounded",
        "enabled": True,
        "max_in_flight": scheduler.max_in_flight,
        "queue_capacity": scheduler.queue_capacity,
        "eval_batch_size": scheduler.eval_batch_size,
        "timeout_seconds": scheduler.eval_timeout_seconds,
        "memory_budget_mb": scheduler.memory_budget_mb,
        "context_mb": float(context_bytes / (1024 * 1024)),
        "estimated_inflight_mb": float(estimated_inflight_bytes / (1024 * 1024)),
        "peak_in_flight": peak_in_flight,
        "submitted_chunks": submitted_chunks,
        "completed_chunks": completed_chunks,
        "saturation_reason_code": "ok",
        "elapsed_seconds": elapsed,
    }
