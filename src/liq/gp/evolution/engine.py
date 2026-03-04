"""Main evolution loop for the GP engine (FR-5.5).

Orchestrates population initialization, evaluation, selection, genetic
operators, constraints, simplification, constant optimization, semantic
deduplication, and statistics tracking.
"""

from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from inspect import signature
from collections.abc import Callable, Sequence
from dataclasses import replace
import logging
import time
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
    get_elites,
    non_dominated_sort,
    select,
)
from liq.gp.primitives.registry import PrimitiveRegistry
from liq.gp.program.ast import Program
from liq.gp.program.simplify import simplify
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
    evaluator: object,
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
        evaluator: Object with ``evaluate(programs, context)`` returning
            ``list[FitnessResult]``.
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

    for gen in range(config.generations):
        # --- Determine evaluation context (batch vs full) ---
        batch_size = config.fitness.batch_size
        if batch_size is None or gen % config.fitness.full_eval_interval == 0:
            eval_context = context
        else:
            eval_context = _batch_context(context, batch_size, rng)

        # --- Evaluate ---
        fitnesses = _evaluate_population(
            population,
            evaluator,
            eval_context,
            config,
        )
        fitnesses, scheduler_metrics = fitnesses

        # --- Apply parsimony pressure ---
        fitnesses = apply_parsimony(fitnesses, population, config)

        # --- Reuse non-dominated sorting for NSGA-II ---
        fronts = None
        ranks = None
        crowding = None
        if config.selection_mode == "nsga2":
            fronts, ranks, crowding = compute_nsga2_rankings(fitnesses, config)
        selection_context = SelectionContext(fronts=fronts, ranks=ranks, crowding=crowding)

        # --- Compute generation statistics ---
        pareto_front_size = len(fronts[0]) if fronts else None
        stats = _compute_stats(
            gen,
            population,
            fitnesses,
            config,
            pareto_front_size=pareto_front_size,
        )
        fingerprints = _compute_semantic_fingerprints(
            population,
            ref_context,
            config.semantic_precision,
        )
        stats = replace(
            stats,
            unique_semantics_ratio=_compute_unique_semantics_ratio_from_fingerprints(
                fingerprints
            ),
            scheduler_metrics=scheduler_metrics,
        )

        # --- Early stopping (deferred break) ---
        should_stop = False
        if config.early_stop_patience is not None:
            if prev_best is not None:
                improvement = _primary_improvement(
                    previous=prev_best[0],
                    current=stats.best_fitness[0],
                    direction=config.fitness.objective_directions[0],
                )
                if improvement < config.early_stop_threshold:
                    best_stall_count += 1
                else:
                    best_stall_count = 0
            prev_best = stats.best_fitness
            if best_stall_count >= config.early_stop_patience:
                should_stop = True

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
        offspring: list[Program] = []
        pi = 0  # parent index

        while len(offspring) < target_size:
            op = select_operator(config, rng)

            if op == "crossover":
                p1 = parents[pi % len(parents)]
                p2 = parents[(pi + 1) % len(parents)]
                pi += 2
                if config.crossover_mode == "module_preserving":
                    child1, child2 = module_preserving_crossover(
                        p1,
                        p2,
                        registry,
                        config.max_depth,
                        rng,
                        max_attempts=config.max_crossover_attempts,
                    )
                else:
                    child1, child2 = subtree_crossover(
                        p1,
                        p2,
                        registry,
                        config.max_depth,
                        rng,
                        max_attempts=config.max_crossover_attempts,
                    )
                for child in (child1, child2):
                    if enforce_constraints(child, config):
                        offspring.append(child)
            elif op == "subtree_mutation":
                parent = parents[pi % len(parents)]
                pi += 1
                child = subtree_mutation(parent, registry, config.max_depth, rng)
                if enforce_constraints(child, config):
                    offspring.append(child)
            elif op == "point_mutation":
                parent = parents[pi % len(parents)]
                pi += 1
                child = point_mutation(parent, registry, rng)
                if enforce_constraints(child, config):
                    offspring.append(child)
            elif op == "parameter_mutation":
                parent = parents[pi % len(parents)]
                pi += 1
                child = parameter_mutation(parent, rng)
                if enforce_constraints(child, config):
                    offspring.append(child)
            elif op == "hoist_mutation":
                parent = parents[pi % len(parents)]
                pi += 1
                child = hoist_mutation(parent, rng)
                if enforce_constraints(child, config):
                    offspring.append(child)
            else:
                # Reproduction: copy parent unchanged
                parent = parents[pi % len(parents)]
                pi += 1
                offspring.append(parent)

        # Trim to exact target size
        offspring = offspring[:target_size]

        # --- Simplification ---
        if config.simplification_enabled:
            offspring = [simplify(p) for p in offspring]

        # --- Combine elites + offspring ---
        population = list(elites) + offspring

        # --- Constant optimization ---
        if config.constant_opt_enabled:
            from liq.gp.program.constants import (
                optimize_constants,
                select_for_optimization,
            )
            constant_opt_budget = config.constant_opt_max_evals

            # Re-evaluate after combination
            fitnesses = _evaluate_population(
                population,
                evaluator,
                context,
                config,
            )
            fitnesses, _ = fitnesses
            select_for_optimization_sig = signature(select_for_optimization)
            select_kwargs = {}
            if "generation" in select_for_optimization_sig.parameters:
                select_kwargs["generation"] = gen
            if "rng" in select_for_optimization_sig.parameters:
                select_kwargs["rng"] = rng
            if "max_evals" in select_for_optimization_sig.parameters:
                select_kwargs["max_evals"] = constant_opt_budget

            indices = select_for_optimization(
                population,
                fitnesses,
                config,
                **select_kwargs,
            )
            for idx in indices:
                population[idx] = optimize_constants(
                    population[idx],
                    evaluator,
                    context,
                    config,
                    rng,
                )
            if indices:
                fingerprints = _compute_semantic_fingerprints(
                    population,
                    ref_context,
                    config.semantic_precision,
                )

        # --- Seed injection ---
        # NOTE: Stats were computed above from the evaluated population.
        # Injection replaces worst individuals for the *next* generation;
        # injected_count is patched into stats as additive metadata.
        injected_count = 0
        if config.seed_injection is not None:
            from liq.gp.evolution.injection import inject_seeds

            # Always re-evaluate to ensure fitnesses reflect current population
            # state (including any constant optimization modifications).
            fitnesses = _evaluate_population(
                population,
                evaluator,
                context,
                config,
            )
            fitnesses, _ = fitnesses
            # Recompute NSGA-II rankings for the new population
            inj_ranks: list[int] | None = None
            inj_crowding: list[float] | None = None
            if config.selection_mode == "nsga2":
                _, inj_ranks, inj_crowding = compute_nsga2_rankings(fitnesses, config)
            population, injected_count = inject_seeds(
                population,
                fitnesses,
                seed_programs,
                config,
                registry,
                rng,
                gen,
                ranks=inj_ranks,
                crowding=inj_crowding,
                elite_indices=set(range(len(elites))),
                injection_event=injection_event_counter,
                output_type=population[0].output_type if population else None,
            )
            if injected_count > 0:
                injection_event_counter += 1
            # Recompute fingerprints if injection changed the population
            if injected_count > 0:
                fingerprints = _compute_semantic_fingerprints(
                    population,
                    ref_context,
                    config.semantic_precision,
                )

        # --- Semantic deduplication ---
        population, _ = deduplicate_population(
            population,
            ref_context,
            registry,
            config,
            rng,
            fingerprints=fingerprints,
        )

        # --- Generation reporting ---
        if injected_count > 0:
            stats = replace(stats, injected_count=injected_count)
        fitness_history.append(stats)
        if callback is not None:
            callback(stats)

        # --- Deferred early-stop break ---
        if should_stop:
            break

    # --- Final evaluation for result extraction ---
    fitnesses = _evaluate_population(
        population,
        evaluator,
        context,
        config,
    )
    fitnesses, _ = fitnesses
    fitnesses = apply_parsimony(fitnesses, population, config)

    # Find best program (by first objective, respecting direction)
    best_idx = _find_best(fitnesses, config)

    # Compute Pareto front
    directions = _effective_directions(
        config, len(fitnesses[0].objectives) if fitnesses else 0
    )
    fronts = non_dominated_sort(fitnesses, directions)
    pareto_front = [population[i] for i in fronts[0]] if fronts else []

    return EvolutionResult(
        best_program=population[best_idx],
        pareto_front=pareto_front,
        fitness_history=fitness_history,
        config=config,
    )


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
    evaluator: object,
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
            evaluator.evaluate(population, context),  # type: ignore[union-attr]
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
    evaluator: object,
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
    estimated_inflight_bytes = context_bytes * max(1, min(scheduler.max_in_flight, len(chunks)))

    def _sequential(reason_code: str) -> tuple[list[FitnessResult], dict[str, object]]:
        elapsed = float(time.perf_counter() - start)
        logger.warning(
            "bounded_evaluator_saturation_fallback reason=%s stage=%s elapsed_seconds=%.6f",
            reason_code,
            "sequential_fallback",
            elapsed,
        )
        return (
            evaluator.evaluate(population, context),  # type: ignore[union-attr]
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
                fut = pool.submit(evaluator.evaluate, batch, context)  # type: ignore[union-attr]
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
