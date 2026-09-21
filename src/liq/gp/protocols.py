"""Protocols (extension points) for liq-gp (NFR-3.5).

Protocols define structural interfaces that consumers implement.
They use ``typing.Protocol`` (not ABCs) so consumers never need to
inherit from liq-gp classes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from liq.gp.program.ast import Program
    from liq.gp.program.eval import EvaluationContext
    from liq.gp.types import FitnessResult, GenerationStats


@runtime_checkable
class FitnessEvaluator(Protocol):
    """Consumer-provided fitness evaluation (FR-5.4.1).

    Implement this protocol in your domain-specific evaluator.  The
    :func:`~liq.gp.evolve` function calls ``evaluate()`` once per generation
    with the full population, plus small follow-up batches for programs
    changed by constant optimisation or seed injection.  Constant
    optimisation itself probes candidates one program at a time, so
    ``evaluate()`` must accept lists of any length.

    Example::

        class MyEvaluator:
            def evaluate(self, programs, context):
                return [FitnessResult(objectives=(...,)) for p in programs]
    """

    def evaluate(
        self,
        programs: list[Program],
        context: EvaluationContext,
    ) -> list[FitnessResult]: ...


@runtime_checkable
class GenerationCallback(Protocol):
    """Callback invoked after each generation with statistics.

    Passed as ``callback=`` to :func:`~liq.gp.evolve`.  Receives a
    :class:`~liq.gp.GenerationStats` with per-generation metrics.
    """

    def __call__(self, stats: GenerationStats) -> None: ...
