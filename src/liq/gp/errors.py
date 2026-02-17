"""Exception hierarchy for liq-gp.

All exceptions inherit from :class:`GPError`.  Consumers can catch
``GPError`` for a blanket handler or individual subclasses for
fine-grained control.

Fatality semantics:

- Most exceptions are **fatal** -- they indicate a programming error
  or unrecoverable condition and should propagate.
- :class:`ConstantOptError` is **non-fatal** -- it is logged and the
  original program constants are kept.  The evolution loop never
  raises this to the caller.
"""

from __future__ import annotations


class GPError(Exception):
    """Base exception for all liq-gp errors.

    All liq-gp exceptions subclass this, so ``except GPError`` acts as
    a catch-all for library errors.
    """


class PrimitiveError(GPError):
    """Raised on primitive registry issues.

    Trigger conditions:

    - Registering a primitive with a name that is already taken.
    - Looking up a primitive name that does not exist in the registry.
    - Arity mismatch: explicit ``arity`` argument does not match
      ``len(input_types)``.
    """


class TypeCheckError(GPError):
    """Raised when a program node violates the GP type system.

    Trigger conditions:

    - Constructing a :class:`~liq.gp.FunctionNode` or
      :class:`~liq.gp.ParameterizedNode` where the number of children
      does not match the primitive's arity.
    - A child node's ``output_type`` does not match the corresponding
      ``input_type`` in the primitive's signature.

    This is the GP-specific analogue of Python's built-in ``TypeError``,
    scoped to the GP type system so consumers can catch it separately.
    """


class EvaluationError(GPError):
    """Raised when program evaluation fails.

    Trigger conditions:

    - A terminal name is not found in the evaluation context.
    - A primitive callable raises an unrecoverable error during
      vectorised evaluation.
    """


class EvolutionError(GPError):
    """Raised when the evolution loop encounters an unrecoverable error.

    Trigger conditions:

    - The primitive registry has no terminals or no functions of a
      required output type, making tree construction impossible.
    - Internal consistency failures during the evolution loop.
    """


class SimplificationError(GPError):
    """Raised when program simplification fails.

    Trigger conditions:

    - A custom simplification rule raises an unexpected exception.

    In practice this is rare; the built-in rules are deterministic and
    well-tested.
    """


class ConstantOptError(GPError):
    """Raised when constant optimization fails.

    **Non-fatal.**  The evolution engine catches this exception, logs a
    warning, and keeps the program's original constant values.  It is
    never propagated to the caller of :func:`~liq.gp.evolve`.

    Trigger conditions:

    - ``scipy.optimize.minimize`` raises during Nelder-Mead optimisation.
    - The objective function produces only NaN values.
    - The wall-clock or iteration limit is exceeded (handled gracefully,
      using the best constants found so far).
    """


class SerializationError(GPError):
    """Raised on serialization or deserialization failures.

    Trigger conditions:

    - The ``schema_version`` in the serialized data does not match the
      current library version.
    - A primitive name in the serialized tree is not found in the
      provided :class:`~liq.gp.PrimitiveRegistry`.
    - A GP type name in the serialized data is not registered.
    - The node type field is unrecognised or the data is corrupt.
    """


class ConfigurationError(GPError):
    """Raised when a :class:`~liq.gp.GPConfig` or
    :class:`~liq.gp.FitnessConfig` is constructed with invalid values.

    Trigger conditions:

    - ``population_size < 10``, ``max_depth < 2``
    - Operator rates do not sum to 1.0
    - ``tournament_size < 2`` or ``> population_size``
    - ``elitism_count < 0`` or ``>= population_size``
    - ``constant_opt_top_k`` not in ``(0.0, 1.0]``
    - ``len(objectives) != len(objective_directions)``
    - NSGA-II with fewer than 2 objectives
    - Pareto parsimony without NSGA-II selection

    Raised at construction time (fail-fast) so invalid configs never
    reach the evolution loop.
    """
