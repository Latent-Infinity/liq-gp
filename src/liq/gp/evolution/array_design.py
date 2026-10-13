"""Low-discrepancy initial designs over the unit cube.

The design is a scrambled Sobol' sequence drawn with :mod:`scipy.stats.qmc`.
Points lie in ``[0, 1)``; mapping them onto declared ranges is the caller's
job. The module knows nothing about what the coordinates mean.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import qmc

__all__ = ["sobol_design"]


def _check_size(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int | np.integer):
        raise TypeError(f"{name} must be an int, got {type(value).__name__}")
    if value < 1:
        raise ValueError(f"{name} must be >= 1, got {value}")
    return int(value)


def sobol_design(n: int, d: int, rng: np.random.Generator) -> np.ndarray:
    """Draw ``n`` scrambled Sobol' points in ``d`` dimensions.

    A fresh ``qmc.Sobol(d, scramble=True, rng=rng)`` engine supplies the first
    ``n`` points of its sequence, so the scrambling is drawn from ``rng`` and the
    same generator state always yields the same design.

    Balance holds only when ``n`` is a power of two: then every column puts
    exactly one point in each of the ``n`` equal strata of ``[0, 1)``, and the
    leading coordinates form a (0, m, 2)-net. For any other ``n`` the result is
    the ``n``-point prefix of the next power-of-two net, which keeps the
    sequence's low discrepancy but not its balance. scipy warns about that case;
    this function draws the whole power-of-two net and returns its prefix, which
    is the same set of points without the warning and without touching the
    process-wide warning filters.

    Args:
        n: Number of points, ``>= 1``.
        d: Dimension, ``>= 1`` and at most ``qmc.Sobol.MAXDIM``.
        rng: Generator the scrambling is drawn from; it is advanced.

    Returns:
        ``float64`` array of shape ``(n, d)`` with entries in ``[0, 1)``.

    Raises:
        TypeError: ``n`` or ``d`` is not an int, or ``rng`` is not a Generator.
        ValueError: ``n`` or ``d`` is below 1.
    """
    n = _check_size("n", n)
    d = _check_size("d", d)
    if not isinstance(rng, np.random.Generator):
        raise TypeError(
            f"rng must be a numpy.random.Generator, got {type(rng).__name__}"
        )
    engine = qmc.Sobol(d=d, scramble=True, rng=rng)
    m = (n - 1).bit_length()
    return engine.random_base2(m)[:n]
