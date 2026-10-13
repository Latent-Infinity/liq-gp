"""Variation primitives for fixed-length real-valued arrays.

The family is the one an array genome is varied with: a self-adaptive
log-normal step for a per-individual mutation scale, uniform recombination,
and Gaussian mutation in unit coordinates with reflection at the bounds, where
positions declared as bits are flipped instead of moved. Every primitive takes
plain ``numpy`` arrays and draws only from the ``numpy.random.Generator`` it is
given, in a fixed order, so a seed reproduces a birth exactly. The module knows
nothing about what a position means; the caller declares bounds and bits.

A composed birth draws, on one generator: the scale step
(:func:`lognormal_scale_step`), then recombination (:func:`uniform_crossover`),
then mutation (:func:`gaussian_mutation`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

__all__ = [
    "Mutation",
    "ScaleStep",
    "gaussian_mutation",
    "lognormal_scale_step",
    "reflect_unit",
    "uniform_crossover",
]


@dataclass(frozen=True)
class ScaleStep:
    """One self-adaptive scale step.

    Attributes:
        parent_scale: The scale the step started from: the parents' geometric
            mean (the common value when they are equal).
        scale: The child's scale, within ``[low, high]``.
        fired: Whether the unreflected step left ``[low, high]``.
    """

    parent_scale: float
    scale: float
    fired: bool


@dataclass(frozen=True)
class Mutation:
    """One mutation of an array.

    Attributes:
        values: The mutated array; the input is not modified.
        selected: Boolean mask of the positions chosen for mutation.
        fired: Boolean mask of the positions where reflection acted. Summing it
            over calls gives per-position firing counts.
    """

    values: np.ndarray
    selected: np.ndarray
    fired: np.ndarray


def _check_rng(rng: object) -> None:
    if not isinstance(rng, np.random.Generator):
        raise TypeError(
            f"rng must be a numpy.random.Generator, got {type(rng).__name__}"
        )


def _vector(name: str, value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    return array


def _positive_finite(name: str, value: float) -> float:
    value = float(value)
    if not (math.isfinite(value) and value > 0.0):
        raise ValueError(f"{name} must be finite and > 0, got {value}")
    return value


def _fold(u: np.ndarray) -> np.ndarray:
    v = np.mod(u, 2.0)
    return np.where(v > 1.0, 2.0 - v, v)


def reflect_unit(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Reflect values into ``[0, 1]``.

    A value outside ``[0, 1]`` is folded back: taken modulo 2 and mirrored
    about 1. No random draw is made.

    Args:
        values: One-dimensional finite array.

    Returns:
        ``(reflected, fired)``: a new array within ``[0, 1]``, and the boolean
        mask of the positions that were outside it.

    Raises:
        ValueError: ``values`` is not one-dimensional or not finite.
    """
    u = _vector("values", values).copy()
    fired = (u < 0.0) | (u > 1.0)
    if fired.any():
        u[fired] = _fold(u[fired])
    return u, fired


def lognormal_scale_step(
    scale_1: float,
    scale_2: float,
    n: int,
    rng: np.random.Generator,
    *,
    low: float,
    high: float,
) -> ScaleStep:
    """Draw a child's mutation scale from its parents' scales.

    The step starts from the parents' geometric mean ``s`` and returns
    ``s * exp(z / sqrt(n))`` for one standard normal draw ``z``. A value outside
    ``[low, high]`` is reflected in log space: its position in
    ``[log(low), log(high)]`` is folded as by :func:`reflect_unit`, and a fold
    landing exactly on an end returns that end.

    Args:
        scale_1: First parent's scale, within ``[low, high]``.
        scale_2: Second parent's scale, within ``[low, high]``.
        n: Number of positions the scale serves; the learning rate is
            ``1 / sqrt(n)``.
        rng: Generator for the one normal draw; it is advanced.
        low: Lower bound on the scale, ``> 0``.
        high: Upper bound on the scale, ``> low`` and finite.

    Returns:
        The :class:`ScaleStep`.

    Raises:
        TypeError: ``n`` is not an int, or ``rng`` is not a Generator.
        ValueError: A bound or scale is not finite and positive, the bounds are
            not increasing, a parent scale lies outside them, or ``n < 1``.
    """
    if isinstance(n, bool) or not isinstance(n, int | np.integer):
        raise TypeError(f"n must be an int, got {type(n).__name__}")
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    _check_rng(rng)
    low = _positive_finite("low", low)
    high = _positive_finite("high", high)
    if not low < high:
        raise ValueError(f"low must be < high, got [{low}, {high}]")
    s1 = _positive_finite("scale_1", scale_1)
    s2 = _positive_finite("scale_2", scale_2)
    if not (low <= s1 <= high and low <= s2 <= high):
        raise ValueError(f"parent scales {s1}, {s2} must lie within [{low}, {high}]")

    parent = s1 if s1 == s2 else math.sqrt(s1 * s2)
    raw = parent * math.exp(rng.standard_normal() / math.sqrt(n))
    if low <= raw <= high:
        return ScaleStep(parent, raw, False)
    log_low, log_high = math.log(low), math.log(high)
    v = float(_fold(np.array([(math.log(raw) - log_low) / (log_high - log_low)]))[0])
    if v == 0.0:
        return ScaleStep(parent, low, True)
    if v == 1.0:
        return ScaleStep(parent, high, True)
    return ScaleStep(parent, math.exp(log_low + v * (log_high - log_low)), True)


def uniform_crossover(
    parent_1: np.ndarray, parent_2: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Recombine two arrays position by position.

    Each position is taken from ``parent_2`` where a uniform draw is below one
    half and from ``parent_1`` otherwise: one draw per position, in order. No
    value is invented, so identical parents give a copy of either.

    Args:
        parent_1: One-dimensional finite array.
        parent_2: Array of the same shape.
        rng: Generator for the draws; it is advanced.

    Returns:
        A new array.

    Raises:
        TypeError: ``rng`` is not a Generator.
        ValueError: The parents are not finite one-dimensional arrays of equal
            length.
    """
    _check_rng(rng)
    a = _vector("parent_1", parent_1)
    b = _vector("parent_2", parent_2)
    if a.shape != b.shape:
        raise ValueError(f"parents differ in shape: {a.shape} and {b.shape}")
    return np.where(rng.random(a.size) < 0.5, b, a)


def gaussian_mutation(
    values: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    scale: float,
    rng: np.random.Generator,
    *,
    bits: np.ndarray | None = None,
) -> Mutation:
    """Mutate an array within per-position bounds.

    Each of the ``n`` positions is selected with probability ``1 / n``; if none
    is, one is chosen uniformly. A selected bit position is flipped
    (``1 - x``). Every other selected position moves in unit coordinates,
    ``u' = u + scale * z`` with ``u = (x - low) / (high - low)`` and one standard
    normal ``z`` per moved position in order, is reflected into ``[0, 1]`` by
    :func:`reflect_unit`, and is mapped back; a unit value of exactly 0 or 1
    returns the bound itself.

    Draw order: ``n`` uniforms for the selection, one integer only if nothing
    was selected, then the normals.

    Args:
        values: One-dimensional finite array within ``[low, high]``; bit
            positions must hold 0 or 1.
        low: Per-position lower bounds.
        high: Per-position upper bounds, each ``> low``.
        scale: Step size in unit coordinates, finite and ``> 0``.
        rng: Generator for the draws; it is advanced.
        bits: Optional boolean mask of the positions held at ``{0, 1}``.

    Returns:
        The :class:`Mutation`.

    Raises:
        TypeError: ``rng`` is not a Generator.
        ValueError: Shapes disagree, an input is not finite, a bound pair is not
            increasing, a value lies outside its bounds, a bit position holds
            neither 0 nor 1, ``bits`` is not boolean, or ``scale`` is not
            finite and positive.
    """
    _check_rng(rng)
    x = _vector("values", values)
    lo = _vector("low", low)
    hi = _vector("high", high)
    if not (lo.shape == hi.shape == x.shape):
        raise ValueError(
            f"values, low and high differ in shape: {x.shape}, {lo.shape}, {hi.shape}"
        )
    if not np.all(lo < hi):
        raise ValueError("every low must be < its high")
    if not np.all((x >= lo) & (x <= hi)):
        raise ValueError("every value must lie within its bounds")
    scale = _positive_finite("scale", scale)
    n = x.size
    if bits is None:
        is_bit = np.zeros(n, dtype=bool)
    else:
        is_bit = np.asarray(bits)
        if is_bit.dtype != np.bool_ or is_bit.shape != x.shape:
            raise ValueError("bits must be a boolean mask of the values' shape")
        if not np.all((x[is_bit] == 0.0) | (x[is_bit] == 1.0)):
            raise ValueError("every bit position must hold 0 or 1")

    selected = rng.random(n) < 1.0 / n
    if not selected.any():
        selected[rng.integers(n)] = True
    out = x.copy()
    flip = selected & is_bit
    out[flip] = 1.0 - x[flip]
    fired = np.zeros(n, dtype=bool)
    idx = np.flatnonzero(selected & ~is_bit)
    if idx.size:
        width = hi - lo
        u = (x[idx] - lo[idx]) / width[idx]
        u, hit = reflect_unit(u + scale * rng.standard_normal(idx.size))
        fired[idx] = hit
        moved = lo[idx] + u * width[idx]
        moved = np.where(u == 0.0, lo[idx], moved)
        out[idx] = np.where(u == 1.0, hi[idx], moved)
    return Mutation(out, selected, fired)
