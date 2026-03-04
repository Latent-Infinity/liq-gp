"""Tests for smooth switching primitives."""

from __future__ import annotations

import numpy as np

from liq.gp.primitives.registry import PrimitiveRegistry
from liq.gp.primitives.smooth_gates import (
    cooldown_gate,
    hysteresis_gate,
    register_smooth_gate_primitives,
    smooth_gate,
    softmax_gate,
)
from liq.gp.types import Series


def test_smooth_gate_is_bounded_and_continuous_in_threshold() -> None:
    """Smooth gate outputs remain in [0, 1] and vary continuously."""
    signal = np.linspace(-1.0, 1.0, 41)
    base = smooth_gate(signal, threshold=0.0, slope=2.0)
    shifted = smooth_gate(signal, threshold=0.01, slope=2.0)

    assert base.min() >= 0.0
    assert base.max() <= 1.0
    assert shifted.min() >= 0.0
    assert shifted.max() <= 1.0
    # Small threshold shift induces small but non-zero output change.
    assert np.max(np.abs(base - shifted)) < 0.05


def test_softmax_gate_stays_bounded_and_stable() -> None:
    """Softmax gate stays bounded and produces midpoint for equal scores."""
    left = np.array([0.0, 1.0, 2.0])
    right = np.array([0.0, 0.0, 1.0])
    soft_half = softmax_gate(left, right, alpha=1.0)
    soft_small_alpha = softmax_gate(left, right, alpha=0.0)
    soft_high_alpha = softmax_gate(left, right, alpha=2.0)

    assert np.all(soft_half >= 0.0)
    assert np.all(soft_half <= 1.0)
    assert np.isclose(soft_half[0], 0.5)
    assert soft_half[1] > soft_half[0]
    assert soft_half[1] < 1.0
    assert np.all(soft_small_alpha >= 0.0)
    assert np.all(soft_small_alpha <= 1.0)
    # Alpha changes should move outputs smoothly, not jump abruptly.
    assert np.max(np.abs(soft_small_alpha - soft_half)) < 0.5
    assert np.max(np.abs(soft_half - soft_high_alpha)) < 0.5


def test_hysteresis_gate_holds_state() -> None:
    """Hysteresis gate only flips when crossing the band thresholds."""
    signal = np.array([-1.0, 0.0, 0.2, 0.55, 0.7, 0.55, 0.42, 0.35, 0.25, -0.1])
    result = hysteresis_gate(signal, threshold=0.4, bandwidth=0.1)
    assert np.array_equal(
        result,
        np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0]),
    )


def test_cooldown_gate_maintains_activation() -> None:
    """Cooldown gate stays active for a fixed budget after activation."""
    signal = np.array([0.0, 1.1, 0.0, 0.0, 0.0, 1.2, 0.0, 0.0, 0.0, 0.0])
    result = cooldown_gate(signal, threshold=0.5, cooldown=3.0)
    assert np.array_equal(
        result,
        np.array([0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0]),
    )


def test_register_smooth_gate_primitives() -> None:
    """All smooth-gate primitives can be registered into a registry."""
    registry = PrimitiveRegistry()
    register_smooth_gate_primitives(registry)

    smooth = registry.get("smooth_gate")
    softmax = registry.get("softmax_gate")
    hysteresis = registry.get("hysteresis_gate")
    cooldown = registry.get("cooldown_gate")

    assert smooth.arity == 3
    assert softmax.arity == 3
    assert hysteresis.arity == 3
    assert cooldown.arity == 3
    assert smooth.output_type is Series
    assert softmax.output_type is Series
    assert hysteresis.output_type is Series
    assert smooth.input_types == (Series, Series, Series)
    assert cooldown.output_type is Series
