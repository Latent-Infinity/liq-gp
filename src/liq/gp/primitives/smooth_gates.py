"""Smooth gating primitives for regime-aware control flow."""

from __future__ import annotations

import numpy as np

from liq.gp.primitives.registry import PrimitiveInfo, PrimitiveRegistry
from liq.gp.types import Series


def smooth_gate(
    signal: np.ndarray,
    threshold: float = 0.0,
    slope: float = 1.0,
) -> np.ndarray:
    """Logistic gate with bounded, smooth output in ``[0, 1]``."""
    slope = np.float64(slope)
    threshold = np.float64(threshold)
    logits = slope * (signal - threshold)
    logits = np.clip(logits, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-logits))


def softmax_gate(a: np.ndarray, b: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """Smooth branch selector between two score streams.

    Returns the smooth probability of selecting ``a`` over ``b``.
    """
    logits = np.asarray(alpha, dtype=np.float64) * (a - b)
    logits = np.clip(logits, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-logits))


def hysteresis_gate(
    signal: np.ndarray,
    threshold: float = 0.0,
    bandwidth: float = 0.1,
    _slope: float = 1.0,  # reserved for API symmetry
) -> np.ndarray:
    """Hysteretic gate with upper/lower switching bands.

    The state is held while ``signal`` stays inside the hysteresis band.
    """
    lower = float(threshold - abs(float(bandwidth)))
    upper = float(threshold + abs(float(bandwidth)))
    output = np.zeros_like(signal, dtype=np.float64)
    state = 0.0
    for index, value in enumerate(signal):
        if value >= upper:
            state = 1.0
        elif value <= lower:
            state = 0.0
        output[index] = state
    return output


def cooldown_gate(
    signal: np.ndarray,
    threshold: float = 0.0,
    cooldown: float = 3.0,
    _slope: float = 1.0,  # reserved for API symmetry
) -> np.ndarray:
    """Cooldown gate that keeps the switch high for ``cooldown`` samples."""
    remaining = 0
    budget = int(max(1.0, round(float(cooldown))))
    out = np.zeros_like(signal, dtype=np.float64)
    for index, value in enumerate(signal):
        if value > threshold:
            remaining = max(0, budget - 1)
            out[index] = 1.0
            continue
        if remaining > 0:
            remaining -= 1
            out[index] = 1.0
            continue
        out[index] = 0.0
    return out


def register_smooth_gate_primitives(registry: PrimitiveRegistry) -> None:
    """Register smooth gate primitives into a ``PrimitiveRegistry``."""
    registry.register(
        "smooth_gate",
        smooth_gate,
        category="regime",
        input_types=(Series, Series, Series),
        output_type=Series,
    )
    registry.register(
        "softmax_gate",
        softmax_gate,
        category="regime",
        input_types=(Series, Series, Series),
        output_type=Series,
    )
    registry.register(
        "hysteresis_gate",
        hysteresis_gate,
        category="regime",
        input_types=(Series, Series, Series),
        output_type=Series,
    )
    registry.register(
        "cooldown_gate",
        cooldown_gate,
        category="regime",
        input_types=(Series, Series, Series),
        output_type=Series,
    )


def make_default_gate_primitive_registry() -> PrimitiveRegistry:
    """Create a stand-alone registry with smooth gate primitives pre-registered."""
    registry = PrimitiveRegistry()
    register_smooth_gate_primitives(registry)
    return registry
