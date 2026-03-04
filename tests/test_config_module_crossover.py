"""Stage-16 config tests for module-preserving crossover mode."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from liq.gp.config import GPConfig


def test_crossover_mode_defaults_to_standard() -> None:
    config = GPConfig()
    assert config.crossover_mode == "standard"


def test_crossover_mode_accepts_module_preserving() -> None:
    config = GPConfig(crossover_mode="module_preserving")
    assert config.crossover_mode == "module_preserving"


def test_crossover_mode_rejects_unknown_value() -> None:
    with pytest.raises(ValidationError):
        GPConfig(crossover_mode="invalid_mode")
