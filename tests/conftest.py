"""Shared fixtures for liq-gp tests."""

from __future__ import annotations

import pytest

from liq.gp.types import GPType


@pytest.fixture(autouse=True)
def _reset_type_registry() -> None:  # noqa: PT004
    """Reset GPType registry to built-ins after each test."""
    yield
    GPType._reset_registry()
