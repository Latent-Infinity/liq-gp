"""Validation edge cases for :mod:`liq.gp.types` core data types."""

from __future__ import annotations

import pytest

from liq.gp.types import ParamSpec


class TestParamSpecAllowedValuesValidation:
    """Discrete parameter validation (FR-3.3)."""

    def test_float_allowed_values_rejected_for_int_dtype(self) -> None:
        with pytest.raises(TypeError, match="all allowed_values must be int"):
            ParamSpec(
                name="period",
                dtype=int,
                default=2,
                allowed_values=[1.5, 2.5],
            )

    def test_non_numeric_allowed_values_rejected(self) -> None:
        with pytest.raises(
            TypeError, match="allowed_values must contain only int/float values"
        ):
            ParamSpec(
                name="mode",
                dtype=float,
                default=1.0,
                allowed_values=["a", "b"],  # type: ignore[list-item]
            )

    def test_bool_allowed_values_rejected(self) -> None:
        with pytest.raises(TypeError, match="allowed_values must not contain bool"):
            ParamSpec(
                name="flag",
                dtype=int,
                default=1,
                allowed_values=[True, False],  # type: ignore[list-item]
            )

    def test_non_finite_allowed_values_rejected(self) -> None:
        with pytest.raises(ValueError, match="allowed_values must be finite"):
            ParamSpec(
                name="alpha",
                dtype=float,
                default=1.0,
                allowed_values=[1.0, float("inf")],
            )


class TestParamSpecBoundsValidation:
    """Continuous parameter bound validation (FR-3.3)."""

    def test_missing_bounds_rejected(self) -> None:
        with pytest.raises(ValueError, match="min_value and max_value are required"):
            ParamSpec(name="period", dtype=int, default=1)

    def test_missing_max_value_rejected(self) -> None:
        with pytest.raises(ValueError, match="min_value and max_value are required"):
            ParamSpec(name="period", dtype=int, default=1, min_value=0)

    def test_float_min_value_rejected_for_int_dtype(self) -> None:
        with pytest.raises(TypeError, match="min_value must be int when dtype is int"):
            ParamSpec(name="period", dtype=int, default=1, min_value=0.5, max_value=10)

    def test_bool_min_value_rejected_for_int_dtype(self) -> None:
        with pytest.raises(TypeError, match="min_value must be int, not bool"):
            ParamSpec(
                name="period",
                dtype=int,
                default=1,
                min_value=True,  # type: ignore[arg-type]
                max_value=10,
            )

    def test_bool_min_value_rejected_for_float_dtype(self) -> None:
        with pytest.raises(TypeError, match="min_value must be int or float, not bool"):
            ParamSpec(
                name="alpha",
                dtype=float,
                default=0.5,
                min_value=True,  # type: ignore[arg-type]
                max_value=1.0,
            )

    def test_float_max_value_rejected_for_int_dtype(self) -> None:
        with pytest.raises(TypeError, match="max_value must be int when dtype is int"):
            ParamSpec(name="period", dtype=int, default=1, min_value=0, max_value=10.5)

    def test_bool_max_value_rejected_for_int_dtype(self) -> None:
        with pytest.raises(TypeError, match="max_value must be int, not bool"):
            ParamSpec(
                name="period",
                dtype=int,
                default=0,
                min_value=0,
                max_value=True,  # type: ignore[arg-type]
            )

    def test_bool_max_value_rejected_for_float_dtype(self) -> None:
        with pytest.raises(TypeError, match="max_value must be int or float, not bool"):
            ParamSpec(
                name="alpha",
                dtype=float,
                default=0.5,
                min_value=0.0,
                max_value=True,  # type: ignore[arg-type]
            )

    def test_non_numeric_min_value_rejected(self) -> None:
        with pytest.raises(TypeError, match="min_value must be int or float"):
            ParamSpec(
                name="alpha",
                dtype=float,
                default=0.5,
                min_value="low",  # type: ignore[arg-type]
                max_value=1.0,
            )

    def test_non_numeric_max_value_rejected(self) -> None:
        with pytest.raises(TypeError, match="max_value must be int or float"):
            ParamSpec(
                name="alpha",
                dtype=float,
                default=0.5,
                min_value=0.0,
                max_value="high",  # type: ignore[arg-type]
            )

    def test_non_finite_bounds_rejected(self) -> None:
        with pytest.raises(ValueError, match="min_value and max_value must be finite"):
            ParamSpec(
                name="alpha",
                dtype=float,
                default=0.0,
                min_value=float("-inf"),
                max_value=1.0,
            )
        with pytest.raises(ValueError, match="min_value and max_value must be finite"):
            ParamSpec(
                name="alpha",
                dtype=float,
                default=0.0,
                min_value=0.0,
                max_value=float("inf"),
            )


class TestParamSpecDefaultValidation:
    """Default-value validation (FR-3.3)."""

    def test_float_default_rejected_for_int_dtype(self) -> None:
        with pytest.raises(TypeError, match="default must be int when dtype is int"):
            ParamSpec(name="period", dtype=int, default=5.0, min_value=0, max_value=10)

    def test_bool_default_rejected_for_int_dtype(self) -> None:
        with pytest.raises(TypeError, match="default must be int when dtype is int"):
            ParamSpec(
                name="period",
                dtype=int,
                default=True,  # type: ignore[arg-type]
                min_value=0,
                max_value=10,
            )

    def test_bool_default_rejected_for_float_dtype(self) -> None:
        with pytest.raises(
            TypeError, match="default must be int or float when dtype is float"
        ):
            ParamSpec(
                name="alpha",
                dtype=float,
                default=True,  # type: ignore[arg-type]
                min_value=0.0,
                max_value=1.0,
            )

    def test_non_numeric_default_rejected_for_float_dtype(self) -> None:
        with pytest.raises(
            TypeError, match="default must be int or float when dtype is float"
        ):
            ParamSpec(
                name="alpha",
                dtype=float,
                default="mid",  # type: ignore[arg-type]
                min_value=0.0,
                max_value=1.0,
            )

    def test_default_above_max_rejected(self) -> None:
        with pytest.raises(
            ValueError, match=r"default \(20\) must be within \[0, 10\]"
        ):
            ParamSpec(name="period", dtype=int, default=20, min_value=0, max_value=10)

    def test_default_below_min_rejected(self) -> None:
        with pytest.raises(
            ValueError, match=r"default \(-1\) must be within \[0, 10\]"
        ):
            ParamSpec(name="period", dtype=int, default=-1, min_value=0, max_value=10)
