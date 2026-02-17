"""Tests for PrimitiveRegistry (FR-3)."""

from __future__ import annotations

import numpy as np
import pytest

from liq.gp.errors import PrimitiveError
from liq.gp.primitives.registry import PrimitiveInfo, PrimitiveRegistry
from liq.gp.types import BoolSeries, Int, ParamSpec, Series

# --- helpers ---------------------------------------------------------------


def _add(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a + b


def _neg(a: np.ndarray) -> np.ndarray:
    return -a


def _close(ctx: dict[str, np.ndarray]) -> np.ndarray:
    return ctx["close"]


def _const() -> float:
    return 1.0


def _highest(a: np.ndarray, *, period: int = 20) -> np.ndarray:
    # simplified stub
    return a


# --- direct registration ---------------------------------------------------


class TestDirectRegistration:
    """registry.register(name, callable, ...)."""

    def test_register_function(self) -> None:
        reg = PrimitiveRegistry()
        reg.register(
            "add",
            _add,
            category="numeric",
            input_types=(Series, Series),
            output_type=Series,
        )
        info = reg.get("add")
        assert info.name == "add"
        assert info.category == "numeric"
        assert info.arity == 2
        assert info.input_types == (Series, Series)
        assert info.output_type is Series
        assert info.callable is _add
        assert info.param_specs == []

    def test_register_terminal(self) -> None:
        reg = PrimitiveRegistry()
        reg.register(
            "close",
            _close,
            category="data",
            input_types=(),
            output_type=Series,
        )
        info = reg.get("close")
        assert info.arity == 0

    def test_register_with_param_specs(self) -> None:
        ps = ParamSpec(name="period", dtype=int, default=20, min_value=2, max_value=200)
        reg = PrimitiveRegistry()
        reg.register(
            "highest",
            _highest,
            category="indicator",
            input_types=(Series,),
            output_type=Series,
            param_specs=[ps],
        )
        info = reg.get("highest")
        assert len(info.param_specs) == 1
        assert info.param_specs[0].name == "period"


# --- decorator registration ------------------------------------------------


class TestDecoratorRegistration:
    """@registry.primitive(name, ...)."""

    def test_decorator_registers(self) -> None:
        reg = PrimitiveRegistry()

        @reg.primitive(
            "neg",
            category="numeric",
            input_types=(Series,),
            output_type=Series,
        )
        def neg_fn(a: np.ndarray) -> np.ndarray:
            return -a

        info = reg.get("neg")
        assert info.name == "neg"
        assert info.callable is neg_fn

    def test_decorator_returns_original_function(self) -> None:
        reg = PrimitiveRegistry()

        @reg.primitive(
            "neg2",
            category="numeric",
            input_types=(Series,),
            output_type=Series,
        )
        def neg_fn(a: np.ndarray) -> np.ndarray:
            return -a

        # decorated function should still be the original callable
        result = neg_fn(np.array([1.0, 2.0]))
        np.testing.assert_array_equal(result, np.array([-1.0, -2.0]))


# --- bulk registration -----------------------------------------------------


class TestBulkRegistration:
    """registry.register_from_metadata(metadata_list) (FR-3.4)."""

    def test_bulk_register(self) -> None:
        reg = PrimitiveRegistry()
        metadata = [
            {
                "name": "add",
                "callable": _add,
                "category": "numeric",
                "input_types": (Series, Series),
                "output_type": Series,
            },
            {
                "name": "neg",
                "callable": _neg,
                "category": "numeric",
                "input_types": (Series,),
                "output_type": Series,
            },
        ]
        reg.register_from_metadata(metadata)
        assert reg.get("add").arity == 2
        assert reg.get("neg").arity == 1

    def test_bulk_register_with_param_specs(self) -> None:
        ps = ParamSpec(name="period", dtype=int, default=20, min_value=2, max_value=200)
        reg = PrimitiveRegistry()
        metadata = [
            {
                "name": "highest",
                "callable": _highest,
                "category": "indicator",
                "input_types": (Series,),
                "output_type": Series,
                "param_specs": [ps],
            },
        ]
        reg.register_from_metadata(metadata)
        info = reg.get("highest")
        assert len(info.param_specs) == 1


# --- query methods ----------------------------------------------------------


class TestQueryMethods:
    """list_primitives, get, terminals, functions (FR-3.5)."""

    @pytest.fixture()
    def populated_registry(self) -> PrimitiveRegistry:
        reg = PrimitiveRegistry()
        reg.register(
            "close", _close, category="data", input_types=(), output_type=Series
        )
        reg.register(
            "volume", _close, category="data", input_types=(), output_type=Series
        )
        reg.register(
            "add",
            _add,
            category="numeric",
            input_types=(Series, Series),
            output_type=Series,
        )
        reg.register(
            "neg", _neg, category="numeric", input_types=(Series,), output_type=Series
        )
        reg.register(
            "gt",
            lambda a, b: (a > b).astype(float),
            category="comparison",
            input_types=(Series, Series),
            output_type=BoolSeries,
        )
        return reg

    def test_list_all(self, populated_registry: PrimitiveRegistry) -> None:
        all_prims = populated_registry.list_primitives()
        assert len(all_prims) == 5

    def test_list_by_category(self, populated_registry: PrimitiveRegistry) -> None:
        numeric = populated_registry.list_primitives(category="numeric")
        assert len(numeric) == 2
        assert all(p.category == "numeric" for p in numeric)

    def test_list_by_category_empty(
        self, populated_registry: PrimitiveRegistry
    ) -> None:
        result = populated_registry.list_primitives(category="nonexistent")
        assert result == []

    def test_terminals(self, populated_registry: PrimitiveRegistry) -> None:
        terms = populated_registry.terminals()
        assert len(terms) == 2
        assert all(t.arity == 0 for t in terms)

    def test_terminals_by_output_type(
        self, populated_registry: PrimitiveRegistry
    ) -> None:
        terms = populated_registry.terminals(output_type=Series)
        assert len(terms) == 2

    def test_terminals_no_match(self, populated_registry: PrimitiveRegistry) -> None:
        terms = populated_registry.terminals(output_type=Int)
        assert terms == []

    def test_functions(self, populated_registry: PrimitiveRegistry) -> None:
        funcs = populated_registry.functions()
        assert len(funcs) == 3
        assert all(f.arity > 0 for f in funcs)

    def test_functions_by_output_type(
        self, populated_registry: PrimitiveRegistry
    ) -> None:
        funcs = populated_registry.functions(output_type=BoolSeries)
        assert len(funcs) == 1
        assert funcs[0].name == "gt"

    def test_cache_updates_after_registration(self, populated_registry: PrimitiveRegistry) -> None:
        assert len(populated_registry.terminals(output_type=Series)) == 2
        assert len(populated_registry.functions(output_type=Series)) == 2

        populated_registry.register(
            "const",
            lambda x: x,
            input_types=(Series,),
            output_type=Series,
            category="custom",
        )

        # Cache invalidation should make new terminal/function results visible.
        assert len(populated_registry.terminals(output_type=Series)) == 2
        assert len(populated_registry.functions(output_type=Series)) == 3
        names = {p.name for p in populated_registry.functions()}
        assert "const" in names

    def test_get_returns_primitive_info(
        self, populated_registry: PrimitiveRegistry
    ) -> None:
        info = populated_registry.get("add")
        assert isinstance(info, PrimitiveInfo)


# --- error paths ------------------------------------------------------------


class TestErrorPaths:
    """Duplicate and unknown primitive errors (FR-3.6, FR-3.7)."""

    def test_duplicate_registration_raises(self) -> None:
        reg = PrimitiveRegistry()
        reg.register("add", _add, input_types=(Series, Series), output_type=Series)
        with pytest.raises(PrimitiveError, match="already registered"):
            reg.register("add", _add, input_types=(Series, Series), output_type=Series)

    def test_unknown_primitive_raises(self) -> None:
        reg = PrimitiveRegistry()
        with pytest.raises(PrimitiveError, match="not found"):
            reg.get("nonexistent")

    def test_arity_mismatch_raises(self) -> None:
        """arity must equal len(input_types) -- enforced at registration."""
        reg = PrimitiveRegistry()
        # _add expects 2 args but we provide 3 input_types
        with pytest.raises(PrimitiveError, match="arity"):
            reg.register(
                "bad",
                _add,
                input_types=(Series, Series, Series),
                output_type=Series,
                arity=2,
            )


# --- no global mutable state (NFR-3.6) ------------------------------------


class TestNoGlobalState:
    """Registry is an explicit object with no shared state."""

    def test_separate_registries_independent(self) -> None:
        reg1 = PrimitiveRegistry()
        reg2 = PrimitiveRegistry()
        reg1.register("add", _add, input_types=(Series, Series), output_type=Series)
        with pytest.raises(PrimitiveError):
            reg2.get("add")

    def test_registry_is_empty_on_creation(self) -> None:
        reg = PrimitiveRegistry()
        assert reg.list_primitives() == []


class TestPrimitiveInfoRepr:
    """PrimitiveInfo repr."""

    def test_repr(self) -> None:
        info = PrimitiveInfo(
            name="add",
            category="numeric",
            arity=2,
            input_types=(Series, Series),
            output_type=Series,
            callable=_add,
        )
        r = repr(info)
        assert "add" in r
        assert "numeric" in r
        assert "arity=2" in r
