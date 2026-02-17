"""Primitive registry for GP nodes (FR-3)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from liq.gp.errors import PrimitiveError
from liq.gp.types import GPType, ParamSpec


class PrimitiveInfo:
    """Metadata for a registered primitive (FR-3.2).

    Attributes:
        name: Unique primitive name.
        category: Grouping label (e.g. ``"arithmetic"``, ``"terminal"``).
        arity: Number of child inputs (0 for terminals).
        input_types: Tuple of :class:`~liq.gp.GPType` for each child.
        output_type: The :class:`~liq.gp.GPType` this primitive produces.
        callable: The function invoked during evaluation.
        param_specs: Evolvable parameter specs (empty for non-parameterized).
    """

    __slots__ = (
        "name",
        "category",
        "arity",
        "input_types",
        "output_type",
        "callable",
        "param_specs",
    )

    def __init__(
        self,
        *,
        name: str,
        category: str,
        arity: int,
        input_types: tuple[GPType, ...],
        output_type: GPType,
        callable: Callable[..., Any],
        param_specs: list[ParamSpec] | None = None,
    ) -> None:
        self.name = name
        self.category = category
        self.arity = arity
        self.input_types = input_types
        self.output_type = output_type
        self.callable = callable
        self.param_specs = param_specs or []

    def __repr__(self) -> str:
        return (
            f"PrimitiveInfo(name={self.name!r}, category={self.category!r}, "
            f"arity={self.arity})"
        )


class PrimitiveRegistry:
    """Registry for GP primitives (FR-3.1).

    Each instance is an independent registry with no shared global state
    (NFR-3.6).
    """

    def __init__(self) -> None:
        self._primitives: dict[str, PrimitiveInfo] = {}
        self._terminals_cache: dict[GPType | None, list[PrimitiveInfo]] = {}
        self._functions_cache: dict[GPType | None, list[PrimitiveInfo]] = {}

    def register(
        self,
        name: str,
        callable: Callable[..., Any],
        *,
        category: str = "default",
        input_types: tuple[GPType, ...] = (),
        output_type: GPType | None = None,
        param_specs: list[ParamSpec] | None = None,
        arity: int | None = None,
    ) -> None:
        """Register a primitive (FR-3.4 direct call).

        Args:
            name: Unique primitive name.
            callable: The function to invoke during evaluation.
            category: Grouping label.
            input_types: Type of each child input.
            output_type: Type this primitive produces.
            param_specs: Evolvable parameters (empty for non-parameterized).
            arity: Explicit arity override. If given, must match len(input_types).

        Raises:
            PrimitiveError: On duplicate name or arity mismatch.
        """
        if name in self._primitives:
            msg = f"Primitive {name!r} is already registered"
            raise PrimitiveError(msg)

        computed_arity = len(input_types)
        if arity is not None and arity != computed_arity:
            msg = (
                f"Primitive {name!r}: explicit arity={arity} does not match "
                f"len(input_types)={computed_arity}"
            )
            raise PrimitiveError(msg)

        if output_type is None:
            from liq.gp.types import Series

            output_type = Series

        self._primitives[name] = PrimitiveInfo(
            name=name,
            category=category,
            arity=computed_arity,
            input_types=input_types,
            output_type=output_type,
            callable=callable,
            param_specs=param_specs,
        )
        self._terminals_cache = {}
        self._functions_cache = {}

    def primitive(
        self,
        name: str,
        *,
        category: str = "default",
        input_types: tuple[GPType, ...] = (),
        output_type: GPType | None = None,
        param_specs: list[ParamSpec] | None = None,
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Decorator registration (FR-3.4).

        Usage::

            @registry.primitive("neg", input_types=(Series,), output_type=Series)
            def neg_fn(a):
                return -a
        """

        def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
            self.register(
                name,
                fn,
                category=category,
                input_types=input_types,
                output_type=output_type,
                param_specs=param_specs,
            )
            return fn

        return decorator

    def register_from_metadata(self, metadata_list: list[dict[str, Any]]) -> None:
        """Bulk registration from metadata dicts (FR-3.4).

        Each dict must have at minimum ``name`` and ``callable`` keys.
        Other keys map to ``register()`` keyword arguments.
        """
        for entry in metadata_list:
            self.register(
                name=entry["name"],
                callable=entry["callable"],
                category=entry.get("category", "default"),
                input_types=entry.get("input_types", ()),
                output_type=entry.get("output_type"),
                param_specs=entry.get("param_specs"),
            )

    def get(self, name: str) -> PrimitiveInfo:
        """Look up a primitive by name (FR-3.5).

        Raises:
            PrimitiveError: If the primitive is not found.
        """
        try:
            return self._primitives[name]
        except KeyError:
            msg = f"Primitive {name!r} not found in registry"
            raise PrimitiveError(msg) from None

    def list_primitives(self, category: str | None = None) -> list[PrimitiveInfo]:
        """List registered primitives, optionally filtered by category (FR-3.5)."""
        prims = list(self._primitives.values())
        if category is not None:
            prims = [p for p in prims if p.category == category]
        return prims

    def terminals(self, output_type: GPType | None = None) -> list[PrimitiveInfo]:
        """List terminal primitives (arity 0) (FR-3.5)."""
        if output_type in self._terminals_cache:
            return list(self._terminals_cache[output_type])

        if output_type is None:
            result = [p for p in self._primitives.values() if p.arity == 0]
        else:
            result = [
                p
                for p in self._primitives.values()
                if p.arity == 0 and p.output_type == output_type
            ]
        self._terminals_cache[output_type] = result
        return list(result)

    def functions(self, output_type: GPType | None = None) -> list[PrimitiveInfo]:
        """List function primitives (arity > 0) (FR-3.5)."""
        if output_type in self._functions_cache:
            return list(self._functions_cache[output_type])

        if output_type is None:
            result = [p for p in self._primitives.values() if p.arity > 0]
        else:
            result = [
                p
                for p in self._primitives.values()
                if p.arity > 0 and p.output_type == output_type
            ]
        self._functions_cache[output_type] = result
        return list(result)
