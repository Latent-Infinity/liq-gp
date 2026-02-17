"""AST node types for GP programs (FR-2).

All nodes are immutable. Mutation and crossover produce new trees.
Type checking is enforced at construction (FR-1.4).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from liq.gp.errors import TypeCheckError

if TYPE_CHECKING:
    from liq.gp.primitives.registry import PrimitiveInfo
    from liq.gp.types import GPType


@dataclass(frozen=True, eq=False)
class TerminalNode:
    """Wraps a zero-arity primitive (reads from evaluation context) (FR-2.2)."""

    name: str
    output_type: GPType

    @property
    def depth(self) -> int:
        return 0

    @property
    def size(self) -> int:
        return 1

    @property
    def constants(self) -> list[ConstantNode]:
        return []

    @property
    def parameters(self) -> dict[str, int | float]:
        return {}

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TerminalNode):
            return NotImplemented
        return self.name == other.name and self.output_type == other.output_type

    def __hash__(self) -> int:
        return hash(("TerminalNode", self.name, self.output_type))


@dataclass(frozen=True, eq=False)
class ConstantNode:
    """Stores a literal float value (ephemeral random constant) (FR-2.2)."""

    value: float
    output_type: GPType = field(default=None)  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.output_type is None:
            from liq.gp.types import Series

            object.__setattr__(self, "output_type", Series)

    @property
    def depth(self) -> int:
        return 0

    @property
    def size(self) -> int:
        return 1

    @property
    def constants(self) -> list[ConstantNode]:
        return [self]

    @property
    def parameters(self) -> dict[str, int | float]:
        return {}

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ConstantNode):
            return NotImplemented
        return self.value == other.value and self.output_type == other.output_type

    def __hash__(self) -> int:
        return hash(("ConstantNode", self.value, self.output_type))


def _check_children_types(
    primitive: PrimitiveInfo,
    children: tuple[Program, ...],
) -> None:
    """Validate child count and types against primitive signature (FR-1.4)."""
    expected_arity = primitive.arity
    actual_arity = len(children)
    if actual_arity != expected_arity:
        msg = (
            f"Primitive {primitive.name!r}: arity mismatch, "
            f"expected {expected_arity} children, got {actual_arity}"
        )
        raise TypeCheckError(msg)

    for i, (child, expected_type) in enumerate(
        zip(children, primitive.input_types, strict=True)
    ):
        if child.output_type != expected_type:
            msg = (
                f"Primitive {primitive.name!r}: type mismatch at child {i}, "
                f"expected {expected_type.name}, got {child.output_type.name}"
            )
            raise TypeCheckError(msg)


@dataclass(frozen=True, eq=False)
class FunctionNode:
    """Wraps a function primitive with typed children (FR-2.2)."""

    primitive: PrimitiveInfo
    children: tuple[Program, ...]
    _depth: int = field(init=False, repr=False)
    _size: int = field(init=False, repr=False)

    def __post_init__(self) -> None:
        _check_children_types(self.primitive, self.children)
        object.__setattr__(self, "_depth", 1 + max(c.depth for c in self.children))
        object.__setattr__(self, "_size", 1 + sum(c.size for c in self.children))

    @property
    def output_type(self) -> GPType:
        return self.primitive.output_type

    @property
    def depth(self) -> int:
        return self._depth

    @property
    def size(self) -> int:
        return self._size

    @property
    def constants(self) -> list[ConstantNode]:
        result: list[ConstantNode] = []
        for c in self.children:
            result.extend(c.constants)
        return result

    @property
    def parameters(self) -> dict[str, int | float]:
        return {}

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, FunctionNode):
            return NotImplemented
        return (
            self.primitive.name == other.primitive.name
            and self.children == other.children
        )

    def __hash__(self) -> int:
        return hash(("FunctionNode", self.primitive.name, self.children))


@dataclass(frozen=True, eq=False)
class ParameterizedNode:
    """Function with children AND evolvable parameters (FR-2.2)."""

    primitive: PrimitiveInfo
    children: tuple[Program, ...]
    params: dict[str, int | float]
    _depth: int = field(init=False, repr=False)
    _size: int = field(init=False, repr=False)

    def __post_init__(self) -> None:
        _check_children_types(self.primitive, self.children)
        object.__setattr__(self, "_depth", 1 + max(c.depth for c in self.children))
        object.__setattr__(self, "_size", 1 + sum(c.size for c in self.children))

    @property
    def output_type(self) -> GPType:
        return self.primitive.output_type

    @property
    def depth(self) -> int:
        return self._depth

    @property
    def size(self) -> int:
        return self._size

    @property
    def constants(self) -> list[ConstantNode]:
        result: list[ConstantNode] = []
        for c in self.children:
            result.extend(c.constants)
        return result

    @property
    def parameters(self) -> dict[str, int | float]:
        return dict(self.params)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ParameterizedNode):
            return NotImplemented
        return (
            self.primitive.name == other.primitive.name
            and self.children == other.children
            and self.params == other.params
        )

    def __hash__(self) -> int:
        return hash(
            (
                "ParameterizedNode",
                self.primitive.name,
                self.children,
                tuple(sorted(self.params.items())),
            )
        )


# Program is the union of all node kinds (FR-2)
Program = TerminalNode | ConstantNode | FunctionNode | ParameterizedNode
