"""Shared metadata for things that can appear as nodes on the canvas."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, ClassVar


@dataclass(frozen=True)
class NodeConfig:
    name: str
    value_type: str
    required: bool = True
    default: Any = None
    description: str = ""


@dataclass(frozen=True)
class NodePort:
    name: str
    description: str
    multiple: bool = False
    dynamic: bool = False


@dataclass(frozen=True)
class NodeInfo:
    type: str
    label: str
    category: str
    description: str
    config: tuple[NodeConfig, ...] = ()
    inputs: tuple[NodePort, ...] = ()
    outputs: tuple[NodePort, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class _NodeRegistration:
    node_class: type[Node]
    info: NodeInfo


class Node:
    """Base class for runtime objects represented by a canvas node.

    Concrete classes own their UI-facing description. The registry lets the
    API expose those descriptions without maintaining a second catalog.
    """

    node_info: ClassVar[NodeInfo | None] = None
    _registry: ClassVar[dict[str, _NodeRegistration]] = {}

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        # Accessing the declared class variable preserves its type. Using
        # cls.__dict__.get(...) would return Any because __dict__ is a generic
        # runtime mapping, and would make that imprecision leak into callers.
        info = cls.node_info
        if info is None:
            return
        if info.type in Node._registry:
            raise TypeError(f"Duplicate node type: {info.type}")
        Node._registry[info.type] = _NodeRegistration(cls, info)

    @classmethod
    def node_types(cls) -> list[NodeInfo]:
        return [registration.info for _, registration in sorted(Node._registry.items())]

    @classmethod
    def info_for(cls, node_type: str) -> NodeInfo:
        try:
            return Node._registry[node_type].info
        except KeyError as error:
            raise KeyError(f"Unknown node type: {node_type}") from error

    @classmethod
    def class_for(cls, node_type: str) -> type[Node]:
        try:
            return Node._registry[node_type].node_class
        except KeyError as error:
            raise KeyError(f"Unknown node type: {node_type}") from error
