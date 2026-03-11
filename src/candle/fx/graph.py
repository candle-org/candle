"""candle.fx.graph -- Minimal Graph class for the FX IR.

A Graph owns a doubly-linked list of :class:`~candle.fx.node.Node` objects
and provides convenience methods for building the IR programmatically.

This is a *minimal* implementation: just enough to support Node creation,
name uniquification, and linked-list management.  Full graph utilities
(e.g., ``output``, ``get_attr``, ``print_tabular``) will be added later.
"""

from __future__ import annotations

import re
from typing import Any, Callable, Dict, Iterator, Optional, Tuple

from candle.fx.node import Node


class _NodeList:
    """Sentinel-based circular doubly-linked list for Nodes."""

    def __init__(self) -> None:
        # The sentinel is a plain object; it is never exposed externally.
        self._root = _Sentinel()

    def append(self, node: Node) -> None:
        """Insert *node* at the end of the list (before the sentinel)."""
        last = self._root.prev
        node._prev = last
        node._next = self._root
        last._next = node  # type: ignore[union-attr]
        self._root.prev = node

    def __iter__(self) -> Iterator[Node]:
        cur = self._root.next
        while cur is not self._root:
            assert isinstance(cur, Node)
            yield cur
            cur = cur._next

    def __len__(self) -> int:
        return sum(1 for _ in self)


class _Sentinel:
    """Sentinel node for the circular doubly-linked list.

    Has ``_prev`` and ``_next`` attributes but is *not* a real Node.
    """

    def __init__(self) -> None:
        self._prev: Any = self
        self._next: Any = self

    @property
    def prev(self) -> Any:
        return self._prev

    @prev.setter
    def prev(self, value: Any) -> None:
        self._prev = value

    @property
    def next(self) -> Any:
        return self._next

    @next.setter
    def next(self, value: Any) -> None:
        self._next = value


class Graph:
    """A minimal FX Graph: owns Nodes in a doubly-linked list.

    >>> g = Graph()
    >>> x = g.placeholder("x")
    >>> y = g.placeholder("y")
    >>> add = g.call_function(operator.add, (x, y))
    """

    def __init__(self) -> None:
        self._node_list = _NodeList()
        self._used_names: Dict[str, int] = {}

    # ------------------------------------------------------------------
    # Name uniquification
    # ------------------------------------------------------------------

    def _unique_name(self, candidate: str) -> str:
        """Return a name based on *candidate* that is unique within this graph.

        On collision, appends ``_1``, ``_2``, etc.
        """
        # Sanitize: replace non-identifier characters with underscore.
        candidate = re.sub(r"[^a-zA-Z0-9_]", "_", candidate)
        if not candidate:
            candidate = "_"

        if candidate not in self._used_names:
            self._used_names[candidate] = 0
            return candidate

        self._used_names[candidate] += 1
        new_name = f"{candidate}_{self._used_names[candidate]}"
        # Recursively ensure the suffixed name is also unique.
        while new_name in self._used_names:
            self._used_names[candidate] += 1
            new_name = f"{candidate}_{self._used_names[candidate]}"
        self._used_names[new_name] = 0
        return new_name

    # ------------------------------------------------------------------
    # Node creation
    # ------------------------------------------------------------------

    def create_node(
        self,
        op: str,
        target: Any,
        args: Optional[Tuple[Any, ...]] = None,
        kwargs: Optional[Dict[str, Any]] = None,
        name: Optional[str] = None,
    ) -> Node:
        """Create a new Node and append it to the graph.

        Parameters
        ----------
        op : str
            Operation type (one of :data:`Node._VALID_OPS`).
        target : Any
            The callable / attribute path / module name.
        args : tuple, optional
            Positional arguments (may reference other Nodes).
        kwargs : dict, optional
            Keyword arguments (may reference other Nodes).
        name : str, optional
            Desired name.  Will be uniquified if it collides.

        Returns
        -------
        Node
            The newly created node (already inserted into the graph).
        """
        if args is None:
            args = ()
        if kwargs is None:
            kwargs = {}
        if name is None:
            # Derive a default name from the target.
            if isinstance(target, str):
                name = target
            elif callable(target):
                name = getattr(target, "__name__", "_")
            else:
                name = "_"
        name = self._unique_name(name)
        node = Node(self, name, op, target, args, kwargs)
        self._insert_node(node)
        return node

    def _insert_node(self, node: Node) -> None:
        """Insert *node* at the tail of the linked list."""
        # Walk to get the last real Node (or sentinel) and hook up prev/next.
        sentinel = self._node_list._root
        last = sentinel.prev

        node._prev = last if not isinstance(last, _Sentinel) else None
        node._next = None

        if isinstance(last, _Sentinel):
            # First real node.
            sentinel._next = node
        else:
            last._next = node

        sentinel._prev = node

    # ------------------------------------------------------------------
    # Convenience constructors
    # ------------------------------------------------------------------

    def placeholder(self, name: str) -> Node:
        """Create a ``placeholder`` node (graph input)."""
        return self.create_node("placeholder", name, name=name)

    def call_function(
        self,
        fn: Callable[..., Any],
        args: Optional[Tuple[Any, ...]] = None,
        kwargs: Optional[Dict[str, Any]] = None,
    ) -> Node:
        """Create a ``call_function`` node."""
        return self.create_node("call_function", fn, args=args, kwargs=kwargs)

    # ------------------------------------------------------------------
    # Iteration
    # ------------------------------------------------------------------

    @property
    def nodes(self) -> Iterator[Node]:
        """Iterate over all Nodes in linked-list order."""
        sentinel = self._node_list._root
        cur = sentinel._next
        while cur is not sentinel and cur is not None:
            assert isinstance(cur, Node)
            yield cur
            cur = cur._next
