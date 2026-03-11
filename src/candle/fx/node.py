"""candle.fx.node -- Node class, the atomic unit of the FX graph IR.

Each Node represents a single operation in the graph.  Nodes are organized
in a circular doubly-linked list owned by a Graph, and they track their
producer/consumer relationships (``args`` / ``kwargs`` / ``users``) automatically.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Set, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from candle.fx.graph import Graph


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _find_nodes_in_arg(arg: Any) -> list["Node"]:
    """Recursively find all Node references in an argument tree."""
    nodes: list[Node] = []
    if isinstance(arg, Node):
        nodes.append(arg)
    elif isinstance(arg, (tuple, list)):
        for item in arg:
            nodes.extend(_find_nodes_in_arg(item))
    elif isinstance(arg, dict):
        for value in arg.values():
            nodes.extend(_find_nodes_in_arg(value))
    return nodes


def _map_arg(arg: Any, fn: Callable[["Node"], Any]) -> Any:
    """Recursively apply *fn* to each Node in *arg*, preserving structure.

    - Node        -> fn(node)
    - tuple/list  -> same container type with elements mapped
    - dict        -> dict with values mapped
    - other       -> unchanged
    """
    if isinstance(arg, Node):
        return fn(arg)
    if isinstance(arg, tuple):
        return tuple(_map_arg(item, fn) for item in arg)
    if isinstance(arg, list):
        return [_map_arg(item, fn) for item in arg]
    if isinstance(arg, dict):
        return {k: _map_arg(v, fn) for k, v in arg.items()}
    return arg


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

class Node:
    """A single operation in the FX graph.

    Attributes:
        graph:       The owning Graph.
        name:        Unique name within the graph (also used for ``__repr__``).
        op:          One of :data:`_VALID_OPS`.
        target:      The callable / attribute / module path this node refers to.
        args:        Positional arguments (may contain Nodes).
        kwargs:      Keyword arguments (may contain Nodes).
        meta:        Arbitrary metadata dict.
        return_type: Optional return-type annotation.
    """

    _VALID_OPS = frozenset([
        "placeholder",
        "get_attr",
        "call_function",
        "call_method",
        "call_module",
        "output",
    ])

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        graph: "Graph",
        name: str,
        op: str,
        target: Any,
        args: Tuple[Any, ...] = (),
        kwargs: Optional[Dict[str, Any]] = None,
        return_type: Optional[type] = None,
    ) -> None:
        assert op in self._VALID_OPS, f"Invalid op {op!r}; must be one of {self._VALID_OPS}"
        self.graph = graph
        self.name = name
        self.op = op
        self.target = target
        self.return_type = return_type
        self.meta: Dict[str, Any] = {}

        # Users: set of Nodes that consume this Node as an input.
        self._users: Set[Node] = set()

        # Linked-list pointers (managed by Graph).
        self._prev: Optional[Node] = None
        self._next: Optional[Node] = None

        # Set args/kwargs through properties so user tracking is updated.
        self._args: Tuple[Any, ...] = ()
        self._kwargs: Dict[str, Any] = {}
        self.args = args
        self.kwargs = kwargs if kwargs is not None else {}

    # ------------------------------------------------------------------
    # args / kwargs properties  (auto-update user tracking)
    # ------------------------------------------------------------------

    @property
    def args(self) -> Tuple[Any, ...]:
        return self._args

    @args.setter
    def args(self, new_args: Tuple[Any, ...]) -> None:
        # Remove self from old input nodes' user sets.
        for node in _find_nodes_in_arg(self._args):
            node._users.discard(self)
        self._args = new_args
        # Add self to new input nodes' user sets.
        for node in _find_nodes_in_arg(self._args):
            node._users.add(self)

    @property
    def kwargs(self) -> Dict[str, Any]:
        return self._kwargs

    @kwargs.setter
    def kwargs(self, new_kwargs: Dict[str, Any]) -> None:
        for node in _find_nodes_in_arg(self._kwargs):
            node._users.discard(self)
        self._kwargs = new_kwargs
        for node in _find_nodes_in_arg(self._kwargs):
            node._users.add(self)

    # ------------------------------------------------------------------
    # Read-only derived properties
    # ------------------------------------------------------------------

    @property
    def users(self) -> Set["Node"]:
        """Nodes that consume this node as an input (read-only view)."""
        return self._users

    @property
    def next(self) -> Optional["Node"]:
        return self._next

    @property
    def prev(self) -> Optional["Node"]:
        return self._prev

    @property
    def all_input_nodes(self) -> list["Node"]:
        """Unique Nodes referenced in args and kwargs, in discovery order."""
        seen: set[int] = set()
        result: list[Node] = []
        for node in _find_nodes_in_arg(self._args) + _find_nodes_in_arg(self._kwargs):
            node_id = id(node)
            if node_id not in seen:
                seen.add(node_id)
                result.append(node)
        return result

    # ------------------------------------------------------------------
    # Mutation helpers
    # ------------------------------------------------------------------

    def update_arg(self, idx: int, arg: Any) -> None:
        """Replace a single positional argument by index."""
        args_list = list(self._args)
        args_list[idx] = arg
        self.args = tuple(args_list)

    def update_kwarg(self, key: str, arg: Any) -> None:
        """Replace a single keyword argument by key."""
        new_kwargs = dict(self._kwargs)
        new_kwargs[key] = arg
        self.kwargs = new_kwargs

    def replace_all_uses_with(self, replace_with: "Node") -> None:
        """In every consumer of *self*, replace references to *self* with *replace_with*."""
        # Snapshot users because the set mutates during iteration.
        for user in list(self._users):
            def _replace(n: "Node") -> "Node":
                return replace_with if n is self else n
            user.args = _map_arg(user.args, _replace)
            user.kwargs = _map_arg(user.kwargs, _replace)

    def replace_input_with(self, old_input: "Node", new_input: "Node") -> None:
        """Replace *old_input* with *new_input* in this node's args and kwargs."""
        def _replace(n: "Node") -> "Node":
            return new_input if n is old_input else n
        self.args = _map_arg(self.args, _replace)
        self.kwargs = _map_arg(self.kwargs, _replace)

    # ------------------------------------------------------------------
    # Linked-list reordering
    # ------------------------------------------------------------------

    def _remove_from_list(self) -> None:
        """Unlink this node from the linked list."""
        if self._prev is not None:
            self._prev._next = self._next
        if self._next is not None:
            self._next._prev = self._prev
        self._prev = None
        self._next = None

    def prepend(self, x: "Node") -> None:
        """Move *x* to immediately before *self* in the linked list."""
        assert x is not self, "Cannot prepend a node before itself"
        x._remove_from_list()
        x._prev = self._prev
        x._next = self
        if self._prev is not None:
            self._prev._next = x
        self._prev = x

    def append(self, x: "Node") -> None:
        """Move *x* to immediately after *self* in the linked list."""
        assert x is not self, "Cannot append a node after itself"
        x._remove_from_list()
        x._prev = self
        x._next = self._next
        if self._next is not None:
            self._next._prev = x
        self._next = x

    # ------------------------------------------------------------------
    # Formatting
    # ------------------------------------------------------------------

    def format_node(self) -> str:
        """Return a human-readable formatted string for this node."""
        args_str = ", ".join(repr(a) for a in self._args)
        kwargs_str = ", ".join(f"{k}={v!r}" for k, v in self._kwargs.items())
        all_args = ", ".join(filter(None, [args_str, kwargs_str]))
        return f"%{self.name} : [{self.op}] = {self.target}({all_args})"

    def __repr__(self) -> str:
        return self.name

    def __str__(self) -> str:
        return self.name

    # ------------------------------------------------------------------
    # Identity-based equality
    # ------------------------------------------------------------------

    def __eq__(self, other: object) -> bool:
        return self is other

    def __hash__(self) -> int:
        return id(self)
