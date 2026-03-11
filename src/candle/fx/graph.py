"""candle.fx.graph -- Graph class for the FX IR.

A Graph owns a doubly-linked list of :class:`~candle.fx.node.Node` objects
and provides convenience methods for building the IR programmatically.
"""

from __future__ import annotations

import contextlib
import re
from typing import Any, Callable, Dict, Iterator, Optional, Tuple

from candle.fx.node import Node, _map_arg, _find_nodes_in_arg


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


class _NodeList:
    """Sentinel-based circular doubly-linked list for Nodes.

    Supports iteration, reverse iteration, length, and membership testing.
    """

    def __init__(self) -> None:
        # The sentinel is a plain object; it is never exposed externally.
        self._root = _Sentinel()

    def _append_node(self, node: Node) -> None:
        """Insert *node* at the end of the list (before the sentinel)."""
        last = self._root.prev
        node._prev = last
        node._next = self._root
        if isinstance(last, _Sentinel):
            last._next = node
        else:
            last._next = node  # type: ignore[union-attr]
        self._root.prev = node

    def _insert_before(self, node: Node, before: Node) -> None:
        """Insert *node* immediately before *before*."""
        node._prev = before._prev
        node._next = before
        if before._prev is not None:
            before._prev._next = node
        elif isinstance(self._root.next, Node) and self._root.next is before:
            # before is the first real node
            self._root._next = node
        before._prev = node

    def _insert_after(self, node: Node, after: Node) -> None:
        """Insert *node* immediately after *after*."""
        node._prev = after
        node._next = after._next
        if after._next is not None:
            after._next._prev = node
        else:
            # after is the last real node
            self._root.prev = node
        after._next = node

    def _remove(self, node: Node) -> None:
        """Remove *node* from the linked list."""
        prev_node = node._prev
        next_node = node._next
        if prev_node is not None:
            prev_node._next = next_node
        elif isinstance(self._root.next, Node):
            # node was the first real node
            if isinstance(next_node, _Sentinel):
                self._root._next = self._root
            else:
                self._root._next = next_node

        if isinstance(next_node, _Sentinel):
            self._root.prev = prev_node if prev_node is not None else self._root
        elif next_node is not None:
            next_node._prev = prev_node

        node._prev = None
        node._next = None

    def __iter__(self) -> Iterator[Node]:
        cur = self._root.next
        while not isinstance(cur, _Sentinel):
            assert isinstance(cur, Node)
            yield cur
            cur = cur._next

    def __reversed__(self) -> Iterator[Node]:
        cur = self._root.prev
        while not isinstance(cur, _Sentinel):
            assert isinstance(cur, Node)
            yield cur
            cur = cur._prev

    def __len__(self) -> int:
        count = 0
        for _ in self:
            count += 1
        return count

    def __contains__(self, item: object) -> bool:
        for node in self:
            if node is item:
                return True
        return False


class _InsertPointManager:
    """Context manager for temporarily changing the graph's insert point."""

    def __init__(self, graph: "Graph", new_insert_point: Optional[Node], insert_after: bool) -> None:
        self._graph = graph
        self._new_insert_point = new_insert_point
        self._insert_after = insert_after
        self._old_insert_point: Optional[Node] = None
        self._old_insert_after: bool = False

    def __enter__(self) -> "_InsertPointManager":
        self._old_insert_point = self._graph._insert_point
        self._old_insert_after = self._graph._insert_after
        self._graph._insert_point = self._new_insert_point
        self._graph._insert_after = self._insert_after
        return self

    def __exit__(self, *args: Any) -> None:
        self._graph._insert_point = self._old_insert_point
        self._graph._insert_after = self._old_insert_after


class Graph:
    """An FX Graph: owns Nodes in a doubly-linked list.

    >>> g = Graph()
    >>> x = g.placeholder("x")
    >>> y = g.placeholder("y")
    >>> add = g.call_function(operator.add, (x, y))
    """

    def __init__(self) -> None:
        self._node_list = _NodeList()
        self._used_names: Dict[str, int] = {}
        self._insert_point: Optional[Node] = None
        self._insert_after: bool = False

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
        type_expr: Optional[type] = None,
    ) -> Node:
        """Create a new Node and insert it into the graph.

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
        type_expr : type, optional
            Optional return type annotation.

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
        node = Node(self, name, op, target, args, kwargs, return_type=type_expr)
        self._insert(node)
        return node

    def _insert(self, node: Node) -> None:
        """Insert *node* at the current insert point (or append to end)."""
        if self._insert_point is not None:
            if self._insert_after:
                self._node_list._insert_after(node, self._insert_point)
            else:
                self._node_list._insert_before(node, self._insert_point)
        else:
            self._node_list._append_node(node)

    # ------------------------------------------------------------------
    # Convenience constructors
    # ------------------------------------------------------------------

    def placeholder(self, name: str, type_expr: Optional[type] = None) -> Node:
        """Create a ``placeholder`` node (graph input)."""
        return self.create_node("placeholder", name, name=name, type_expr=type_expr)

    def get_attr(self, qualified_name: str, type_expr: Optional[type] = None) -> Node:
        """Create a ``get_attr`` node (fetch attribute from module)."""
        # Replace dots with underscores for the default name.
        name = qualified_name.replace(".", "_")
        return self.create_node("get_attr", qualified_name, name=name, type_expr=type_expr)

    def call_function(
        self,
        fn: Callable[..., Any],
        args: Optional[Tuple[Any, ...]] = None,
        kwargs: Optional[Dict[str, Any]] = None,
        type_expr: Optional[type] = None,
    ) -> Node:
        """Create a ``call_function`` node (call a free function)."""
        return self.create_node("call_function", fn, args=args, kwargs=kwargs, type_expr=type_expr)

    def call_method(
        self,
        method_name: str,
        args: Optional[Tuple[Any, ...]] = None,
        kwargs: Optional[Dict[str, Any]] = None,
        type_expr: Optional[type] = None,
    ) -> Node:
        """Create a ``call_method`` node (call a method on first arg)."""
        return self.create_node("call_method", method_name, args=args, kwargs=kwargs, type_expr=type_expr)

    def call_module(
        self,
        module_name: str,
        args: Optional[Tuple[Any, ...]] = None,
        kwargs: Optional[Dict[str, Any]] = None,
        type_expr: Optional[type] = None,
    ) -> Node:
        """Create a ``call_module`` node (invoke a submodule)."""
        # Replace dots with underscores for the default name.
        name = module_name.replace(".", "_")
        return self.create_node("call_module", module_name, args=args, kwargs=kwargs, name=name, type_expr=type_expr)

    def output(self, result: Any, type_expr: Optional[type] = None) -> Node:
        """Create an ``output`` node (graph return value)."""
        return self.create_node("output", "output", args=(result,), type_expr=type_expr)

    # ------------------------------------------------------------------
    # Node manipulation
    # ------------------------------------------------------------------

    def erase_node(self, to_erase: Node) -> None:
        """Remove *to_erase* from the graph.

        Raises
        ------
        RuntimeError
            If the node still has users (consumers).
        """
        if to_erase.users:
            raise RuntimeError(
                f"Node '{to_erase.name}' still has {len(to_erase.users)} user(s) "
                f"and cannot be erased. Replace all uses first."
            )
        # Remove from inputs' user sets
        for input_node in to_erase.all_input_nodes:
            input_node._users.discard(to_erase)
        # Remove from linked list
        self._node_list._remove(to_erase)

    def eliminate_dead_code(self) -> bool:
        """Remove nodes with zero users (except placeholders and output).

        Returns
        -------
        bool
            True if any nodes were removed, False otherwise.
        """
        changed = False
        # Iterate in reverse order to avoid issues with removing nodes
        # that feed into other dead nodes.
        for node in list(reversed(self.nodes)):
            if node.op in ("placeholder", "output"):
                continue
            if not node.users:
                self.erase_node(node)
                changed = True
        return changed

    # ------------------------------------------------------------------
    # Insert point management
    # ------------------------------------------------------------------

    @contextlib.contextmanager
    def inserting_before(self, n: Optional[Node]) -> Iterator[None]:
        """Context manager to insert nodes before *n*."""
        manager = _InsertPointManager(self, n, insert_after=False)
        yield manager.__enter__()
        manager.__exit__(None, None, None)

    @contextlib.contextmanager
    def inserting_after(self, n: Optional[Node]) -> Iterator[None]:
        """Context manager to insert nodes after *n*."""
        manager = _InsertPointManager(self, n, insert_after=True)
        yield manager.__enter__()
        manager.__exit__(None, None, None)

    # ------------------------------------------------------------------
    # Copying
    # ------------------------------------------------------------------

    def node_copy(
        self,
        node: Node,
        arg_transform: Callable[[Node], Any] = lambda x: x,
    ) -> Node:
        """Copy a single node into this graph.

        Parameters
        ----------
        node : Node
            Node to copy from another graph.
        arg_transform : callable
            Function to transform Node references in args/kwargs.

        Returns
        -------
        Node
            The newly created node in this graph.
        """
        new_args = _map_arg(node.args, arg_transform)
        new_kwargs = _map_arg(node.kwargs, arg_transform)
        return self.create_node(
            op=node.op,
            target=node.target,
            args=new_args if isinstance(new_args, tuple) else tuple(new_args) if isinstance(new_args, list) else (new_args,),
            kwargs=new_kwargs if isinstance(new_kwargs, dict) else {},
            name=node.name,
            type_expr=node.return_type,
        )

    def graph_copy(self, g: "Graph", val_map: Optional[Dict[Node, Node]] = None) -> Dict[Node, Node]:
        """Copy all nodes from another graph into this graph.

        Parameters
        ----------
        g : Graph
            Source graph to copy from.
        val_map : dict, optional
            Dictionary to populate with old_node -> new_node mappings.
            If None, a new dict is created.

        Returns
        -------
        dict
            Mapping from source nodes to newly created nodes.
        """
        if val_map is None:
            val_map = {}

        for node in g.nodes:
            new_node = self.node_copy(node, lambda n: val_map.get(n, n))
            val_map[node] = new_node

        return val_map

    # ------------------------------------------------------------------
    # Code generation
    # ------------------------------------------------------------------

    def python_code(self, root_module: str = "self") -> str:
        """Generate Python source code for this graph.

        Parameters
        ----------
        root_module : str
            The name of the root module (default: "self").

        Returns
        -------
        str
            Python source code implementing the graph's forward function.
        """
        # Collect placeholder names for function signature
        placeholders = []
        body_lines = []

        for node in self.nodes:
            if node.op == "placeholder":
                placeholders.append(node.name)
                # No body line for placeholder
            elif node.op == "get_attr":
                body_lines.append(f"    {node.name} = {root_module}.{node.target}")
            elif node.op == "call_function":
                args_str = self._format_args(node.args, node.kwargs)
                target_name = self._get_qualified_name(node.target)
                body_lines.append(f"    {node.name} = {target_name}({args_str})")
            elif node.op == "call_method":
                args_str = self._format_args(node.args[1:], node.kwargs) if len(node.args) > 1 else self._format_args((), node.kwargs)
                self_arg = self._format_arg(node.args[0]) if node.args else ""
                if args_str:
                    body_lines.append(f"    {node.name} = {self_arg}.{node.target}({args_str})")
                else:
                    body_lines.append(f"    {node.name} = {self_arg}.{node.target}()")
            elif node.op == "call_module":
                args_str = self._format_args(node.args, node.kwargs)
                body_lines.append(f"    {node.name} = {root_module}.{node.target}({args_str})")
            elif node.op == "output":
                result = self._format_arg(node.args[0]) if node.args else "None"
                body_lines.append(f"    return {result}")

        # Build the function
        params = ", ".join(["self"] + placeholders)
        lines = [f"def forward({params}):"]
        lines.extend(body_lines)

        return "\n".join(lines)

    def _format_arg(self, arg: Any) -> str:
        """Format a single argument for code generation."""
        if isinstance(arg, Node):
            return arg.name
        elif isinstance(arg, tuple):
            items = ", ".join(self._format_arg(a) for a in arg)
            if len(arg) == 1:
                return f"({items},)"
            return f"({items})"
        elif isinstance(arg, list):
            items = ", ".join(self._format_arg(a) for a in arg)
            return f"[{items}]"
        elif isinstance(arg, dict):
            items = ", ".join(f"{k}: {self._format_arg(v)}" for k, v in arg.items())
            return f"{{{items}}}"
        else:
            return repr(arg)

    def _format_args(self, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> str:
        """Format positional and keyword arguments for code generation."""
        parts = [self._format_arg(a) for a in args]
        parts.extend(f"{k}={self._format_arg(v)}" for k, v in kwargs.items())
        return ", ".join(parts)

    def _get_qualified_name(self, target: Any) -> str:
        """Get a qualified name for a callable target."""
        if hasattr(target, "__module__") and target.__module__:
            module = target.__module__
            name = getattr(target, "__name__", str(target))
            # Don't qualify builtins
            if module == "builtins":
                return name
            return f"{module}.{name}"
        return getattr(target, "__name__", str(target))

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def nodes(self) -> _NodeList:
        """Return the node list (supports iteration, len, reversed, contains)."""
        return self._node_list

    def __str__(self) -> str:
        """Return a formatted representation of all nodes."""
        lines = []
        for node in self.nodes:
            lines.append(node.format_node())
        return "\n".join(lines)

    def __repr__(self) -> str:
        node_count = len(self._node_list)
        return f"<Graph with {node_count} nodes>"
