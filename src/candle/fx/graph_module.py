"""candle.fx.graph_module -- GraphModule, an executable Module wrapping a Graph.

A :class:`GraphModule` is a :class:`~candle.nn.Module` whose ``forward``
method is generated from a :class:`~candle.fx.graph.Graph`.  This lets you
construct, transform, and execute FX graphs as regular modules.
"""

from __future__ import annotations

import importlib
from typing import Any, Dict, Optional, TYPE_CHECKING, Union

from candle.nn.module import Module

if TYPE_CHECKING:
    from candle.fx.graph import Graph


class GraphModule(Module):
    """An :class:`~candle.nn.Module` whose ``forward`` is compiled from a
    :class:`~candle.fx.graph.Graph`.

    Parameters
    ----------
    root : Module or dict
        The original module (or a dict of named attributes) from which
        submodules, parameters, and buffers are copied.
    graph : Graph
        The FX graph that defines the computation.
    class_name : str
        Optional name for the generated class (default ``"GraphModule"``).
    """

    def __init__(
        self,
        root: Union[Module, Dict[str, Any]],
        graph: "Graph",
        class_name: str = "GraphModule",
    ) -> None:
        super().__init__()
        self._class_name = class_name

        # Copy state from root
        if isinstance(root, Module):
            # Copy submodules
            for name, mod in root._modules.items():
                self._modules[name] = mod
                # Also set as attribute so that `self.<name>` works in forward
                super(Module, self).__setattr__(name, mod)

            # Copy parameters
            for name, param in root._parameters.items():
                self._parameters[name] = param
                if param is not None:
                    super(Module, self).__setattr__(name, param)

            # Copy buffers
            for name, buf in root._buffers.items():
                self._buffers[name] = buf
                if buf is not None:
                    super(Module, self).__setattr__(name, buf)
        elif isinstance(root, dict):
            for name, value in root.items():
                setattr(self, name, value)

        self._graph = graph
        self.recompile()

    # ------------------------------------------------------------------
    # Recompilation
    # ------------------------------------------------------------------

    def recompile(self) -> None:
        """Regenerate the ``forward`` method from ``self._graph``."""
        code = self._graph.python_code("self")
        # Build the globals dict needed for exec
        globals_dict = self._collect_globals()
        # Compile and exec
        local_ns: Dict[str, Any] = {}
        exec(compile(code, "<graph>", "exec"), globals_dict, local_ns)  # noqa: S102
        forward_fn = local_ns["forward"]
        # Bind as a method on this instance
        self.forward = forward_fn.__get__(self, type(self))

    def _collect_globals(self) -> Dict[str, Any]:
        """Walk the graph and collect modules needed by call_function targets.

        For each ``call_function`` node whose target lives in some module
        (e.g. ``_operator.add``), we import that module and add it to the
        globals dict so that the ``exec``'d code can reference it.
        """
        globals_dict: Dict[str, Any] = {"__builtins__": __builtins__}

        for node in self._graph.nodes:
            if node.op == "call_function":
                target = node.target
                mod_name = getattr(target, "__module__", None)
                if mod_name and mod_name != "builtins":
                    # Import the module and add it to globals
                    try:
                        mod = importlib.import_module(mod_name)
                        globals_dict[mod_name] = mod
                    except ImportError:
                        pass

        return globals_dict

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def graph(self) -> "Graph":
        """The underlying :class:`Graph`."""
        return self._graph

    @graph.setter
    def graph(self, g: "Graph") -> None:
        self._graph = g
        self.recompile()

    @property
    def code(self) -> str:
        """The generated Python source code for ``forward``."""
        return self._graph.python_code("self")

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def print_readable(self, print_output: bool = True) -> str:
        """Print or return the generated forward code.

        Parameters
        ----------
        print_output : bool
            If True, print the code to stdout and return it.
            If False, just return the code string.
        """
        code = self.code
        if print_output:
            print(code)
        return code

    def __str__(self) -> str:
        graph_str = str(self._graph)
        return f"GraphModule(\n  {graph_str}\n)"

    def _get_name(self) -> str:
        return self._class_name
