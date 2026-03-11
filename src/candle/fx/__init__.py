"""candle.fx -- FX graph intermediate representation.

This package provides a PyTorch-compatible ``torch.fx``-style graph IR
for tracing, transforming, and executing neural-network computations.

Exports are added incrementally as each component is implemented:

- **Node** (Task 1) -- the atomic unit of the graph
- Graph, GraphModule, Interpreter -- added in later tasks
"""

from candle.fx.node import Node

# Graph is available once graph.py is implemented.
try:
    from candle.fx.graph import Graph
except ImportError:
    pass

# GraphModule will be added in a later task.
try:
    from candle.fx.graph_module import GraphModule  # type: ignore[import-not-found]
except ImportError:
    pass

# Interpreter will be added in a later task.
try:
    from candle.fx.interpreter import Interpreter  # type: ignore[import-not-found]
except ImportError:
    pass

__all__ = [
    "Node",
    "Graph",
    "GraphModule",
    "Interpreter",
]
