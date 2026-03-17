"""Pure-Python fallback for _autograd_node.pyx — FastNode."""


class FastNode:
    """Pure-Python mirror of Cython FastNode.

    Provides the same typed-slot layout so Node can inherit from it
    unconditionally.  When Cython is available the .pyx version replaces
    this module.
    """
    __slots__ = (
        "backward", "inputs", "_saved_tensors_list", "_saved_fields",
        "_hooks", "_prehooks", "_next_functions_cache", "_metadata",
        "_name", "_anomaly_trace", "_anomaly_parent", "__dict__",
    )
