# Autograd Node Virtual Base Class Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Expose `candle.autograd.graph.Node` as a virtual base class that passes `isinstance/issubclass` checks without appearing in the concrete MRO, matching PyTorch semantics.

**Architecture:** Add a module-level `__getattr__` in `src/candle/autograd/graph.py` that lazily returns a `_VirtualNode` class (using `ABCMeta`) and custom `__instancecheck__`/`__subclasscheck__` implementations. Provide a contract test that asserts `isinstance` and `issubclass` on a real autograd node.

**Tech Stack:** Python, Candle autograd graph module, pytest.

---

### Task 1: Add Failing Contract Test For Node Semantics

**Files:**
- Create: `tests/contract/test_autograd_graph_node.py`

**Step 1: Write the failing test**

```python
import candle as torch
from candle.autograd.graph import Node


def test_autograd_graph_node_virtual_base():
    x = torch.randn(2, 2, requires_grad=True)
    y = (x * x).sum()
    assert isinstance(y.grad_fn, Node)
    assert issubclass(type(y.grad_fn), Node)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/contract/test_autograd_graph_node.py::test_autograd_graph_node_virtual_base -v --tb=short`
Expected: FAIL with `ImportError` / `AttributeError` or `AssertionError` for Node checks.

---

### Task 2: Implement Virtual Node In autograd.graph

**Files:**
- Modify: `src/candle/autograd/graph.py`

**Step 1: Implement minimal code**

```python
__all__ = ["saved_tensors_hooks", "GradientEdge", "get_gradient_edge"]


def __getattr__(name):
    if name == "Node":
        from abc import ABCMeta

        class _VirtualNode(metaclass=ABCMeta):
            @classmethod
            def __instancecheck__(cls, instance):
                return hasattr(instance, "next_functions") and hasattr(instance, "name")

            @classmethod
            def __subclasscheck__(cls, subclass):
                return hasattr(subclass, "__mro__") and any(
                    hasattr(base, "next_functions") and hasattr(base, "name")
                    for base in subclass.__mro__
                )

        return _VirtualNode
    raise AttributeError(f"module {__name__} has no attribute {name}")


__all__.append("Node")
```

**Step 2: Run test to verify it passes**

Run: `pytest tests/contract/test_autograd_graph_node.py::test_autograd_graph_node_virtual_base -v --tb=short`
Expected: PASS

---

### Task 3: Contract Suite Gate

**Files:**
- Test: `tests/contract/`

**Step 1: Run contract tests**

Run: `pytest tests/contract/ -v --tb=short`
Expected: PASS

---

### Task 4: Pylint Gate

**Files:**
- Lint: `src/candle/`

**Step 1: Run pylint**

Run: `pylint src/candle/ --rcfile=pyproject.toml`
Expected: PASS (if it crashes due to environment, capture the traceback for PR notes)

---

### Task 5: Commit And PR

**Step 1: Commit**

```bash
git add tests/contract/test_autograd_graph_node.py src/candle/autograd/graph.py
git commit -m "fix: add virtual autograd graph Node"
```

**Step 2: Rebase upstream**

```bash
git fetch upstream main
git rebase upstream/main
```

**Step 3: Push**

```bash
git push -u origin compat/autograd-pr2
```

**Step 4: Create PR**

```bash
gh pr create \
  -R candle-org/candle \
  --base main \
  --head lvyufeng:compat/autograd-pr2 \
  --title "fix: add virtual autograd graph Node" \
  --body-file /tmp/pr_body_pr2.md
```

**Step 5: Verify PR**

```bash
gh pr list -R candle-org/candle --head lvyufeng:compat/autograd-pr2 --state open
```
