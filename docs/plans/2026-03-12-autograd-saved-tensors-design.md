# Autograd Saved-Tensor Interface Design

**Goal:** Implement a PyTorch-compatible saved-tensor interface (`_saved_*`, `_raw_saved_*`, and `torch._C._autograd.SavedTensor`) to satisfy `test_autograd.py` behaviors with minimal scope.

**Scope:** CPU-only, focus on autograd semantics needed by PyTorch `test_autograd.py` (no full checkpoint engine rewrite). Keep changes localized to autograd Node/SavedTensor, graph hooks, and minimal autograd wrapper metadata.

## Requirements (from `test_autograd.py`)

- `grad_fn._saved_*` returns materialized tensors/values.
- `grad_fn._raw_saved_*` returns `torch._C._autograd.SavedTensor` objects.
- Access after backward (release) raises `RuntimeError` containing `after they have already been freed`.
- `SavedTensor.register_hooks(pack, unpack)`:
  - requires two callables, otherwise `TypeError`.
  - forbids registering on `None` (`RuntimeError: None is forbidden`).
  - forbids double registration (`RuntimeError: already been set`).
  - if pack hook modifies input in-place (version counter changes), raise `RuntimeError: A saved tensor pack hook is modifying its input in place.`
  - unpack hook must return a Tensor or raise `TypeError: Output of saved tensor unpack_hook expected to be a Tensor`.
- Global hooks via `torch.autograd.graph.saved_tensors_hooks(pack, unpack)`; hooks stack must be nestable.
- Disabling hooks via `torch.autograd.graph.disable_saved_tensors_hooks(msg)` causes registration to raise `RuntimeError(msg)`.
- `_saved_tensors_hooks_is_enabled()` returns current enable state.
- For common ops, saved field names match PyTorch (at least `_saved_self`, `_saved_other`, `_saved_result`, `_saved_indices`, `_saved_dim`, and scalar/optional saved fields used in tests).

## Design Overview

### 1) `SavedTensor` wrapper

- Holds:
  - `_tensor_ref` (Tensor or None), `_saved_version`, `_released` flag.
  - `_packed` (if default/global hooks used), per-object hook pair.
- `register_hooks(pack, unpack)`:
  - Validate callables and one-time registration.
  - Reject if `tensor_ref is None`.
  - In-place modification check: capture version, run pack, compare version.
- `materialize()`:
  - If released, raise runtime error with “after they have already been freed”.
  - Apply hooks: per-object if present, else global hooks if enabled.
  - If unpack returns non-Tensor, raise TypeError.

### 2) `Node` saved fields

- Add `_raw_saved_tensors` list and `_saved_fields` dict.
- `save_for_backward()` populates `_raw_saved_tensors` with `SavedTensor` for each input (including `None`).
- `__getattr__` resolution:
  - `_raw_saved_*` returns corresponding `SavedTensor`(s).
  - `_saved_*` returns `materialize()` of those values (for tuples/lists, preserve structure).
- `release_saved_tensors()` marks all SavedTensor entries as released.

### 3) Global hooks state

- Extend `autograd.graph` with:
  - `disable_saved_tensors_hooks(msg)` context manager
  - `_saved_tensors_hooks_is_enabled()`
  - `current_saved_tensors_hooks()` checks enabled flag.

### 4) `_C._autograd` shim

- `candle._C._autograd` object exposing:
  - `SavedTensor` class
  - `_saved_tensors_hooks_is_enabled()`
- `torch.autograd.SavedTensor` should raise “forbidden” on direct construction.

### 5) Minimal per-op mapping

- Update autograd wrappers to save named fields:
  - Binary ops: `_saved_self`, `_saved_other`
  - Unary ops: `_saved_self`
  - Output-saved ops: `_saved_result` for `exp`, `tanh`, etc.
  - For index/select: `_saved_indices` and `_saved_dim` where used by tests
  - For optional scalar fields (e.g., logit eps, rounding_mode), store directly in `_saved_fields`.

## Test Plan

1. Add unit tests under `tests/cpu/`:
   - `SavedTensor.register_hooks` behavior
   - `_saved_*` vs `_raw_saved_*` access
   - release errors
   - pack-hook in-place modification detection
2. Run:
   - `pytest tests/cpu/test_saved_tensor_hooks.py -v`
   - `python compat/pytorch/run.py --file test_autograd.py -x -vv --maxfail=1`

## Risks / Notes

- Field naming is derived from PyTorch tests; we will add mappings incrementally to satisfy test coverage.
- Keep changes minimal; do not attempt to fully mirror PyTorch autograd internals beyond test needs.
