# Forward-Mode AD (CPU) Design

## Goal
Implement PyTorch-aligned forward-mode AD for Candle on CPU: dual levels, make/unpack dual tensors, JVP propagation, and required autograd helpers to run `compat/pytorch/test_autograd.py` with minimal xfails.

## Scope
- Tensor-level forward-grad storage with support for nested levels.
- `candle.autograd.forward_ad` API (`enter_dual_level`, `exit_dual_level`, `dual_level`, `make_dual`, `unpack_dual`).
- JVP registry and dispatcher integration (raise if missing rule when tangents are present).
- Autograd helpers required by `test_autograd.py` (`GradientEdge`, `get_gradient_edge`, `_calculate_shape`, etc.).
- Core JVP rules for common ops used in training and `test_autograd.py`.

## Architecture
Forward-mode information lives on `Tensor` (per-level tangent map). The `forward_ad` module manages the active level stack and reads/writes tangents. Dispatcher computes primal values via existing kernels, then applies a JVP rule if any input has a tangent for the current level. Missing JVP rules raise a `RuntimeError` with the op name.

## Data Flow
1. `enter_dual_level()` creates a new level.
2. `make_dual(primal, tangent)` stores tangent on the current level in `Tensor`.
3. `dispatch()` runs kernel → checks for active level + input tangents → applies JVP rule → stores output tangents.
4. `unpack_dual(tensor)` returns `(tensor, tangent)` for the level.
5. `exit_dual_level()` clears tangents for the level.

## Error Handling
- `make_dual` requires an active level; otherwise `RuntimeError`.
- `make_dual` enforces floating/complex dtype and exact shape match.
- Missing JVP rule when tangents exist raises `RuntimeError`.
- Forward AD level exit is LIFO-only, matching PyTorch.

## Testing
- Add CPU tests for forward AD stack, make/unpack, and a few key JVP rules.
- Run `compat/pytorch/run.py --file test_autograd.py` and iterate on failures.
- Allow explicit xfails for forward‑AD gradcheck/higher‑order/edge tests that do not impact training/transformers usage.
