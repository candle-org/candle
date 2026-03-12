# Autograd Package Rename Design

## Goal
Rename `candle._autograd` to `candle.autograd`, update internal imports, remove the `_autograd` public module, and add minimal `candle.autograd._functions` support to unblock PyTorch autograd compatibility tests.

## Scope
- Rename the package directory from `src/candle/_autograd/` to `src/candle/autograd/`.
- Update all internal imports to reference `candle.autograd` instead of `candle._autograd`.
- Remove the special autograd alias in `src/_candle_torch_compat.py` so `torch.autograd` resolves to `candle.autograd` via default mapping.
- Add `src/candle/autograd/_functions/__init__.py` and `src/candle/autograd/_functions/tensor.py` with a minimal `Resize` autograd `Function` (and only add `Type` if test coverage requires it).
- Update compatibility tests that currently assert `candle._autograd` is the target.

## Architecture
The rename is a direct package move plus import path updates. `candle.autograd` becomes the canonical location for autograd public APIs. The compatibility hook no longer rewrites `torch.autograd` to `_autograd`, so `torch.autograd` imports are redirected to `candle.autograd`. The `_functions` module is added to satisfy `torch.autograd._functions.Resize.apply` used by PyTorch tests, with behavior aligned to PyTorch’s shape checks and gradient reshaping.

## Data Flow
- `import torch.autograd` -> `import candle.autograd` (via meta path redirect).
- `torch.autograd._functions.Resize.apply` -> `candle.autograd._functions.Resize.apply`.
- `Resize.forward` validates `numel` against the requested shape, saves input shape, and returns a view with the new shape (using `contiguous()` when needed).
- `Resize.backward` reshapes `grad_output` back to the saved input shape.

## Error Handling
- On `numel` mismatch, `Resize.forward` raises `RuntimeError` with the PyTorch-compatible message describing the requested and actual element counts.

## Testing
- Update `tests/test_torch_compat.py` to assert `torch.autograd is candle.autograd` and to update name resolution expectations.
- Run `pytest tests/contract/ -v --tb=short` after changes.
- Run `python compat/pytorch/run.py --file test_autograd.py` and capture failures.
- Add any remaining, well-understood failures to `compat/pytorch/xfail.yaml` with reasons after collecting evidence.

## Compatibility Notes
This is a breaking change for `import candle._autograd`, which will no longer resolve. Internal references are updated accordingly. External users must switch to `import candle.autograd`.
