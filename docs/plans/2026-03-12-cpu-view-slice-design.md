# CPU View/Slice Ops Design

**Goal:** Add slice/expand_copy/as_strided_* CPU ops with schema-first dispatch and PyTorch parity contract tests.

## Scope
- Ops: slice, slice_copy, slice_scatter, expand_copy, as_strided_, as_strided_copy, as_strided_scatter.
- CPU backend only in this batch.
- Follow schema-first contract: schemas -> contract tests -> kernels -> API.

## Architecture
- Register schemas in `src/candle/_dispatch/schemas.py`.
- Add contract tests in `tests/contract/` using the training-core parity helpers.
- Implement CPU kernels in `src/candle/_backends/cpu/ops.py` and register in
  `src/candle/_backends/cpu/__init__.py`.
- Expose functional/tensor API in `src/candle/_functional.py` and `src/candle/_tensor.py`
  if needed.

## Semantics
- `slice`: use basic indexing path; return a view when step==1 and forward; return a
  contiguous copy for step!=1 or negative strides (matching current getitem behavior).
- `slice_copy`: always return a contiguous copy.
- `slice_scatter`: write `src` into the slice view, return the updated base tensor.
- `expand_copy`: expand to target size, then return a contiguous copy (no shared storage).
- `as_strided_`: in-place metadata update (shape/stride/offset), no data copy; validate bounds.
- `as_strided_copy`: create as-strided view then return a contiguous copy.
- `as_strided_scatter`: write into the as-strided view and return the updated base tensor.
- Error behavior matches PyTorch where contract tests assert it.

## Testing
- Contract parity tests for each op: shapes, values, and view-vs-copy behavior.
- Error cases for invalid stride/offset where PyTorch is stable.
- Required gate: `pytest tests/contract/ -v --tb=short`.
- Run focused CPU view tests if applicable.

## Risks
- `as_strided` validation rules may differ slightly from PyTorch; keep tests conservative.
- View semantics for strided slices depend on existing getitem behavior.
