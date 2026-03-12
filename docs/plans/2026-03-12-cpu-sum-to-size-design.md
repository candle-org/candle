# CPU sum_to_size Design

**Goal:** Add `sum_to_size` on CPU with full PyTorch error-message parity, correct forward/backward semantics, and contract tests that compare candle vs. real torch.

**Scope**
- Schema registration already exists: `sum_to_size(Tensor input, int[] size) -> Tensor`.
- Add full argument validation to match PyTorch error strings.
- Implement CPU kernel, meta inference, autograd backward, and contract tests.
- Export functional/tensor API already exists.

**Non-Goals**
- No new GPU/NPU kernels.
- No additional operator families beyond `sum_to_size`.

**Architecture**
- `src/candle/_dispatch/schema.py`: add `sum_to_size`-specific validation that mirrors PyTorch type errors for `size` (top-level types and per-element errors with exact messages).
- `src/candle/_backends/cpu/ops.py`: implement `sum_to_size` by reducing over extra leading dims and dims where `target==1` and input dim > 1, matching PyTorch expandability rules. If target size equals input shape, return the input view directly.
- `src/candle/_backends/meta/infer.py`: add `infer_sum_to_size` returning the target shape with correct error handling for non-expandable sizes.
- `src/candle/_backends/cpu/__init__.py`: register `sum_to_size` with CPU kernel and meta.
- `src/candle/_backends/autograd.py`: add backward that expands grad to input shape using `expand` (consistent with PyTorch semantics).

**Error Semantics**
- Invalid top-level `size` types raise `TypeError` with exact PyTorch text: "must be tuple of ints, not <type>".
- Invalid first element raises: "must be tuple of ints, but found element of type <type> at pos 0".
- Invalid later element raises: "failed to unpack the object at pos <idx> with error \"type must be tuple of ints,but got <type>\"".
- Bool elements are treated like ints except at position 0 (matches PyTorch).
- Non-expandable sizes raise `RuntimeError`: "size {[...] } is not expandable to size {[...]}".

**Testing**
- Expand `tests/contract/test_training_core_sum_to_size_parity.py` to cover:
  - Forward parity for typical sizes and scalar output.
  - Full error-message matrix vs real torch.
  - Backward parity for reduction cases.
  - View-like behavior when target size equals input shape (storage sharing).

**Risks**
- Error-message strings must match exactly; tests compare against real torch.
- Backward must broadcast correctly for all rank reductions.
