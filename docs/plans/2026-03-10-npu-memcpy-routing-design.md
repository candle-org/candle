# NPU Memcpy Routing Design

## Goal

Route all NPU memcpy operations through framework-provided runtime helpers instead of direct
`acl.rt.memcpy` calls. This ensures consistent stream semantics and prevents bypassing the
NPU runtime abstraction.

## Scope

- Replace direct `acl.rt.memcpy` usage across NPU-related code paths (functionalize, storage,
  ops, distributed, and runtime helpers).
- Add runtime-level `memcpy_h2d`, `memcpy_d2h`, and `memcpy_d2d` helpers.
- Preserve existing synchronization behavior by making sync an explicit call-site choice.

## Stream Semantics

- If a call-site does not provide a stream, runtime helpers will use
  `npu_state.current_stream()` for the corresponding device.
- If a call-site passes a stream, runtime helpers will use that stream exactly.
- `non_blocking=True` will use async memcpy when available; otherwise fall back to
  synchronous memcpy.

## Error Handling

- Runtime helpers validate ACL return codes and raise `RuntimeError` with existing message
  patterns.
- D2H synchronization remains explicit at call-sites (no implicit sync inside helpers unless
  the call-site passes a `sync=True` option).

## Implementation Plan

1. Add runtime helpers:
   - `memcpy_h2d(dst_ptr, size, src_ptr, runtime=None, stream=None, non_blocking=False)`
   - `memcpy_d2h(dst_ptr, size, src_ptr, runtime=None, stream=None, non_blocking=False)`
   - `memcpy_d2d(dst_ptr, size, src_ptr, runtime=None, stream=None, non_blocking=False)`
2. Replace direct `acl.rt.memcpy` usage with these helpers:
   - `src/candle/_dispatch/functionalize.py`
   - `src/candle/_storage.py`
   - `src/candle/_backends/npu/ops.py`
   - `src/candle/distributed/_process_group.py`
   - `src/candle/distributed/__init__.py`
3. Update tests to cover:
   - Stream selection when `stream=None`.
   - Stream passthrough when a stream is provided.

## Testing

- Run `pytest tests/npu/test_npu_streams.py -v --tb=short`.
- If any additional NPU copy paths are covered, add targeted tests in `tests/npu/`.

## Notes

- Avoid introducing implicit synchronization in runtime helpers; keep sync control at
  call-sites for parity with existing behavior.
- This is a mechanical change; no functional changes beyond stream routing should be
  introduced.
