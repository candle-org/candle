# MPS GPU RNG — Philox 4x32-10

## Goal

Replace CPU-side numpy RNG for MPS tensors with GPU-native Philox 4x32-10 PRNG via Metal shaders. Per-device generators with deterministic self-consistent reproducibility.

## Philox Metal Shaders

Core Philox 4x32-10 implemented as MSL utility functions. Each thread computes `philox4x32_10(uint4 counter, uint2 key) -> uint4` independently.

- **Counter**: `(gid, 0, offset, 0)` — gid = thread index, offset = per-op counter
- **Key**: `(seed_lo, seed_hi)` — from 64-bit seed
- **Output**: 4 random uint32 per call

Kernels (each thread produces 4 output elements):

| Kernel | Conversion |
|--------|------------|
| `philox_uniform_f32/f16` | `(uint >> 8) * (1/2^24)` → [0,1), then `low + val*(high-low)` |
| `philox_normal_f32/f16` | Box-Muller on 2 uniform pairs → 4 normals, then `mean + val*std` |
| `philox_bernoulli_f32` | uniform < p ? 1.0 : 0.0 |
| `philox_randint_i32/i64` | `low + (uint % range)` |
| `philox_dropout_f32/f16` | Fused: generate mask + multiply + scale in one kernel |

Generator offset advances by `ceil(N/4)` per op.

## Generator Class

`_random.py::Generator` gains `device='mps'` support:
- Stores `(seed, offset)`, no numpy RNG
- `philox_engine_inputs(num_elements)` returns `(seed, offset)`, advances offset

New `src/candle/mps.py`:
- `is_available()`, `manual_seed(seed)`, `_default_generator`

`manual_seed()` propagates to MPS (like NPU).

## Op Integration

### Creation (`mps/creation.py`)
- `rand_create` → `philox_uniform` kernel
- `randn_create` → `philox_normal` kernel
- `randint_create` → `philox_randint` kernel
- `randperm_create` → `philox_uniform` on GPU → CPU argsort → copy back

### In-place (`mps/ops.py`)
- `uniform_`, `normal_`, `bernoulli_` → GPU Philox kernels with `_can_use_gpu` guard

### Dropout (`mps/ops.py`)
- Fused `philox_dropout` kernel (mask + multiply + scale)

### Dispatch (`metal_compute.py`)
- `dispatch_philox_fill(kernel, out_buf, seed_lo, seed_hi, offset, N)`
- `dispatch_philox_fill_params(kernel, out_buf, seed_lo, seed_hi, offset, p1, p2, N)`
- `dispatch_philox_dropout(kernel, a_buf, out_buf, seed_lo, seed_hi, offset, prob, scale, N)`

## Files

| File | Changes |
|------|---------|
| `metal_shaders.py` | Philox utility + 5 kernel templates + generators |
| `metal_compute.py` | 3 dispatch methods + ctypes encoders |
| `creation.py` | GPU paths for rand/randn/randint/randperm |
| `ops.py` | GPU paths for uniform_/normal_/bernoulli_/dropout |
| `_random.py` | Generator MPS support, seed propagation |
| `mps.py` | New module |
| `test_mps_rng.py` | ~20 tests |
