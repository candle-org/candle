# NPU L0 Parity Tranche Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make every benchmarked L0 single-operator on NPU at least as fast as torch_npu (median_ratio <= 1.0).

**Architecture:** Fix benchmark routing first (rms_norm measures wrong path), then correctness blockers (rope/cross_entropy), then host-overhead reduction for small ops via Cython fast-path optimization, then remaining ops. Each batch produces a JSON gate artifact proving progress.

**Tech Stack:** Python 3.11, Cython (`_C/_npu_ops.pyx`, `_C/_aclnn_ffi.pyx`), ACLNN ctypes bindings, pytest, conda envs candle311/torchnpu311, CANN 9.0.0.

---

## Scope

Implements `docs/superpowers/specs/2026-05-30-npu-l0-parity-tranche-design.md`.

Does NOT change L1/L2 graph/fusion paths, C++ autograd engine, or dispatcher architecture.

## File Structure

Create:

- `benchmarks/op_benchmark_npu/cases_native.py` — native-route benchmark case builders (rms_norm_native, etc.)
- `tests/npu/test_rope_l0_parity.py` — rope correctness and parity test
- `tests/npu/test_cross_entropy_npu_dtype.py` — cross_entropy dtype correctness test
- `tests/npu/test_rms_norm_native_route.py` — rms_norm native routing test

Modify:

- `benchmarks/op_benchmark_npu/cases.py` — add rms_norm_native case, rename old to rms_norm_composite
- `src/candle/_backends/npu/ops/math.py` — fix neg workspace issue for rope shapes
- `src/candle/nn/functional.py` — fix cross_entropy dtype promotion
- `src/candle/_C/_npu_ops.pyx` — optimize fast_add/fast_mul/fast_silu host overhead
- `docs/known-kernel-issues.md` — document neg CANN issue if confirmed

---

### Task 1: Add rms_norm_native benchmark case

**Files:**
- Modify: `benchmarks/op_benchmark_npu/cases.py`
- Test: run benchmark with `--ops rms_norm_native`

- [ ] **Step 1: Write rms_norm_native case builder**

In `benchmarks/op_benchmark_npu/cases.py`, add after the existing `_build_rms_norm`:

```python
def _build_rms_norm_native(torch_mod, F, device, dtype, batch, seq):
    """Benchmark the native RMSNorm path (nn.functional or nn.Module)."""
    x = torch_mod.randn((batch, seq, HIDDEN), device=device, dtype=dtype)
    weight = torch_mod.ones(HIDDEN, device=device, dtype=dtype)
    eps = 1e-6

    def fn():
        return F.rms_norm(x, (HIDDEN,), weight, eps)

    return fn
```

Then in the `CASES` dict, add:

```python
"rms_norm_native": {
    "builder": _build_rms_norm_native,
    "modes": ["fwd"],
    "scenarios": SCENARIOS,
},
```

Rename the existing `"rms_norm"` key to `"rms_norm_composite"`.

- [ ] **Step 2: Verify F.rms_norm exists in Candle functional API**

Run:

```bash
cd /home/jenkins/lvyufeng/candle/.worktrees/npu-l0-parity-tranche
source /usr/local/Ascend/cann-9.0.0/set_env.sh
source /home/jenkins/anaconda3/etc/profile.d/conda.sh
conda run -n candle311 python -c "import candle; print(hasattr(candle.nn.functional, 'rms_norm'))"
```

Expected: `True`. If `False`, add `rms_norm` to `candle/nn/functional.py` that dispatches to the existing `rms_norm` op.

- [ ] **Step 3: Run benchmark for rms_norm_native only**

```bash
source /usr/local/Ascend/cann-9.0.0/set_env.sh
source /home/jenkins/anaconda3/etc/profile.d/conda.sh
CONDA_SH=/home/jenkins/anaconda3/etc/profile.d/conda.sh \
CANN_SET_ENV=/usr/local/Ascend/cann-9.0.0/set_env.sh \
CANDLE_CONDA_ENV=candle311 \
TORCH_NPU_CONDA_ENV=torchnpu311 \
conda run -n candle311 --no-capture-output \
  python -m benchmarks.op_benchmark_npu.run \
  --ops rms_norm_native --dtype fp16 --scenario infer --mode fwd \
  --warmup 3 --iters 10 --json-output /tmp/rms_norm_native_check.json
```

Expected: row with status `ok` and a ratio much lower than 11289x. If it errors, debug the routing.

- [ ] **Step 4: Commit**

```bash
git add benchmarks/op_benchmark_npu/cases.py
git commit -m "bench(npu): add rms_norm_native case, rename old to composite"
```

---

### Task 2: Fix rope correctness blocker (neg workspace failure)

**Files:**
- Modify: `src/candle/_backends/npu/ops/math.py` (neg function)
- Modify: `docs/known-kernel-issues.md`
- Create: `tests/npu/test_rope_l0_parity.py`

- [ ] **Step 1: Write failing test for neg on rope-shaped tensor**

Create `tests/npu/test_rope_l0_parity.py`:

```python
"""Test that unary neg works on NPU for RoPE-shaped fp16 tensors."""
import candle as torch


def test_neg_fp16_rope_shape():
    """neg must work on (1, 32, 2048, 64) fp16 NPU tensor without workspace error."""
    x = torch.randn((1, 32, 2048, 64), device="npu", dtype=torch.float16)
    result = -x
    assert result.device.type == "npu"
    assert result.shape == x.shape
    assert result.dtype == torch.float16


def test_rope_composite_runs_on_npu():
    """Full RoPE composite must run without error on NPU."""
    x = torch.randn((1, 32, 2048, 128), device="npu", dtype=torch.float16)
    cos = torch.randn_like(x)
    sin = torch.randn_like(x)
    half = 64
    t1 = x[..., :half]
    t2 = x[..., half:]
    rotated = torch.cat((-t2, t1), dim=-1)
    result = x * cos + rotated * sin
    assert result.device.type == "npu"
    assert result.shape == x.shape
```

- [ ] **Step 2: Run test to verify it fails**

```bash
source /usr/local/Ascend/cann-9.0.0/set_env.sh
source /home/jenkins/anaconda3/etc/profile.d/conda.sh
conda run -n candle311 python -m pytest tests/npu/test_rope_l0_parity.py -v --tb=short
```

Expected: FAIL with `GetWorkspaceSize failed: 561103 [op=neg, device=npu]`.

- [ ] **Step 3: Diagnose neg failure**

In `src/candle/_C/_npu_ops.pyx`, `fast_neg` calls ACLNN neg directly. The workspace query fails for this shape/dtype.

Test whether `mul(x, -1)` works as an on-device workaround:

```bash
source /usr/local/Ascend/cann-9.0.0/set_env.sh
source /home/jenkins/anaconda3/etc/profile.d/conda.sh
conda run -n candle311 python -c "
import candle as torch
x = torch.randn((1, 32, 2048, 64), device='npu', dtype=torch.float16)
result = x * (-1)
print('OK, shape:', result.shape)
"
```

If this succeeds, the workaround is to use `mul(x, -1)` when ACLNN neg workspace query fails.

- [ ] **Step 4: Implement on-device composite workaround for neg**

In `src/candle/_C/_npu_ops.pyx`, modify `fast_neg` to catch the workspace error and fall back to `fast_mul(a, -1)`:

```python
def fast_neg(a):
    """Optimized out-of-place neg(a). Falls back to mul(a, -1) if ACLNN neg workspace fails."""
    _ensure_npu_imports()
    _ensure_ffi_neg()
    cdef int dev_idx = a.device.index or 0
    try:
        # existing ACLNN neg implementation
        ...
    except RuntimeError as e:
        if "GetWorkspaceSize failed" in str(e) or "561103" in str(e):
            # CANN neg workspace bug — use on-device mul workaround
            # TODO: re-enable native neg when CANN fixes workspace for this shape
            return fast_mul(a, -1)
        raise
```

Keep the native neg path as the primary attempt; only fall back on the specific error.

- [ ] **Step 5: Document in known-kernel-issues.md**

Add to `docs/known-kernel-issues.md`:

```markdown
## neg — NPU workspace failure on certain shapes

- **Op:** neg
- **Backend:** NPU (ACLNN)
- **Error:** `GetWorkspaceSize failed: 561103 [op=neg, device=npu]`
- **Affected shapes:** (1, 32, 2048, 64) fp16 and similar 4D tensors
- **CANN version:** 9.0.0
- **Workaround:** `mul(x, -1)` on-device composite
- **Native path:** Preserved in fast_neg, attempted first
```

- [ ] **Step 6: Run test to verify it passes**

```bash
source /usr/local/Ascend/cann-9.0.0/set_env.sh
source /home/jenkins/anaconda3/etc/profile.d/conda.sh
conda run -n candle311 python -m pytest tests/npu/test_rope_l0_parity.py -v --tb=short
```

Expected: PASS.

- [ ] **Step 7: Run rope benchmark case**

```bash
source /usr/local/Ascend/cann-9.0.0/set_env.sh
source /home/jenkins/anaconda3/etc/profile.d/conda.sh
CONDA_SH=/home/jenkins/anaconda3/etc/profile.d/conda.sh \
CANN_SET_ENV=/usr/local/Ascend/cann-9.0.0/set_env.sh \
CANDLE_CONDA_ENV=candle311 \
TORCH_NPU_CONDA_ENV=torchnpu311 \
conda run -n candle311 --no-capture-output \
  python -m benchmarks.op_benchmark_npu.run \
  --ops rope --dtype fp16 --scenario infer --mode fwd \
  --warmup 3 --iters 10 --json-output /tmp/rope_check.json
```

Expected: status `ok`, ratio reported.

- [ ] **Step 8: Commit**

```bash
git add src/candle/_C/_npu_ops.pyx docs/known-kernel-issues.md tests/npu/test_rope_l0_parity.py
git commit -m "fix(npu): work around neg workspace failure for rope shapes

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Fix cross_entropy dtype mismatch

**Files:**
- Modify: `src/candle/nn/functional.py` (cross_entropy / nll_loss)
- Create: `tests/npu/test_cross_entropy_npu_dtype.py`

- [ ] **Step 1: Write failing test**

Create `tests/npu/test_cross_entropy_npu_dtype.py`:

```python
"""Test cross_entropy works on NPU with fp16 input and int64 target."""
import candle as torch
import candle.nn.functional as F


def test_cross_entropy_fp16_npu():
    """cross_entropy must not raise dtype mismatch on NPU."""
    x = torch.randn((2048, 32000), device="npu", dtype=torch.float16)
    target = torch.randint(0, 32000, (2048,), device="npu")
    loss = F.cross_entropy(x, target)
    assert loss.device.type == "npu"
    assert loss.numel() == 1
```

- [ ] **Step 2: Run test to verify it fails**

```bash
source /usr/local/Ascend/cann-9.0.0/set_env.sh
source /home/jenkins/anaconda3/etc/profile.d/conda.sh
conda run -n candle311 python -m pytest tests/npu/test_cross_entropy_npu_dtype.py -v --tb=short
```

Expected: FAIL with `NPU mul requires matching dtypes`.

- [ ] **Step 3: Trace the dtype mismatch**

The cross_entropy composite does: log_softmax → gather → neg → mul → sum/div.

The `mul` step fails because one operand is fp16 and the other is a different dtype (likely a mask or count tensor that's int/float32). Trace in `src/candle/nn/functional.py` the `nll_loss` function to find where `valid_float` or similar intermediate has wrong dtype.

Run diagnostic:

```bash
source /usr/local/Ascend/cann-9.0.0/set_env.sh
source /home/jenkins/anaconda3/etc/profile.d/conda.sh
conda run -n candle311 python -c "
import candle as torch
import candle.nn.functional as F
x = torch.randn((4, 10), device='npu', dtype=torch.float16)
target = torch.randint(0, 10, (4,), device='npu')
try:
    F.cross_entropy(x, target)
except Exception as e:
    print('Error:', e)
    # Now trace step by step
    log_probs = F.log_softmax(x, dim=1)
    print('log_softmax ok, dtype:', log_probs.dtype)
    gathered = torch.gather(log_probs, 1, target.unsqueeze(1))
    print('gather ok, dtype:', gathered.dtype)
    neg_gathered = -gathered
    print('neg ok, dtype:', neg_gathered.dtype)
    # The mul with valid_count is likely the issue
"
```

- [ ] **Step 4: Fix dtype promotion in nll_loss**

In `src/candle/nn/functional.py`, in the `nll_loss` implementation, ensure that any scalar/count tensor used in `mul` or `div` is cast to match the input dtype before the NPU dispatch:

```python
# Before: valid_float may be int or float32
# After: cast to input dtype for NPU compatibility
valid_float = valid_count.to(dtype=input.dtype)
```

This preserves torch semantics (torch does this promotion internally) without weakening schema validation.

- [ ] **Step 5: Run test to verify it passes**

```bash
source /usr/local/Ascend/cann-9.0.0/set_env.sh
source /home/jenkins/anaconda3/etc/profile.d/conda.sh
conda run -n candle311 python -m pytest tests/npu/test_cross_entropy_npu_dtype.py -v --tb=short
```

Expected: PASS.

- [ ] **Step 6: Run cross_entropy benchmark**

```bash
source /usr/local/Ascend/cann-9.0.0/set_env.sh
source /home/jenkins/anaconda3/etc/profile.d/conda.sh
CONDA_SH=/home/jenkins/anaconda3/etc/profile.d/conda.sh \
CANN_SET_ENV=/usr/local/Ascend/cann-9.0.0/set_env.sh \
CANDLE_CONDA_ENV=candle311 \
TORCH_NPU_CONDA_ENV=torchnpu311 \
conda run -n candle311 --no-capture-output \
  python -m benchmarks.op_benchmark_npu.run \
  --ops cross_entropy --dtype fp16 --scenario infer --mode fwd \
  --warmup 3 --iters 10 --json-output /tmp/ce_check.json
```

Expected: status `ok`.

- [ ] **Step 7: Run CPU/contract regression tests**

```bash
source /home/jenkins/anaconda3/etc/profile.d/conda.sh
conda run -n candle311 python -m pytest tests/cpu/ tests/contract/ -v --tb=short -x
```

Expected: no regressions from the dtype fix.

- [ ] **Step 8: Commit**

```bash
git add src/candle/nn/functional.py tests/npu/test_cross_entropy_npu_dtype.py
git commit -m "fix(npu): align cross_entropy dtype promotion for NPU mul

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Profile and optimize add/mul/silu host overhead

**Files:**
- Modify: `src/candle/_C/_npu_ops.pyx`
- Modify: `src/candle/_C/_aclnn_ffi.pyx` (if needed)

- [ ] **Step 1: Confirm fast path routing**

Verify add/mul/silu reach Cython fast paths:

```bash
source /usr/local/Ascend/cann-9.0.0/set_env.sh
source /home/jenkins/anaconda3/etc/profile.d/conda.sh
conda run -n candle311 python -c "
import candle as torch
x = torch.randn((1, 2048, 4096), device='npu', dtype=torch.float16)
y = torch.randn_like(x)

# These should use fast paths without error
r1 = x + y
r2 = x * y
r3 = torch.nn.functional.silu(x)
print('add ok:', r1.shape)
print('mul ok:', r2.shape)
print('silu ok:', r3.shape)
"
```

Expected: all succeed. If any fails, fix routing before profiling.

- [ ] **Step 2: Profile per-call overhead**

```bash
source /usr/local/Ascend/cann-9.0.0/set_env.sh
source /home/jenkins/anaconda3/etc/profile.d/conda.sh
conda run -n candle311 python -c "
import time
import candle as torch

x = torch.randn((1, 2048, 4096), device='npu', dtype=torch.float16)
y = torch.randn_like(x)
torch.npu.synchronize()

# Warmup
for _ in range(10):
    _ = x + y
torch.npu.synchronize()

# Measure
N = 100
torch.npu.synchronize()
t0 = time.perf_counter()
for _ in range(N):
    _ = x + y
torch.npu.synchronize()
t1 = time.perf_counter()
print(f'add: {(t1-t0)/N*1000:.4f} ms/call')

torch.npu.synchronize()
t0 = time.perf_counter()
for _ in range(N):
    _ = x * y
torch.npu.synchronize()
t1 = time.perf_counter()
print(f'mul: {(t1-t0)/N*1000:.4f} ms/call')

torch.npu.synchronize()
t0 = time.perf_counter()
for _ in range(N):
    _ = torch.nn.functional.silu(x)
torch.npu.synchronize()
t1 = time.perf_counter()
print(f'silu: {(t1-t0)/N*1000:.4f} ms/call')
"
```

Record baseline numbers. Compare with torch_npu target (add ~0.08ms, mul ~0.08ms, silu ~0.11ms).

- [ ] **Step 3: Identify dominant overhead in fast_add/fast_mul/fast_silu**

In `src/candle/_C/_npu_ops.pyx`, each fast binary/unary op typically:

1. Calls `_ensure_npu_imports()` — one-time check, should be negligible after first call.
2. Creates output tensor via allocator.
3. Creates ACLNN descriptor/executor.
4. Queries workspace size.
5. Allocates workspace.
6. Launches kernel.
7. Returns wrapped tensor.

Add timing instrumentation (temporary, not committed) to identify which step dominates:

```python
# Temporary profiling in fast_add:
import time
t0 = time.perf_counter()
out = _allocate_output(a)  # step 2
t1 = time.perf_counter()
desc = _create_add_desc(a, b, out)  # step 3
t2 = time.perf_counter()
ws_size = _get_workspace_size(desc)  # step 4
t3 = time.perf_counter()
ws = _alloc_workspace(ws_size)  # step 5
t4 = time.perf_counter()
_launch(desc, ws)  # step 6
t5 = time.perf_counter()
print(f'alloc={t1-t0:.6f} desc={t2-t1:.6f} ws_query={t3-t2:.6f} ws_alloc={t4-t3:.6f} launch={t5-t4:.6f}')
```

- [ ] **Step 4: Implement optimizations based on profiling**

Based on prior execution-alignment analysis, likely optimizations (implement only what profiling confirms):

**a) Descriptor/executor reuse for same shape/dtype/stride:**

```python
# In _npu_ops.pyx, add a shape-keyed LRU cache for descriptors
_add_desc_cache = {}  # (a_shape, a_dtype, a_stride, b_shape, b_dtype, b_stride) -> descriptor

def _get_or_create_add_desc(a, b, out):
    key = (a.shape, a.dtype, tuple(a.stride()), b.shape, b.dtype, tuple(b.stride()))
    if key in _add_desc_cache:
        desc = _add_desc_cache[key]
        _rebind_addresses(desc, a, b, out)  # rebind data pointers only
        return desc
    desc = _create_add_desc(a, b, out)
    _add_desc_cache[key] = desc
    return desc
```

**b) Workspace pool (stream-scoped):**

```python
# Reuse workspace buffer if current allocation >= required size
_workspace_pool = {}  # stream_id -> (buffer, size)

def _get_workspace(size, stream_id):
    if stream_id in _workspace_pool:
        buf, cur_size = _workspace_pool[stream_id]
        if cur_size >= size:
            return buf
    buf = _alloc_workspace(size)
    _workspace_pool[stream_id] = (buf, size)
    return buf
```

**c) Output allocation fast path:**

If the allocator has a cached free block of the right size, skip the full allocation path. This is allocator-internal and may already exist.

- [ ] **Step 5: Verify improvement**

Rerun the profiling script from Step 2. Target: add/mul/silu each <= torch_npu median.

- [ ] **Step 6: Run benchmark gate for add/mul/silu**

```bash
source /usr/local/Ascend/cann-9.0.0/set_env.sh
source /home/jenkins/anaconda3/etc/profile.d/conda.sh
CONDA_SH=/home/jenkins/anaconda3/etc/profile.d/conda.sh \
CANN_SET_ENV=/usr/local/Ascend/cann-9.0.0/set_env.sh \
CANDLE_CONDA_ENV=candle311 \
TORCH_NPU_CONDA_ENV=torchnpu311 \
conda run -n candle311 --no-capture-output \
  python -m benchmarks.op_benchmark_npu.run \
  --ops add,mul,silu --dtype fp16 --scenario infer --mode fwd \
  --warmup 5 --iters 30 --json-output /tmp/batch3_after.json \
  --fail-on-ratio --max-ratio 1.0
```

Expected: exit 0, all three ops median_ratio <= 1.0.

- [ ] **Step 7: Run regression tests**

```bash
source /usr/local/Ascend/cann-9.0.0/set_env.sh
source /home/jenkins/anaconda3/etc/profile.d/conda.sh
conda run -n candle311 python -m pytest tests/cpu/ tests/contract/ -v --tb=short -x
```

- [ ] **Step 8: Commit**

```bash
git add src/candle/_C/_npu_ops.pyx src/candle/_C/_aclnn_ffi.pyx
git commit -m "perf(npu): reduce host overhead for add/mul/silu fast paths

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Optimize matmul/bmm ops

**Files:**
- Modify: `src/candle/_C/_npu_ops.pyx` (fast_matmul and related)
- Modify: `src/candle/_backends/npu/ops/linalg.py` (if routing issues found)

- [ ] **Step 1: Confirm batched ACLNN path**

```bash
source /usr/local/Ascend/cann-9.0.0/set_env.sh
source /home/jenkins/anaconda3/etc/profile.d/conda.sh
conda run -n candle311 python -c "
import candle as torch
# matmul_qkv shape: (batch*heads, seq, head_dim) @ (batch*heads, head_dim, seq)
a = torch.randn((32, 2048, 128), device='npu', dtype=torch.float16)
b = torch.randn((32, 128, 2048), device='npu', dtype=torch.float16)
r = torch.matmul(a, b)
print('matmul ok:', r.shape)

# bmm
r2 = torch.bmm(a, b)
print('bmm ok:', r2.shape)
"
```

- [ ] **Step 2: Profile matmul overhead**

Same approach as Task 4 Step 2 but for matmul shapes. Measure per-call latency and compare with torch_npu targets.

- [ ] **Step 3: Apply descriptor/workspace optimizations**

Reuse the same patterns from Task 4 (descriptor cache, workspace pool) for `fast_matmul` and `fast_bmm`. The matmul shapes in the benchmark are static, so caching is safe.

- [ ] **Step 4: Check for unnecessary contiguous conversions**

In `src/candle/_backends/npu/ops/linalg.py`, check if matmul/bmm forces contiguous on inputs that are already contiguous or have ACLNN-safe strides. Remove unnecessary `.contiguous()` calls.

- [ ] **Step 5: Run benchmark gate**

```bash
source /usr/local/Ascend/cann-9.0.0/set_env.sh
source /home/jenkins/anaconda3/etc/profile.d/conda.sh
CONDA_SH=/home/jenkins/anaconda3/etc/profile.d/conda.sh \
CANN_SET_ENV=/usr/local/Ascend/cann-9.0.0/set_env.sh \
CANDLE_CONDA_ENV=candle311 \
TORCH_NPU_CONDA_ENV=torchnpu311 \
conda run -n candle311 --no-capture-output \
  python -m benchmarks.op_benchmark_npu.run \
  --ops matmul_qkv,bmm_attn_scores,bmm_attn_output,matmul_ffn_up,matmul_ffn_down \
  --dtype fp16 --scenario infer --mode fwd \
  --warmup 5 --iters 30 --json-output /tmp/batch4_matmul_after.json \
  --fail-on-ratio --max-ratio 1.0
```

Expected: all matmul/bmm ops median_ratio <= 1.0.

- [ ] **Step 6: Commit**

```bash
git add src/candle/_C/_npu_ops.pyx src/candle/_backends/npu/ops/linalg.py
git commit -m "perf(npu): optimize matmul/bmm descriptor and contiguity paths

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: Optimize softmax, embedding, dropout

**Files:**
- Modify: `src/candle/_C/_npu_ops.pyx`
- Modify: `src/candle/_backends/npu/ops/activation.py` (if routing issues)

- [ ] **Step 1: Profile softmax**

```bash
source /usr/local/Ascend/cann-9.0.0/set_env.sh
source /home/jenkins/anaconda3/etc/profile.d/conda.sh
conda run -n candle311 python -c "
import time, candle as torch
x = torch.randn((32, 2048, 2048), device='npu', dtype=torch.float16)
torch.npu.synchronize()
for _ in range(10): torch.nn.functional.softmax(x, dim=-1)
torch.npu.synchronize()
N=50
t0=time.perf_counter()
for _ in range(N): torch.nn.functional.softmax(x, dim=-1)
torch.npu.synchronize()
t1=time.perf_counter()
print(f'softmax: {(t1-t0)/N*1000:.4f} ms')
"
```

Softmax is only 1.15x slower — likely just descriptor overhead. Apply same cache pattern.

- [ ] **Step 2: Profile embedding**

```bash
source /usr/local/Ascend/cann-9.0.0/set_env.sh
source /home/jenkins/anaconda3/etc/profile.d/conda.sh
conda run -n candle311 python -c "
import time, candle as torch
weight = torch.randn((32000, 4096), device='npu', dtype=torch.float16)
indices = torch.randint(0, 32000, (1, 2048), device='npu')
torch.npu.synchronize()
for _ in range(10): torch.nn.functional.embedding(indices, weight)
torch.npu.synchronize()
N=50
t0=time.perf_counter()
for _ in range(N): torch.nn.functional.embedding(indices, weight)
torch.npu.synchronize()
t1=time.perf_counter()
print(f'embedding: {(t1-t0)/N*1000:.4f} ms')
"
```

Embedding is 3.56x slower — may have routing or allocation issues beyond descriptor overhead.

- [ ] **Step 3: Profile dropout**

```bash
source /usr/local/Ascend/cann-9.0.0/set_env.sh
source /home/jenkins/anaconda3/etc/profile.d/conda.sh
conda run -n candle311 python -c "
import time, candle as torch
x = torch.randn((1, 2048, 4096), device='npu', dtype=torch.float16)
torch.npu.synchronize()
for _ in range(10): torch.nn.functional.dropout(x, p=0.1, training=True)
torch.npu.synchronize()
N=50
t0=time.perf_counter()
for _ in range(N): torch.nn.functional.dropout(x, p=0.1, training=True)
torch.npu.synchronize()
t1=time.perf_counter()
print(f'dropout: {(t1-t0)/N*1000:.4f} ms')
"
```

- [ ] **Step 4: Apply optimizations**

For each op, apply the relevant subset of:
- Descriptor/executor cache (same as Task 4).
- Workspace pool reuse.
- Remove unnecessary contiguous/view conversions.
- For embedding: check if output allocation or index handling is the bottleneck.
- For dropout: check if random mask generation or sync is the bottleneck.

- [ ] **Step 5: Run benchmark gate**

```bash
source /usr/local/Ascend/cann-9.0.0/set_env.sh
source /home/jenkins/anaconda3/etc/profile.d/conda.sh
CONDA_SH=/home/jenkins/anaconda3/etc/profile.d/conda.sh \
CANN_SET_ENV=/usr/local/Ascend/cann-9.0.0/set_env.sh \
CANDLE_CONDA_ENV=candle311 \
TORCH_NPU_CONDA_ENV=torchnpu311 \
conda run -n candle311 --no-capture-output \
  python -m benchmarks.op_benchmark_npu.run \
  --ops softmax,embedding,dropout --dtype fp16 --scenario infer --mode fwd \
  --warmup 5 --iters 30 --json-output /tmp/batch4_remaining_after.json \
  --fail-on-ratio --max-ratio 1.0
```

Expected: all three ops median_ratio <= 1.0.

- [ ] **Step 6: Commit**

```bash
git add src/candle/_C/_npu_ops.pyx src/candle/_backends/npu/ops/activation.py
git commit -m "perf(npu): optimize softmax/embedding/dropout host overhead

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 7: Final L0 gate validation

**Files:**
- No new code changes (validation only)

- [ ] **Step 1: Run full L0 gate**

```bash
source /usr/local/Ascend/cann-9.0.0/set_env.sh
source /home/jenkins/anaconda3/etc/profile.d/conda.sh
mkdir -p results/npu_perf_gate/l0_final
CONDA_SH=/home/jenkins/anaconda3/etc/profile.d/conda.sh \
CANN_SET_ENV=/usr/local/Ascend/cann-9.0.0/set_env.sh \
CANDLE_CONDA_ENV=candle311 \
TORCH_NPU_CONDA_ENV=torchnpu311 \
conda run -n candle311 --no-capture-output \
  python -m benchmarks.op_benchmark_npu.run \
  --mode fwd \
  --dtype fp16 \
  --scenario infer \
  --warmup 5 \
  --iters 30 \
  --json-output results/npu_perf_gate/l0_final/l0_parity_final.json \
  --fail-on-ratio \
  --max-ratio 1.0
```

Expected: exit 0, no failures, all target ops median_ratio <= 1.0.

- [ ] **Step 2: Run full regression suite**

```bash
source /usr/local/Ascend/cann-9.0.0/set_env.sh
source /home/jenkins/anaconda3/etc/profile.d/conda.sh
conda run -n candle311 pylint src/candle/ --rcfile=.github/pylint.conf
conda run -n candle311 python -m pytest tests/cpu/ tests/contract/ -v --tb=short
```

Expected: pylint clean, all tests pass.

- [ ] **Step 3: Run L1/L2 observation (not a gate)**

```bash
source /usr/local/Ascend/cann-9.0.0/set_env.sh
source /home/jenkins/anaconda3/etc/profile.d/conda.sh
CONDA_SH=/home/jenkins/anaconda3/etc/profile.d/conda.sh \
CANN_SET_ENV=/usr/local/Ascend/cann-9.0.0/set_env.sh \
CANDLE_CONDA_ENV=candle311 \
TORCH_NPU_CONDA_ENV=torchnpu311 \
conda run -n candle311 --no-capture-output \
  python -m benchmarks.pipeline_npu.run \
  --cases A1,A2s,B1s --mode eager --warmup 3 --iters 10 \
  --json-output results/npu_perf_gate/l0_final/l1_observation.json
```

Record L1 ratios for comparison. Not a merge gate but ensures no model-level regression.

- [ ] **Step 4: Commit final validation evidence**

```bash
git add results/npu_perf_gate/l0_final/
git commit -m "evidence(npu): L0 parity final gate results

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```


