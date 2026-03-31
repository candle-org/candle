# Candle Parallelism Roadmap Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Upgrade Candle from a "basic distributed + NPU-first parallel runtime" into a training-focused parallel stack with real async overlap, public FSDP, shard-aware checkpointing, CUDA/NCCL parity on hot paths, and usable DTensor / tensor-parallel foundations.

**Architecture:** Deliver parallelism in waves. First, make the existing distributed training path real and fast: `Work` / `Future` / DDP overlap must become true async hot paths implemented in Cython. Next, expose public FSDP on top of the existing composable implementation and make checkpointing shard-aware. Only after the NPU training path is stable should the project invest in CUDA/NCCL parity, then DTensor / tensor parallel, and finally pipeline-parallel and ecosystem polish.

**Tech Stack:** Cython >= 3, Python 3.11, Candle distributed runtime (`src/candle/distributed/`), Candle autograd / nn / tensor internals, HCCL / ACLRT, CUDA / NCCL, pytest, optional benchmark scripts under `benchmarks/`

---

## Non-Negotiable Constraints

1. **Cython-first hot paths**
   - If PyTorch implements a parallel / distributed path in C++, Candle must implement the corresponding hot path in **Cython**, not pure Python.
   - Pure Python is acceptable only for:
     - API wrappers
     - configuration and validation
     - fallback glue outside the main performance path
     - tests and documentation helpers

2. **Do not regress working NPU paths**
   - Existing HCCL / NPU collectives and stream/event functionality must remain usable while Cython hot paths are introduced.

3. **Preserve public API where already exposed**
   - Existing imports from `candle.distributed`, `candle.nn.parallel`, `candle.cuda`, `candle.npu`, and `candle.utils.data` must continue to work unless the roadmap explicitly changes them.

4. **Use TDD for every hot-path migration**
   - For each phase: add failing contract tests first, then add Cython implementation, then run targeted tests, then run broader regression tests.

5. **Phase gates matter**
   - Do not start CUDA/NCCL parity before async runtime semantics are stable.
   - Do not start TP/PP before public FSDP and shard-aware checkpointing are stable.

---

## Phase Order

- **P0:** True async runtime + DDP overlap
- **P1:** Public FSDP (NPU-first) + Cython shard hot paths
- **P2:** Distributed checkpoint / resume for DDP + FSDP
- **P3:** CUDA / NCCL main training stack
- **P4:** DTensor / DeviceMesh / Tensor Parallel
- **P5:** Pipeline Parallel / MoE primitives
- **P6:** CPU thread-control APIs, RPC / elastic decisions, DataParallel policy, docs cleanup

---

### Task 1: Lock async runtime contracts with failing tests

**Files:**
- Create: `tests/distributed/test_work_async.py`
- Create: `tests/distributed/test_ddp_async_overlap.py`
- Modify if needed: `tests/distributed/conftest.py`

**Step 1: Write failing `Work` / `Future` contract tests**

Cover these behaviors explicitly:
- `Work.get_future()` must not eagerly call `wait()`.
- `Future` returned by `get_future()` must be unresolved immediately after submit on async backend paths.
- `Work.wait()` must resolve the future exactly once.
- `is_completed()` must reflect device completion, not just Python object creation.

Suggested test names:
- `test_work_get_future_is_not_precompleted()`
- `test_work_wait_completes_future_once()`
- `test_async_collective_returns_unfinished_work()`

**Step 2: Write failing DDP overlap tests**

Cover these behaviors explicitly:
- DDP gradient bucket reduction should not force full sync at every bucket arrival.
- Comm hook futures should represent pending work, not already-completed wrappers.
- `no_sync()` must still skip sync while preserving later final reduction.

Suggested test names:
- `test_ddp_comm_hook_future_is_pending_before_wait()`
- `test_ddp_no_sync_still_allows_final_sync()`

**Step 3: Run targeted tests and confirm failure**

Run:
```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda run -n candle python -m pytest tests/distributed/test_work_async.py tests/distributed/test_ddp_async_overlap.py -v --tb=short
```

Expected:
- Failures showing `get_future()` resolves immediately.
- Failures showing DDP hook futures behave synchronously.

**Step 4: Commit**

```bash
git add tests/distributed/test_work_async.py tests/distributed/test_ddp_async_overlap.py tests/distributed/conftest.py
git commit -m "test(distributed): lock async work and ddp overlap contracts"
```

---

### Task 2: Move `Work` hot path into Cython

**Files:**
- Modify: `src/candle/distributed/_c10d.pyx`
- Modify: `src/candle/distributed/_c10d.pxd`
- Modify: `src/candle/distributed/_work.py`
- Modify: `src/candle/distributed/__init__.py`
- Modify: `setup.py`
- Test: `tests/distributed/test_work_async.py`

**Step 1: Define the Cython `Work` contract**

`_c10d.pyx` / `_c10d.pxd` must expose a fast-path `Work` with typed fields for:
- completion flag
- exception
- stream handle / device id
- callback or completion hook storage
- optional result payload

Required public methods:
- `wait(timeout=None)`
- `is_completed()`
- `is_success()`
- `exception()`
- `source_rank()`
- `result()`
- `synchronize()`
- `get_future()`

**Step 2: Ensure the Cython `Work` does not resolve futures eagerly**

Implement `get_future()` so it:
- creates and stores a `Future`
- returns immediately
- resolves only when device completion or explicit wait finishes

**Step 3: Keep `_work.py` as API-compatible fallback only**

Do not delete the Python fallback. Make it clearly secondary and keep import fallback in `distributed/__init__.py`.

**Step 4: Build extension and run targeted tests**

Run:
```bash
python setup.py build_ext --inplace
source ~/miniconda3/etc/profile.d/conda.sh && conda run -n candle python -m pytest tests/distributed/test_work_async.py -v --tb=short
```

Expected:
- `candle.distributed._c10d` builds successfully.
- `Work` async contract tests pass.

**Step 5: Commit**

```bash
git add setup.py src/candle/distributed/_c10d.pyx src/candle/distributed/_c10d.pxd src/candle/distributed/_work.py src/candle/distributed/__init__.py tests/distributed/test_work_async.py
git commit -m "feat(distributed): add Cython work fast path"
```

---

### Task 3: Wire HCCL completion into real async future resolution

**Files:**
- Modify: `src/candle/distributed/_c10d_hccl.pyx`
- Modify: `src/candle/distributed/_process_group.py`
- Modify: `src/candle/_backends/npu/streams.py`
- Test: `tests/distributed/test_work_async.py`
- Test: `tests/distributed/test_ddp_async_overlap.py`

**Step 1: Identify the HCCL submission points**

Audit all HCCL collective entry points used by:
- `all_reduce`
- `all_gather`
- `reduce_scatter`
- `all_to_all(_single)`
- `send` / `recv`

**Step 2: Attach completion state to submitted HCCL work**

The hot path must be Cython. At minimum:
- return a pending `Work`
- store stream and completion metadata on the `Work`
- complete the associated future when the stream completes

**Step 3: Keep `wait()` correct**

`wait()` must still be the synchronous escape hatch and must resolve any pending future exactly once.

**Step 4: Run targeted tests**

Run:
```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda run -n candle python -m pytest tests/distributed/test_work_async.py tests/distributed/test_ddp_async_overlap.py -v --tb=short
```

Expected:
- Async work tests pass on HCCL-enabled machines.
- Pending futures are no longer precompleted.

**Step 5: Commit**

```bash
git add src/candle/distributed/_c10d_hccl.pyx src/candle/distributed/_process_group.py src/candle/_backends/npu/streams.py tests/distributed/test_work_async.py tests/distributed/test_ddp_async_overlap.py
git commit -m "feat(hccl): resolve work futures on real stream completion"
```

---

### Task 4: Cythonize DDP bucket bookkeeping and reduction hot path

**Files:**
- Modify: `src/candle/nn/parallel/distributed.py`
- Create: `src/candle/distributed/_ddp_fastpath.pyx`
- Create: `src/candle/distributed/_ddp_fastpath.pxd`
- Modify: `setup.py`
- Test: `tests/distributed/test_ddp_async_overlap.py`
- Create: `tests/distributed/test_ddp_bucket_fastpath.py`

**Step 1: Add failing tests for bucket sizing and bucket lifecycle**

Cover these cases:
- bucket sizing must use real dtype byte width, not `numel() * 4`
- gradient_as_bucket_view path must not mis-handle dtype-specific sizes
- bucket reduction must preserve parameter ordering and views

**Step 2: Move hot loops out of Python**

The following belong in Cython, not Python:
- bucket byte-size accounting
- bucket flatten / offsets / slice bookkeeping
- flat-buffer rebuild / view reconstruction
- bucket-ready countdown bookkeeping

Keep Python responsible only for module-level orchestration and hook registration.

**Step 3: Verify DDP behavior**

Run:
```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda run -n candle python -m pytest tests/distributed/test_ddp_async_overlap.py tests/distributed/test_ddp_bucket_fastpath.py -v --tb=short
```

Expected:
- dtype-aware sizing tests pass
- bucket-view tests pass
- no regressions in comm-hook behavior

**Step 4: Commit**

```bash
git add setup.py src/candle/nn/parallel/distributed.py src/candle/distributed/_ddp_fastpath.pyx src/candle/distributed/_ddp_fastpath.pxd tests/distributed/test_ddp_async_overlap.py tests/distributed/test_ddp_bucket_fastpath.py
git commit -m "perf(ddp): move bucket bookkeeping to Cython"
```

---

### Task 5: Add overlap benchmarks and phase gate for P0 exit

**Files:**
- Create: `benchmarks/distributed_ddp_overlap.py`
- Modify: `tests/distributed/test_ddp_async_overlap.py`
- Modify: `docs/known-kernel-issues.md` if backend limitations appear

**Step 1: Add a minimal DDP overlap benchmark**

Benchmark at least:
- synchronous baseline
- async HCCL path
- bucketed path with comm hook

**Step 2: Define P0 exit metrics**

Document these as explicit success criteria in the benchmark script header:
- async path must produce identical numerical results
- async path must not regress throughput vs sync baseline
- overlap path should show measurable reduction in exposed communication time

**Step 3: Run benchmark and tests**

Run:
```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda run -n candle python benchmarks/distributed_ddp_overlap.py
source ~/miniconda3/etc/profile.d/conda.sh && conda run -n candle python -m pytest tests/distributed/test_ddp_async_overlap.py -v --tb=short
```

Expected:
- benchmark prints sync vs async measurements
- tests pass

**Step 4: Commit**

```bash
git add benchmarks/distributed_ddp_overlap.py tests/distributed/test_ddp_async_overlap.py docs/known-kernel-issues.md
git commit -m "bench(distributed): add ddp overlap benchmark and phase gate"
```

---

### Task 6: Promote composable FSDP to public API

**Files:**
- Modify: `src/candle/distributed/fsdp/__init__.py`
- Modify: `src/candle/distributed/_composable/__init__.py`
- Modify: `src/candle/distributed/__init__.py` if export glue is needed
- Create: `tests/distributed/test_fsdp_public_api.py`

**Step 1: Add failing public API tests**

Cover these imports and calls:
- `from candle.distributed.fsdp import fully_shard`
- `from candle.distributed._composable import fully_shard`
- `fully_shard(module, mesh=mesh)` returns a usable wrapped module

If the project wants to keep `FullyShardedDataParallel` unsupported, write tests that lock that policy clearly. If the project wants to expose it, add failing tests for the desired constructor behavior.

**Step 2: Remove the unconditional public `raise` path**

Current blockers live at:
- `src/candle/distributed/fsdp/__init__.py:4`
- `src/candle/distributed/fsdp/__init__.py:27`

Replace hard-fail exports with a deliberate public API decision.

**Step 3: Run tests**

Run:
```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda run -n candle python -m pytest tests/distributed/test_fsdp_public_api.py -v --tb=short
```

Expected:
- public `fully_shard` import path works
- no unconditional `RuntimeError` / `NotImplementedError`

**Step 4: Commit**

```bash
git add src/candle/distributed/fsdp/__init__.py src/candle/distributed/_composable/__init__.py src/candle/distributed/__init__.py tests/distributed/test_fsdp_public_api.py
git commit -m "feat(fsdp): expose composable fsdp through public api"
```

---

### Task 7: Move FSDP shard bookkeeping into Cython

**Files:**
- Create: `src/candle/distributed/_fsdp_fastpath.pyx`
- Create: `src/candle/distributed/_fsdp_fastpath.pxd`
- Modify: `src/candle/distributed/_composable/fsdp/_fsdp_param.py`
- Modify: `src/candle/distributed/_composable/fsdp/_fsdp_param_group.py`
- Modify: `src/candle/distributed/_composable/fsdp/_fsdp_state.py`
- Modify: `setup.py`
- Test: `tests/distributed/test_fsdp_public_api.py`
- Create: `tests/distributed/test_fsdp_shard_fastpath.py`

**Step 1: Add failing tests for shard layout invariants**

Cover:
- flat-buffer offsets are stable across runs
- unshard / reshard preserves tensor values and shapes
- `no_sync()` still accumulates correctly on unsharded params

**Step 2: Move hot bookkeeping to Cython**

The following must not remain pure Python on the main path:
- shard offset calculation
- flat-buffer pack / unpack loops
- owner / leaf-parameter mapping
- local shard writeback bookkeeping

**Step 3: Preserve Python orchestrator logic only where needed**

Keep Python for:
- hook registration
- public API surface
- error messaging

Move repeated per-parameter loops into Cython.

**Step 4: Run tests**

Run:
```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda run -n candle python -m pytest tests/distributed/test_fsdp_public_api.py tests/distributed/test_fsdp_shard_fastpath.py -v --tb=short
```

Expected:
- shard layout tests pass
- public FSDP entry points remain usable

**Step 5: Commit**

```bash
git add setup.py src/candle/distributed/_fsdp_fastpath.pyx src/candle/distributed/_fsdp_fastpath.pxd src/candle/distributed/_composable/fsdp/_fsdp_param.py src/candle/distributed/_composable/fsdp/_fsdp_param_group.py src/candle/distributed/_composable/fsdp/_fsdp_state.py tests/distributed/test_fsdp_public_api.py tests/distributed/test_fsdp_shard_fastpath.py
git commit -m "perf(fsdp): move shard bookkeeping to Cython"
```

---

### Task 8: Add NPU-first FSDP integration tests and phase gate for P1 exit

**Files:**
- Create: `tests/distributed/test_fsdp_npu_integration.py`
- Modify: `tests/distributed/test_fsdp_public_api.py`
- Optional: create `benchmarks/fsdp_npu_step.py`

**Step 1: Write NPU-first integration tests**

Cover at least:
- forward + backward + optimizer step
- `no_sync()` accumulation
- `summon_full_params()` round-trip
- `reshard_after_forward=True/False`
- mixed precision behavior if already supported

**Step 2: Run integration tests**

Run:
```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda run -n candle python -m pytest tests/distributed/test_fsdp_npu_integration.py -v --tb=short
```

Expected:
- at least one realistic NPU FSDP training step passes

**Step 3: Commit**

```bash
git add tests/distributed/test_fsdp_npu_integration.py tests/distributed/test_fsdp_public_api.py benchmarks/fsdp_npu_step.py
git commit -m "test(fsdp): add npu integration coverage"
```

---

### Task 9: Make distributed checkpoint shard-aware for DDP and FSDP

**Files:**
- Modify: `src/candle/distributed/checkpoint/state_dict.py`
- Modify: `src/candle/distributed/checkpoint/planner.py`
- Modify: `src/candle/distributed/checkpoint/storage.py`
- Modify: `src/candle/distributed/tensor/dtensor.py`
- Create: `tests/distributed/test_distributed_checkpoint.py`

**Step 1: Add failing checkpoint tests**

Cover:
- DDP save + load round-trip
- FSDP save + load round-trip
- shard-aware tensor metadata preservation
- optimizer state round-trip where already supported

**Step 2: Implement shard-aware checkpoint plumbing**

Make sure checkpoint planning understands:
- local shard offsets
- global tensor metadata
- rank ownership of chunks

**Step 3: Run tests**

Run:
```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda run -n candle python -m pytest tests/distributed/test_distributed_checkpoint.py -v --tb=short
```

Expected:
- DDP/FSDP checkpoint round-trips pass

**Step 4: Commit**

```bash
git add src/candle/distributed/checkpoint/state_dict.py src/candle/distributed/checkpoint/planner.py src/candle/distributed/checkpoint/storage.py src/candle/distributed/tensor/dtensor.py tests/distributed/test_distributed_checkpoint.py
git commit -m "feat(checkpoint): add shard-aware distributed checkpointing"
```

---

### Task 10: Move checkpoint metadata and chunk planning hot paths into Cython

**Files:**
- Create: `src/candle/distributed/_checkpoint_fastpath.pyx`
- Create: `src/candle/distributed/_checkpoint_fastpath.pxd`
- Modify: `src/candle/distributed/checkpoint/planner.py`
- Modify: `src/candle/distributed/checkpoint/metadata.py`
- Modify: `setup.py`
- Test: `tests/distributed/test_distributed_checkpoint.py`

**Step 1: Identify repeated per-chunk loops**

Move these into Cython:
- chunk offset calculation
- shard-to-global metadata packing
- repeated chunk-list generation for DTensor / FSDP state

**Step 2: Keep Python only for planner orchestration**

Planner policy can stay Python. The repeated metadata and chunk math should move to Cython.

**Step 3: Run tests**

Run:
```bash
python setup.py build_ext --inplace
source ~/miniconda3/etc/profile.d/conda.sh && conda run -n candle python -m pytest tests/distributed/test_distributed_checkpoint.py -v --tb=short
```

Expected:
- checkpoint tests still pass
- compiled fast path builds cleanly

**Step 4: Commit**

```bash
git add setup.py src/candle/distributed/_checkpoint_fastpath.pyx src/candle/distributed/_checkpoint_fastpath.pxd src/candle/distributed/checkpoint/planner.py src/candle/distributed/checkpoint/metadata.py tests/distributed/test_distributed_checkpoint.py
git commit -m "perf(checkpoint): move chunk planning to Cython"
```

---

### Task 11: Add failing CUDA/NCCL parity tests before implementation

**Files:**
- Create: `tests/cuda/test_nccl_collectives.py`
- Create: `tests/cuda/test_cuda_stream_api.py`
- Modify: `tests/cuda/conftest.py` if needed

**Step 1: Write failing NCCL collective tests**

Cover:
- `init_process_group(backend="nccl")`
- `all_reduce`
- `all_gather`
- `reduce_scatter`
- `barrier`

**Step 2: Write failing CUDA stream/event parity tests**

Cover:
- `current_stream()`
- `default_stream()`
- `stream()` context manager
- `wait_event()` / `wait_stream()`
- `Event.query()` / `elapsed_time()`

**Step 3: Run tests and confirm failure**

Run:
```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda run -n candle python -m pytest tests/cuda/test_nccl_collectives.py tests/cuda/test_cuda_stream_api.py -v --tb=short
```

Expected:
- NCCL backend unavailable
- missing CUDA stream/event APIs fail loudly

**Step 4: Commit**

```bash
git add tests/cuda/test_nccl_collectives.py tests/cuda/test_cuda_stream_api.py tests/cuda/conftest.py
git commit -m "test(cuda): add nccl and stream parity coverage"
```

---

### Task 12: Implement NCCL backend and close CUDA stream/event gaps in Cython

**Files:**
- Create: `src/candle/distributed/_c10d_nccl.pyx`
- Create: `src/candle/distributed/_c10d_nccl.pxd`
- Modify: `src/candle/distributed/_backend.py`
- Modify: `src/candle/distributed/__init__.py`
- Modify: `src/candle/cuda.py`
- Modify: `src/candle/_backends/cuda/runtime.py`
- Modify: `setup.py`
- Test: `tests/cuda/test_nccl_collectives.py`
- Test: `tests/cuda/test_cuda_stream_api.py`

**Step 1: Add NCCL process-group fast path in Cython**

Required hot paths:
- collective submit
- P2P submit where supported
- device / dtype extraction
- `Work` creation and completion bookkeeping

**Step 2: Close CUDA stream/event surface gaps**

Implement at least:
- `current_stream()`
- `default_stream()`
- `stream()` context manager
- `wait_event()`
- `wait_stream()`
- `Event.query()`
- `Event.elapsed_time()`

**Step 3: Run tests**

Run:
```bash
python setup.py build_ext --inplace
source ~/miniconda3/etc/profile.d/conda.sh && conda run -n candle python -m pytest tests/cuda/test_nccl_collectives.py tests/cuda/test_cuda_stream_api.py -v --tb=short
```

Expected:
- NCCL tests pass on CUDA hardware
- CUDA stream/event parity tests pass

**Step 4: Commit**

```bash
git add setup.py src/candle/distributed/_c10d_nccl.pyx src/candle/distributed/_c10d_nccl.pxd src/candle/distributed/_backend.py src/candle/distributed/__init__.py src/candle/cuda.py src/candle/_backends/cuda/runtime.py tests/cuda/test_nccl_collectives.py tests/cuda/test_cuda_stream_api.py
git commit -m "feat(cuda): add nccl backend and stream parity"
```

---

### Task 13: Make DTensor redistribution and tensor-parallel entry points real

**Files:**
- Modify: `src/candle/distributed/tensor/dtensor.py`
- Modify: `src/candle/distributed/device_mesh.py`
- Modify: `src/candle/distributed/tensor/placement.py`
- Modify: `src/candle/distributed/tensor/parallel.py`
- Create: `tests/distributed/test_dtensor_redistribute.py`
- Create: `tests/distributed/test_tensor_parallel.py`

**Step 1: Add failing DTensor redistribution tests**

Cover:
- local shard -> replicated transition
- replicated -> sharded transition
- shape / offset bookkeeping correctness

**Step 2: Add failing tensor-parallel API tests**

Cover:
- `parallelize_module()` no longer raising `NotImplementedError`
- at least one linear-layer or embedding split path works

**Step 3: Implement real redistribution semantics**

`DTensor` must stop being metadata-only on the main path. It needs actual placement-aware movement for supported layouts.

**Step 4: Run tests**

Run:
```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda run -n candle python -m pytest tests/distributed/test_dtensor_redistribute.py tests/distributed/test_tensor_parallel.py -v --tb=short
```

Expected:
- supported redistribution cases pass
- tensor-parallel API is no longer a stub

**Step 5: Commit**

```bash
git add src/candle/distributed/tensor/dtensor.py src/candle/distributed/device_mesh.py src/candle/distributed/tensor/placement.py src/candle/distributed/tensor/parallel.py tests/distributed/test_dtensor_redistribute.py tests/distributed/test_tensor_parallel.py
git commit -m "feat(dtensor): add redistribution and tensor parallel entry points"
```

---

### Task 14: Move DTensor / TP layout transforms into Cython

**Files:**
- Create: `src/candle/distributed/_dtensor_fastpath.pyx`
- Create: `src/candle/distributed/_dtensor_fastpath.pxd`
- Modify: `src/candle/distributed/tensor/dtensor.py`
- Modify: `src/candle/distributed/tensor/parallel.py`
- Modify: `setup.py`
- Test: `tests/distributed/test_dtensor_redistribute.py`
- Test: `tests/distributed/test_tensor_parallel.py`

**Step 1: Identify layout-transform hot loops**

Move these to Cython:
- shard offset calculations
- placement transform loops
- repeated gather/scatter index calculations
- all_to_all split bookkeeping for TP flows

**Step 2: Keep Python for orchestration only**

Python should choose the algorithm. Cython should do the repeated layout math and buffer bookkeeping.

**Step 3: Run tests**

Run:
```bash
python setup.py build_ext --inplace
source ~/miniconda3/etc/profile.d/conda.sh && conda run -n candle python -m pytest tests/distributed/test_dtensor_redistribute.py tests/distributed/test_tensor_parallel.py -v --tb=short
```

Expected:
- redistribution and TP tests keep passing
- compiled module builds cleanly

**Step 4: Commit**

```bash
git add setup.py src/candle/distributed/_dtensor_fastpath.pyx src/candle/distributed/_dtensor_fastpath.pxd src/candle/distributed/tensor/dtensor.py src/candle/distributed/tensor/parallel.py tests/distributed/test_dtensor_redistribute.py tests/distributed/test_tensor_parallel.py
git commit -m "perf(dtensor): move layout transforms to Cython"
```

---

### Task 15: Add minimal pipeline-parallel runtime only after TP foundations pass

**Files:**
- Create: `src/candle/distributed/pipeline.py`
- Create: `tests/distributed/test_pipeline_parallel.py`
- Optional: create `benchmarks/pipeline_parallel_step.py`

**Step 1: Add failing minimal pipeline tests**

Cover:
- stage execution order
- micro-batch scheduling
- activation handoff between two stages

**Step 2: Implement only the smallest viable pipeline runtime**

Do not add a full scheduler zoo. Start with:
- 2-stage pipeline
- fixed micro-batch schedule
- explicit stage boundary handoff

**Step 3: Run tests**

Run:
```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda run -n candle python -m pytest tests/distributed/test_pipeline_parallel.py -v --tb=short
```

Expected:
- minimal pipeline test passes

**Step 4: Commit**

```bash
git add src/candle/distributed/pipeline.py tests/distributed/test_pipeline_parallel.py benchmarks/pipeline_parallel_step.py
git commit -m "feat(pipeline): add minimal pipeline parallel runtime"
```

---

### Task 16: Finish low-priority compatibility and documentation cleanup

**Files:**
- Modify: `src/candle/__init__.py`
- Modify: `src/candle/backends/__init__.py`
- Modify or create: `src/candle/parallel.py`
- Create: `tests/cpu/test_threading_api.py`
- Create: `docs/parallel-status.md`

**Step 1: Add failing thread-control API tests**

Cover:
- `set_num_threads`
- `get_num_threads`
- `set_num_interop_threads`
- `get_num_interop_threads`
- `parallel_info`

**Step 2: Decide and document non-goals explicitly**

Examples:
- whether `nn.DataParallel` remains a stub permanently
- whether RPC / elastic / torchrun are deferred indefinitely

**Step 3: Publish a support matrix**

`docs/parallel-status.md` should list, per backend:
- DDP
- FSDP
- checkpoint
- stream/event
- NCCL/HCCL/Gloo
- DTensor/TP
- PP

**Step 4: Run tests**

Run:
```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda run -n candle python -m pytest tests/cpu/test_threading_api.py tests/distributed -v --tb=short -x
```

Expected:
- new compatibility tests pass
- no regression in distributed coverage

**Step 5: Commit**

```bash
git add src/candle/__init__.py src/candle/backends/__init__.py src/candle/parallel.py tests/cpu/test_threading_api.py docs/parallel-status.md
git commit -m "docs(parallel): publish final support matrix and cleanup apis"
```

---

## Dependency Graph

- **P0** depends on existing HCCL stream/event support and must finish before any serious CUDA/NCCL or public FSDP work.
- **P1** depends on P0 because public FSDP needs real async work semantics.
- **P2** depends on P1 because shard-aware checkpointing needs stable FSDP shard behavior.
- **P3** depends on P0 because CUDA/NCCL should reuse the unified async `Work` model, not invent its own.
- **P4** depends on P1 and P3 for stable collectives, shard semantics, and backend coverage.
- **P5** depends on P4 because pipeline / MoE scheduling needs working DTensor / mesh / collective foundations.
- **P6** can happen last and must not distract from training-critical phases.

---

## Phase Exit Criteria

### Exit P0 when:
- `Work.get_future()` is truly asynchronous on supported async backends.
- DDP overlap tests pass.
- Benchmarks show no regression vs synchronous baseline.

### Exit P1 when:
- Public `fully_shard()` works on NPU.
- FSDP integration tests pass for forward/backward/step.
- FSDP shard bookkeeping hot loops are on Cython fast path.

### Exit P2 when:
- DDP and FSDP checkpoints round-trip correctly.
- Shard-aware metadata / chunk planning are compiled fast paths.

### Exit P3 when:
- NCCL backend is real and tested.
- `torch.cuda`-style stream/event gaps listed in this plan are closed.

### Exit P4 when:
- DTensor redistribution works for supported layouts.
- `parallelize_module()` is no longer a stub for at least one model-family pattern.

### Exit P5 when:
- A minimal 2-stage pipeline path works.
- Pipeline tests pass without regressing DDP/FSDP/TP.

### Exit P6 when:
- Support matrix is documented.
- Thread-control compatibility decisions are implemented and tested.
- Deferred non-goals are written down explicitly.

---

## Suggested Execution Strategy

1. Implement **P0-P2** in one development stream focused on NPU/HCCL training realism.
2. Implement **P3** only after P0 is benchmark-stable.
3. Implement **P4-P5** only after public FSDP and checkpointing are solid.
4. Keep every hot-path migration behind targeted tests and a compiled extension build check.
5. Prefer frequent commits after each numbered task in this plan.
