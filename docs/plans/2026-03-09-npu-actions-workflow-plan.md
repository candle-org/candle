# NPU Actions Workflow Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a dedicated GitHub Actions NPU workflow that uses the local self-hosted Ascend runner with device groups `0-3`, `4-5`, and `6-7` matched to the current NPU and distributed test requirements.

**Architecture:** Keep the existing CPU/MPS CI workflow unchanged and introduce a separate NPU workflow. Split jobs by device-count needs: single-card NPU regression on `6-7`, 2-card distributed coverage on `4-5`, and the existing 4/8-card HCCL parameterized tests on `0-3` with an explicit second pass for the full 8-card cases.

**Tech Stack:** GitHub Actions, self-hosted Ascend runner, pytest, candle editable install.

---

### Task 1: Add a CI contract for the NPU workflow shape

**Files:**
- Create: `tests/contract/test_npu_workflow_contract.py`
- Test: `tests/contract/test_npu_workflow_contract.py`

**Step 1: Write the failing test**

Assert that `.github/workflows/npu.yaml` exists and encodes the three device groups plus their intended pytest entrypoints.

**Step 2: Run test to verify it fails**

Run: `pytest tests/contract/test_npu_workflow_contract.py -v --tb=short`
Expected: FAIL because `.github/workflows/npu.yaml` does not exist yet.

**Step 3: Write minimal implementation**

Create `.github/workflows/npu.yaml` with jobs for single-card NPU tests, 2-card distributed tests, and 4/8-card HCCL tests.

**Step 4: Run test to verify it passes**

Run: `pytest tests/contract/test_npu_workflow_contract.py -v --tb=short`
Expected: PASS.

### Task 2: Implement the workflow details

**Files:**
- Create: `.github/workflows/npu.yaml`
- Modify: `tests/contract/test_npu_workflow_contract.py`

**Step 1: Add runner and environment setup**

Use `runs-on: [self-hosted, ascend]`, install `requirements/requirements-test.txt`, install candle editable, and export device visibility per job.

**Step 2: Add test partitioning**

Use:
- `pytest tests/npu/ -v --tb=short` for single-card coverage on `6,7`
- `pytest tests/distributed/ -v --tb=short -k 'not all_to_all_single_async_unequal_multicard and not all_to_all_single_invalid_split_pairing_multicard and not all_to_all_single_split_numel_validation_multicard'` for 2-card coverage on `4,5`
- targeted pytest invocations for the 4-card and 8-card `all_to_all_single*multicard` cases on `0,1,2,3`

**Step 3: Verify contract**

Run: `pytest tests/contract/test_npu_workflow_contract.py -v --tb=short`
Expected: PASS.
