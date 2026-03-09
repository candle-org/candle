from pathlib import Path


def test_npu_workflow_partitions_suites_by_runner_pool():
    payload = Path('.github/workflows/npu.yaml').read_text(encoding='utf-8')

    assert 'name: NPU' in payload
    assert 'workflow_dispatch:' in payload
    assert 'CONDA_EXE: /home/lvyufeng/miniconda3/bin/conda' in payload
    assert 'create -y -p "$JOB_CONDA_ENV" python=3.11 pip' in payload
    assert 'run -p "$JOB_CONDA_ENV" python --version' in payload

    assert 'runs-on: [self-hosted, linux, ascend, 910a, npu-6-7]' in payload
    assert 'ASCEND_RT_VISIBLE_DEVICES: 6,7' in payload
    assert 'run -p "$JOB_CONDA_ENV" pytest tests/npu/ -v --tb=short' in payload

    assert 'runs-on: [self-hosted, linux, ascend, 910a, npu-4-5]' in payload
    assert 'ASCEND_RT_VISIBLE_DEVICES: 4,5' in payload
    assert 'run -p "$JOB_CONDA_ENV" pytest tests/distributed/ -v --tb=short -k "not all_to_all_single_async_unequal_multicard and not all_to_all_single_invalid_split_pairing_multicard and not all_to_all_single_split_numel_validation_multicard"' in payload
    assert "test_hccl_all_to_all_single_async_unequal_multicard[2-29714]" in payload
    assert "test_hccl_all_to_all_single_invalid_split_pairing_multicard[2-29715]" in payload
    assert "test_hccl_all_to_all_single_split_numel_validation_multicard[input_sum_mismatch-2-29716]" in payload
    assert "test_hccl_all_to_all_single_split_numel_validation_multicard[output_sum_mismatch-2-29716]" in payload

    assert 'runs-on: [self-hosted, linux, ascend, 910a, npu-0-3]' in payload
    assert 'ASCEND_RT_VISIBLE_DEVICES: 0,1,2,3' in payload
    assert "test_hccl_all_to_all_single_async_unequal_multicard[4-29724]" in payload
    assert "test_hccl_all_to_all_single_invalid_split_pairing_multicard[4-29725]" in payload
    assert "test_hccl_all_to_all_single_split_numel_validation_multicard[input_sum_mismatch-4-29726]" in payload
    assert "test_hccl_all_to_all_single_split_numel_validation_multicard[output_sum_mismatch-4-29726]" in payload
