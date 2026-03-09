from pathlib import Path


def test_ci_runs_cpu_and_contract_tests():
    payload = Path('.github/workflows/ci.yaml').read_text(encoding='utf-8')
    assert 'pytest tests/cpu/ tests/contract/' in payload
