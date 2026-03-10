from pathlib import Path


def test_distributed_tests_do_not_write_workers_to_fixed_tmp_paths():
    offenders = []
    for path in sorted(Path('tests/distributed').glob('test_*.py')):
        payload = path.read_text(encoding='utf-8')
        if '/tmp/' in payload and ('with open(worker_file, "w")' in payload or 'with open(worker, "w")' in payload):
            offenders.append(path.as_posix())

    assert not offenders, f'distributed worker scripts must not use fixed /tmp paths: {offenders}'
