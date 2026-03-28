import argparse
import importlib.util
from pathlib import Path


SPEC = importlib.util.spec_from_file_location(
    "openi_ci",
    str(Path(__file__).resolve().parents[1] / "scripts" / "openi_ci.py"),
)
openi_ci = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(openi_ci)


class DummySession:
    pass


def test_runner_api_token_prefers_openi_runner_token(monkeypatch):
    monkeypatch.setenv("OPENI_RUNNER_TOKEN", "runner-token")
    monkeypatch.setenv("GITHUB_TOKEN", "github-token")

    token = openi_ci._get_runner_api_token()

    assert token == "runner-token"


def test_runner_api_token_falls_back_to_github_token(monkeypatch):
    monkeypatch.delenv("OPENI_RUNNER_TOKEN", raising=False)
    monkeypatch.setenv("GITHUB_TOKEN", "github-token")

    token = openi_ci._get_runner_api_token()

    assert token == "github-token"


def test_ensure_task_by_id_restarts_stopped_task(monkeypatch, tmp_path):
    monkeypatch.setattr(openi_ci, "ARTIFACT_ROOT", tmp_path)
    monkeypatch.setattr(openi_ci, "REMOTE_ARTIFACTS", tmp_path / "remote")
    monkeypatch.setattr(openi_ci, "_load_session_config", lambda: {"cookie": "", "csrf": ""})
    monkeypatch.setattr(openi_ci, "_make_requests_session", lambda _cfg: DummySession())

    calls = []

    def fake_api_call(_session, method, path, **_kwargs):
        calls.append((method, path))
        assert method == "get"
        assert path == "/api/v1/ai_task/brief?id=123"
        return {"data": {"id": 123, "status": "STOPPED"}}

    restarted = {"called": False}

    def fake_restart(args):
        restarted["called"] = True
        assert args.task_id == "123"
        return 0

    monkeypatch.setattr(openi_ci, "_api_call", fake_api_call)
    monkeypatch.setattr(openi_ci, "_handle_restart_task", fake_restart)

    args = argparse.Namespace(task_id="123")
    result = openi_ci._handle_ensure_task_by_id(args)

    assert result == 0
    assert restarted["called"] is True
    saved = openi_ci._load_json_state("task")
    assert saved["id"] == 123
    assert saved["status"] == "STOPPED"
    assert calls == [("get", "/api/v1/ai_task/brief?id=123")]


def test_ensure_task_by_id_falls_back_to_my_list_on_403(monkeypatch, tmp_path):
    monkeypatch.setattr(openi_ci, "ARTIFACT_ROOT", tmp_path)
    monkeypatch.setattr(openi_ci, "REMOTE_ARTIFACTS", tmp_path / "remote")
    monkeypatch.setattr(openi_ci, "_load_session_config", lambda: {"cookie": "", "csrf": ""})
    monkeypatch.setattr(openi_ci, "_make_requests_session", lambda _cfg: DummySession())

    def fake_api_call(_session, method, path, **_kwargs):
        assert method == "get"
        raise openi_ci.requests.exceptions.HTTPError(response=type("Resp", (), {"status_code": 403})())

    matched = {"id": 999, "status": "RUNNING", "cluster": "C2Net", "compute_source": "NPU"}

    monkeypatch.setattr(openi_ci, "_api_call", fake_api_call)
    monkeypatch.setattr(openi_ci, "_find_matching_task_in_my_list", lambda _session, _args: matched)

    args = argparse.Namespace(task_id="old", spec_id="328", cluster="C2Net", compute_source="NPU", image_id="", image_name="")
    result = openi_ci._handle_ensure_task_by_id(args)

    assert result == 0
    saved = openi_ci._load_json_state("task")
    assert saved["id"] == 999
    assert saved["status"] == "RUNNING"


def test_ensure_task_by_id_waits_for_stopping_then_restarts(monkeypatch, tmp_path):
    monkeypatch.setattr(openi_ci, "ARTIFACT_ROOT", tmp_path)
    monkeypatch.setattr(openi_ci, "REMOTE_ARTIFACTS", tmp_path / "remote")
    monkeypatch.setattr(openi_ci, "_load_session_config", lambda: {"cookie": "", "csrf": ""})
    monkeypatch.setattr(openi_ci, "_make_requests_session", lambda _cfg: DummySession())

    def fake_api_call(_session, method, path, **_kwargs):
        assert method == "get"
        assert path == "/api/v1/ai_task/brief?id=123"
        return {"data": {"id": 123, "status": "STOPPING"}}

    waited = {"called": False}
    restarted = {"called": False}

    def fake_wait(_session, task_id, timeout=300):
        waited["called"] = True
        assert str(task_id) == "123"
        return None

    def fake_restart(args):
        restarted["called"] = True
        assert args.task_id == "123"
        return 0

    monkeypatch.setattr(openi_ci, "_api_call", fake_api_call)
    monkeypatch.setattr(openi_ci, "_wait_task_stopped", fake_wait)
    monkeypatch.setattr(openi_ci, "_handle_restart_task", fake_restart)

    args = argparse.Namespace(task_id="123")
    result = openi_ci._handle_ensure_task_by_id(args)

    assert result == 0
    assert waited["called"] is True
    assert restarted["called"] is True


def test_cleanup_task_only_stops_not_deletes(monkeypatch, tmp_path):
    monkeypatch.setattr(openi_ci, "ARTIFACT_ROOT", tmp_path)
    monkeypatch.setattr(openi_ci, "REMOTE_ARTIFACTS", tmp_path / "remote")
    monkeypatch.setattr(openi_ci, "_load_session_config", lambda: {"cookie": "", "csrf": ""})
    monkeypatch.setattr(openi_ci, "_make_requests_session", lambda _cfg: DummySession())
    openi_ci._save_json_state("task", {"id": 456})
    monkeypatch.setattr(openi_ci, "_wait_task_stopped", lambda *_args, **_kwargs: None)

    calls = []

    def fake_api_call(_session, method, path, **_kwargs):
        calls.append((method, path))
        return {"data": {}}

    monkeypatch.setattr(openi_ci, "_api_call", fake_api_call)

    result = openi_ci._handle_cleanup_task(argparse.Namespace(task_id=""))

    assert result == 0
    assert calls == [("post", "/api/v1/ai_task/stop?id=456")]


def test_cleanup_task_can_stop_by_explicit_task_id(monkeypatch, tmp_path):
    monkeypatch.setattr(openi_ci, "ARTIFACT_ROOT", tmp_path)
    monkeypatch.setattr(openi_ci, "REMOTE_ARTIFACTS", tmp_path / "remote")
    monkeypatch.setattr(openi_ci, "_load_session_config", lambda: {"cookie": "", "csrf": ""})
    monkeypatch.setattr(openi_ci, "_make_requests_session", lambda _cfg: DummySession())
    monkeypatch.setattr(openi_ci, "_wait_task_stopped", lambda *_args, **_kwargs: None)

    calls = []

    def fake_api_call(_session, method, path, **_kwargs):
        calls.append((method, path))
        return {"data": {}}

    monkeypatch.setattr(openi_ci, "_api_call", fake_api_call)

    result = openi_ci._handle_cleanup_task(
        argparse.Namespace(task_id="789", spec_id=None, cluster=None, compute_source=None, image_id="", image_name="")
    )

    assert result == 0
    assert calls == [("post", "/api/v1/ai_task/stop?id=789")]


def test_cleanup_task_falls_back_to_my_list_on_403(monkeypatch, tmp_path):
    monkeypatch.setattr(openi_ci, "ARTIFACT_ROOT", tmp_path)
    monkeypatch.setattr(openi_ci, "REMOTE_ARTIFACTS", tmp_path / "remote")
    monkeypatch.setattr(openi_ci, "_load_session_config", lambda: {"cookie": "", "csrf": ""})
    monkeypatch.setattr(openi_ci, "_make_requests_session", lambda _cfg: DummySession())
    monkeypatch.setattr(openi_ci, "_wait_task_stopped", lambda *_args, **_kwargs: None)

    calls = []

    def fake_api_call(_session, method, path, **_kwargs):
        calls.append((method, path))
        if path == "/api/v1/ai_task/stop?id=old":
            raise openi_ci.requests.exceptions.HTTPError(response=type("Resp", (), {"status_code": 403})())
        return {"data": {}}

    monkeypatch.setattr(openi_ci, "_api_call", fake_api_call)
    monkeypatch.setattr(openi_ci, "_find_matching_task_in_my_list", lambda _session, _args: {"id": 1001})

    args = argparse.Namespace(task_id="old", spec_id="340", cluster="C2Net", compute_source="NPU", image_id="", image_name="")
    result = openi_ci._handle_cleanup_task(args)

    assert result == 0
    assert calls == [
        ("post", "/api/v1/ai_task/stop?id=old"),
        ("post", "/api/v1/ai_task/stop?id=1001"),
    ]


def test_cleanup_task_prefers_saved_task_when_no_explicit_id(monkeypatch, tmp_path):
    monkeypatch.setattr(openi_ci, "ARTIFACT_ROOT", tmp_path)
    monkeypatch.setattr(openi_ci, "REMOTE_ARTIFACTS", tmp_path / "remote")
    monkeypatch.setattr(openi_ci, "_load_session_config", lambda: {"cookie": "", "csrf": ""})
    monkeypatch.setattr(openi_ci, "_make_requests_session", lambda _cfg: DummySession())
    monkeypatch.setattr(openi_ci, "_wait_task_stopped", lambda *_args, **_kwargs: None)
    openi_ci._save_json_state("task", {"id": 456})

    calls = []

    def fake_api_call(_session, method, path, **_kwargs):
        calls.append((method, path))
        return {"data": {}}

    monkeypatch.setattr(openi_ci, "_api_call", fake_api_call)

    result = openi_ci._handle_cleanup_task(
        argparse.Namespace(task_id="", spec_id=None, cluster=None, compute_source=None, image_id="", image_name="")
    )

    assert result == 0
    assert calls == [("post", "/api/v1/ai_task/stop?id=456")]


def test_cleanup_task_returns_when_no_state_and_no_task_id(monkeypatch, tmp_path):
    monkeypatch.setattr(openi_ci, "ARTIFACT_ROOT", tmp_path)
    monkeypatch.setattr(openi_ci, "REMOTE_ARTIFACTS", tmp_path / "remote")
    monkeypatch.setattr(openi_ci, "_load_session_config", lambda: {"cookie": "", "csrf": ""})
    monkeypatch.setattr(openi_ci, "_make_requests_session", lambda _cfg: DummySession())

    called = {"value": False}

    def fake_api_call(*_args, **_kwargs):
        called["value"] = True
        return {"data": {}}

    monkeypatch.setattr(openi_ci, "_api_call", fake_api_call)

    result = openi_ci._handle_cleanup_task(
        argparse.Namespace(task_id="", spec_id=None, cluster=None, compute_source=None, image_id="", image_name="")
    )

    assert result == 0
    assert called["value"] is False


def test_cleanup_task_waits_until_task_stops(monkeypatch, tmp_path):
    monkeypatch.setattr(openi_ci, "ARTIFACT_ROOT", tmp_path)
    monkeypatch.setattr(openi_ci, "REMOTE_ARTIFACTS", tmp_path / "remote")
    monkeypatch.setattr(openi_ci, "_load_session_config", lambda: {"cookie": "", "csrf": ""})
    monkeypatch.setattr(openi_ci, "_make_requests_session", lambda _cfg: DummySession())
    monkeypatch.setattr(openi_ci.time, "sleep", lambda _n: None)

    calls = []
    statuses = iter([
        {"data": {}},
        {"data": {"status": "STOPPING"}},
        {"data": {"status": "STOPPED"}},
    ])

    def fake_api_call(_session, method, path, **_kwargs):
        calls.append((method, path))
        if method == "post":
            return next(statuses)
        if path == "/api/v1/ai_task/brief?id=789":
            return next(statuses)
        raise AssertionError((method, path))

    monkeypatch.setattr(openi_ci, "_api_call", fake_api_call)

    result = openi_ci._handle_cleanup_task(
        argparse.Namespace(task_id="789", spec_id=None, cluster=None, compute_source=None, image_id="", image_name="")
    )

    assert result == 0
    assert calls == [
        ("post", "/api/v1/ai_task/stop?id=789"),
        ("get", "/api/v1/ai_task/brief?id=789"),
        ("get", "/api/v1/ai_task/brief?id=789"),
    ]
