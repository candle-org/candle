import importlib.util
import json
from pathlib import Path

import pytest
import requests

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / ".github/scripts/openi_ci.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("openi_ci", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def openi_ci_module():
    assert SCRIPT_PATH.exists(), f"Expected script at {SCRIPT_PATH}"
    return _load_module()


def test_script_exists():
    assert SCRIPT_PATH.exists()


def test_parse_jupyter_url_strips_lab_suffix(openi_ci_module):
    base, token = openi_ci_module._parse_jupyter_url(
        "https://host.example/lab?token=abc123"
    )
    assert base == "https://host.example"
    assert token == "abc123"


def test_parse_jupyter_url_strips_lab_trailing_slash(openi_ci_module):
    base, token = openi_ci_module._parse_jupyter_url(
        "https://host.example/lab/?token=xyz"
    )
    assert base == "https://host.example"
    assert token == "xyz"


def test_state_paths_live_under_openi_artifact_root(openi_ci_module, tmp_path, monkeypatch):
    monkeypatch.setattr(openi_ci_module, "ARTIFACT_ROOT", tmp_path / ".artifacts" / "openi-910a")
    task_path = openi_ci_module._state_path("task")
    run_path = openi_ci_module._state_path("run")
    assert str(task_path).startswith(str(openi_ci_module.ARTIFACT_ROOT))
    assert str(run_path).startswith(str(openi_ci_module.ARTIFACT_ROOT))
    assert task_path != run_path


def test_remote_exec_command_uses_bash_lc(openi_ci_module):
    command = openi_ci_module._remote_exec_cmd("echo hello")
    assert command[:2] == ["bash", "-lc"]
    assert command[2] == "echo hello"


def test_remote_script_exports_required_python_env(openi_ci_module):
    script = openi_ci_module._build_remote_test_script(
        repo_url="https://github.com/candle-org/candle.git",
        ref="main",
    )
    assert "export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1" in script
    assert "export PYTHONNOUSERSITE=1" in script


def test_remote_prepare_script_disables_git_proxy_explicitly(openi_ci_module):
    script = openi_ci_module._build_remote_prepare_script(
        repo_url="https://github.com/candle-org/candle.git",
        ref="main",
        remote_repo="/tmp/repo",
    )
    assert "unset http_proxy https_proxy ftp_proxy no_proxy" in script
    assert "unset HTTP_PROXY HTTPS_PROXY FTP_PROXY NO_PROXY" in script
    assert "git -c http.proxy= -c https.proxy= clone https://github.com/candle-org/candle.git /tmp/repo" in script
    assert "git -c http.proxy= -c https.proxy= fetch --all --tags" in script


@pytest.mark.parametrize(
    "artifact_name",
    ["pytest.log", "junit.xml", "summary.json", "remote_env.txt"],
)
def test_remote_script_writes_expected_artifacts(openi_ci_module, artifact_name):
    script = openi_ci_module._build_remote_test_script(
        repo_url="https://github.com/candle-org/candle.git",
        ref="main",
    )
    assert artifact_name in script


def test_save_and_load_json_state(openi_ci_module, tmp_path, monkeypatch):
    monkeypatch.setattr(openi_ci_module, "ARTIFACT_ROOT", tmp_path / ".artifacts" / "openi-910a")
    payload = {"id": 123, "status": "RUNNING"}
    openi_ci_module._save_json_state("task", payload)
    assert openi_ci_module._load_json_state("task") == payload


@pytest.mark.parametrize("state_name", ["session", "task", "kernel", "run"])
def test_named_state_files_round_trip(openi_ci_module, tmp_path, monkeypatch, state_name):
    monkeypatch.setattr(openi_ci_module, "ARTIFACT_ROOT", tmp_path / ".artifacts" / "openi-910a")
    payload = {"name": state_name}
    openi_ci_module._save_json_state(state_name, payload)
    assert openi_ci_module._state_path(state_name).name == f"{state_name}.json"
    assert openi_ci_module._load_json_state(state_name) == payload


def test_env_session_overrides_stale_session_file(openi_ci_module, tmp_path, monkeypatch):
    monkeypatch.setattr(openi_ci_module, "ARTIFACT_ROOT", tmp_path / ".artifacts" / "openi-910a")
    openi_ci_module._save_json_state("session", {"cookie": "stale", "csrf": "stale"})
    monkeypatch.setenv("OPENI_COOKIE", "fresh-cookie")
    monkeypatch.setenv("OPENI_CSRF", "fresh-csrf")
    session = openi_ci_module._load_session_config()
    assert session["cookie"] == "fresh-cookie"
    assert session["csrf"] == "fresh-csrf"
    assert session["source"] == "env"


def test_loads_session_from_persisted_state_when_env_missing(openi_ci_module, tmp_path, monkeypatch):
    monkeypatch.setattr(openi_ci_module, "ARTIFACT_ROOT", tmp_path / ".artifacts" / "openi-910a")
    openi_ci_module._save_json_state(
        "session",
        {"cookie": "persisted-cookie", "csrf": "persisted-csrf", "source": "session-file"},
    )
    monkeypatch.delenv("OPENI_COOKIE", raising=False)
    monkeypatch.delenv("OPENI_CSRF", raising=False)
    monkeypatch.delenv("OPENI_USER_NAME", raising=False)
    monkeypatch.delenv("OPENI_USER_PASSWORD", raising=False)
    session = openi_ci_module._load_session_config()
    assert session["cookie"] == "persisted-cookie"
    assert session["csrf"] == "persisted-csrf"
    assert session["source"] == "session-file"


def test_unsupported_auth_mode_fails_clearly(openi_ci_module, tmp_path, monkeypatch):
    monkeypatch.setattr(openi_ci_module, "ARTIFACT_ROOT", tmp_path / ".artifacts" / "openi-910a")
    monkeypatch.delenv("OPENI_COOKIE", raising=False)
    monkeypatch.delenv("OPENI_CSRF", raising=False)
    monkeypatch.setenv("OPENI_USER_NAME", "demo")
    monkeypatch.setenv("OPENI_USER_PASSWORD", "secret")
    monkeypatch.setattr(
        openi_ci_module,
        "_login_with_password",
        lambda _username, _password: (_ for _ in ()).throw(openi_ci_module.UnsupportedAuthError("username/password login is not implemented")),
    )
    with pytest.raises(openi_ci_module.UnsupportedAuthError, match="not implemented"):
        openi_ci_module._load_session_config()


def test_load_session_uses_password_login_when_supported(openi_ci_module, tmp_path, monkeypatch):
    monkeypatch.setattr(openi_ci_module, "ARTIFACT_ROOT", tmp_path / ".artifacts" / "openi-910a")
    monkeypatch.delenv("OPENI_COOKIE", raising=False)
    monkeypatch.delenv("OPENI_CSRF", raising=False)
    monkeypatch.setenv("OPENI_USER_NAME", "demo")
    monkeypatch.setenv("OPENI_USER_PASSWORD", "secret")
    monkeypatch.setattr(
        openi_ci_module,
        "_login_with_password",
        lambda username, password: {"cookie": f"cookie-for-{username}", "csrf": "csrf-token", "source": "password-login"},
    )
    session = openi_ci_module._load_session_config()
    assert session["cookie"] == "cookie-for-demo"
    assert session["csrf"] == "csrf-token"
    assert session["source"] == "password-login"


def test_login_with_password_uses_post_login_csrf_cookie(openi_ci_module, monkeypatch):
    login_page = '''
    <form class="ui form" action="/user/login" method="post" onclick="return checkform();">
      <input type="hidden" name="_csrf" value="csrf-login-page">
      <input id="user_name" name="user_name" value="" placeholder="Username/Email/Phone number" autofocus required>
      <input id="input_password" type="password" value="" placeholder="Password" autocomplete="off" required>
      <input id="password" name="password" type="hidden" value="">
      <input id="remember" name="remember" type="checkbox">
    </form>
    '''

    class _LoginResponse:
        def __init__(self, text="", headers=None, cookies=None, history=None, url="https://openi.pcl.ac.cn/user/login"):
            self.text = text
            self.headers = headers or {}
            self.cookies = cookies or requests.cookies.RequestsCookieJar()
            self.history = history or []
            self.url = url

        def raise_for_status(self):
            return None

    class _PasswordLoginSession:
        def __init__(self):
            self.headers = {}
            self.calls = []
            self.cookies = requests.cookies.RequestsCookieJar()

        def get(self, url, **kwargs):
            self.calls.append(("get", url, None, kwargs))
            return _LoginResponse(text=login_page)

        def post(self, url, data=None, allow_redirects=True, **kwargs):
            self.calls.append(("post", url, data, kwargs))
            self.cookies.set("i_like_openi", "cookie-value", domain="openi.pcl.ac.cn")
            self.cookies.set("_csrf", "csrf-post-login", domain="openi.pcl.ac.cn")
            return _LoginResponse(headers={"Location": "/dashboard"}, cookies=self.cookies, history=[object()], url="https://openi.pcl.ac.cn/dashboard")

    monkeypatch.setattr(openi_ci_module, "_encrypt_password", lambda password, public_key: f"enc::{password}::{public_key[:8]}")
    monkeypatch.setattr(openi_ci_module.requests, "Session", _PasswordLoginSession)
    session = openi_ci_module._login_with_password("demo", "secret")
    assert session["source"] == "password-login"
    assert session["csrf"] == "csrf-post-login"
    assert "i_like_openi=cookie-value" in session["cookie"]


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class _FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []
        self.headers = {}

    def post(self, url, json=None, **kwargs):
        self.calls.append(("post", url, json, kwargs))
        if not self.responses:
            raise AssertionError(f"Unexpected POST to {url}: no fake responses left")
        return _FakeResponse(self.responses.pop(0))

    def get(self, url, **kwargs):
        self.calls.append(("get", url, None, kwargs))
        if not self.responses:
            raise AssertionError(f"Unexpected GET to {url}: no fake responses left")
        return _FakeResponse(self.responses.pop(0))


class _FakeJupyterClient:
    instances = []

    def __init__(self, base_url, token, kernel_id=None, session=None):
        self.base_url = base_url
        self.token = token
        self.kernel_id = kernel_id
        self.session = session
        self.commands = []
        self.downloads = []
        _FakeJupyterClient.instances.append(self)

    def create_kernel(self):
        self.kernel_id = "kernel-123"
        return self.kernel_id

    def execute_shell(self, command, timeout=600):
        self.commands.append((command, timeout))
        return {"output": "remote ok\n", "exit_code": 0}

    def download_file(self, remote_path, local_path):
        self.downloads.append((remote_path, str(local_path)))
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        Path(local_path).write_text(f"downloaded {remote_path}", encoding="utf-8")


@pytest.fixture(autouse=True)
def _reset_fake_jupyter_client_instances():
    _FakeJupyterClient.instances = []
    yield
    _FakeJupyterClient.instances = []


def test_create_task_payload_uses_numeric_spec_and_network_values(openi_ci_module):
    args = openi_ci_module._build_parser().parse_args([
        "create-task",
        "--repo-url", "https://github.com/candle-org/candle.git",
        "--ref", "main",
        "--image-id", "image-1",
        "--image-name", "image-name-1",
        "--spec-id", "328",
        "--cluster", "C2Net",
        "--compute-source", "NPU",
        "--has-internet", "2",
    ])
    payload = openi_ci_module._build_create_payload(args)
    assert payload["spec_id"] == 328
    assert payload["has_internet"] == 2


def test_create_task_quota_limit_raises_instead_of_auto_fallback(openi_ci_module, tmp_path, monkeypatch):
    monkeypatch.setattr(openi_ci_module, "ARTIFACT_ROOT", tmp_path / ".artifacts" / "openi-910a")
    monkeypatch.setattr(openi_ci_module, "_load_session_config", lambda: {"cookie": "c", "csrf": "x", "source": "env"})

    class _QuotaSession(_FakeSession):
        def post(self, url, json=None, **kwargs):
            self.calls.append(("post", url, json, kwargs))
            return _FakeResponse({"code": 2004, "msg": "quota reached", "data": None})

    fake_session = _QuotaSession([])
    monkeypatch.setattr(openi_ci_module, "_make_requests_session", lambda session_cfg: fake_session)

    args = openi_ci_module._build_parser().parse_args([
        "create-task",
        "--repo-url", "https://github.com/candle-org/candle.git",
        "--ref", "main",
        "--image-id", "image-1",
        "--image-name", "image-name-1",
        "--spec-id", "328",
        "--cluster", "C2Net",
        "--compute-source", "NPU",
        "--has-internet", "2",
    ])

    with pytest.raises(openi_ci_module.OpenIAPIError, match="2004"):
        openi_ci_module._handle_create_task(args)



def test_ensure_task_creates_when_no_prior_task_state(openi_ci_module, tmp_path, monkeypatch):
    monkeypatch.setattr(openi_ci_module, "ARTIFACT_ROOT", tmp_path / ".artifacts" / "openi-910a")
    monkeypatch.setattr(openi_ci_module, "_load_session_config", lambda: {"cookie": "c", "csrf": "x", "source": "env"})

    fake_session = _FakeSession([
        {"code": 0, "data": {"id": 111, "status": "WAITING"}},
    ])
    monkeypatch.setattr(openi_ci_module, "_make_requests_session", lambda session_cfg: fake_session)

    def _fake_create_task(_args):
        openi_ci_module._save_json_state("task", {"id": 111, "status": "WAITING", "repo_url": _args.repo_url, "ref": _args.ref})
        return 0

    monkeypatch.setattr(openi_ci_module, "_handle_create_task", _fake_create_task)

    args = openi_ci_module._build_parser().parse_args([
        "ensure-task",
        "--repo-url", "https://github.com/candle-org/candle.git",
        "--ref", "main",
        "--image-id", "image-1",
        "--image-name", "image-name-1",
        "--spec-id", "328",
        "--cluster", "C2Net",
        "--compute-source", "NPU",
        "--has-internet", "2",
    ])

    assert openi_ci_module._handle_ensure_task(args) == 0
    saved = openi_ci_module._load_json_state("task")
    assert saved["id"] == 111
    assert saved["status"] == "WAITING"
    assert saved["repo_url"] == "https://github.com/candle-org/candle.git"
    assert saved["ref"] == "main"
    assert fake_session.calls == []



def test_ensure_task_falls_back_to_create_when_saved_task_brief_is_forbidden(openi_ci_module, tmp_path, monkeypatch):
    monkeypatch.setattr(openi_ci_module, "ARTIFACT_ROOT", tmp_path / ".artifacts" / "openi-910a")
    openi_ci_module._save_json_state("task", {"id": 999, "status": "STOPPED"})
    monkeypatch.setattr(openi_ci_module, "_load_session_config", lambda: {"cookie": "c", "csrf": "x", "source": "env"})

    def _api_call(_session, method, path, **kwargs):
        if path.startswith("/api/v1/ai_task/brief?id=999"):
            response = requests.Response()
            response.status_code = 403
            response.url = path
            raise requests.exceptions.HTTPError(response=response)
        if path == "/api/v1/ai_task/my_list":
            return {"data": {"list": []}}
        raise AssertionError(f"Unexpected api call: {method} {path}")

    monkeypatch.setattr(openi_ci_module, "_api_call", _api_call)
    monkeypatch.setattr(openi_ci_module, "_make_requests_session", lambda session_cfg: object())

    def _fake_create_task(_args):
        openi_ci_module._save_json_state("task", {"id": 111, "status": "WAITING", "repo_url": _args.repo_url, "ref": _args.ref})
        return 0

    monkeypatch.setattr(openi_ci_module, "_handle_create_task", _fake_create_task)

    args = openi_ci_module._build_parser().parse_args([
        "ensure-task",
        "--repo-url", "https://github.com/candle-org/candle.git",
        "--ref", "main",
        "--image-id", "image-1",
        "--image-name", "image-name-1",
        "--spec-id", "328",
        "--cluster", "C2Net",
        "--compute-source", "NPU",
        "--has-internet", "2",
    ])

    assert openi_ci_module._handle_ensure_task(args) == 0
    saved = openi_ci_module._load_json_state("task")
    assert saved["id"] == 111
    assert saved["status"] == "WAITING"



def test_ensure_task_reuses_recent_running_task_discovered_from_my_list(openi_ci_module, tmp_path, monkeypatch):
    monkeypatch.setattr(openi_ci_module, "ARTIFACT_ROOT", tmp_path / ".artifacts" / "openi-910a")
    openi_ci_module._save_json_state("task", {"id": 999, "status": "STOPPED"})
    monkeypatch.setattr(openi_ci_module, "_load_session_config", lambda: {"cookie": "c", "csrf": "x", "source": "env"})
    monkeypatch.setattr(openi_ci_module.time, "time", lambda: 1060)

    def _api_call(_session, method, path, **kwargs):
        if path.startswith("/api/v1/ai_task/brief?id=999"):
            response = requests.Response()
            response.status_code = 403
            response.url = path
            raise requests.exceptions.HTTPError(response=response)
        if path == "/api/v1/ai_task/my_list":
            return {"data": {"tasks": [
                {"task": {"id": 202, "status": "RUNNING", "start_time": 1000, "end_time": 0, "image_id": "image-1", "image_name": "image-name-1", "cluster": "C2Net", "compute_source": "NPU"}}
            ]}}
        raise AssertionError(f"Unexpected api call: {method} {path}")

    monkeypatch.setattr(openi_ci_module, "_api_call", _api_call)
    monkeypatch.setattr(openi_ci_module, "_make_requests_session", lambda session_cfg: object())

    args = openi_ci_module._build_parser().parse_args([
        "ensure-task",
        "--repo-url", "https://github.com/candle-org/candle.git",
        "--ref", "main",
        "--image-id", "image-1",
        "--image-name", "image-name-1",
        "--spec-id", "328",
        "--cluster", "C2Net",
        "--compute-source", "NPU",
        "--has-internet", "2",
    ])

    assert openi_ci_module._handle_ensure_task(args) == 0
    saved = openi_ci_module._load_json_state("task")
    assert saved["id"] == 202
    assert saved["status"] == "RUNNING"
    assert saved["repo_url"] == "https://github.com/candle-org/candle.git"
    assert saved["ref"] == "main"



def test_ensure_task_restarts_discovered_task_after_reuse_timeout(openi_ci_module, tmp_path, monkeypatch):
    monkeypatch.setattr(openi_ci_module, "ARTIFACT_ROOT", tmp_path / ".artifacts" / "openi-910a")
    openi_ci_module._save_json_state("task", {"id": 999, "status": "STOPPED"})
    monkeypatch.setattr(openi_ci_module, "_load_session_config", lambda: {"cookie": "c", "csrf": "x", "source": "env"})
    monkeypatch.setattr(openi_ci_module.time, "time", lambda: 1000 + 12601)

    def _api_call(_session, method, path, **kwargs):
        if path.startswith("/api/v1/ai_task/brief?id=999"):
            response = requests.Response()
            response.status_code = 403
            response.url = path
            raise requests.exceptions.HTTPError(response=response)
        if path == "/api/v1/ai_task/my_list":
            return {"data": {"tasks": [
                {"task": {"id": 202, "status": "RUNNING", "start_time": 1000, "end_time": 0, "image_id": "image-1", "image_name": "image-name-1", "cluster": "C2Net", "compute_source": "NPU"}}
            ]}}
        raise AssertionError(f"Unexpected api call: {method} {path}")

    monkeypatch.setattr(openi_ci_module, "_api_call", _api_call)
    monkeypatch.setattr(openi_ci_module, "_make_requests_session", lambda session_cfg: object())

    def _fake_restart_task(_args):
        openi_ci_module._save_json_state("task", {"id": 303, "status": "WAITING"})
        return 0

    monkeypatch.setattr(openi_ci_module, "_handle_restart_task", _fake_restart_task)

    args = openi_ci_module._build_parser().parse_args([
        "ensure-task",
        "--repo-url", "https://github.com/candle-org/candle.git",
        "--ref", "main",
        "--image-id", "image-1",
        "--image-name", "image-name-1",
        "--spec-id", "328",
        "--cluster", "C2Net",
        "--compute-source", "NPU",
        "--has-internet", "2",
    ])

    assert openi_ci_module._handle_ensure_task(args) == 0
    saved = openi_ci_module._load_json_state("task")
    assert saved["id"] == 303
    assert saved["status"] == "WAITING"
    assert saved["repo_url"] == "https://github.com/candle-org/candle.git"
    assert saved["ref"] == "main"



def test_restart_task_stops_then_restarts_and_persists_new_task(openi_ci_module, tmp_path, monkeypatch):
    monkeypatch.setattr(openi_ci_module, "ARTIFACT_ROOT", tmp_path / ".artifacts" / "openi-910a")
    openi_ci_module._save_json_state("task", {"id": 202, "status": "RUNNING"})
    monkeypatch.setattr(openi_ci_module, "_load_session_config", lambda: {"cookie": "c", "csrf": "x", "source": "env"})

    class _RestartSession(_FakeSession):
        pass

    fake_session = _RestartSession([
        {"code": 0, "data": {"id": 202, "status": "STOPPING"}},
        {"code": 0, "data": {"id": 202, "status": "STOPPING"}},
        {"code": 0, "data": {"id": 202, "status": "STOPPED"}},
        {"code": 0, "data": {"id": 303, "status": "WAITING"}},
    ])
    monkeypatch.setattr(openi_ci_module, "_make_requests_session", lambda session_cfg: fake_session)
    monkeypatch.setattr(openi_ci_module.time, "sleep", lambda *_args, **_kwargs: None)

    args = openi_ci_module._build_parser().parse_args(["restart-task"])
    assert openi_ci_module._handle_restart_task(args) == 0
    saved = openi_ci_module._load_json_state("task")
    assert saved["id"] == 303
    assert saved["status"] == "WAITING"
    methods = [call[0] for call in fake_session.calls]
    assert methods == ["post", "get", "get", "post"]
    assert "/stop?id=202" in fake_session.calls[0][1]
    assert "/api/v1/ai_task/restart?id=202" in fake_session.calls[3][1]



def test_restart_task_times_out_if_task_never_stops(openi_ci_module, tmp_path, monkeypatch):
    monkeypatch.setattr(openi_ci_module, "ARTIFACT_ROOT", tmp_path / ".artifacts" / "openi-910a")
    openi_ci_module._save_json_state("task", {"id": 404, "status": "RUNNING"})
    monkeypatch.setattr(openi_ci_module, "_load_session_config", lambda: {"cookie": "c", "csrf": "x", "source": "env"})

    class _NeverStopsSession(_FakeSession):
        pass

    fake_session = _NeverStopsSession([
        {"code": 0, "data": {"id": 404, "status": "STOPPING"}},
    ] * 100)
    monkeypatch.setattr(openi_ci_module, "_make_requests_session", lambda session_cfg: fake_session)

    times = iter([0, 1, 2, 3, 4, 605])
    monkeypatch.setattr(openi_ci_module.time, "time", lambda: next(times))
    monkeypatch.setattr(openi_ci_module.time, "sleep", lambda *_args, **_kwargs: None)

    args = openi_ci_module._build_parser().parse_args(["restart-task"])
    with pytest.raises(openi_ci_module.OpenITaskError, match="did not stop within"):
        openi_ci_module._handle_restart_task(args)



def test_wait_task_polls_until_running(openi_ci_module, tmp_path, monkeypatch):
    monkeypatch.setattr(openi_ci_module, "ARTIFACT_ROOT", tmp_path / ".artifacts" / "openi-910a")
    openi_ci_module._save_json_state("task", {"id": 101, "status": "CREATED"})
    monkeypatch.setattr(openi_ci_module, "_load_session_config", lambda: {"cookie": "c", "csrf": "x", "source": "env"})
    fake_session = _FakeSession([
        {"code": 0, "data": {"id": 101, "status": "CREATED"}},
        {"code": 0, "data": {"id": 101, "status": "RUNNING"}},
    ])
    monkeypatch.setattr(openi_ci_module, "_make_requests_session", lambda session_cfg: fake_session)
    monkeypatch.setattr(openi_ci_module.time, "sleep", lambda *_args, **_kwargs: None)
    args = openi_ci_module._build_parser().parse_args(["wait-task"])
    assert openi_ci_module._handle_wait_task(args) == 0
    saved = openi_ci_module._load_json_state("task")
    assert saved["status"] == "RUNNING"
    get_calls = [call for call in fake_session.calls if call[0] == "get"]
    assert len(get_calls) == 2


def test_ensure_task_reuses_recent_running_task(openi_ci_module, tmp_path, monkeypatch):
    monkeypatch.setattr(openi_ci_module, "ARTIFACT_ROOT", tmp_path / ".artifacts" / "openi-910a")
    openi_ci_module._save_json_state("task", {"id": 202, "status": "RUNNING"})
    monkeypatch.setattr(openi_ci_module, "_load_session_config", lambda: {"cookie": "c", "csrf": "x", "source": "env"})
    fake_session = _FakeSession([
        {"code": 0, "data": {"id": 202, "status": "RUNNING", "start_time": 1000, "end_time": 0, "image_id": "image-1", "image_name": "image-name-1", "cluster": "C2Net", "compute_source": "NPU"}},
    ])
    monkeypatch.setattr(openi_ci_module, "_make_requests_session", lambda session_cfg: fake_session)
    monkeypatch.setattr(openi_ci_module.time, "time", lambda: 1060)

    args = openi_ci_module._build_parser().parse_args([
        "ensure-task",
        "--repo-url", "https://github.com/candle-org/candle.git",
        "--ref", "main",
        "--image-id", "image-1",
        "--image-name", "image-name-1",
        "--spec-id", "328",
        "--cluster", "C2Net",
        "--compute-source", "NPU",
        "--has-internet", "2",
    ])

    assert openi_ci_module._handle_ensure_task(args) == 0
    saved = openi_ci_module._load_json_state("task")
    assert saved["id"] == 202
    assert saved["status"] == "RUNNING"
    assert saved["repo_url"] == "https://github.com/candle-org/candle.git"
    assert saved["ref"] == "main"
    assert [call[0] for call in fake_session.calls] == ["get"]



def test_ensure_task_restarts_after_reuse_timeout(openi_ci_module, tmp_path, monkeypatch):
    monkeypatch.setattr(openi_ci_module, "ARTIFACT_ROOT", tmp_path / ".artifacts" / "openi-910a")
    openi_ci_module._save_json_state("task", {"id": 202, "status": "RUNNING"})
    monkeypatch.setattr(openi_ci_module, "_load_session_config", lambda: {"cookie": "c", "csrf": "x", "source": "env"})
    fake_session = _FakeSession([
        {"code": 0, "data": {"id": 202, "status": "RUNNING", "start_time": 1000, "end_time": 0, "image_id": "image-1", "image_name": "image-name-1", "cluster": "C2Net", "compute_source": "NPU"}},
        {"code": 0, "data": {"id": 202, "status": "STOPPING"}},
        {"code": 0, "data": {"id": 202, "status": "STOPPED"}},
        {"code": 0, "data": {"id": 303, "status": "WAITING"}},
    ])
    monkeypatch.setattr(openi_ci_module, "_make_requests_session", lambda session_cfg: fake_session)
    monkeypatch.setattr(openi_ci_module.time, "time", lambda: 1000 + 12601)
    monkeypatch.setattr(openi_ci_module.time, "sleep", lambda *_args, **_kwargs: None)

    args = openi_ci_module._build_parser().parse_args([
        "ensure-task",
        "--repo-url", "https://github.com/candle-org/candle.git",
        "--ref", "main",
        "--image-id", "image-1",
        "--image-name", "image-name-1",
        "--spec-id", "328",
        "--cluster", "C2Net",
        "--compute-source", "NPU",
        "--has-internet", "2",
    ])

    assert openi_ci_module._handle_ensure_task(args) == 0
    saved = openi_ci_module._load_json_state("task")
    assert saved["id"] == 303
    assert saved["status"] == "WAITING"
    assert saved["repo_url"] == "https://github.com/candle-org/candle.git"
    assert saved["ref"] == "main"
    assert [call[0] for call in fake_session.calls] == ["get", "post", "get", "post"]

    monkeypatch.setattr(openi_ci_module, "ARTIFACT_ROOT", tmp_path / ".artifacts" / "openi-910a")
    openi_ci_module._save_json_state("task", {"id": 202, "status": "RUNNING"})
    monkeypatch.setattr(openi_ci_module, "_load_session_config", lambda: {"cookie": "c", "csrf": "x", "source": "env"})

    class _RestartSession(_FakeSession):
        pass

    fake_session = _RestartSession([
        {"code": 0, "data": {"id": 202, "status": "STOPPING"}},
        {"code": 0, "data": {"id": 202, "status": "STOPPING"}},
        {"code": 0, "data": {"id": 202, "status": "STOPPED"}},
        {"code": 0, "data": {"id": 303, "status": "WAITING"}},
    ])
    monkeypatch.setattr(openi_ci_module, "_make_requests_session", lambda session_cfg: fake_session)
    monkeypatch.setattr(openi_ci_module.time, "sleep", lambda *_args, **_kwargs: None)

    args = openi_ci_module._build_parser().parse_args(["restart-task"])
    assert openi_ci_module._handle_restart_task(args) == 0
    saved = openi_ci_module._load_json_state("task")
    assert saved["id"] == 303
    assert saved["status"] == "WAITING"
    methods = [call[0] for call in fake_session.calls]
    assert methods == ["post", "get", "get", "post"]
    assert "/stop?id=202" in fake_session.calls[0][1]
    assert "/api/v1/ai_task/restart?id=202" in fake_session.calls[3][1]


def test_restart_task_times_out_if_task_never_stops(openi_ci_module, tmp_path, monkeypatch):
    monkeypatch.setattr(openi_ci_module, "ARTIFACT_ROOT", tmp_path / ".artifacts" / "openi-910a")
    openi_ci_module._save_json_state("task", {"id": 404, "status": "RUNNING"})
    monkeypatch.setattr(openi_ci_module, "_load_session_config", lambda: {"cookie": "c", "csrf": "x", "source": "env"})

    class _NeverStopsSession(_FakeSession):
        pass

    fake_session = _NeverStopsSession([
        {"code": 0, "data": {"id": 404, "status": "STOPPING"}},
    ] * 100)
    monkeypatch.setattr(openi_ci_module, "_make_requests_session", lambda session_cfg: fake_session)

    times = iter([0, 1, 2, 3, 4, 605])
    monkeypatch.setattr(openi_ci_module.time, "time", lambda: next(times))
    monkeypatch.setattr(openi_ci_module.time, "sleep", lambda *_args, **_kwargs: None)

    args = openi_ci_module._build_parser().parse_args(["restart-task"])
    with pytest.raises(openi_ci_module.OpenITaskError, match="did not stop within"):
        openi_ci_module._handle_restart_task(args)



def test_wait_task_polls_until_running(openi_ci_module, tmp_path, monkeypatch):
    monkeypatch.setattr(openi_ci_module, "ARTIFACT_ROOT", tmp_path / ".artifacts" / "openi-910a")
    openi_ci_module._save_json_state("task", {"id": 101, "status": "CREATED"})
    monkeypatch.setattr(openi_ci_module, "_load_session_config", lambda: {"cookie": "c", "csrf": "x", "source": "env"})
    fake_session = _FakeSession([
        {"code": 0, "data": {"id": 101, "status": "CREATED"}},
        {"code": 0, "data": {"id": 101, "status": "RUNNING"}},
    ])
    monkeypatch.setattr(openi_ci_module, "_make_requests_session", lambda session_cfg: fake_session)
    monkeypatch.setattr(openi_ci_module.time, "sleep", lambda *_args, **_kwargs: None)
    args = openi_ci_module._build_parser().parse_args(["wait-task"])
    assert openi_ci_module._handle_wait_task(args) == 0
    saved = openi_ci_module._load_json_state("task")
    assert saved["status"] == "RUNNING"
    get_calls = [call for call in fake_session.calls if call[0] == "get"]
    assert len(get_calls) == 2


def test_cleanup_task_stops_then_deletes_when_not_kept(openi_ci_module, tmp_path, monkeypatch):
    monkeypatch.setattr(openi_ci_module, "ARTIFACT_ROOT", tmp_path / ".artifacts" / "openi-910a")
    openi_ci_module._save_json_state("task", {"id": 202, "status": "RUNNING"})
    monkeypatch.setattr(openi_ci_module, "_load_session_config", lambda: {"cookie": "c", "csrf": "x", "source": "env"})
    fake_session = _FakeSession([
        {"code": 0, "data": {"ok": True}},
        {"code": 0, "data": {"ok": True}},
    ])
    monkeypatch.setattr(openi_ci_module, "_make_requests_session", lambda session_cfg: fake_session)
    args = openi_ci_module._build_parser().parse_args(["cleanup-task"])
    assert openi_ci_module._handle_cleanup_task(args) == 0
    methods = [call[0] for call in fake_session.calls]
    assert methods == ["post", "post"]
    assert "/stop?id=202" in fake_session.calls[0][1]
    assert "/del?id=202" in fake_session.calls[1][1]


def test_remote_script_unsets_proxy_variables(openi_ci_module):
    script = openi_ci_module._build_remote_test_script(
        repo_url="https://github.com/candle-org/candle.git",
        ref="main",
    )
    assert "grep -i proxy" in script
    assert 'unset "$key"' in script


def test_remote_script_collects_npu_smi_and_build_log(openi_ci_module):
    script = openi_ci_module._build_remote_test_script(
        repo_url="https://github.com/candle-org/candle.git",
        ref="main",
    )
    assert "npu-smi.txt" in script
    assert "build.log" in script


def test_remote_run_suite_script_activates_conda_env_instead_of_conda_run(openi_ci_module):
    script = openi_ci_module._build_remote_run_suite_script("/tmp/repo")
    assert "conda activate /home/ma-user/work/.conda/envs/candle-py311" in script
    assert "conda.sh" in script
    assert "conda run -n candle-py311" not in script


def test_remote_run_suite_script_reuses_persistent_work_env_before_recreate(openi_ci_module):
    script = openi_ci_module._build_remote_run_suite_script("/tmp/repo")
    assert "/home/ma-user/work" in script
    assert 'if [ -x "/home/ma-user/work/.conda/envs/candle-py311/bin/python" ]' in script
    assert "-p /home/ma-user/work/.conda/envs/candle-py311 python=3.11" in script
    assert script.index("/home/ma-user/work/.conda/envs/candle-py311") < script.index("python setup.py build_ext --inplace")


def test_remote_run_suite_script_installs_requirements_test_in_py311_env(openi_ci_module):
    script = openi_ci_module._build_remote_run_suite_script("/tmp/repo")
    assert "requirements/requirements-test.txt" in script
    assert "python -m pip install" in script
    assert "-r requirements/requirements-test.txt" in script
    assert script.index("requirements/requirements-test.txt") < script.index("python setup.py build_ext --inplace")


def test_remote_run_suite_script_uses_aarch64_friendly_domestic_conda_source(openi_ci_module):
    script = openi_ci_module._build_remote_run_suite_script("/tmp/repo")
    assert "conda create -y --override-channels" in script
    assert "-c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge" in script
    assert "https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main" not in script


def test_remote_run_suite_script_uses_domestic_pip_index(openi_ci_module):
    script = openi_ci_module._build_remote_run_suite_script("/tmp/repo")
    assert "python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements/requirements-test.txt" in script
    assert "python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple Cython" in script



def test_prepare_remote_creates_kernel_and_persists_state(openi_ci_module, tmp_path, monkeypatch):
    monkeypatch.setattr(openi_ci_module, "ARTIFACT_ROOT", tmp_path / ".artifacts" / "openi-910a")
    monkeypatch.setattr(openi_ci_module, "REMOTE_ARTIFACTS", openi_ci_module.ARTIFACT_ROOT / "remote")
    openi_ci_module._save_json_state("task", {"id": 404, "status": "RUNNING"})
    monkeypatch.setattr(openi_ci_module, "_load_session_config", lambda: {"cookie": "c", "csrf": "x", "source": "env"})
    monkeypatch.setattr(openi_ci_module, "_make_requests_session", lambda session_cfg: object())
    monkeypatch.setattr(openi_ci_module, "_api_call", lambda _session, method, path, **kwargs: {"data": {"url": "https://host.example/lab?token=tok"}})
    monkeypatch.setattr(openi_ci_module, "OpenIJupyterClient", _FakeJupyterClient)
    args = openi_ci_module._build_parser().parse_args(["prepare-remote", "--repo-url", "https://github.com/candle-org/candle.git", "--ref", "main"])
    assert openi_ci_module._handle_prepare_remote(args) == 0
    kernel_state = openi_ci_module._load_json_state("kernel")
    run_state = openi_ci_module._load_json_state("run")
    assert kernel_state["kernel_id"] == "kernel-123"
    assert kernel_state["base_url"] == "https://host.example"
    assert run_state["repo_url"] == "https://github.com/candle-org/candle.git"
    assert run_state["ref"] == "main"
    command, _timeout = _FakeJupyterClient.instances[-1].commands[-1]
    assert command[:2] == ["bash", "-lc"]
    assert "git -c http.proxy= -c https.proxy= clone https://github.com/candle-org/candle.git" in command[2]
    assert openi_ci_module.PYTEST_910A_COMMAND not in command[2]


def test_prepare_remote_raises_when_remote_prepare_command_fails(openi_ci_module, tmp_path, monkeypatch):
    monkeypatch.setattr(openi_ci_module, "ARTIFACT_ROOT", tmp_path / ".artifacts" / "openi-910a")
    monkeypatch.setattr(openi_ci_module, "REMOTE_ARTIFACTS", openi_ci_module.ARTIFACT_ROOT / "remote")
    openi_ci_module._save_json_state("task", {"id": 406, "status": "RUNNING"})
    monkeypatch.setattr(openi_ci_module, "_load_session_config", lambda: {"cookie": "c", "csrf": "x", "source": "env"})
    monkeypatch.setattr(openi_ci_module, "_make_requests_session", lambda session_cfg: object())
    monkeypatch.setattr(openi_ci_module, "_api_call", lambda _session, method, path, **kwargs: {"data": {"url": "https://host.example/lab?token=tok"}})

    class _FailingPrepareJupyterClient(_FakeJupyterClient):
        def execute_shell(self, command, timeout=600):
            self.commands.append((command, timeout))
            return {"output": "git clone failed\n", "exit_code": 1}

    monkeypatch.setattr(openi_ci_module, "OpenIJupyterClient", _FailingPrepareJupyterClient)
    args = openi_ci_module._build_parser().parse_args(["prepare-remote", "--repo-url", "https://github.com/candle-org/candle.git", "--ref", "main"])
    with pytest.raises(openi_ci_module.OpenITaskError, match="Remote prepare failed with exit code 1"):
        openi_ci_module._handle_prepare_remote(args)
    assert not openi_ci_module._state_path("kernel").exists()
    assert not openi_ci_module._state_path("run").exists()


def test_prepare_remote_clears_stale_local_remote_artifacts(openi_ci_module, tmp_path, monkeypatch):
    monkeypatch.setattr(openi_ci_module, "ARTIFACT_ROOT", tmp_path / ".artifacts" / "openi-910a")
    monkeypatch.setattr(openi_ci_module, "REMOTE_ARTIFACTS", openi_ci_module.ARTIFACT_ROOT / "remote")
    openi_ci_module.REMOTE_ARTIFACTS.mkdir(parents=True, exist_ok=True)
    stale_junit = openi_ci_module.REMOTE_ARTIFACTS / "junit.xml"
    stale_pytest = openi_ci_module.REMOTE_ARTIFACTS / "pytest.log"
    stale_junit.write_text("stale junit", encoding="utf-8")
    stale_pytest.write_text("stale pytest", encoding="utf-8")
    openi_ci_module._save_json_state("task", {"id": 405, "status": "RUNNING"})
    monkeypatch.setattr(openi_ci_module, "_load_session_config", lambda: {"cookie": "c", "csrf": "x", "source": "env"})
    monkeypatch.setattr(openi_ci_module, "_make_requests_session", lambda session_cfg: object())
    monkeypatch.setattr(openi_ci_module, "_api_call", lambda _session, method, path, **kwargs: {"data": {"url": "https://host.example/lab?token=tok"}})
    monkeypatch.setattr(openi_ci_module, "OpenIJupyterClient", _FakeJupyterClient)
    args = openi_ci_module._build_parser().parse_args(["prepare-remote", "--repo-url", "https://github.com/candle-org/candle.git", "--ref", "main"])
    assert openi_ci_module._handle_prepare_remote(args) == 0
    assert not stale_junit.exists()
    assert not stale_pytest.exists()


def test_run_910a_suite_updates_run_state(openi_ci_module, tmp_path, monkeypatch):
    monkeypatch.setattr(openi_ci_module, "ARTIFACT_ROOT", tmp_path / ".artifacts" / "openi-910a")
    monkeypatch.setattr(openi_ci_module, "REMOTE_ARTIFACTS", openi_ci_module.ARTIFACT_ROOT / "remote")
    openi_ci_module._save_json_state("kernel", {"base_url": "https://host.example", "token": "tok", "kernel_id": "kernel-123"})
    openi_ci_module._save_json_state("run", {"repo_url": "https://github.com/candle-org/candle.git", "ref": "main", "remote_repo": "candle-openi-ci/repo", "remote_artifacts": {name: f"candle-openi-ci/repo/{name}" for name in ["pytest.log", "junit.xml", "summary.json", "remote_env.txt", "build.log", "npu-smi.txt"]}})
    monkeypatch.setattr(openi_ci_module, "OpenIJupyterClient", _FakeJupyterClient)
    args = openi_ci_module._build_parser().parse_args(["run-910a-suite"])
    assert openi_ci_module._handle_run_910a_suite(args) == 0
    run_state = openi_ci_module._load_json_state("run")
    assert run_state["exit_code"] == 0
    assert run_state["output"] == "remote ok\n"
    command, _timeout = _FakeJupyterClient.instances[-1].commands[-1]
    assert command[:2] == ["bash", "-lc"]
    assert openi_ci_module.PYTEST_910A_COMMAND in command[2]
    assert "git clone" not in command[2]
    assert "git checkout" not in command[2]


def test_real_jupyter_client_create_kernel_uses_api(openi_ci_module, monkeypatch):
    class _Resp:
        def __init__(self, payload):
            self._payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self._payload

    class _Session:
        def __init__(self):
            self.calls = []

        def post(self, url, json=None, timeout=None, headers=None):
            self.calls.append((url, json, timeout, headers))
            return _Resp({"id": "kernel-real-1"})

    session = _Session()
    client = openi_ci_module.OpenIJupyterClient(base_url="https://host/base", token="tok", session=session)
    kernel_id = client.create_kernel()
    assert kernel_id == "kernel-real-1"
    url, body, timeout, headers = session.calls[0]
    assert url == "https://host/base/api/kernels?token=tok"
    assert body == {"name": "python3"}
    assert timeout == 30
    assert headers["Content-Type"] == "application/json"


def test_real_jupyter_client_download_file_uses_contents_api(openi_ci_module, tmp_path, monkeypatch):
    class _Resp:
        def __init__(self, payload):
            self._payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self._payload

    class _Session:
        def __init__(self):
            self.calls = []

        def get(self, url, timeout=None):
            self.calls.append((url, timeout))
            return _Resp({"type": "file", "format": "text", "content": "hello world"})

    session = _Session()
    client = openi_ci_module.OpenIJupyterClient(base_url="https://host/base", token="tok", session=session)
    local_path = tmp_path / "remote" / "pytest.log"
    client.download_file("/home/ma-user/work/file.txt", local_path)
    assert local_path.read_text() == "hello world"
    url, timeout = session.calls[0]
    assert url == "https://host/base/api/contents/home/ma-user/work/file.txt?token=tok"
    assert timeout == 30


def test_real_jupyter_client_execute_shell_uses_kernel_channels(openi_ci_module, monkeypatch):
    events = []

    class _Resp:
        def __init__(self, payload):
            self._payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self._payload

    class _Session:
        def __init__(self):
            self.posts = []

        def post(self, url, json=None, timeout=None, headers=None):
            self.posts.append((url, json, timeout, headers))
            return _Resp({"id": "kernel-real-1"})

    class _WS:
        def __init__(self, url, header=None, on_open=None, on_message=None, on_error=None, on_close=None):
            self.url = url
            self.header = header
            self.on_open = on_open
            self.on_message = on_message
            self.on_error = on_error
            self.on_close = on_close
            self.sent = []
            events.append(self)

        def send(self, data):
            self.sent.append(data)

        def run_forever(self, sslopt=None):
            self.on_open(self)
            self.on_message(self, '{"parent_header":{"msg_id":"msg-1"},"msg_type":"stream","content":{"text":"hello\\n"}}')
            self.on_message(self, '{"parent_header":{"msg_id":"msg-1"},"msg_type":"execute_reply","content":{"status":"ok"}}')

        def close(self):
            return None

    monkeypatch.setattr(openi_ci_module, "WebSocketApp", _WS)
    monkeypatch.setattr(openi_ci_module.uuid, "uuid4", lambda: "msg-1")
    session = _Session()
    client = openi_ci_module.OpenIJupyterClient(base_url="https://host/base", token="tok", kernel_id="kernel-real-1", session=session)
    result = client.execute_shell(["bash", "-lc", "echo hello"], timeout=30)
    assert result["exit_code"] == 0
    assert result["output"] == "hello\n"
    ws = events[0]
    assert ws.url == "wss://host/base/api/kernels/kernel-real-1/channels?token=tok"
    payload = ws.sent[0]
    assert "msg-1" in payload
    assert "bash -lc 'echo hello'" in payload


def test_fetch_artifacts_keeps_downloading_when_some_files_are_missing(openi_ci_module, tmp_path, monkeypatch):
    monkeypatch.setattr(openi_ci_module, "ARTIFACT_ROOT", tmp_path / ".artifacts" / "openi-910a")
    monkeypatch.setattr(openi_ci_module, "REMOTE_ARTIFACTS", openi_ci_module.ARTIFACT_ROOT / "remote")
    openi_ci_module._save_json_state("kernel", {"base_url": "https://host.example", "token": "tok", "kernel_id": "kernel-123"})
    openi_ci_module._save_json_state(
        "run",
        {
            "remote_repo": "candle-openi-ci/repo",
            "remote_artifacts": {
                "build.log": "candle-openi-ci/repo/build.log",
                "junit.xml": "candle-openi-ci/repo/junit.xml",
                "pytest.log": "candle-openi-ci/repo/pytest.log",
            },
        },
    )

    class _MissingFileJupyterClient(_FakeJupyterClient):
        def download_file(self, remote_path, local_path):
            if remote_path.endswith("junit.xml"):
                response = requests.Response()
                response.status_code = 404
                raise requests.exceptions.HTTPError(response=response)
            super().download_file(remote_path, local_path)

    monkeypatch.setattr(openi_ci_module, "OpenIJupyterClient", _MissingFileJupyterClient)
    args = openi_ci_module._build_parser().parse_args(["fetch-artifacts"])
    assert openi_ci_module._handle_fetch_artifacts(args) == 0
    assert (openi_ci_module.REMOTE_ARTIFACTS / "build.log").exists()
    assert (openi_ci_module.REMOTE_ARTIFACTS / "pytest.log").exists()
    assert not (openi_ci_module.REMOTE_ARTIFACTS / "junit.xml").exists()


@pytest.mark.parametrize("artifact_name", ["pytest.log", "junit.xml", "summary.json", "remote_env.txt", "build.log", "npu-smi.txt"])
def test_fetch_artifacts_downloads_manifest(openi_ci_module, tmp_path, monkeypatch, artifact_name):
    monkeypatch.setattr(openi_ci_module, "ARTIFACT_ROOT", tmp_path / ".artifacts" / "openi-910a")
    monkeypatch.setattr(openi_ci_module, "REMOTE_ARTIFACTS", openi_ci_module.ARTIFACT_ROOT / "remote")
    openi_ci_module._save_json_state("kernel", {"base_url": "https://host.example", "token": "tok", "kernel_id": "kernel-123"})
    openi_ci_module._save_json_state("run", {"remote_repo": "candle-openi-ci/repo", "remote_artifacts": {name: f"candle-openi-ci/repo/{name}" for name in ["pytest.log", "junit.xml", "summary.json", "remote_env.txt", "build.log", "npu-smi.txt"]}})
    monkeypatch.setattr(openi_ci_module, "OpenIJupyterClient", _FakeJupyterClient)
    args = openi_ci_module._build_parser().parse_args(["fetch-artifacts"])
    assert openi_ci_module._handle_fetch_artifacts(args) == 0
    assert (openi_ci_module.REMOTE_ARTIFACTS / artifact_name).exists()


def test_build_remote_run_dist_script_2card_sets_visible_devices(openi_ci_module):
    script = openi_ci_module._build_remote_run_dist_script(
        "/home/ma-user/work/candle-openi-ci/repo",
        card_count=2,
        visible_devices="0,1",
    )
    assert "export ASCEND_RT_VISIBLE_DEVICES=0,1" in script
    assert "export ASCEND_VISIBLE_DEVICES=0,1" in script


def test_build_remote_run_dist_script_2card_runs_dist_tests(openi_ci_module):
    script = openi_ci_module._build_remote_run_dist_script(
        "/home/ma-user/work/candle-openi-ci/repo",
        card_count=2,
        visible_devices="0,1",
    )
    assert "tests/distributed/" in script
    assert "not all_to_all_single_async_unequal_multicard" in script
    assert "29715" in script  # 2-card HCCL port


def test_build_remote_run_dist_script_4card_runs_4card_tests(openi_ci_module):
    script = openi_ci_module._build_remote_run_dist_script(
        "/home/ma-user/work/candle-openi-ci/repo",
        card_count=4,
        visible_devices="0,1,2,3",
    )
    assert "export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3" in script
    assert "29724" in script  # 4-card HCCL port
    assert "not test_ddp" in script


def test_run_910a_dist_subcommand_registered(openi_ci_module):
    parser = openi_ci_module._build_parser()
    args = parser.parse_args(["run-910a-dist", "--card-count", "2", "--visible-devices", "0,1"])
    assert args.card_count == 2
    assert args.visible_devices == "0,1"


def test_resolve_artifact_root_default(openi_ci_module, monkeypatch):
    monkeypatch.delenv("OPENI_ARTIFACT_SUFFIX", raising=False)
    root = openi_ci_module._resolve_artifact_root()
    assert root.name == "openi-910a"


def test_resolve_artifact_root_with_suffix(openi_ci_module, monkeypatch):
    monkeypatch.setenv("OPENI_ARTIFACT_SUFFIX", "suite")
    root = openi_ci_module._resolve_artifact_root()
    assert root.name == "openi-910a-suite"


def test_load_json_state_falls_back_to_base_dir(openi_ci_module, tmp_path, monkeypatch):
    base_dir = tmp_path / ".artifacts" / "openi-910a"
    base_dir.mkdir(parents=True)
    (base_dir / "task.json").write_text('{"id": 123}', encoding="utf-8")

    suffixed_dir = tmp_path / ".artifacts" / "openi-910a-suite"
    suffixed_dir.mkdir(parents=True)

    monkeypatch.setattr(openi_ci_module, "ARTIFACT_ROOT", suffixed_dir)
    monkeypatch.setattr(openi_ci_module, "_REPO_ROOT", tmp_path)
    monkeypatch.setattr(openi_ci_module, "_BASE_ARTIFACT_DIR", "openi-910a")

    result = openi_ci_module._load_json_state("task")
    assert result["id"] == 123
