#!/usr/bin/env python3
"""OpenI orchestration helper CLI."""

from __future__ import annotations

import argparse
import base64
import json
import os
import subprocess
import time
from html.parser import HTMLParser
from pathlib import Path
import shlex
import shutil
import uuid
from urllib.parse import parse_qs, quote, urlparse

import requests
from websocket import WebSocketApp

BASE_URL = "https://openi.pcl.ac.cn"
LOGIN_URL = f"{BASE_URL}/user/login"
LOGIN_RSA_PUBLIC_KEY = (
    "MIGfMA0GCSqGSIb3DQEBAQUAA4GNADCBiQKBgQC8qOB41dNhDZyhdgiIxvCYv8fS\n"
    "kfWZOCWUZ3qhl//nMDz5RjemCxCQ2C3o63kbzWW6fEKRIWydhrVIWBu8eCEe7MfT\n"
    "YFe7IeOlwDH9mLqbMDzcLjFHphXNb2rRUii+PFJovdL9ys8utCDkWTSnP2G2x1RZ\n"
    "xUfxfQqoYkMaAEio0QIDAQAB\n"
)
_REPO_ROOT = Path(__file__).resolve().parents[2]
_BASE_ARTIFACT_DIR = "openi-910a"


def _resolve_artifact_root() -> Path:
    suffix = os.environ.get("OPENI_ARTIFACT_SUFFIX", "")
    name = f"{_BASE_ARTIFACT_DIR}-{suffix}" if suffix else _BASE_ARTIFACT_DIR
    return _REPO_ROOT / ".artifacts" / name


ARTIFACT_ROOT = _resolve_artifact_root()
REMOTE_ARTIFACTS = ARTIFACT_ROOT / "remote"
REMOTE_WORKDIR = "/home/ma-user/work/candle-openi-ci"
DEFAULT_WAIT_TIMEOUT_SECONDS = 600
DEFAULT_WAIT_INTERVAL_SECONDS = 10
RESTART_STOP_TIMEOUT_SECONDS = 600

REMOTE_ARTIFACT_NAMES = [
    "pytest.log",
    "junit.xml",
    "summary.json",
    "remote_env.txt",
    "build.log",
    "npu-smi.txt",
]
DOMESTIC_CONDA_FORGE_CHANNEL = "https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge"
DOMESTIC_PIP_INDEX = "https://pypi.tuna.tsinghua.edu.cn/simple"

PYTEST_910A_COMMAND = (
    "python -m pytest tests/npu/ -v --tb=short "
    "--ignore=tests/npu/test_pipeline_npu_bench_smoke.py "
    "--ignore=tests/npu/910b/ "
    "--ignore=tests/npu/310b/ "
    "--ignore=tests/npu/310p/"
)


class _LoginFormParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.current_form = None
        self.forms = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "form":
            self.current_form = {"attrs": attrs, "inputs": []}
        elif tag == "input" and self.current_form is not None:
            self.current_form["inputs"].append(attrs)

    def handle_endtag(self, tag):
        if tag == "form" and self.current_form is not None:
            self.forms.append(self.current_form)
            self.current_form = None


REUSABLE_TASK_STATUSES = ("RUNNING", "WAITING", "CREATED")
TERMINAL_FAILURE_STATES = {"CREATED_FAILED", "STOPPED", "STOP"}


class UnsupportedAuthError(RuntimeError):
    """Raised when the requested auth mode is recognized but unsupported."""


class OpenIAPIError(RuntimeError):
    """Raised when the OpenI API returns an error payload."""


class OpenITaskError(RuntimeError):
    """Raised when a task enters a terminal failure state."""


class OpenIJupyterClient:
    """Minimal Jupyter client implementation for kernel, execution, and file download."""

    def __init__(self, base_url: str, token: str, kernel_id: str | None = None, session=None):
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.kernel_id = kernel_id
        self.session = session or requests.Session()

    def create_kernel(self) -> str:
        response = self.session.post(
            f"{self.base_url}/api/kernels?token={self.token}",
            json={"name": "python3"},
            timeout=30,
            headers={"Content-Type": "application/json"},
        )
        response.raise_for_status()
        payload = response.json()
        self.kernel_id = payload["id"]
        return self.kernel_id

    def execute_shell(self, command: list[str], timeout: int = 600) -> dict:
        if not self.kernel_id:
            raise OpenITaskError("Kernel id is required before executing commands")
        ws_scheme = "wss" if self.base_url.startswith("https://") else "ws"
        host_path = self.base_url.split("://", 1)[1]
        ws_url = f"{ws_scheme}://{host_path}/api/kernels/{self.kernel_id}/channels?token={self.token}"
        message_id = str(uuid.uuid4())
        quoted = " ".join(shlex.quote(part) for part in command)
        code = (
            "import subprocess\n"
            f"_result = subprocess.run({quoted!r}, shell=True, text=True, capture_output=True)\n"
            "print(_result.stdout, end='')\n"
            "print(f'__CLAUDE_EXIT_CODE__={_result.returncode}')\n"
        )
        payload = {
            "header": {"msg_id": message_id, "msg_type": "execute_request", "username": "openi-ci", "version": "5.3"},
            "parent_header": {},
            "metadata": {},
            "content": {"code": code, "silent": False, "store_history": False, "user_expressions": {}, "allow_stdin": False, "stop_on_error": True},
            "channel": "shell",
        }
        state = {"output": [], "done": False, "exit_code": 1, "error": None}

        def on_open(ws):
            ws.send(json.dumps(payload))

        def on_message(ws, message):
            data = json.loads(message)
            if data.get("parent_header", {}).get("msg_id") != message_id:
                return
            msg_type = data.get("msg_type")
            content = data.get("content", {})
            if msg_type == "stream":
                state["output"].append(content.get("text", ""))
            elif msg_type == "execute_result":
                state["output"].append(content.get("data", {}).get("text/plain", ""))
            elif msg_type == "error":
                state["error"] = "\n".join(content.get("traceback", [])) or content.get("evalue", "")
                state["done"] = True
                ws.close()
            elif msg_type == "execute_reply":
                state["done"] = True
                ws.close()

        def on_error(_ws, error):
            state["error"] = str(error)
            state["done"] = True

        ws = WebSocketApp(ws_url, on_open=on_open, on_message=on_message, on_error=on_error, on_close=lambda *_args: None)
        ws.run_forever(sslopt={"cert_reqs": 0})
        if state["error"]:
            raise OpenITaskError(state["error"])
        output = "".join(state["output"])
        marker = "__CLAUDE_EXIT_CODE__="
        exit_code = 0
        if marker in output:
            prefix, suffix = output.rsplit(marker, 1)
            output = prefix
            exit_code = int(suffix.strip().splitlines()[0])
        return {"output": output, "exit_code": exit_code}

    def download_file(self, remote_path: str, local_path: Path) -> None:
        encoded = quote(remote_path.lstrip("/"), safe="/")
        response = self.session.get(f"{self.base_url}/api/contents/{encoded}?token={self.token}", timeout=30)
        response.raise_for_status()
        payload = response.json()
        local_path.parent.mkdir(parents=True, exist_ok=True)
        if payload.get("format") == "base64":
            local_path.write_bytes(base64.b64decode(payload.get("content", "")))
        else:
            local_path.write_text(payload.get("content", ""), encoding="utf-8")




class _LoginFormParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.current_form = None
        self.forms = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "form":
            self.current_form = {"attrs": attrs, "inputs": []}
        elif tag == "input" and self.current_form is not None:
            self.current_form["inputs"].append(attrs)

    def handle_endtag(self, tag):
        if tag == "form" and self.current_form is not None:
            self.forms.append(self.current_form)
            self.current_form = None


def _is_quota_limit_error(error: Exception) -> bool:
    if not isinstance(error, OpenIAPIError):
        return False
    return "'code': 2004" in str(error) or '"code": 2004' in str(error)


def _select_reusable_task(tasks: list[dict]) -> dict | None:
    priority = {"RUNNING": 0, "WAITING": 1, "CREATED": 2}
    candidates = [task for task in tasks if task.get("status") in priority]
    if not candidates:
        return None
    return sorted(candidates, key=lambda task: (priority[task["status"]], -int(task.get("id", 0))))[0]


def _list_reusable_tasks(session: requests.Session) -> list[dict]:
    payload = _api_call(session, "get", "/api/v1/ai_task/operation_profile", params={"id": 0})
    data = payload.get("data")
    if isinstance(data, list):
        return [task for task in data if task.get("status") in REUSABLE_TASK_STATUSES]
    return []



def _ensure_artifact_root() -> None:
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    REMOTE_ARTIFACTS.mkdir(parents=True, exist_ok=True)


def _state_path(name: str) -> Path:
    return ARTIFACT_ROOT / f"{name}.json"


def _save_json_state(name: str, payload: dict) -> Path:
    _ensure_artifact_root()
    path = _state_path(name)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _load_json_state(name: str) -> dict:
    path = _state_path(name)
    if not path.exists():
        base_path = _REPO_ROOT / ".artifacts" / _BASE_ARTIFACT_DIR / f"{name}.json"
        if base_path.exists():
            path = base_path
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_jupyter_url(lab_url: str) -> tuple[str, str]:
    parsed = urlparse(lab_url)
    path = parsed.path
    if path.endswith("/lab"):
        path = path[:-4]
    elif path.endswith("/lab/"):
        path = path[:-5]
    base = f"{parsed.scheme}://{parsed.netloc}{path.rstrip('/')}"
    token = parse_qs(parsed.query).get("token", [""])[0]
    return base, token


def _remote_exec_cmd(payload: str) -> list[str]:
    return ["bash", "-lc", payload]


def _build_jupyter_repo_path() -> str:
    return "candle-openi-ci/repo"


def _build_remote_absolute_repo_path() -> str:
    return f"{REMOTE_WORKDIR}/repo"


def _build_remote_artifact_manifest(remote_repo: str) -> dict:
    return {name: f"{remote_repo}/{name}" for name in REMOTE_ARTIFACT_NAMES}


def _clear_local_remote_artifacts() -> None:
    if REMOTE_ARTIFACTS.exists():
        shutil.rmtree(REMOTE_ARTIFACTS)


def _build_remote_prepare_script(*, repo_url: str, ref: str, remote_repo: str) -> str:
    return f"""set -euo pipefail
for key in $(env | cut -d= -f1 | grep -i proxy || true); do
  unset "$key"
done
unset http_proxy https_proxy ftp_proxy no_proxy
unset HTTP_PROXY HTTPS_PROXY FTP_PROXY NO_PROXY
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export PYTHONNOUSERSITE=1
mkdir -p {REMOTE_WORKDIR}
rm -rf {remote_repo}
git -c http.proxy= -c https.proxy= clone {repo_url} {remote_repo}
cd {remote_repo}
git -c http.proxy= -c https.proxy= fetch --all --tags
git checkout {ref}
env | sort > remote_env.txt
npu-smi info > npu-smi.txt 2>&1 || true
"""


def _build_remote_test_script(*, repo_url: str, ref: str) -> str:
    remote_repo = f"{REMOTE_WORKDIR}/repo"
    return f"""{_build_remote_prepare_script(repo_url=repo_url, ref=ref, remote_repo=remote_repo)}
python setup.py build_ext --inplace > build.log 2>&1
: > pytest.log
: > junit.xml
: > summary.json
python - <<'PY'
from pathlib import Path
Path('summary.json').write_text('{{}}', encoding='utf-8')
PY
{PYTEST_910A_COMMAND} --junitxml junit.xml > pytest.log 2>&1
"""


def _build_remote_run_suite_script(remote_repo: str) -> str:
    env_dir = "/home/ma-user/work/.conda/envs/candle-py311"
    return f"""set -euo pipefail
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export PYTHONNOUSERSITE=1
cd {remote_repo}
for key in $(env | cut -d= -f1 | grep -i proxy || true); do
  unset "$key"
done
CONDA_SH="/home/ma-user/anaconda3/etc/profile.d/conda.sh"
if [ ! -f "$CONDA_SH" ]; then
  echo "conda.sh not found" >&2
  exit 98
fi
source "$CONDA_SH"
if [ -x "{env_dir}/bin/python" ]; then
  : > build.log
else
  mkdir -p /home/ma-user/work/.conda/envs
  conda create -y --override-channels -c {DOMESTIC_CONDA_FORGE_CHANNEL} -p {env_dir} python=3.11 > build.log 2>&1
fi
conda activate {env_dir}
python -m pip install -i {DOMESTIC_PIP_INDEX} -r requirements/requirements-test.txt >> build.log 2>&1
python -m pip install -i {DOMESTIC_PIP_INDEX} Cython >> build.log 2>&1
python setup.py build_ext --inplace >> build.log 2>&1
: > pytest.log
: > junit.xml
: > summary.json
python - <<'PY'
from pathlib import Path
Path('summary.json').write_text('{{}}', encoding='utf-8')
PY
{PYTEST_910A_COMMAND} --junitxml junit.xml > pytest.log 2>&1
"""


def _build_remote_run_dist_script(remote_repo: str, card_count: int, visible_devices: str) -> str:
    env_dir = "/home/ma-user/work/.conda/envs/candle-py311"
    if card_count == 2:
        k_filter = (
            "not all_to_all_single_async_unequal_multicard "
            "and not all_to_all_single_invalid_split_pairing_multicard "
            "and not all_to_all_single_split_numel_validation_multicard "
            "and not test_ddp"
        )
        hccl_tests = (
            "'tests/distributed/test_hccl_all_to_all_single_async_unequal_multicard.py"
            "::test_hccl_all_to_all_single_async_unequal_multicard[2-29715]' "
            "'tests/distributed/test_hccl_all_to_all_single_invalid_splits_multicard.py"
            "::test_hccl_all_to_all_single_invalid_split_pairing_multicard[2-29715]' "
            "'tests/distributed/test_hccl_all_to_all_single_split_numel_validation_multicard.py"
            "::test_hccl_all_to_all_single_split_numel_validation_multicard[input_sum_mismatch-2-29716]' "
            "'tests/distributed/test_hccl_all_to_all_single_split_numel_validation_multicard.py"
            "::test_hccl_all_to_all_single_split_numel_validation_multicard[output_sum_mismatch-2-29716]'"
        )
    else:
        k_filter = "not test_ddp"
        hccl_tests = (
            "'tests/distributed/test_hccl_all_to_all_single_async_unequal_multicard.py"
            "::test_hccl_all_to_all_single_async_unequal_multicard[4-29724]' "
            "'tests/distributed/test_hccl_all_to_all_single_invalid_splits_multicard.py"
            "::test_hccl_all_to_all_single_invalid_split_pairing_multicard[4-29725]' "
            "'tests/distributed/test_hccl_all_to_all_single_split_numel_validation_multicard.py"
            "::test_hccl_all_to_all_single_split_numel_validation_multicard[input_sum_mismatch-4-29726]' "
            "'tests/distributed/test_hccl_all_to_all_single_split_numel_validation_multicard.py"
            "::test_hccl_all_to_all_single_split_numel_validation_multicard[output_sum_mismatch-4-29726]'"
        )
    return f"""set -euo pipefail
export ASCEND_RT_VISIBLE_DEVICES={visible_devices}
export ASCEND_VISIBLE_DEVICES={visible_devices}
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export PYTHONNOUSERSITE=1
cd {remote_repo}
for key in $(env | cut -d= -f1 | grep -i proxy || true); do
  unset "$key"
done
CONDA_SH="/home/ma-user/anaconda3/etc/profile.d/conda.sh"
if [ ! -f "$CONDA_SH" ]; then
  echo "conda.sh not found" >&2
  exit 98
fi
source "$CONDA_SH"
conda activate {env_dir}
python -m pytest tests/distributed/ -v --tb=short -k "{k_filter}" > dist.log 2>&1
HCCL_LOG=hccl.log
python -m pytest \\
  {hccl_tests} \\
  -v -rs --tb=short > "$HCCL_LOG" 2>&1
if grep -q 'SKIPPED' "$HCCL_LOG"; then
  echo "ERROR: SKIPPED tests found in HCCL log" >&2
  exit 1
fi
"""


def _handle_run_910a_dist(args: argparse.Namespace) -> int:
    client = _load_kernel_client()
    run_state = _load_json_state("run")
    script = _build_remote_run_dist_script(
        _build_remote_absolute_repo_path(),
        card_count=args.card_count,
        visible_devices=args.visible_devices,
    )
    command = _remote_exec_cmd(script)
    result = client.execute_shell(command, timeout=1800)
    run_state.update({"exit_code": result.get("exit_code", 0), "output": result.get("output", "")})
    _save_json_state("run", run_state)
    if run_state["exit_code"] != 0:
        raise OpenITaskError(f"Remote dist suite failed with exit code {run_state['exit_code']}")
    return 0


def _extract_login_form(html: str) -> tuple[dict, dict[str, dict]]:
    parser = _LoginFormParser()
    parser.feed(html)
    for form in parser.forms:
        inputs = {item.get("name") or item.get("id"): item for item in form["inputs"]}
        if "user_name" in inputs and "password" in inputs:
            return form["attrs"], inputs
    raise UnsupportedAuthError("OpenI login form not found")


def _encrypt_password(password: str, public_key: str) -> str:
    pem = "-----BEGIN PUBLIC KEY-----\n" + public_key + "-----END PUBLIC KEY-----\n"
    node_script = (
        "const crypto = require('crypto');"
        "const key = process.env.OPENI_RSA_PUBLIC_KEY;"
        "const password = process.env.OPENI_RSA_PASSWORD;"
        "const out = crypto.publicEncrypt({key, padding: crypto.constants.RSA_PKCS1_PADDING}, Buffer.from(password, 'utf8')).toString('base64');"
        "process.stdout.write(out);"
    )
    env = os.environ.copy()
    env["OPENI_RSA_PUBLIC_KEY"] = pem
    env["OPENI_RSA_PASSWORD"] = password
    completed = subprocess.run(
        ["node", "-e", node_script],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    return completed.stdout.strip()


def _cookie_header_from_jar(jar: requests.cookies.RequestsCookieJar) -> str:
    return "; ".join(f"{cookie.name}={cookie.value}" for cookie in jar)


def _login_with_password(username: str, password: str) -> dict:
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0"})
    page = session.get(LOGIN_URL, timeout=30)
    page.raise_for_status()
    form_attrs, inputs = _extract_login_form(page.text)
    action = form_attrs.get("action", "/user/login")
    action_url = action if action.startswith("http") else f"{BASE_URL}{action}"
    csrf = inputs.get("_csrf", {}).get("value", "")
    encrypted_password = _encrypt_password(password, LOGIN_RSA_PUBLIC_KEY)
    form_data = {
        "_csrf": csrf,
        "user_name": username,
        "password": encrypted_password,
        "remember": "on",
    }
    response = session.post(action_url, data=form_data, allow_redirects=True, timeout=30)
    response.raise_for_status()
    if response.url.rstrip("/") == LOGIN_URL.rstrip("/") and not response.history:
        raise UnsupportedAuthError("username/password login did not complete successfully")
    current_csrf = session.cookies.get("_csrf", domain="openi.pcl.ac.cn") or session.cookies.get("_csrf") or csrf
    return {
        "cookie": _cookie_header_from_jar(session.cookies),
        "csrf": current_csrf,
        "source": "password-login",
    }


def _load_session_config() -> dict:
    cookie = os.environ.get("OPENI_COOKIE")
    csrf = os.environ.get("OPENI_CSRF")
    if cookie and csrf:
        return {"cookie": cookie, "csrf": csrf, "source": "env"}

    session_path = _state_path("session")
    if session_path.exists():
        return _load_json_state("session")

    username = os.environ.get("OPENI_USER_NAME")
    password = os.environ.get("OPENI_USER_PASSWORD")
    if username and password:
        return _login_with_password(username, password)
    if username or password:
        raise UnsupportedAuthError("username/password login is not implemented")

    raise UnsupportedAuthError("OpenI session auth is not implemented without OPENI_COOKIE/OPENI_CSRF")


def _make_requests_session(session_cfg: dict) -> requests.Session:
    session = requests.Session()
    cookie = session_cfg.get("cookie", "")
    csrf = session_cfg.get("csrf", "")
    if cookie:
        for part in cookie.split(";"):
            part = part.strip()
            if "=" not in part:
                continue
            key, value = part.split("=", 1)
            session.cookies.set(key.strip(), value.strip(), domain="openi.pcl.ac.cn")
    if csrf:
        session.cookies.set("_csrf", csrf, domain="openi.pcl.ac.cn")
    session.headers.update(
        {
            "X-Csrf-Token": csrf,
            "Content-Type": "application/json",
            "User-Agent": "Mozilla/5.0",
        }
    )
    return session


def _api_call(session: requests.Session, method: str, path: str, **kwargs) -> dict:
    csrf = session.headers.get("X-Csrf-Token", "")
    separator = "&" if "?" in path else "?"
    url = f"{BASE_URL}{path}{separator}_csrf={csrf}"
    response = getattr(session, method)(url, **kwargs)
    response.raise_for_status()
    payload = response.json()
    if payload.get("code") != 0:
        raise OpenIAPIError(str(payload))
    return payload


def _build_create_payload(args: argparse.Namespace) -> dict:
    return {
        "repoOwnerName": "-",
        "repoName": "-",
        "job_type": "DEBUG",
        "cluster": args.cluster,
        "compute_source": args.compute_source,
        "description": "",
        "branch_name": args.ref,
        "image_id": args.image_id,
        "image_name": args.image_name,
        "pretrain_model_id_str": "",
        "dataset_uuid_str": "",
        "has_internet": int(args.has_internet),
        "spec_id": int(args.spec_id),
        "display_job_name": f"openi-{int(time.time())}",
    }


def _bool_arg(value: str | None) -> bool:
    if value is None:
        return False
    return str(value).lower() in {"1", "true", "yes", "on"}


def _load_kernel_client() -> OpenIJupyterClient:
    kernel_state = _load_json_state("kernel")
    return OpenIJupyterClient(
        base_url=kernel_state["base_url"],
        token=kernel_state["token"],
        kernel_id=kernel_state.get("kernel_id"),
    )


def _save_task_with_context(task: dict, args: argparse.Namespace) -> None:
    task.update({"repo_url": args.repo_url, "ref": args.ref, "spec_id": str(args.spec_id)})
    _save_json_state("task", task)


def _task_matches_request(task: dict, args: argparse.Namespace) -> bool:
    saved_spec = task.get("spec_id")
    spec_id_match = str(saved_spec) == str(args.spec_id) if saved_spec is not None else True
    return (
        task.get("image_id") == args.image_id
        and task.get("image_name") == args.image_name
        and task.get("cluster") == args.cluster
        and task.get("compute_source") == args.compute_source
        and spec_id_match
    )


def _find_matching_task_in_my_list(session: requests.Session, args: argparse.Namespace) -> dict | None:
    payload = _api_call(session, "get", "/api/v1/ai_task/my_list")
    data = payload.get("data")
    tasks = []
    if isinstance(data, dict):
        raw_tasks = data.get("tasks") or data.get("list") or []
        for item in raw_tasks:
            if isinstance(item, dict) and isinstance(item.get("task"), dict):
                tasks.append(item["task"])
            elif isinstance(item, dict):
                tasks.append(item)
    elif isinstance(data, list):
        tasks = data
    candidates = [task for task in tasks if _task_matches_request(task, args)]
    if not candidates:
        return None
    candidates.sort(key=lambda task: (int(task.get("start_time") or 0), int(task.get("id") or 0)), reverse=True)
    return candidates[0]


def _handle_create_task(args: argparse.Namespace) -> int:
    session_cfg = _load_session_config()
    session = _make_requests_session(session_cfg)
    payload = _build_create_payload(args)
    response = _api_call(session, "post", "/api/v1/-/-/ai_task/create", json=payload)
    task = response["data"]
    _save_task_with_context(task, args)
    return 0




def _handle_ensure_task(args: argparse.Namespace) -> int:
    task_path = _state_path("task")
    if not task_path.exists():
        return _handle_create_task(args)

    session_cfg = _load_session_config()
    session = _make_requests_session(session_cfg)
    existing_task = _load_json_state("task")
    try:
        response = _api_call(session, "get", f"/api/v1/ai_task/brief?id={existing_task['id']}")
        task = response["data"]
    except requests.exceptions.HTTPError as exc:
        response_obj = getattr(exc, "response", None)
        if response_obj is None or response_obj.status_code != 403:
            raise
        task = _find_matching_task_in_my_list(session, args)
        if task is None:
            return _handle_create_task(args)

    if not _task_matches_request(task, args):
        discovered_task = _find_matching_task_in_my_list(session, args)
        if discovered_task is None:
            return _handle_create_task(args)
        task = discovered_task

    status = task.get("status", "")
    reusable_statuses = {"RUNNING", "WAITING", "CREATED"}
    restartable_statuses = reusable_statuses | {"STOPPED", "STOP"}
    reuse_timeout_seconds = int(args.reuse_timeout_minutes) * 60
    started_at = task.get("start_time") or 0
    age_seconds = max(0, int(time.time() - started_at)) if started_at else 0

    if status in reusable_statuses and age_seconds < reuse_timeout_seconds:
        _save_task_with_context(task, args)
        return 0

    if status in restartable_statuses:
        _save_json_state("task", task)
        _handle_restart_task(args)
        restarted_task = _load_json_state("task")
        _save_task_with_context(restarted_task, args)
        return 0

    return _handle_create_task(args)


def _handle_restart_task(args: argparse.Namespace) -> int:
    session_cfg = _load_session_config()
    session = _make_requests_session(session_cfg)
    task = _load_json_state("task")
    task_id = task["id"]

    _api_call(session, "post", f"/api/v1/ai_task/stop?id={task_id}")
    deadline = time.time() + RESTART_STOP_TIMEOUT_SECONDS
    while time.time() < deadline:
        response = _api_call(session, "get", f"/api/v1/ai_task/brief?id={task_id}")
        task = response["data"]
        status = task.get("status", "")
        if status in {"STOPPED", "STOP"}:
            restart_response = _api_call(session, "post", f"/api/v1/ai_task/restart?id={task_id}")
            _save_json_state("task", restart_response["data"])
            return 0
        time.sleep(DEFAULT_WAIT_INTERVAL_SECONDS)
    raise OpenITaskError(f"Task {task_id} did not stop within {RESTART_STOP_TIMEOUT_SECONDS}s")


def _handle_wait_task(args: argparse.Namespace) -> int:
    session_cfg = _load_session_config()
    session = _make_requests_session(session_cfg)
    task = _load_json_state("task")
    task_id = task["id"]
    deadline = time.time() + DEFAULT_WAIT_TIMEOUT_SECONDS
    while time.time() < deadline:
        response = _api_call(session, "get", f"/api/v1/ai_task/brief?id={task_id}")
        task = response["data"]
        _save_json_state("task", task)
        status = task.get("status", "")
        if status == "RUNNING":
            return 0
        if status in TERMINAL_FAILURE_STATES:
            raise OpenITaskError(f"Task {task_id} failed with status {status}")
        time.sleep(DEFAULT_WAIT_INTERVAL_SECONDS)
    raise OpenITaskError(f"Task {task_id} did not reach RUNNING within {DEFAULT_WAIT_TIMEOUT_SECONDS}s")


def _handle_prepare_remote(args: argparse.Namespace) -> int:
    _clear_local_remote_artifacts()
    session_cfg = _load_session_config()
    session = _make_requests_session(session_cfg)
    task = _load_json_state("task")
    response = _api_call(session, "get", f"/api/v1/ai_task/debug_url?id={task['id']}&file=")
    lab_url = response["data"]["url"]
    base_url, token = _parse_jupyter_url(lab_url)
    client = OpenIJupyterClient(base_url=base_url, token=token, session=session)
    kernel_id = client.create_kernel()
    absolute_repo = _build_remote_absolute_repo_path()
    jupyter_repo = _build_jupyter_repo_path()
    prepare_command = _remote_exec_cmd(
        _build_remote_prepare_script(repo_url=args.repo_url, ref=args.ref, remote_repo=absolute_repo)
    )
    prepare_result = client.execute_shell(prepare_command, timeout=600)
    if prepare_result.get("exit_code", 0) != 0:
        raise OpenITaskError(
            f"Remote prepare failed with exit code {prepare_result.get('exit_code', 1)}"
        )
    _save_json_state(
        "kernel",
        {"base_url": base_url, "token": token, "kernel_id": kernel_id, "task_id": task["id"]},
    )
    _save_json_state(
        "run",
        {
            "repo_url": args.repo_url,
            "ref": args.ref,
            "remote_repo": jupyter_repo,
            "remote_artifacts": _build_remote_artifact_manifest(jupyter_repo),
        },
    )
    return 0


def _handle_run_910a_suite(args: argparse.Namespace) -> int:
    client = _load_kernel_client()
    run_state = _load_json_state("run")
    command = _remote_exec_cmd(_build_remote_run_suite_script(_build_remote_absolute_repo_path()))
    result = client.execute_shell(command, timeout=600)
    run_state.update({"exit_code": result.get("exit_code", 0), "output": result.get("output", "")})
    _save_json_state("run", run_state)
    if run_state["exit_code"] != 0:
        raise OpenITaskError(f"Remote suite failed with exit code {run_state['exit_code']}")
    return 0


def _handle_fetch_artifacts(args: argparse.Namespace) -> int:
    _ensure_artifact_root()
    client = _load_kernel_client()
    run_state = _load_json_state("run")
    manifest = run_state.get("remote_artifacts") or _build_remote_artifact_manifest(run_state["remote_repo"])
    for artifact_name, remote_path in manifest.items():
        try:
            client.download_file(remote_path, REMOTE_ARTIFACTS / artifact_name)
        except requests.exceptions.HTTPError as exc:
            response = getattr(exc, "response", None)
            if response is not None and response.status_code == 404:
                continue
            raise
    return 0


def _handle_cleanup_task(args: argparse.Namespace) -> int:
    if _bool_arg(args.keep_task_on_failure):
        return 0
    task_path = _state_path("task")
    if not task_path.exists():
        return 0
    task = _load_json_state("task")
    task_id = task["id"]
    session_cfg = _load_session_config()
    session = _make_requests_session(session_cfg)
    _api_call(session, "post", f"/api/v1/ai_task/stop?id={task_id}")
    _api_call(session, "post", f"/api/v1/ai_task/del?id={task_id}")
    return 0


def _not_implemented(_: argparse.Namespace) -> int:
    raise NotImplementedError("OpenI networked operations are implemented in later tasks")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="OpenI orchestration helper")
    subparsers = parser.add_subparsers(dest="command", required=True)

    create = subparsers.add_parser("create-task", help="Create OpenI task")
    create.add_argument("--repo-url", required=True)
    create.add_argument("--ref", required=True)
    create.add_argument("--image-id", required=True)
    create.add_argument("--image-name", required=True)
    create.add_argument("--spec-id", required=True)
    create.add_argument("--cluster", required=True)
    create.add_argument("--compute-source", required=True)
    create.add_argument("--has-internet", required=True)
    create.add_argument("--keep-task-on-failure", required=False)

    ensure = subparsers.add_parser("ensure-task", help="Create, reuse, or restart OpenI task")
    ensure.add_argument("--repo-url", required=True)
    ensure.add_argument("--ref", required=True)
    ensure.add_argument("--image-id", required=True)
    ensure.add_argument("--image-name", required=True)
    ensure.add_argument("--spec-id", required=True)
    ensure.add_argument("--cluster", required=True)
    ensure.add_argument("--compute-source", required=True)
    ensure.add_argument("--has-internet", required=True)
    ensure.add_argument("--reuse-timeout-minutes", required=False, default="210")
    ensure.add_argument("--keep-task-on-failure", required=False)

    subparsers.add_parser("restart-task", help="Stop current task and restart it")
    subparsers.add_parser("wait-task", help="Wait for OpenI task to run")

    prepare = subparsers.add_parser("prepare-remote", help="Prepare remote checkout")
    prepare.add_argument("--repo-url", required=True)
    prepare.add_argument("--ref", required=True)

    subparsers.add_parser("run-910a-suite", help="Run remote 910A suite")

    dist = subparsers.add_parser("run-910a-dist", help="Run remote 910A distributed tests")
    dist.add_argument("--card-count", required=True, type=int, choices=[2, 4])
    dist.add_argument("--visible-devices", required=True)
    subparsers.add_parser("fetch-artifacts", help="Fetch remote artifacts")

    cleanup = subparsers.add_parser("cleanup-task", help="Cleanup remote task")
    cleanup.add_argument("--keep-task-on-failure", required=False)

    subparsers.add_parser("login-wechat", help="Login locally via WeChat")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    handlers = {
        "create-task": _handle_create_task,
        "ensure-task": _handle_ensure_task,
        "restart-task": _handle_restart_task,
        "wait-task": _handle_wait_task,
        "prepare-remote": _handle_prepare_remote,
        "run-910a-suite": _handle_run_910a_suite,
        "run-910a-dist": _handle_run_910a_dist,
        "fetch-artifacts": _handle_fetch_artifacts,
        "cleanup-task": _handle_cleanup_task,
        "login-wechat": _not_implemented,
    }
    return handlers[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
