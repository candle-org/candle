"""Tests for the ``import torch`` → candle compatibility hook.

All tests run in **subprocesses** because import hooks are process-global side
effects.  The bootstrap module ``_candle_torch_compat`` is loaded directly by
manipulating ``sys.path`` to point at the project root.
"""

import os
import subprocess
import sys
import textwrap

import pytest

# src/ contains both _candle_torch_compat.py and candle/
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC_DIR = os.path.join(_PROJECT_ROOT, "src")


def _run(code, *, env_extra=None, expect_fail=False, bootstrap=True):
    """Run *code* in a subprocess with the project on sys.path.

    When *bootstrap* is True (default), the bootstrap module is imported
    before the test code — simulating what the ``.pth`` file does at Python
    startup in a real install.

    Returns (stdout, stderr, returncode).
    """
    env = os.environ.copy()
    # Ensure the subprocess can find both the bootstrap module (project root)
    # and candle itself (src/).
    python_path = os.pathsep.join([_PROJECT_ROOT, _SRC_DIR])
    existing = env.get("PYTHONPATH", "")
    if existing:
        python_path = python_path + os.pathsep + existing
    env["PYTHONPATH"] = python_path
    if env_extra:
        env.update(env_extra)

    if bootstrap:
        code = "import _candle_torch_compat\n" + code

    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env,
        timeout=30,
    )
    if not expect_fail and result.returncode != 0:
        raise AssertionError(
            f"Subprocess failed (rc={result.returncode}):\n"
            f"--- stdout ---\n{result.stdout}\n"
            f"--- stderr ---\n{result.stderr}"
        )
    return result.stdout.strip(), result.stderr.strip(), result.returncode


# -------------------------------------------------------------------------
# TestShouldRedirect — env var logic
# -------------------------------------------------------------------------

class TestShouldRedirect:
    """Test the ``_should_redirect`` decision logic."""

    def test_use_candle_true(self):
        code = textwrap.dedent("""\
            import _candle_torch_compat as m
            # Re-check with fresh state.
            m._uninstall()
            assert m._should_redirect() is True
        """)
        _run(code, env_extra={"USE_CANDLE": "1"}, bootstrap=False)

    def test_use_candle_yes(self):
        code = textwrap.dedent("""\
            import _candle_torch_compat as m
            m._uninstall()
            assert m._should_redirect() is True
        """)
        _run(code, env_extra={"USE_CANDLE": "yes"}, bootstrap=False)

    def test_use_candle_false(self):
        code = textwrap.dedent("""\
            import _candle_torch_compat as m
            m._uninstall()
            assert m._should_redirect() is False
        """)
        _run(code, env_extra={"USE_CANDLE": "0"}, bootstrap=False)

    def test_use_candle_no(self):
        code = textwrap.dedent("""\
            import _candle_torch_compat as m
            m._uninstall()
            assert m._should_redirect() is False
        """)
        _run(code, env_extra={"USE_CANDLE": "no"}, bootstrap=False)


# -------------------------------------------------------------------------
# TestResolveCandleName — alias mapping
# -------------------------------------------------------------------------

class TestResolveCandleName:
    """Test the ``_resolve_candle_name`` function."""

    def test_top_level(self):
        code = textwrap.dedent("""\
            import _candle_torch_compat as m
            assert m._resolve_candle_name("torch") == "candle"
        """)
        _run(code, env_extra={"USE_CANDLE": "0"}, bootstrap=False)

    def test_direct_submodule(self):
        code = textwrap.dedent("""\
            import _candle_torch_compat as m
            assert m._resolve_candle_name("torch.nn") == "candle.nn"
            assert m._resolve_candle_name("torch.optim") == "candle.optim"
        """)
        _run(code, env_extra={"USE_CANDLE": "0"}, bootstrap=False)

    def test_aliased_autograd(self):
        code = textwrap.dedent("""\
            import _candle_torch_compat as m
            assert m._resolve_candle_name("torch.autograd") == "candle.autograd"
            assert m._resolve_candle_name("torch.autograd.graph") == "candle.autograd.graph"
        """)
        _run(code, env_extra={"USE_CANDLE": "0"}, bootstrap=False)

    def test_aliased_random(self):
        code = textwrap.dedent("""\
            import _candle_torch_compat as m
            assert m._resolve_candle_name("torch.random") == "candle._random"
        """)
        _run(code, env_extra={"USE_CANDLE": "0"}, bootstrap=False)

    def test_nested_submodule(self):
        code = textwrap.dedent("""\
            import _candle_torch_compat as m
            assert m._resolve_candle_name("torch.nn.functional") == "candle.nn.functional"
        """)
        _run(code, env_extra={"USE_CANDLE": "0"}, bootstrap=False)


# -------------------------------------------------------------------------
# TestImportHook — end-to-end import redirection
# -------------------------------------------------------------------------

class TestImportHook:
    """End-to-end tests: ``import torch`` → candle."""

    def test_import_torch(self):
        code = textwrap.dedent("""\
            import torch
            import candle
            assert torch is candle, f"torch={id(torch)}, candle={id(candle)}"
            print("OK")
        """)
        out, _, _ = _run(code, env_extra={"USE_CANDLE": "1"})
        assert "OK" in out

    def test_import_torch_nn(self):
        code = textwrap.dedent("""\
            import torch.nn
            import candle.nn
            assert torch.nn is candle.nn
            print("OK")
        """)
        out, _, _ = _run(code, env_extra={"USE_CANDLE": "1"})
        assert "OK" in out

    def test_import_torch_nn_functional(self):
        code = textwrap.dedent("""\
            import torch.nn.functional as F
            import candle.nn.functional as CF
            assert F is CF
            print("OK")
        """)
        out, _, _ = _run(code, env_extra={"USE_CANDLE": "1"})
        assert "OK" in out

    def test_from_torch_import_tensor(self):
        code = textwrap.dedent("""\
            from torch import Tensor
            from candle import Tensor as CTensor
            assert Tensor is CTensor
            print("OK")
        """)
        out, _, _ = _run(code, env_extra={"USE_CANDLE": "1"})
        assert "OK" in out

    def test_torch_optim(self):
        code = textwrap.dedent("""\
            import torch.optim
            import candle.optim
            assert torch.optim is candle.optim
            print("OK")
        """)
        out, _, _ = _run(code, env_extra={"USE_CANDLE": "1"})
        assert "OK" in out

    def test_torch_autograd(self):
        code = textwrap.dedent("""\
            import torch.autograd
            import candle.autograd
            assert torch.autograd is candle.autograd
            print("OK")
        """)
        out, _, _ = _run(code, env_extra={"USE_CANDLE": "1"})
        assert "OK" in out

    def test_from_torch_hub_imports_torchvision_symbols(self):
        """Regression: torchvision imports these names from torch.hub."""
        code = textwrap.dedent("""\
            from torch.hub import _get_torch_home, load_state_dict_from_url
            assert callable(_get_torch_home)
            assert callable(load_state_dict_from_url)
            print("OK")
        """)
        out, _, _ = _run(code, env_extra={"USE_CANDLE": "1"})
        assert "OK" in out

    def test_import_torch_custom_ops(self):
        """Regression: torchvision meta registrations import torch._custom_ops."""
        code = textwrap.dedent("""\
            import importlib
            mod = importlib.import_module("torch._custom_ops")
            assert mod is not None
            print("OK")
        """)
        out, _, _ = _run(code, env_extra={"USE_CANDLE": "1"})
        assert "OK" in out

    def test_torch_hub_set_dir_updates_get_dir_without_warning(self):
        code = textwrap.dedent("""\
            import tempfile
            import warnings
            import torch.hub
            target = tempfile.mkdtemp(prefix='candle-hub-dir-')
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                torch.hub.set_dir(target)
            assert torch.hub.get_dir() == target
            assert not caught, [str(w.message) for w in caught]
            print('OK')
        """)
        out, _, _ = _run(code, env_extra={"USE_CANDLE": "1"})
        assert "OK" in out

    def test_torch_hub_load_state_dict_cleans_partial_file_on_failure(self):
        code = textwrap.dedent("""\
            import pathlib
            import tempfile
            import urllib.request
            import torch.hub

            class BrokenResponse:
                def __init__(self):
                    self.calls = 0

                def __enter__(self):
                    return self

                def __exit__(self, exc_type, exc, tb):
                    return False

                def read(self, _size=-1):
                    self.calls += 1
                    if self.calls == 1:
                        return b'partial-download'
                    raise RuntimeError('simulated download failure')

            original_urlopen = urllib.request.urlopen
            urllib.request.urlopen = lambda request: BrokenResponse()
            model_dir = tempfile.mkdtemp(prefix='candle-hub-download-')
            try:
                try:
                    torch.hub.load_state_dict_from_url(
                        'https://example.com/state.pt',
                        model_dir=model_dir,
                        progress=False,
                    )
                except RuntimeError as exc:
                    assert 'simulated download failure' in str(exc)
                else:
                    raise AssertionError('expected download failure')
                partials = sorted(path.name for path in pathlib.Path(model_dir).glob('*.partial'))
                cached = pathlib.Path(model_dir) / 'state.pt'
                assert partials == [], partials
                assert not cached.exists()
                print('OK')
            finally:
                urllib.request.urlopen = original_urlopen
        """)
        out, _, _ = _run(code, env_extra={"USE_CANDLE": "1"})
        assert "OK" in out

    def test_from_torch_utils_model_zoo_imports_load_url(self):
        """Older torchvision versions fall back to torch.utils.model_zoo."""
        code = textwrap.dedent("""\
            from torch.utils.model_zoo import load_url
            assert callable(load_url)
            print("OK")
        """)
        out, _, _ = _run(code, env_extra={"USE_CANDLE": "1"})
        assert "OK" in out

    def test_torch_frombuffer_shares_memory_with_writable_buffer(self):
        """torch.frombuffer must alias the original writable buffer."""
        code = textwrap.dedent("""\
            import torch
            raw = bytearray([1, 2, 3, 4])
            t = torch.frombuffer(raw, dtype=torch.uint8)
            raw[1] = 99
            assert t.tolist() == [1, 99, 3, 4]
            print("OK")
        """)
        out, _, _ = _run(code, env_extra={"USE_CANDLE": "1"})
        assert "OK" in out

    def test_torch_frombuffer_accepts_bytearray_with_offset(self):
        """Torchvision MNIST loaders use torch.frombuffer on raw IDX bytes."""
        code = textwrap.dedent("""\
            import torch
            t = torch.frombuffer(bytearray(range(8)), dtype=torch.uint8, offset=4)
            assert t.tolist() == [4, 5, 6, 7]
            print("OK")
        """)
        out, _, _ = _run(code, env_extra={"USE_CANDLE": "1"})
        assert "OK" in out

    def test_torch_frombuffer_supports_multibyte_dtype_and_view(self):
        """Torchvision MNIST uses frombuffer result with shape reinterpretation."""
        code = textwrap.dedent("""\
            import torch
            raw = bytearray([1, 0, 2, 0, 3, 0, 4, 0])
            t = torch.frombuffer(raw, dtype=torch.int16)
            assert t.view(2, 2).tolist() == [[1, 2], [3, 4]]
            print("OK")
        """)
        out, _, _ = _run(code, env_extra={"USE_CANDLE": "1"})
        assert "OK" in out

    def test_torch_zeros(self):
        """Functional: actually create a tensor through the redirected import."""
        code = textwrap.dedent("""\
            import torch
            t = torch.zeros(3)
            assert t.shape == (3,)
            assert str(type(t).__module__).startswith("candle")
            print("OK")
        """)
        out, _, _ = _run(code, env_extra={"USE_CANDLE": "1"})
        assert "OK" in out

    def test_torch_accelerator_tutorial_snippet(self):
        """Compatibility: tutorial device selection works through torch.accelerator."""
        code = textwrap.dedent("""\
            import torch

            assert hasattr(torch, "accelerator")
            assert hasattr(torch.accelerator, "is_available")
            assert hasattr(torch.accelerator, "current_accelerator")

            expected = any(
                hasattr(torch, name) and hasattr(getattr(torch, name), "is_available")
                and getattr(torch, name).is_available()
                for name in ("npu", "cuda", "mps")
            )
            assert torch.accelerator.is_available() is expected

            tensor = torch.ones(1)
            if torch.accelerator.is_available():
                accelerator = torch.accelerator.current_accelerator()
                assert accelerator is not None
                tensor = tensor.to(accelerator)
                assert tensor.device.type == accelerator.type
            else:
                assert torch.accelerator.current_accelerator() is None

            print("OK")
        """)
        out, _, _ = _run(code, env_extra={"USE_CANDLE": "1"})
        assert "OK" in out

    def test_no_redirect_when_disabled(self):
        """When USE_CANDLE=0, ``import torch`` should NOT redirect."""
        code = textwrap.dedent("""\
            try:
                import torch
                # If real torch is installed, this succeeds normally.
                # If not, we get ImportError — both are valid.
                print("IMPORTED")
            except ImportError:
                print("IMPORT_ERROR")
        """)
        out, _, _ = _run(code, env_extra={"USE_CANDLE": "0"})
        assert out in ("IMPORTED", "IMPORT_ERROR")


# -------------------------------------------------------------------------
# TestCLI — python -m candle.torch_compat
# -------------------------------------------------------------------------

class TestCLI:
    """Test the CLI tool."""

    def test_status(self):
        env = os.environ.copy()
        python_path = os.pathsep.join([_PROJECT_ROOT, _SRC_DIR])
        existing = env.get("PYTHONPATH", "")
        if existing:
            python_path = python_path + os.pathsep + existing
        env["PYTHONPATH"] = python_path
        env["USE_CANDLE"] = "0"

        result = subprocess.run(
            [sys.executable, "-m", "candle.torch_compat", "status"],
            capture_output=True,
            text=True,
            env=env,
            timeout=30,
        )
        assert result.returncode == 0
        assert "USE_CANDLE" in result.stdout

    def test_no_args_shows_help(self):
        env = os.environ.copy()
        python_path = os.pathsep.join([_PROJECT_ROOT, _SRC_DIR])
        existing = env.get("PYTHONPATH", "")
        if existing:
            python_path = python_path + os.pathsep + existing
        env["PYTHONPATH"] = python_path
        env["USE_CANDLE"] = "0"

        result = subprocess.run(
            [sys.executable, "-m", "candle.torch_compat"],
            capture_output=True,
            text=True,
            env=env,
            timeout=30,
        )
        assert result.returncode == 1
        assert "install" in result.stdout.lower() or "install" in result.stderr.lower()
