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
