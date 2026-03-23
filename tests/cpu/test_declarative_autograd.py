"""Tests for declarative autograd codegen pipeline.

Tests both the codegen tooling (parse, generate) and the generated code
(forward wrappers + backward Nodes) against numerical gradients.
"""
import math
import pytest
import candle as torch

try:
    import yaml as _yaml  # noqa: F401
    _has_yaml = True
except ImportError:
    _has_yaml = False

_skip_no_yaml = pytest.mark.skipif(not _has_yaml, reason="pyyaml not installed")


# ---------------------------------------------------------------------------
# Part 1: Codegen unit tests (parse + generate)
# ---------------------------------------------------------------------------

class TestDerivativesYamlParsing:
    @_skip_no_yaml
    def test_load_derivatives(self):
        from tools.autograd.load_derivatives import load_derivatives
        from pathlib import Path
        yaml_path = Path(__file__).resolve().parents[2] / "tools" / "autograd" / "derivatives.yaml"
        infos = load_derivatives(yaml_path)
        assert len(infos) >= 20
        names = {i.name for i in infos}
        assert "exp" in names
        assert "mul" in names
        assert "matmul" in names

    def test_parse_unary_schema(self):
        from tools.autograd.model import parse_schema
        func_name, args, returns = parse_schema("exp(Tensor self) -> Tensor")
        assert func_name == "exp"
        assert len(args) == 1
        assert args[0].name == "self"
        assert args[0].is_tensor
        assert len(returns) == 1

    def test_parse_binary_schema(self):
        from tools.autograd.model import parse_schema
        func_name, args, returns = parse_schema("add.Tensor(Tensor self, Tensor other) -> Tensor")
        assert func_name == "add.Tensor"
        assert len(args) == 2
        assert args[0].name == "self"
        assert args[1].name == "other"

    def test_parse_schema_with_non_tensor_args(self):
        from tools.autograd.model import parse_schema
        func_name, args, returns = parse_schema("transpose(Tensor self, int dim0, int dim1) -> Tensor")
        assert func_name == "transpose"
        assert len(args) == 3
        assert args[0].is_tensor
        assert not args[1].is_tensor
        assert args[1].type == "int"

    @_skip_no_yaml
    def test_saved_inputs_only_tensors(self):
        """Non-tensor args (shape, dim) should NOT appear in all_saved_inputs."""
        from tools.autograd.load_derivatives import load_derivatives
        from pathlib import Path
        yaml_path = Path(__file__).resolve().parents[2] / "tools" / "autograd" / "derivatives.yaml"
        infos = load_derivatives(yaml_path)
        info_map = {i.name: i for i in infos}

        reshape_info = info_map["reshape"]
        # 'shape' is int[], not a tensor — should not be in saved_inputs
        assert "shape" not in reshape_info.all_saved_inputs

        transpose_info = info_map["transpose"]
        assert "dim0" not in transpose_info.all_saved_inputs
        assert "dim1" not in transpose_info.all_saved_inputs

    @_skip_no_yaml
    def test_differentiability_info_properties(self):
        from tools.autograd.load_derivatives import load_derivatives
        from pathlib import Path
        yaml_path = Path(__file__).resolve().parents[2] / "tools" / "autograd" / "derivatives.yaml"
        infos = load_derivatives(yaml_path)

        mul_info = next(i for i in infos if i.func_name == "mul.Scalar")
        assert mul_info.backward_name == "MulScalarBackward0"
        assert len(mul_info.differentiable_inputs) == 1
        assert not mul_info.is_inplace
        assert not mul_info.is_multi_output


class TestCodegenOutput:
    def test_generated_files_syntax(self):
        """All generated files must be valid Python."""
        import ast
        from pathlib import Path
        gen_dir = Path(__file__).resolve().parents[2] / "src" / "candle" / "_generated"
        for name in ("functions.py", "variable_type.py", "registration.py"):
            path = gen_dir / name
            assert path.exists(), f"{name} not found"
            ast.parse(path.read_text())

    def test_generated_node_classes_exist(self):
        from candle._generated import functions as F
        assert hasattr(F, "ExpBackward0")
        assert hasattr(F, "MulBackward0")
        assert hasattr(F, "MatmulBackward0")
        assert hasattr(F, "ReshapeBackward0")
        assert hasattr(F, "TransposeBackward0")

    def test_generated_wrappers_exist(self):
        from candle._generated import variable_type as VT
        assert callable(getattr(VT, "exp_autograd", None))
        assert callable(getattr(VT, "mul_autograd", None))
        assert callable(getattr(VT, "add_autograd", None))
        assert callable(getattr(VT, "matmul_autograd", None))

    def test_node_has_expected_interface(self):
        from candle._generated.functions import ExpBackward0
        # Should accept inputs + keyset kwargs
        x = torch.tensor([1.0], requires_grad=True)
        node = ExpBackward0((x,), raw_keyset=None, active_keyset=None)
        assert node.name() == "ExpBackward0"
        assert hasattr(node, "_save")
        assert hasattr(node, "backward")


# ---------------------------------------------------------------------------
# Part 2: Numerical gradient checks for generated backward
# ---------------------------------------------------------------------------

def _numerical_grad(fn, x, eps=1e-5):
    """Compute numerical gradient of scalar fn w.r.t. 1-D tensor x."""
    x_np = x.tolist()
    grads = []
    for i in range(len(x_np)):
        x_plus = x_np.copy()
        x_minus = x_np.copy()
        x_plus[i] += eps
        x_minus[i] -= eps
        f_plus = fn(torch.tensor(x_plus, dtype=torch.float64)).item()
        f_minus = fn(torch.tensor(x_minus, dtype=torch.float64)).item()
        grads.append((f_plus - f_minus) / (2 * eps))
    return grads


class TestGeneratedBackwardNumerical:
    """Test generated backward formulas against numerical gradients."""

    def _check(self, fn, x_vals, atol=1e-4):
        x = torch.tensor(x_vals, dtype=torch.float64, requires_grad=True)
        y = fn(x)
        if y.numel() > 1:
            y = y.sum()
        y.backward()
        analytic = x.grad.tolist()
        numerical = _numerical_grad(lambda t: fn(t).sum() if fn(t).numel() > 1 else fn(t), x)
        for a, n in zip(analytic, numerical):
            assert abs(a - n) < atol, f"analytic={a}, numerical={n}"

    def test_exp(self):
        self._check(torch.exp, [0.5, 1.0, -0.5])

    def test_log(self):
        self._check(torch.log, [0.5, 1.0, 2.0])

    def test_sqrt(self):
        self._check(torch.sqrt, [1.0, 4.0, 9.0])

    def test_neg(self):
        self._check(torch.neg, [1.0, -2.0, 3.0])

    def test_abs(self):
        self._check(torch.abs, [1.0, -2.0, 3.0])

    def test_sigmoid(self):
        self._check(torch.sigmoid, [0.0, 1.0, -1.0])

    def test_tanh(self):
        self._check(torch.tanh, [0.0, 0.5, -0.5])

    def test_sin(self):
        self._check(torch.sin, [0.0, 1.0, -1.0])

    def test_cos(self):
        self._check(torch.cos, [0.0, 1.0, -1.0])

    def test_relu(self):
        self._check(torch.relu, [1.0, -1.0, 0.5])

    def test_add(self):
        x = torch.tensor([1.0, 2.0], requires_grad=True)
        y = torch.tensor([3.0, 4.0], requires_grad=True)
        z = (x + y).sum()
        z.backward()
        assert x.grad.tolist() == [1.0, 1.0]
        assert y.grad.tolist() == [1.0, 1.0]

    def test_mul(self):
        x = torch.tensor([2.0, 3.0], requires_grad=True)
        y = torch.tensor([4.0, 5.0], requires_grad=True)
        z = (x * y).sum()
        z.backward()
        assert x.grad.tolist() == [4.0, 5.0]
        assert y.grad.tolist() == [2.0, 3.0]

    def test_sub(self):
        x = torch.tensor([5.0, 6.0], requires_grad=True)
        y = torch.tensor([1.0, 2.0], requires_grad=True)
        z = (x - y).sum()
        z.backward()
        assert x.grad.tolist() == [1.0, 1.0]
        assert y.grad.tolist() == [-1.0, -1.0]

    def test_div(self):
        x = torch.tensor([6.0, 8.0], requires_grad=True)
        y = torch.tensor([2.0, 4.0], requires_grad=True)
        z = (x / y).sum()
        z.backward()
        # dx = 1/y, dy = -x/y^2
        assert abs(x.grad.tolist()[0] - 0.5) < 1e-5
        assert abs(x.grad.tolist()[1] - 0.25) < 1e-5
        assert abs(y.grad.tolist()[0] - (-1.5)) < 1e-5
        assert abs(y.grad.tolist()[1] - (-0.5)) < 1e-5

    def test_matmul(self):
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        y = torch.tensor([[5.0, 6.0], [7.0, 8.0]], requires_grad=True)
        z = torch.matmul(x, y).sum()
        z.backward()
        assert x.grad is not None
        assert y.grad is not None
        # dx = ones @ y^T, dy = x^T @ ones
        assert x.grad.shape == (2, 2)
        assert y.grad.shape == (2, 2)

    def test_reshape(self):
        x = torch.tensor([[1.0, 2.0, 3.0]], requires_grad=True)
        y = x.reshape(3)
        y.sum().backward()
        assert x.grad.tolist() == [[1.0, 1.0, 1.0]]

    def test_transpose(self):
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        y = x.transpose(0, 1)
        y.sum().backward()
        assert x.grad.tolist() == [[1.0, 1.0], [1.0, 1.0]]

    def test_sum(self):
        x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        y = x.sum()
        y.backward()
        assert x.grad.tolist() == [1.0, 1.0, 1.0]

    def test_chain_exp_log(self):
        """Test chain rule: d/dx log(exp(x)) = 1."""
        x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        y = torch.log(torch.exp(x)).sum()
        y.backward()
        for g in x.grad.tolist():
            assert abs(g - 1.0) < 1e-4

    def test_chain_mul_add(self):
        """Test chain: d/dx (x*x + x) = 2x + 1."""
        x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        y = (x * x + x).sum()
        y.backward()
        expected = [3.0, 5.0, 7.0]  # 2x + 1
        for g, e in zip(x.grad.tolist(), expected):
            assert abs(g - e) < 1e-4


# ---------------------------------------------------------------------------
# Part 3: Phase 1 — Activation backward numerical checks
# ---------------------------------------------------------------------------

class TestActivationBackwardNumerical:
    """Test generated activation backward formulas against numerical gradients."""

    def _check(self, fn, x_vals, atol=1e-4):
        x = torch.tensor(x_vals, dtype=torch.float64, requires_grad=True)
        y = fn(x)
        if y.numel() > 1:
            y = y.sum()
        y.backward()
        analytic = x.grad.tolist()
        numerical = _numerical_grad(lambda t: fn(t).sum() if fn(t).numel() > 1 else fn(t), x)
        for a, n in zip(analytic, numerical):
            assert abs(a - n) < atol, f"analytic={a}, numerical={n}"

    def test_erf(self):
        self._check(torch.erf, [0.0, 0.5, -0.5, 1.0])

    def test_softplus(self):
        self._check(torch.nn.functional.softplus, [0.0, 1.0, -1.0, 2.0])

    def test_silu(self):
        self._check(torch.nn.functional.silu, [0.0, 1.0, -1.0, 2.0])

    def test_gelu(self):
        self._check(torch.nn.functional.gelu, [0.0, 1.0, -1.0, 0.5])

    def test_mish(self):
        self._check(torch.nn.functional.mish, [0.0, 1.0, -1.0, 0.5])

    def test_leaky_relu(self):
        self._check(lambda x: torch.nn.functional.leaky_relu(x, 0.01), [1.0, -1.0, 0.5, -0.5])

    def test_elu(self):
        self._check(lambda x: torch.nn.functional.elu(x, 1.0), [1.0, -1.0, 0.5, -0.5])

    def test_celu(self):
        self._check(lambda x: torch.nn.functional.celu(x, 1.0), [1.0, -1.0, 0.5, -0.5])

    def test_hardtanh(self):
        self._check(lambda x: torch.nn.functional.hardtanh(x, -1.0, 1.0), [0.5, -0.5, 0.0, 0.9])

    def test_relu6(self):
        self._check(torch.nn.functional.relu6, [1.0, -1.0, 3.0, 5.0])

    def test_hardswish(self):
        self._check(torch.nn.functional.hardswish, [0.0, 1.0, -1.0, 4.0])

    def test_hardsigmoid(self):
        self._check(torch.nn.functional.hardsigmoid, [0.0, 1.0, -1.0, 4.0])

    def test_selu(self):
        self._check(torch.nn.functional.selu, [1.0, -1.0, 0.5, -0.5])

    def test_softmax(self):
        x = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float64, requires_grad=True)
        y = torch.nn.functional.softmax(x, dim=-1)
        # Pick one output element to get a scalar loss
        loss = y[0, 0]
        loss.backward()
        assert x.grad is not None
        # softmax grad: s_i * (delta_ij - s_j)
        s = y[0].tolist()
        expected = [s[0] * (1 - s[0]), s[0] * (-s[1]), s[0] * (-s[2])]
        for a, e in zip(x.grad[0].tolist(), expected):
            assert abs(a - e) < 1e-4, f"analytic={a}, expected={e}"

    def test_log_softmax(self):
        x = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float64, requires_grad=True)
        y = torch.nn.functional.log_softmax(x, dim=-1)
        loss = y[0, 0]
        loss.backward()
        assert x.grad is not None
        # log_softmax grad for element 0: 1 - softmax(x)[0] for dim 0, -softmax(x)[j] for others
        import math
        vals = [1.0, 2.0, 3.0]
        max_v = max(vals)
        exps = [math.exp(v - max_v) for v in vals]
        s_sum = sum(exps)
        s = [e / s_sum for e in exps]
        expected = [1 - s[0], -s[1], -s[2]]
        for a, e in zip(x.grad[0].tolist(), expected):
            assert abs(a - e) < 1e-4, f"analytic={a}, expected={e}"

    def test_prelu(self):
        x = torch.tensor([1.0, -2.0, 3.0, -0.5], dtype=torch.float64, requires_grad=True)
        w = torch.tensor([0.25], dtype=torch.float64, requires_grad=True)
        y = torch.nn.functional.prelu(x, w).sum()
        y.backward()
        # For x > 0: grad_x = 1, for x <= 0: grad_x = w
        expected_x = [1.0, 0.25, 1.0, 0.25]
        for a, e in zip(x.grad.tolist(), expected_x):
            assert abs(a - e) < 1e-4, f"analytic={a}, expected={e}"
        # grad_w = sum of x[i] where x[i] <= 0
        expected_w = -2.0 + (-0.5)  # = -2.5
        assert abs(w.grad.item() - expected_w) < 1e-4


# ---------------------------------------------------------------------------
# Part 4: Phase 1 — Norm backward numerical checks
# ---------------------------------------------------------------------------

class TestNormBackwardNumerical:
    """Test generated norm backward formulas against numerical gradients."""

    def test_layer_norm(self):
        x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                         dtype=torch.float64, requires_grad=True)
        y = torch.nn.functional.layer_norm(x, (3,))
        y.sum().backward()
        assert x.grad is not None
        assert x.grad.shape == (2, 3)
        # Numerical check
        eps = 1e-5
        x_np = x.tolist()
        grads = []
        for i in range(2):
            for j in range(3):
                xp = [row[:] for row in x_np]
                xm = [row[:] for row in x_np]
                xp[i][j] += eps
                xm[i][j] -= eps
                tp = torch.tensor(xp, dtype=torch.float64)
                tm = torch.tensor(xm, dtype=torch.float64)
                fp = torch.nn.functional.layer_norm(tp, (3,)).sum().item()
                fm = torch.nn.functional.layer_norm(tm, (3,)).sum().item()
                grads.append((fp - fm) / (2 * eps))
        analytic = [x.grad[i][j].item() for i in range(2) for j in range(3)]
        for a, n in zip(analytic, grads):
            assert abs(a - n) < 1e-4, f"analytic={a}, numerical={n}"

    def test_layer_norm_with_weight(self):
        x = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float64, requires_grad=True)
        w = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        b = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64)
        y = torch.nn.functional.layer_norm(x, (3,), weight=w, bias=b)
        y.sum().backward()
        assert x.grad is not None
        assert x.grad.shape == (1, 3)

    def test_batch_norm(self):
        x = torch.tensor([[[1.0, 2.0], [3.0, 4.0]],
                          [[5.0, 6.0], [7.0, 8.0]]],
                         dtype=torch.float64, requires_grad=True)
        # x shape: (2, 2, 2) — N=2, C=2, spatial=2
        y = torch.nn.functional.batch_norm(x, None, None, training=True)
        y.sum().backward()
        assert x.grad is not None
        assert x.grad.shape == (2, 2, 2)

    def test_group_norm(self):
        x = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]],
                           [[5.0, 6.0], [7.0, 8.0]]]],
                         dtype=torch.float64, requires_grad=True)
        # x shape: (1, 2, 2, 2) — N=1, C=2, H=2, W=2
        y = torch.nn.functional.group_norm(x, num_groups=2)
        y.sum().backward()
        assert x.grad is not None
        assert x.grad.shape == (1, 2, 2, 2)

    def test_rms_norm(self):
        # rms_norm is not in torch.nn.functional, use candle directly
        x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                         dtype=torch.float64, requires_grad=True)
        from candle._functional import rms_norm
        y = rms_norm(x, (3,))
        y.sum().backward()
        assert x.grad is not None
        assert x.grad.shape == (2, 3)


# ---------------------------------------------------------------------------
# Part 5: Cython compiled-module contract tests (Task 1)
# ---------------------------------------------------------------------------
# These tests pin the contract that future compiled Cython modules must satisfy.
# They are expected to FAIL until Tasks 3–5 are implemented (modules compiled).
# Do NOT skip or xfail them — they must fail loudly so CI catches regressions.

def test_generated_cython_variable_type_module_exists():
    """candle._generated._variable_type_cy must expose autograd wrapper callables."""
    import importlib
    mod = importlib.import_module("candle._generated._variable_type_cy")
    assert hasattr(mod, "exp_autograd"), "_variable_type_cy missing exp_autograd"
    assert hasattr(mod, "matmul_autograd"), "_variable_type_cy missing matmul_autograd"
    assert hasattr(mod, "mul_tensor_autograd"), "_variable_type_cy missing mul_tensor_autograd"


def test_generated_cython_functions_module_exists():
    """candle._generated._functions_cy must expose backward Node classes."""
    import importlib
    mod = importlib.import_module("candle._generated._functions_cy")
    assert hasattr(mod, "ExpBackward0"), "_functions_cy missing ExpBackward0"
    assert hasattr(mod, "MulBackward0"), "_functions_cy missing MulBackward0"
    assert hasattr(mod, "MatmulBackward0"), "_functions_cy missing MatmulBackward0"


# ---------------------------------------------------------------------------
# Parity smoke tests: registration surface and class-name preservation
# ---------------------------------------------------------------------------

def test_registration_surface_exports_generated_variable_type_wrappers():
    """registration.py must expose register_generated_autograd_kernels and
    variable_type.py must have exp_autograd / exp_autograd_post callables."""
    from candle._generated import registration as reg
    from candle._generated import variable_type as vt

    assert callable(getattr(vt, "exp_autograd", None)), "vt.exp_autograd not callable"
    assert callable(getattr(vt, "exp_autograd_post", None)), "vt.exp_autograd_post not callable"
    assert hasattr(reg, "register_generated_autograd_kernels"), (
        "registration missing register_generated_autograd_kernels"
    )


def test_py_functions_backward_class_names():
    """Python functions.py must expose the expected backward Node class names.
    This is the baseline that must hold regardless of any Cython module state."""
    from candle._generated import functions as py_functions

    assert hasattr(py_functions, "ExpBackward0"), "functions.py missing ExpBackward0"
    assert type(py_functions.ExpBackward0.__name__) is str


def test_cy_functions_backward_class_names():
    """Future: _functions_cy must expose the same class names as functions.py
    so registration.py can swap them transparently. Requires Task 2 Cython build."""
    import importlib
    cy_functions = importlib.import_module("candle._generated._functions_cy")
    assert hasattr(cy_functions, "ExpBackward0"), (
        "_functions_cy missing ExpBackward0 — compile the Cython module first"
    )
    assert hasattr(cy_functions, "MulBackward0"), (
        "_functions_cy missing MulBackward0"
    )


@_skip_no_yaml
def test_generated_variable_type_cython_header_contains_cached_refs(tmp_path):
    """_variable_type_cy.pyx must contain cached module-level refs and _ensure_refs.

    This test drives Task 3: it fails while gen_variable_type_pyx is a placeholder
    stub.  It passes once gen_variable_type_pyx emits the full Cython header with
    cdef object refs for GradMode, annotate_node_creation, etc.
    """
    from tools.autograd.gen_variable_type import gen_variable_type_pyx
    from tools.autograd.load_derivatives import load_derivatives
    from pathlib import Path

    yaml_path = Path(__file__).resolve().parents[2] / "tools" / "autograd" / "derivatives.yaml"
    infos = load_derivatives(yaml_path)
    content = gen_variable_type_pyx(infos)

    # Must have the Cython language_level directive
    assert "# cython: language_level=3" in content, "missing language_level directive"

    # Must have wraparound=False for safety
    assert "wraparound=False" in content, "missing wraparound=False directive"

    # Must have module-level cdef object refs for all required symbols
    assert "cdef object _GradMode" in content, "missing cdef object _GradMode"
    assert "cdef object _annotate_node_creation" in content, "missing cdef object _annotate_node_creation"
    assert "cdef object _strip_autograd_keys" in content, "missing cdef object _strip_autograd_keys"
    assert "cdef object _current_dispatch_keyset" in content, "missing cdef object _current_dispatch_keyset"
    assert "cdef object _redispatch" in content, "missing cdef object _redispatch"
    assert "cdef object _F" in content, "missing cdef object _F"

    # Must have the lazy-init helper
    assert "cdef inline void _ensure_refs()" in content, "missing _ensure_refs() function"

    # Must have wrapper functions for key ops
    assert "def exp_autograd(" in content, "missing exp_autograd wrapper"
    assert "def matmul_autograd(" in content or "def matmul_tensor_autograd(" in content, \
        "missing matmul_autograd wrapper"
    assert "def mul_tensor_autograd(" in content, "missing mul_tensor_autograd wrapper"

    # Must have post-wrapper functions
    assert "def exp_autograd_post(" in content, "missing exp_autograd_post wrapper"

    # Must NOT use negative indexing (wraparound=False footgun)
    import re
    neg_index_matches = re.findall(r'\[\s*-\d+\s*\]', content)
    assert not neg_index_matches, (
        f"Negative indexing found in generated .pyx (wraparound=False footgun): "
        f"{neg_index_matches[:3]}"
    )

    # Also verify gen_autograd writes the content to disk
    from tools.autograd.gen_autograd import main
    main(yaml_path, tmp_path)
    pyx_path = tmp_path / "_variable_type_cy.pyx"
    assert pyx_path.exists(), "gen_autograd did not write _variable_type_cy.pyx"
    disk_content = pyx_path.read_text()
    assert "cdef object _GradMode" in disk_content, \
        "written _variable_type_cy.pyx missing cached refs"


@_skip_no_yaml
def test_generated_functions_cython_header_contains_cached_refs(tmp_path):
    """_functions_cy.pyx must contain cached module-level refs and _ensure_refs.

    This test drives Task 4: it fails while gen_functions_pyx is a placeholder
    stub.  It passes once gen_functions_pyx emits the full Cython header with
    cdef object refs for Node, grad_context, redispatch, etc.
    """
    from tools.autograd.gen_functions import gen_functions_pyx
    from tools.autograd.load_derivatives import load_derivatives
    from pathlib import Path

    yaml_path = Path(__file__).resolve().parents[2] / "tools" / "autograd" / "derivatives.yaml"
    infos = load_derivatives(yaml_path)
    content = gen_functions_pyx(infos)

    # Must have the Cython language_level directive
    assert "# cython: language_level=3" in content, "missing language_level directive"

    # Must have wraparound=False for safety
    assert "wraparound=False" in content, "missing wraparound=False directive"

    # Must have module-level cdef object refs for all required symbols
    # _Node is now eagerly imported (not cdef), so check for import line instead
    assert "from candle.autograd.node import Node as _Node" in content, "missing eager _Node import"
    assert "cdef object _grad_context" in content, "missing cdef object _grad_context"
    assert "cdef object _strip_autograd_keys" in content, "missing cdef object _strip_autograd_keys"
    assert "cdef object _backward_dispatch_keyset" in content, "missing cdef object _backward_dispatch_keyset"
    assert "cdef object _scalar_tensor_like" in content, "missing cdef object _scalar_tensor_like"
    assert "cdef object _redispatch" in content, "missing cdef object _redispatch"
    assert "cdef object _reduce_grad" in content, "missing cdef object _reduce_grad"
    assert "cdef object _current_dispatch_keyset" in content, "missing cdef object _current_dispatch_keyset"

    # Must have the lazy-init helper
    assert "cdef inline void _ensure_refs()" in content, "missing _ensure_refs() function"

    # Must have backward node classes for key ops
    assert "class ExpBackward0" in content, "missing ExpBackward0 class"
    assert "class MulBackward0" in content or "class MulTensorBackward0" in content, \
        "missing MulBackward0/MulTensorBackward0 class"
    assert "class MatmulBackward0" in content or "class MatmulTensorBackward0" in content, \
        "missing MatmulBackward0/MatmulTensorBackward0 class"

    # Must NOT use negative indexing (wraparound=False footgun)
    import re
    neg_index_matches = re.findall(r'\[\s*-\d+\s*\]', content)
    assert not neg_index_matches, (
        f"Negative indexing found in generated .pyx (wraparound=False footgun): "
        f"{neg_index_matches[:3]}"
    )

    # Cython-safe generator contract: avoid known compile-blocker free names.
    blocker_patterns = [
        r'\b_raise_not_implemented\(',
        r'\b_as_strided_backward_helper\(',
        r'\b_erf_backward_helper\(',
        r'\b_hardsigmoid_backward_helper\(',
        r'\b_hardswish_backward_helper\(',
        r'\bM_PI\b',
        r'\bM_LN2\b',
        r'\bgrad_sign\b',
        r'\bgrad_logabsdet\b',
        r'\bgrad_L\b',
        r'\bgrad_U\b',
        r'\bgrads\[0\]',
        r'\bgrads\[1\]',
        r'\bgrads\[2\]',
    ]
    unexpected = [pattern for pattern in blocker_patterns if re.search(pattern, content)]
    assert not unexpected, f"known Cython compile blockers still present: {unexpected[:8]}"

    # Also verify gen_autograd writes the content to disk
    from tools.autograd.gen_autograd import main
    main(yaml_path, tmp_path)
    pyx_path = tmp_path / "_functions_cy.pyx"
    assert pyx_path.exists(), "gen_autograd did not write _functions_cy.pyx"
    disk_content = pyx_path.read_text()
    assert "from candle.autograd.node import Node as _Node" in disk_content, \
        "written _functions_cy.pyx missing eager _Node import"


def test_generated_cython_modules_import_after_build():
    """Smoke test: both generated Cython modules must be importable after build.

    This test drives Task 5: it fails before the generated .pyx files are
    compiled and installed.  It passes once setup.py includes them and the
    editable package is reinstalled.
    """
    import importlib
    importlib.import_module("candle._generated._variable_type_cy")
    importlib.import_module("candle._generated._functions_cy")


# ---------------------------------------------------------------------------
# Part 7: Generated package Cython availability flag
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Part A5: Runtime preference after registration split
# ---------------------------------------------------------------------------

def test_registration_prefers_compiled_generated_variable_type_for_generated_safe_ops():
    import importlib

    cy_vt = importlib.import_module("candle._generated._variable_type_cy")
    reg = importlib.import_module("candle._generated.registration")

    assert hasattr(cy_vt, "exp_autograd")
    assert hasattr(reg, "register_generated_autograd_kernels")

    ns = {}
    exec(
        "from candle._generated import variable_type as _VT_PY\n"
        "try:\n"
        "    from candle._generated import _variable_type_cy as _VT_CY\n"
        "except ImportError:\n"
        "    _VT_CY = None\n"
        "_VT = _VT_CY if _VT_CY is not None else _VT_PY\n",
        ns,
    )
    assert ns["_VT"].__name__.endswith("_variable_type_cy")


def test_registration_keeps_python_fallback_for_legacy_ops():
    import importlib

    py_vt = importlib.import_module("candle._generated.variable_type")
    reg_text = importlib.import_module("candle._generated.registration").__file__

    assert hasattr(py_vt, "sum_to_size_autograd_post")
    assert hasattr(py_vt, "diff_autograd")
    assert hasattr(py_vt, "diff_autograd_post")

    from pathlib import Path
    text = Path(reg_text).read_text()
    legacy = text.split("# === UPSTREAM LEGACY REGISTRATIONS ===", 1)[1]
    assert "_VT_PY.sum_to_size_autograd_post" in legacy
    assert "_VT_PY.diff_autograd" in legacy
    assert "_VT_PY.diff_autograd_post" in legacy
def test_gen_autograd_writes_cython_outputs(tmp_path):
    """gen_autograd.main() must write both .pyx outputs alongside the .py outputs.

    This test drives Task 2: it fails today because gen_autograd only writes
    functions.py / variable_type.py / registration.py.  It will pass once
    gen_autograd is extended to call gen_functions_pyx and gen_variable_type_pyx.
    """
    from tools.autograd.gen_autograd import main
    from pathlib import Path

    yaml_path = Path(__file__).resolve().parents[2] / "tools" / "autograd" / "derivatives.yaml"
    main(yaml_path, tmp_path)

    # Existing .py outputs must still be written
    assert (tmp_path / "functions.py").exists(), "functions.py missing"
    assert (tmp_path / "variable_type.py").exists(), "variable_type.py missing"
    assert (tmp_path / "registration.py").exists(), "registration.py missing"

    # New .pyx outputs required by Task 2
    assert (tmp_path / "_functions_cy.pyx").exists(), (
        "_functions_cy.pyx not written — update gen_autograd to call gen_functions_pyx"
    )
    assert (tmp_path / "_variable_type_cy.pyx").exists(), (
        "_variable_type_cy.pyx not written — update gen_autograd to call gen_variable_type_pyx"
    )

