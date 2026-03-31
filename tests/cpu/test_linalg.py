"""Tests for torch.linalg module."""

import sys
import os
import math
import types
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import candle as torch


class TestLinalgNorms:
    """Tests for linalg norm functions."""

    def test_vector_norm_l2(self):
        x = torch.tensor([3.0, 4.0])
        result = torch.linalg.vector_norm(x)
        assert abs(result.item() - 5.0) < 1e-5

    def test_vector_norm_l1(self):
        x = torch.tensor([3.0, -4.0])
        result = torch.linalg.vector_norm(x, ord=1)
        assert abs(result.item() - 7.0) < 1e-5

    def test_vector_norm_inf(self):
        x = torch.tensor([3.0, -4.0, 1.0])
        result = torch.linalg.vector_norm(x, ord=float('inf'))
        assert abs(result.item() - 4.0) < 1e-5

    def test_vector_norm_dim(self):
        x = torch.tensor([[3.0, 4.0], [5.0, 12.0]])
        result = torch.linalg.vector_norm(x, dim=1)
        np.testing.assert_allclose(result.numpy(), [5.0, 13.0], atol=1e-5)

    def test_norm(self):
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        result = torch.linalg.norm(x)
        expected = np.linalg.norm(np.array([[1, 2], [3, 4]]))
        assert abs(result.item() - expected) < 1e-5

    def test_norm_with_dim(self):
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        result = torch.linalg.norm(x, dim=1)
        expected = np.linalg.norm(np.array([[1, 2], [3, 4]]), axis=1)
        np.testing.assert_allclose(result.numpy(), expected, atol=1e-5)

    def test_matrix_norm_frobenius(self):
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        result = torch.linalg.matrix_norm(x)
        expected = np.sqrt(1 + 4 + 9 + 16)
        assert abs(result.item() - expected) < 1e-5


class TestLinalgDecompositions:
    """Tests for matrix decompositions."""

    def test_qr(self):
        A = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        Q, R = torch.linalg.qr(A)
        # Q @ R should reconstruct A
        reconstructed = Q.numpy() @ R.numpy()
        np.testing.assert_allclose(reconstructed, A.numpy(), atol=1e-5)

    def test_svd(self):
        A = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        U, S, Vh = torch.linalg.svd(A)
        # U @ diag(S) @ Vh should reconstruct A (need Sigma as m x n)
        m, n = A.shape
        Sigma = np.zeros((m, n))
        np.fill_diagonal(Sigma, S.numpy())
        reconstructed = U.numpy() @ Sigma @ Vh.numpy()
        np.testing.assert_allclose(reconstructed, A.numpy(), atol=1e-5)

    def test_svd_reduced(self):
        A = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        U, S, Vh = torch.linalg.svd(A, full_matrices=False)
        assert U.shape == (3, 2)
        assert S.shape == (2,)
        assert Vh.shape == (2, 2)

    def test_svdvals(self):
        A = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        S = torch.linalg.svdvals(A)
        assert S.shape == (2,)
        assert S.numpy()[0] > S.numpy()[1]  # Singular values in descending order

    def test_cholesky(self):
        # Create a positive definite matrix
        A = torch.tensor([[4.0, 2.0], [2.0, 3.0]])
        L = torch.linalg.cholesky(A)
        reconstructed = L.numpy() @ L.numpy().T
        np.testing.assert_allclose(reconstructed, A.numpy(), atol=1e-5)

    def test_cholesky_upper(self):
        A = torch.tensor([[4.0, 2.0], [2.0, 3.0]])
        U = torch.linalg.cholesky(A, upper=True)
        reconstructed = U.numpy().T @ U.numpy()
        np.testing.assert_allclose(reconstructed, A.numpy(), atol=1e-5)

    def test_eig(self):
        A = torch.tensor([[2.0, 1.0], [1.0, 3.0]])
        eigenvalues, eigenvectors = torch.linalg.eig(A)
        # Verify A @ v = lambda * v for each eigenpair
        for i in range(2):
            lam = eigenvalues.numpy()[i]
            v = eigenvectors.numpy()[:, i]
            np.testing.assert_allclose(A.numpy() @ v, lam * v, atol=1e-5)

    def test_eigh(self):
        A = torch.tensor([[2.0, 1.0], [1.0, 3.0]])
        eigenvalues, eigenvectors = torch.linalg.eigh(A)
        # Eigenvalues should be real and sorted
        vals = eigenvalues.numpy()
        assert vals[0] <= vals[1]

    def test_eigvals(self):
        A = torch.tensor([[2.0, 1.0], [1.0, 3.0]])
        vals = torch.linalg.eigvals(A)
        assert vals.shape == (2,)

    def test_eigvalsh(self):
        A = torch.tensor([[2.0, 1.0], [1.0, 3.0]])
        vals = torch.linalg.eigvalsh(A)
        expected = np.linalg.eigvalsh(np.array([[2, 1], [1, 3]]))
        np.testing.assert_allclose(vals.numpy(), expected, atol=1e-5)


class TestLinalgSolvers:
    """Tests for linear system solvers."""

    def test_solve(self):
        A = torch.tensor([[3.0, 1.0], [1.0, 2.0]])
        B = torch.tensor([[9.0], [8.0]])
        X = torch.linalg.solve(A, B)
        np.testing.assert_allclose(A.numpy() @ X.numpy(), B.numpy(), atol=1e-5)

    def test_inv(self):
        A = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        A_inv = torch.linalg.inv(A)
        identity = A.numpy() @ A_inv.numpy()
        np.testing.assert_allclose(identity, np.eye(2), atol=1e-5)

    def test_pinv(self):
        A = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        A_pinv = torch.linalg.pinv(A)
        # A @ pinv(A) @ A ≈ A
        reconstructed = A.numpy() @ A_pinv.numpy() @ A.numpy()
        np.testing.assert_allclose(reconstructed, A.numpy(), atol=1e-5)

    def test_lstsq(self):
        A = torch.tensor([[1.0, 1.0], [1.0, 2.0], [1.0, 3.0]])
        B = torch.tensor([[1.0], [2.0], [2.0]])
        result = torch.linalg.lstsq(A, B)
        assert result.solution.shape == (2, 1)


class TestLinalgProperties:
    """Tests for matrix properties."""

    def test_det(self):
        A = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        d = torch.linalg.det(A)
        assert abs(d.item() - (-2.0)) < 1e-5

    def test_slogdet(self):
        A = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        sign, logabsdet = torch.linalg.slogdet(A)
        assert abs(sign.item() - (-1.0)) < 1e-5
        assert abs(logabsdet.item() - math.log(2.0)) < 1e-5

    def test_matrix_rank(self):
        A = torch.tensor([[1.0, 2.0], [2.0, 4.0]])  # Rank 1
        rank = torch.linalg.matrix_rank(A)
        assert rank.item() == 1

    def test_matrix_rank_full(self):
        A = torch.tensor([[1.0, 2.0], [3.0, 4.0]])  # Rank 2
        rank = torch.linalg.matrix_rank(A)
        assert rank.item() == 2

    def test_cond(self):
        A = torch.tensor([[1.0, 0.0], [0.0, 1.0]])  # Identity matrix
        c = torch.linalg.cond(A)
        assert abs(c.item() - 1.0) < 1e-5

    def test_matrix_power(self):
        A = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        A2 = torch.linalg.matrix_power(A, 2)
        expected = A.numpy() @ A.numpy()
        np.testing.assert_allclose(A2.numpy(), expected, atol=1e-5)

    def test_matrix_power_zero(self):
        A = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        I = torch.linalg.matrix_power(A, 0)
        np.testing.assert_allclose(I.numpy(), np.eye(2), atol=1e-5)


def test_npu_matmul_out_preserves_user_output_tensor(monkeypatch):
    import candle._backends.npu.ops.linalg as npu_linalg
    from candle._backends.npu.ops._helpers import float_dtype
    from candle._dtype import float16 as float16_dtype
    from candle._dtype import float32 as float32_dtype

    assert float_dtype == float32_dtype


    class FakeTensor:
        def __init__(self, name, dtype=float_dtype, shape=(2, 2), stride=(2, 1), device=None):
            self.name = name
            self.dtype = dtype
            self.shape = shape
            self.stride = stride
            self.offset = 0
            self.device = device or types.SimpleNamespace(type="npu", index=0)
            self.copy_calls = []

        def copy_(self, other):
            self.copy_calls.append(other)
            return self

    class FakeStorage:
        def data_ptr(self):
            return 1234

    runtime = types.SimpleNamespace(device_id=0, _contiguous_stride=lambda shape: (shape[1], 1), _alloc_device=lambda size, runtime=None: 5678)
    stream = types.SimpleNamespace(stream=0)

    a = FakeTensor("a")
    b = FakeTensor("b")
    user_out = FakeTensor("user_out")
    wrapped = FakeTensor("wrapped", dtype=float16_dtype)
    cast_result = FakeTensor("cast_result", dtype=float_dtype)
    cast_input = FakeTensor("cast_input", dtype=float16_dtype)

    monkeypatch.setattr(npu_linalg.npu_runtime, "get_runtime", lambda device_id=0: runtime)
    monkeypatch.setattr(npu_linalg.npu_runtime, "_alloc_device", lambda size, runtime=None: 5678)
    monkeypatch.setattr(npu_linalg.npu_state, "current_stream", lambda device_id=0: stream)
    monkeypatch.setattr(npu_linalg, "_unwrap_storage", lambda tensor: FakeStorage())
    monkeypatch.setattr(npu_linalg, "_dtype_itemsize", lambda dtype: 4)
    monkeypatch.setattr(npu_linalg, "_matmul_out_shape", lambda a_shape, b_shape: (2, 2))
    monkeypatch.setattr(npu_linalg, "_numel", lambda shape: 4)
    monkeypatch.setattr(npu_linalg, "npu_typed_storage_from_ptr", lambda ptr, numel, dtype, device=None: object())
    monkeypatch.setattr(npu_linalg, "_wrap_tensor", lambda storage, shape, stride: wrapped)
    monkeypatch.setattr(npu_linalg, "_use_soc_fallback", lambda op_name: True)

    def fake_cast_tensor_dtype(tensor, dst_dtype):
        if dst_dtype == float16_dtype:
            return cast_input
        if dst_dtype == float_dtype:
            return cast_result
        raise AssertionError(f"unexpected dst_dtype: {dst_dtype}")

    monkeypatch.setattr(npu_linalg, "_cast_tensor_dtype", fake_cast_tensor_dtype)
    monkeypatch.setattr(npu_linalg.aclnn, "matmul", lambda *args, **kwargs: None)

    result = npu_linalg.matmul(a, b, out=user_out)

    assert result is user_out
    assert user_out.copy_calls == [cast_result]
    assert wrapped.copy_calls == []
    assert cast_result.copy_calls == []
    assert result.dtype == float_dtype


class TestLinalgMisc:
    """Tests for misc linalg functions."""

    def test_cross(self):
        a = torch.tensor([1.0, 0.0, 0.0])
        b = torch.tensor([0.0, 1.0, 0.0])
        c = torch.linalg.cross(a, b)
        np.testing.assert_allclose(c.numpy(), [0.0, 0.0, 1.0], atol=1e-5)

    def test_diagonal(self):
        A = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        d = torch.linalg.diagonal(A)
        np.testing.assert_allclose(d.numpy(), [1.0, 4.0], atol=1e-5)

    def test_multi_dot(self):
        A = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        B = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
        C = torch.tensor([[1.0], [0.0]])
        result = torch.linalg.multi_dot([A, B, C])
        expected = A.numpy() @ B.numpy() @ C.numpy()
        np.testing.assert_allclose(result.numpy(), expected, atol=1e-5)

    def test_vander(self):
        x = torch.tensor([1.0, 2.0, 3.0])
        V = torch.linalg.vander(x)
        expected = np.vander([1, 2, 3], increasing=True)
        np.testing.assert_allclose(V.numpy(), expected, atol=1e-5)

    def test_vander_N(self):
        x = torch.tensor([1.0, 2.0, 3.0])
        V = torch.linalg.vander(x, N=4)
        assert V.shape == (3, 4)

    def test_matmul(self):
        A = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        B = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
        result = torch.linalg.matmul(A, B)
        expected = A.numpy() @ B.numpy()
        np.testing.assert_allclose(result.numpy(), expected, atol=1e-5)


class TestLinalgLU:
    """Tests for LU decomposition functions."""

    def test_lu(self):
        A = torch.tensor([[2.0, 1.0], [4.0, 3.0]])
        try:
            P, L, U = torch.linalg.lu(A)
            reconstructed = P.numpy() @ L.numpy() @ U.numpy()
            np.testing.assert_allclose(reconstructed, A.numpy(), atol=1e-5)
        except ImportError:
            pytest.skip("scipy not available")

    def test_lu_factor(self):
        A = torch.tensor([[2.0, 1.0], [4.0, 3.0]])
        try:
            result = torch.linalg.lu_factor(A)
            assert result.LU.shape == (2, 2)
        except ImportError:
            pytest.skip("scipy not available")


class TestLinalgMatrixExp:
    """Tests for matrix exponential."""

    def test_matrix_exp_identity(self):
        try:
            I = torch.tensor([[0.0, 0.0], [0.0, 0.0]])
            result = torch.linalg.matrix_exp(I)
            np.testing.assert_allclose(result.numpy(), np.eye(2), atol=1e-5)
        except ImportError:
            pytest.skip("scipy not available")
