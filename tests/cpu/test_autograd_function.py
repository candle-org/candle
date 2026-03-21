import pytest
import numpy as np
import candle as torch
from candle.autograd import Function
from candle.autograd.engine import backward, grad


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _allclose(t, expected, atol=1e-6):
    arr = t._numpy_view().flatten()
    return np.allclose(arr, expected, atol=atol)


# ---------------------------------------------------------------------------
# 1. Old-style Function (ctx as first param)
# ---------------------------------------------------------------------------

class _DoubleOldStyle(Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.mul(x, torch.tensor([2.0]))

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        return (torch.mul(grad_output, torch.tensor([2.0])),)


def test_old_style_function():
    x = torch.tensor([3.0], requires_grad=True)
    y = _DoubleOldStyle.apply(x)
    assert y.grad_fn is not None
    backward(y.sum())
    # dy/dx = 2, so grad should be [2.0]
    assert _allclose(x.grad, [2.0])


class _CaptureMaterializeFlag(Function):
    captured = None

    @staticmethod
    def forward(ctx, x):
        ctx.set_materialize_grads(False)
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        _CaptureMaterializeFlag.captured = ctx._materialize_grads
        return (grad_output,)


class _DuplicateOutput(Function):
    @staticmethod
    def forward(ctx, x):
        return x.clone(), x.clone()

    @staticmethod
    def backward(ctx, grad_output_a, grad_output_b):
        return (grad_output_a + grad_output_b,)



# ---------------------------------------------------------------------------
# 2. New-style Function (no ctx in forward)
# ---------------------------------------------------------------------------

class _DoubleNewStyle(Function):
    @staticmethod
    def forward(x):
        return torch.mul(x, torch.tensor([2.0]))

    @staticmethod
    def setup_context(ctx, inputs, output):
        (x,) = inputs
        ctx.save_for_backward(x)

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        return (torch.mul(grad_output, torch.tensor([2.0])),)


def test_new_style_function():
    x = torch.tensor([3.0], requires_grad=True)
    y = _DoubleNewStyle.apply(x)
    assert y.grad_fn is not None
    backward(y.sum())
    assert _allclose(x.grad, [2.0])


# ---------------------------------------------------------------------------
# 3. No-grad path — inputs without requires_grad
# ---------------------------------------------------------------------------

def test_no_grad_path():
    x = torch.tensor([3.0])  # requires_grad=False by default
    y = _DoubleOldStyle.apply(x)
    assert y.grad_fn is None


# ---------------------------------------------------------------------------
# 4. mark_non_differentiable
# ---------------------------------------------------------------------------

class _NonDiffFunc(Function):
    @staticmethod
    def forward(ctx, x):
        out = torch.mul(x, torch.tensor([2.0]))
        ctx.mark_non_differentiable(out)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        return (grad_output,)


def test_mark_non_differentiable():
    x = torch.tensor([3.0], requires_grad=True)
    y = _NonDiffFunc.apply(x)
    # Output is marked non-differentiable, so no grad_fn
    assert y.grad_fn is None


# ---------------------------------------------------------------------------
# 5. needs_input_grad
# ---------------------------------------------------------------------------

class _CheckNeedsGrad(Function):
    captured_needs = None

    @staticmethod
    def forward(ctx, x, y):
        _CheckNeedsGrad.captured_needs = ctx.needs_input_grad
        ctx.save_for_backward(x, y)
        return torch.add(x, y)

    @staticmethod
    def backward(ctx, grad_output):
        return (grad_output, grad_output)


def test_needs_input_grad_values():
    x = torch.tensor([1.0], requires_grad=True)
    y = torch.tensor([2.0])  # no grad
    out = _CheckNeedsGrad.apply(x, y)
    assert _CheckNeedsGrad.captured_needs == (True, False)


# ---------------------------------------------------------------------------
# 6. Version check — inplace modification after save_for_backward
# ---------------------------------------------------------------------------

class _SaveAndReturn(Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.mul(x, torch.tensor([1.0]))

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        return (grad_output,)


def test_version_check():
    x = torch.tensor([1.0], requires_grad=True)
    y = _SaveAndReturn.apply(x)
    # Mutate x in-place to bump its version counter
    x._version_counter.bump()
    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        backward(y.sum())


# ---------------------------------------------------------------------------
# 7. Double backward without retain_graph raises RuntimeError
# ---------------------------------------------------------------------------

def test_double_backward_raises():
    x = torch.tensor([3.0], requires_grad=True)
    y = _DoubleOldStyle.apply(x)
    backward(y.sum())
    # Second backward should fail — saved tensors are released
    x.grad = None
    with pytest.raises(RuntimeError, match="Trying to backward through the graph a second time"):
        backward(y.sum())


# ---------------------------------------------------------------------------
# 8. With retain_graph — double backward works
# ---------------------------------------------------------------------------

def test_retain_graph():
    x = torch.tensor([3.0], requires_grad=True)
    y = _DoubleOldStyle.apply(x)
    backward(y.sum(), retain_graph=True)
    first_grad = x.grad._numpy_view().copy()
    x.grad = None
    backward(y.sum())
    second_grad = x.grad._numpy_view().copy()
    assert np.allclose(first_grad, second_grad)


# ---------------------------------------------------------------------------
# 9. Gradient accumulation — .grad is accumulated correctly
# ---------------------------------------------------------------------------

def test_gradient_accumulation():
    x = torch.tensor([3.0], requires_grad=True)
    y = _DoubleOldStyle.apply(x)
    backward(y.sum(), retain_graph=True)
    # grad should be [2.0]
    assert _allclose(x.grad, [2.0])
    # Run backward again without clearing grad -> should accumulate
    backward(y.sum())
    assert _allclose(x.grad, [4.0])


def test_gradient_accumulation_preserves_reference_without_create_graph():
    x = torch.tensor([3.0], requires_grad=True)
    y = _DoubleOldStyle.apply(x)

    backward(y.sum(), retain_graph=True)
    grad_ref = x.grad
    grad_clone = x.grad.clone()

    backward(y.sum())

    assert id(grad_ref) == id(x.grad)
    assert _allclose(grad_ref, (grad_clone * 2)._numpy_view().flatten())


def test_gradient_accumulation_replaces_reference_with_create_graph():
    x = torch.tensor([3.0], requires_grad=True)
    y = _DoubleOldStyle.apply(x)

    backward(y.sum(), retain_graph=True)
    grad_ref = x.grad

    backward(y.sum(), create_graph=True)

    assert id(grad_ref) != id(x.grad)


def test_set_materialize_grads_flag_is_preserved_on_ctx():
    x = torch.tensor([3.0], requires_grad=True)
    y = _CaptureMaterializeFlag.apply(x)
    backward(y.sum())
    assert _CaptureMaterializeFlag.captured is False


def test_custom_function_multiple_outputs_accumulate_input_grad():
    x = torch.tensor([3.0], requires_grad=True)
    out_a, out_b = _DuplicateOutput.apply(x)
    assert out_a.grad_fn is not None
    assert out_b.grad_fn is not None
    backward((out_a + out_b).sum())
    assert _allclose(x.grad, [2.0])


def test_next_functions_accumulate_grad_hooks_observe_tensor_hook():
    events = []

    def tensor_hook(g):
        events.append(("tensor", float(g.item())))
        return g * 2

    def acc_prehook(grads):
        events.append(("acc_pre", float(grads[0].item())))
        return grads

    def acc_posthook(_grad_out, grad_in):
        events.append(("acc_post", float(grad_in[0].item())))

    a = torch.tensor([1.0], requires_grad=True)
    b = a.clone()
    acc = b.grad_fn.next_functions[0][0]

    a.register_hook(tensor_hook)
    acc.register_prehook(acc_prehook)
    acc.register_hook(acc_posthook)

    backward(b.sum())

    assert events == [("tensor", 1.0), ("acc_pre", 2.0), ("acc_post", 2.0)]


def test_tensor_backward_accepts_inputs_subset():
    a = torch.tensor([1.0], requires_grad=True)
    b = torch.tensor([2.0], requires_grad=True)
    out = a + b

    out.backward(inputs=[b])

    assert a.grad is None
    assert _allclose(b.grad, [1.0])



def test_tensor_backward_inputs_skip_unrequested_leaf_hooks():
    a = torch.tensor([1.0], requires_grad=True)
    b = torch.tensor([2.0], requires_grad=True)
    c = a.clone()

    def hook(_grad):
        raise RuntimeError("hook should not run")

    a.register_hook(hook)
    out = a + b + c

    out.sum().backward(inputs=[b])

    assert a.grad is None
    assert _allclose(b.grad, [1.0])



def test_tensor_backward_inputs_skip_unrequested_clone_branch_hooks():
    def tensor_prehook(_grad):
        raise RuntimeError("hook should not run")

    def posthook(_grad_out, _grad_in):
        raise RuntimeError("posthook should not run")

    a = torch.tensor([1.0], requires_grad=True)
    b = torch.tensor([1.0], requires_grad=True)
    c = a.clone()
    acc = c.grad_fn.next_functions[0][0]

    a.register_hook(tensor_prehook)
    acc.register_hook(posthook)

    out = a + b + c
    out.sum().backward(inputs=[b])

    assert a.grad is None
    assert _allclose(b.grad, [1.0])


# ---------------------------------------------------------------------------
# 10. Chain two custom Functions
# ---------------------------------------------------------------------------

class _AddOne(Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.add(x, torch.tensor([1.0]))

    @staticmethod
    def backward(ctx, grad_output):
        return (grad_output,)


def test_chain_functions():
    x = torch.tensor([3.0], requires_grad=True)
    y = _DoubleOldStyle.apply(x)  # y = 2*x
    z = _AddOne.apply(y)          # z = 2*x + 1
    backward(z.sum())
    # dz/dx = 2
    assert _allclose(x.grad, [2.0])


# ---------------------------------------------------------------------------
# 12. autograd.grad() with custom Function
# ---------------------------------------------------------------------------

def test_autograd_grad():
    x = torch.tensor([3.0], requires_grad=True)
    y = _DoubleOldStyle.apply(x)
    (dx,) = grad(y.sum(), (x,))
    assert _allclose(dx, [2.0])
    # x.grad should NOT be set (autograd.grad doesn't accumulate by default)
    assert x.grad is None

def test_autograd_resize_apply_roundtrip_shape():
    import candle.autograd as autograd
    import candle

    x = candle.rand(2, requires_grad=True)
    y = autograd._functions.Resize.apply(x, (2,))
    assert y.shape == (2,)
    y.sum().backward()
    assert x.grad.shape == (2,)


def test_gradient_accumulation_sparse_reference_behavior():
    def sparse_grad():
        grad = torch.sparse_coo_tensor(
            torch.tensor([[1, 1]]).long(), torch.tensor([1.0, 1.0])
        )
        assert grad.is_sparse
        assert grad.layout is torch.sparse_coo
        return grad

    for create_graph in (False, True):
        x = torch.tensor([1.5, 1.5], requires_grad=True)
        x.grad = sparse_grad()
        grad_ref = x.grad
        x.backward(torch.tensor([1.5, 1.5]), create_graph=create_graph)
        assert id(grad_ref) != id(x.grad)

        x = torch.tensor([1.5, 1.5], requires_grad=True)
        x.grad = sparse_grad()
        grad_ref = x.grad
        x.backward(sparse_grad(), create_graph=create_graph)
        assert (id(grad_ref) == id(x.grad)) is (not create_graph)


def test_grad_fn_metadata_does_not_keep_graph_alive():
    import weakref

    def make_ref():
        x = torch.tensor([1.0], requires_grad=True)
        y = x.exp()

        class Payload:
            pass

        payload = Payload()
        y.grad_fn.metadata[0] = payload
        ref = weakref.ref(payload)
        return y, ref

    y, ref = make_ref()
    assert ref() is not None
    del y
    import gc
    gc.collect()
    assert ref() is None




def test_detect_anomaly_warns_and_reports_missing_forward_info():
    import warnings
    import candle.autograd as autograd

    class MyFunc(Function):
        @staticmethod
        def forward(ctx, x):
            return x.clone()

        @staticmethod
        def backward(ctx, grad_output):
            bad = grad_output.clone()
            bad[0] = 0
            bad[0] /= 0
            return (bad,)

    x = torch.tensor([1.0, 2.0], requires_grad=True)
    y = MyFunc.apply(x)

    with warnings.catch_warnings(record=True) as caught:
        with pytest.raises(RuntimeError, match=r"Function 'MyFuncBackward' returned nan values in its 0th output"):
            with autograd.detect_anomaly():
                y.backward(torch.ones_like(y))

    messages = [str(w.message) for w in caught]
    assert any("Anomaly Detection has been enabled" in msg for msg in messages)
    assert any("No forward pass information" in msg for msg in messages)


def test_detect_anomaly_forward_warning_mentions_apply_site():
    import warnings
    import candle.autograd as autograd

    class MyFunc(Function):
        @staticmethod
        def forward(ctx, x):
            return x.clone()

        @staticmethod
        def backward(ctx, grad_output):
            bad = grad_output.clone()
            bad[0] = 0
            bad[0] /= 0
            return (bad,)

    x = torch.tensor([1.0, 2.0], requires_grad=True)

    with warnings.catch_warnings(record=True) as caught:
        with pytest.raises(RuntimeError, match=r"Function 'MyFuncBackward' returned nan values in its 0th output"):
            with autograd.detect_anomaly():
                out = MyFunc.apply(x)
                out.backward(torch.ones_like(out))

    messages = [str(w.message) for w in caught]
    assert any("MyFunc.apply" in msg for msg in messages)


def test_grad_fn_name_matches_custom_function_backward_name():
    x = torch.tensor([1.0], requires_grad=True)
    y = _DoubleOldStyle.apply(x)
    assert y.grad_fn.name() == "_DoubleOldStyleBackward"



def test_autograd_grad_detects_inplace_version_error_for_used_input():
    a = torch.randn(5, requires_grad=True)
    d1 = a + 1
    d2 = d1**2
    d1 += 1

    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        grad(d2.sum(), (a,))



def test_tensor_attribute_deletion_matches_pytorch_grad_behavior():
    x = torch.randn((2, 2), requires_grad=True)
    del x.grad
    assert x.grad is None



def test_tensor_attribute_deletion_protects_autograd_internals():
    x = torch.randn((2, 2), requires_grad=True)
    for name in ("data", "requires_grad", "_grad_fn", "_backward_hooks"):
        with pytest.raises(RuntimeError):
            delattr(x, name)
