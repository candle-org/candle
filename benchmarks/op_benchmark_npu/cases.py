"""Op benchmark case definitions for NPU workloads."""

# LLM-style constants used by the existing forward cases.
HIDDEN = 4096
INTERMEDIATE = 11008
HEADS = 32
HEAD_DIM = 128
VOCAB = 32000

SCENARIOS = {
    "infer": {"batch": 1, "seq": 2048, "label": "Inference (batch=1, seq=2048)"},
    "train": {"batch": 4, "seq": 512, "label": "Training (batch=4, seq=512)"},
}

DTYPES = {
    "fp16": "float16",
    "bf16": "bfloat16",
    "fp32": "float32",
}

MODES = {
    "fwd": "Forward",
    "bwd": "Backward",
}


def _clear_grads(*values):
    for value in values:
        if value is None:
            continue
        if isinstance(value, (list, tuple)):
            _clear_grads(*value)
            continue
        if getattr(value, "grad", None) is not None:
            value.grad = None


def _with_cleanup(fn, *values, setup=None):
    def cleanup():
        _clear_grads(*values)
    if setup is not None:
        fn.setup = setup
    fn.cleanup = cleanup
    return fn


def _build_matmul_qkv(torch_mod, F, device, dtype, batch, seq):
    del F
    x = torch_mod.randn((batch, seq, HIDDEN), device=device, dtype=dtype)
    w = torch_mod.randn((HIDDEN, HIDDEN), device=device, dtype=dtype)
    def fn():
        return torch_mod.matmul(x, w)
    return fn


def _build_matmul_ffn_up(torch_mod, F, device, dtype, batch, seq):
    del F
    x = torch_mod.randn((batch, seq, HIDDEN), device=device, dtype=dtype)
    w = torch_mod.randn((HIDDEN, INTERMEDIATE), device=device, dtype=dtype)
    def fn():
        return torch_mod.matmul(x, w)
    return fn


def _build_matmul_ffn_down(torch_mod, F, device, dtype, batch, seq):
    del F
    x = torch_mod.randn((batch, seq, INTERMEDIATE), device=device, dtype=dtype)
    w = torch_mod.randn((INTERMEDIATE, HIDDEN), device=device, dtype=dtype)
    def fn():
        return torch_mod.matmul(x, w)
    return fn


def _build_bmm_attn_scores(torch_mod, F, device, dtype, batch, seq):
    del F
    n = batch * HEADS
    q = torch_mod.randn((n, seq, HEAD_DIM), device=device, dtype=dtype)
    k = torch_mod.randn((n, HEAD_DIM, seq), device=device, dtype=dtype)
    def fn():
        return torch_mod.bmm(q, k)
    return fn


def _build_bmm_attn_output(torch_mod, F, device, dtype, batch, seq):
    del F
    n = batch * HEADS
    s = torch_mod.randn((n, seq, seq), device=device, dtype=dtype)
    v = torch_mod.randn((n, seq, HEAD_DIM), device=device, dtype=dtype)
    def fn():
        return torch_mod.bmm(s, v)
    return fn


def _build_softmax(torch_mod, F, device, dtype, batch, seq):
    n = batch * HEADS
    x = torch_mod.randn((n, seq, seq), device=device, dtype=dtype)
    def fn():
        return F.softmax(x, dim=-1)
    return fn


def _build_rms_norm(torch_mod, F, device, dtype, batch, seq):
    del F
    x = torch_mod.randn((batch, seq, HIDDEN), device=device, dtype=dtype)
    eps = 1e-6
    def fn():
        return x * torch_mod.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return fn


def _build_rms_norm_native(torch_mod, F, device, dtype, batch, seq):
    x = torch_mod.randn((batch, seq, HIDDEN), device=device, dtype=dtype)
    weight = torch_mod.ones(HIDDEN, device=device, dtype=dtype)
    eps = 1e-6
    def fn():
        return F.rms_norm(x, (HIDDEN,), weight, eps)
    return fn


def _build_silu(torch_mod, F, device, dtype, batch, seq):
    x = torch_mod.randn((batch, seq, INTERMEDIATE), device=device, dtype=dtype)
    def fn():
        return F.silu(x)
    return fn


def _build_mul(torch_mod, F, device, dtype, batch, seq):
    del F
    a = torch_mod.randn((batch, seq, INTERMEDIATE), device=device, dtype=dtype)
    b = torch_mod.randn((batch, seq, INTERMEDIATE), device=device, dtype=dtype)
    def fn():
        return torch_mod.mul(a, b)
    return fn


def _build_add(torch_mod, F, device, dtype, batch, seq):
    del F
    a = torch_mod.randn((batch, seq, HIDDEN), device=device, dtype=dtype)
    b = torch_mod.randn((batch, seq, HIDDEN), device=device, dtype=dtype)
    def fn():
        return torch_mod.add(a, b)
    return fn


def _build_rope(torch_mod, F, device, dtype, batch, seq):
    del F
    x = torch_mod.randn((batch, HEADS, seq, HEAD_DIM), device=device, dtype=dtype)
    cos = torch_mod.randn((batch, HEADS, seq, HEAD_DIM), device=device, dtype=dtype)
    sin = torch_mod.randn((batch, HEADS, seq, HEAD_DIM), device=device, dtype=dtype)
    def _rotate_half(t):
        half = HEAD_DIM // 2
        t1 = t[..., :half].contiguous()
        t2 = t[..., half:].contiguous()
        return torch_mod.cat((-t2, t1), dim=-1)
    def fn():
        return x * cos + _rotate_half(x) * sin
    return fn


def _build_cross_entropy(torch_mod, F, device, dtype, batch, seq):
    total = batch * seq
    x = torch_mod.randn((total, VOCAB), device=device, dtype=dtype)
    target = torch_mod.randint(0, VOCAB, (total,), device=device)
    def fn():
        return F.cross_entropy(x, target)
    return fn


def _build_embedding(torch_mod, F, device, dtype, batch, seq):
    weight = torch_mod.randn((VOCAB, HIDDEN), device=device, dtype=dtype)
    idx = torch_mod.randint(0, VOCAB, (batch, seq), device=device)
    def fn():
        return F.embedding(idx, weight)
    return fn


def _build_dropout(torch_mod, F, device, dtype, batch, seq):
    x = torch_mod.randn((batch, seq, HIDDEN), device=device, dtype=dtype)
    def fn():
        return F.dropout(x, p=0.1, training=True)
    return fn


# Backward cases use fixed shapes from benchmarks/perf_candle_vs_torch_npu.py.
def _build_linear_bwd(torch_mod, F, device, dtype, batch, in_features, out_features):
    del F
    x = torch_mod.randn((batch, in_features), device=device, dtype=dtype, requires_grad=True)
    w = torch_mod.randn((in_features, out_features), device=device, dtype=dtype, requires_grad=True)
    loss = None
    def setup():
        nonlocal loss
        loss = torch_mod.matmul(x, w).sum()
    def fn():
        loss.backward()
    return _with_cleanup(fn, x, w, setup=setup)


def _build_linear_bwd_mlp_1024_4096(torch_mod, F, device, dtype, batch, seq):
    del seq
    return _build_linear_bwd(torch_mod, F, device, dtype, 32, 1024, 4096)


def _build_linear_bwd_mlp_4096_4096(torch_mod, F, device, dtype, batch, seq):
    del batch, seq
    return _build_linear_bwd(torch_mod, F, device, dtype, 32, 4096, 4096)


def _build_linear_bwd_xfmr_512_512(torch_mod, F, device, dtype, batch, seq):
    del batch, seq
    return _build_linear_bwd(torch_mod, F, device, dtype, 2 * 128, 512, 512)


def _build_linear_bwd_xfmr_ff1(torch_mod, F, device, dtype, batch, seq):
    del batch, seq
    return _build_linear_bwd(torch_mod, F, device, dtype, 2 * 128, 512, 2048)


def _build_linear_bwd_xfmr_ff2(torch_mod, F, device, dtype, batch, seq):
    del batch, seq
    return _build_linear_bwd(torch_mod, F, device, dtype, 2 * 128, 2048, 512)


def _build_gelu_bwd(torch_mod, F, device, dtype, batch, seq):
    del batch, seq
    x = torch_mod.randn((32, 4096), device=device, dtype=dtype, requires_grad=True)
    loss = None
    def setup():
        nonlocal loss
        loss = F.gelu(x).sum()
    def fn():
        loss.backward()
    return _with_cleanup(fn, x, setup=setup)


def _build_layer_norm_bwd(torch_mod, F, device, dtype, batch, seq):
    del F, batch, seq
    module = torch_mod.nn.LayerNorm(512).to(device).to(dtype)
    x = torch_mod.randn((2, 128, 512), device=device, dtype=dtype, requires_grad=True)
    params = list(module.parameters())
    loss = None
    def setup():
        nonlocal loss
        loss = module(x).sum()
    def fn():
        loss.backward()
    return _with_cleanup(fn, x, params, setup=setup)


def _build_softmax_bwd(torch_mod, F, device, dtype, batch, seq):
    del batch, seq
    x = torch_mod.randn((2, 8, 128, 128), device=device, dtype=dtype, requires_grad=True)
    grad = torch_mod.randn((2, 8, 128, 128), device=device, dtype=dtype)
    loss = None
    def setup():
        nonlocal loss
        loss = (F.softmax(x, dim=-1) * grad).sum()
    def fn():
        loss.backward()
    return _with_cleanup(fn, x, setup=setup)


def _build_matmul_bwd_attn_scores(torch_mod, F, device, dtype, batch, seq):
    del F, batch, seq
    q = torch_mod.randn((2, 8, 128, 64), device=device, dtype=dtype, requires_grad=True)
    k = torch_mod.randn((2, 8, 64, 128), device=device, dtype=dtype, requires_grad=True)
    loss = None
    def setup():
        nonlocal loss
        loss = torch_mod.matmul(q, k).sum()
    def fn():
        loss.backward()
    return _with_cleanup(fn, q, k, setup=setup)


def _build_matmul_bwd_attn_output(torch_mod, F, device, dtype, batch, seq):
    del F, batch, seq
    attn = torch_mod.randn((2, 8, 128, 128), device=device, dtype=dtype, requires_grad=True)
    v = torch_mod.randn((2, 8, 128, 64), device=device, dtype=dtype, requires_grad=True)
    loss = None
    def setup():
        nonlocal loss
        loss = torch_mod.matmul(attn, v).sum()
    def fn():
        loss.backward()
    return _with_cleanup(fn, attn, v, setup=setup)


def _build_conv2d_bwd_resnet(torch_mod, F, device, dtype, batch, seq):
    del F, batch, seq
    module = torch_mod.nn.Conv2d(64, 64, 3, padding=1, bias=False).to(device).to(dtype)
    x = torch_mod.randn((8, 64, 32, 32), device=device, dtype=dtype, requires_grad=True)
    params = list(module.parameters())
    loss = None
    def setup():
        nonlocal loss
        loss = module(x).sum()
    def fn():
        loss.backward()
    return _with_cleanup(fn, x, params, setup=setup)


def _build_batch_norm_bwd_resnet(torch_mod, F, device, dtype, batch, seq):
    del F, batch, seq
    module = torch_mod.nn.BatchNorm2d(64).to(device).to(dtype)
    x = torch_mod.randn((8, 64, 32, 32), device=device, dtype=dtype, requires_grad=True)
    params = list(module.parameters())
    loss = None
    def setup():
        nonlocal loss
        loss = module(x).sum()
    def fn():
        loss.backward()
    return _with_cleanup(fn, x, params, setup=setup)


def _build_relu_bwd_resnet(torch_mod, F, device, dtype, batch, seq):
    del batch, seq
    x = torch_mod.randn((8, 64, 32, 32), device=device, dtype=dtype, requires_grad=True)
    loss = None
    def setup():
        nonlocal loss
        loss = F.relu(x).sum()
    def fn():
        loss.backward()
    return _with_cleanup(fn, x, setup=setup)


def _build_add_bwd_residual(torch_mod, F, device, dtype, batch, seq):
    del F, batch, seq
    a = torch_mod.randn((8, 64, 32, 32), device=device, dtype=dtype, requires_grad=True)
    b = torch_mod.randn((8, 64, 32, 32), device=device, dtype=dtype, requires_grad=True)
    loss = None
    def setup():
        nonlocal loss
        loss = torch_mod.add(a, b).sum()
    def fn():
        loss.backward()
    return _with_cleanup(fn, a, b, setup=setup)


OP_CASES = [
    {"name": "matmul_qkv", "mode": "fwd", "build": _build_matmul_qkv},
    {"name": "matmul_ffn_up", "mode": "fwd", "build": _build_matmul_ffn_up},
    {"name": "matmul_ffn_down", "mode": "fwd", "build": _build_matmul_ffn_down},
    {"name": "bmm_attn_scores", "mode": "fwd", "build": _build_bmm_attn_scores},
    {"name": "bmm_attn_output", "mode": "fwd", "build": _build_bmm_attn_output},
    {"name": "softmax", "mode": "fwd", "build": _build_softmax},
    {"name": "rms_norm_composite", "mode": "fwd", "build": _build_rms_norm},
    {"name": "rms_norm_native", "mode": "fwd", "build": _build_rms_norm_native},
    {"name": "silu", "mode": "fwd", "build": _build_silu},
    {"name": "mul", "mode": "fwd", "build": _build_mul},
    {"name": "add", "mode": "fwd", "build": _build_add},
    {"name": "rope", "mode": "fwd", "build": _build_rope},
    {"name": "cross_entropy", "mode": "fwd", "build": _build_cross_entropy},
    {"name": "embedding", "mode": "fwd", "build": _build_embedding},
    {"name": "dropout", "mode": "fwd", "build": _build_dropout},
    {"name": "linear_bwd_mlp_1024_4096", "mode": "bwd", "build": _build_linear_bwd_mlp_1024_4096},
    {"name": "linear_bwd_mlp_4096_4096", "mode": "bwd", "build": _build_linear_bwd_mlp_4096_4096},
    {"name": "linear_bwd_xfmr_512_512", "mode": "bwd", "build": _build_linear_bwd_xfmr_512_512},
    {"name": "linear_bwd_xfmr_ff1", "mode": "bwd", "build": _build_linear_bwd_xfmr_ff1},
    {"name": "linear_bwd_xfmr_ff2", "mode": "bwd", "build": _build_linear_bwd_xfmr_ff2},
    {"name": "gelu_bwd", "mode": "bwd", "build": _build_gelu_bwd},
    {"name": "layer_norm_bwd", "mode": "bwd", "build": _build_layer_norm_bwd},
    {"name": "softmax_bwd", "mode": "bwd", "build": _build_softmax_bwd},
    {"name": "matmul_bwd_attn_scores", "mode": "bwd", "build": _build_matmul_bwd_attn_scores},
    {"name": "matmul_bwd_attn_output", "mode": "bwd", "build": _build_matmul_bwd_attn_output},
    {"name": "conv2d_bwd_resnet", "mode": "bwd", "build": _build_conv2d_bwd_resnet},
    {"name": "batch_norm_bwd_resnet", "mode": "bwd", "build": _build_batch_norm_bwd_resnet},
    {"name": "relu_bwd_resnet", "mode": "bwd", "build": _build_relu_bwd_resnet},
    {"name": "add_bwd_residual", "mode": "bwd", "build": _build_add_bwd_residual},
]
