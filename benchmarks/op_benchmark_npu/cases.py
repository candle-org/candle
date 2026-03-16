"""Op benchmark case definitions for Llama-7B / Qwen-7B shapes."""

# Model constants
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


def _build_matmul_qkv(torch_mod, F, device, dtype, batch, seq):
    x = torch_mod.randn((batch, seq, HIDDEN), device=device, dtype=dtype)
    w = torch_mod.randn((HIDDEN, HIDDEN), device=device, dtype=dtype)
    def fn():
        return torch_mod.matmul(x, w)
    return fn


def _build_matmul_ffn_up(torch_mod, F, device, dtype, batch, seq):
    x = torch_mod.randn((batch, seq, HIDDEN), device=device, dtype=dtype)
    w = torch_mod.randn((HIDDEN, INTERMEDIATE), device=device, dtype=dtype)
    def fn():
        return torch_mod.matmul(x, w)
    return fn


def _build_matmul_ffn_down(torch_mod, F, device, dtype, batch, seq):
    x = torch_mod.randn((batch, seq, INTERMEDIATE), device=device, dtype=dtype)
    w = torch_mod.randn((INTERMEDIATE, HIDDEN), device=device, dtype=dtype)
    def fn():
        return torch_mod.matmul(x, w)
    return fn


def _build_bmm_attn_scores(torch_mod, F, device, dtype, batch, seq):
    n = batch * HEADS
    q = torch_mod.randn((n, seq, HEAD_DIM), device=device, dtype=dtype)
    k = torch_mod.randn((n, HEAD_DIM, seq), device=device, dtype=dtype)
    def fn():
        return torch_mod.bmm(q, k)
    return fn


def _build_bmm_attn_output(torch_mod, F, device, dtype, batch, seq):
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
    x = torch_mod.randn((batch, seq, HIDDEN), device=device, dtype=dtype)
    eps = 1e-6
    def fn():
        return x * torch_mod.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return fn


def _build_silu(torch_mod, F, device, dtype, batch, seq):
    x = torch_mod.randn((batch, seq, INTERMEDIATE), device=device, dtype=dtype)
    def fn():
        return F.silu(x)
    return fn


def _build_mul(torch_mod, F, device, dtype, batch, seq):
    a = torch_mod.randn((batch, seq, INTERMEDIATE), device=device, dtype=dtype)
    b = torch_mod.randn((batch, seq, INTERMEDIATE), device=device, dtype=dtype)
    def fn():
        return torch_mod.mul(a, b)
    return fn


def _build_add(torch_mod, F, device, dtype, batch, seq):
    a = torch_mod.randn((batch, seq, HIDDEN), device=device, dtype=dtype)
    b = torch_mod.randn((batch, seq, HIDDEN), device=device, dtype=dtype)
    def fn():
        return torch_mod.add(a, b)
    return fn


def _build_rope(torch_mod, F, device, dtype, batch, seq):
    x = torch_mod.randn((batch, HEADS, seq, HEAD_DIM), device=device, dtype=dtype)
    cos = torch_mod.randn((batch, HEADS, seq, HEAD_DIM), device=device, dtype=dtype)
    sin = torch_mod.randn((batch, HEADS, seq, HEAD_DIM), device=device, dtype=dtype)
    def _rotate_half(t):
        half = HEAD_DIM // 2
        t1 = t[..., :half]
        t2 = t[..., half:]
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


OP_CASES = [
    {"name": "matmul_qkv", "build": _build_matmul_qkv},
    {"name": "matmul_ffn_up", "build": _build_matmul_ffn_up},
    {"name": "matmul_ffn_down", "build": _build_matmul_ffn_down},
    {"name": "bmm_attn_scores", "build": _build_bmm_attn_scores},
    {"name": "bmm_attn_output", "build": _build_bmm_attn_output},
    {"name": "softmax", "build": _build_softmax},
    {"name": "rms_norm", "build": _build_rms_norm},
    {"name": "silu", "build": _build_silu},
    {"name": "mul", "build": _build_mul},
    {"name": "add", "build": _build_add},
    {"name": "rope", "build": _build_rope},
    {"name": "cross_entropy", "build": _build_cross_entropy},
    {"name": "embedding", "build": _build_embedding},
    {"name": "dropout", "build": _build_dropout},
]
