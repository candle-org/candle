def _case_a1(torch_mod, F, device, dtype):
    x = torch_mod.randn((1, 128, 1024), device=device, dtype=dtype)
    w1 = torch_mod.randn((1024, 1024), device=device, dtype=dtype)
    w2 = torch_mod.randn((1024, 1024), device=device, dtype=dtype)

    def forward():
        y = (x @ w1).relu()
        z = y @ w2
        return z

    return forward


def _case_a2(torch_mod, F, device, dtype):
    b, s, h, heads = 1, 512, 1024, 16
    x = torch_mod.randn((b, s, h), device=device, dtype=dtype)
    wq = torch_mod.randn((h, h), device=device, dtype=dtype)
    wk = torch_mod.randn((h, h), device=device, dtype=dtype)
    wv = torch_mod.randn((h, h), device=device, dtype=dtype)
    wo = torch_mod.randn((h, h), device=device, dtype=dtype)
    head_dim = h // heads

    def forward():
        q = x @ wq
        k = x @ wk
        v = x @ wv
        q = q.reshape((b, s, heads, head_dim)).transpose(1, 2)
        k = k.reshape((b, s, heads, head_dim)).transpose(1, 2)
        v = v.reshape((b, s, heads, head_dim)).transpose(1, 2)
        attn = torch_mod.matmul(q.contiguous(), k.contiguous().transpose(-2, -1))
        attn = F.softmax(attn, dim=-1)
        out = torch_mod.matmul(attn.contiguous(), v.contiguous())
        out = out.transpose(1, 2).reshape((b, s, h))
        return out @ wo

    return forward


def _case_a2s(torch_mod, F, device, dtype):
    b, s, h, heads = 2, 256, 512, 8
    x = torch_mod.randn((b, s, h), device=device, dtype=dtype)
    wq = torch_mod.randn((h, h), device=device, dtype=dtype)
    wk = torch_mod.randn((h, h), device=device, dtype=dtype)
    wv = torch_mod.randn((h, h), device=device, dtype=dtype)
    wo = torch_mod.randn((h, h), device=device, dtype=dtype)
    head_dim = h // heads

    def forward():
        q = x @ wq
        k = x @ wk
        v = x @ wv
        q = q.reshape((b, s, heads, head_dim)).transpose(1, 2)
        k = k.reshape((b, s, heads, head_dim)).transpose(1, 2)
        v = v.reshape((b, s, heads, head_dim)).transpose(1, 2)
        attn = torch_mod.matmul(q.contiguous(), k.contiguous().transpose(-2, -1))
        attn = F.softmax(attn, dim=-1)
        out = torch_mod.matmul(attn.contiguous(), v.contiguous())
        out = out.transpose(1, 2).reshape((b, s, h))
        return out @ wo

    return forward


def _case_a3(torch_mod, F, device, dtype):
    b, s, h = 8, 128, 2048
    x = torch_mod.randn((b, s, h), device=device, dtype=dtype)
    w1 = torch_mod.randn((h, 4 * h), device=device, dtype=dtype)
    w2 = torch_mod.randn((4 * h, h), device=device, dtype=dtype)

    def forward():
        y = F.silu(x @ w1)
        return y @ w2

    return forward


def _block(torch_mod, F, device, dtype, *, b, s, h, heads):
    x = torch_mod.randn((b, s, h), device=device, dtype=dtype)
    wq = torch_mod.randn((h, h), device=device, dtype=dtype)
    wk = torch_mod.randn((h, h), device=device, dtype=dtype)
    wv = torch_mod.randn((h, h), device=device, dtype=dtype)
    wo = torch_mod.randn((h, h), device=device, dtype=dtype)
    w1 = torch_mod.randn((h, 4 * h), device=device, dtype=dtype)
    w2 = torch_mod.randn((4 * h, h), device=device, dtype=dtype)
    head_dim = h // heads

    def forward():
        y = F.layer_norm(x, (h,))
        q = y @ wq
        k = y @ wk
        v = y @ wv
        q = q.reshape((b, s, heads, head_dim)).transpose(1, 2)
        k = k.reshape((b, s, heads, head_dim)).transpose(1, 2)
        v = v.reshape((b, s, heads, head_dim)).transpose(1, 2)
        attn = torch_mod.matmul(q.contiguous(), k.contiguous().transpose(-2, -1))
        attn = F.softmax(attn, dim=-1)
        out = torch_mod.matmul(attn.contiguous(), v.contiguous())
        out = out.transpose(1, 2).reshape((b, s, h))
        x1 = x + (out @ wo)
        z = F.layer_norm(x1, (h,))
        z = F.gelu(z @ w1)
        return x1 + (z @ w2)

    return forward


def _case_b1(torch_mod, F, device, dtype):
    return _block(torch_mod, F, device, dtype, b=1, s=512, h=2048, heads=16)


def _case_b1s(torch_mod, F, device, dtype):
    return _block(torch_mod, F, device, dtype, b=2, s=256, h=512, heads=8)


def _case_b2(torch_mod, F, device, dtype):
    return _block(torch_mod, F, device, dtype, b=4, s=128, h=1024, heads=8)


def _case_b3(torch_mod, F, device, dtype):
    return _block(torch_mod, F, device, dtype, b=1, s=2048, h=1024, heads=16)


def _case_c1(torch_mod, F, device, dtype):
    block = _block(torch_mod, F, device, dtype, b=1, s=512, h=1024, heads=16)

    def forward():
        out = block()
        out = block()
        out = block()
        out = block()
        return out

    return forward


def _case_c2(torch_mod, F, device, dtype):
    block = _block(torch_mod, F, device, dtype, b=2, s=256, h=1024, heads=16)

    def forward():
        out = block()
        out = block()
        out = block()
        out = block()
        return out

    return forward


def _case_d1(torch_mod, F, device, dtype):
    block = _block(torch_mod, F, device, dtype, b=2, s=256, h=512, heads=8)

    def forward():
        out = block()
        out = block()
        out = block()
        out = block()
        out = block()
        out = block()
        out = block()
        out = block()
        return out

    return forward


def _case_d2(torch_mod, F, device, dtype):
    b, s, h = 8, 128, 256
    x = torch_mod.randn((b, s, h), device=device, dtype=dtype)
    bias = torch_mod.randn((h,), device=device, dtype=dtype)
    scale = torch_mod.randn((h,), device=device, dtype=dtype)

    def forward():
        y = x
        for _ in range(12):
            y = y + bias
            y = F.relu(y)
            y = y * scale
            y = y + y
        return y

    return forward


CASES = {
    "A1": {
        "case_id": "A1",
        "batch": 1,
        "seq": 128,
        "hidden": 1024,
        "heads": None,
        "dtype": "float16",
        "builder": _case_a1,
    },
    "A2": {
        "case_id": "A2",
        "batch": 1,
        "seq": 512,
        "hidden": 1024,
        "heads": 16,
        "dtype": "float16",
        "builder": _case_a2,
    },
    "A2s": {
        "case_id": "A2s",
        "batch": 2,
        "seq": 256,
        "hidden": 512,
        "heads": 8,
        "dtype": "float16",
        "builder": _case_a2s,
    },
    "A3": {
        "case_id": "A3",
        "batch": 8,
        "seq": 128,
        "hidden": 2048,
        "heads": None,
        "dtype": "float16",
        "builder": _case_a3,
    },
    "B1": {
        "case_id": "B1",
        "batch": 1,
        "seq": 512,
        "hidden": 2048,
        "heads": 16,
        "dtype": "float16",
        "builder": _case_b1,
    },
    "B1s": {
        "case_id": "B1s",
        "batch": 2,
        "seq": 256,
        "hidden": 512,
        "heads": 8,
        "dtype": "float16",
        "builder": _case_b1s,
    },
    "B2": {
        "case_id": "B2",
        "batch": 4,
        "seq": 128,
        "hidden": 1024,
        "heads": 8,
        "dtype": "float16",
        "builder": _case_b2,
    },
    "B3": {
        "case_id": "B3",
        "batch": 1,
        "seq": 2048,
        "hidden": 1024,
        "heads": 16,
        "dtype": "float16",
        "builder": _case_b3,
    },
    "C1": {
        "case_id": "C1",
        "batch": 1,
        "seq": 512,
        "hidden": 1024,
        "heads": 16,
        "dtype": "float16",
        "builder": _case_c1,
    },
    "C2": {
        "case_id": "C2",
        "batch": 2,
        "seq": 256,
        "hidden": 1024,
        "heads": 16,
        "dtype": "float16",
        "builder": _case_c2,
    },
    "D1": {
        "case_id": "D1",
        "batch": 2,
        "seq": 256,
        "hidden": 512,
        "heads": 8,
        "dtype": "float16",
        "builder": _case_d1,
    },
    "D2": {
        "case_id": "D2",
        "batch": 8,
        "seq": 128,
        "hidden": 256,
        "heads": None,
        "dtype": "float16",
        "builder": _case_d2,
    },
}
