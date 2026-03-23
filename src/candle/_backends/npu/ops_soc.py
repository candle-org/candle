"""SoC-aware capability lookup for NPU ops.

Centralises all chip-specific differences:
  1. Operator fallback  — ops needing composite workaround
  2. Dtype support       — global and per-op dtype restrictions
  3. Distributed ops     — collective op availability
  4. Chip flags          — fine-grained feature switches
"""

from . import runtime as npu_runtime

# ── 1. Operator fallback ─────────────────────────────────────────────
# Ops that must use on-device composite workaround on a given chip.
#
# NOTE: torch_npu (Ascend's official PyTorch backend) does NOT implement
# these ops natively either — they fall through to CPU via PyTorch's
# `at::native::cpu_fallback` (see VariableFallbackKernel.cpp).  Candle
# instead uses on-device ACLNN small-op composites so that tensors never
# leave the NPU, avoiding the D2H/H2D round-trip penalty.

_FALLBACK_OPS = {
    "910a": frozenset({
        # 6-op allclose composite (abs/sub/mul/add/le/all_) triggers ACLNN 561000
        # after executor pool pressure; use isclose (single kernel) + all_ instead.
        "allclose",
        "isinf",            # aclnnIsInf returns 161001 (unavailable)
        "frac",             # aclnnFrac returns 561000 (unsupported)
        "gather",           # aclnnGather returns 561103 on multi-dim inputs
        "matmul",           # aclnnMatmul on 910A only supports float16; float32 inputs are cast
        "addmm",            # aclnnAddmm on 910A only supports float16; float32 inputs are cast
        "mv",               # aclnnMv on 910A only supports float16; float32 inputs are cast
        "repeat_interleave_tensor",  # aclnnRepeatInterleave* tensor-repeats poisons later ops on 910A
    }),
    "910b": frozenset({
        # torch_npu: CPU fallback; candle: on-device composite
        "std",              # aclnnVar all-reduce fails with 161002
        "nansum",           # aclnnReduceNansum returns 161002
        "instance_norm",    # aclnnInstanceNorm returns 161002
        "avg_pool2d",           # aclnnAvgPool2d returns 161002
        "adaptive_avg_pool2d",  # cross-op contamination (cubeMathType=1 corrupts state)
        "upsample_nearest1d",  # aclnnUpsampleNearest1d broken; composite always used
        "einsum",           # aclnnEinsum returns 161002
        "isinf",            # aclnnIsInf returns 161001 (unavailable)
        "im2col",           # aclnnIm2col returns 561103
    }),
    "310b": frozenset({
        # Confirmed broken/missing native kernels on 310B (locally tested):
        "allclose",     # le_tensor path in allclose segfaults after common test sequence
        "isinf",        # aclnnIsInf returns 161001 (unavailable)
        "dot",          # aclnnDot returns 561103
        "matmul",       # aclnnMatmul float32 unsupported; cast to float16
        "addmm",        # aclnnAddmm float32 unsupported; cast to float16
        "mv",           # aclnnMv float32 unsupported; cast to float16
        "remainder",    # aclnnRemainderTensorTensor returns 161002; composite uses where
        "where",        # aclnnSWhere returns 561000 on 310B
        "atan2",        # aclnnAtan2 returns 561103
        "lerp",         # aclnnLerps returns 561103
        "dropout",      # aclnnDropoutDoMask returns 561103
        "softplus",     # aclnnSoftplus returns 561103; mish composite depends on this
        "isclose",      # aclnnIsClose returns 561103 on 310B
        "flip",         # aclnnFlip returns 561000
        "argsort",      # aclnnTopk (used by argsort) returns 561103
        "sort",         # aclnnTopk returns 561103
        "topk",         # aclnnTopk returns 561103
        "diag",         # aclnnDiag returns 561103
        "gather",       # aclnnGather returns 561103
        "take_along_dim",  # aclnnGather returns 561103
        "layer_norm",   # aclnnLayerNorm returns 561103 for float32 (float16 works)
        "mish",         # aclnnMish returns 561103
        "batch_norm",   # aclnnBatchNorm returns 161002
        "avg_pool2d",           # aclnnAvgPool2d returns 161002 (same as 910B)
        "adaptive_avg_pool2d",  # aclnnAdaptiveAvgPool2d cubeMathType contamination
        "einsum",       # aclnnEinsum untested on 310B; composite works
    }),
    "310p": frozenset(),
}

# ── 2. Dtype support ─────────────────────────────────────────────────
# 2a. Dtypes globally unsupported on a chip (all ops).
_UNSUPPORTED_DTYPES_GLOBAL = {
    "910a": frozenset({"bfloat16"}),
    "910b": frozenset(),
    "310b": frozenset({"bfloat16"}),
    "310p": frozenset(),
}

# 2b. Per-op dtype restrictions (override / extend global).
_UNSUPPORTED_DTYPES_PER_OP = {
    "910a": {},
    "910b": {},
    "310b": {},
    "310p": {},
}

# ── 3. Distributed op support ────────────────────────────────────────
_SUPPORTED_DISTRIBUTED_OPS = {
    "910a": frozenset({"all_reduce", "broadcast", "all_gather"}),
    "910b": frozenset({
        "all_reduce", "broadcast", "all_gather",
        "alltoall_v", "reduce_scatter",
    }),
    "310b": frozenset(),
    "310p": frozenset(),
}

# ── 4. Chip flags ────────────────────────────────────────────────────
_CHIP_FLAGS = {
    "910a": {
        "use_smallop_arange_1d": False,
        "use_smallop_linspace": True,
        "aclgraph_supported": False,
    },
    "910b": {
        "use_smallop_arange_1d": False,
        "use_smallop_linspace": True,
        "aclgraph_supported": True,
    },
    "310b": {
        "use_smallop_arange_1d": True,
        "use_smallop_linspace": True,
        "use_safe_int64_index_compare": True,
        "aclgraph_supported": False,
    },
    "310p": {
        "use_smallop_arange_1d": False,
        "use_smallop_linspace": False,
        "aclgraph_supported": False,
    },
}

# ── Internal helpers ─────────────────────────────────────────────────

_cached_profile = None


def current_profile():
    """Return cached SoC profile string (e.g. '910b')."""
    global _cached_profile  # pylint: disable=global-statement
    if _cached_profile is None:
        _cached_profile = npu_runtime.soc_profile()
    return _cached_profile


def _resolve_profile(profile):
    return current_profile() if profile is None else str(profile).lower()


# ── Public API ───────────────────────────────────────────────────────

def use_fallback(op_name, profile=None):
    """Return True if *op_name* should use composite fallback."""
    p = _resolve_profile(profile)
    ops = _FALLBACK_OPS.get(p, frozenset())
    return op_name in ops


def fallback_ops(profile=None):
    """Return the full set of fallback ops for *profile*."""
    return _FALLBACK_OPS.get(_resolve_profile(profile), frozenset())


def check_dtype_support(op_name, dtype, profile=None):
    """Return True if *dtype* is supported for *op_name* on *profile*.

    Checks global restrictions first, then per-op restrictions.
    """
    p = _resolve_profile(profile)
    dtype_name = dtype if isinstance(dtype, str) else getattr(dtype, "name", str(dtype))
    # Global restriction
    if dtype_name in _UNSUPPORTED_DTYPES_GLOBAL.get(p, frozenset()):
        return False
    # Per-op restriction
    per_op = _UNSUPPORTED_DTYPES_PER_OP.get(p, {})
    if op_name in per_op and dtype_name in per_op[op_name]:
        return False
    return True


def is_distributed_op_supported(op_name, profile=None):
    """Return True if distributed *op_name* is available on *profile*."""
    p = _resolve_profile(profile)
    supported = _SUPPORTED_DISTRIBUTED_OPS.get(p, frozenset())
    return op_name in supported


def chip_flag(name, profile=None, default=False):
    """Query a chip-specific feature flag."""
    p = _resolve_profile(profile)
    flags = _CHIP_FLAGS.get(p)
    if flags is None:
        return bool(default)
    return bool(flags.get(name, default))


# ── Convenience aliases (backward-compatible) ────────────────────────

def capability(name, profile=None, default=False):
    """Alias for chip_flag (backward compatibility)."""
    return chip_flag(name, profile=profile, default=default)


def use_smallop_arange_1d(profile=None):
    return chip_flag("use_smallop_arange_1d", profile=profile, default=False)


def use_smallop_linspace(profile=None):
    return chip_flag("use_smallop_linspace", profile=profile, default=False)


def aclgraph_supported(profile=None):
    return chip_flag("aclgraph_supported", profile=profile, default=False)
