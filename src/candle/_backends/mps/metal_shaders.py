"""Metal Shading Language (MSL) kernel source strings for GPU compute.

All kernels are compiled once at first use via MetalKernelDispatcher.
Naming convention: <op>_<suffix> for each dtype variant (f32, f16, i32, i64, u8).
"""

# ---------------------------------------------------------------------------
# Dtype mapping: (MSL type, kernel suffix)
# ---------------------------------------------------------------------------
DTYPE_MAP = [
    ("float", "f32"),
    ("half",  "f16"),
    ("int",   "i32"),
    ("long",  "i64"),
    ("uchar", "u8"),     # bool
]

_FLOAT_TYPES = ("float", "half")
_ALL_TYPES = tuple(t for t, _ in DTYPE_MAP)
_SUFFIX = dict(DTYPE_MAP)

# Ops that require math functions — only valid for float/half
_FLOAT_ONLY_OPS = frozenset({
    "sqrt", "rsqrt", "exp", "log", "log2", "sin", "cos", "tanh",
    "sigmoid", "gelu", "silu", "floor", "ceil", "round",
})

# ---------------------------------------------------------------------------
# Templates: element-wise kernels
# ---------------------------------------------------------------------------

_UNARY_TEMPLATE = """
kernel void {name}_{suffix}(device const {type}* a [[buffer(0)]],
                             device {type}* out      [[buffer(1)]],
                             constant uint& N        [[buffer(2)]],
                             uint id [[thread_position_in_grid]]) {{
    if (id < N) {{
        {type} x = a[id];
        out[id] = {expr};
    }}
}}
"""

_BINARY_TEMPLATE = """
kernel void {name}_{suffix}(device const {type}* a [[buffer(0)]],
                             device const {type}* b [[buffer(1)]],
                             device {type}* out      [[buffer(2)]],
                             constant uint& N        [[buffer(3)]],
                             uint id [[thread_position_in_grid]]) {{
    if (id < N) out[id] = {expr};
}}
"""

_BINARY_SCALAR_TEMPLATE = """
kernel void {name}_scalar_{suffix}(device const {type}* a  [[buffer(0)]],
                                    constant {type}& scalar [[buffer(1)]],
                                    device {type}* out      [[buffer(2)]],
                                    constant uint& N        [[buffer(3)]],
                                    uint id [[thread_position_in_grid]]) {{
    if (id < N) out[id] = {expr};
}}
"""

# ---------------------------------------------------------------------------
# Templates: strided element-wise kernels
# ---------------------------------------------------------------------------

_STRIDED_INDEX_HELPER = """
// Strided indexing helper: convert linear index to byte offset
inline uint index_to_offset(uint linear_idx,
                            constant uint* shape,
                            constant int*  strides,
                            uint ndim) {
    uint offset = 0;
    for (uint d = ndim; d > 0; d--) {
        offset += (linear_idx % shape[d-1]) * uint(strides[d-1]);
        linear_idx /= shape[d-1];
    }
    return offset;
}
"""

_UNARY_STRIDED_TEMPLATE = """
kernel void {name}_strided_{suffix}(
    device const {type}* a   [[buffer(0)]],
    device {type}* out       [[buffer(1)]],
    constant uint& N         [[buffer(2)]],
    constant uint* shape     [[buffer(3)]],
    constant int*  strides_a [[buffer(4)]],
    constant uint& ndim      [[buffer(5)]],
    uint id [[thread_position_in_grid]]) {{
    if (id < N) {{
        uint off_a = index_to_offset(id, shape, strides_a, ndim);
        {type} x = a[off_a];
        out[id] = {expr};
    }}
}}
"""

_BINARY_STRIDED_TEMPLATE = """
kernel void {name}_strided_{suffix}(
    device const {type}* a   [[buffer(0)]],
    device const {type}* b   [[buffer(1)]],
    device {type}* out       [[buffer(2)]],
    constant uint& N         [[buffer(3)]],
    constant uint* shape     [[buffer(4)]],
    constant int*  strides_a [[buffer(5)]],
    constant int*  strides_b [[buffer(6)]],
    constant uint& ndim      [[buffer(7)]],
    uint id [[thread_position_in_grid]]) {{
    if (id < N) {{
        uint off_a = index_to_offset(id, shape, strides_a, ndim);
        uint off_b = index_to_offset(id, shape, strides_b, ndim);
        out[id] = {expr_strided};
    }}
}}
"""

_BINARY_SCALAR_STRIDED_TEMPLATE = """
kernel void {name}_scalar_strided_{suffix}(
    device const {type}* a   [[buffer(0)]],
    constant {type}& scalar  [[buffer(1)]],
    device {type}* out       [[buffer(2)]],
    constant uint& N         [[buffer(3)]],
    constant uint* shape     [[buffer(4)]],
    constant int*  strides_a [[buffer(5)]],
    constant uint& ndim      [[buffer(6)]],
    uint id [[thread_position_in_grid]]) {{
    if (id < N) {{
        uint off_a = index_to_offset(id, shape, strides_a, ndim);
        out[id] = {expr_strided};
    }}
}}
"""

# ---------------------------------------------------------------------------
# Templates: unary predicate kernels (float → bool/uchar)
# ---------------------------------------------------------------------------

_UNARY_PREDICATE_TEMPLATE = """
kernel void {name}_{suffix}(device const {type}* a [[buffer(0)]],
                              device uchar* out      [[buffer(1)]],
                              constant uint& N       [[buffer(2)]],
                              uint id [[thread_position_in_grid]]) {{
    if (id < N) out[id] = {expr} ? 1 : 0;
}}
"""

_UNARY_PREDICATE_STRIDED_TEMPLATE = """
kernel void {name}_strided_{suffix}(
    device const {type}* a   [[buffer(0)]],
    device uchar* out        [[buffer(1)]],
    constant uint& N         [[buffer(2)]],
    constant uint* shape     [[buffer(3)]],
    constant int*  strides_a [[buffer(4)]],
    constant uint& ndim      [[buffer(5)]],
    uint id [[thread_position_in_grid]]) {{
    if (id < N) {{
        uint off_a = index_to_offset(id, shape, strides_a, ndim);
        out[id] = {expr_strided} ? 1 : 0;
    }}
}}
"""

# ---------------------------------------------------------------------------
# Templates: clamp with 2 scalars (min + max)
# ---------------------------------------------------------------------------

_CLAMP_TEMPLATE = """
kernel void clamp_{suffix}(device const {type}* a    [[buffer(0)]],
                             constant {type}& min_val [[buffer(1)]],
                             constant {type}& max_val [[buffer(2)]],
                             device {type}* out       [[buffer(3)]],
                             constant uint& N         [[buffer(4)]],
                             uint id [[thread_position_in_grid]]) {{
    if (id < N) out[id] = clamp(a[id], min_val, max_val);
}}
"""

_CLAMP_STRIDED_TEMPLATE = """
kernel void clamp_strided_{suffix}(
    device const {type}* a    [[buffer(0)]],
    constant {type}& min_val  [[buffer(1)]],
    constant {type}& max_val  [[buffer(2)]],
    device {type}* out        [[buffer(3)]],
    constant uint& N          [[buffer(4)]],
    constant uint* shape      [[buffer(5)]],
    constant int*  strides_a  [[buffer(6)]],
    constant uint& ndim       [[buffer(7)]],
    uint id [[thread_position_in_grid]]) {{
    if (id < N) {{
        uint off_a = index_to_offset(id, shape, strides_a, ndim);
        out[id] = clamp(a[off_a], min_val, max_val);
    }}
}}
"""

# ---------------------------------------------------------------------------
# Templates: comparison kernels (typed input, uchar output)
# ---------------------------------------------------------------------------

_COMPARISON_TEMPLATE = """
kernel void {name}_{suffix}(device const {type}* a [[buffer(0)]],
                              device const {type}* b [[buffer(1)]],
                              device uchar* out      [[buffer(2)]],
                              constant uint& N       [[buffer(3)]],
                              uint id [[thread_position_in_grid]]) {{
    if (id < N) out[id] = (a[id] {op} b[id]) ? 1 : 0;
}}
"""

_COMPARISON_SCALAR_TEMPLATE = """
kernel void {name}_scalar_{suffix}(device const {type}* a  [[buffer(0)]],
                                    constant {type}& scalar [[buffer(1)]],
                                    device uchar* out       [[buffer(2)]],
                                    constant uint& N        [[buffer(3)]],
                                    uint id [[thread_position_in_grid]]) {{
    if (id < N) out[id] = (a[id] {op} scalar) ? 1 : 0;
}}
"""

# ---------------------------------------------------------------------------
# Templates: axis-reduce kernels
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Templates: shape & index kernels (Phase 3)
# ---------------------------------------------------------------------------

_WHERE_TEMPLATE = """
kernel void where_{suffix}(device const uchar* cond [[buffer(0)]],
                            device const {type}* x   [[buffer(1)]],
                            device const {type}* y   [[buffer(2)]],
                            device {type}* out        [[buffer(3)]],
                            constant uint& N          [[buffer(4)]],
                            uint id [[thread_position_in_grid]]) {{
    if (id < N) out[id] = cond[id] ? x[id] : y[id];
}}
"""

_WHERE_SCALAR_Y_TEMPLATE = """
kernel void where_scalar_y_{suffix}(device const uchar* cond [[buffer(0)]],
                                     device const {type}* x   [[buffer(1)]],
                                     constant {type}& y_val    [[buffer(2)]],
                                     device {type}* out        [[buffer(3)]],
                                     constant uint& N          [[buffer(4)]],
                                     uint id [[thread_position_in_grid]]) {{
    if (id < N) out[id] = cond[id] ? x[id] : y_val;
}}
"""

_WHERE_SCALAR_X_TEMPLATE = """
kernel void where_scalar_x_{suffix}(device const uchar* cond [[buffer(0)]],
                                     constant {type}& x_val    [[buffer(1)]],
                                     device const {type}* y    [[buffer(2)]],
                                     device {type}* out        [[buffer(3)]],
                                     constant uint& N          [[buffer(4)]],
                                     uint id [[thread_position_in_grid]]) {{
    if (id < N) out[id] = cond[id] ? x_val : y[id];
}}
"""

_MASKED_FILL_TEMPLATE = """
kernel void masked_fill_{suffix}(device const {type}* a  [[buffer(0)]],
                                  device const uchar* mask [[buffer(1)]],
                                  constant {type}& value   [[buffer(2)]],
                                  device {type}* out       [[buffer(3)]],
                                  constant uint& N         [[buffer(4)]],
                                  uint id [[thread_position_in_grid]]) {{
    if (id < N) out[id] = mask[id] ? value : a[id];
}}
"""

_TRIL_TEMPLATE = """
kernel void tril_{suffix}(device const {type}* a [[buffer(0)]],
                           device {type}* out      [[buffer(1)]],
                           constant uint& rows     [[buffer(2)]],
                           constant uint& cols     [[buffer(3)]],
                           constant int& diagonal  [[buffer(4)]],
                           constant uint& N        [[buffer(5)]],
                           uint id [[thread_position_in_grid]]) {{
    if (id >= N) return;
    uint plane = rows * cols;
    uint pos = id % plane;
    uint row = pos / cols;
    uint col = pos % cols;
    out[id] = ((int)col <= (int)row + diagonal) ? a[id] : ({type})0;
}}
"""

_TRIU_TEMPLATE = """
kernel void triu_{suffix}(device const {type}* a [[buffer(0)]],
                           device {type}* out      [[buffer(1)]],
                           constant uint& rows     [[buffer(2)]],
                           constant uint& cols     [[buffer(3)]],
                           constant int& diagonal  [[buffer(4)]],
                           constant uint& N        [[buffer(5)]],
                           uint id [[thread_position_in_grid]]) {{
    if (id >= N) return;
    uint plane = rows * cols;
    uint pos = id % plane;
    uint row = pos / cols;
    uint col = pos % cols;
    out[id] = ((int)col >= (int)row + diagonal) ? a[id] : ({type})0;
}}
"""

_INDEX_SELECT_TEMPLATE = """
kernel void index_select_{suffix}(device const {type}* input   [[buffer(0)]],
                                   device const int* indices    [[buffer(1)]],
                                   device {type}* output        [[buffer(2)]],
                                   constant uint& outer_size    [[buffer(3)]],
                                   constant uint& idx_size      [[buffer(4)]],
                                   constant uint& inner_size    [[buffer(5)]],
                                   constant uint& input_dim_size [[buffer(6)]],
                                   uint gid [[thread_position_in_grid]]) {{
    if (gid >= outer_size * idx_size * inner_size) return;
    uint inner_idx = gid % inner_size;
    uint k = (gid / inner_size) % idx_size;
    uint outer = gid / (idx_size * inner_size);
    output[gid] = input[(outer * input_dim_size + (uint)indices[k]) * inner_size + inner_idx];
}}
"""

_GATHER_TEMPLATE = """
kernel void gather_{suffix}(device const {type}* input   [[buffer(0)]],
                             device const int* indices    [[buffer(1)]],
                             device {type}* output        [[buffer(2)]],
                             constant uint& outer_size    [[buffer(3)]],
                             constant uint& idx_size      [[buffer(4)]],
                             constant uint& inner_size    [[buffer(5)]],
                             constant uint& input_dim_size [[buffer(6)]],
                             uint gid [[thread_position_in_grid]]) {{
    if (gid >= outer_size * idx_size * inner_size) return;
    uint inner_idx = gid % inner_size;
    uint k = (gid / inner_size) % idx_size;
    uint outer = gid / (idx_size * inner_size);
    output[gid] = input[(outer * input_dim_size + (uint)indices[gid]) * inner_size + inner_idx];
}}
"""

_CAT_COPY_TEMPLATE = """
kernel void cat_copy_{suffix}(device const {type}* src [[buffer(0)]],
                               device {type}* dst       [[buffer(1)]],
                               constant uint& outer_size [[buffer(2)]],
                               constant uint& src_dim    [[buffer(3)]],
                               constant uint& inner_size [[buffer(4)]],
                               constant uint& dst_dim    [[buffer(5)]],
                               constant uint& offset     [[buffer(6)]],
                               uint gid [[thread_position_in_grid]]) {{
    if (gid >= outer_size * src_dim * inner_size) return;
    uint inner_idx = gid % inner_size;
    uint dim_idx = (gid / inner_size) % src_dim;
    uint outer = gid / (src_dim * inner_size);
    dst[(outer * dst_dim + offset + dim_idx) * inner_size + inner_idx] = src[gid];
}}
"""

# ---------------------------------------------------------------------------
# Templates: axis-reduce kernels
# ---------------------------------------------------------------------------

_REDUCE_DIM_TEMPLATE = """
kernel void reduce_{name}_dim_{suffix}(
    device const {type}* input  [[buffer(0)]],
    device {out_type}* output   [[buffer(1)]],
    constant uint& outer_size   [[buffer(2)]],
    constant uint& reduce_size  [[buffer(3)]],
    constant uint& inner_size   [[buffer(4)]],
    uint gid [[thread_position_in_grid]]) {{
    if (gid >= outer_size * inner_size) return;
    uint outer = gid / inner_size;
    uint inner = gid % inner_size;
    {init}
    for (uint r = 0; r < reduce_size; r++) {{
        {type} val = input[(outer * reduce_size + r) * inner_size + inner];
        {body}
    }}
    output[gid] = {finalize};
}}
"""

# ---------------------------------------------------------------------------
# Generator functions
# ---------------------------------------------------------------------------

def _gen_unary(name, expr, types=None, float_only=False):
    """Generate unary kernel source for given types (contiguous + strided)."""
    if types is None:
        types = _FLOAT_TYPES if float_only else _ALL_TYPES
    parts = []
    for t in types:
        suffix = _SUFFIX[t]
        parts.append(_UNARY_TEMPLATE.format(
            name=name, suffix=suffix, type=t, expr=expr))
        parts.append(_UNARY_STRIDED_TEMPLATE.format(
            name=name, suffix=suffix, type=t, expr=expr))
    return "".join(parts)


def _gen_binary(name, expr, types=None, float_only=False):
    """Generate binary kernel source for given types (contiguous + strided)."""
    if types is None:
        types = _FLOAT_TYPES if float_only else _ALL_TYPES
    parts = []
    for t in types:
        suffix = _SUFFIX[t]
        # Contiguous variants
        parts.append(_BINARY_TEMPLATE.format(
            name=name, suffix=suffix, type=t, expr=expr))
        scalar_expr = expr.replace("b[id]", "scalar")
        parts.append(_BINARY_SCALAR_TEMPLATE.format(
            name=name, suffix=suffix, type=t, expr=scalar_expr))
        # Strided variants: replace a[id]/b[id] with a[off_a]/b[off_b]
        strided_expr = expr.replace("a[id]", "a[off_a]").replace("b[id]", "b[off_b]")
        parts.append(_BINARY_STRIDED_TEMPLATE.format(
            name=name, suffix=suffix, type=t, expr_strided=strided_expr))
        scalar_strided_expr = expr.replace("a[id]", "a[off_a]").replace("b[id]", "scalar")
        parts.append(_BINARY_SCALAR_STRIDED_TEMPLATE.format(
            name=name, suffix=suffix, type=t, expr_strided=scalar_strided_expr))
    return "".join(parts)


def _gen_comparison(name, op, types=None):
    """Generate comparison kernels (typed input → uchar output)."""
    if types is None:
        types = ("float", "half", "int", "long")
    parts = []
    for t in types:
        suffix = _SUFFIX[t]
        parts.append(_COMPARISON_TEMPLATE.format(
            name=name, suffix=suffix, type=t, op=op))
        parts.append(_COMPARISON_SCALAR_TEMPLATE.format(
            name=name, suffix=suffix, type=t, op=op))
    return "".join(parts)


def _gen_unary_predicate(name, expr, types=None):
    """Generate unary predicate kernels (typed input → uchar output)."""
    if types is None:
        types = _FLOAT_TYPES
    parts = []
    for t in types:
        suffix = _SUFFIX[t]
        parts.append(_UNARY_PREDICATE_TEMPLATE.format(
            name=name, suffix=suffix, type=t, expr=expr))
        strided_expr = expr.replace("a[id]", "a[off_a]")
        parts.append(_UNARY_PREDICATE_STRIDED_TEMPLATE.format(
            name=name, suffix=suffix, type=t, expr_strided=strided_expr))
    return "".join(parts)


def _gen_clamp(types=None):
    """Generate clamp kernels (2-scalar: min + max) for given types."""
    if types is None:
        types = ("float", "half", "int", "long")
    parts = []
    for t in types:
        suffix = _SUFFIX[t]
        parts.append(_CLAMP_TEMPLATE.format(suffix=suffix, type=t))
        parts.append(_CLAMP_STRIDED_TEMPLATE.format(suffix=suffix, type=t))
    return "".join(parts)


def _gen_reduce_dim(name, init, body, finalize, out_type=None, types=None):
    """Generate axis-reduce kernels for each dtype."""
    if types is None:
        types = ("float", "half", "int", "long")
    parts = []
    for t in types:
        suffix = _SUFFIX[t]
        ot = out_type if out_type else t
        parts.append(_REDUCE_DIM_TEMPLATE.format(
            name=name, suffix=suffix, type=t, out_type=ot,
            init=init.replace("{type}", t).replace("{out_type}", ot),
            body=body.replace("{type}", t),
            finalize=finalize.replace("{type}", t)))
    return "".join(parts)


def _gen_where(types=None):
    """Generate where kernels (3 variants per type)."""
    if types is None:
        types = ("float", "half", "int", "long")
    parts = []
    for t in types:
        suffix = _SUFFIX[t]
        parts.append(_WHERE_TEMPLATE.format(suffix=suffix, type=t))
        parts.append(_WHERE_SCALAR_Y_TEMPLATE.format(suffix=suffix, type=t))
        parts.append(_WHERE_SCALAR_X_TEMPLATE.format(suffix=suffix, type=t))
    return "".join(parts)


def _gen_masked_fill(types=None):
    """Generate masked_fill kernels (1 per type)."""
    if types is None:
        types = ("float", "half", "int", "long")
    parts = []
    for t in types:
        suffix = _SUFFIX[t]
        parts.append(_MASKED_FILL_TEMPLATE.format(suffix=suffix, type=t))
    return "".join(parts)


def _gen_tril_triu(types=None):
    """Generate tril and triu kernels (2 per type)."""
    if types is None:
        types = ("float", "half", "int", "long")
    parts = []
    for t in types:
        suffix = _SUFFIX[t]
        parts.append(_TRIL_TEMPLATE.format(suffix=suffix, type=t))
        parts.append(_TRIU_TEMPLATE.format(suffix=suffix, type=t))
    return "".join(parts)


def _gen_index_select(types=None):
    """Generate index_select kernels (1 per type)."""
    if types is None:
        types = ("float", "half", "int", "long")
    parts = []
    for t in types:
        suffix = _SUFFIX[t]
        parts.append(_INDEX_SELECT_TEMPLATE.format(suffix=suffix, type=t))
    return "".join(parts)


def _gen_gather(types=None):
    """Generate gather kernels (1 per type)."""
    if types is None:
        types = ("float", "half", "int", "long")
    parts = []
    for t in types:
        suffix = _SUFFIX[t]
        parts.append(_GATHER_TEMPLATE.format(suffix=suffix, type=t))
    return "".join(parts)


def _gen_cat_copy(types=None):
    """Generate cat_copy kernels (1 per type, includes bool/uchar)."""
    if types is None:
        types = ("float", "half", "int", "long", "uchar")
    parts = []
    for t in types:
        suffix = _SUFFIX[t]
        parts.append(_CAT_COPY_TEMPLATE.format(suffix=suffix, type=t))
    return "".join(parts)


# ---------------------------------------------------------------------------
# Build the full MSL source
# ---------------------------------------------------------------------------

_HEADER = """
#include <metal_stdlib>
using namespace metal;
"""

# --- Binary element-wise ---
_BINARY_OPS = (
    ("add", "a[id] + b[id]"),
    ("sub", "a[id] - b[id]"),
    ("mul", "a[id] * b[id]"),
    ("div", "a[id] / b[id]"),
    ("maximum", "max(a[id], b[id])"),
    ("minimum", "min(a[id], b[id])"),
)

_BINARY_FLOAT_OPS = (
    ("pow", "pow(a[id], b[id])"),
    ("fmod", "fmod(a[id], b[id])"),
    ("remainder", "a[id] - b[id] * floor(a[id] / b[id])"),
)

# --- Unary element-wise (numeric types: float, half, int, long) ---
_UNARY_NUMERIC_OPS = (
    ("neg", "-x"),
    ("abs", "abs(x)"),
    ("relu", "max(x, ({type})0)"),
)

_UNARY_FLOAT_OPS = (
    ("sign", "sign(x)"),
    ("sqrt", "sqrt(x)"),
    ("rsqrt", "rsqrt(x)"),
    ("exp", "exp(x)"),
    ("log", "log(x)"),
    ("log2", "log2(x)"),
    ("sin", "sin(x)"),
    ("cos", "cos(x)"),
    ("tanh", "tanh(x)"),
    ("sigmoid", "1.0 / (1.0 + exp(-x))"),
    ("floor", "floor(x)"),
    ("ceil", "ceil(x)"),
    ("round", "rint(x)"),
    ("gelu", "0.5f * x * (1.0f + tanh(0.7978845608f * (x + 0.044715f * x * x * x)))"),
    ("silu", "x / (1.0f + exp(-x))"),
)

# --- Comparison ops ---
_COMPARISON_OPS = (
    ("eq", "=="), ("ne", "!="), ("lt", "<"),
    ("le", "<="), ("gt", ">"),  ("ge", ">="),
)

# --- Reduction templates (full-tensor, two-pass parallel) ---

_REDUCTION_PARTIAL_TEMPLATE = """
kernel void {name}_partial_{suffix}(device const {type}* input [[buffer(0)]],
                            device {type}* partials     [[buffer(1)]],
                            constant uint& N           [[buffer(2)]],
                            uint gid [[thread_position_in_grid]],
                            uint lid [[thread_position_in_threadgroup]],
                            uint group_id [[threadgroup_position_in_grid]],
                            uint group_size [[threads_per_threadgroup]]) {{
    threadgroup {type} shared[256];
    {type} val = (gid < N) ? input[gid] : {identity};
    shared[lid] = val;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = group_size / 2; s > 0; s >>= 1) {{
        if (lid < s) shared[lid] = {combine};
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}
    if (lid == 0) partials[group_id] = shared[0];
}}
"""

_REDUCTION_FINAL_TEMPLATE = """
kernel void {name}_final_{suffix}(device const {type}* partials [[buffer(0)]],
                          device {type}* output          [[buffer(1)]],
                          constant uint& N              [[buffer(2)]],
                          uint lid [[thread_position_in_threadgroup]],
                          uint group_size [[threads_per_threadgroup]]) {{
    threadgroup {type} shared[256];
    {type} val = (lid < N) ? partials[lid] : {identity};
    shared[lid] = val;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = group_size / 2; s > 0; s >>= 1) {{
        if (lid < s) shared[lid] = {combine};
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}
    if (lid == 0) output[0] = shared[0];
}}
"""


def _gen_reduction(name, identity, combine, types=None):
    """Generate full-tensor two-pass reduction kernels per dtype."""
    if types is None:
        types = ("float", "half", "int", "long")
    parts = []
    for t in types:
        suffix = _SUFFIX[t]
        ident = identity.replace("{type}", t)
        comb = combine.replace("{type}", t)
        parts.append(_REDUCTION_PARTIAL_TEMPLATE.format(
            name=name, suffix=suffix, type=t, identity=ident, combine=comb))
        parts.append(_REDUCTION_FINAL_TEMPLATE.format(
            name=name, suffix=suffix, type=t, identity=ident, combine=comb))
    return "".join(parts)


# Argmax/argmin reduction templates (value + index pairs)
_ARG_REDUCTION_PARTIAL_TEMPLATE = """
kernel void {name}_partial_{suffix}(device const {type}* input [[buffer(0)]],
                               device {type}* partial_vals [[buffer(1)]],
                               device uint* partial_idxs   [[buffer(2)]],
                               constant uint& N            [[buffer(3)]],
                               uint gid [[thread_position_in_grid]],
                               uint lid [[thread_position_in_threadgroup]],
                               uint group_id [[threadgroup_position_in_grid]],
                               uint group_size [[threads_per_threadgroup]]) {{
    threadgroup {type} s_val[256];
    threadgroup uint s_idx[256];
    {type} val = (gid < N) ? input[gid] : {identity};
    uint idx = gid;
    s_val[lid] = val;
    s_idx[lid] = idx;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = group_size / 2; s > 0; s >>= 1) {{
        if (lid < s && s_val[lid + s] {cmp} s_val[lid]) {{
            s_val[lid] = s_val[lid + s];
            s_idx[lid] = s_idx[lid + s];
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}
    if (lid == 0) {{
        partial_vals[group_id] = s_val[0];
        partial_idxs[group_id] = s_idx[0];
    }}
}}
"""

_ARG_REDUCTION_FINAL_TEMPLATE = """
kernel void {name}_final_{suffix}(device const {type}* partial_vals [[buffer(0)]],
                             device const uint* partial_idxs   [[buffer(1)]],
                             device uint* output               [[buffer(2)]],
                             constant uint& N                  [[buffer(3)]],
                             uint lid [[thread_position_in_threadgroup]],
                             uint group_size [[threads_per_threadgroup]]) {{
    threadgroup {type} s_val[256];
    threadgroup uint s_idx[256];
    {type} val = (lid < N) ? partial_vals[lid] : {identity};
    uint idx = (lid < N) ? partial_idxs[lid] : 0;
    s_val[lid] = val;
    s_idx[lid] = idx;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = group_size / 2; s > 0; s >>= 1) {{
        if (lid < s && s_val[lid + s] {cmp} s_val[lid]) {{
            s_val[lid] = s_val[lid + s];
            s_idx[lid] = s_idx[lid + s];
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}
    if (lid == 0) output[0] = s_idx[0];
}}
"""


def _gen_arg_reduction(name, identity, cmp, types=None):
    """Generate full-tensor argmax/argmin two-pass reduction kernels."""
    if types is None:
        types = ("float", "half", "int", "long")
    parts = []
    for t in types:
        suffix = _SUFFIX[t]
        ident = identity.replace("{type}", t)
        parts.append(_ARG_REDUCTION_PARTIAL_TEMPLATE.format(
            name=name, suffix=suffix, type=t, identity=ident, cmp=cmp))
        parts.append(_ARG_REDUCTION_FINAL_TEMPLATE.format(
            name=name, suffix=suffix, type=t, identity=ident, cmp=cmp))
    return "".join(parts)


# --- In-place binary kernels ---
_INPLACE_BINARY_TEMPLATE = """
kernel void {name}_inplace_{suffix}(device {type}* a       [[buffer(0)]],
                                     device const {type}* b [[buffer(1)]],
                                     constant uint& N       [[buffer(2)]],
                                     uint id [[thread_position_in_grid]]) {{
    if (id < N) a[id] {op}= b[id];
}}

kernel void {name}_inplace_scalar_{suffix}(device {type}* a      [[buffer(0)]],
                                            constant {type}& scalar [[buffer(1)]],
                                            constant uint& N        [[buffer(2)]],
                                            uint id [[thread_position_in_grid]]) {{
    if (id < N) a[id] {op}= scalar;
}}
"""

_INPLACE_UNARY_SOURCE = """
kernel void relu_inplace_f32(device float* a [[buffer(0)]],
                              constant uint& N [[buffer(1)]],
                              uint id [[thread_position_in_grid]]) {
    if (id < N && a[id] < 0.0f) a[id] = 0.0f;
}

kernel void relu_inplace_f16(device half* a [[buffer(0)]],
                              constant uint& N [[buffer(1)]],
                              uint id [[thread_position_in_grid]]) {
    if (id < N && a[id] < (half)0.0f) a[id] = (half)0.0f;
}
"""


def _build_msl_source():
    """Assemble the complete MSL source string."""
    parts = [_HEADER, _STRIDED_INDEX_HELPER]

    # Binary ops (all dtypes)
    for name, expr in _BINARY_OPS:
        parts.append(_gen_binary(name, expr))

    # Binary ops (float-only)
    for name, expr in _BINARY_FLOAT_OPS:
        parts.append(_gen_binary(name, expr, float_only=True))

    # Unary ops (numeric types: float, half, int, long)
    _NUMERIC_TYPES = ("float", "half", "int", "long")
    for name, expr in _UNARY_NUMERIC_OPS:
        for t in _NUMERIC_TYPES:
            suffix = _SUFFIX[t]
            e = expr.replace("{type}", t)
            parts.append(_UNARY_TEMPLATE.format(name=name, suffix=suffix, type=t, expr=e))
            parts.append(_UNARY_STRIDED_TEMPLATE.format(name=name, suffix=suffix, type=t, expr=e))

    # Unary ops (float-only)
    for name, expr in _UNARY_FLOAT_OPS:
        for t in _FLOAT_TYPES:
            suffix = _SUFFIX[t]
            e = expr.replace("{type}", t)
            parts.append(_UNARY_TEMPLATE.format(name=name, suffix=suffix, type=t, expr=e))
            parts.append(_UNARY_STRIDED_TEMPLATE.format(name=name, suffix=suffix, type=t, expr=e))

    # Leaky ReLU via binary-scalar template (float/half)
    for t in _FLOAT_TYPES:
        suffix = _SUFFIX[t]
        expr = f"(a[id] > ({t})0) ? a[id] : scalar * a[id]"
        parts.append(_BINARY_SCALAR_TEMPLATE.format(
            name="leaky_relu", suffix=suffix, type=t, expr=expr))
        strided_expr = expr.replace("a[id]", "a[off_a]")
        parts.append(_BINARY_SCALAR_STRIDED_TEMPLATE.format(
            name="leaky_relu", suffix=suffix, type=t, expr_strided=strided_expr))

    # clamp_min / clamp_max via binary-scalar template (float, half, int, long)
    _CLAMP_SCALAR_TYPES = ("float", "half", "int", "long")
    for t in _CLAMP_SCALAR_TYPES:
        suffix = _SUFFIX[t]
        # clamp_min: max(a, scalar)
        parts.append(_BINARY_SCALAR_TEMPLATE.format(
            name="clamp_min", suffix=suffix, type=t, expr="max(a[id], scalar)"))
        parts.append(_BINARY_SCALAR_STRIDED_TEMPLATE.format(
            name="clamp_min", suffix=suffix, type=t, expr_strided="max(a[off_a], scalar)"))
        # clamp_max: min(a, scalar)
        parts.append(_BINARY_SCALAR_TEMPLATE.format(
            name="clamp_max", suffix=suffix, type=t, expr="min(a[id], scalar)"))
        parts.append(_BINARY_SCALAR_STRIDED_TEMPLATE.format(
            name="clamp_max", suffix=suffix, type=t, expr_strided="min(a[off_a], scalar)"))

    # Clamp with 2 scalars (min + max)
    parts.append(_gen_clamp())

    # Unary predicate ops (float → bool): isinf, isnan, isfinite
    parts.append(_gen_unary_predicate("isinf", "isinf(a[id])"))
    parts.append(_gen_unary_predicate("isnan", "isnan(a[id])"))
    parts.append(_gen_unary_predicate("isfinite", "isfinite(a[id])"))

    # Full-tensor reductions (multi-dtype)
    # sum: identity=0, combine=add
    parts.append(_gen_reduction(
        "sum", "({type})0",
        "shared[lid] + shared[lid + s]"))
    # max: identity=-INFINITY for float, type_min for int
    parts.append(_gen_reduction(
        "max", "-INFINITY",
        "max(shared[lid], shared[lid + s])",
        types=("float", "half")))
    parts.append(_gen_reduction(
        "max", "INT_MIN",
        "max(shared[lid], shared[lid + s])",
        types=("int",)))
    parts.append(_gen_reduction(
        "max", "LONG_MIN",
        "max(shared[lid], shared[lid + s])",
        types=("long",)))
    # min
    parts.append(_gen_reduction(
        "min", "INFINITY",
        "min(shared[lid], shared[lid + s])",
        types=("float", "half")))
    parts.append(_gen_reduction(
        "min", "INT_MAX",
        "min(shared[lid], shared[lid + s])",
        types=("int",)))
    parts.append(_gen_reduction(
        "min", "LONG_MAX",
        "min(shared[lid], shared[lid + s])",
        types=("long",)))

    # argmax/argmin (multi-dtype)
    parts.append(_gen_arg_reduction(
        "argmax", "-INFINITY", ">", types=("float", "half")))
    parts.append(_gen_arg_reduction(
        "argmax", "INT_MIN", ">", types=("int",)))
    parts.append(_gen_arg_reduction(
        "argmax", "LONG_MIN", ">", types=("long",)))
    parts.append(_gen_arg_reduction(
        "argmin", "INFINITY", "<", types=("float", "half")))
    parts.append(_gen_arg_reduction(
        "argmin", "INT_MAX", "<", types=("int",)))
    parts.append(_gen_arg_reduction(
        "argmin", "LONG_MAX", "<", types=("long",)))

    # Full-tensor prod reduction
    parts.append(_gen_reduction(
        "prod", "({type})1",
        "shared[lid] * shared[lid + s]"))

    # Full-tensor any/all reductions (uchar only — input is pre-cast to bool)
    parts.append(_gen_reduction(
        "any", "0",
        "(shared[lid] || shared[lid + s]) ? (uchar)1 : (uchar)0",
        types=("uchar",)))
    parts.append(_gen_reduction(
        "all", "1",
        "(shared[lid] && shared[lid + s]) ? (uchar)1 : (uchar)0",
        types=("uchar",)))

    # Axis-reduce kernels (dim reduction)
    parts.append(_gen_reduce_dim(
        "sum",
        init="{type} acc = ({type})0;",
        body="acc += val;",
        finalize="acc"))
    parts.append(_gen_reduce_dim(
        "mean",
        init="float acc = 0.0f;",
        body="acc += (float)val;",
        finalize="({type})(acc / (float)reduce_size)",
        types=("float", "half")))
    parts.append(_gen_reduce_dim(
        "prod",
        init="{type} acc = ({type})1;",
        body="acc *= val;",
        finalize="acc"))
    parts.append(_gen_reduce_dim(
        "max",
        init="{type} acc = input[(outer * reduce_size + 0) * inner_size + inner];",
        body="if (val > acc) acc = val;",
        finalize="acc"))
    parts.append(_gen_reduce_dim(
        "min",
        init="{type} acc = input[(outer * reduce_size + 0) * inner_size + inner];",
        body="if (val < acc) acc = val;",
        finalize="acc"))
    parts.append(_gen_reduce_dim(
        "argmax",
        init="{type} best = input[(outer * reduce_size + 0) * inner_size + inner]; uint best_idx = 0;",
        body="if (val > best) { best = val; best_idx = r; }",
        finalize="best_idx",
        out_type="uint"))
    parts.append(_gen_reduce_dim(
        "argmin",
        init="{type} best = input[(outer * reduce_size + 0) * inner_size + inner]; uint best_idx = 0;",
        body="if (val < best) { best = val; best_idx = r; }",
        finalize="best_idx",
        out_type="uint"))
    # any/all axis-reduce (typed input → uchar output)
    parts.append(_gen_reduce_dim(
        "any",
        init="uchar acc = 0;",
        body="if (val != 0) acc = 1;",
        finalize="acc",
        out_type="uchar",
        types=("float", "half", "int", "long", "uchar")))
    parts.append(_gen_reduce_dim(
        "all",
        init="uchar acc = 1;",
        body="if (val == 0) acc = 0;",
        finalize="acc",
        out_type="uchar",
        types=("float", "half", "int", "long", "uchar")))

    # Shape & index kernels (Phase 3)
    parts.append(_gen_where())
    parts.append(_gen_masked_fill())
    parts.append(_gen_tril_triu())
    parts.append(_gen_index_select())
    parts.append(_gen_gather())
    parts.append(_gen_cat_copy())

    # Comparison ops
    for name, op in _COMPARISON_OPS:
        parts.append(_gen_comparison(name, op))

    # Fill kernels (multi-dtype)
    for t in _ALL_TYPES:
        suffix = _SUFFIX[t]
        parts.append(f"""
kernel void fill_{suffix}(device {t}* out    [[buffer(0)]],
                     constant {t}& val  [[buffer(1)]],
                     constant uint& N     [[buffer(2)]],
                     uint id [[thread_position_in_grid]]) {{
    if (id < N) out[id] = val;
}}
""")

    # Copy kernels (multi-dtype)
    for t in _ALL_TYPES:
        suffix = _SUFFIX[t]
        parts.append(f"""
kernel void copy_{suffix}(device const {t}* src [[buffer(0)]],
                     device {t}* dst       [[buffer(1)]],
                     constant uint& N        [[buffer(2)]],
                     uint id [[thread_position_in_grid]]) {{
    if (id < N) dst[id] = src[id];
}}
""")

    # Softmax (float only — existing)
    parts.append("""
// ---- softmax (per-row, last dim) ----
kernel void softmax_f32(device const float* input [[buffer(0)]],
                        device float* output       [[buffer(1)]],
                        constant uint& rows        [[buffer(2)]],
                        constant uint& cols        [[buffer(3)]],
                        uint2 pos [[thread_position_in_grid]]) {
    uint row = pos.y;
    uint col = pos.x;
    if (row >= rows || col >= cols) return;

    // Find max in this row
    float max_val = -INFINITY;
    for (uint c = 0; c < cols; c++) {
        max_val = max(max_val, input[row * cols + c]);
    }

    // Compute exp(x - max)
    float exp_val = exp(input[row * cols + col] - max_val);

    // Sum of exp in this row
    float sum_exp = 0.0f;
    for (uint c = 0; c < cols; c++) {
        sum_exp += exp(input[row * cols + c] - max_val);
    }

    output[row * cols + col] = exp_val / sum_exp;
}
""")

    # Softmax float16 (compute in float32 for numerical stability)
    parts.append("""
kernel void softmax_f16(device const half* input [[buffer(0)]],
                        device half* output       [[buffer(1)]],
                        constant uint& rows       [[buffer(2)]],
                        constant uint& cols       [[buffer(3)]],
                        uint2 pos [[thread_position_in_grid]]) {
    uint row = pos.y;
    uint col = pos.x;
    if (row >= rows || col >= cols) return;

    float max_val = -INFINITY;
    for (uint c = 0; c < cols; c++) {
        max_val = max(max_val, (float)input[row * cols + c]);
    }

    float exp_val = exp((float)input[row * cols + col] - max_val);

    float sum_exp = 0.0f;
    for (uint c = 0; c < cols; c++) {
        sum_exp += exp((float)input[row * cols + c] - max_val);
    }

    output[row * cols + col] = (half)(exp_val / sum_exp);
}
""")

    # In-place binary ops (float types only — existing behavior)
    for name, op in (("add", "+"), ("sub", "-"), ("mul", "*"), ("div", "/")):
        for t, suffix in (("float", "f32"), ("half", "f16")):
            parts.append(_INPLACE_BINARY_TEMPLATE.format(
                name=name, op=op, suffix=suffix, type=t))

    parts.append(_INPLACE_UNARY_SOURCE)

    return "".join(parts)


MSL_SOURCE = _build_msl_source()
