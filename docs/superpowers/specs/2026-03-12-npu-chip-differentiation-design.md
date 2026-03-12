# NPU Chip Differentiation Design

## Problem

The candle NPU backend treats all Ascend chips (910A, 910B, 310B, 310P) uniformly, but they differ in:

1. **Operator support** — some ACLNN kernels are unavailable or buggy on certain chips
2. **dtype support** — e.g., 310B does not support bfloat16
3. **Distributed op support** — e.g., 910A lacks alltoall_v; 310B has no distributed support

Additionally, `ops.py` is 11,361 lines and difficult to maintain.

## Solution

**Approach A: Functional domain split + centralized capability table.**

Split the monolithic `ops.py` into domain modules. Extend `ops_soc.py` to cover all three chip-difference dimensions. Chip routing stays centralized — individual op modules query `ops_soc` for decisions.

## Directory Structure

### After refactoring

```
src/candle/_backends/npu/
├── ops/                        # ops.py split into sub-modules
│   ├── __init__.py             # Unified re-export, dispatch registration
│   ├── _helpers.py             # Shared utilities (~40 functions)
│   ├── math.py                 # Arithmetic + unary math (add, exp, sin, pow, ...)
│   ├── comparison.py           # Comparison / logical / bitwise ops
│   ├── reduce.py               # Reductions + cumulative (sum, argmax, topk, cumsum, ...)
│   ├── shape.py                # Shape / view / indexing (reshape, flip, cat, gather, ...)
│   ├── activation.py           # Activation functions + softmax
│   ├── norm.py                 # Normalization (layer_norm, batch_norm, group_norm, ...)
│   ├── linalg.py               # Linear algebra (matmul, dot, qr, svd, eig, ...)
│   ├── conv.py                 # Convolution + pooling + upsampling
│   ├── random.py               # Random / initialization / dropout
│   ├── elementwise.py          # Misc element-wise (where, lerp, clamp, remainder, ...)
│   ├── special.py              # Special functions + FFT
│   └── optim.py                # Optimizer step ops (_adam_step_op, ...)
├── distributed/                # Distributed collective ops
│   ├── __init__.py
│   └── collective.py           # all_reduce, alltoall_v, broadcast, ...
├── ops_soc.py                  # Chip capability table (extended)
├── runtime.py                  # Device detection, initialization
├── aclnn.py                    # ACLNN kernel ctypes bindings
├── cann_discovery.py           # CANN path auto-discovery
├── acl_loader.py               # ACL shared library loader
├── allocator.py                # Device memory allocator
├── state.py                    # Device state management
├── streams.py                  # Stream management
├── creation.py                 # Tensor creation ops
├── backward.py                 # Backward pass implementations
├── custom_kernel.py            # Custom kernel support
└── __init__.py
```

### Estimated module sizes

| Module | Lines | Description |
|--------|-------|-------------|
| `_helpers.py` | ~400 | Shared: `_unary_op`, `_binary_op`, `_broadcast_shape`, scalar conversion, ... |
| `math.py` | ~300 | `add`, `sub`, `mul`, `div`, `exp`, `log`, `sin`, `cos`, `sqrt`, `pow`, `erf`, ... |
| `comparison.py` | ~150 | `eq`, `ne`, `lt`, `gt`, `le`, `ge`, `logical_*`, `bitwise_*` |
| `reduce.py` | ~850 | `sum`, `mean`, `argmax`, `argmin`, `topk`, `argsort`, `sort`, `cumsum`, `cumprod`, ... |
| `shape.py` | ~2500 | `reshape`, `flip`, `roll`, `cat`, `stack`, `gather`, `scatter`, `getitem`, `setitem`, ... |
| `activation.py` | ~500 | `relu`, `gelu`, `silu`, `softmax`, `log_softmax`, `elu`, `prelu`, ... |
| `norm.py` | ~400 | `layer_norm`, `batch_norm`, `group_norm`, `instance_norm` (with composite fallbacks) |
| `linalg.py` | ~2000 | `matmul`, `dot`, `mm`, `bmm`, `qr`, `svd`, `eig`, `solve`, `cholesky`, ... |
| `conv.py` | ~2000 | `conv1d/2d/3d`, `conv_transpose*`, `max_pool*`, `avg_pool*`, `upsample_*`, ... |
| `random.py` | ~500 | `uniform_`, `normal_`, `bernoulli_`, `dropout`, `randperm`, ... |
| `elementwise.py` | ~400 | `where`, `lerp`, `clamp`, `addcmul`, `remainder`, `isclose`, `heaviside`, ... |
| `special.py` | ~800 | `digamma`, `erfinv`, `sinc`, `fft_*`, `special_*`, ... |
| `optim.py` | ~500 | `_adam_step_op`, `_sgd_step_op`, `_adamw_step_op`, ... |

### Files to delete after refactoring

- `ops.py` — replaced by `ops/` package
- `ops_910a.py`, `ops_910b.py`, `ops_310b.py`, `ops_310p.py` — empty shells, merged into `ops_soc.py`

## Capability Table Design (`ops_soc.py`)

### Data structures

```python
# 1. Operator fallback — ops that need composite workaround on specific chips
_FALLBACK_OPS = {
    "910a": frozenset({...}),
    "910b": frozenset(),
    "310b": frozenset({"atan2", "where", "flip", "argsort", "sort", "topk",
                        "diag", "lerp", "remainder", "isclose", "softplus",
                        "uniform_", "normal_", "layer_norm", "mish",
                        "batch_norm", "dropout", "take_along_dim", "gather"}),
    "310p": frozenset({...}),
}

# 2a. Global unsupported dtypes (all ops on this chip)
_UNSUPPORTED_DTYPES_GLOBAL = {
    "910a": frozenset(),
    "910b": frozenset(),
    "310b": frozenset({"bfloat16"}),
    "310p": frozenset(),
}

# 2b. Per-op unsupported dtypes (overrides global)
_UNSUPPORTED_DTYPES_PER_OP = {
    "910a": {"matmul": frozenset({"bfloat16"})},
    "910b": {},
    "310b": {},
    "310p": {},
}

# 3. Distributed op support
_SUPPORTED_DISTRIBUTED_OPS = {
    "910a": frozenset({"all_reduce", "broadcast", "all_gather"}),
    "910b": frozenset({"all_reduce", "broadcast", "all_gather",
                        "alltoall_v", "reduce_scatter"}),
    "310b": frozenset(),
    "310p": frozenset({...}),
}

# 4. Chip-specific flags (fine-grained control)
_CHIP_FLAGS = {
    "910a": {"use_smallop_arange_1d": False, "use_smallop_linspace": False},
    "910b": {"use_smallop_arange_1d": False, "use_smallop_linspace": False},
    "310b": {"use_smallop_arange_1d": True,  "use_smallop_linspace": True},
    "310p": {"use_smallop_arange_1d": False, "use_smallop_linspace": False},
}
```

### Query API

```python
def use_fallback(op_name, profile=None) -> bool:
    """Check if op should use composite fallback on current chip."""

def check_dtype_support(op_name, dtype, profile=None) -> bool:
    """Return True if dtype is supported for op on current chip."""

def is_distributed_op_supported(op_name, profile=None) -> bool:
    """Return True if distributed op is available on current chip."""

def chip_flag(name, profile=None, default=False):
    """Query chip-specific feature flag."""

def current_profile() -> str:
    """Return cached SoC profile string (e.g., '910b')."""
```

## Module Collaboration Pattern

### `ops/__init__.py` — unified export

```python
from .math import *
from .comparison import *
from .reduce import *
from .shape import *
from .activation import *
from .norm import *
from .linalg import *
from .conv import *
from .random import *
from .elementwise import *
from .special import *
from .optim import *
```

External import paths remain unchanged: `from candle._backends.npu.ops import xxx`.

### Op module pattern

Each module follows this pattern:

```python
# ops/norm.py
from ._helpers import _unwrap_storage, _wrap_tensor, ...
from .. import ops_soc
from ..aclnn import aclnn

def layer_norm(input, normalized_shape, weight, bias, eps):
    if ops_soc.use_fallback("layer_norm"):
        return _layer_norm_composite(input, normalized_shape, weight, bias, eps)
    # Native ACLNN kernel path
    ...

def _layer_norm_composite(input, normalized_shape, weight, bias, eps):
    """On-device composite implementation for chips without native kernel."""
    ...
```

### dtype check injection

Centralized in `_helpers.py`, called at op entry points:

```python
# ops/_helpers.py
from .. import ops_soc

def check_op_dtype(op_name, *tensors):
    for t in tensors:
        if not ops_soc.check_dtype_support(op_name, t.dtype):
            raise RuntimeError(
                f"Op '{op_name}' does not support dtype {t.dtype} "
                f"on {ops_soc.current_profile()}"
            )
```

### Distributed op guard

```python
# distributed/collective.py
from .. import ops_soc

def alltoall_v(input, ...):
    if not ops_soc.is_distributed_op_supported("alltoall_v"):
        raise RuntimeError(
            f"alltoall_v is not supported on {ops_soc.current_profile()}"
        )
    ...
```

## Migration Strategy

1. Create `ops/` package with `_helpers.py` first (extract shared utilities)
2. Move functions domain-by-domain, one module at a time
3. After each module extraction, verify imports and run tests
4. Extend `ops_soc.py` with dtype and distributed tables
5. Delete old `ops.py` and empty `ops_*.py` files
6. Update any external imports referencing the old layout
