"""Data model for parsed derivatives.yaml entries."""
from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class Argument:
    name: str
    type: str  # "Tensor", "int", "int[]", "float", "bool", "ScalarType?", etc.
    default: str | None = None
    is_tensor: bool = False
    is_optional_tensor: bool = False
    is_mutating: bool = False  # Tensor(a!)
    is_tensor_list: bool = False  # Tensor[]


@dataclass
class Return:
    name: str | None = None
    type: str = "Tensor"


@dataclass
class Derivative:
    formula: str
    var_names: tuple[str, ...]
    saved_inputs: tuple[str, ...]
    saved_outputs: tuple[str, ...]


@dataclass
class DifferentiabilityInfo:
    name: str                                    # e.g. "mul"
    func_name: str                               # e.g. "mul.Tensor"
    schema: str                                  # full signature string
    derivatives: list[Derivative]
    all_saved_inputs: list[str]
    all_saved_outputs: list[str]
    args: list[Argument]
    returns: list[Return]
    output_differentiability: list[bool] | None = None
    non_differentiable: set[str] = field(default_factory=set)
    view_autograd: bool = False

    @property
    def op_name(self) -> str:
        """Dispatch op name (before the dot)."""
        return self.name

    @property
    def generated_func_stem(self) -> str:
        """Stable Python-safe stem derived from the full overloaded function name."""
        return re.sub(r"[^0-9A-Za-z_]+", "_", self.func_name).lower()

    @property
    def backward_name(self) -> str:
        preserved_names = {
            "column_stack": "Column_stackBackward0",
            "fft_fft": "Fft_fftBackward0",
            "fft_fft2": "Fft_fft2Backward0",
            "fft_fftn": "Fft_fftnBackward0",
            "fft_fftshift": "Fft_fftshiftBackward0",
            "fft_hfft": "Fft_hfftBackward0",
            "fft_ifft": "Fft_ifftBackward0",
            "fft_ifft2": "Fft_ifft2Backward0",
            "fft_ifftn": "Fft_ifftnBackward0",
            "fft_ifftshift": "Fft_ifftshiftBackward0",
            "fft_ihfft": "Fft_ihfftBackward0",
            "fft_irfft": "Fft_irfftBackward0",
            "fft_irfft2": "Fft_irfft2Backward0",
            "fft_irfftn": "Fft_irfftnBackward0",
            "fft_rfft": "Fft_rfftBackward0",
            "fft_rfft2": "Fft_rfft2Backward0",
            "fft_rfftn": "Fft_rfftnBackward0",
            "floor_divide": "Floor_divideBackward0",
            "adaptive_avg_pool1d": "Adaptive_avg_pool1dBackward0",
            "adaptive_avg_pool2d": "Adaptive_avg_pool2dBackward0",
            "adaptive_avg_pool3d": "Adaptive_avg_pool3dBackward0",
            "adaptive_max_pool1d": "Adaptive_max_pool1dBackward0",
            "avg_pool1d": "Avg_pool1dBackward0",
            "max_pool1d": "Max_pool1dBackward0",
            "max_pool3d": "Max_pool3dBackward0",
            "conv_transpose1d": "Conv_transpose1dBackward0",
            "conv_transpose2d": "Conv_transpose2dBackward0",
            "conv_transpose3d": "Conv_transpose3dBackward0",
            "affine_grid": "Affine_gridBackward0",
            "grid_sample": "Grid_sampleBackward0",
            "ctc_loss": "Ctc_lossBackward0",
            "instance_norm": "Instance_normBackward0",
            "linalg_slogdet": "Linalg_slogdetBackward0",
            "batch_norm": "Batch_normBackward0",
            "group_norm": "Group_normBackward0",
            "layer_norm": "Layer_normBackward0",
            "rms_norm": "Rms_normBackward0",
            "linalg_cholesky": "Linalg_choleskyBackward0",
            "linalg_cond": "Linalg_condBackward0",
            "linalg_det": "Linalg_detBackward0",
            "linalg_eigvals": "Linalg_eigvalsBackward0",
            "linalg_eigvalsh": "Linalg_eigvalshBackward0",
            "linalg_inv": "Linalg_invBackward0",
            "linalg_matrix_norm": "Linalg_matrix_normBackward0",
            "linalg_matrix_power": "Linalg_matrix_powerBackward0",
            "linalg_matrix_rank": "Linalg_matrix_rankBackward0",
            "linalg_norm": "Linalg_normBackward0",
            "linalg_solve": "Linalg_solveBackward0",
            "linalg_svdvals": "Linalg_svdvalsBackward0",
            "linalg_tensorinv": "Linalg_tensorinvBackward0",
            "linalg_tensorsolve": "Linalg_tensorsolveBackward0",
            "linalg_vander": "Linalg_vanderBackward0",
            "log_softmax": "Log_softmaxBackward0",
            "matrix_power": "Matrix_powerBackward0",
            "pad_sequence": "Pad_sequenceBackward0",
            "row_stack": "Row_stackBackward0",
            "special_digamma": "Special_digammaBackward0",
            "special_erfinv": "Special_erfinvBackward0",
            "special_gammainc": "Special_gammaincBackward0",
            "special_gammaincc": "Special_gammainccBackward0",
            "special_gammaln": "Special_gammalnBackward0",
            "special_i0": "Special_i0Backward0",
            "special_logit": "Special_logitBackward0",
            "special_multigammaln": "Special_multigammalnBackward0",
            "special_ndtr": "Special_ndtrBackward0",
            "special_polygamma": "Special_polygammaBackward0",
            "special_sinc": "Special_sincBackward0",
            "special_xlogy": "Special_xlogyBackward0",
            "true_divide": "True_divideBackward0",
        }
        if self.func_name in preserved_names:
            return preserved_names[self.func_name]
        parts = [p for p in re.split(r"[^0-9A-Za-z]+", self.func_name) if p]
        camel = "".join(p[:1].upper() + p[1:] for p in parts)
        return f"{camel}Backward0"

    @property
    def tensor_args(self) -> list[Argument]:
        return [a for a in self.args if a.is_tensor or a.is_optional_tensor or a.is_tensor_list]

    @property
    def differentiable_inputs(self) -> list[Argument]:
        all_diff = set()
        for d in self.derivatives:
            all_diff.update(d.var_names)
        return [a for a in self.tensor_args if a.name in all_diff]

    @property
    def non_tensor_args(self) -> list[Argument]:
        return [a for a in self.args if not a.is_tensor and not a.is_optional_tensor and not a.is_tensor_list]

    @property
    def num_outputs(self) -> int:
        return len(self.returns)

    @property
    def is_multi_output(self) -> bool:
        return self.num_outputs > 1

    @property
    def is_inplace(self) -> bool:
        return any(a.is_mutating for a in self.args)


# ---------------------------------------------------------------------------
# Schema parsing helpers
# ---------------------------------------------------------------------------

_TENSOR_TYPES = {"Tensor", "Tensor?", "Tensor(a!)", "Tensor(a)", "Tensor[]"}

_TYPE_MAP = {
    "Tensor": "Tensor",
    "Tensor?": "Tensor?",
    "Tensor(a!)": "Tensor",
    "Tensor(a)": "Tensor",
    "Tensor[]": "Tensor[]",
    "Tensor?[]": "Tensor?[]",
    "int": "int",
    "int[]": "int[]",
    "int?": "int?",
    "int[1]": "int[]",
    "int[1]?": "int[]?",
    "int[2]": "int[]",
    "int[2]?": "int[]?",
    "int[3]": "int[]",
    "int[4]": "int[]",
    "float": "float",
    "float?": "float?",
    "bool": "bool",
    "bool?": "bool?",
    "bool[3]": "bool[]",
    "bool[4]": "bool[]",
    "ScalarType": "ScalarType",
    "ScalarType?": "ScalarType?",
    "Scalar": "Scalar",
    "Scalar?": "Scalar?",
    "Scalar[]": "Scalar[]",
    "str": "str",
    "str?": "str?",
    "SymInt": "int",
    "SymInt?": "int?",
    "SymInt[]": "int[]",
    "SymInt[]?": "int[]?",
    "SymInt[1]": "int[]",
    "SymInt[2]": "int[]",
    "SymInt[3]": "int[]",
    "SymInt[4]": "int[]",
    "SymInt[5]": "int[]",
    "SymInt[6]": "int[]",
    "Generator?": "Generator?",
    "MemoryFormat?": "MemoryFormat?",
    "Layout?": "Layout?",
    "Device?": "Device?",
}

# Regex for Tensor alias types: Tensor(a), Tensor(a!), Tensor(b!), Tensor(a -> *), etc.
_TENSOR_ALIAS_RE = re.compile(r"^Tensor\([a-z](?:!| -> \*)*\)(?:\[\])?$")


_SCHEMA_RE = re.compile(
    r"^(\w+(?:\.\w+)?)\((.*?)\)\s*->\s*(.+)$"
)

# Matches: "Type name", "Type name=default", handles Tensor(a!), Tensor(a -> *), etc.
_ARG_RE = re.compile(
    r"^\s*([\w\[\]?()! >*-]+?)\s+(\w+)\s*(?:=\s*(.+))?\s*$"
)

_RETURN_RE = re.compile(
    r"^\s*(?:(\w+)\s+)?(\w+)\s*$"
)


_REDUCTION_DEFAULTS = {
    "None": "0",
    "Mean": "1",
    "Sum": "2",
}


def _normalize_default(type_str: str, default: str | None) -> str | None:
    """Normalize schema defaults into valid Python literals."""
    if default is None:
        return None
    if type_str == "int" and default in _REDUCTION_DEFAULTS:
        return _REDUCTION_DEFAULTS[default]
    return default


def _resolve_type(type_str: str) -> str:
    """Resolve a raw type string to its canonical form."""
    if type_str in _TYPE_MAP:
        return _TYPE_MAP[type_str]
    # Handle dynamic Tensor alias types: Tensor(b!), Tensor(c!), Tensor(a -> *), etc.
    if _TENSOR_ALIAS_RE.match(type_str):
        if type_str.endswith("[]"):
            return "Tensor[]"
        return "Tensor"
    return type_str


def _is_tensor_list_type(type_str: str) -> bool:
    return type_str in ("Tensor[]", "Tensor?[]") or (
        type_str.endswith("[]") and _TENSOR_ALIAS_RE.match(type_str) is not None
    )


def _is_tensor_type(type_str: str) -> bool:
    return type_str == "Tensor" or (
        _TENSOR_ALIAS_RE.match(type_str) is not None and not type_str.endswith("[]")
    )


def _is_optional_tensor_type(type_str: str) -> bool:
    return type_str == "Tensor?"


def parse_schema(schema: str) -> tuple[str, list[Argument], list[Return]]:
    """Parse a function schema string into (func_name, args, returns)."""
    m = _SCHEMA_RE.match(schema.strip())
    if not m:
        raise ValueError(f"Cannot parse schema: {schema!r}")
    func_name = m.group(1)
    args_str = m.group(2)
    returns_str = m.group(3).strip()

    # Parse arguments
    args = []
    if args_str.strip():
        for part in _split_args(args_str):
            part = part.strip()
            if not part:
                continue
            # Skip * keyword-only separator
            if part == '*':
                continue
            am = _ARG_RE.match(part)
            if not am:
                raise ValueError(f"Cannot parse argument: {part!r} in {schema!r}")
            type_str = am.group(1)
            arg_name = am.group(2)
            default = _normalize_default(type_str, am.group(3))
            is_tensor_list = _is_tensor_list_type(type_str)
            is_tensor = _is_tensor_type(type_str) and not is_tensor_list
            is_optional = _is_optional_tensor_type(type_str)
            is_mutating = "!" in type_str
            args.append(Argument(
                name=arg_name,
                type=_resolve_type(type_str),
                default=default,
                is_tensor=is_tensor,
                is_optional_tensor=is_optional,
                is_mutating=is_mutating,
                is_tensor_list=is_tensor_list,
            ))

    # Parse returns
    returns = _parse_returns(returns_str)
    return func_name, args, returns


def _split_args(s: str) -> list[str]:
    """Split argument string respecting parentheses and bracketed defaults."""
    parts = []
    paren_depth = 0
    bracket_depth = 0
    current = []
    for ch in s:
        if ch == '(':
            paren_depth += 1
            current.append(ch)
        elif ch == ')':
            paren_depth -= 1
            current.append(ch)
        elif ch == '[':
            bracket_depth += 1
            current.append(ch)
        elif ch == ']':
            bracket_depth -= 1
            current.append(ch)
        elif ch == ',' and paren_depth == 0 and bracket_depth == 0:
            parts.append(''.join(current))
            current = []
        else:
            current.append(ch)
    if current:
        parts.append(''.join(current))
    return parts


def _parse_returns(returns_str: str) -> list[Return]:
    """Parse return type(s)."""
    returns_str = returns_str.strip()
    # Multi-output: (Type name, Type name, ...)
    if returns_str.startswith("(") and returns_str.endswith(")"):
        inner = returns_str[1:-1]
        parts = [p.strip() for p in inner.split(",")]
        returns = []
        for p in parts:
            tokens = p.split()
            if len(tokens) == 2:
                returns.append(Return(name=tokens[1], type=tokens[0]))
            else:
                returns.append(Return(name=None, type=tokens[0]))
        return returns
    # Single output
    return [Return(name=None, type=returns_str)]
