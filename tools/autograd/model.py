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

    @property
    def op_name(self) -> str:
        """Dispatch op name (before the dot)."""
        return self.name

    @property
    def backward_name(self) -> str:
        return f"{self.name.capitalize()}Backward0"

    @property
    def tensor_args(self) -> list[Argument]:
        return [a for a in self.args if a.is_tensor or a.is_optional_tensor]

    @property
    def differentiable_inputs(self) -> list[Argument]:
        all_diff = set()
        for d in self.derivatives:
            all_diff.update(d.var_names)
        return [a for a in self.tensor_args if a.name in all_diff]

    @property
    def non_tensor_args(self) -> list[Argument]:
        return [a for a in self.args if not a.is_tensor and not a.is_optional_tensor]

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
    "int": "int",
    "int[]": "int[]",
    "int?": "int?",
    "float": "float",
    "float?": "float?",
    "bool": "bool",
    "bool?": "bool?",
    "ScalarType": "ScalarType",
    "ScalarType?": "ScalarType?",
    "Scalar": "Scalar",
    "Scalar?": "Scalar?",
    "str": "str",
    "str?": "str?",
    "SymInt": "int",
    "SymInt[]": "int[]",
}

_SCHEMA_RE = re.compile(
    r"^(\w+(?:\.\w+)?)\((.*?)\)\s*->\s*(.+)$"
)

_ARG_RE = re.compile(
    r"^\s*([\w\[\]?()!]+)\s+(\w+)\s*(?:=\s*(.+))?\s*$"
)

_RETURN_RE = re.compile(
    r"^\s*(?:(\w+)\s+)?(\w+)\s*$"
)


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
            am = _ARG_RE.match(part)
            if not am:
                raise ValueError(f"Cannot parse argument: {part!r} in {schema!r}")
            type_str = am.group(1)
            arg_name = am.group(2)
            default = am.group(3)
            is_tensor = type_str in ("Tensor", "Tensor(a!)", "Tensor(a)", "Tensor[]")
            is_optional = type_str == "Tensor?"
            is_mutating = "!" in type_str
            args.append(Argument(
                name=arg_name,
                type=_TYPE_MAP.get(type_str, type_str),
                default=default,
                is_tensor=is_tensor,
                is_optional_tensor=is_optional,
                is_mutating=is_mutating,
            ))

    # Parse returns
    returns = _parse_returns(returns_str)
    return func_name, args, returns


def _split_args(s: str) -> list[str]:
    """Split argument string respecting parentheses."""
    parts = []
    depth = 0
    current = []
    for ch in s:
        if ch == '(' :
            depth += 1
            current.append(ch)
        elif ch == ')':
            depth -= 1
            current.append(ch)
        elif ch == ',' and depth == 0:
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
