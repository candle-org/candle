"""Parse derivatives.yaml into DifferentiabilityInfo objects."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Sequence

import yaml

from .model import (
    Argument,
    Derivative,
    DifferentiabilityInfo,
    parse_schema,
)

# Variables that refer to forward outputs
_OUTPUT_NAMES = {
    "result",
    "result0",
    "result1",
    "result2",
    "result3",
    "values",
    "indices",
    "Q",
    "R",
    "LU",
    "pivots",
    "info",
}

# Identifiers that are NOT saved variables (builtins / helpers)
_FORMULA_BUILTINS = {
    "grad",
    "grads",
    "grad_input_mask",
    "grad_output",
    "grad_out",
    "retain_variables",
    "None",
    "True",
    "False",
    "reduce_grad",
    "maybe_multiply",
    "handle_r_to_c",
    "redispatch",
    "_grad_context",
    "_scalar_tensor_like",
    "at",
    "std",
    "GradMode",
    "Tensor",
    "TensorGeometry",
    "IntArrayRef",
    "auto_linear",
    "auto_element_wise",
    "not_implemented",
    "zeros_like",
    "ones_like",
}



def load_derivatives(path: str | Path) -> list[DifferentiabilityInfo]:
    """Load and parse a derivatives.yaml file."""
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        entries = yaml.safe_load(f)
    if not entries:
        return []
    result = []
    for entry in entries:
        info = _parse_entry(entry)
        result.append(info)
    return result



def _parse_entry(entry: dict) -> DifferentiabilityInfo:
    schema = entry["name"]
    func_name, args, returns = parse_schema(schema)

    # Base name (before the dot)
    name = func_name.split(".")[0]

    output_diff = entry.get("output_differentiability", None)
    non_diff: set[str] = set()

    derivative_items = _iter_derivative_items(entry)
    arg_names = {a.name for a in args}
    save_inputs_override = entry.get("save_inputs", None)
    derivatives: list[Derivative] = []
    for key, value in derivative_items:
        var_names = tuple(v.strip() for v in key.split(","))
        if not all(v in arg_names or v in _OUTPUT_NAMES for v in var_names):
            continue
        # Skip forward-derivative / output-derivative entries for now.
        # gen_functions.py only emits backward formulas for input gradients.
        if not any(v in arg_names for v in var_names):
            continue
        if value == "non_differentiable":
            non_diff.update(v for v in var_names if v in arg_names)
            continue
        formula = str(value)
        saved_inputs, saved_outputs = _analyze_formula(formula, args)
        derivatives.append(Derivative(
            formula=formula,
            var_names=tuple(v for v in var_names if v in arg_names),
            saved_inputs=tuple(saved_inputs),
            saved_outputs=tuple(saved_outputs),
        ))

    all_saved_inputs: list[str] = []
    all_saved_outputs: list[str] = []
    seen_in = set()
    seen_out = set()
    for derivative in derivatives:
        for name_ in derivative.saved_inputs:
            if name_ not in seen_in:
                all_saved_inputs.append(name_)
                seen_in.add(name_)
        for name_ in derivative.saved_outputs:
            if name_ not in seen_out:
                all_saved_outputs.append(name_)
                seen_out.add(name_)

    if save_inputs_override is not None:
        allowed = set(save_inputs_override)
        all_saved_inputs = [name_ for name_ in all_saved_inputs if name_ in allowed]

    return DifferentiabilityInfo(
        name=name,
        func_name=func_name,
        schema=schema,
        derivatives=derivatives,
        all_saved_inputs=all_saved_inputs,
        all_saved_outputs=all_saved_outputs,
        args=args,
        returns=returns,
        output_differentiability=output_diff,
        non_differentiable=non_diff,
    )



def _iter_derivative_items(entry: dict) -> list[tuple[str, object]]:
    dispatch = entry.get("dispatch")
    if not dispatch:
        return [
            (key, value)
            for key, value in entry.items()
            if key not in ("name", "output_differentiability", "save_inputs", "dispatch")
        ]

    default_dispatch = dispatch.get("Default", {})
    return list(default_dispatch.items()) + [
        (key, value)
        for key, value in entry.items()
        if key not in ("name", "output_differentiability", "save_inputs", "dispatch")
    ]


# Regex to find identifiers in a formula
_IDENT_RE = re.compile(r"\b([a-zA-Z_]\w*)\b")



def _analyze_formula(
    formula: str, args: Sequence[Argument]
) -> tuple[list[str], list[str]]:
    """Determine which forward inputs and outputs are referenced in a formula."""
    tensor_arg_names = {
        a.name for a in args if a.is_tensor or a.is_optional_tensor or a.is_tensor_list
    }
    identifiers = set(_IDENT_RE.findall(formula))

    saved_inputs = []
    saved_outputs = []
    for ident in identifiers:
        if ident in _FORMULA_BUILTINS:
            continue
        if ident in tensor_arg_names:
            saved_inputs.append(ident)
        elif ident in _OUTPUT_NAMES or ident.startswith("result"):
            saved_outputs.append(ident)

    return sorted(saved_inputs), sorted(saved_outputs)
