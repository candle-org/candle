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
    Return,
    parse_schema,
)

# Variables that refer to forward outputs
_OUTPUT_NAMES = {"result", "result0", "result1", "result2", "result3"}

# Identifiers that are NOT saved variables (builtins / helpers)
_FORMULA_BUILTINS = {
    "grad", "grads", "grad_input_mask", "None", "True", "False",
    "reduce_grad", "maybe_multiply", "handle_r_to_c",
    "redispatch", "_grad_context", "_scalar_tensor_like",
}


def load_derivatives(path: str | Path) -> list[DifferentiabilityInfo]:
    """Load and parse a derivatives.yaml file."""
    path = Path(path)
    with open(path) as f:
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

    # Parse output_differentiability
    output_diff = entry.get("output_differentiability", None)

    # Parse non_differentiable
    non_diff: set[str] = set()

    # Collect derivatives
    arg_names = {a.name for a in args}
    save_inputs_override = entry.get("save_inputs", None)
    derivatives: list[Derivative] = []
    for key, value in entry.items():
        if key in ("name", "output_differentiability", "save_inputs"):
            continue
        # key is comma-separated var names like "self" or "input, weight, bias"
        var_names = tuple(v.strip() for v in key.split(","))
        # Check all var_names are actual arguments
        if not all(v in arg_names for v in var_names):
            continue
        if value == "non_differentiable":
            non_diff.update(var_names)
            continue
        formula = str(value)
        saved_inputs, saved_outputs = _analyze_formula(formula, args)
        derivatives.append(Derivative(
            formula=formula,
            var_names=var_names,
            saved_inputs=tuple(saved_inputs),
            saved_outputs=tuple(saved_outputs),
        ))

    # Aggregate all saved inputs/outputs
    all_saved_inputs: list[str] = []
    all_saved_outputs: list[str] = []
    seen_in = set()
    seen_out = set()
    for d in derivatives:
        for s in d.saved_inputs:
            if s not in seen_in:
                all_saved_inputs.append(s)
                seen_in.add(s)
        for s in d.saved_outputs:
            if s not in seen_out:
                all_saved_outputs.append(s)
                seen_out.add(s)

    # Apply explicit save_inputs override if provided
    if save_inputs_override is not None:
        allowed = set(save_inputs_override)
        all_saved_inputs = [s for s in all_saved_inputs if s in allowed]

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


# Regex to find Python identifiers in a formula
_IDENT_RE = re.compile(r"\b([a-zA-Z_]\w*)\b")


def _analyze_formula(
    formula: str, args: Sequence[Argument]
) -> tuple[list[str], list[str]]:
    """Determine which forward inputs and outputs are referenced in a formula.

    Only tensor arguments go into saved_inputs (non-tensor args are stored
    separately as plain attributes on the Node).
    """
    tensor_arg_names = {a.name for a in args if a.is_tensor or a.is_optional_tensor}
    identifiers = set(_IDENT_RE.findall(formula))

    saved_inputs = []
    saved_outputs = []
    for ident in identifiers:
        if ident in _FORMULA_BUILTINS:
            continue
        if ident in tensor_arg_names:
            saved_inputs.append(ident)
        elif ident in _OUTPUT_NAMES:
            saved_outputs.append(ident)
        # non-tensor args and function names are skipped

    return sorted(saved_inputs), sorted(saved_outputs)
