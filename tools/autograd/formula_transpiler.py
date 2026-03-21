"""Transpile PyTorch C++ derivative formulas into Candle Python formulas."""
from __future__ import annotations

import re


_STD_GET_RE = re.compile(r"^std::get<(\d+)>\((.*)\)$")
_STD_VECTOR_REPEAT_RE = re.compile(r"^std::vector<c10::SymInt>\((.*),\s*(.*)\)$")
_SIMPLE_NAME_RE = re.compile(r"^[A-Za-z_:][A-Za-z0-9_:<>]*$")
_METHOD_NAME_RE = re.compile(r"^[A-Za-z_]\w*$")
_ENUM_CONSTANT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*::[A-Za-z_][A-Za-z0-9_]*::[A-Za-z_][A-Za-z0-9_]*$")
_NONE_TUPLE_RE = re.compile(r"^\((?:None\s*,\s*)*None(?:\s*,\s*None)*\)$")

_SPECIAL_IDENTIFIERS = {
    "grad",
    "grads",
    "result",
    "result0",
    "result1",
    "result2",
    "result3",
    "True",
    "False",
    "None",
    "keyset",
}

_BINARY_OPS = {
    "+": "add",
    "-": "sub",
    "*": "mul",
    "/": "div",
}

_COMPARISON_OPS = {
    "==": "eq",
    "!=": "ne",
    "<=": "le",
    ">=": "ge",
    "<": "lt",
    ">": "gt",
}

_BOOL_LITERALS = {
    "true": "True",
    "false": "False",
    "nullptr": "None",
}

_SPECIAL_CALLS = {
    "zeros_like": lambda args: f"{args[0]}._zeros_like()",
    "ones_like": lambda args: f"{args[0]}._ones_like()",
    "handle_r_to_c": lambda args: args[1],
    "maybe_multiply": lambda args: f'redispatch("mul", keyset, {args[0]}, {args[1]})',
    "not_implemented": lambda args: f'_raise_not_implemented({args[0]})',
    "std::sqrt": lambda args: f'redispatch("sqrt", keyset, {args[0]})',
    "std::real": lambda args: args[0],
    "sum_backward": lambda args: f'_sum_backward_helper({", ".join([*args, "keyset"])})',
    "mean_backward": lambda args: f'_mean_backward_helper({", ".join([*args, "keyset"])})',
    "sigmoid_backward": lambda args: f'_sigmoid_backward_helper({", ".join([*args, "keyset"])})',
    "tanh_backward": lambda args: f'_tanh_backward_helper({", ".join([*args, "keyset"])})',
    "threshold_backward": lambda args: f'_threshold_backward_helper({", ".join([*args, "keyset"])})',
    "softplus_backward": lambda args: f'_softplus_backward_helper({", ".join([*args, "keyset"])})',
    "hardtanh_backward": lambda args: f'_hardtanh_backward_helper({", ".join([*args, "keyset"])})',
    "hardswish_backward": lambda args: f'_hardswish_backward_helper({", ".join([*args, "keyset"])})',
    "hardsigmoid_backward": lambda args: f'_hardsigmoid_backward_helper({", ".join([*args, "keyset"])})',
    "mul_tensor_backward": lambda args: f'_mul_tensor_backward_helper({", ".join([*args, "keyset"])})',
    "div_tensor_self_backward": lambda args: f'_div_tensor_self_backward_helper({", ".join([*args, "keyset"])})',
    "div_tensor_other_backward": lambda args: f'_div_tensor_other_backward_helper({", ".join([*args, "keyset"])})',
    "matmul_backward": lambda args: f'_matmul_backward_helper({", ".join([*args, "keyset"])})',
    "gelu_backward": lambda args: f'_gelu_backward_helper({", ".join([*args, "keyset"])})',
    "unsqueeze_to": lambda args: f'_unsqueeze_to_backward_helper({", ".join([*args, "keyset"])})',
    "sum_to": lambda args: f'_sum_to_backward_helper({", ".join([*args, "keyset"])})',
    "pow_backward": lambda args: f'_pow_backward_helper({", ".join([*args, "keyset"])})',
    "pow_backward_self": lambda args: f'_pow_backward_self_helper({", ".join([*args, "keyset"])})',
    "pow_backward_exponent": lambda args: f'_pow_backward_exponent_helper({", ".join([*args, "keyset"])})',
    "TensorGeometry": lambda args: args[0],
    "slice_backward_wrapper": lambda args: f'_slice_backward_wrapper_helper({", ".join([*args, "keyset"])})',
    "as_strided_backward": lambda args: f'_as_strided_backward_helper({", ".join([*args, "keyset"])})',
    "as_strided_scatter_backward": lambda args: f'_as_strided_scatter_backward_helper({", ".join([*args, "keyset"])})',
    "clamp_backward": lambda args: f'_clamp_backward_helper({", ".join([*args, "keyset"])})',
    "expand_symint": lambda args: f'redispatch("expand", keyset, {", ".join(args)})',
    "permute_backwards": lambda args: f'_permute_backward_helper({", ".join([*args, "keyset"])})',
    "select_backward_symint": lambda args: f'_select_backward_symint_helper({", ".join([*args, "keyset"])})',
    "cat_tensors_backward": lambda args: f'_cat_backward_helper({args[0]}, tensors, {args[-1]}, keyset)',
    "stack_tensors_backward": lambda args: f'_stack_backward_helper({args[0]}, tensors, {args[1]}, keyset)',
}

_HELPER_FALLBACKS = {
    "infinitely_differentiable_native_group_norm_backward": "_native_group_norm_helper(grads, input, result1, result2, weight, N, C, HxW, group, eps, grad_input_mask)",
    "infinitely_differentiable_silu_backward": "_silu_grad(grad, self, keyset)",
    "clamp_backward(grad, self, min, max)": "_clamp_backward_helper(grad, self, min, max, keyset)",
    "fmap(reverse_list_symint": "_roll_backward_helper(grad, shifts, dims)",
    "grads[0].defined() ? (grads[1].defined() ? at::where(self >= 0, grads[0], grads[0] * weight + grads[1] * self)": "_prelu_kernel_backward_grad_output_helper(grads, self, weight, grad_output)",
    "std::get<0>(convolution_backward_symint(grad_output_p, input_p, weight_t": "_convolution_backward_jvp_input_helper(grad_output_p, grad_output_t, input_p, weight_p, weight_t, bias_sizes, stride, padding, dilation, transposed, output_padding, groups)",
    "std::get<1>(convolution_backward_symint(grad_output_p, input_t": "_convolution_backward_jvp_weight_helper(grad_output_p, grad_output_t, input_p, input_t, weight_p, bias_sizes, stride, padding, dilation, transposed, output_padding, groups)",
    "self.layout() == c10::kJagged": "_to_padded_tensor_backward_helper(grad, self)",
    "grad / (2 * result.conj())": "_sqrt_backward_helper(grad, result, keyset)",
    "-0.5 * grad * result.pow(3).conj()": "_rsqrt_backward_helper(grad, result, keyset)",
    "grad * result.conj() * M_LN2": "_exp2_backward_helper(grad, result, keyset)",
    "2.0 / sqrt(M_PI) * exp(-(self.pow(2))) * grad": "_erf_backward_helper(grad, self, keyset)",
    "at::sum_to(grad, self.sym_sizes())": "_sum_to_backward_helper(grad, self.shape, keyset)",
}


class TranspileError(ValueError):
    """Raised when a PyTorch derivative formula cannot be transpiled."""



def transpile(formula: str) -> str:
    text = formula.strip()
    if not text:
        return text
    if text.startswith('"') and text.endswith('"'):
        text = text[1:-1]
    text = _replace_cpp_literals(text)
    text = _strip_comments(text)
    if text in {"auto_linear", "auto_element_wise", "non_differentiable"}:
        return text
    for prefix, helper in _HELPER_FALLBACKS.items():
        if prefix in text:
            return helper
    return _transpile_expr(text)



def _replace_cpp_literals(text: str) -> str:
    for old, new in _BOOL_LITERALS.items():
        text = re.sub(rf"\b{re.escape(old)}\b", new, text)
    text = text.replace("std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor>()", "(None, None, None, None, None)")
    text = text.replace("std::tuple<Tensor, Tensor, Tensor>()", "(None, None, None)")
    text = text.replace("std::tuple<Tensor, Tensor>()", "(None, None)")
    text = text.replace("Tensor()", "None")
    return text



def _strip_comments(text: str) -> str:
    return re.sub(r"/\*.*?\*/", "", text).strip()



def _transpile_expr(expr: str) -> str:
    expr = expr.strip()
    if not expr:
        return expr

    if _NONE_TUPLE_RE.fullmatch(expr):
        return expr

    expr = _unwrap_parens(expr)

    ternary = _split_ternary(expr)
    if ternary is not None:
        cond, yes, no = ternary
        return f"({_transpile_expr(yes)} if {_transpile_expr(cond)} else {_transpile_expr(no)})"

    std_get = _STD_GET_RE.match(expr)
    if std_get:
        index, value = std_get.groups()
        return f"{_transpile_expr(value)}[{index}]"

    vector_repeat = _STD_VECTOR_REPEAT_RE.match(expr)
    if vector_repeat:
        size_expr, value_expr = vector_repeat.groups()
        return f"[{_transpile_expr(value_expr)}] * {_transpile_expr(size_expr)}"

    if expr.startswith('{') and expr.endswith('}'):
        inner = expr[1:-1].strip()
        if not inner:
            return '[]'
        items = [_transpile_expr(arg) for arg in _parse_args(inner)]
        return f"[{', '.join(items)}]"

    if expr in _SPECIAL_IDENTIFIERS or re.fullmatch(r"\d+(?:\.\d+)?", expr):
        return expr

    if expr in _BOOL_LITERALS.values():
        return expr

    if expr.startswith('grad_input_mask[') and expr.endswith(']'):
        return 'True'

    if expr.startswith('GradMode::'):
        return 'True'

    if _ENUM_CONSTANT_RE.fullmatch(expr):
        return expr

    if expr.endswith('.sizes()'):
        return f"{_transpile_expr(expr[:-8])}.shape"
    if expr.endswith('.sym_sizes()'):
        return f"{_transpile_expr(expr[:-12])}.shape"

    func_call = _split_function_call(expr)
    if func_call is not None:
        name, arg_string = func_call
        args = [_transpile_expr(arg) for arg in _parse_args(arg_string)]
        if name in _SPECIAL_CALLS:
            return _SPECIAL_CALLS[name](args)
        if name.startswith('at::'):
            name = name[4:]
        if name.startswith('std::'):
            raise TranspileError(f"Unsupported std:: call: {expr}")
        return f'redispatch("{name}", keyset, {", ".join(args)})'

    for op in ('||', '&&'):
        parts = _split_top_level(expr, op)
        if parts is not None:
            lhs, rhs = parts
            joiner = 'or' if op == '||' else 'and'
            return f"({_transpile_expr(lhs)} {joiner} {_transpile_expr(rhs)})"

    if expr.startswith('!') and not expr.startswith('!='):
        return f"(not {_transpile_expr(expr[1:].strip())})"

    for op, name in _COMPARISON_OPS.items():
        parts = _split_top_level(expr, op)
        if parts is not None:
            lhs, rhs = parts
            return f'redispatch("{name}", keyset, {_transpile_expr(lhs)}, {_transpile_expr(rhs)})'

    for op in ('+', '-'):
        parts = _split_top_level(expr, op)
        if parts is not None:
            lhs, rhs = parts
            return f'redispatch("{_BINARY_OPS[op]}", keyset, {_transpile_expr(lhs)}, {_transpile_expr(rhs)})'

    for op in ('*', '/'):
        parts = _split_top_level(expr, op)
        if parts is not None:
            lhs, rhs = parts
            return f'redispatch("{_BINARY_OPS[op]}", keyset, {_transpile_expr(lhs)}, {_transpile_expr(rhs)})'

    if expr.startswith('-') and not expr.startswith('->'):
        return f'redispatch("neg", keyset, {_transpile_expr(expr[1:].strip())})'

    method_call = _split_method_call(expr)
    if method_call is not None:
        target, name, arg_string = method_call
        target_py = _transpile_expr(target)
        args_py = [_transpile_expr(arg) for arg in _parse_args(arg_string)]
        if name == 't' and not args_py:
            return f'redispatch("transpose", keyset, {target_py}, 0, 1)'
        if name == 'conj' and not args_py:
            return target_py
        if name == 'sgn' and not args_py:
            return f'redispatch("sign", keyset, {target_py})'
        if name == 'toDouble' and not args_py:
            return f'float({target_py})'
        if name == 'toFloat' and not args_py:
            return f'float({target_py})'
        if name == 'scalar_type' and not args_py:
            return f'{target_py}.dtype'
        if name == 'options' and not args_py:
            return f'{target_py}.options'
        if name == 'defined' and not args_py:
            return f'({target_py} is not None)'
        if name == 'sym_size' and len(args_py) == 1:
            return f'{target_py}.shape[{args_py[0]}]'
        if name == 'sym_sizes' and not args_py:
            return f'{target_py}.shape'
        if name == 'sym_numel' and not args_py:
            return f'{target_py}.numel()'
        if name == 'contiguous' and not args_py:
            return f'redispatch("contiguous", keyset, {target_py})'
        if name == 'expand_symint' and len(args_py) == 1:
            return f'redispatch("expand", keyset, {target_py}, {args_py[0]})'
        if name == 'reshape' and len(args_py) == 1:
            return f'redispatch("reshape", keyset, {target_py}, {args_py[0]})'
        return f'redispatch("{name}", keyset, {", ".join([target_py, *args_py])})'

    if '::' in expr:
        raise TranspileError(f"Unsupported namespace expression: {expr}")

    return expr



def _split_function_call(expr: str):
    open_idx = expr.find('(')
    if open_idx <= 0 or not expr.endswith(')'):
        return None
    name = expr[:open_idx].strip()
    if not _SIMPLE_NAME_RE.fullmatch(name):
        return None
    if _matching_paren(expr, open_idx) != len(expr) - 1:
        return None
    return name, expr[open_idx + 1:-1]



def _split_method_call(expr: str):
    if not expr.endswith(')'):
        return None
    open_idx = _find_call_open(expr)
    if open_idx is None:
        return None
    dot_idx = _find_method_dot(expr, open_idx)
    if dot_idx is None:
        return None
    sep_len = 2 if expr[dot_idx:dot_idx + 2] == '->' else 1
    target = expr[:dot_idx].strip()
    name = expr[dot_idx + sep_len:open_idx].strip()
    if not target or not _METHOD_NAME_RE.fullmatch(name):
        return None
    if _matching_paren(expr, open_idx) != len(expr) - 1:
        return None
    return target, name, expr[open_idx + 1:-1]



def _find_call_open(expr: str):
    depth = 0
    for idx in range(len(expr) - 1, -1, -1):
        ch = expr[idx]
        if ch == ')':
            depth += 1
        elif ch == '(':
            depth -= 1
            if depth == 0:
                return idx
    return None



def _find_method_dot(expr: str, open_idx: int):
    paren_depth = 0
    bracket_depth = 0
    brace_depth = 0
    for idx in range(open_idx - 1, -1, -1):
        ch = expr[idx]
        if ch == ')':
            paren_depth += 1
        elif ch == '(':
            paren_depth -= 1
        elif ch == ']':
            bracket_depth += 1
        elif ch == '[':
            bracket_depth -= 1
        elif ch == '}':
            brace_depth += 1
        elif ch == '{':
            brace_depth -= 1
        elif ch == '.' and paren_depth == 0 and bracket_depth == 0 and brace_depth == 0:
            return idx
        elif ch == '>' and idx > 0 and expr[idx - 1] == '-' and paren_depth == 0 and bracket_depth == 0 and brace_depth == 0:
            return idx - 1
    return None



def _matching_paren(expr: str, open_idx: int) -> int:
    depth = 0
    for idx in range(open_idx, len(expr)):
        ch = expr[idx]
        if ch == '(':
            depth += 1
        elif ch == ')':
            depth -= 1
            if depth == 0:
                return idx
    return -1



def _parse_args(arg_string: str) -> list[str]:
    if not arg_string.strip():
        return []
    args = []
    paren_depth = 0
    bracket_depth = 0
    brace_depth = 0
    current: list[str] = []
    for ch in arg_string:
        if ch == '(':
            paren_depth += 1
        elif ch == ')':
            paren_depth -= 1
        elif ch == '[':
            bracket_depth += 1
        elif ch == ']':
            bracket_depth -= 1
        elif ch == '{':
            brace_depth += 1
        elif ch == '}':
            brace_depth -= 1
        elif ch == ',' and paren_depth == 0 and bracket_depth == 0 and brace_depth == 0:
            args.append(''.join(current).strip())
            current = []
            continue
        current.append(ch)
    if current:
        args.append(''.join(current).strip())
    return args



def _unwrap_parens(expr: str) -> str:
    while expr.startswith('(') and expr.endswith(')'):
        depth = 0
        valid = True
        for idx, ch in enumerate(expr):
            if ch == '(':
                depth += 1
            elif ch == ')':
                depth -= 1
                if depth == 0 and idx != len(expr) - 1:
                    valid = False
                    break
        if not valid:
            break
        expr = expr[1:-1].strip()
    return expr



def _split_top_level(expr: str, op: str):
    depth = 0
    bracket_depth = 0
    brace_depth = 0
    in_string = False
    last_match = None
    idx = 0
    while idx <= len(expr) - len(op):
        ch = expr[idx]
        if ch == '"':
            in_string = not in_string
            idx += 1
            continue
        if in_string:
            idx += 1
            continue
        if ch == '(':
            depth += 1
            idx += 1
            continue
        if ch == ')':
            depth -= 1
            idx += 1
            continue
        if ch == '[':
            bracket_depth += 1
            idx += 1
            continue
        if ch == ']':
            bracket_depth -= 1
            idx += 1
            continue
        if ch == '{':
            brace_depth += 1
            idx += 1
            continue
        if ch == '}':
            brace_depth -= 1
            idx += 1
            continue
        if depth == 0 and bracket_depth == 0 and brace_depth == 0 and expr[idx: idx + len(op)] == op:
            if op == '>':
                if idx > 0 and expr[idx - 1] == '-':
                    idx += 1
                    continue
            if op == '-':
                if idx + 1 < len(expr) and expr[idx + 1] == '>':
                    idx += 1
                    continue
                prev_idx = idx - 1
                while prev_idx >= 0 and expr[prev_idx].isspace():
                    prev_idx -= 1
                prev_char = expr[prev_idx] if prev_idx >= 0 else ''
                if prev_char in ('', '+', '-', '*', '/', '(', ',', '?', ':'):
                    idx += 1
                    continue
            last_match = idx
            idx += len(op)
            continue
        idx += 1
    if last_match is None:
        return None
    lhs = expr[:last_match].strip()
    rhs = expr[last_match + len(op):].strip()
    if lhs and rhs:
        return lhs, rhs
    return None



def _split_ternary(expr: str):
    depth = 0
    bracket_depth = 0
    brace_depth = 0
    in_string = False
    question = None
    colon = None
    for idx, ch in enumerate(expr):
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == '(':
            depth += 1
        elif ch == ')':
            depth -= 1
        elif ch == '[':
            bracket_depth += 1
        elif ch == ']':
            bracket_depth -= 1
        elif ch == '{':
            brace_depth += 1
        elif ch == '}':
            brace_depth -= 1
        elif ch == '?' and depth == 0 and bracket_depth == 0 and brace_depth == 0:
            question = idx
        elif ch == ':' and depth == 0 and bracket_depth == 0 and brace_depth == 0 and question is not None:
            if idx + 1 < len(expr) and expr[idx + 1] == ':':
                continue
            if idx > 0 and expr[idx - 1] == ':':
                continue
            colon = idx
            break
    if question is None or colon is None:
        return None
    return expr[:question].strip(), expr[question + 1:colon].strip(), expr[colon + 1:].strip()



def _raise_not_implemented(name):
    raise NotImplementedError(name)


__all__ = ['transpile', 'TranspileError']
