from .engine import grad
from .grad_mode import enable_grad, is_grad_enabled, set_grad_enabled


__all__ = ["vjp", "jvp", "jacobian", "hessian"]


def _stack(tensors, dim=0):
    if len(tensors) == 0:
        raise RuntimeError("stack expects a non-empty TensorList")
    from .._functional import stack as _stack_fn

    return _stack_fn(tensors, dim=dim)


def _zeros_like(tensor):
    from .._functional import zeros_like as _zeros_like_fn

    return _zeros_like_fn(tensor)


def _as_tuple_nocheck(x):
    if isinstance(x, tuple):
        return x
    if isinstance(x, list):
        return tuple(x)
    return (x,)


def _as_tuple(inp, arg_name=None, fn_name=None):
    if arg_name is None and fn_name is None:
        return _as_tuple_nocheck(inp)

    is_inp_tuple = isinstance(inp, tuple)
    if not is_inp_tuple:
        inp = (inp,)

    from .._tensor import Tensor

    for i, el in enumerate(inp):
        if not isinstance(el, Tensor):
            if is_inp_tuple:
                raise TypeError(
                    f"The {arg_name} given to {fn_name} must be either a Tensor or a tuple of Tensors but the"
                    f" value at index {i} has type {type(el)}."
                )
            raise TypeError(
                f"The {arg_name} given to {fn_name} must be either a Tensor or a tuple of Tensors but the"
                f" given {arg_name} has type {type(el)}."
            )

    return is_inp_tuple, inp


def _tuple_postprocess(res, to_unpack):
    if isinstance(to_unpack, tuple):
        if len(to_unpack) != 2:
            raise AssertionError("Expected to_unpack tuple to have exactly 2 elements")
        if not to_unpack[1]:
            res = tuple(el[0] for el in res)
        if not to_unpack[0]:
            res = res[0]
    elif not to_unpack:
        res = res[0]
    return res


def _grad_preprocess(inputs, create_graph, need_graph):
    res = []
    for inp in inputs:
        if create_graph and inp.requires_grad:
            if not inp.is_sparse:
                res.append(inp.view_as(inp))
            else:
                res.append(inp.clone())
        else:
            res.append(inp.detach().requires_grad_(need_graph))
    return tuple(res)


def _grad_postprocess(inputs, create_graph):
    from .._tensor import Tensor

    if isinstance(inputs[0], Tensor):
        if not create_graph:
            return tuple(inp.detach() for inp in inputs)
        return inputs
    return tuple(_grad_postprocess(inp, create_graph) for inp in inputs)


def _validate_v(v, other, is_other_tuple):
    if len(other) != len(v):
        if is_other_tuple:
            raise RuntimeError(
                f"v is a tuple of invalid length: should be {len(other)} but got {len(v)}."
            )
        raise RuntimeError("The given v should contain a single Tensor.")

    for idx, (el_v, el_other) in enumerate(zip(v, other)):
        if el_v.size() != el_other.size():
            prepend = f"Entry {idx} in " if is_other_tuple else ""
            raise RuntimeError(
                f"{prepend}v has invalid size: should be {el_other.size()} but got {el_v.size()}."
            )


def _check_requires_grad(inputs, input_type, strict):
    if not strict:
        return

    if input_type not in ["outputs", "grad_inputs", "jacobian", "hessian"]:
        raise RuntimeError("Invalid input_type to _check_requires_grad")
    for i, inp in enumerate(inputs):
        if inp is None:
            raise RuntimeError(
                f"The output of the user-provided function is independent of input {i}."
                " This is not allowed in strict mode."
            )
        if not inp.requires_grad:
            if input_type == "hessian":
                raise RuntimeError(
                    f"The hessian of the user-provided function with respect to input {i}"
                    " is independent of the input. This is not allowed in strict mode."
                    " You should ensure that your function is thrice differentiable and that"
                    " the hessian depends on the inputs."
                )
            if input_type == "jacobian":
                raise RuntimeError(
                    "While computing the hessian, found that the jacobian of the user-provided"
                    f" function with respect to input {i} is independent of the input. This is not"
                    " allowed in strict mode. You should ensure that your function is twice"
                    " differentiable and that the jacobian depends on the inputs (this would be"
                    " violated by a linear function for example)."
                )
            if input_type == "grad_inputs":
                raise RuntimeError(
                    f"The gradient with respect to input {i} is independent of the inputs of the"
                    " user-provided function. This is not allowed in strict mode."
                )
            raise RuntimeError(
                f"Output {i} of the user-provided function does not require gradients."
                " The outputs must be computed in a differentiable manner from the input"
                " when running in strict mode."
            )


def _autograd_grad(outputs, inputs, grad_outputs=None, create_graph=False, retain_graph=None):
    if not isinstance(outputs, tuple):
        raise AssertionError("Expected outputs to be a tuple")
    if grad_outputs is None:
        grad_outputs = (None,) * len(outputs)
    if not isinstance(grad_outputs, tuple):
        raise AssertionError("Expected grad_outputs to be a tuple")
    if len(outputs) != len(grad_outputs):
        raise AssertionError(
            f"Expected outputs and grad_outputs to have the same length, "
            f"but got {len(outputs)} and {len(grad_outputs)}"
        )

    new_outputs = ()
    new_grad_outputs = ()
    for out, grad_out in zip(outputs, grad_outputs):
        if out is not None and out.requires_grad:
            new_outputs += (out,)
            new_grad_outputs += (grad_out,)

    if len(new_outputs) == 0:
        return (None,) * len(inputs)
    grad_outputs_arg = None if all(grad_out is None for grad_out in new_grad_outputs) else new_grad_outputs
    return grad(
        new_outputs,
        inputs,
        grad_outputs_arg,
        allow_unused=True,
        create_graph=create_graph,
        retain_graph=retain_graph,
    )


def _fill_in_zeros(grads, refs, strict, create_graph, stage):
    if stage not in ["back", "back_trick", "double_back", "double_back_trick"]:
        raise RuntimeError(f"Invalid stage argument '{stage}' to _fill_in_zeros")

    res = ()
    for i, grads_i in enumerate(grads):
        if grads_i is None:
            if strict:
                if stage == "back":
                    raise RuntimeError(
                        "The output of the user-provided function is independent of "
                        f"input {i}. This is not allowed in strict mode."
                    )
                if stage == "back_trick":
                    raise RuntimeError(
                        f"The gradient with respect to the input is independent of entry {i}"
                        " in the grad_outputs when using the double backward trick to compute"
                        " forward mode gradients. This is not allowed in strict mode."
                    )
                if stage == "double_back":
                    raise RuntimeError(
                        "The jacobian of the user-provided function is independent of "
                        f"input {i}. This is not allowed in strict mode."
                    )
                raise RuntimeError(
                    "The hessian of the user-provided function is independent of "
                    f"entry {i} in the grad_jacobian. This is not allowed in strict "
                    "mode as it prevents from using the double backward trick to "
                    "replace forward mode AD."
                )
            grads_i = _zeros_like(refs[i])
        elif strict and create_graph and not grads_i.requires_grad:
            if "double" not in stage:
                raise RuntimeError(
                    "The jacobian of the user-provided function is independent of "
                    f"input {i}. This is not allowed in strict mode when create_graph=True."
                )
            raise RuntimeError(
                "The hessian of the user-provided function is independent of "
                f"input {i}. This is not allowed in strict mode when create_graph=True."
            )
        res += (grads_i,)
    return res


def _zeros_like_requires_grad(tensor):
    result = _zeros_like(tensor)
    result.requires_grad_(True)
    return result


def vjp(func, inputs, v=None, create_graph=False, strict=False):
    with enable_grad():
        is_inputs_tuple, inputs = _as_tuple(inputs, "inputs", "vjp")
        inputs = _grad_preprocess(inputs, create_graph=create_graph, need_graph=True)

        outputs = func(*inputs)
        is_outputs_tuple, outputs = _as_tuple(
            outputs, "outputs of the user-provided function", "vjp"
        )
        _check_requires_grad(outputs, "outputs", strict=strict)

        if v is not None:
            _, v = _as_tuple(v, "v", "vjp")
            v = _grad_preprocess(v, create_graph=create_graph, need_graph=False)
            _validate_v(v, outputs, is_outputs_tuple)
        elif len(outputs) != 1 or outputs[0].nelement() != 1:
            raise RuntimeError(
                "The vector v can only be None if the "
                "user-provided function returns "
                "a single Tensor with a single element."
            )

    enable_grad_flag = True if create_graph else is_grad_enabled()
    with set_grad_enabled(enable_grad_flag):
        grad_res = _autograd_grad(outputs, inputs, v, create_graph=create_graph)
        result = _fill_in_zeros(grad_res, inputs, strict, create_graph, "back")

    outputs = _grad_postprocess(outputs, create_graph)
    result = _grad_postprocess(result, create_graph)

    return _tuple_postprocess(outputs, is_outputs_tuple), _tuple_postprocess(
        result, is_inputs_tuple
    )


def jvp(func, inputs, v=None, create_graph=False, strict=False):
    with enable_grad():
        is_inputs_tuple, inputs = _as_tuple(inputs, "inputs", "jvp")
        inputs = _grad_preprocess(inputs, create_graph=create_graph, need_graph=True)

        if v is not None:
            _, v = _as_tuple(v, "v", "jvp")
            v = _grad_preprocess(v, create_graph=create_graph, need_graph=False)
            _validate_v(v, inputs, is_inputs_tuple)
        elif len(inputs) != 1 or inputs[0].nelement() != 1:
            raise RuntimeError(
                "The vector v can only be None if the input to "
                "the user-provided function is a single Tensor "
                "with a single element."
            )

        outputs = func(*inputs)
        is_outputs_tuple, outputs = _as_tuple(
            outputs, "outputs of the user-provided function", "jvp"
        )
        _check_requires_grad(outputs, "outputs", strict=strict)
        grad_outputs = tuple(_zeros_like_requires_grad(out) for out in outputs)
        grad_inputs = _autograd_grad(outputs, inputs, grad_outputs, create_graph=True)
        _check_requires_grad(grad_inputs, "grad_inputs", strict=strict)

    if create_graph:
        with enable_grad():
            grad_res = _autograd_grad(
                grad_inputs, grad_outputs, v, create_graph=create_graph
            )
            result = _fill_in_zeros(grad_res, outputs, strict, create_graph, "back_trick")
    else:
        grad_res = _autograd_grad(
            grad_inputs, grad_outputs, v, create_graph=create_graph
        )
        result = _fill_in_zeros(grad_res, outputs, strict, create_graph, "back_trick")

    outputs = _grad_postprocess(outputs, create_graph)
    result = _grad_postprocess(result, create_graph)

    return _tuple_postprocess(outputs, is_outputs_tuple), _tuple_postprocess(
        result, is_outputs_tuple
    )


def jacobian(
    func,
    inputs,
    create_graph=False,
    strict=False,
    vectorize=False,
    strategy="reverse-mode",
):
    if strategy not in ("forward-mode", "reverse-mode"):
        raise AssertionError(
            'Expected strategy to be either "forward-mode" or "reverse-mode". Hint: If your '
            'function has more outputs than inputs, "forward-mode" tends to be more performant. '
            'Otherwise, prefer to use "reverse-mode".'
        )
    if strategy == "forward-mode":
        if create_graph:
            raise NotImplementedError(
                "torch.autograd.functional.jacobian: `create_graph=True` "
                'and `strategy="forward-mode"` are not supported together (yet). '
                "Please either set `create_graph=False` or "
                '`strategy="reverse-mode"`.'
            )
        if strict:
            raise RuntimeError(
                "torch.autograd.functional.jacobian: `strict=True` "
                'and `strategy="forward-mode"` are not supported together (yet). '
                "Please either set `strict=False` or "
                '`strategy="reverse-mode"`.'
            )
        if not vectorize:
            raise NotImplementedError(
                "Computing Jacobian using forward-AD or forward-over-reverse Hessian is"
                "only implemented for `vectorize=True`."
            )
        vectorize = False
    if vectorize:
        if strict:
            raise RuntimeError(
                "torch.autograd.functional.jacobian: `strict=True` "
                "and `vectorized=True` are not supported together. "
                "Please either set `strict=False` or "
                "`vectorize=False`."
            )
        vectorize = False

    with enable_grad():
        is_inputs_tuple, inputs = _as_tuple(inputs, "inputs", "jacobian")
        inputs = _grad_preprocess(inputs, create_graph=create_graph, need_graph=True)

        outputs = func(*inputs)
        is_outputs_tuple, outputs = _as_tuple(
            outputs, "outputs of the user-provided function", "jacobian"
        )
        _check_requires_grad(outputs, "outputs", strict=strict)

        jacobian_result = ()
        for i, out in enumerate(outputs):
            jac_i = tuple([] for _ in range(len(inputs)))
            for j in range(out.nelement()):
                vj = _autograd_grad(
                    (out.reshape(-1)[j],),
                    inputs,
                    retain_graph=True,
                    create_graph=create_graph,
                )

                for el_idx, (jac_i_el, vj_el, inp_el) in enumerate(
                    zip(jac_i, vj, inputs)
                ):
                    if vj_el is not None:
                        if strict and create_graph and not vj_el.requires_grad:
                            raise RuntimeError(
                                "The jacobian of the user-provided function is "
                                f"independent of input {i}. This is not allowed in "
                                "strict mode when create_graph=True."
                            )
                        jac_i_el.append(vj_el)
                    else:
                        if strict:
                            raise RuntimeError(
                                f"Output {i} of the user-provided function is "
                                f"independent of input {el_idx}. This is not allowed in "
                                "strict mode."
                            )
                        jac_i_el.append(_zeros_like(inp_el))

            jacobian_result += (
                tuple(
                    _stack(jac_i_el, dim=0).view(out.size() + inputs[el_idx].size())
                    for el_idx, jac_i_el in enumerate(jac_i)
                ),
            )

        jacobian_result = _grad_postprocess(jacobian_result, create_graph)

        return _tuple_postprocess(jacobian_result, (is_outputs_tuple, is_inputs_tuple))


def hessian(
    func,
    inputs,
    create_graph=False,
    strict=False,
    vectorize=False,
    outer_jacobian_strategy="reverse-mode",
):
    is_inputs_tuple, inputs = _as_tuple(inputs, "inputs", "hessian")
    if outer_jacobian_strategy not in ("forward-mode", "reverse-mode"):
        raise AssertionError('Expected strategy to be either "forward-mode" or "reverse-mode".')

    def ensure_single_output_function(*inp):
        out = func(*inp)
        is_out_tuple, t_out = _as_tuple(
            out, "outputs of the user-provided function", "hessian"
        )
        _check_requires_grad(t_out, "outputs", strict=strict)

        from .._tensor import Tensor

        if is_out_tuple or not isinstance(out, Tensor):
            raise RuntimeError("The function given to hessian should return a single Tensor")
        if out.nelement() != 1:
            raise RuntimeError(
                "The Tensor returned by the function given to hessian should contain a single element"
            )
        return out.squeeze()

    def jac_func(*inp):
        if outer_jacobian_strategy == "forward-mode":
            inp = tuple(t.requires_grad_(True) for t in inp)
        jac = jacobian(ensure_single_output_function, inp, create_graph=True)
        _check_requires_grad(jac, "jacobian", strict=strict)
        return jac

    result = jacobian(
        jac_func,
        inputs,
        create_graph=create_graph,
        strict=strict,
        vectorize=vectorize,
        strategy=outer_jacobian_strategy,
    )
    return _tuple_postprocess(result, (is_inputs_tuple, is_inputs_tuple))
