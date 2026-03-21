from tools.autograd.formula_transpiler import transpile


def test_transpile_division_with_parenthesized_conj_expression():
    formula = 'grad / (2 * self.conj())'
    expected = 'redispatch("div", keyset, grad, redispatch("mul", keyset, 2, self))'
    assert transpile(formula) == expected


def test_transpile_ternary_with_nested_method_chains():
    formula = 'self.is_complex() ? grad * ((self + 1).rsqrt() * (self - 1).rsqrt()).conj() : grad * (self * self - 1).rsqrt()'
    result = transpile(formula)
    assert 'if redispatch("is_complex", keyset, self) else' in result
    assert 'redispatch("mul", keyset, grad, redispatch("mul", keyset, redispatch("rsqrt", keyset, redispatch("add", keyset, self, 1)), redispatch("rsqrt", keyset, redispatch("sub", keyset, self, 1))))' in result
    assert 'redispatch("mul", keyset, grad, redispatch("rsqrt", keyset, redispatch("sub", keyset, redispatch("mul", keyset, self, self), 1)))' in result


def test_transpile_at_where_with_scalar_tensor_argument():
    formula = 'where(self >= min, grad, at::scalar_tensor(0., grad.options()))'
    expected = 'redispatch("where", keyset, redispatch("ge", keyset, self, min), grad, redispatch("scalar_tensor", keyset, 0., self.options if False else grad.options))'
    result = transpile(formula)
    assert 'redispatch("where"' in result
    assert 'redispatch("scalar_tensor"' in result


def test_transpile_namespace_call_followed_by_method_chain():
    formula = 'at::where(self == other, grad / 2, grad).masked_fill_(self < other, 0)'
    expected = 'redispatch("masked_fill_", keyset, redispatch("where", keyset, redispatch("eq", keyset, self, other), redispatch("div", keyset, grad, 2), grad), redispatch("lt", keyset, self, other), 0)'
    assert transpile(formula) == expected


def test_transpile_std_sqrt_call():
    formula = 'grad * std::sqrt(2 * M_PI) * (result.square() / 2).exp()'
    expected = 'redispatch("mul", keyset, redispatch("mul", keyset, grad, redispatch("sqrt", keyset, redispatch("mul", keyset, 2, M_PI))), redispatch("exp", keyset, redispatch("div", keyset, redispatch("square", keyset, result), 2)))'
    assert transpile(formula) == expected


def test_transpile_enum_constant_passthrough():
    formula = 'apply_loss_reduction(binary_cross_entropy_backward(self_t, self_p, target_p, weight, at::Reduction::None), reduction)'
    result = transpile(formula)
    assert 'redispatch("apply_loss_reduction"' in result
    assert 'at::Reduction::None' in result


def test_transpile_std_vector_constructor():
    formula = 'std::vector<c10::SymInt>(padding.size(), 0)'
    expected = '[0] * redispatch("size", keyset, padding)'
    assert transpile(formula) == expected


def test_transpile_brace_init_bool_mask():
    formula = '{true, false, false}'
    expected = '[True, False, False]'
    assert transpile(formula) == expected


def test_transpile_generic_std_tuple_empty_constructor():
    formula = 'std::tuple<Tensor, Tensor, Tensor>()'
    expected = '(None, None, None)'
    assert transpile(formula) == expected


def test_transpile_scalar_toDouble_ternary():
    formula = 'other.toDouble() > 0. ? at::xlogy(grad,  other) : at::xlogy(grad,  other).masked_fill(self == 0., 0.)'
    result = transpile(formula)
    assert 'float(other)' in result
    assert 'redispatch("xlogy"' in result
    assert 'redispatch("masked_fill"' in result


def test_transpile_prelu_kernel_backward_uses_helper_fallback():
    formula = 'grads[0].defined() ? (grads[1].defined() ? at::where(self >= 0, grads[0], grads[0] * weight + grads[1] * self) : at::where(self >= 0, grads[0], grads[0] * weight)) : at::where(self >= 0, at::zeros({}, grad_output.options()), grads[1] * self)'
    assert transpile(formula) == '_prelu_kernel_backward_grad_output_helper(grads, self, weight, grad_output)'


def test_transpile_native_group_norm_uses_helper_fallback():
    formula = 'GradMode::is_enabled() || grads[1].defined() || grads[2].defined() ? infinitely_differentiable_native_group_norm_backward(grads[0], grads[1], grads[2], input, result1, result2, weight, N, C, HxW, group, eps, grad_input_mask) : (grads[0].defined() ? native_group_norm_backward_symint(grads[0].device().is_xpu() ? grads[0] : grads[0].contiguous(grads[0].device().is_cpu() ? input.suggest_memory_format() : c10::MemoryFormat::Contiguous), input.device().is_xpu() ? input : input.contiguous(input.device().is_cpu() ? input.suggest_memory_format() : c10::MemoryFormat::Contiguous), result1, result2, weight, N, C, HxW, group, grad_input_mask) : std::tuple<Tensor, Tensor, Tensor>())'
    assert transpile(formula) == '_native_group_norm_helper(grads, input, result1, result2, weight, N, C, HxW, group, eps, grad_input_mask)'


def test_transpile_roll_uses_helper_fallback():
    formula = 'grad.roll_symint(fmap(reverse_list_symint(shifts), [](c10::SymInt i){return -i;}), reverse_list(dims))'
    assert transpile(formula) == '_roll_backward_helper(grad, shifts, dims)'


def test_transpile_convolution_backward_jvp_uses_helper_fallback():
    formula = 'std::get<0>(convolution_backward_symint(grad_output_p, input_p, weight_t, bias_sizes, stride, padding, dilation, transposed, output_padding, groups, {true, false, false})) + std::get<0>(convolution_backward_symint(grad_output_t, input_p, weight_p, bias_sizes, stride, padding, dilation, transposed, output_padding, groups, {true, false, false}))'
    assert transpile(formula) == '_convolution_backward_jvp_input_helper(grad_output_p, grad_output_t, input_p, weight_p, weight_t, bias_sizes, stride, padding, dilation, transposed, output_padding, groups)'


def test_transpile_to_padded_tensor_uses_helper_fallback():
    formula = 'self.layout() == c10::kJagged ? at::_nested_from_padded_tensor_symint(grad, at::_nested_get_offsets(self), at::_nested_get_jagged_dummy(self), at::_nested_get_ragged_idx(self), at::_nested_get_min_seqlen(self).defined() ? std::optional<Tensor>(at::_nested_get_min_seqlen(self)) : ::std::nullopt, at::_nested_get_max_seqlen(self).defined() ? std::optional<Tensor>(at::_nested_get_max_seqlen(self)) : ::std::nullopt, std::optional<c10::SymInt>(at::_nested_get_values(self).sym_size(0))) : at::_nested_from_padded(grad, self._nested_tensor_size())'
    assert transpile(formula) == '_to_padded_tensor_backward_helper(grad, self)'


def test_transpile_unary_minus_after_multiplication():
    formula = 'grad * -((-self * self + 1).rsqrt()).conj()'
    expected = 'redispatch("mul", keyset, grad, redispatch("neg", keyset, redispatch("rsqrt", keyset, redispatch("add", keyset, redispatch("mul", keyset, redispatch("neg", keyset, self), self), 1))))'
    assert transpile(formula) == expected


def test_transpile_unary_minus_before_method_chain():
    formula = 'grad * -self.sin().conj()'
    expected = 'redispatch("mul", keyset, grad, redispatch("neg", keyset, redispatch("sin", keyset, self)))'
    assert transpile(formula) == expected


def test_transpile_namespace_call_minus_value():
    formula = 'grad * (at::special_i1e(self) - self.sgn() * result)'
    expected = 'redispatch("mul", keyset, grad, redispatch("sub", keyset, redispatch("special_i1e", keyset, self), redispatch("mul", keyset, redispatch("sign", keyset, self), result)))'
    assert transpile(formula) == expected


def test_transpile_native_dropout_nested_ternary_condition():
    formula = '!train.has_value() || !train.value() ? 1 : (p == 1 ? 0.0 : 1.0 / (1.0 - p))'
    expected = '(1 if ((not redispatch("has_value", keyset, train)) or (not redispatch("value", keyset, train))) else (0.0 if redispatch("eq", keyset, p, 1) else redispatch("div", keyset, 1.0, redispatch("sub", keyset, 1.0, p))))'
    assert transpile(formula) == expected


def test_transpile_pointer_style_method_call():
    formula = 'bias->sym_sizes()'
    expected = 'bias.shape'
    assert transpile(formula) == expected


def test_transpile_brace_shape_list_with_method_calls():
    formula = 'maybe_multiply(grad.unsqueeze(0).expand_symint({ batch1.sym_size(0), batch1.sym_size(1), batch2.sym_size(2) }).bmm(batch2.transpose(1, 2).conj()), alpha.conj())'
    result = transpile(formula)
    assert '[batch1.shape[0], batch1.shape[1], batch2.shape[2]]' in result
    assert '{ batch1, 0)' not in result


def test_transpile_scalar_type_method_to_dtype_attr():
    formula = 'self.scalar_type()'
    expected = 'self.dtype'
    assert transpile(formula) == expected


def test_transpile_options_method_to_options_attr():
    formula = 'self.options()'
    expected = 'self.options'
    assert transpile(formula) == expected


def test_transpile_defined_method_to_is_not_none():
    formula = 'grad.defined()'
    expected = '(grad is not None)'
    assert transpile(formula) == expected


def test_transpile_sym_size_method_to_shape_index():
    formula = 'batch1.sym_size(0)'
    expected = 'batch1.shape[0]'
    assert transpile(formula) == expected


def test_transpile_sym_sizes_method_to_shape_attr():
    formula = 'self.sym_sizes()'
    expected = 'self.shape'
    assert transpile(formula) == expected


def test_transpile_toFloat_method_to_float_call():
    formula = 'alpha.toFloat()'
    expected = 'float(alpha)'
    assert transpile(formula) == expected


def test_transpile_sgn_method_to_sign_dispatch():
    formula = 'self.sgn()'
    expected = 'redispatch("sign", keyset, self)'
    assert transpile(formula) == expected


def test_transpile_expand_symint_method_to_expand_op():
    formula = 'grad.expand_symint(self.sym_sizes())'
    expected = 'redispatch("expand", keyset, grad, self.shape)'
    assert transpile(formula) == expected


def test_transpile_sum_to_function_to_local_helper():
    formula = 'sum_to(grad, self.sym_sizes())'
    expected = '_sum_to_backward_helper(grad, self.shape, keyset)'
    assert transpile(formula) == expected
