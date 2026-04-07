# Global Residual Inventory

## Tensor(storage) hits
```
Binary file src/candle/_cython/_storage.cpython-312-aarch64-linux-gnu.so matches
src/candle/_cython/_storage.c:3403:PyDoc_STRVAR(__pyx_doc_6candle_7_cython_8_storage_2cy_make_npu_tensor, "Construct an NPU Tensor entirely in Cython via the unified tensor factory.\n\n    Equivalent to::\n\n        storage = npu_typed_storage_from_ptr(device_ptr, n_elements, dtype, device)\n        return Tensor(storage, shape, stride)\n\n    Routes through cy_make_tensor_from_storage so all tensor births share a\n    single initialisation path.\n    ");
src/candle/_cython/_storage.pyx:103:        return Tensor(storage, shape, stride)

```

## _Tensor(storage) hits
```

```

## Tensor.__new__(Tensor) hits
```
src/candle/_cython/_tensor_impl.c:15069: *     cdef TensorImpl t = Tensor.__new__(Tensor)
src/candle/_cython/_tensor_impl.c:15097: *     cdef TensorImpl t = Tensor.__new__(Tensor)             # <<<<<<<<<<<<<<
src/candle/_cython/_tensor_impl.c:15117: *     cdef TensorImpl t = Tensor.__new__(Tensor)
src/candle/_cython/_tensor_impl.c:15421: *     cdef TensorImpl t = Tensor.__new__(Tensor)
src/candle/_cython/_tensor_impl.c:15432: *     cdef TensorImpl t = Tensor.__new__(Tensor)
src/candle/_cython/_tensor_impl.c:15449: *     cdef TensorImpl t = Tensor.__new__(Tensor)             # <<<<<<<<<<<<<<
src/candle/_cython/_tensor_impl.c:15489: *     cdef TensorImpl t = Tensor.__new__(Tensor)
src/candle/_cython/_tensor_impl.pyx:535:    cdef TensorImpl t = Tensor.__new__(Tensor)
src/candle/_cython/_tensor_impl.pyx:568:    cdef TensorImpl t = Tensor.__new__(Tensor)

```

## _dtype_code assignments
```
src/candle/_backends/npu/aclnn.py:10049:    self_dtype_code = _dtype_to_acl(self_dtype)
src/candle/_backends/npu/aclnn.py:10050:    mask_dtype_code = _dtype_to_acl(mask_dtype)
src/candle/_backends/npu/aclnn.py:10051:    source_dtype_code = _dtype_to_acl(source_dtype)
src/candle/_backends/npu/aclnn.py:10174:    self_dtype_code = _dtype_to_acl(dtype)
src/candle/_backends/npu/aclnn.py:10175:    out_dtype_code = _dtype_to_acl(out_dtype)
src/candle/_backends/npu/aclnn.py:10596:    mask_dtype_code = _dtype_to_acl("int8")
src/candle/_backends/npu/aclnn.py:10906:    stats_dtype_code = _dtype_to_acl("float32")
src/candle/_backends/npu/aclnn.py:11066:    stats_dtype_code = _dtype_to_acl("float32")
src/candle/_backends/npu/aclnn.py:11151:    grad_dtype_code = _dtype_to_acl(grad_dtype)
src/candle/_backends/npu/aclnn.py:11152:    indices_dtype_code = _dtype_to_acl(indices_dtype)
src/candle/_backends/npu/aclnn.py:11225:    mask_dtype_code = _dtype_to_acl("int8")
src/candle/_backends/npu/aclnn.py:11521:    out_dtype_code = _dtype_to_acl(out_dtype)
src/candle/_backends/npu/aclnn.py:11680:    self_dtype_code = _dtype_to_acl(self_dtype)
src/candle/_backends/npu/aclnn.py:11681:    on_dtype_code = _dtype_to_acl(on_dtype)
src/candle/_backends/npu/aclnn.py:11682:    off_dtype_code = _dtype_to_acl(off_dtype)
src/candle/_backends/npu/aclnn.py:11683:    out_dtype_code = _dtype_to_acl(out_dtype)
src/candle/_backends/npu/aclnn.py:11910:    indices_dtype_code = _dtype_to_acl("int64")
src/candle/_backends/npu/aclnn.py:11967:    indices_dtype_code = _dtype_to_acl("int64")
src/candle/_backends/npu/aclnn.py:12023:    input_dtype_code = _dtype_to_acl(dtype)
src/candle/_backends/npu/aclnn.py:12025:    out_dtype_code = _dtype_to_acl(out_dtype)
src/candle/_backends/npu/aclnn.py:12082:    self_dtype_code = _dtype_to_acl(dtype)
src/candle/_backends/npu/aclnn.py:12083:    inverse_dtype_code = _dtype_to_acl("int64")
src/candle/_backends/npu/aclnn.py:12136:    out_dtype_code = _dtype_to_acl(dtype)
src/candle/_backends/npu/aclnn.py:13287:    stats_dtype_code = _dtype_to_acl(dtype)
src/candle/_backends/npu/aclnn.py:15119:    stats_dtype_code = _dtype_to_acl("float32")
src/candle/_backends/npu/aclnn.py:15325:    self_dtype_code = _dtype_to_acl(self_dtype)
src/candle/_backends/npu/aclnn.py:15326:    weights_dtype_code = self_dtype_code if weights_dtype is None else _dtype_to_acl(weights_dtype)
src/candle/_backends/npu/aclnn.py:15327:    out_dtype_code = _dtype_to_acl(out_dtype)
src/candle/_backends/npu/aclnn.py:15730:    idx_dtype_code = _dtype_to_acl("int32")
src/candle/_backends/npu/aclnn.py:15805:    idx_dtype_code = _dtype_to_acl("int32")
src/candle/_backends/npu/aclnn.py:15885:    idx_dtype_code = _dtype_to_acl("int64")
src/candle/_backends/npu/aclnn.py:15955:    idx_dtype_code = _dtype_to_acl("int64")
src/candle/_backends/npu/aclnn.py:16021:    grad_dtype_code = _dtype_to_acl(grad_dtype)
src/candle/_backends/npu/aclnn.py:16022:    out_dtype_code = _dtype_to_acl(out_dtype)
src/candle/_backends/npu/aclnn.py:16023:    repeats_dtype_code = _dtype_to_acl("int64")
src/candle/_backends/npu/aclnn.py:5474:    self_dtype_code = _dtype_to_acl(self_dtype)
src/candle/_backends/npu/aclnn.py:5475:    out_dtype_code = _dtype_to_acl(out_dtype)
src/candle/_backends/npu/aclnn.py:5520:    self_dtype_code = _dtype_to_acl(self_dtype)
src/candle/_backends/npu/aclnn.py:5521:    out_dtype_code = _dtype_to_acl(out_dtype)
src/candle/_backends/npu/aclnn.py:5982:    repeats_dtype_code = _dtype_to_acl(repeats_dtype)
src/candle/_backends/npu/aclnn.py:6294:    self_dtype_code = _dtype_to_acl(self_dtype)
src/candle/_backends/npu/aclnn.py:6295:    values_dtype_code = _dtype_to_acl(values_dtype)
src/candle/_backends/npu/aclnn.py:8138:    self_dtype_code = _dtype_to_acl(self_dtype)
src/candle/_backends/npu/aclnn.py:8139:    out_dtype_code = _dtype_to_acl(out_dtype)
src/candle/_backends/npu/aclnn.py:8723:    self_dtype_code = _dtype_to_acl(self_dtype)
src/candle/_backends/npu/aclnn.py:8724:    out_dtype_code = _dtype_to_acl(out_dtype)
src/candle/_backends/npu/aclnn.py:9092:    stats_dtype_code = _dtype_to_acl("float32")
src/candle/_backends/npu/aclnn.py:9181:    stats_dtype_code = _dtype_to_acl("float32")
src/candle/_backends/npu/aclnn.py:9294:    mask_dtype_code = _dtype_to_acl("uint8")
src/candle/_backends/npu/aclnn.py:9340:    mask_dtype_code = _dtype_to_acl("uint8")
src/candle/_backends/npu/aclnn.py:9533:    dst_dtype_code = _dtype_to_acl(dst_dtype)
src/candle/_backends/npu/aclnn.py:9534:    src_dtype_code = _dtype_to_acl(src_dtype)
src/candle/_backends/npu/aclnn.py:9754:    self_dtype_code = _dtype_to_acl(self_dtype)
src/candle/_backends/npu/aclnn.py:9755:    mask_dtype_code = _dtype_to_acl(mask_dtype)
src/candle/_backends/npu/aclnn.py:9809:    self_dtype_code = _dtype_to_acl(self_dtype)
src/candle/_backends/npu/aclnn.py:9810:    index_dtype_code = _dtype_to_acl(index_dtype)
src/candle/_backends/npu/aclnn.py:9811:    source_dtype_code = _dtype_to_acl(source_dtype)
src/candle/_backends/npu/aclnn.py:9864:    self_dtype_code = _dtype_to_acl(self_dtype)
src/candle/_backends/npu/aclnn.py:9865:    index_dtype_code = _dtype_to_acl(index_dtype)
src/candle/_backends/npu/aclnn.py:9921:    self_dtype_code = _dtype_to_acl(self_dtype)
src/candle/_backends/npu/aclnn.py:9922:    index_dtype_code = _dtype_to_acl(index_dtype)
src/candle/_backends/npu/aclnn.py:9923:    source_dtype_code = _dtype_to_acl(source_dtype)
src/candle/_backends/npu/aclnn.py:9924:    out_dtype_code = _dtype_to_acl(out_dtype)
src/candle/_backends/npu/aclnn.py:9988:    self_dtype_code = _dtype_to_acl(self_dtype)
src/candle/_backends/npu/aclnn.py:9989:    index_dtype_code = _dtype_to_acl(index_dtype)
src/candle/_backends/npu/aclnn.py:9990:    src_dtype_code = _dtype_to_acl(src_dtype)
src/candle/_backends/npu/aclnn.py:9991:    out_dtype_code = _dtype_to_acl(out_dtype)
src/candle/_cython/_aclnn_ffi.c:100502:    __pyx_v_self_dtype_code = __Pyx_PyLong_As_int32_t(values[12]); if (unlikely((__pyx_v_self_dtype_code == ((int32_t)-1)) && PyErr_Occurred())) __PYX_ERR(0, 5778, __pyx_L3_error)
src/candle/_cython/_aclnn_ffi.c:100503:    __pyx_v_index_dtype_code = __Pyx_PyLong_As_int32_t(values[13]); if (unlikely((__pyx_v_index_dtype_code == ((int32_t)-1)) && PyErr_Occurred())) __PYX_ERR(0, 5778, __pyx_L3_error)
src/candle/_cython/_aclnn_ffi.c:100504:    __pyx_v_src_dtype_code = __Pyx_PyLong_As_int32_t(values[14]); if (unlikely((__pyx_v_src_dtype_code == ((int32_t)-1)) && PyErr_Occurred())) __PYX_ERR(0, 5778, __pyx_L3_error)
src/candle/_cython/_aclnn_ffi.c:100505:    __pyx_v_out_dtype_code = __Pyx_PyLong_As_int32_t(values[15]); if (unlikely((__pyx_v_out_dtype_code == ((int32_t)-1)) && PyErr_Occurred())) __PYX_ERR(0, 5778, __pyx_L3_error)
src/candle/_cython/_aclnn_ffi.c:102191:    __pyx_v_self_dtype_code = __Pyx_PyLong_As_int32_t(values[10]); if (unlikely((__pyx_v_self_dtype_code == ((int32_t)-1)) && PyErr_Occurred())) __PYX_ERR(0, 5871, __pyx_L3_error)
src/candle/_cython/_aclnn_ffi.c:102192:    __pyx_v_out_dtype_code = __Pyx_PyLong_As_int32_t(values[11]); if (unlikely((__pyx_v_out_dtype_code == ((int32_t)-1)) && PyErr_Occurred())) __PYX_ERR(0, 5871, __pyx_L3_error)
src/candle/_cython/_aclnn_ffi.c:103466:    __pyx_v_self_dtype_code = __Pyx_PyLong_As_int32_t(values[9]); if (unlikely((__pyx_v_self_dtype_code == ((int32_t)-1)) && PyErr_Occurred())) __PYX_ERR(0, 5939, __pyx_L3_error)
src/candle/_cython/_aclnn_ffi.c:103467:    __pyx_v_q_dtype_code = __Pyx_PyLong_As_int32_t(values[10]); if (unlikely((__pyx_v_q_dtype_code == ((int32_t)-1)) && PyErr_Occurred())) __PYX_ERR(0, 5939, __pyx_L3_error)
src/candle/_cython/_aclnn_ffi.c:103468:    __pyx_v_r_dtype_code = __Pyx_PyLong_As_int32_t(values[11]); if (unlikely((__pyx_v_r_dtype_code == ((int32_t)-1)) && PyErr_Occurred())) __PYX_ERR(0, 5939, __pyx_L3_error)
src/candle/_cython/_aclnn_ffi.c:104731:    __pyx_v_self_dtype_code = __Pyx_PyLong_As_int32_t(values[12]); if (unlikely((__pyx_v_self_dtype_code == ((int32_t)-1)) && PyErr_Occurred())) __PYX_ERR(0, 6010, __pyx_L3_error)
src/candle/_cython/_aclnn_ffi.c:104732:    __pyx_v_b_dtype_code = __Pyx_PyLong_As_int32_t(values[13]); if (unlikely((__pyx_v_b_dtype_code == ((int32_t)-1)) && PyErr_Occurred())) __PYX_ERR(0, 6010, __pyx_L3_error)
src/candle/_cython/_aclnn_ffi.c:104733:    __pyx_v_c_dtype_code = __Pyx_PyLong_As_int32_t(values[14]); if (unlikely((__pyx_v_c_dtype_code == ((int32_t)-1)) && PyErr_Occurred())) __PYX_ERR(0, 6010, __pyx_L3_error)
src/candle/_cython/_aclnn_ffi.c:104734:    __pyx_v_out_dtype_code = __Pyx_PyLong_As_int32_t(values[15]); if (unlikely((__pyx_v_out_dtype_code == ((int32_t)-1)) && PyErr_Occurred())) __PYX_ERR(0, 6010, __pyx_L3_error)
src/candle/_cython/_aclnn_ffi.c:106100:    __pyx_v_self_dtype_code = __Pyx_PyLong_As_int32_t(values[11]); if (unlikely((__pyx_v_self_dtype_code == ((int32_t)-1)) && PyErr_Occurred())) __PYX_ERR(0, 6090, __pyx_L3_error)
src/candle/_cython/_aclnn_ffi.c:106101:    __pyx_v_values_dtype_code = __Pyx_PyLong_As_int32_t(values[12]); if (unlikely((__pyx_v_values_dtype_code == ((int32_t)-1)) && PyErr_Occurred())) __PYX_ERR(0, 6090, __pyx_L3_error)
src/candle/_cython/_aclnn_ffi.c:106102:    __pyx_v_indices_dtype_code = __Pyx_PyLong_As_int32_t(values[13]); if (unlikely((__pyx_v_indices_dtype_code == ((int32_t)-1)) && PyErr_Occurred())) __PYX_ERR(0, 6090, __pyx_L3_error)
src/candle/_cython/_aclnn_ffi.c:107629:    __pyx_v_a_dtype_code = __Pyx_PyLong_As_int32_t(values[11]); if (unlikely((__pyx_v_a_dtype_code == ((int32_t)-1)) && PyErr_Occurred())) __PYX_ERR(0, 6173, __pyx_L3_error)
src/candle/_cython/_aclnn_ffi.c:107630:    __pyx_v_b_dtype_code = __Pyx_PyLong_As_int32_t(values[12]); if (unlikely((__pyx_v_b_dtype_code == ((int32_t)-1)) && PyErr_Occurred())) __PYX_ERR(0, 6173, __pyx_L3_error)
src/candle/_cython/_aclnn_ffi.c:107631:    __pyx_v_c_dtype_code = __Pyx_PyLong_As_int32_t(values[13]); if (unlikely((__pyx_v_c_dtype_code == ((int32_t)-1)) && PyErr_Occurred())) __PYX_ERR(0, 6173, __pyx_L3_error)
src/candle/_cython/_aclnn_ffi.c:107632:    __pyx_v_out_dtype_code = __Pyx_PyLong_As_int32_t(values[14]); if (unlikely((__pyx_v_out_dtype_code == ((int32_t)-1)) && PyErr_Occurred())) __PYX_ERR(0, 6173, __pyx_L3_error)
src/candle/_cython/_aclnn_ffi.c:108984:    __pyx_v_self_dtype_code = __Pyx_PyLong_As_int32_t(values[9]); if (unlikely((__pyx_v_self_dtype_code == ((int32_t)-1)) && PyErr_Occurred())) __PYX_ERR(0, 6254, __pyx_L3_error)
src/candle/_cython/_aclnn_ffi.c:108985:    __pyx_v_out_a_dtype_code = __Pyx_PyLong_As_int32_t(values[10]); if (unlikely((__pyx_v_out_a_dtype_code == ((int32_t)-1)) && PyErr_Occurred())) __PYX_ERR(0, 6254, __pyx_L3_error)
src/candle/_cython/_aclnn_ffi.c:108986:    __pyx_v_out_b_dtype_code = __Pyx_PyLong_As_int32_t(values[11]); if (unlikely((__pyx_v_out_b_dtype_code == ((int32_t)-1)) && PyErr_Occurred())) __PYX_ERR(0, 6254, __pyx_L3_error)
src/candle/_cython/_aclnn_ffi.c:110695:    __pyx_v_self_dtype_code = __Pyx_PyLong_As_int32_t(values[10]); if (unlikely((__pyx_v_self_dtype_code == ((int32_t)-1)) && PyErr_Occurred())) __PYX_ERR(0, 6346, __pyx_L3_error)
src/candle/_cython/_aclnn_ffi.c:110696:    __pyx_v_out_a_dtype_code = __Pyx_PyLong_As_int32_t(values[11]); if (unlikely((__pyx_v_out_a_dtype_code == ((int32_t)-1)) && PyErr_Occurred())) __PYX_ERR(0, 6346, __pyx_L3_error)
src/candle/_cython/_aclnn_ffi.c:110697:    __pyx_v_out_b_dtype_code = __Pyx_PyLong_As_int32_t(values[12]); if (unlikely((__pyx_v_out_b_dtype_code == ((int32_t)-1)) && PyErr_Occurred())) __PYX_ERR(0, 6346, __pyx_L3_error)
src/candle/_cython/_aclnn_ffi.c:112420:    __pyx_v_a_dtype_code = __Pyx_PyLong_As_int32_t(values[10]); if (unlikely((__pyx_v_a_dtype_code == ((int32_t)-1)) && PyErr_Occurred())) __PYX_ERR(0, 6438, __pyx_L3_error)
src/candle/_cython/_aclnn_ffi.c:112421:    __pyx_v_b_dtype_code = __Pyx_PyLong_As_int32_t(values[11]); if (unlikely((__pyx_v_b_dtype_code == ((int32_t)-1)) && PyErr_Occurred())) __PYX_ERR(0, 6438, __pyx_L3_error)
src/candle/_cython/_aclnn_ffi.c:112422:    __pyx_v_c_dtype_code = __Pyx_PyLong_As_int32_t(values[12]); if (unlikely((__pyx_v_c_dtype_code == ((int32_t)-1)) && PyErr_Occurred())) __PYX_ERR(0, 6438, __pyx_L3_error)
src/candle/_cython/_aclnn_ffi.c:112423:    __pyx_v_out_dtype_code = __Pyx_PyLong_As_int32_t(values[13]); if (unlikely((__pyx_v_out_dtype_code == ((int32_t)-1)) && PyErr_Occurred())) __PYX_ERR(0, 6438, __pyx_L3_error)
src/candle/_cython/_aclnn_ffi.c:114163:    __pyx_v_a_dtype_code = __Pyx_PyLong_As_int32_t(values[15]); if (unlikely((__pyx_v_a_dtype_code == ((int32_t)-1)) && PyErr_Occurred())) __PYX_ERR(0, 6531, __pyx_L3_error)
src/candle/_cython/_aclnn_ffi.c:114164:    __pyx_v_b_dtype_code = __Pyx_PyLong_As_int32_t(values[16]); if (unlikely((__pyx_v_b_dtype_code == ((int32_t)-1)) && PyErr_Occurred())) __PYX_ERR(0, 6531, __pyx_L3_error)
src/candle/_cython/_aclnn_ffi.c:114165:    __pyx_v_out_dtype_code = __Pyx_PyLong_As_int32_t(values[17]); if (unlikely((__pyx_v_out_dtype_code == ((int32_t)-1)) && PyErr_Occurred())) __PYX_ERR(0, 6531, __pyx_L3_error)
src/candle/_cython/_aclnn_ffi.c:117007:    __pyx_v_a_dtype_code = __Pyx_PyLong_As_int32_t(values[15]); if (unlikely((__pyx_v_a_dtype_code == ((int32_t)-1)) && PyErr_Occurred())) __PYX_ERR(0, 6683, __pyx_L3_error)
src/candle/_cython/_aclnn_ffi.c:117008:    __pyx_v_b_dtype_code = __Pyx_PyLong_As_int32_t(values[16]); if (unlikely((__pyx_v_b_dtype_code == ((int32_t)-1)) && PyErr_Occurred())) __PYX_ERR(0, 6683, __pyx_L3_error)
src/candle/_cython/_aclnn_ffi.c:117009:    __pyx_v_c_dtype_code = __Pyx_PyLong_As_int32_t(values[17]); if (unlikely((__pyx_v_c_dtype_code == ((int32_t)-1)) && PyErr_Occurred())) __PYX_ERR(0, 6683, __pyx_L3_error)
src/candle/_cython/_aclnn_ffi.c:117010:    __pyx_v_out_dtype_code = __Pyx_PyLong_As_int32_t(values[18]); if (unlikely((__pyx_v_out_dtype_code == ((int32_t)-1)) && PyErr_Occurred())) __PYX_ERR(0, 6683, __pyx_L3_error)
src/candle/_cython/_aclnn_ffi.c:120467:    __pyx_v_self_dtype_code = __Pyx_PyLong_As_int32_t(values[10]); if (unlikely((__pyx_v_self_dtype_code == ((int32_t)-1)) && PyErr_Occurred())) __PYX_ERR(0, 6873, __pyx_L3_error)
src/candle/_cython/_aclnn_ffi.c:120468:    __pyx_v_out_dtype_code = __Pyx_PyLong_As_int32_t(values[11]); if (unlikely((__pyx_v_out_dtype_code == ((int32_t)-1)) && PyErr_Occurred())) __PYX_ERR(0, 6873, __pyx_L3_error)
src/candle/_cython/_aclnn_ffi.c:121968:    __pyx_v_first_dtype_code = __Pyx_PyLong_As_int32_t(values[10]); if (unlikely((__pyx_v_first_dtype_code == ((int32_t)-1)) && PyErr_Occurred())) __PYX_ERR(0, 6954, __pyx_L3_error)
src/candle/_cython/_aclnn_ffi.c:121969:    __pyx_v_second_dtype_code = __Pyx_PyLong_As_int32_t(values[11]); if (unlikely((__pyx_v_second_dtype_code == ((int32_t)-1)) && PyErr_Occurred())) __PYX_ERR(0, 6954, __pyx_L3_error)
src/candle/_cython/_aclnn_ffi.c:121970:    __pyx_v_out_dtype_code = __Pyx_PyLong_As_int32_t(values[12]); if (unlikely((__pyx_v_out_dtype_code == ((int32_t)-1)) && PyErr_Occurred())) __PYX_ERR(0, 6954, __pyx_L3_error)
src/candle/_cython/_aclnn_ffi.c:123455:    __pyx_v_self_dtype_code = __Pyx_PyLong_As_int32_t(values[11]); if (unlikely((__pyx_v_self_dtype_code == ((int32_t)-1)) && PyErr_Occurred())) __PYX_ERR(0, 7035, __pyx_L3_error)
src/candle/_cython/_aclnn_ffi.c:123456:    __pyx_v_out_dtype_code = __Pyx_PyLong_As_int32_t(values[12]); if (unlikely((__pyx_v_out_dtype_code == ((int32_t)-1)) && PyErr_Occurred())) __PYX_ERR(0, 7035, __pyx_L3_error)
src/candle/_cython/_aclnn_ffi.c:12436:    __pyx_v_dtype_code = __Pyx_PyLong_As_int32_t(values[8]); if (unlikely((__pyx_v_dtype_code == ((int32_t)-1)) && PyErr_Occurred())) __PYX_ERR(0, 770, __pyx_L3_error)
src/candle/_cython/_aclnn_ffi.c:125385:    __pyx_v_self_dtype_code = __Pyx_PyLong_As_int32_t(values[10]); if (unlikely((__pyx_v_self_dtype_code == ((int32_t)-1)) && PyErr_Occurred())) __PYX_ERR(0, 7138, __pyx_L3_error)
src/candle/_cython/_aclnn_ffi.c:125386:    __pyx_v_out_dtype_code = __Pyx_PyLong_As_int32_t(values[11]); if (unlikely((__pyx_v_out_dtype_code == ((int32_t)-1)) && PyErr_Occurred())) __PYX_ERR(0, 7138, __pyx_L3_error)
src/candle/_cython/_aclnn_ffi.c:127327:    __pyx_v_a_dtype_code = __Pyx_PyLong_As_int32_t(values[8]); if (unlikely((__pyx_v_a_dtype_code == ((int32_t)-1)) && PyErr_Occurred())) __PYX_ERR(0, 7240, __pyx_L3_error)
src/candle/_cython/_aclnn_ffi.c:127328:    __pyx_v_b_dtype_code = __Pyx_PyLong_As_int32_t(values[9]); if (unlikely((__pyx_v_b_dtype_code == ((int32_t)-1)) && PyErr_Occurred())) __PYX_ERR(0, 7240, __pyx_L3_error)
src/candle/_cython/_aclnn_ffi.c:127329:    __pyx_v_out_dtype_code = __Pyx_PyLong_As_int32_t(values[10]); if (unlikely((__pyx_v_out_dtype_code == ((int32_t)-1)) && PyErr_Occurred())) __PYX_ERR(0, 7240, __pyx_L3_error)
src/candle/_cython/_aclnn_ffi.c:128820:    __pyx_v_a_dtype_code = __Pyx_PyLong_As_int32_t(values[8]); if (unlikely((__pyx_v_a_dtype_code == ((int32_t)-1)) && PyErr_Occurred())) __PYX_ERR(0, 7319, __pyx_L3_error)
src/candle/_cython/_aclnn_ffi.c:128821:    __pyx_v_b_dtype_code = __Pyx_PyLong_As_int32_t(values[9]); if (unlikely((__pyx_v_b_dtype_code == ((int32_t)-1)) && PyErr_Occurred())) __PYX_ERR(0, 7319, __pyx_L3_error)
src/candle/_cython/_aclnn_ffi.c:128822:    __pyx_v_out_dtype_code = __Pyx_PyLong_As_int32_t(values[10]); if (unlikely((__pyx_v_out_dtype_code == ((int32_t)-1)) && PyErr_Occurred())) __PYX_ERR(0, 7319, __pyx_L3_error)
src/candle/_cython/_aclnn_ffi.c:130327:    __pyx_v_a_dtype_code = __Pyx_PyLong_As_int32_t(values[8]); if (unlikely((__pyx_v_a_dtype_code == ((int32_t)-1)) && PyErr_Occurred())) __PYX_ERR(0, 7398, __pyx_L3_error)
src/candle/_cython/_aclnn_ffi.c:130328:    __pyx_v_b_dtype_code = __Pyx_PyLong_As_int32_t(values[9]); if (unlikely((__pyx_v_b_dtype_code == ((int32_t)-1)) && PyErr_Occurred())) __PYX_ERR(0, 7398, __pyx_L3_error)
src/candle/_cython/_aclnn_ffi.c:130329:    __pyx_v_out_dtype_code = __Pyx_PyLong_As_int32_t(values[10]); if (unlikely((__pyx_v_out_dtype_code == ((int32_t)-1)) && PyErr_Occurred())) __PYX_ERR(0, 7398, __pyx_L3_error)
src/candle/_cython/_aclnn_ffi.c:131907:    __pyx_v_tensor_dtype_code = __Pyx_PyLong_As_int32_t(values[16]); if (unlikely((__pyx_v_tensor_dtype_code == ((int32_t)-1)) && PyErr_Occurred())) __PYX_ERR(0, 7482, __pyx_L3_error)
src/candle/_cython/_aclnn_ffi.c:131908:    __pyx_v_stats_dtype_code = __Pyx_PyLong_As_int32_t(values[17]); if (unlikely((__pyx_v_stats_dtype_code == ((int32_t)-1)) && PyErr_Occurred())) __PYX_ERR(0, 7482, __pyx_L3_error)
src/candle/_cython/_aclnn_ffi.c:134158:    __pyx_v_tensor_dtype_code = __Pyx_PyLong_As_int32_t(values[14]); if (unlikely((__pyx_v_tensor_dtype_code == ((int32_t)-1)) && PyErr_Occurred())) __PYX_ERR(0, 7632, __pyx_L3_error)
src/candle/_cython/_aclnn_ffi.c:135882:    __pyx_v_tensor_dtype_code = __Pyx_PyLong_As_int32_t(values[25]); if (unlikely((__pyx_v_tensor_dtype_code == ((int32_t)-1)) && PyErr_Occurred())) __PYX_ERR(0, 7734, __pyx_L3_error)
src/candle/_cython/_aclnn_ffi.c:135883:    __pyx_v_stats_dtype_code = __Pyx_PyLong_As_int32_t(values[26]); if (unlikely((__pyx_v_stats_dtype_code == ((int32_t)-1)) && PyErr_Occurred())) __PYX_ERR(0, 7734, __pyx_L3_error)
src/candle/_cython/_aclnn_ffi.c:13755:    __pyx_v_dtype_code = __Pyx_PyLong_As_int32_t(values[8]); if (unlikely((__pyx_v_dtype_code == ((int32_t)-1)) && PyErr_Occurred())) __PYX_ERR(0, 867, __pyx_L3_error)
src/candle/_cython/_aclnn_ffi.c:138506:    __pyx_v_tensor_dtype_code = __Pyx_PyLong_As_int32_t(values[23]); if (unlikely((__pyx_v_tensor_dtype_code == ((int32_t)-1)) && PyErr_Occurred())) __PYX_ERR(0, 7897, __pyx_L3_error)
src/candle/_cython/_aclnn_ffi.c:138507:    __pyx_v_stats_dtype_code = __Pyx_PyLong_As_int32_t(values[24]); if (unlikely((__pyx_v_stats_dtype_code == ((int32_t)-1)) && PyErr_Occurred())) __PYX_ERR(0, 7897, __pyx_L3_error)
src/candle/_cython/_aclnn_ffi.c:140731:    __pyx_v_tensor_dtype_code = __Pyx_PyLong_As_int32_t(values[23]); if (unlikely((__pyx_v_tensor_dtype_code == ((int32_t)-1)) && PyErr_Occurred())) __PYX_ERR(0, 8034, __pyx_L3_error)
src/candle/_cython/_aclnn_ffi.c:143298:    __pyx_v_tensor_dtype_code = __Pyx_PyLong_As_int32_t(values[16]); if (unlikely((__pyx_v_tensor_dtype_code == ((int32_t)-1)) && PyErr_Occurred())) __PYX_ERR(0, 8211, __pyx_L3_error)
src/candle/_cython/_aclnn_ffi.c:144905:    __pyx_v_a_dtype_code = __Pyx_PyLong_As_int32_t(values[12]); if (unlikely((__pyx_v_a_dtype_code == ((int32_t)-1)) && PyErr_Occurred())) __PYX_ERR(0, 8305, __pyx_L3_error)
src/candle/_cython/_aclnn_ffi.c:144906:    __pyx_v_b_dtype_code = __Pyx_PyLong_As_int32_t(values[13]); if (unlikely((__pyx_v_b_dtype_code == ((int32_t)-1)) && PyErr_Occurred())) __PYX_ERR(0, 8305, __pyx_L3_error)
src/candle/_cython/_aclnn_ffi.c:144907:    __pyx_v_c_dtype_code = __Pyx_PyLong_As_int32_t(values[14]); if (unlikely((__pyx_v_c_dtype_code == ((int32_t)-1)) && PyErr_Occurred())) __PYX_ERR(0, 8305, __pyx_L3_error)
src/candle/_cython/_aclnn_ffi.c:144908:    __pyx_v_out_a_dtype_code = __Pyx_PyLong_As_int32_t(values[15]); if (unlikely((__pyx_v_out_a_dtype_code == ((int32_t)-1)) && PyErr_Occurred())) __PYX_ERR(0, 8306, __pyx_L3_error)
src/candle/_cython/_aclnn_ffi.c:144909:    __pyx_v_out_b_dtype_code = __Pyx_PyLong_As_int32_t(values[16]); if (unlikely((__pyx_v_out_b_dtype_code == ((int32_t)-1)) && PyErr_Occurred())) __PYX_ERR(0, 8306, __pyx_L3_error)
src/candle/_cython/_aclnn_ffi.c:146809:    __pyx_v_a_dtype_code = __Pyx_PyLong_As_int32_t(values[8]); if (unlikely((__pyx_v_a_dtype_code == ((int32_t)-1)) && PyErr_Occurred())) __PYX_ERR(0, 8411, __pyx_L3_error)
src/candle/_cython/_aclnn_ffi.c:146810:    __pyx_v_b_dtype_code = __Pyx_PyLong_As_int32_t(values[9]); if (unlikely((__pyx_v_b_dtype_code == ((int32_t)-1)) && PyErr_Occurred())) __PYX_ERR(0, 8411, __pyx_L3_error)
src/candle/_cython/_aclnn_ffi.c:146811:    __pyx_v_out_dtype_code = __Pyx_PyLong_As_int32_t(values[10]); if (unlikely((__pyx_v_out_dtype_code == ((int32_t)-1)) && PyErr_Occurred())) __PYX_ERR(0, 8411, __pyx_L3_error)
src/candle/_cython/_aclnn_ffi.c:148295:    __pyx_v_a_dtype_code = __Pyx_PyLong_As_int32_t(values[9]); if (unlikely((__pyx_v_a_dtype_code == ((int32_t)-1)) && PyErr_Occurred())) __PYX_ERR(0, 8491, __pyx_L3_error)
src/candle/_cython/_aclnn_ffi.c:148296:    __pyx_v_b_dtype_code = __Pyx_PyLong_As_int32_t(values[10]); if (unlikely((__pyx_v_b_dtype_code == ((int32_t)-1)) && PyErr_Occurred())) __PYX_ERR(0, 8491, __pyx_L3_error)
src/candle/_cython/_aclnn_ffi.c:148297:    __pyx_v_out_dtype_code = __Pyx_PyLong_As_int32_t(values[11]); if (unlikely((__pyx_v_out_dtype_code == ((int32_t)-1)) && PyErr_Occurred())) __PYX_ERR(0, 8491, __pyx_L3_error)
src/candle/_cython/_aclnn_ffi.c:149809:    __pyx_v_a_dtype_code = __Pyx_PyLong_As_int32_t(values[10]); if (unlikely((__pyx_v_a_dtype_code == ((int32_t)-1)) && PyErr_Occurred())) __PYX_ERR(0, 8570, __pyx_L3_error)
src/candle/_cython/_aclnn_ffi.c:149810:    __pyx_v_b_dtype_code = __Pyx_PyLong_As_int32_t(values[11]); if (unlikely((__pyx_v_b_dtype_code == ((int32_t)-1)) && PyErr_Occurred())) __PYX_ERR(0, 8570, __pyx_L3_error)
src/candle/_cython/_aclnn_ffi.c:149811:    __pyx_v_c_dtype_code = __Pyx_PyLong_As_int32_t(values[12]); if (unlikely((__pyx_v_c_dtype_code == ((int32_t)-1)) && PyErr_Occurred())) __PYX_ERR(0, 8570, __pyx_L3_error)
src/candle/_cython/_aclnn_ffi.c:149812:    __pyx_v_out_dtype_code = __Pyx_PyLong_As_int32_t(values[13]); if (unlikely((__pyx_v_out_dtype_code == ((int32_t)-1)) && PyErr_Occurred())) __PYX_ERR(0, 8570, __pyx_L3_error)
src/candle/_cython/_aclnn_ffi.c:15064:    __pyx_v_dtype_code = __Pyx_PyLong_As_int32_t(values[6]); if (unlikely((__pyx_v_dtype_code == ((int32_t)-1)) && PyErr_Occurred())) __PYX_ERR(0, 961, __pyx_L3_error)
src/candle/_cython/_aclnn_ffi.c:151617:    __pyx_v_a_dtype_code = __Pyx_PyLong_As_int32_t(values[16]); if (unlikely((__pyx_v_a_dtype_code == ((int32_t)-1)) && PyErr_Occurred())) __PYX_ERR(0, 8666, __pyx_L3_error)
src/candle/_cython/_aclnn_ffi.c:151618:    __pyx_v_b_dtype_code = __Pyx_PyLong_As_int32_t(values[17]); if (unlikely((__pyx_v_b_dtype_code == ((int32_t)-1)) && PyErr_Occurred())) __PYX_ERR(0, 8666, __pyx_L3_error)
src/candle/_cython/_aclnn_ffi.c:151619:    __pyx_v_c_dtype_code = __Pyx_PyLong_As_int32_t(values[18]); if (unlikely((__pyx_v_c_dtype_code == ((int32_t)-1)) && PyErr_Occurred())) __PYX_ERR(0, 8666, __pyx_L3_error)
src/candle/_cython/_aclnn_ffi.c:151620:    __pyx_v_out_dtype_code = __Pyx_PyLong_As_int32_t(values[19]); if (unlikely((__pyx_v_out_dtype_code == ((int32_t)-1)) && PyErr_Occurred())) __PYX_ERR(0, 8667, __pyx_L3_error)
src/candle/_cython/_aclnn_ffi.c:151621:    __pyx_v_stats_a_dtype_code = __Pyx_PyLong_As_int32_t(values[20]); if (unlikely((__pyx_v_stats_a_dtype_code == ((int32_t)-1)) && PyErr_Occurred())) __PYX_ERR(0, 8667, __pyx_L3_error)
src/candle/_cython/_aclnn_ffi.c:151622:    __pyx_v_stats_b_dtype_code = __Pyx_PyLong_As_int32_t(values[21]); if (unlikely((__pyx_v_stats_b_dtype_code == ((int32_t)-1)) && PyErr_Occurred())) __PYX_ERR(0, 8667, __pyx_L3_error)
src/candle/_cython/_aclnn_ffi.c:153292:    __pyx_v_a_dtype_code = __Pyx_PyLong_As_int32_t(values[21]); if (unlikely((__pyx_v_a_dtype_code == ((int32_t)-1)) && PyErr_Occurred())) __PYX_ERR(0, 8766, __pyx_L3_error)
src/candle/_cython/_aclnn_ffi.c:153293:    __pyx_v_b_dtype_code = __Pyx_PyLong_As_int32_t(values[22]); if (unlikely((__pyx_v_b_dtype_code == ((int32_t)-1)) && PyErr_Occurred())) __PYX_ERR(0, 8766, __pyx_L3_error)
src/candle/_cython/_aclnn_ffi.c:153294:    __pyx_v_c_dtype_code = __Pyx_PyLong_As_int32_t(values[23]); if (unlikely((__pyx_v_c_dtype_code == ((int32_t)-1)) && PyErr_Occurred())) __PYX_ERR(0, 8766, __pyx_L3_error)
src/candle/_cython/_aclnn_ffi.c:153295:    __pyx_v_d_dtype_code = __Pyx_PyLong_As_int32_t(values[24]); if (unlikely((__pyx_v_d_dtype_code == ((int32_t)-1)) && PyErr_Occurred())) __PYX_ERR(0, 8767, __pyx_L3_error)
src/candle/_cython/_aclnn_ffi.c:153296:    __pyx_v_e_dtype_code = __Pyx_PyLong_As_int32_t(values[25]); if (unlikely((__pyx_v_e_dtype_code == ((int32_t)-1)) && PyErr_Occurred())) __PYX_ERR(0, 8767, __pyx_L3_error)
src/candle/_cython/_aclnn_ffi.c:153297:    __pyx_v_f_dtype_code = __Pyx_PyLong_As_int32_t(values[26]); if (unlikely((__pyx_v_f_dtype_code == ((int32_t)-1)) && PyErr_Occurred())) __PYX_ERR(0, 8767, __pyx_L3_error)
src/candle/_cython/_aclnn_ffi.c:154874:    __pyx_v_cond_dtype_code = __Pyx_PyLong_As_int32_t(values[10]); if (unlikely((__pyx_v_cond_dtype_code == ((int32_t)-1)) && PyErr_Occurred())) __PYX_ERR(0, 8862, __pyx_L3_error)
src/candle/_cython/_aclnn_ffi.c:154875:    __pyx_v_self_dtype_code = __Pyx_PyLong_As_int32_t(values[11]); if (unlikely((__pyx_v_self_dtype_code == ((int32_t)-1)) && PyErr_Occurred())) __PYX_ERR(0, 8862, __pyx_L3_error)
src/candle/_cython/_aclnn_ffi.c:154876:    __pyx_v_other_dtype_code = __Pyx_PyLong_As_int32_t(values[12]); if (unlikely((__pyx_v_other_dtype_code == ((int32_t)-1)) && PyErr_Occurred())) __PYX_ERR(0, 8862, __pyx_L3_error)
src/candle/_cython/_aclnn_ffi.c:154877:    __pyx_v_out_dtype_code = __Pyx_PyLong_As_int32_t(values[13]); if (unlikely((__pyx_v_out_dtype_code == ((int32_t)-1)) && PyErr_Occurred())) __PYX_ERR(0, 8862, __pyx_L3_error)
src/candle/_cython/_aclnn_ffi.c:156196:    __pyx_v_dtype_code = __Pyx_PyLong_As_int32_t(values[6]); if (unlikely((__pyx_v_dtype_code == ((int32_t)-1)) && PyErr_Occurred())) __PYX_ERR(0, 8939, __pyx_L3_error)
src/candle/_cython/_aclnn_ffi.c:157299:    __pyx_v_dtype_code = __Pyx_PyLong_As_int32_t(values[10]); if (unlikely((__pyx_v_dtype_code == ((int32_t)-1)) && PyErr_Occurred())) __PYX_ERR(0, 8999, __pyx_L3_error)
src/candle/_cython/_aclnn_ffi.c:158625:    __pyx_v_self_dtype_code = __Pyx_PyLong_As_int32_t(values[6]); if (unlikely((__pyx_v_self_dtype_code == ((int32_t)-1)) && PyErr_Occurred())) __PYX_ERR(0, 9076, __pyx_L3_error)
src/candle/_cython/_aclnn_ffi.c:158626:    __pyx_v_out_dtype_code = __Pyx_PyLong_As_int32_t(values[7]); if (unlikely((__pyx_v_out_dtype_code == ((int32_t)-1)) && PyErr_Occurred())) __PYX_ERR(0, 9076, __pyx_L3_error)
src/candle/_cython/_aclnn_ffi.c:159728:    __pyx_v_self_dtype_code = __Pyx_PyLong_As_int32_t(values[9]); if (unlikely((__pyx_v_self_dtype_code == ((int32_t)-1)) && PyErr_Occurred())) __PYX_ERR(0, 9136, __pyx_L3_error)
src/candle/_cython/_aclnn_ffi.c:159729:    __pyx_v_other_dtype_code = __Pyx_PyLong_As_int32_t(values[10]); if (unlikely((__pyx_v_other_dtype_code == ((int32_t)-1)) && PyErr_Occurred())) __PYX_ERR(0, 9136, __pyx_L3_error)
src/candle/_cython/_aclnn_ffi.c:159730:    __pyx_v_out_dtype_code = __Pyx_PyLong_As_int32_t(values[11]); if (unlikely((__pyx_v_out_dtype_code == ((int32_t)-1)) && PyErr_Occurred())) __PYX_ERR(0, 9136, __pyx_L3_error)
src/candle/_cython/_aclnn_ffi.c:161213:    __pyx_v_self_dtype_code = __Pyx_PyLong_As_int32_t(values[9]); if (unlikely((__pyx_v_self_dtype_code == ((int32_t)-1)) && PyErr_Occurred())) __PYX_ERR(0, 9217, __pyx_L3_error)
src/candle/_cython/_aclnn_ffi.c:161214:    __pyx_v_other_dtype_code = __Pyx_PyLong_As_int32_t(values[10]); if (unlikely((__pyx_v_other_dtype_code == ((int32_t)-1)) && PyErr_Occurred())) __PYX_ERR(0, 9217, __pyx_L3_error)
src/candle/_cython/_aclnn_ffi.c:161215:    __pyx_v_out_dtype_code = __Pyx_PyLong_As_int32_t(values[11]); if (unlikely((__pyx_v_out_dtype_code == ((int32_t)-1)) && PyErr_Occurred())) __PYX_ERR(0, 9217, __pyx_L3_error)
src/candle/_cython/_aclnn_ffi.c:162470:    __pyx_v_first_dtype_code = __Pyx_PyLong_As_int32_t(values[11]); if (unlikely((__pyx_v_first_dtype_code == ((int32_t)-1)) && PyErr_Occurred())) __PYX_ERR(0, 9286, __pyx_L3_error)
src/candle/_cython/_aclnn_ffi.c:162471:    __pyx_v_second_dtype_code = __Pyx_PyLong_As_int32_t(values[12]); if (unlikely((__pyx_v_second_dtype_code == ((int32_t)-1)) && PyErr_Occurred())) __PYX_ERR(0, 9286, __pyx_L3_error)
src/candle/_cython/_aclnn_ffi.c:162472:    __pyx_v_out_dtype_code = __Pyx_PyLong_As_int32_t(values[13]); if (unlikely((__pyx_v_out_dtype_code == ((int32_t)-1)) && PyErr_Occurred())) __PYX_ERR(0, 9286, __pyx_L3_error)
src/candle/_cython/_aclnn_ffi.c:16253:    __pyx_v_dtype_code = __Pyx_PyLong_As_int32_t(values[6]); if (unlikely((__pyx_v_dtype_code == ((int32_t)-1)) && PyErr_Occurred())) __PYX_ERR(0, 1040, __pyx_L3_error)
src/candle/_cython/_aclnn_ffi.c:163715:    __pyx_v_self_dtype_code = __Pyx_PyLong_As_int32_t(values[11]); if (unlikely((__pyx_v_self_dtype_code == ((int32_t)-1)) && PyErr_Occurred())) __PYX_ERR(0, 9356, __pyx_L3_error)
src/candle/_cython/_aclnn_ffi.c:163716:    __pyx_v_other_dtype_code = __Pyx_PyLong_As_int32_t(values[12]); if (unlikely((__pyx_v_other_dtype_code == ((int32_t)-1)) && PyErr_Occurred())) __PYX_ERR(0, 9356, __pyx_L3_error)
src/candle/_cython/_aclnn_ffi.c:163717:    __pyx_v_out_dtype_code = __Pyx_PyLong_As_int32_t(values[13]); if (unlikely((__pyx_v_out_dtype_code == ((int32_t)-1)) && PyErr_Occurred())) __PYX_ERR(0, 9356, __pyx_L3_error)
src/candle/_cython/_aclnn_ffi.c:164934:    __pyx_v_self_dtype_code = __Pyx_PyLong_As_int32_t(values[8]); if (unlikely((__pyx_v_self_dtype_code == ((int32_t)-1)) && PyErr_Occurred())) __PYX_ERR(0, 9424, __pyx_L3_error)
src/candle/_cython/_aclnn_ffi.c:164935:    __pyx_v_other_dtype_code = __Pyx_PyLong_As_int32_t(values[9]); if (unlikely((__pyx_v_other_dtype_code == ((int32_t)-1)) && PyErr_Occurred())) __PYX_ERR(0, 9424, __pyx_L3_error)
src/candle/_cython/_aclnn_ffi.c:164936:    __pyx_v_out_dtype_code = __Pyx_PyLong_As_int32_t(values[10]); if (unlikely((__pyx_v_out_dtype_code == ((int32_t)-1)) && PyErr_Occurred())) __PYX_ERR(0, 9424, __pyx_L3_error)
src/candle/_cython/_aclnn_ffi.c:166381:    __pyx_v_dtype_code = __Pyx_PyLong_As_int32_t(values[6]); if (unlikely((__pyx_v_dtype_code == ((int32_t)-1)) && PyErr_Occurred())) __PYX_ERR(0, 9503, __pyx_L3_error)
src/candle/_cython/_aclnn_ffi.c:167491:    __pyx_v_dtype_code = __Pyx_PyLong_As_int32_t(values[11]); if (unlikely((__pyx_v_dtype_code == ((int32_t)-1)) && PyErr_Occurred())) __PYX_ERR(0, 9564, __pyx_L3_error)
src/candle/_cython/_aclnn_ffi.c:17441:    __pyx_v_self_dtype_code = __Pyx_PyLong_As_int32_t(values[6]); if (unlikely((__pyx_v_self_dtype_code == ((int32_t)-1)) && PyErr_Occurred())) __PYX_ERR(0, 1119, __pyx_L3_error)
src/candle/_cython/_aclnn_ffi.c:17442:    __pyx_v_out_dtype_code = __Pyx_PyLong_As_int32_t(values[7]); if (unlikely((__pyx_v_out_dtype_code == ((int32_t)-1)) && PyErr_Occurred())) __PYX_ERR(0, 1119, __pyx_L3_error)
src/candle/_cython/_aclnn_ffi.c:18628:    __pyx_v_self_dtype_code = __Pyx_PyLong_As_int32_t(values[6]); if (unlikely((__pyx_v_self_dtype_code == ((int32_t)-1)) && PyErr_Occurred())) __PYX_ERR(0, 1194, __pyx_L3_error)
src/candle/_cython/_aclnn_ffi.c:18629:    __pyx_v_out_dtype_code = __Pyx_PyLong_As_int32_t(values[7]); if (unlikely((__pyx_v_out_dtype_code == ((int32_t)-1)) && PyErr_Occurred())) __PYX_ERR(0, 1194, __pyx_L3_error)
src/candle/_cython/_aclnn_ffi.c:19875:    __pyx_v_dtype_code = __Pyx_PyLong_As_int32_t(values[8]); if (unlikely((__pyx_v_dtype_code == ((int32_t)-1)) && PyErr_Occurred())) __PYX_ERR(0, 1261, __pyx_L3_error)
src/candle/_cython/_aclnn_ffi.c:21279:    __pyx_v_self_dtype_code = __Pyx_PyLong_As_int32_t(values[8]); if (unlikely((__pyx_v_self_dtype_code == ((int32_t)-1)) && PyErr_Occurred())) __PYX_ERR(0, 1351, __pyx_L3_error)
src/candle/_cython/_aclnn_ffi.c:21280:    __pyx_v_out_dtype_code = __Pyx_PyLong_As_int32_t(values[9]); if (unlikely((__pyx_v_out_dtype_code == ((int32_t)-1)) && PyErr_Occurred())) __PYX_ERR(0, 1351, __pyx_L3_error)
src/candle/_cython/_aclnn_ffi.c:22696:    __pyx_v_self_dtype_code = __Pyx_PyLong_As_int32_t(values[8]); if (unlikely((__pyx_v_self_dtype_code == ((int32_t)-1)) && PyErr_Occurred())) __PYX_ERR(0, 1428, __pyx_L3_error)
src/candle/_cython/_aclnn_ffi.c:23882:    __pyx_v_src_dtype_code = __Pyx_PyLong_As_int32_t(values[6]); if (unlikely((__pyx_v_src_dtype_code == ((int32_t)-1)) && PyErr_Occurred())) __PYX_ERR(0, 1504, __pyx_L3_error)
src/candle/_cython/_aclnn_ffi.c:23883:    __pyx_v_dst_dtype_code = __Pyx_PyLong_As_int32_t(values[7]); if (unlikely((__pyx_v_dst_dtype_code == ((int32_t)-1)) && PyErr_Occurred())) __PYX_ERR(0, 1504, __pyx_L3_error)
src/candle/_cython/_aclnn_ffi.c:25096:    __pyx_v_self_dtype_code = __Pyx_PyLong_As_int32_t(values[8]); if (unlikely((__pyx_v_self_dtype_code == ((int32_t)-1)) && PyErr_Occurred())) __PYX_ERR(0, 1568, __pyx_L3_error)
src/candle/_cython/_aclnn_ffi.c:26363:    __pyx_v_self_dtype_code = __Pyx_PyLong_As_int32_t(values[13]); if (unlikely((__pyx_v_self_dtype_code == ((int32_t)-1)) && PyErr_Occurred())) __PYX_ERR(0, 1634, __pyx_L3_error)
src/candle/_cython/_aclnn_ffi.c:26364:    __pyx_v_values_dtype_code = __Pyx_PyLong_As_int32_t(values[14]); if (unlikely((__pyx_v_values_dtype_code == ((int32_t)-1)) && PyErr_Occurred())) __PYX_ERR(0, 1634, __pyx_L3_error)
src/candle/_cython/_aclnn_ffi.c:28060:    __pyx_v_self_dtype_code = __Pyx_PyLong_As_int32_t(values[6]); if (unlikely((__pyx_v_self_dtype_code == ((int32_t)-1)) && PyErr_Occurred())) __PYX_ERR(0, 1729, __pyx_L3_error)
src/candle/_cython/_aclnn_ffi.c:28061:    __pyx_v_out_dtype_code = __Pyx_PyLong_As_int32_t(values[7]); if (unlikely((__pyx_v_out_dtype_code == ((int32_t)-1)) && PyErr_Occurred())) __PYX_ERR(0, 1729, __pyx_L3_error)
src/candle/_cython/_aclnn_ffi.c:29281:    __pyx_v_self_dtype_code = __Pyx_PyLong_As_int32_t(values[8]); if (unlikely((__pyx_v_self_dtype_code == ((int32_t)-1)) && PyErr_Occurred())) __PYX_ERR(0, 1795, __pyx_L3_error)
src/candle/_cython/_aclnn_ffi.c:29282:    __pyx_v_out_dtype_code = __Pyx_PyLong_As_int32_t(values[9]); if (unlikely((__pyx_v_out_dtype_code == ((int32_t)-1)) && PyErr_Occurred())) __PYX_ERR(0, 1795, __pyx_L3_error)
src/candle/_cython/_aclnn_ffi.c:30494:    __pyx_v_out_dtype_code = __Pyx_PyLong_As_int32_t(values[7]); if (unlikely((__pyx_v_out_dtype_code == ((int32_t)-1)) && PyErr_Occurred())) __PYX_ERR(0, 1859, __pyx_L3_error)
src/candle/_cython/_aclnn_ffi.c:30495:    __pyx_v_self_dtype_code = __Pyx_PyLong_As_int32_t(values[8]); if (unlikely((__pyx_v_self_dtype_code == ((int32_t)-1)) && PyErr_Occurred())) __PYX_ERR(0, 1859, __pyx_L3_error)
src/candle/_cython/_aclnn_ffi.c:31699:    __pyx_v_self_dtype_code = __Pyx_PyLong_As_int32_t(values[6]); if (unlikely((__pyx_v_self_dtype_code == ((int32_t)-1)) && PyErr_Occurred())) __PYX_ERR(0, 1924, __pyx_L3_error)
src/candle/_cython/_aclnn_ffi.c:31799: *     cdef int32_t bool_dtype_code = 12             # <<<<<<<<<<<<<<
src/candle/_cython/_aclnn_ffi.c:31803:  __pyx_v_bool_dtype_code = 12;
src/candle/_cython/_aclnn_ffi.c:31807: *     cdef int32_t bool_dtype_code = 12
src/candle/_cython/_aclnn_ffi.c:31818: *     cdef int32_t bool_dtype_code = 12
src/candle/_cython/_aclnn_ffi.c:32949:    __pyx_v_out_dtype_code = __Pyx_PyLong_As_int32_t(values[6]); if (unlikely((__pyx_v_out_dtype_code == ((int32_t)-1)) && PyErr_Occurred())) __PYX_ERR(0, 2000, __pyx_L3_error)
src/candle/_cython/_aclnn_ffi.c:32950:    __pyx_v_self_dtype_code = __Pyx_PyLong_As_int32_t(values[7]); if (unlikely((__pyx_v_self_dtype_code == ((int32_t)-1)) && PyErr_Occurred())) __PYX_ERR(0, 2000, __pyx_L3_error)
src/candle/_cython/_aclnn_ffi.c:34156:    __pyx_v_dtype_code = __Pyx_PyLong_As_int32_t(values[7]); if (unlikely((__pyx_v_dtype_code == ((int32_t)-1)) && PyErr_Occurred())) __PYX_ERR(0, 2066, __pyx_L3_error)
src/candle/_cython/_aclnn_ffi.c:35445:    __pyx_v_dtype_code = __Pyx_PyLong_As_int32_t(values[14]); if (unlikely((__pyx_v_dtype_code == ((int32_t)-1)) && PyErr_Occurred())) __PYX_ERR(0, 2134, __pyx_L3_error)
src/candle/_cython/_aclnn_ffi.c:37289:    __pyx_v_self_dtype_code = __Pyx_PyLong_As_int32_t(values[7]); if (unlikely((__pyx_v_self_dtype_code == ((int32_t)-1)) && PyErr_Occurred())) __PYX_ERR(0, 2255, __pyx_L3_error)
src/candle/_cython/_aclnn_ffi.c:37290:    __pyx_v_out_dtype_code = __Pyx_PyLong_As_int32_t(values[8]); if (unlikely((__pyx_v_out_dtype_code == ((int32_t)-1)) && PyErr_Occurred())) __PYX_ERR(0, 2255, __pyx_L3_error)
src/candle/_cython/_aclnn_ffi.c:38552:    __pyx_v_self_dtype_code = __Pyx_PyLong_As_int32_t(values[9]); if (unlikely((__pyx_v_self_dtype_code == ((int32_t)-1)) && PyErr_Occurred())) __PYX_ERR(0, 2324, __pyx_L3_error)
src/candle/_cython/_aclnn_ffi.c:38553:    __pyx_v_out_dtype_code = __Pyx_PyLong_As_int32_t(values[10]); if (unlikely((__pyx_v_out_dtype_code == ((int32_t)-1)) && PyErr_Occurred())) __PYX_ERR(0, 2324, __pyx_L3_error)
src/candle/_cython/_aclnn_ffi.c:40039:    __pyx_v_self_dtype_code = __Pyx_PyLong_As_int32_t(values[10]); if (unlikely((__pyx_v_self_dtype_code == ((int32_t)-1)) && PyErr_Occurred())) __PYX_ERR(0, 2406, __pyx_L3_error)
src/candle/_cython/_aclnn_ffi.c:40040:    __pyx_v_out_dtype_code = __Pyx_PyLong_As_int32_t(values[11]); if (unlikely((__pyx_v_out_dtype_code == ((int32_t)-1)) && PyErr_Occurred())) __PYX_ERR(0, 2406, __pyx_L3_error)
src/candle/_cython/_aclnn_ffi.c:41518:    __pyx_v_self_dtype_code = __Pyx_PyLong_As_int32_t(values[9]); if (unlikely((__pyx_v_self_dtype_code == ((int32_t)-1)) && PyErr_Occurred())) __PYX_ERR(0, 2487, __pyx_L3_error)
src/candle/_cython/_aclnn_ffi.c:41519:    __pyx_v_out_dtype_code = __Pyx_PyLong_As_int32_t(values[10]); if (unlikely((__pyx_v_out_dtype_code == ((int32_t)-1)) && PyErr_Occurred())) __PYX_ERR(0, 2487, __pyx_L3_error)
src/candle/_cython/_aclnn_ffi.c:42996:    __pyx_v_self_dtype_code = __Pyx_PyLong_As_int32_t(values[8]); if (unlikely((__pyx_v_self_dtype_code == ((int32_t)-1)) && PyErr_Occurred())) __PYX_ERR(0, 2567, __pyx_L3_error)
src/candle/_cython/_aclnn_ffi.c:42997:    __pyx_v_out_dtype_code = __Pyx_PyLong_As_int32_t(values[9]); if (unlikely((__pyx_v_out_dtype_code == ((int32_t)-1)) && PyErr_Occurred())) __PYX_ERR(0, 2567, __pyx_L3_error)
src/candle/_cython/_aclnn_ffi.c:4386:    __pyx_v_dtype_code = __Pyx_PyLong_As_int32_t(values[3]); if (unlikely((__pyx_v_dtype_code == ((int32_t)-1)) && PyErr_Occurred())) __PYX_ERR(0, 128, __pyx_L3_error)
src/candle/_cython/_aclnn_ffi.c:44473:    __pyx_v_self_dtype_code = __Pyx_PyLong_As_int32_t(values[6]); if (unlikely((__pyx_v_self_dtype_code == ((int32_t)-1)) && PyErr_Occurred())) __PYX_ERR(0, 2647, __pyx_L3_error)
src/candle/_cython/_aclnn_ffi.c:44474:    __pyx_v_out_dtype_code = __Pyx_PyLong_As_int32_t(values[7]); if (unlikely((__pyx_v_out_dtype_code == ((int32_t)-1)) && PyErr_Occurred())) __PYX_ERR(0, 2647, __pyx_L3_error)
src/candle/_cython/_aclnn_ffi.c:45683:    __pyx_v_dtype_code = __Pyx_PyLong_As_int32_t(values[4]); if (unlikely((__pyx_v_dtype_code == ((int32_t)-1)) && PyErr_Occurred())) __PYX_ERR(0, 2713, __pyx_L3_error)
src/candle/_cython/_aclnn_ffi.c:46677:    __pyx_v_dtype_code = __Pyx_PyLong_As_int32_t(values[4]); if (unlikely((__pyx_v_dtype_code == ((int32_t)-1)) && PyErr_Occurred())) __PYX_ERR(0, 2763, __pyx_L3_error)
src/candle/_cython/_aclnn_ffi.c:47585:    __pyx_v_dtype_code = __Pyx_PyLong_As_int32_t(values[4]); if (unlikely((__pyx_v_dtype_code == ((int32_t)-1)) && PyErr_Occurred())) __PYX_ERR(0, 2811, __pyx_L3_error)
src/candle/_cython/_aclnn_ffi.c:48472:    __pyx_v_dtype_code = __Pyx_PyLong_As_int32_t(values[4]); if (unlikely((__pyx_v_dtype_code == ((int32_t)-1)) && PyErr_Occurred())) __PYX_ERR(0, 2859, __pyx_L3_error)
src/candle/_cython/_aclnn_ffi.c:49469:    __pyx_v_dst_dtype_code = __Pyx_PyLong_As_int32_t(values[6]); if (unlikely((__pyx_v_dst_dtype_code == ((int32_t)-1)) && PyErr_Occurred())) __PYX_ERR(0, 2911, __pyx_L3_error)
src/candle/_cython/_aclnn_ffi.c:49470:    __pyx_v_src_dtype_code = __Pyx_PyLong_As_int32_t(values[7]); if (unlikely((__pyx_v_src_dtype_code == ((int32_t)-1)) && PyErr_Occurred())) __PYX_ERR(0, 2911, __pyx_L3_error)
src/candle/_cython/_aclnn_ffi.c:50713:    __pyx_v_self_dtype_code = __Pyx_PyLong_As_int32_t(values[6]); if (unlikely((__pyx_v_self_dtype_code == ((int32_t)-1)) && PyErr_Occurred())) __PYX_ERR(0, 2977, __pyx_L3_error)
src/candle/_cython/_aclnn_ffi.c:50714:    __pyx_v_mask_dtype_code = __Pyx_PyLong_As_int32_t(values[7]); if (unlikely((__pyx_v_mask_dtype_code == ((int32_t)-1)) && PyErr_Occurred())) __PYX_ERR(0, 2977, __pyx_L3_error)
src/candle/_cython/_aclnn_ffi.c:51966:    __pyx_v_self_dtype_code = __Pyx_PyLong_As_int32_t(values[7]); if (unlikely((__pyx_v_self_dtype_code == ((int32_t)-1)) && PyErr_Occurred())) __PYX_ERR(0, 3045, __pyx_L3_error)
src/candle/_cython/_aclnn_ffi.c:51967:    __pyx_v_index_dtype_code = __Pyx_PyLong_As_int32_t(values[8]); if (unlikely((__pyx_v_index_dtype_code == ((int32_t)-1)) && PyErr_Occurred())) __PYX_ERR(0, 3045, __pyx_L3_error)
src/candle/_cython/_aclnn_ffi.c:53242:    __pyx_v_self_dtype_code = __Pyx_PyLong_As_int32_t(values[9]); if (unlikely((__pyx_v_self_dtype_code == ((int32_t)-1)) && PyErr_Occurred())) __PYX_ERR(0, 3114, __pyx_L3_error)
src/candle/_cython/_aclnn_ffi.c:53243:    __pyx_v_index_dtype_code = __Pyx_PyLong_As_int32_t(values[10]); if (unlikely((__pyx_v_index_dtype_code == ((int32_t)-1)) && PyErr_Occurred())) __PYX_ERR(0, 3114, __pyx_L3_error)
src/candle/_cython/_aclnn_ffi.c:53244:    __pyx_v_source_dtype_code = __Pyx_PyLong_As_int32_t(values[11]); if (unlikely((__pyx_v_source_dtype_code == ((int32_t)-1)) && PyErr_Occurred())) __PYX_ERR(0, 3114, __pyx_L3_error)
src/candle/_cython/_aclnn_ffi.c:54757:    __pyx_v_self_dtype_code = __Pyx_PyLong_As_int32_t(values[11]); if (unlikely((__pyx_v_self_dtype_code == ((int32_t)-1)) && PyErr_Occurred())) __PYX_ERR(0, 3196, __pyx_L3_error)
src/candle/_cython/_aclnn_ffi.c:54758:    __pyx_v_index_dtype_code = __Pyx_PyLong_As_int32_t(values[12]); if (unlikely((__pyx_v_index_dtype_code == ((int32_t)-1)) && PyErr_Occurred())) __PYX_ERR(0, 3196, __pyx_L3_error)
src/candle/_cython/_aclnn_ffi.c:54759:    __pyx_v_src_dtype_code = __Pyx_PyLong_As_int32_t(values[13]); if (unlikely((__pyx_v_src_dtype_code == ((int32_t)-1)) && PyErr_Occurred())) __PYX_ERR(0, 3197, __pyx_L3_error)
src/candle/_cython/_aclnn_ffi.c:54760:    __pyx_v_out_dtype_code = __Pyx_PyLong_As_int32_t(values[14]); if (unlikely((__pyx_v_out_dtype_code == ((int32_t)-1)) && PyErr_Occurred())) __PYX_ERR(0, 3197, __pyx_L3_error)
src/candle/_cython/_aclnn_ffi.c:56444:    __pyx_v_self_dtype_code = __Pyx_PyLong_As_int32_t(values[8]); if (unlikely((__pyx_v_self_dtype_code == ((int32_t)-1)) && PyErr_Occurred())) __PYX_ERR(0, 3292, __pyx_L3_error)
src/candle/_cython/_aclnn_ffi.c:56445:    __pyx_v_mask_dtype_code = __Pyx_PyLong_As_int32_t(values[9]); if (unlikely((__pyx_v_mask_dtype_code == ((int32_t)-1)) && PyErr_Occurred())) __PYX_ERR(0, 3292, __pyx_L3_error)
src/candle/_cython/_aclnn_ffi.c:56446:    __pyx_v_source_dtype_code = __Pyx_PyLong_As_int32_t(values[10]); if (unlikely((__pyx_v_source_dtype_code == ((int32_t)-1)) && PyErr_Occurred())) __PYX_ERR(0, 3292, __pyx_L3_error)
src/candle/_cython/_aclnn_ffi.c:57708:    __pyx_v_self_dtype_code = __Pyx_PyLong_As_int32_t(values[11]); if (unlikely((__pyx_v_self_dtype_code == ((int32_t)-1)) && PyErr_Occurred())) __PYX_ERR(0, 3363, __pyx_L3_error)
src/candle/_cython/_aclnn_ffi.c:57709:    __pyx_v_index_dtype_code = __Pyx_PyLong_As_int32_t(values[12]); if (unlikely((__pyx_v_index_dtype_code == ((int32_t)-1)) && PyErr_Occurred())) __PYX_ERR(0, 3363, __pyx_L3_error)
src/candle/_cython/_aclnn_ffi.c:57710:    __pyx_v_source_dtype_code = __Pyx_PyLong_As_int32_t(values[13]); if (unlikely((__pyx_v_source_dtype_code == ((int32_t)-1)) && PyErr_Occurred())) __PYX_ERR(0, 3364, __pyx_L3_error)
src/candle/_cython/_aclnn_ffi.c:57711:    __pyx_v_out_dtype_code = __Pyx_PyLong_As_int32_t(values[14]); if (unlikely((__pyx_v_out_dtype_code == ((int32_t)-1)) && PyErr_Occurred())) __PYX_ERR(0, 3364, __pyx_L3_error)
src/candle/_cython/_aclnn_ffi.c:59478:    __pyx_v_self_dtype_code = __Pyx_PyLong_As_int32_t(values[4]); if (unlikely((__pyx_v_self_dtype_code == ((int32_t)-1)) && PyErr_Occurred())) __PYX_ERR(0, 3456, __pyx_L3_error)
src/candle/_cython/_aclnn_ffi.c:59482:    __pyx_v_values_dtype_code = __Pyx_PyLong_As_int32_t(values[8]); if (unlikely((__pyx_v_values_dtype_code == ((int32_t)-1)) && PyErr_Occurred())) __PYX_ERR(0, 3458, __pyx_L3_error)
src/candle/_cython/_aclnn_ffi.c:61627:    __pyx_v_self_dtype_code = __Pyx_PyLong_As_int32_t(values[4]); if (unlikely((__pyx_v_self_dtype_code == ((int32_t)-1)) && PyErr_Occurred())) __PYX_ERR(0, 3599, __pyx_L3_error)
src/candle/_cython/_aclnn_ffi.c:61631:    __pyx_v_out_dtype_code = __Pyx_PyLong_As_int32_t(values[8]); if (unlikely((__pyx_v_out_dtype_code == ((int32_t)-1)) && PyErr_Occurred())) __PYX_ERR(0, 3601, __pyx_L3_error)
src/candle/_cython/_aclnn_ffi.c:63638:    __pyx_v_out_dtype_code = __Pyx_PyLong_As_int32_t(values[9]); if (unlikely((__pyx_v_out_dtype_code == ((int32_t)-1)) && PyErr_Occurred())) __PYX_ERR(0, 3731, __pyx_L3_error)
src/candle/_cython/_aclnn_ffi.c:65791:    __pyx_v_out_dtype_code = __Pyx_PyLong_As_int32_t(values[9]); if (unlikely((__pyx_v_out_dtype_code == ((int32_t)-1)) && PyErr_Occurred())) __PYX_ERR(0, 3852, __pyx_L3_error)
src/candle/_cython/_aclnn_ffi.c:67335:    __pyx_v_first_dtype_code = __Pyx_PyLong_As_int32_t(values[10]); if (unlikely((__pyx_v_first_dtype_code == ((int32_t)-1)) && PyErr_Occurred())) __PYX_ERR(0, 3938, __pyx_L3_error)
src/candle/_cython/_aclnn_ffi.c:67336:    __pyx_v_second_dtype_code = __Pyx_PyLong_As_int32_t(values[11]); if (unlikely((__pyx_v_second_dtype_code == ((int32_t)-1)) && PyErr_Occurred())) __PYX_ERR(0, 3938, __pyx_L3_error)
src/candle/_cython/_aclnn_ffi.c:67337:    __pyx_v_out_dtype_code = __Pyx_PyLong_As_int32_t(values[12]); if (unlikely((__pyx_v_out_dtype_code == ((int32_t)-1)) && PyErr_Occurred())) __PYX_ERR(0, 3938, __pyx_L3_error)
src/candle/_cython/_aclnn_ffi.c:68828:    __pyx_v_self_dtype_code = __Pyx_PyLong_As_int32_t(values[10]); if (unlikely((__pyx_v_self_dtype_code == ((int32_t)-1)) && PyErr_Occurred())) __PYX_ERR(0, 4020, __pyx_L3_error)
src/candle/_cython/_aclnn_ffi.c:68829:    __pyx_v_out_dtype_code = __Pyx_PyLong_As_int32_t(values[11]); if (unlikely((__pyx_v_out_dtype_code == ((int32_t)-1)) && PyErr_Occurred())) __PYX_ERR(0, 4020, __pyx_L3_error)
src/candle/_cython/_aclnn_ffi.c:68830:    __pyx_v_inverse_dtype_code = __Pyx_PyLong_As_int32_t(values[12]); if (unlikely((__pyx_v_inverse_dtype_code == ((int32_t)-1)) && PyErr_Occurred())) __PYX_ERR(0, 4020, __pyx_L3_error)
src/candle/_cython/_aclnn_ffi.c:70266:    __pyx_v_out_dtype_code = __Pyx_PyLong_As_int32_t(values[4]); if (unlikely((__pyx_v_out_dtype_code == ((int32_t)-1)) && PyErr_Occurred())) __PYX_ERR(0, 4096, __pyx_L3_error)
src/candle/_cython/_aclnn_ffi.c:71251:    __pyx_v_out_dtype_code = __Pyx_PyLong_As_int32_t(values[6]); if (unlikely((__pyx_v_out_dtype_code == ((int32_t)-1)) && PyErr_Occurred())) __PYX_ERR(0, 4146, __pyx_L3_error)
src/candle/_cython/_aclnn_ffi.c:72241:    __pyx_v_out_dtype_code = __Pyx_PyLong_As_int32_t(values[7]); if (unlikely((__pyx_v_out_dtype_code == ((int32_t)-1)) && PyErr_Occurred())) __PYX_ERR(0, 4195, __pyx_L3_error)
src/candle/_cython/_aclnn_ffi.c:73239:    __pyx_v_out_dtype_code = __Pyx_PyLong_As_int32_t(values[8]); if (unlikely((__pyx_v_out_dtype_code == ((int32_t)-1)) && PyErr_Occurred())) __PYX_ERR(0, 4244, __pyx_L3_error)
src/candle/_cython/_aclnn_ffi.c:74364:    __pyx_v_self_dtype_code = __Pyx_PyLong_As_int32_t(values[8]); if (unlikely((__pyx_v_self_dtype_code == ((int32_t)-1)) && PyErr_Occurred())) __PYX_ERR(0, 4303, __pyx_L3_error)
src/candle/_cython/_aclnn_ffi.c:74365:    __pyx_v_out_dtype_code = __Pyx_PyLong_As_int32_t(values[9]); if (unlikely((__pyx_v_out_dtype_code == ((int32_t)-1)) && PyErr_Occurred())) __PYX_ERR(0, 4303, __pyx_L3_error)
src/candle/_cython/_aclnn_ffi.c:75615:    __pyx_v_self_dtype_code = __Pyx_PyLong_As_int32_t(values[8]); if (unlikely((__pyx_v_self_dtype_code == ((int32_t)-1)) && PyErr_Occurred())) __PYX_ERR(0, 4372, __pyx_L3_error)
src/candle/_cython/_aclnn_ffi.c:75616:    __pyx_v_out_dtype_code = __Pyx_PyLong_As_int32_t(values[9]); if (unlikely((__pyx_v_out_dtype_code == ((int32_t)-1)) && PyErr_Occurred())) __PYX_ERR(0, 4372, __pyx_L3_error)
src/candle/_cython/_aclnn_ffi.c:76876:    __pyx_v_self_dtype_code = __Pyx_PyLong_As_int32_t(values[9]); if (unlikely((__pyx_v_self_dtype_code == ((int32_t)-1)) && PyErr_Occurred())) __PYX_ERR(0, 4440, __pyx_L3_error)
src/candle/_cython/_aclnn_ffi.c:76877:    __pyx_v_out_dtype_code = __Pyx_PyLong_As_int32_t(values[10]); if (unlikely((__pyx_v_out_dtype_code == ((int32_t)-1)) && PyErr_Occurred())) __PYX_ERR(0, 4440, __pyx_L3_error)
src/candle/_cython/_aclnn_ffi.c:77954:    __pyx_v_self_dtype_code = __Pyx_PyLong_As_int32_t(values[7]); if (unlikely((__pyx_v_self_dtype_code == ((int32_t)-1)) && PyErr_Occurred())) __PYX_ERR(0, 4498, __pyx_L3_error)
src/candle/_cython/_aclnn_ffi.c:77955:    __pyx_v_out_dtype_code = __Pyx_PyLong_As_int32_t(values[8]); if (unlikely((__pyx_v_out_dtype_code == ((int32_t)-1)) && PyErr_Occurred())) __PYX_ERR(0, 4498, __pyx_L3_error)
src/candle/_cython/_aclnn_ffi.c:7959:    __pyx_v_dtype_code = __Pyx_PyLong_As_int32_t(values[6]); if (unlikely((__pyx_v_dtype_code == ((int32_t)-1)) && PyErr_Occurred())) __PYX_ERR(0, 438, __pyx_L3_error)
src/candle/_cython/_aclnn_ffi.c:79633:    __pyx_v_self_dtype_code = __Pyx_PyLong_As_int32_t(values[7]); if (unlikely((__pyx_v_self_dtype_code == ((int32_t)-1)) && PyErr_Occurred())) __PYX_ERR(0, 4589, __pyx_L3_error)
src/candle/_cython/_aclnn_ffi.c:79634:    __pyx_v_out_dtype_code = __Pyx_PyLong_As_int32_t(values[8]); if (unlikely((__pyx_v_out_dtype_code == ((int32_t)-1)) && PyErr_Occurred())) __PYX_ERR(0, 4589, __pyx_L3_error)
src/candle/_cython/_aclnn_ffi.c:7964:    __pyx_v_alpha_dtype_code = __Pyx_PyLong_As_int32_t(values[11]); if (unlikely((__pyx_v_alpha_dtype_code == ((int32_t)-1)) && PyErr_Occurred())) __PYX_ERR(0, 440, __pyx_L3_error)
src/candle/_cython/_aclnn_ffi.c:81479:    __pyx_v_tensor_dtype_code = __Pyx_PyLong_As_int32_t(values[21]); if (unlikely((__pyx_v_tensor_dtype_code == ((int32_t)-1)) && PyErr_Occurred())) __PYX_ERR(0, 4687, __pyx_L3_error)
src/candle/_cython/_aclnn_ffi.c:81480:    __pyx_v_stats_dtype_code = __Pyx_PyLong_As_int32_t(values[22]); if (unlikely((__pyx_v_stats_dtype_code == ((int32_t)-1)) && PyErr_Occurred())) __PYX_ERR(0, 4687, __pyx_L3_error)
src/candle/_cython/_aclnn_ffi.c:83861:    __pyx_v_tensor_dtype_code = __Pyx_PyLong_As_int32_t(values[19]); if (unlikely((__pyx_v_tensor_dtype_code == ((int32_t)-1)) && PyErr_Occurred())) __PYX_ERR(0, 4841, __pyx_L3_error)
src/candle/_cython/_aclnn_ffi.c:83862:    __pyx_v_stats_dtype_code = __Pyx_PyLong_As_int32_t(values[20]); if (unlikely((__pyx_v_stats_dtype_code == ((int32_t)-1)) && PyErr_Occurred())) __PYX_ERR(0, 4841, __pyx_L3_error)
src/candle/_cython/_aclnn_ffi.c:85732:    __pyx_v_tensor_dtype_code = __Pyx_PyLong_As_int32_t(values[17]); if (unlikely((__pyx_v_tensor_dtype_code == ((int32_t)-1)) && PyErr_Occurred())) __PYX_ERR(0, 4965, __pyx_L3_error)
src/candle/_cython/_aclnn_ffi.c:87854:    __pyx_v_self_dtype_code = __Pyx_PyLong_As_int32_t(values[8]); if (unlikely((__pyx_v_self_dtype_code == ((int32_t)-1)) && PyErr_Occurred())) __PYX_ERR(0, 5103, __pyx_L3_error)
src/candle/_cython/_aclnn_ffi.c:87855:    __pyx_v_out_dtype_code = __Pyx_PyLong_As_int32_t(values[9]); if (unlikely((__pyx_v_out_dtype_code == ((int32_t)-1)) && PyErr_Occurred())) __PYX_ERR(0, 5103, __pyx_L3_error)
src/candle/_cython/_aclnn_ffi.c:8950:    __pyx_v_dtype_code = __Pyx_PyLong_As_int32_t(values[2]); if (unlikely((__pyx_v_dtype_code == ((int32_t)-1)) && PyErr_Occurred())) __PYX_ERR(0, 554, __pyx_L3_error)
src/candle/_cython/_aclnn_ffi.c:89990:    __pyx_v_self_dtype_code = __Pyx_PyLong_As_int32_t(values[12]); if (unlikely((__pyx_v_self_dtype_code == ((int32_t)-1)) && PyErr_Occurred())) __PYX_ERR(0, 5215, __pyx_L3_error)
src/candle/_cython/_aclnn_ffi.c:89991:    __pyx_v_out_dtype_code = __Pyx_PyLong_As_int32_t(values[13]); if (unlikely((__pyx_v_out_dtype_code == ((int32_t)-1)) && PyErr_Occurred())) __PYX_ERR(0, 5215, __pyx_L3_error)
src/candle/_cython/_aclnn_ffi.c:93054:    __pyx_v_self_dtype_code = __Pyx_PyLong_As_int32_t(values[13]); if (unlikely((__pyx_v_self_dtype_code == ((int32_t)-1)) && PyErr_Occurred())) __PYX_ERR(0, 5379, __pyx_L3_error)
src/candle/_cython/_aclnn_ffi.c:93055:    __pyx_v_out_dtype_code = __Pyx_PyLong_As_int32_t(values[14]); if (unlikely((__pyx_v_out_dtype_code == ((int32_t)-1)) && PyErr_Occurred())) __PYX_ERR(0, 5379, __pyx_L3_error)
src/candle/_cython/_aclnn_ffi.c:9451:    __pyx_v_dtype_code = __Pyx_PyLong_As_int32_t(values[1]); if (unlikely((__pyx_v_dtype_code == ((int32_t)-1)) && PyErr_Occurred())) __PYX_ERR(0, 585, __pyx_L3_error)
src/candle/_cython/_aclnn_ffi.c:95658:    __pyx_v_self_dtype_code = __Pyx_PyLong_As_int32_t(values[13]); if (unlikely((__pyx_v_self_dtype_code == ((int32_t)-1)) && PyErr_Occurred())) __PYX_ERR(0, 5517, __pyx_L3_error)
src/candle/_cython/_aclnn_ffi.c:95659:    __pyx_v_out_a_dtype_code = __Pyx_PyLong_As_int32_t(values[14]); if (unlikely((__pyx_v_out_a_dtype_code == ((int32_t)-1)) && PyErr_Occurred())) __PYX_ERR(0, 5517, __pyx_L3_error)
src/candle/_cython/_aclnn_ffi.c:95660:    __pyx_v_out_b_dtype_code = __Pyx_PyLong_As_int32_t(values[15]); if (unlikely((__pyx_v_out_b_dtype_code == ((int32_t)-1)) && PyErr_Occurred())) __PYX_ERR(0, 5517, __pyx_L3_error)
src/candle/_cython/_aclnn_ffi.c:98914:    __pyx_v_self_dtype_code = __Pyx_PyLong_As_int32_t(values[9]); if (unlikely((__pyx_v_self_dtype_code == ((int32_t)-1)) && PyErr_Occurred())) __PYX_ERR(0, 5694, __pyx_L3_error)
src/candle/_cython/_aclnn_ffi.c:98915:    __pyx_v_optional_dtype_code = __Pyx_PyLong_As_int32_t(values[10]); if (unlikely((__pyx_v_optional_dtype_code == ((int32_t)-1)) && PyErr_Occurred())) __PYX_ERR(0, 5694, __pyx_L3_error)
src/candle/_cython/_aclnn_ffi.c:98916:    __pyx_v_out_dtype_code = __Pyx_PyLong_As_int32_t(values[11]); if (unlikely((__pyx_v_out_dtype_code == ((int32_t)-1)) && PyErr_Occurred())) __PYX_ERR(0, 5694, __pyx_L3_error)
src/candle/_cython/_aclnn_ffi.pyx:1934:    cdef int32_t bool_dtype_code = 12
src/candle/_cython/_npu_ops.c:10457:  __pyx_v_dtype_code = __pyx_t_1;
src/candle/_cython/_npu_ops.c:11752:  __pyx_v_dtype_code = __pyx_t_1;
src/candle/_cython/_npu_ops.c:13068:  __pyx_v_dtype_code = __pyx_t_1;
src/candle/_cython/_npu_ops.c:4231: *         a_dtype_code = (<TensorImpl>a)._dtype_code
src/candle/_cython/_npu_ops.c:4240: *         a_dtype_code = (<TensorImpl>a)._dtype_code
src/candle/_cython/_npu_ops.c:4249: *         a_dtype_code = (<TensorImpl>a)._dtype_code             # <<<<<<<<<<<<<<
src/candle/_cython/_npu_ops.c:4254:    __pyx_v_a_dtype_code = __pyx_t_2;
src/candle/_cython/_npu_ops.c:4267: *         a_dtype_code = (<TensorImpl>a)._dtype_code
src/candle/_cython/_npu_ops.c:4284: *         a_dtype_code = -1  # will use Python path
src/candle/_cython/_npu_ops.c:4301: *         a_dtype_code = -1  # will use Python path
src/candle/_cython/_npu_ops.c:4322: *         a_dtype_code = -1  # will use Python path             # <<<<<<<<<<<<<<
src/candle/_cython/_npu_ops.c:4326:    __pyx_v_a_dtype_code = -1;
src/candle/_cython/_npu_ops.c:4331: *         a_dtype_code = -1  # will use Python path
src/candle/_cython/_npu_ops.c:4335: *         b_dtype_code = (<TensorImpl>b)._dtype_code
src/candle/_cython/_npu_ops.c:4344: *         b_dtype_code = (<TensorImpl>b)._dtype_code
src/candle/_cython/_npu_ops.c:4353: *         b_dtype_code = (<TensorImpl>b)._dtype_code             # <<<<<<<<<<<<<<
src/candle/_cython/_npu_ops.c:4358:    __pyx_v_b_dtype_code = __pyx_t_2;
src/candle/_cython/_npu_ops.c:4361: *         a_dtype_code = -1  # will use Python path
src/candle/_cython/_npu_ops.c:4365: *         b_dtype_code = (<TensorImpl>b)._dtype_code
src/candle/_cython/_npu_ops.c:4371: *         b_dtype_code = (<TensorImpl>b)._dtype_code
src/candle/_cython/_npu_ops.c:4375: *         b_dtype_code = -1
src/candle/_cython/_npu_ops.c:4387: *         b_dtype_code = -1
src/candle/_cython/_npu_ops.c:4404: *         b_dtype_code = -1             # <<<<<<<<<<<<<<
src/candle/_cython/_npu_ops.c:4408:    __pyx_v_b_dtype_code = -1;
src/candle/_cython/_npu_ops.c:4413: *         b_dtype_code = -1
src/candle/_cython/_npu_ops.c:4461: *         b_dtype_code = -1
src/candle/_cython/_npu_ops.c:7401:    __pyx_v_dtype_code = 0;
src/candle/_cython/_npu_ops.c:8133:  __pyx_v_dtype_code = __pyx_t_1;
src/candle/_cython/_npu_ops.pyx:156:        a_dtype_code = (<TensorImpl>a)._dtype_code
src/candle/_cython/_npu_ops.pyx:161:        a_dtype_code = -1  # will use Python path
src/candle/_cython/_npu_ops.pyx:165:        b_dtype_code = (<TensorImpl>b)._dtype_code
src/candle/_cython/_npu_ops.pyx:169:        b_dtype_code = -1
src/candle/_cython/_tensor_impl.c:12868:  __pyx_v_self->_dtype_code = __pyx_t_1;
src/candle/_cython/_tensor_impl.c:4304: *             self._dtype_code = 0
src/candle/_cython/_tensor_impl.c:4316: *             self._dtype_code = 0
src/candle/_cython/_tensor_impl.c:4325: *             self._dtype_code = 0             # <<<<<<<<<<<<<<
src/candle/_cython/_tensor_impl.c:4327: *             self._dtype_code = 1
src/candle/_cython/_tensor_impl.c:4329:    __pyx_v_self->_dtype_code = 0;
src/candle/_cython/_tensor_impl.c:4335: *             self._dtype_code = 0
src/candle/_cython/_tensor_impl.c:4343: *             self._dtype_code = 0
src/candle/_cython/_tensor_impl.c:4345: *             self._dtype_code = 1
src/candle/_cython/_tensor_impl.c:4352: *             self._dtype_code = 0
src/candle/_cython/_tensor_impl.c:4354: *             self._dtype_code = 1             # <<<<<<<<<<<<<<
src/candle/_cython/_tensor_impl.c:4356: *             self._dtype_code = 2
src/candle/_cython/_tensor_impl.c:4358:    __pyx_v_self->_dtype_code = 1;
src/candle/_cython/_tensor_impl.c:4362: *             self._dtype_code = 0
src/candle/_cython/_tensor_impl.c:4364: *             self._dtype_code = 1
src/candle/_cython/_tensor_impl.c:4372: *             self._dtype_code = 1
src/candle/_cython/_tensor_impl.c:4374: *             self._dtype_code = 2
src/candle/_cython/_tensor_impl.c:4381: *             self._dtype_code = 1
src/candle/_cython/_tensor_impl.c:4383: *             self._dtype_code = 2             # <<<<<<<<<<<<<<
src/candle/_cython/_tensor_impl.c:4385: *             self._dtype_code = 3
src/candle/_cython/_tensor_impl.c:4387:    __pyx_v_self->_dtype_code = 2;
src/candle/_cython/_tensor_impl.c:4391: *             self._dtype_code = 1
src/candle/_cython/_tensor_impl.c:4393: *             self._dtype_code = 2
src/candle/_cython/_tensor_impl.c:4401: *             self._dtype_code = 2
src/candle/_cython/_tensor_impl.c:4403: *             self._dtype_code = 3
src/candle/_cython/_tensor_impl.c:4410: *             self._dtype_code = 2
src/candle/_cython/_tensor_impl.c:4412: *             self._dtype_code = 3             # <<<<<<<<<<<<<<
src/candle/_cython/_tensor_impl.c:4414: *             self._dtype_code = 4
src/candle/_cython/_tensor_impl.c:4416:    __pyx_v_self->_dtype_code = 3;
src/candle/_cython/_tensor_impl.c:4420: *             self._dtype_code = 2
src/candle/_cython/_tensor_impl.c:4422: *             self._dtype_code = 3
src/candle/_cython/_tensor_impl.c:4430: *             self._dtype_code = 3
src/candle/_cython/_tensor_impl.c:4432: *             self._dtype_code = 4
src/candle/_cython/_tensor_impl.c:4439: *             self._dtype_code = 3
src/candle/_cython/_tensor_impl.c:4441: *             self._dtype_code = 4             # <<<<<<<<<<<<<<
src/candle/_cython/_tensor_impl.c:4443: *             self._dtype_code = 5
src/candle/_cython/_tensor_impl.c:4445:    __pyx_v_self->_dtype_code = 4;
src/candle/_cython/_tensor_impl.c:4449: *             self._dtype_code = 3
src/candle/_cython/_tensor_impl.c:4451: *             self._dtype_code = 4
src/candle/_cython/_tensor_impl.c:4459: *             self._dtype_code = 4
src/candle/_cython/_tensor_impl.c:4461: *             self._dtype_code = 5
src/candle/_cython/_tensor_impl.c:4468: *             self._dtype_code = 4
src/candle/_cython/_tensor_impl.c:4470: *             self._dtype_code = 5             # <<<<<<<<<<<<<<
src/candle/_cython/_tensor_impl.c:4472: *             self._dtype_code = 6
src/candle/_cython/_tensor_impl.c:4474:    __pyx_v_self->_dtype_code = 5;
src/candle/_cython/_tensor_impl.c:4478: *             self._dtype_code = 4
src/candle/_cython/_tensor_impl.c:4480: *             self._dtype_code = 5
src/candle/_cython/_tensor_impl.c:4488: *             self._dtype_code = 5
src/candle/_cython/_tensor_impl.c:4490: *             self._dtype_code = 6
src/candle/_cython/_tensor_impl.c:4497: *             self._dtype_code = 5
src/candle/_cython/_tensor_impl.c:4499: *             self._dtype_code = 6             # <<<<<<<<<<<<<<
src/candle/_cython/_tensor_impl.c:4501: *             self._dtype_code = 7
src/candle/_cython/_tensor_impl.c:4503:    __pyx_v_self->_dtype_code = 6;
src/candle/_cython/_tensor_impl.c:4507: *             self._dtype_code = 5
src/candle/_cython/_tensor_impl.c:4509: *             self._dtype_code = 6
src/candle/_cython/_tensor_impl.c:4517: *             self._dtype_code = 6
src/candle/_cython/_tensor_impl.c:4519: *             self._dtype_code = 7
src/candle/_cython/_tensor_impl.c:4526: *             self._dtype_code = 6
src/candle/_cython/_tensor_impl.c:4528: *             self._dtype_code = 7             # <<<<<<<<<<<<<<
src/candle/_cython/_tensor_impl.c:4530: *             self._dtype_code = 8
src/candle/_cython/_tensor_impl.c:4532:    __pyx_v_self->_dtype_code = 7;
src/candle/_cython/_tensor_impl.c:4536: *             self._dtype_code = 6
src/candle/_cython/_tensor_impl.c:4538: *             self._dtype_code = 7
src/candle/_cython/_tensor_impl.c:4546: *             self._dtype_code = 7
src/candle/_cython/_tensor_impl.c:4548: *             self._dtype_code = 8
src/candle/_cython/_tensor_impl.c:4555: *             self._dtype_code = 7
src/candle/_cython/_tensor_impl.c:4557: *             self._dtype_code = 8             # <<<<<<<<<<<<<<
src/candle/_cython/_tensor_impl.c:4559: *             self._dtype_code = 9
src/candle/_cython/_tensor_impl.c:4561:    __pyx_v_self->_dtype_code = 8;
src/candle/_cython/_tensor_impl.c:4565: *             self._dtype_code = 7
src/candle/_cython/_tensor_impl.c:4567: *             self._dtype_code = 8
src/candle/_cython/_tensor_impl.c:4575: *             self._dtype_code = 8
src/candle/_cython/_tensor_impl.c:4577: *             self._dtype_code = 9
src/candle/_cython/_tensor_impl.c:4584: *             self._dtype_code = 8
src/candle/_cython/_tensor_impl.c:4586: *             self._dtype_code = 9             # <<<<<<<<<<<<<<
src/candle/_cython/_tensor_impl.c:4588: *             self._dtype_code = -1
src/candle/_cython/_tensor_impl.c:4590:    __pyx_v_self->_dtype_code = 9;
src/candle/_cython/_tensor_impl.c:4594: *             self._dtype_code = 8
src/candle/_cython/_tensor_impl.c:4596: *             self._dtype_code = 9
src/candle/_cython/_tensor_impl.c:4603: *             self._dtype_code = 9
src/candle/_cython/_tensor_impl.c:4605: *             self._dtype_code = -1             # <<<<<<<<<<<<<<
src/candle/_cython/_tensor_impl.c:4610:    __pyx_v_self->_dtype_code = -1;
src/candle/_cython/_tensor_impl.pyx:115:            self._dtype_code = 0
src/candle/_cython/_tensor_impl.pyx:117:            self._dtype_code = 1
src/candle/_cython/_tensor_impl.pyx:119:            self._dtype_code = 2
src/candle/_cython/_tensor_impl.pyx:121:            self._dtype_code = 3
src/candle/_cython/_tensor_impl.pyx:123:            self._dtype_code = 4
src/candle/_cython/_tensor_impl.pyx:125:            self._dtype_code = 5
src/candle/_cython/_tensor_impl.pyx:127:            self._dtype_code = 6
src/candle/_cython/_tensor_impl.pyx:129:            self._dtype_code = 7
src/candle/_cython/_tensor_impl.pyx:131:            self._dtype_code = 8
src/candle/_cython/_tensor_impl.pyx:133:            self._dtype_code = 9
src/candle/_cython/_tensor_impl.pyx:135:            self._dtype_code = -1
src/candle/_tensor.py:219:        self._dtype_code = {

```

## _device_type assignments
```
src/candle/_cython/_dispatcher_core.c:10211:  PyObject *__pyx_v_device_type = NULL;
src/candle/_cython/_dispatcher_core.c:11126:    __pyx_v_device_type = __pyx_t_4;
src/candle/_cython/_dispatcher_core.c:11136:    __pyx_t_21 = (__pyx_v_device_type == Py_None);
src/candle/_cython/_dispatcher_core.c:8118:  PyObject *__pyx_v_autocast_device_type = 0;
src/candle/_cython/_dispatcher_core.c:9524: *     cdef object autocast_device_type = getattr(dispatch_device, "type", dispatch_device)
src/candle/_cython/_dispatcher_core.c:9540: *     cdef object autocast_device_type = getattr(dispatch_device, "type", dispatch_device)             # <<<<<<<<<<<<<<
src/candle/_cython/_dispatcher_core.c:9542: *         autocast_device_type = getattr(tensors[0].device, "type", None)
src/candle/_cython/_dispatcher_core.c:9546:  __pyx_v_autocast_device_type = __pyx_t_1;
src/candle/_cython/_dispatcher_core.c:9551: *     cdef object autocast_device_type = getattr(dispatch_device, "type", dispatch_device)
src/candle/_cython/_dispatcher_core.c:9553: *         autocast_device_type = getattr(tensors[0].device, "type", None)
src/candle/_cython/_dispatcher_core.c:9556:  __pyx_t_9 = (__pyx_v_autocast_device_type == Py_None);
src/candle/_cython/_dispatcher_core.c:9575: *     cdef object autocast_device_type = getattr(dispatch_device, "type", dispatch_device)
src/candle/_cython/_dispatcher_core.c:9577: *         autocast_device_type = getattr(tensors[0].device, "type", None)             # <<<<<<<<<<<<<<
src/candle/_cython/_dispatcher_core.c:9595: *     cdef object autocast_device_type = getattr(dispatch_device, "type", dispatch_device)
src/candle/_cython/_dispatcher_core.c:9597: *         autocast_device_type = getattr(tensors[0].device, "type", None)
src/candle/_cython/_dispatcher_core.c:9604: *         autocast_device_type = getattr(tensors[0].device, "type", None)
src/candle/_cython/_dispatcher_core.c:9645: *         autocast_device_type = getattr(tensors[0].device, "type", None)
src/candle/_cython/_dispatcher_core.c:9655: *         autocast_device_type = getattr(tensors[0].device, "type", None)
src/candle/_cython/_dispatcher_core.pyx:572:    cdef object autocast_device_type = getattr(dispatch_device, "type", dispatch_device)
src/candle/_cython/_dispatcher_core.pyx:574:        autocast_device_type = getattr(tensors[0].device, "type", None)
src/candle/_cython/_storage_impl.c:2663: *         self._device_type = 0
src/candle/_cython/_storage_impl.c:2671: *         self._device_type = 0
src/candle/_cython/_storage_impl.c:2679: *         self._device_type = 0             # <<<<<<<<<<<<<<
src/candle/_cython/_storage_impl.c:2683:  __pyx_v_self->_device_type = 0;
src/candle/_cython/_storage_impl.c:2687: *         self._device_type = 0
src/candle/_cython/_storage_impl.c:2695: *         self._device_type = 0
src/candle/_cython/_storage_impl.c:2734: *         if self._owner is None and self._data_ptr != NULL and self._device_type == 0 and self._resizable:
src/candle/_cython/_storage_impl.c:2758: *         if self._owner is None and self._data_ptr != NULL and self._device_type == 0 and self._resizable:             # <<<<<<<<<<<<<<
src/candle/_cython/_storage_impl.c:2774:  __pyx_t_2 = (__pyx_v_self->_device_type == 0);
src/candle/_cython/_storage_impl.c:2786: *         if self._owner is None and self._data_ptr != NULL and self._device_type == 0 and self._resizable:
src/candle/_cython/_storage_impl.c:2794: *         if self._owner is None and self._data_ptr != NULL and self._device_type == 0 and self._resizable:
src/candle/_cython/_storage_impl.c:2805: *         if self._owner is None and self._data_ptr != NULL and self._device_type == 0 and self._resizable:             # <<<<<<<<<<<<<<
src/candle/_cython/_storage_impl.c:2815: *         if self._owner is None and self._data_ptr != NULL and self._device_type == 0 and self._resizable:
src/candle/_cython/_storage_impl.c:3005: *         s._device_type = 0
src/candle/_cython/_storage_impl.c:3020: *         s._device_type = 0
src/candle/_cython/_storage_impl.c:3032: *         s._device_type = 0             # <<<<<<<<<<<<<<
src/candle/_cython/_storage_impl.c:3036:  __pyx_v_s->_device_type = 0;
src/candle/_cython/_storage_impl.c:3040: *         s._device_type = 0
src/candle/_cython/_storage_impl.c:3048: *         s._device_type = 0
src/candle/_cython/_storage_impl.c:3287: *         s._device_type = 0
src/candle/_cython/_storage_impl.c:3295: *         s._device_type = 0
src/candle/_cython/_storage_impl.c:3303: *         s._device_type = 0             # <<<<<<<<<<<<<<
src/candle/_cython/_storage_impl.c:3307:  __pyx_v_s->_device_type = 0;
src/candle/_cython/_storage_impl.c:3311: *         s._device_type = 0
src/candle/_cython/_storage_impl.c:3319: *         s._device_type = 0
src/candle/_cython/_storage_impl.c:3488:    __pyx_v_device_type = __Pyx_PyLong_As_int(values[2]); if (unlikely((__pyx_v_device_type == (int)-1) && PyErr_Occurred())) __PYX_ERR(0, 50, __pyx_L3_error)
src/candle/_cython/_storage_impl.c:3709: *         s._device_type = device_type
src/candle/_cython/_storage_impl.c:3717: *         s._device_type = device_type
src/candle/_cython/_storage_impl.c:3725: *         s._device_type = device_type             # <<<<<<<<<<<<<<
src/candle/_cython/_storage_impl.c:3729:  __pyx_v_s->_device_type = __pyx_v_device_type;
src/candle/_cython/_storage_impl.c:3733: *         s._device_type = device_type
src/candle/_cython/_storage_impl.c:3741: *         s._device_type = device_type
src/candle/_cython/_storage_impl.pyx:10:        self._device_type = 0
src/candle/_cython/_storage_impl.pyx:16:        if self._owner is None and self._data_ptr != NULL and self._device_type == 0 and self._resizable:
src/candle/_cython/_storage_impl.pyx:29:        s._device_type = 0
src/candle/_cython/_storage_impl.pyx:43:        s._device_type = 0
src/candle/_cython/_storage_impl.pyx:62:        s._device_type = device_type
src/candle/_cython/_tensor_impl.c:12601:  __pyx_v_self->_device_type = __pyx_t_1;
src/candle/_cython/_tensor_impl.c:18070: *         if self._device_type == 0:
src/candle/_cython/_tensor_impl.c:3724: *             self._device_type = 0
src/candle/_cython/_tensor_impl.c:3744: *             self._device_type = 0
src/candle/_cython/_tensor_impl.c:3753: *             self._device_type = 0             # <<<<<<<<<<<<<<
src/candle/_cython/_tensor_impl.c:3755: *             self._device_type = 1
src/candle/_cython/_tensor_impl.c:3757:    __pyx_v_self->_device_type = 0;
src/candle/_cython/_tensor_impl.c:3763: *             self._device_type = 0
src/candle/_cython/_tensor_impl.c:3771: *             self._device_type = 0
src/candle/_cython/_tensor_impl.c:3773: *             self._device_type = 1
src/candle/_cython/_tensor_impl.c:3780: *             self._device_type = 0
src/candle/_cython/_tensor_impl.c:3782: *             self._device_type = 1             # <<<<<<<<<<<<<<
src/candle/_cython/_tensor_impl.c:3784: *             self._device_type = 2
src/candle/_cython/_tensor_impl.c:3786:    __pyx_v_self->_device_type = 1;
src/candle/_cython/_tensor_impl.c:3790: *             self._device_type = 0
src/candle/_cython/_tensor_impl.c:3792: *             self._device_type = 1
src/candle/_cython/_tensor_impl.c:3800: *             self._device_type = 1
src/candle/_cython/_tensor_impl.c:3802: *             self._device_type = 2
src/candle/_cython/_tensor_impl.c:3809: *             self._device_type = 1
src/candle/_cython/_tensor_impl.c:3811: *             self._device_type = 2             # <<<<<<<<<<<<<<
src/candle/_cython/_tensor_impl.c:3813: *             self._device_type = 3
src/candle/_cython/_tensor_impl.c:3815:    __pyx_v_self->_device_type = 2;
src/candle/_cython/_tensor_impl.c:3819: *             self._device_type = 1
src/candle/_cython/_tensor_impl.c:3821: *             self._device_type = 2
src/candle/_cython/_tensor_impl.c:3829: *             self._device_type = 2
src/candle/_cython/_tensor_impl.c:3831: *             self._device_type = 3
src/candle/_cython/_tensor_impl.c:3838: *             self._device_type = 2
src/candle/_cython/_tensor_impl.c:3840: *             self._device_type = 3             # <<<<<<<<<<<<<<
src/candle/_cython/_tensor_impl.c:3842: *             self._device_type = 4
src/candle/_cython/_tensor_impl.c:3844:    __pyx_v_self->_device_type = 3;
src/candle/_cython/_tensor_impl.c:3848: *             self._device_type = 2
src/candle/_cython/_tensor_impl.c:3850: *             self._device_type = 3
src/candle/_cython/_tensor_impl.c:3858: *             self._device_type = 3
src/candle/_cython/_tensor_impl.c:3860: *             self._device_type = 4
src/candle/_cython/_tensor_impl.c:3867: *             self._device_type = 3
src/candle/_cython/_tensor_impl.c:3869: *             self._device_type = 4             # <<<<<<<<<<<<<<
src/candle/_cython/_tensor_impl.c:3871: *             self._device_type = -1
src/candle/_cython/_tensor_impl.c:3873:    __pyx_v_self->_device_type = 4;
src/candle/_cython/_tensor_impl.c:3877: *             self._device_type = 3
src/candle/_cython/_tensor_impl.c:3879: *             self._device_type = 4
src/candle/_cython/_tensor_impl.c:3886: *             self._device_type = 4
src/candle/_cython/_tensor_impl.c:3888: *             self._device_type = -1             # <<<<<<<<<<<<<<
src/candle/_cython/_tensor_impl.c:3893:    __pyx_v_self->_device_type = -1;
src/candle/_cython/_tensor_impl.c:3899: *             self._device_type = -1
src/candle/_cython/_tensor_impl.c:3910: *             self._device_type = -1
src/candle/_cython/_tensor_impl.c:6094: *         return self._device_type == 2
src/candle/_cython/_tensor_impl.c:6124: *         return self._device_type == 2             # <<<<<<<<<<<<<<
src/candle/_cython/_tensor_impl.c:6129:  __pyx_t_1 = __Pyx_PyBool_FromLong((__pyx_v_self->_device_type == 2)); if (unlikely(!__pyx_t_1)) __PYX_ERR(0, 244, __pyx_L1_error)
src/candle/_cython/_tensor_impl.c:6140: *         return self._device_type == 2
src/candle/_cython/_tensor_impl.c:6155: *         return self._device_type == 2
src/candle/_cython/_tensor_impl.c:6159: *         return self._device_type == 0
src/candle/_cython/_tensor_impl.c:6189: *         return self._device_type == 0             # <<<<<<<<<<<<<<
src/candle/_cython/_tensor_impl.c:6194:  __pyx_t_1 = __Pyx_PyBool_FromLong((__pyx_v_self->_device_type == 0)); if (unlikely(!__pyx_t_1)) __PYX_ERR(0, 248, __pyx_L1_error)
src/candle/_cython/_tensor_impl.c:6201: *         return self._device_type == 2
src/candle/_cython/_tensor_impl.c:6205: *         return self._device_type == 0
src/candle/_cython/_tensor_impl.c:6220: *         return self._device_type == 0
src/candle/_cython/_tensor_impl.c:6224: *         return self._device_type == 1
src/candle/_cython/_tensor_impl.c:6254: *         return self._device_type == 1             # <<<<<<<<<<<<<<
src/candle/_cython/_tensor_impl.c:6259:  __pyx_t_1 = __Pyx_PyBool_FromLong((__pyx_v_self->_device_type == 1)); if (unlikely(!__pyx_t_1)) __PYX_ERR(0, 252, __pyx_L1_error)
src/candle/_cython/_tensor_impl.c:6266: *         return self._device_type == 0
src/candle/_cython/_tensor_impl.c:6270: *         return self._device_type == 1
src/candle/_cython/_tensor_impl.c:6285: *         return self._device_type == 1
src/candle/_cython/_tensor_impl.c:6289: *         return self._device_type == 4
src/candle/_cython/_tensor_impl.c:6319: *         return self._device_type == 4             # <<<<<<<<<<<<<<
src/candle/_cython/_tensor_impl.c:6324:  __pyx_t_1 = __Pyx_PyBool_FromLong((__pyx_v_self->_device_type == 4)); if (unlikely(!__pyx_t_1)) __PYX_ERR(0, 256, __pyx_L1_error)
src/candle/_cython/_tensor_impl.c:6331: *         return self._device_type == 1
src/candle/_cython/_tensor_impl.c:6335: *         return self._device_type == 4
src/candle/_cython/_tensor_impl.c:6350: *         return self._device_type == 4
src/candle/_cython/_tensor_impl.c:6398: *         return self._device_type == 4
src/candle/_cython/_tensor_impl.c:6758: *         if self._device_type == 0:
src/candle/_cython/_tensor_impl.c:6818: *         if self._device_type == 0:             # <<<<<<<<<<<<<<
src/candle/_cython/_tensor_impl.c:6822:  __pyx_t_1 = (__pyx_v_self->_device_type == 0);
src/candle/_cython/_tensor_impl.c:6827: *         if self._device_type == 0:
src/candle/_cython/_tensor_impl.c:6840: *         if self._device_type == 0:             # <<<<<<<<<<<<<<
src/candle/_cython/_tensor_impl.c:6847: *         if self._device_type == 0:
src/candle/_cython/_tensor_impl.c:6872: *         if self._device_type == 0:
src/candle/_cython/_tensor_impl.pyx:244:        return self._device_type == 2
src/candle/_cython/_tensor_impl.pyx:248:        return self._device_type == 0
src/candle/_cython/_tensor_impl.pyx:252:        return self._device_type == 1
src/candle/_cython/_tensor_impl.pyx:256:        return self._device_type == 4
src/candle/_cython/_tensor_impl.pyx:282:        if self._device_type == 0:
src/candle/_cython/_tensor_impl.pyx:63:            self._device_type = 0
src/candle/_cython/_tensor_impl.pyx:65:            self._device_type = 1
src/candle/_cython/_tensor_impl.pyx:67:            self._device_type = 2
src/candle/_cython/_tensor_impl.pyx:69:            self._device_type = 3
src/candle/_cython/_tensor_impl.pyx:71:            self._device_type = 4
src/candle/_cython/_tensor_impl.pyx:73:            self._device_type = -1
src/candle/_dispatch/dispatcher.py:534:    autocast_device_type = getattr(dispatch_device, "type", dispatch_device)
src/candle/_dispatch/dispatcher.py:536:        autocast_device_type = getattr(tensors[0].device, "type", None)
src/candle/distributed/tensor/dtensor.py:192:                local_device_type = getattr(self._local_tensor.device, "type", "cpu")
src/candle/distributed/tensor/dtensor.py:233:                local_device_type = getattr(self._local_tensor.device, "type", "cpu")
src/candle/_tensor.py:184:        self._device_type = devt
src/candle/utils/data/_pin_memory_thread.py:50:        self._device_type = device_type

```

## _dispatch_keys assignments
```
src/candle/_cython/_tensor_impl.c:14472:  __pyx_v_self->_dispatch_keys = __pyx_t_1;
src/candle/_cython/_tensor_impl.c:17589:  __pyx_vtable_6candle_7_cython_12_tensor_impl_TensorImpl._recompute_dispatch_keys = (void (*)(struct __pyx_obj_6candle_7_cython_12_tensor_impl_TensorImpl *))__pyx_f_6candle_7_cython_12_tensor_impl_10TensorImpl__recompute_dispatch_keys;
src/candle/_cython/_tensor_impl.c:4209: *         self._dispatch_keys = dk
src/candle/_cython/_tensor_impl.c:4219: *         self._dispatch_keys = dk
src/candle/_cython/_tensor_impl.c:4237: *         self._dispatch_keys = dk             # <<<<<<<<<<<<<<
src/candle/_cython/_tensor_impl.c:4241:  __pyx_v_self->_dispatch_keys = __pyx_v_dk;
src/candle/_cython/_tensor_impl.c:4255: *         self._dispatch_keys = dk
src/candle/_cython/_tensor_impl.c:4615: *         self._dispatch_keys = dk
src/candle/_cython/_tensor_impl.pyx:106:        self._dispatch_keys = dk
src/candle/_tensor.py:212:        self._dispatch_keys = dk

```


## Classification

| File | Line | Pattern | Classification | Reason |
|------|------|---------|----------------|--------|
| `src/candle/_cython/_storage.pyx` | 103 | `Tensor(storage, shape, stride)` in docstring example | compatibility shell / docs-only | This is documentation text inside a Cython docstring, not executable runtime logic |
| `src/candle/_cython/_tensor_impl.pyx` | 535, 568 | `Tensor.__new__(Tensor)` | intentional unified birth implementation | These are the official Cython birth factories, not residual ad hoc births |
| `src/candle/_cython/_aclnn_ffi.c` and other generated `.c` files | many | generated assignments | ignore / generated artifact | Generated C output from Cython, not a source-level residual |
| `src/candle/_backends/npu/aclnn.py` | many | `_dtype_code`-like locals | protocol/helper internal state | These are ACLNN FFI call arguments, not Tensor metadata birth leftovers |

## Residual Summary

At source level, after excluding generated C/Cython artifacts and docstring examples, there are **no remaining unexplained direct tensor births** in the runtime-core paths targeted by Phases 1–11.

What remains falls into one of three harmless categories:

1. **documentation examples**
2. **generated C/Cython artifacts**
3. **FFI-local dtype code variables that are not Tensor metadata births**

## Stopline Recommendation

We consider the current Tensor runtime refactor complete enough to pause because:

1. Public deterministic creation paths are unified
2. Runtime internal deterministic births are unified
3. Autograd simple and medium-complex residual births are unified
4. RNG outputs follow the same birth contract
5. Cross-boundary deterministic reconstruction (multiprocessing/shared-memory/distributed boundary/stream) is unified
6. No remaining source-level unexplained direct tensor births exist in the targeted runtime-core layers

The remaining grep hits are confined to:
- generated C/Cython artifacts
- documentation examples
- FFI-local scalar/dtype argument variables

These do **not** represent runtime-level Tensor birth inconsistencies.

Further cleanup should only proceed if one of the following becomes a real maintenance or correctness problem:
- a new source-level direct birth is introduced in an active subsystem
- a protocol-heavy distributed/serialization path proves inconsistent in practice
- compatibility-shell behavior needs to be further narrowed for architectural reasons

## If We Continue Later

Priority order:
1. suspicious leftovers (if any new source-level ones appear)
2. protocol-heavy distributed exceptions
3. low-value utility cleanup only if driven by a concrete bug or maintenance burden

## Final Verification

- build_ext: PASS
- common tensor-core regressions: PASS
- CPU regression set: PASS
- NPU smoke: PASS
