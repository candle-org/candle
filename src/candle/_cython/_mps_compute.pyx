# cython: language_level=3, boundscheck=False, wraparound=False
"""Cython MetalKernelDispatcher — zero-overhead cdef class wrapping all Metal
GPU kernel dispatch methods for the MPS backend.

Imports ctypes encoding helpers from the original metal_compute module rather
than reimplementing them, so this file only contains the class skeleton and
the fast dispatch entry points.
"""

import struct
import threading

from candle._backends.mps.runtime import _HAS_PYOBJC  # pylint: disable=import-error,no-name-in-module

if _HAS_PYOBJC:
    from Metal import MTLSizeMake as _MTLSizeMake  # pylint: disable=import-error,no-name-in-module
    def _MTLSize(w, h, d):
        return _MTLSizeMake(w, h, d)
else:
    def _MTLSize(w, h, d):  # noqa: E302
        return None

# Import ctypes encoding helpers — do NOT duplicate them
from candle._backends.mps.metal_compute import (  # pylint: disable=import-error,no-name-in-module
    _library_get_function_ctypes,
    _encode_unary_ctypes,
    _encode_binary_ctypes,
    _encode_binary_scalar_ctypes,
    _encode_binary_strided_ctypes,
    _encode_binary_scalar_strided_ctypes,
    _encode_inplace_unary_ctypes,
    _encode_inplace_scalar_ctypes,
    _encode_softmax_ctypes,
    _encode_clamp_ctypes,
    _encode_clamp_strided_ctypes,
    _encode_reduce_dim_ctypes,
    _encode_cumsum_ctypes,
    _encode_cumextreme_ctypes,
    _encode_sort_ctypes,
    _encode_index_gather_ctypes,
    _encode_cat_copy_ctypes,
    _encode_conv2d_ctypes,
    _encode_layer_norm_ctypes,
    _encode_rms_norm_ctypes,
    _encode_batch_norm_stats_ctypes,
    _encode_batch_norm_apply_ctypes,
    _encode_group_norm_ctypes,
    _encode_pool2d_ctypes,
    _encode_adaptive_avg_pool2d_ctypes,
    _encode_upsample_ctypes,
    _encode_pad_constant_ctypes,
    _encode_flip_ctypes,
    _encode_roll_ctypes,
    _encode_philox_fill_ctypes,
    _encode_philox_bernoulli_ctypes,
    _encode_philox_randint_ctypes,
    _encode_philox_dropout_ctypes,
    _encode_masked_fill_ctypes,
    _encode_tril_triu_ctypes,
    _encode_where_ctypes,
    _encode_where_scalar_ctypes,
    _encode_arg_partial_ctypes,
    _encode_arg_final_ctypes,
    _encode_unary_strided_ctypes,
)

# ---------------------------------------------------------------------------
# Singleton state
# ---------------------------------------------------------------------------
cdef object _singleton = None
cdef object _singleton_lock = None

def _get_lock():
    global _singleton_lock
    if _singleton_lock is None:
        _singleton_lock = threading.Lock()
    return _singleton_lock


# ---------------------------------------------------------------------------
# cdef class MetalKernelDispatcher
# ---------------------------------------------------------------------------

cdef class MetalKernelDispatcher:
    """Cython cdef class: compiles MSL kernels lazily, caches pipelines,
    dispatches all Metal GPU compute kernels."""

    cdef object _library
    cdef dict _pipeline_cache
    cdef object _lock

    def __cinit__(self):
        self._library = None
        self._pipeline_cache = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Compilation
    # ------------------------------------------------------------------

    cpdef void ensure_compiled(self):
        """Compile all MSL source on first use."""
        if self._library is not None:
            return
        with self._lock:
            if self._library is not None:
                return
            from candle._backends.mps.metal_shaders import MSL_SOURCE  # pylint: disable=import-error,no-name-in-module
            from candle._backends.mps.runtime import get_runtime  # pylint: disable=import-error,no-name-in-module
            rt = get_runtime()
            self._library = rt.compile_library(MSL_SOURCE)

    cpdef object _get_pipeline(self, str kernel_name):
        """Get or create a cached compute pipeline for *kernel_name*."""
        cdef object pipeline
        if kernel_name in self._pipeline_cache:
            return self._pipeline_cache[kernel_name]
        self.ensure_compiled()
        from candle._backends.mps.runtime import get_runtime  # pylint: disable=import-error,no-name-in-module
        rt = get_runtime()
        if _HAS_PYOBJC:
            fn = self._library.newFunctionWithName_(kernel_name)
        else:
            fn = _library_get_function_ctypes(self._library, kernel_name)
        if fn is None or (isinstance(fn, int) and fn == 0):
            raise RuntimeError(f"Metal kernel '{kernel_name}' not found in library")
        pipeline = rt.make_compute_pipeline(fn)
        self._pipeline_cache[kernel_name] = pipeline
        return pipeline

    @staticmethod
    def _threads_per_group(pipeline):
        if _HAS_PYOBJC:
            return min(256, int(pipeline.maxTotalThreadsPerThreadgroup()))
        return 256

    # ------------------------------------------------------------------
    # Dispatch: unary  (a -> out)
    # ------------------------------------------------------------------

    cpdef void dispatch_unary(self, str kernel_name, object a_metal_buf,
                              object out_metal_buf, int numel):
        """Encode and execute a unary element-wise kernel."""
        from candle._backends.mps.runtime import get_runtime  # pylint: disable=import-error,no-name-in-module
        rt = get_runtime()
        cdef object pipeline = self._get_pipeline(kernel_name)
        cdef int tpg = self._threads_per_group(pipeline)
        cdef int groups = (numel + tpg - 1) // tpg

        cmd = rt.create_command_buffer()
        enc = rt.get_compute_encoder(cmd)

        if _HAS_PYOBJC:
            enc.setComputePipelineState_(pipeline)
            enc.setBuffer_offset_atIndex_(a_metal_buf, 0, 0)
            enc.setBuffer_offset_atIndex_(out_metal_buf, 0, 1)
            enc.setBytes_length_atIndex_(struct.pack("I", numel), 4, 2)
            enc.dispatchThreadgroups_threadsPerThreadgroup_(
                _MTLSize(groups, 1, 1), _MTLSize(tpg, 1, 1))
            enc.endEncoding()
        else:
            _encode_unary_ctypes(enc, pipeline, a_metal_buf, out_metal_buf,
                                 numel, groups, tpg)

        rt.commit_and_wait(cmd)

    # ------------------------------------------------------------------
    # Dispatch: binary  (a, b -> out)
    # ------------------------------------------------------------------

    cpdef void dispatch_binary(self, str kernel_name, object a_buf,
                               object b_buf, object out_buf, int numel):
        """Encode and execute a binary element-wise kernel."""
        from candle._backends.mps.runtime import get_runtime  # pylint: disable=import-error,no-name-in-module
        rt = get_runtime()
        cdef object pipeline = self._get_pipeline(kernel_name)
        cdef int tpg = self._threads_per_group(pipeline)
        cdef int groups = (numel + tpg - 1) // tpg

        cmd = rt.create_command_buffer()
        enc = rt.get_compute_encoder(cmd)

        if _HAS_PYOBJC:
            enc.setComputePipelineState_(pipeline)
            enc.setBuffer_offset_atIndex_(a_buf, 0, 0)
            enc.setBuffer_offset_atIndex_(b_buf, 0, 1)
            enc.setBuffer_offset_atIndex_(out_buf, 0, 2)
            enc.setBytes_length_atIndex_(struct.pack("I", numel), 4, 3)
            enc.dispatchThreadgroups_threadsPerThreadgroup_(
                _MTLSize(groups, 1, 1), _MTLSize(tpg, 1, 1))
            enc.endEncoding()
        else:
            _encode_binary_ctypes(enc, pipeline, a_buf, b_buf, out_buf,
                                  numel, groups, tpg)

        rt.commit_and_wait(cmd)

    # ------------------------------------------------------------------
    # Dispatch: binary scalar  (a, scalar -> out)
    # ------------------------------------------------------------------

    cpdef void dispatch_binary_scalar(self, str kernel_name, object a_buf,
                                      object scalar, object out_buf, int numel,
                                      str scalar_fmt="f"):
        """Encode and execute a binary element-wise kernel with a scalar."""
        from candle._backends.mps.runtime import get_runtime  # pylint: disable=import-error,no-name-in-module
        rt = get_runtime()
        cdef object pipeline = self._get_pipeline(kernel_name)
        cdef int tpg = self._threads_per_group(pipeline)
        cdef int groups = (numel + tpg - 1) // tpg

        cmd = rt.create_command_buffer()
        enc = rt.get_compute_encoder(cmd)

        cdef bytes scalar_bytes = struct.pack(scalar_fmt, scalar)
        cdef int scalar_size = len(scalar_bytes)

        if _HAS_PYOBJC:
            enc.setComputePipelineState_(pipeline)
            enc.setBuffer_offset_atIndex_(a_buf, 0, 0)
            enc.setBytes_length_atIndex_(scalar_bytes, scalar_size, 1)
            enc.setBuffer_offset_atIndex_(out_buf, 0, 2)
            enc.setBytes_length_atIndex_(struct.pack("I", numel), 4, 3)
            enc.dispatchThreadgroups_threadsPerThreadgroup_(
                _MTLSize(groups, 1, 1), _MTLSize(tpg, 1, 1))
            enc.endEncoding()
        else:
            _encode_binary_scalar_ctypes(enc, pipeline, a_buf,
                                         scalar_bytes, scalar_size,
                                         out_buf, numel, groups, tpg)

        rt.commit_and_wait(cmd)

    # ------------------------------------------------------------------
    # Dispatch: reduction  (input -> scalar output, two-pass)
    # ------------------------------------------------------------------

    cpdef void dispatch_reduction(self, str partial_kernel, str final_kernel,
                                  object a_buf, object out_buf, int numel):
        """Two-pass parallel reduction: per-threadgroup partials -> final."""
        from candle._backends.mps.runtime import get_runtime  # pylint: disable=import-error,no-name-in-module
        rt = get_runtime()
        cdef object pipeline_p = self._get_pipeline(partial_kernel)
        cdef object pipeline_f = self._get_pipeline(final_kernel)
        cdef int tpg = self._threads_per_group(pipeline_p)
        cdef int num_groups = (numel + tpg - 1) // tpg

        partials_buf = rt.create_buffer(num_groups * 4)

        cmd = rt.create_command_buffer()
        enc = rt.get_compute_encoder(cmd)
        if _HAS_PYOBJC:
            enc.setComputePipelineState_(pipeline_p)
            enc.setBuffer_offset_atIndex_(a_buf, 0, 0)
            enc.setBuffer_offset_atIndex_(partials_buf, 0, 1)
            enc.setBytes_length_atIndex_(struct.pack("I", numel), 4, 2)
            enc.dispatchThreadgroups_threadsPerThreadgroup_(
                _MTLSize(num_groups, 1, 1), _MTLSize(tpg, 1, 1))
            enc.endEncoding()
        else:
            _encode_unary_ctypes(enc, pipeline_p, a_buf, partials_buf,
                                 numel, num_groups, tpg)
        rt.commit_and_wait(cmd)

        cdef int final_tpg = self._threads_per_group(pipeline_f)
        final_tpg = max(final_tpg, num_groups)
        final_tpg = 1
        while final_tpg < num_groups:
            final_tpg *= 2
        final_tpg = min(final_tpg, 256)

        cmd2 = rt.create_command_buffer()
        enc2 = rt.get_compute_encoder(cmd2)
        if _HAS_PYOBJC:
            enc2.setComputePipelineState_(pipeline_f)
            enc2.setBuffer_offset_atIndex_(partials_buf, 0, 0)
            enc2.setBuffer_offset_atIndex_(out_buf, 0, 1)
            enc2.setBytes_length_atIndex_(struct.pack("I", num_groups), 4, 2)
            enc2.dispatchThreadgroups_threadsPerThreadgroup_(
                _MTLSize(1, 1, 1), _MTLSize(final_tpg, 1, 1))
            enc2.endEncoding()
        else:
            _encode_unary_ctypes(enc2, pipeline_f, partials_buf, out_buf,
                                 num_groups, 1, final_tpg)
        rt.commit_and_wait(cmd2)

    # ------------------------------------------------------------------
    # Dispatch: arg reduction  (two-pass with value+index)
    # ------------------------------------------------------------------

    cpdef void dispatch_arg_reduction(self, str partial_kernel,
                                      str final_kernel, object a_buf,
                                      object out_buf, int numel):
        """Two-pass argmax/argmin: partials carry (value, index) pairs."""
        from candle._backends.mps.runtime import get_runtime  # pylint: disable=import-error,no-name-in-module
        rt = get_runtime()
        cdef object pipeline_p = self._get_pipeline(partial_kernel)
        cdef object pipeline_f = self._get_pipeline(final_kernel)
        cdef int tpg = self._threads_per_group(pipeline_p)
        cdef int num_groups = (numel + tpg - 1) // tpg

        partial_vals_buf = rt.create_buffer(num_groups * 4)
        partial_idxs_buf = rt.create_buffer(num_groups * 4)

        cmd = rt.create_command_buffer()
        enc = rt.get_compute_encoder(cmd)
        if _HAS_PYOBJC:
            enc.setComputePipelineState_(pipeline_p)
            enc.setBuffer_offset_atIndex_(a_buf, 0, 0)
            enc.setBuffer_offset_atIndex_(partial_vals_buf, 0, 1)
            enc.setBuffer_offset_atIndex_(partial_idxs_buf, 0, 2)
            enc.setBytes_length_atIndex_(struct.pack("I", numel), 4, 3)
            enc.dispatchThreadgroups_threadsPerThreadgroup_(
                _MTLSize(num_groups, 1, 1), _MTLSize(tpg, 1, 1))
            enc.endEncoding()
        else:
            _encode_arg_partial_ctypes(enc, pipeline_p, a_buf,
                                       partial_vals_buf, partial_idxs_buf,
                                       numel, num_groups, tpg)
        rt.commit_and_wait(cmd)

        cdef int final_tpg = 1
        while final_tpg < num_groups:
            final_tpg *= 2
        final_tpg = min(final_tpg, 256)

        cmd2 = rt.create_command_buffer()
        enc2 = rt.get_compute_encoder(cmd2)
        if _HAS_PYOBJC:
            enc2.setComputePipelineState_(pipeline_f)
            enc2.setBuffer_offset_atIndex_(partial_vals_buf, 0, 0)
            enc2.setBuffer_offset_atIndex_(partial_idxs_buf, 0, 1)
            enc2.setBuffer_offset_atIndex_(out_buf, 0, 2)
            enc2.setBytes_length_atIndex_(struct.pack("I", num_groups), 4, 3)
            enc2.dispatchThreadgroups_threadsPerThreadgroup_(
                _MTLSize(1, 1, 1), _MTLSize(final_tpg, 1, 1))
            enc2.endEncoding()
        else:
            _encode_arg_final_ctypes(enc2, pipeline_f, partial_vals_buf,
                                     partial_idxs_buf, out_buf,
                                     num_groups, final_tpg)
        rt.commit_and_wait(cmd2)

    # ------------------------------------------------------------------
    # Dispatch: in-place unary  (a -> a)
    # ------------------------------------------------------------------

    cpdef void dispatch_inplace_unary(self, str kernel_name, object a_buf,
                                      int numel):
        """In-place unary: writes output back to input buffer."""
        from candle._backends.mps.runtime import get_runtime  # pylint: disable=import-error,no-name-in-module
        rt = get_runtime()
        cdef object pipeline = self._get_pipeline(kernel_name)
        cdef int tpg = self._threads_per_group(pipeline)
        cdef int groups = (numel + tpg - 1) // tpg

        cmd = rt.create_command_buffer()
        enc = rt.get_compute_encoder(cmd)

        if _HAS_PYOBJC:
            enc.setComputePipelineState_(pipeline)
            enc.setBuffer_offset_atIndex_(a_buf, 0, 0)
            enc.setBytes_length_atIndex_(struct.pack("I", numel), 4, 1)
            enc.dispatchThreadgroups_threadsPerThreadgroup_(
                _MTLSize(groups, 1, 1), _MTLSize(tpg, 1, 1))
            enc.endEncoding()
        else:
            _encode_inplace_unary_ctypes(enc, pipeline, a_buf, numel,
                                         groups, tpg)

        rt.commit_and_wait(cmd)

    # ------------------------------------------------------------------
    # Dispatch: in-place binary  (a, b -> a)
    # ------------------------------------------------------------------

    cpdef void dispatch_inplace_binary(self, str kernel_name, object a_buf,
                                       object b_buf, int numel):
        """In-place binary: a[i] op= b[i]."""
        from candle._backends.mps.runtime import get_runtime  # pylint: disable=import-error,no-name-in-module
        rt = get_runtime()
        cdef object pipeline = self._get_pipeline(kernel_name)
        cdef int tpg = self._threads_per_group(pipeline)
        cdef int groups = (numel + tpg - 1) // tpg

        cmd = rt.create_command_buffer()
        enc = rt.get_compute_encoder(cmd)

        if _HAS_PYOBJC:
            enc.setComputePipelineState_(pipeline)
            enc.setBuffer_offset_atIndex_(a_buf, 0, 0)
            enc.setBuffer_offset_atIndex_(b_buf, 0, 1)
            enc.setBytes_length_atIndex_(struct.pack("I", numel), 4, 2)
            enc.dispatchThreadgroups_threadsPerThreadgroup_(
                _MTLSize(groups, 1, 1), _MTLSize(tpg, 1, 1))
            enc.endEncoding()
        else:
            _encode_unary_ctypes(enc, pipeline, a_buf, b_buf, numel,
                                 groups, tpg)

        rt.commit_and_wait(cmd)

    # ------------------------------------------------------------------
    # Dispatch: in-place binary scalar  (a, scalar -> a)
    # ------------------------------------------------------------------

    cpdef void dispatch_inplace_binary_scalar(self, str kernel_name,
                                              object a_buf, object scalar,
                                              int numel, str scalar_fmt="f"):
        """In-place binary scalar: a[i] op= scalar."""
        from candle._backends.mps.runtime import get_runtime  # pylint: disable=import-error,no-name-in-module
        rt = get_runtime()
        cdef object pipeline = self._get_pipeline(kernel_name)
        cdef int tpg = self._threads_per_group(pipeline)
        cdef int groups = (numel + tpg - 1) // tpg

        cdef bytes scalar_bytes = struct.pack(scalar_fmt, scalar)
        cdef int scalar_size = len(scalar_bytes)

        cmd = rt.create_command_buffer()
        enc = rt.get_compute_encoder(cmd)

        if _HAS_PYOBJC:
            enc.setComputePipelineState_(pipeline)
            enc.setBuffer_offset_atIndex_(a_buf, 0, 0)
            enc.setBytes_length_atIndex_(scalar_bytes, scalar_size, 1)
            enc.setBytes_length_atIndex_(struct.pack("I", numel), 4, 2)
            enc.dispatchThreadgroups_threadsPerThreadgroup_(
                _MTLSize(groups, 1, 1), _MTLSize(tpg, 1, 1))
            enc.endEncoding()
        else:
            _encode_inplace_scalar_ctypes(enc, pipeline, a_buf, scalar_bytes,
                                          scalar_size, numel, groups, tpg)

        rt.commit_and_wait(cmd)

    # ------------------------------------------------------------------
    # Dispatch: fill  (out[i] = scalar)
    # ------------------------------------------------------------------

    cpdef void dispatch_fill(self, str kernel_name, object out_buf,
                             object scalar, int numel, str scalar_fmt="f"):
        """Encode and execute a fill kernel."""
        from candle._backends.mps.runtime import get_runtime  # pylint: disable=import-error,no-name-in-module
        rt = get_runtime()
        cdef object pipeline = self._get_pipeline(kernel_name)
        cdef int tpg = self._threads_per_group(pipeline)
        cdef int groups = (numel + tpg - 1) // tpg

        cdef bytes scalar_bytes = struct.pack(scalar_fmt, scalar)
        cdef int scalar_size = len(scalar_bytes)

        cmd = rt.create_command_buffer()
        enc = rt.get_compute_encoder(cmd)

        if _HAS_PYOBJC:
            enc.setComputePipelineState_(pipeline)
            enc.setBuffer_offset_atIndex_(out_buf, 0, 0)
            enc.setBytes_length_atIndex_(scalar_bytes, scalar_size, 1)
            enc.setBytes_length_atIndex_(struct.pack("I", numel), 4, 2)
            enc.dispatchThreadgroups_threadsPerThreadgroup_(
                _MTLSize(groups, 1, 1), _MTLSize(tpg, 1, 1))
            enc.endEncoding()
        else:
            _encode_inplace_scalar_ctypes(enc, pipeline, out_buf, scalar_bytes,
                                          scalar_size, numel, groups, tpg)

        rt.commit_and_wait(cmd)

    # ------------------------------------------------------------------
    # Dispatch: copy  (src -> dst)
    # ------------------------------------------------------------------

    cpdef void dispatch_copy(self, str kernel_name, object src_buf,
                             object dst_buf, int numel):
        """Encode a copy kernel."""
        self.dispatch_unary(kernel_name, src_buf, dst_buf, numel)

    # ------------------------------------------------------------------
    # Dispatch: unary strided
    # ------------------------------------------------------------------

    cpdef void dispatch_unary_strided(self, str kernel_name, object a_metal_buf,
                                      object out_metal_buf, int numel,
                                      object shape_array, object strides_a_array,
                                      int ndim):
        """Encode and execute a strided unary kernel."""
        from candle._backends.mps.runtime import get_runtime  # pylint: disable=import-error,no-name-in-module
        rt = get_runtime()
        cdef object pipeline = self._get_pipeline(kernel_name)
        cdef int tpg = self._threads_per_group(pipeline)
        cdef int groups = (numel + tpg - 1) // tpg

        cdef bytes shape_bytes = struct.pack(f"{ndim}I", *shape_array)
        cdef bytes strides_bytes = struct.pack(f"{ndim}i", *strides_a_array)

        cmd = rt.create_command_buffer()
        enc = rt.get_compute_encoder(cmd)

        if _HAS_PYOBJC:
            enc.setComputePipelineState_(pipeline)
            enc.setBuffer_offset_atIndex_(a_metal_buf, 0, 0)
            enc.setBuffer_offset_atIndex_(out_metal_buf, 0, 1)
            enc.setBytes_length_atIndex_(struct.pack("I", numel), 4, 2)
            enc.setBytes_length_atIndex_(shape_bytes, len(shape_bytes), 3)
            enc.setBytes_length_atIndex_(strides_bytes, len(strides_bytes), 4)
            enc.setBytes_length_atIndex_(struct.pack("I", ndim), 4, 5)
            enc.dispatchThreadgroups_threadsPerThreadgroup_(
                _MTLSize(groups, 1, 1), _MTLSize(tpg, 1, 1))
            enc.endEncoding()
        else:
            _encode_unary_strided_ctypes(enc, pipeline, a_metal_buf,
                                         out_metal_buf, numel, shape_bytes,
                                         strides_bytes, ndim, groups, tpg)

        rt.commit_and_wait(cmd)

    # ------------------------------------------------------------------
    # Dispatch: binary strided
    # ------------------------------------------------------------------

    cpdef void dispatch_binary_strided(self, str kernel_name, object a_buf,
                                       object b_buf, object out_buf, int numel,
                                       object shape_array,
                                       object strides_a_array,
                                       object strides_b_array, int ndim):
        """Encode and execute a strided binary kernel."""
        from candle._backends.mps.runtime import get_runtime  # pylint: disable=import-error,no-name-in-module
        rt = get_runtime()
        cdef object pipeline = self._get_pipeline(kernel_name)
        cdef int tpg = self._threads_per_group(pipeline)
        cdef int groups = (numel + tpg - 1) // tpg

        cdef bytes shape_bytes = struct.pack(f"{ndim}I", *shape_array)
        cdef bytes strides_a_bytes = struct.pack(f"{ndim}i", *strides_a_array)
        cdef bytes strides_b_bytes = struct.pack(f"{ndim}i", *strides_b_array)

        cmd = rt.create_command_buffer()
        enc = rt.get_compute_encoder(cmd)

        if _HAS_PYOBJC:
            enc.setComputePipelineState_(pipeline)
            enc.setBuffer_offset_atIndex_(a_buf, 0, 0)
            enc.setBuffer_offset_atIndex_(b_buf, 0, 1)
            enc.setBuffer_offset_atIndex_(out_buf, 0, 2)
            enc.setBytes_length_atIndex_(struct.pack("I", numel), 4, 3)
            enc.setBytes_length_atIndex_(shape_bytes, len(shape_bytes), 4)
            enc.setBytes_length_atIndex_(strides_a_bytes, len(strides_a_bytes), 5)
            enc.setBytes_length_atIndex_(strides_b_bytes, len(strides_b_bytes), 6)
            enc.setBytes_length_atIndex_(struct.pack("I", ndim), 4, 7)
            enc.dispatchThreadgroups_threadsPerThreadgroup_(
                _MTLSize(groups, 1, 1), _MTLSize(tpg, 1, 1))
            enc.endEncoding()
        else:
            _encode_binary_strided_ctypes(enc, pipeline, a_buf, b_buf,
                                          out_buf, numel, shape_bytes,
                                          strides_a_bytes, strides_b_bytes,
                                          ndim, groups, tpg)

        rt.commit_and_wait(cmd)

    # ------------------------------------------------------------------
    # Dispatch: binary scalar strided
    # ------------------------------------------------------------------

    cpdef void dispatch_binary_scalar_strided(self, str kernel_name,
                                              object a_buf, object scalar,
                                              object out_buf, int numel,
                                              object shape_array,
                                              object strides_a_array, int ndim,
                                              str scalar_fmt="f"):
        """Encode and execute a strided binary-scalar kernel."""
        from candle._backends.mps.runtime import get_runtime  # pylint: disable=import-error,no-name-in-module
        rt = get_runtime()
        cdef object pipeline = self._get_pipeline(kernel_name)
        cdef int tpg = self._threads_per_group(pipeline)
        cdef int groups = (numel + tpg - 1) // tpg

        cdef bytes scalar_bytes = struct.pack(scalar_fmt, scalar)
        cdef int scalar_size = len(scalar_bytes)
        cdef bytes shape_bytes = struct.pack(f"{ndim}I", *shape_array)
        cdef bytes strides_bytes = struct.pack(f"{ndim}i", *strides_a_array)

        cmd = rt.create_command_buffer()
        enc = rt.get_compute_encoder(cmd)

        if _HAS_PYOBJC:
            enc.setComputePipelineState_(pipeline)
            enc.setBuffer_offset_atIndex_(a_buf, 0, 0)
            enc.setBytes_length_atIndex_(scalar_bytes, scalar_size, 1)
            enc.setBuffer_offset_atIndex_(out_buf, 0, 2)
            enc.setBytes_length_atIndex_(struct.pack("I", numel), 4, 3)
            enc.setBytes_length_atIndex_(shape_bytes, len(shape_bytes), 4)
            enc.setBytes_length_atIndex_(strides_bytes, len(strides_bytes), 5)
            enc.setBytes_length_atIndex_(struct.pack("I", ndim), 4, 6)
            enc.dispatchThreadgroups_threadsPerThreadgroup_(
                _MTLSize(groups, 1, 1), _MTLSize(tpg, 1, 1))
            enc.endEncoding()
        else:
            _encode_binary_scalar_strided_ctypes(
                enc, pipeline, a_buf, scalar_bytes, scalar_size,
                out_buf, numel, shape_bytes, strides_bytes, ndim,
                groups, tpg)

        rt.commit_and_wait(cmd)

    # ------------------------------------------------------------------
    # Dispatch: comparison  (a, b -> bool out)
    # ------------------------------------------------------------------

    cpdef void dispatch_comparison(self, str kernel_name, object a_buf,
                                   object b_buf, object out_buf, int numel):
        """Encode a comparison kernel (output is bool/uint8)."""
        self.dispatch_binary(kernel_name, a_buf, b_buf, out_buf, numel)

    cpdef void dispatch_comparison_scalar(self, str kernel_name, object a_buf,
                                          object scalar, object out_buf,
                                          int numel, str scalar_fmt="f"):
        """Encode a scalar comparison kernel."""
        self.dispatch_binary_scalar(kernel_name, a_buf, scalar, out_buf,
                                    numel, scalar_fmt)

    # ------------------------------------------------------------------
    # Dispatch: clamp  (a, lo, hi -> out)
    # ------------------------------------------------------------------

    cpdef void dispatch_clamp(self, str kernel_name, object a_buf,
                              object scalar1, object scalar2, object out_buf,
                              int numel, str scalar_fmt="f"):
        """Encode clamp kernel (2-scalar bounds)."""
        from candle._backends.mps.runtime import get_runtime  # pylint: disable=import-error,no-name-in-module
        rt = get_runtime()
        cdef object pipeline = self._get_pipeline(kernel_name)
        cdef int tpg = self._threads_per_group(pipeline)
        cdef int groups = (numel + tpg - 1) // tpg

        cdef bytes s1_bytes = struct.pack(scalar_fmt, scalar1)
        cdef bytes s2_bytes = struct.pack(scalar_fmt, scalar2)
        cdef int scalar_size = len(s1_bytes)

        cmd = rt.create_command_buffer()
        enc = rt.get_compute_encoder(cmd)

        if _HAS_PYOBJC:
            enc.setComputePipelineState_(pipeline)
            enc.setBuffer_offset_atIndex_(a_buf, 0, 0)
            enc.setBytes_length_atIndex_(s1_bytes, scalar_size, 1)
            enc.setBytes_length_atIndex_(s2_bytes, scalar_size, 2)
            enc.setBuffer_offset_atIndex_(out_buf, 0, 3)
            enc.setBytes_length_atIndex_(struct.pack("I", numel), 4, 4)
            enc.dispatchThreadgroups_threadsPerThreadgroup_(
                _MTLSize(groups, 1, 1), _MTLSize(tpg, 1, 1))
            enc.endEncoding()
        else:
            _encode_clamp_ctypes(enc, pipeline, a_buf, s1_bytes, s2_bytes,
                                 scalar_size, out_buf, numel, groups, tpg)

        rt.commit_and_wait(cmd)

    # ------------------------------------------------------------------
    # Dispatch: clamp strided
    # ------------------------------------------------------------------

    cpdef void dispatch_clamp_strided(self, str kernel_name, object a_buf,
                                      object scalar1, object scalar2,
                                      object out_buf, int numel,
                                      object shape_array,
                                      object strides_a_array, int ndim,
                                      str scalar_fmt="f"):
        """Encode strided clamp kernel."""
        from candle._backends.mps.runtime import get_runtime  # pylint: disable=import-error,no-name-in-module
        rt = get_runtime()
        cdef object pipeline = self._get_pipeline(kernel_name)
        cdef int tpg = self._threads_per_group(pipeline)
        cdef int groups = (numel + tpg - 1) // tpg

        cdef bytes s1_bytes = struct.pack(scalar_fmt, scalar1)
        cdef bytes s2_bytes = struct.pack(scalar_fmt, scalar2)
        cdef int scalar_size = len(s1_bytes)
        cdef bytes shape_bytes = struct.pack(f"{ndim}I", *shape_array)
        cdef bytes strides_bytes = struct.pack(f"{ndim}i", *strides_a_array)

        cmd = rt.create_command_buffer()
        enc = rt.get_compute_encoder(cmd)

        if _HAS_PYOBJC:
            enc.setComputePipelineState_(pipeline)
            enc.setBuffer_offset_atIndex_(a_buf, 0, 0)
            enc.setBytes_length_atIndex_(s1_bytes, scalar_size, 1)
            enc.setBytes_length_atIndex_(s2_bytes, scalar_size, 2)
            enc.setBuffer_offset_atIndex_(out_buf, 0, 3)
            enc.setBytes_length_atIndex_(struct.pack("I", numel), 4, 4)
            enc.setBytes_length_atIndex_(shape_bytes, len(shape_bytes), 5)
            enc.setBytes_length_atIndex_(strides_bytes, len(strides_bytes), 6)
            enc.setBytes_length_atIndex_(struct.pack("I", ndim), 4, 7)
            enc.dispatchThreadgroups_threadsPerThreadgroup_(
                _MTLSize(groups, 1, 1), _MTLSize(tpg, 1, 1))
            enc.endEncoding()
        else:
            _encode_clamp_strided_ctypes(enc, pipeline, a_buf,
                                         s1_bytes, s2_bytes, scalar_size,
                                         out_buf, numel, shape_bytes,
                                         strides_bytes, ndim, groups, tpg)

        rt.commit_and_wait(cmd)

    # ------------------------------------------------------------------
    # Dispatch: where  (cond, x, y -> out)
    # ------------------------------------------------------------------

    cpdef void dispatch_where(self, str kernel_name, object cond_buf,
                              object x_buf, object y_buf, object out_buf,
                              int numel):
        """Encode where kernel (cond ? x : y)."""
        from candle._backends.mps.runtime import get_runtime  # pylint: disable=import-error,no-name-in-module
        rt = get_runtime()
        cdef object pipeline = self._get_pipeline(kernel_name)
        cdef int tpg = self._threads_per_group(pipeline)
        cdef int groups = (numel + tpg - 1) // tpg

        cmd = rt.create_command_buffer()
        enc = rt.get_compute_encoder(cmd)

        if _HAS_PYOBJC:
            enc.setComputePipelineState_(pipeline)
            enc.setBuffer_offset_atIndex_(cond_buf, 0, 0)
            enc.setBuffer_offset_atIndex_(x_buf, 0, 1)
            enc.setBuffer_offset_atIndex_(y_buf, 0, 2)
            enc.setBuffer_offset_atIndex_(out_buf, 0, 3)
            enc.setBytes_length_atIndex_(struct.pack("I", numel), 4, 4)
            enc.dispatchThreadgroups_threadsPerThreadgroup_(
                _MTLSize(groups, 1, 1), _MTLSize(tpg, 1, 1))
            enc.endEncoding()
        else:
            _encode_where_ctypes(enc, pipeline, cond_buf, x_buf, y_buf,
                                 out_buf, numel, groups, tpg)

        rt.commit_and_wait(cmd)

    cpdef void dispatch_where_scalar(self, str kernel_name, object cond_buf,
                                     object tensor_buf, object scalar,
                                     object out_buf, int numel,
                                     str scalar_fmt="f"):
        """Encode where_scalar kernel (cond ? tensor : scalar or cond ? scalar : tensor)."""
        from candle._backends.mps.runtime import get_runtime  # pylint: disable=import-error,no-name-in-module
        rt = get_runtime()
        cdef object pipeline = self._get_pipeline(kernel_name)
        cdef int tpg = self._threads_per_group(pipeline)
        cdef int groups = (numel + tpg - 1) // tpg

        cdef bytes scalar_bytes = struct.pack(scalar_fmt, scalar)
        cdef int scalar_size = len(scalar_bytes)

        cmd = rt.create_command_buffer()
        enc = rt.get_compute_encoder(cmd)

        if _HAS_PYOBJC:
            enc.setComputePipelineState_(pipeline)
            enc.setBuffer_offset_atIndex_(cond_buf, 0, 0)
            enc.setBuffer_offset_atIndex_(tensor_buf, 0, 1)
            enc.setBytes_length_atIndex_(scalar_bytes, scalar_size, 2)
            enc.setBuffer_offset_atIndex_(out_buf, 0, 3)
            enc.setBytes_length_atIndex_(struct.pack("I", numel), 4, 4)
            enc.dispatchThreadgroups_threadsPerThreadgroup_(
                _MTLSize(groups, 1, 1), _MTLSize(tpg, 1, 1))
            enc.endEncoding()
        else:
            _encode_where_scalar_ctypes(enc, pipeline, cond_buf, tensor_buf,
                                        scalar_bytes, scalar_size, out_buf,
                                        numel, groups, tpg)

        rt.commit_and_wait(cmd)

    # ------------------------------------------------------------------
    # Dispatch: masked_fill
    # ------------------------------------------------------------------

    cpdef void dispatch_masked_fill(self, str kernel_name, object a_buf,
                                    object mask_buf, object scalar,
                                    object out_buf, int numel,
                                    str scalar_fmt="f"):
        """Encode masked_fill kernel."""
        from candle._backends.mps.runtime import get_runtime  # pylint: disable=import-error,no-name-in-module
        rt = get_runtime()
        cdef object pipeline = self._get_pipeline(kernel_name)
        cdef int tpg = self._threads_per_group(pipeline)
        cdef int groups = (numel + tpg - 1) // tpg

        cdef bytes scalar_bytes = struct.pack(scalar_fmt, scalar)
        cdef int scalar_size = len(scalar_bytes)

        cmd = rt.create_command_buffer()
        enc = rt.get_compute_encoder(cmd)

        if _HAS_PYOBJC:
            enc.setComputePipelineState_(pipeline)
            enc.setBuffer_offset_atIndex_(a_buf, 0, 0)
            enc.setBuffer_offset_atIndex_(mask_buf, 0, 1)
            enc.setBytes_length_atIndex_(scalar_bytes, scalar_size, 2)
            enc.setBuffer_offset_atIndex_(out_buf, 0, 3)
            enc.setBytes_length_atIndex_(struct.pack("I", numel), 4, 4)
            enc.dispatchThreadgroups_threadsPerThreadgroup_(
                _MTLSize(groups, 1, 1), _MTLSize(tpg, 1, 1))
            enc.endEncoding()
        else:
            _encode_masked_fill_ctypes(enc, pipeline, a_buf, mask_buf,
                                       scalar_bytes, scalar_size, out_buf,
                                       numel, groups, tpg)

        rt.commit_and_wait(cmd)

    # ------------------------------------------------------------------
    # Dispatch: tril / triu
    # ------------------------------------------------------------------

    cpdef void dispatch_tril_triu(self, str kernel_name, object a_buf,
                                  object out_buf, int rows, int cols,
                                  int diagonal, int numel):
        """Encode tril or triu kernel."""
        from candle._backends.mps.runtime import get_runtime  # pylint: disable=import-error,no-name-in-module
        rt = get_runtime()
        cdef object pipeline = self._get_pipeline(kernel_name)
        cdef int tpg = self._threads_per_group(pipeline)
        cdef int groups = (numel + tpg - 1) // tpg

        cmd = rt.create_command_buffer()
        enc = rt.get_compute_encoder(cmd)

        if _HAS_PYOBJC:
            enc.setComputePipelineState_(pipeline)
            enc.setBuffer_offset_atIndex_(a_buf, 0, 0)
            enc.setBuffer_offset_atIndex_(out_buf, 0, 1)
            enc.setBytes_length_atIndex_(struct.pack("I", rows), 4, 2)
            enc.setBytes_length_atIndex_(struct.pack("I", cols), 4, 3)
            enc.setBytes_length_atIndex_(struct.pack("i", diagonal), 4, 4)
            enc.setBytes_length_atIndex_(struct.pack("I", numel), 4, 5)
            enc.dispatchThreadgroups_threadsPerThreadgroup_(
                _MTLSize(groups, 1, 1), _MTLSize(tpg, 1, 1))
            enc.endEncoding()
        else:
            _encode_tril_triu_ctypes(enc, pipeline, a_buf, out_buf,
                                     rows, cols, diagonal, numel,
                                     groups, tpg)

        rt.commit_and_wait(cmd)

    # ------------------------------------------------------------------
    # Dispatch: index_gather
    # ------------------------------------------------------------------

    cpdef void dispatch_index_gather(self, str kernel_name, object input_buf,
                                     object index_buf, object out_buf,
                                     int outer_size, int idx_size,
                                     int inner_size, int input_dim_size,
                                     int numel):
        """Encode index_gather kernel."""
        from candle._backends.mps.runtime import get_runtime  # pylint: disable=import-error,no-name-in-module
        rt = get_runtime()
        cdef object pipeline = self._get_pipeline(kernel_name)
        cdef int tpg = self._threads_per_group(pipeline)
        cdef int groups = (numel + tpg - 1) // tpg

        cmd = rt.create_command_buffer()
        enc = rt.get_compute_encoder(cmd)

        if _HAS_PYOBJC:
            enc.setComputePipelineState_(pipeline)
            enc.setBuffer_offset_atIndex_(input_buf, 0, 0)
            enc.setBuffer_offset_atIndex_(index_buf, 0, 1)
            enc.setBuffer_offset_atIndex_(out_buf, 0, 2)
            enc.setBytes_length_atIndex_(struct.pack("I", outer_size), 4, 3)
            enc.setBytes_length_atIndex_(struct.pack("I", idx_size), 4, 4)
            enc.setBytes_length_atIndex_(struct.pack("I", inner_size), 4, 5)
            enc.setBytes_length_atIndex_(struct.pack("I", input_dim_size), 4, 6)
            enc.dispatchThreadgroups_threadsPerThreadgroup_(
                _MTLSize(groups, 1, 1), _MTLSize(tpg, 1, 1))
            enc.endEncoding()
        else:
            _encode_index_gather_ctypes(enc, pipeline, input_buf, index_buf,
                                        out_buf, outer_size, idx_size,
                                        inner_size, input_dim_size,
                                        groups, tpg)

        rt.commit_and_wait(cmd)

    # ------------------------------------------------------------------
    # Dispatch: cat_copy
    # ------------------------------------------------------------------

    cpdef void dispatch_cat_copy(self, str kernel_name, object src_buf,
                                 object dst_buf, int outer_size,
                                 int src_dim, int inner_size, int dst_dim,
                                 int offset, int numel):
        """Encode cat_copy kernel."""
        from candle._backends.mps.runtime import get_runtime  # pylint: disable=import-error,no-name-in-module
        rt = get_runtime()
        cdef object pipeline = self._get_pipeline(kernel_name)
        cdef int tpg = self._threads_per_group(pipeline)
        cdef int groups = (numel + tpg - 1) // tpg

        cmd = rt.create_command_buffer()
        enc = rt.get_compute_encoder(cmd)

        if _HAS_PYOBJC:
            enc.setComputePipelineState_(pipeline)
            enc.setBuffer_offset_atIndex_(src_buf, 0, 0)
            enc.setBuffer_offset_atIndex_(dst_buf, 0, 1)
            enc.setBytes_length_atIndex_(struct.pack("I", outer_size), 4, 2)
            enc.setBytes_length_atIndex_(struct.pack("I", src_dim), 4, 3)
            enc.setBytes_length_atIndex_(struct.pack("I", inner_size), 4, 4)
            enc.setBytes_length_atIndex_(struct.pack("I", dst_dim), 4, 5)
            enc.setBytes_length_atIndex_(struct.pack("I", offset), 4, 6)
            enc.dispatchThreadgroups_threadsPerThreadgroup_(
                _MTLSize(groups, 1, 1), _MTLSize(tpg, 1, 1))
            enc.endEncoding()
        else:
            _encode_cat_copy_ctypes(enc, pipeline, src_buf, dst_buf,
                                    outer_size, src_dim, inner_size,
                                    dst_dim, offset, groups, tpg)

        rt.commit_and_wait(cmd)

    # ------------------------------------------------------------------
    # Dispatch: conv2d
    # ------------------------------------------------------------------

    cpdef void dispatch_conv2d(self, str kernel_name, object input_buf,
                               object weight_buf, object bias_buf,
                               object output_buf, int N, int C_in, int H_in,
                               int W_in, int C_out, int kH, int kW,
                               int H_out, int W_out, int sH, int sW,
                               int pH, int pW, int dH, int dW,
                               int has_bias, int numel):
        """Encode conv2d kernel (4 bufs + 17 packed uint params)."""
        from candle._backends.mps.runtime import get_runtime  # pylint: disable=import-error,no-name-in-module
        rt = get_runtime()
        cdef object pipeline = self._get_pipeline(kernel_name)
        cdef int tpg = self._threads_per_group(pipeline)
        cdef int groups = (numel + tpg - 1) // tpg

        params = struct.pack("17I", N, C_in, H_in, W_in, C_out, kH, kW,
                             H_out, W_out, sH, sW, pH, pW, dH, dW,
                             has_bias, numel)

        cmd = rt.create_command_buffer()
        enc = rt.get_compute_encoder(cmd)

        if _HAS_PYOBJC:
            enc.setComputePipelineState_(pipeline)
            enc.setBuffer_offset_atIndex_(input_buf, 0, 0)
            enc.setBuffer_offset_atIndex_(weight_buf, 0, 1)
            enc.setBuffer_offset_atIndex_(bias_buf, 0, 2)
            enc.setBuffer_offset_atIndex_(output_buf, 0, 3)
            enc.setBytes_length_atIndex_(params, len(params), 4)
            enc.dispatchThreadgroups_threadsPerThreadgroup_(
                _MTLSize(groups, 1, 1), _MTLSize(tpg, 1, 1))
            enc.endEncoding()
        else:
            _encode_conv2d_ctypes(enc, pipeline, input_buf, weight_buf,
                                  bias_buf, output_buf, params, groups, tpg)

        rt.commit_and_wait(cmd)

    # ------------------------------------------------------------------
    # Dispatch: layer_norm
    # ------------------------------------------------------------------

    cpdef void dispatch_layer_norm(self, str kernel_name, object input_buf,
                                   object weight_buf, object bias_buf,
                                   object output_buf, int outer_size,
                                   int inner_size, float eps,
                                   int has_weight, int has_bias):
        """Encode layer_norm kernel."""
        from candle._backends.mps.runtime import get_runtime  # pylint: disable=import-error,no-name-in-module
        rt = get_runtime()
        cdef object pipeline = self._get_pipeline(kernel_name)
        cdef int tpg = self._threads_per_group(pipeline)
        cdef int numel = outer_size * inner_size
        cdef int groups = (numel + tpg - 1) // tpg

        cmd = rt.create_command_buffer()
        enc = rt.get_compute_encoder(cmd)

        if _HAS_PYOBJC:
            enc.setComputePipelineState_(pipeline)
            enc.setBuffer_offset_atIndex_(input_buf, 0, 0)
            enc.setBuffer_offset_atIndex_(weight_buf, 0, 1)
            enc.setBuffer_offset_atIndex_(bias_buf, 0, 2)
            enc.setBuffer_offset_atIndex_(output_buf, 0, 3)
            enc.setBytes_length_atIndex_(struct.pack("I", outer_size), 4, 4)
            enc.setBytes_length_atIndex_(struct.pack("I", inner_size), 4, 5)
            enc.setBytes_length_atIndex_(struct.pack("f", eps), 4, 6)
            enc.setBytes_length_atIndex_(struct.pack("I", has_weight), 4, 7)
            enc.setBytes_length_atIndex_(struct.pack("I", has_bias), 4, 8)
            enc.dispatchThreadgroups_threadsPerThreadgroup_(
                _MTLSize(groups, 1, 1), _MTLSize(tpg, 1, 1))
            enc.endEncoding()
        else:
            _encode_layer_norm_ctypes(enc, pipeline, input_buf, weight_buf,
                                      bias_buf, output_buf, outer_size,
                                      inner_size, eps, has_weight, has_bias,
                                      groups, tpg)

        rt.commit_and_wait(cmd)

    # ------------------------------------------------------------------
    # Dispatch: rms_norm
    # ------------------------------------------------------------------

    cpdef void dispatch_rms_norm(self, str kernel_name, object input_buf,
                                 object weight_buf, object output_buf,
                                 int outer_size, int inner_size, float eps,
                                 int has_weight):
        """Encode rms_norm kernel."""
        from candle._backends.mps.runtime import get_runtime  # pylint: disable=import-error,no-name-in-module
        rt = get_runtime()
        cdef object pipeline = self._get_pipeline(kernel_name)
        cdef int tpg = self._threads_per_group(pipeline)
        cdef int numel = outer_size * inner_size
        cdef int groups = (numel + tpg - 1) // tpg

        cmd = rt.create_command_buffer()
        enc = rt.get_compute_encoder(cmd)

        if _HAS_PYOBJC:
            enc.setComputePipelineState_(pipeline)
            enc.setBuffer_offset_atIndex_(input_buf, 0, 0)
            enc.setBuffer_offset_atIndex_(weight_buf, 0, 1)
            enc.setBuffer_offset_atIndex_(output_buf, 0, 2)
            enc.setBytes_length_atIndex_(struct.pack("I", outer_size), 4, 3)
            enc.setBytes_length_atIndex_(struct.pack("I", inner_size), 4, 4)
            enc.setBytes_length_atIndex_(struct.pack("f", eps), 4, 5)
            enc.setBytes_length_atIndex_(struct.pack("I", has_weight), 4, 6)
            enc.dispatchThreadgroups_threadsPerThreadgroup_(
                _MTLSize(groups, 1, 1), _MTLSize(tpg, 1, 1))
            enc.endEncoding()
        else:
            _encode_rms_norm_ctypes(enc, pipeline, input_buf, weight_buf,
                                    output_buf, outer_size, inner_size,
                                    eps, has_weight, groups, tpg)

        rt.commit_and_wait(cmd)

    # ------------------------------------------------------------------
    # Dispatch: batch_norm_stats
    # ------------------------------------------------------------------

    cpdef void dispatch_batch_norm_stats(self, str kernel_name, object input_buf,
                                         object mean_buf, object var_buf,
                                         int N, int C, int HW):
        """Encode batch_norm_stats kernel."""
        from candle._backends.mps.runtime import get_runtime  # pylint: disable=import-error,no-name-in-module
        rt = get_runtime()
        cdef object pipeline = self._get_pipeline(kernel_name)
        cdef int tpg = self._threads_per_group(pipeline)
        cdef int groups = (C + tpg - 1) // tpg

        cmd = rt.create_command_buffer()
        enc = rt.get_compute_encoder(cmd)

        if _HAS_PYOBJC:
            enc.setComputePipelineState_(pipeline)
            enc.setBuffer_offset_atIndex_(input_buf, 0, 0)
            enc.setBuffer_offset_atIndex_(mean_buf, 0, 1)
            enc.setBuffer_offset_atIndex_(var_buf, 0, 2)
            enc.setBytes_length_atIndex_(struct.pack("I", N), 4, 3)
            enc.setBytes_length_atIndex_(struct.pack("I", C), 4, 4)
            enc.setBytes_length_atIndex_(struct.pack("I", HW), 4, 5)
            enc.dispatchThreadgroups_threadsPerThreadgroup_(
                _MTLSize(groups, 1, 1), _MTLSize(tpg, 1, 1))
            enc.endEncoding()
        else:
            _encode_batch_norm_stats_ctypes(enc, pipeline, input_buf,
                                            mean_buf, var_buf, N, C, HW,
                                            groups, tpg)

        rt.commit_and_wait(cmd)

    # ------------------------------------------------------------------
    # Dispatch: batch_norm_apply
    # ------------------------------------------------------------------

    cpdef void dispatch_batch_norm_apply(self, str kernel_name, object input_buf,
                                         object mean_buf, object var_buf,
                                         object weight_buf, object bias_buf,
                                         object output_buf,
                                         int C, int spatial_size,
                                         float eps, int has_weight,
                                         int has_bias, int total):
        """Encode batch_norm_apply kernel."""
        from candle._backends.mps.runtime import get_runtime  # pylint: disable=import-error,no-name-in-module
        rt = get_runtime()
        cdef object pipeline = self._get_pipeline(kernel_name)
        cdef int tpg = self._threads_per_group(pipeline)
        cdef int groups = (total + tpg - 1) // tpg

        cmd = rt.create_command_buffer()
        enc = rt.get_compute_encoder(cmd)

        if _HAS_PYOBJC:
            enc.setComputePipelineState_(pipeline)
            enc.setBuffer_offset_atIndex_(input_buf, 0, 0)
            enc.setBuffer_offset_atIndex_(mean_buf, 0, 1)
            enc.setBuffer_offset_atIndex_(var_buf, 0, 2)
            enc.setBuffer_offset_atIndex_(weight_buf, 0, 3)
            enc.setBuffer_offset_atIndex_(bias_buf, 0, 4)
            enc.setBuffer_offset_atIndex_(output_buf, 0, 5)
            enc.setBytes_length_atIndex_(struct.pack("I", C), 4, 6)
            enc.setBytes_length_atIndex_(struct.pack("I", spatial_size), 4, 7)
            enc.setBytes_length_atIndex_(struct.pack("f", eps), 4, 8)
            enc.setBytes_length_atIndex_(struct.pack("I", has_weight), 4, 9)
            enc.setBytes_length_atIndex_(struct.pack("I", has_bias), 4, 10)
            enc.setBytes_length_atIndex_(struct.pack("I", total), 4, 11)
            enc.dispatchThreadgroups_threadsPerThreadgroup_(
                _MTLSize(groups, 1, 1), _MTLSize(tpg, 1, 1))
            enc.endEncoding()
        else:
            _encode_batch_norm_apply_ctypes(enc, pipeline, input_buf,
                                            mean_buf, var_buf, weight_buf,
                                            bias_buf, output_buf,
                                            C, spatial_size, eps,
                                            has_weight, has_bias, total,
                                            groups, tpg)

        rt.commit_and_wait(cmd)

    # ------------------------------------------------------------------
    # Dispatch: group_norm
    # ------------------------------------------------------------------

    cpdef void dispatch_group_norm(self, str kernel_name, object input_buf,
                                   object weight_buf, object bias_buf,
                                   object output_buf, int N, int G, int C,
                                   int HW, float eps, int has_weight,
                                   int has_bias):
        """Encode group_norm kernel."""
        from candle._backends.mps.runtime import get_runtime  # pylint: disable=import-error,no-name-in-module
        rt = get_runtime()
        cdef object pipeline = self._get_pipeline(kernel_name)
        cdef int tpg = self._threads_per_group(pipeline)
        cdef int groups = N * G

        cmd = rt.create_command_buffer()
        enc = rt.get_compute_encoder(cmd)

        if _HAS_PYOBJC:
            enc.setComputePipelineState_(pipeline)
            enc.setBuffer_offset_atIndex_(input_buf, 0, 0)
            enc.setBuffer_offset_atIndex_(weight_buf, 0, 1)
            enc.setBuffer_offset_atIndex_(bias_buf, 0, 2)
            enc.setBuffer_offset_atIndex_(output_buf, 0, 3)
            enc.setBytes_length_atIndex_(struct.pack("I", N), 4, 4)
            enc.setBytes_length_atIndex_(struct.pack("I", G), 4, 5)
            enc.setBytes_length_atIndex_(struct.pack("I", C), 4, 6)
            enc.setBytes_length_atIndex_(struct.pack("I", HW), 4, 7)
            enc.setBytes_length_atIndex_(struct.pack("f", eps), 4, 8)
            enc.setBytes_length_atIndex_(struct.pack("I", has_weight), 4, 9)
            enc.setBytes_length_atIndex_(struct.pack("I", has_bias), 4, 10)
            enc.dispatchThreadgroups_threadsPerThreadgroup_(
                _MTLSize(groups, 1, 1), _MTLSize(tpg, 1, 1))
            enc.endEncoding()
        else:
            _encode_group_norm_ctypes(enc, pipeline, input_buf, weight_buf,
                                      bias_buf, output_buf, N, G, C, HW,
                                      eps, has_weight, has_bias, groups, tpg)

        rt.commit_and_wait(cmd)

    # ------------------------------------------------------------------
    # Dispatch: pool2d
    # ------------------------------------------------------------------

    cpdef void dispatch_pool2d(self, str kernel_name, object input_buf,
                               object output_buf, bytes params_packed,
                               int numel):
        """Encode pool2d kernel (2 bufs + packed params)."""
        from candle._backends.mps.runtime import get_runtime  # pylint: disable=import-error,no-name-in-module
        rt = get_runtime()
        cdef object pipeline = self._get_pipeline(kernel_name)
        cdef int tpg = self._threads_per_group(pipeline)
        cdef int groups = (numel + tpg - 1) // tpg

        cmd = rt.create_command_buffer()
        enc = rt.get_compute_encoder(cmd)

        if _HAS_PYOBJC:
            enc.setComputePipelineState_(pipeline)
            enc.setBuffer_offset_atIndex_(input_buf, 0, 0)
            enc.setBuffer_offset_atIndex_(output_buf, 0, 1)
            enc.setBytes_length_atIndex_(params_packed, len(params_packed), 2)
            enc.dispatchThreadgroups_threadsPerThreadgroup_(
                _MTLSize(groups, 1, 1), _MTLSize(tpg, 1, 1))
            enc.endEncoding()
        else:
            _encode_pool2d_ctypes(enc, pipeline, input_buf, output_buf,
                                  params_packed, groups, tpg)

        rt.commit_and_wait(cmd)

    # ------------------------------------------------------------------
    # Dispatch: cumsum
    # ------------------------------------------------------------------

    cpdef void dispatch_cumsum(self, str kernel_name, object input_buf,
                               object output_buf, int outer_size,
                               int dim_size, int inner_size):
        """Encode cumsum kernel."""
        from candle._backends.mps.runtime import get_runtime  # pylint: disable=import-error,no-name-in-module
        rt = get_runtime()
        cdef object pipeline = self._get_pipeline(kernel_name)
        cdef int tpg = self._threads_per_group(pipeline)
        cdef int numel = outer_size * inner_size
        cdef int groups = (numel + tpg - 1) // tpg

        cmd = rt.create_command_buffer()
        enc = rt.get_compute_encoder(cmd)

        if _HAS_PYOBJC:
            enc.setComputePipelineState_(pipeline)
            enc.setBuffer_offset_atIndex_(input_buf, 0, 0)
            enc.setBuffer_offset_atIndex_(output_buf, 0, 1)
            enc.setBytes_length_atIndex_(struct.pack("I", outer_size), 4, 2)
            enc.setBytes_length_atIndex_(struct.pack("I", dim_size), 4, 3)
            enc.setBytes_length_atIndex_(struct.pack("I", inner_size), 4, 4)
            enc.dispatchThreadgroups_threadsPerThreadgroup_(
                _MTLSize(groups, 1, 1), _MTLSize(tpg, 1, 1))
            enc.endEncoding()
        else:
            _encode_cumsum_ctypes(enc, pipeline, input_buf, output_buf,
                                  outer_size, dim_size, inner_size,
                                  groups, tpg)

        rt.commit_and_wait(cmd)

    # ------------------------------------------------------------------
    # Dispatch: cummax / cummin
    # ------------------------------------------------------------------

    cpdef void dispatch_cummax(self, str kernel_name, object input_buf,
                               object values_buf, object indices_buf,
                               int outer_size, int dim_size, int inner_size):
        """Encode cummax kernel."""
        from candle._backends.mps.runtime import get_runtime  # pylint: disable=import-error,no-name-in-module
        rt = get_runtime()
        cdef object pipeline = self._get_pipeline(kernel_name)
        cdef int tpg = self._threads_per_group(pipeline)
        cdef int numel = outer_size * inner_size
        cdef int groups = (numel + tpg - 1) // tpg

        cmd = rt.create_command_buffer()
        enc = rt.get_compute_encoder(cmd)

        if _HAS_PYOBJC:
            enc.setComputePipelineState_(pipeline)
            enc.setBuffer_offset_atIndex_(input_buf, 0, 0)
            enc.setBuffer_offset_atIndex_(values_buf, 0, 1)
            enc.setBuffer_offset_atIndex_(indices_buf, 0, 2)
            enc.setBytes_length_atIndex_(struct.pack("I", outer_size), 4, 3)
            enc.setBytes_length_atIndex_(struct.pack("I", dim_size), 4, 4)
            enc.setBytes_length_atIndex_(struct.pack("I", inner_size), 4, 5)
            enc.dispatchThreadgroups_threadsPerThreadgroup_(
                _MTLSize(groups, 1, 1), _MTLSize(tpg, 1, 1))
            enc.endEncoding()
        else:
            _encode_cumextreme_ctypes(enc, pipeline, input_buf, values_buf,
                                      indices_buf, outer_size, dim_size,
                                      inner_size, groups, tpg)

        rt.commit_and_wait(cmd)

    # cummin uses the same kernel signature as cummax
    cpdef void dispatch_cummin(self, str kernel_name, object input_buf,
                               object values_buf, object indices_buf,
                               int outer_size, int dim_size, int inner_size):
        """Encode cummin kernel."""
        self.dispatch_cummax(kernel_name, input_buf, values_buf, indices_buf,
                             outer_size, dim_size, inner_size)

    # ------------------------------------------------------------------
    # Dispatch: sort
    # ------------------------------------------------------------------

    cpdef void dispatch_sort(self, str kernel_name, object input_buf,
                             object values_buf, object indices_buf,
                             int outer_size, int dim_size, int inner_size,
                             bint descending):
        """Encode sort kernel."""
        from candle._backends.mps.runtime import get_runtime  # pylint: disable=import-error,no-name-in-module
        rt = get_runtime()
        cdef object pipeline = self._get_pipeline(kernel_name)
        cdef int tpg = self._threads_per_group(pipeline)
        cdef int numel = outer_size * inner_size
        cdef int groups = (numel + tpg - 1) // tpg

        cmd = rt.create_command_buffer()
        enc = rt.get_compute_encoder(cmd)

        if _HAS_PYOBJC:
            enc.setComputePipelineState_(pipeline)
            enc.setBuffer_offset_atIndex_(input_buf, 0, 0)
            enc.setBuffer_offset_atIndex_(values_buf, 0, 1)
            enc.setBuffer_offset_atIndex_(indices_buf, 0, 2)
            enc.setBytes_length_atIndex_(struct.pack("I", outer_size), 4, 3)
            enc.setBytes_length_atIndex_(struct.pack("I", dim_size), 4, 4)
            enc.setBytes_length_atIndex_(struct.pack("I", inner_size), 4, 5)
            enc.setBytes_length_atIndex_(struct.pack("I", 1 if descending else 0), 4, 6)
            enc.dispatchThreadgroups_threadsPerThreadgroup_(
                _MTLSize(groups, 1, 1), _MTLSize(tpg, 1, 1))
            enc.endEncoding()
        else:
            _encode_sort_ctypes(enc, pipeline, input_buf, values_buf,
                                indices_buf, outer_size, dim_size,
                                inner_size, descending, groups, tpg)

        rt.commit_and_wait(cmd)

    # ------------------------------------------------------------------
    # Dispatch: adaptive_avg_pool2d
    # ------------------------------------------------------------------

    cpdef void dispatch_adaptive_avg_pool2d(self, str kernel_name,
                                            object input_buf, object output_buf,
                                            int N, int C, int H_in, int W_in,
                                            int oH, int oW, int total):
        """Encode adaptive_avg_pool2d kernel."""
        from candle._backends.mps.runtime import get_runtime  # pylint: disable=import-error,no-name-in-module
        rt = get_runtime()
        cdef object pipeline = self._get_pipeline(kernel_name)
        cdef int tpg = self._threads_per_group(pipeline)
        cdef int groups = (total + tpg - 1) // tpg

        cmd = rt.create_command_buffer()
        enc = rt.get_compute_encoder(cmd)

        if _HAS_PYOBJC:
            enc.setComputePipelineState_(pipeline)
            enc.setBuffer_offset_atIndex_(input_buf, 0, 0)
            enc.setBuffer_offset_atIndex_(output_buf, 0, 1)
            enc.setBytes_length_atIndex_(struct.pack("I", N), 4, 2)
            enc.setBytes_length_atIndex_(struct.pack("I", C), 4, 3)
            enc.setBytes_length_atIndex_(struct.pack("I", H_in), 4, 4)
            enc.setBytes_length_atIndex_(struct.pack("I", W_in), 4, 5)
            enc.setBytes_length_atIndex_(struct.pack("I", oH), 4, 6)
            enc.setBytes_length_atIndex_(struct.pack("I", oW), 4, 7)
            enc.setBytes_length_atIndex_(struct.pack("I", total), 4, 8)
            enc.dispatchThreadgroups_threadsPerThreadgroup_(
                _MTLSize(groups, 1, 1), _MTLSize(tpg, 1, 1))
            enc.endEncoding()
        else:
            _encode_adaptive_avg_pool2d_ctypes(enc, pipeline, input_buf,
                                               output_buf, N, C, H_in, W_in,
                                               oH, oW, total, groups, tpg)

        rt.commit_and_wait(cmd)

    # ------------------------------------------------------------------
    # Dispatch: upsample
    # ------------------------------------------------------------------

    cpdef void dispatch_upsample(self, str kernel_name, object input_buf,
                                 object output_buf, int N, int C,
                                 int H_in, int W_in, int H_out, int W_out,
                                 int extra_uint, int total):
        """Encode upsample kernel (extra_uint is align_corners for bilinear)."""
        from candle._backends.mps.runtime import get_runtime  # pylint: disable=import-error,no-name-in-module
        rt = get_runtime()
        cdef object pipeline = self._get_pipeline(kernel_name)
        cdef int tpg = self._threads_per_group(pipeline)
        cdef int groups = (total + tpg - 1) // tpg

        cmd = rt.create_command_buffer()
        enc = rt.get_compute_encoder(cmd)

        if _HAS_PYOBJC:
            enc.setComputePipelineState_(pipeline)
            enc.setBuffer_offset_atIndex_(input_buf, 0, 0)
            enc.setBuffer_offset_atIndex_(output_buf, 0, 1)
            enc.setBytes_length_atIndex_(struct.pack("I", N), 4, 2)
            enc.setBytes_length_atIndex_(struct.pack("I", C), 4, 3)
            enc.setBytes_length_atIndex_(struct.pack("I", H_in), 4, 4)
            enc.setBytes_length_atIndex_(struct.pack("I", W_in), 4, 5)
            enc.setBytes_length_atIndex_(struct.pack("I", H_out), 4, 6)
            enc.setBytes_length_atIndex_(struct.pack("I", W_out), 4, 7)
            enc.setBytes_length_atIndex_(struct.pack("I", extra_uint), 4, 8)
            enc.setBytes_length_atIndex_(struct.pack("I", total), 4, 9)
            enc.dispatchThreadgroups_threadsPerThreadgroup_(
                _MTLSize(groups, 1, 1), _MTLSize(tpg, 1, 1))
            enc.endEncoding()
        else:
            _encode_upsample_ctypes(enc, pipeline, input_buf, output_buf,
                                    N, C, H_in, W_in, H_out, W_out,
                                    extra_uint, total, groups, tpg)

        rt.commit_and_wait(cmd)

    # ------------------------------------------------------------------
    # Dispatch: pad_constant
    # ------------------------------------------------------------------

    cpdef void dispatch_pad_constant(self, str kernel_name, object input_buf,
                                     object output_buf, bytes in_shape_packed,
                                     bytes pad_before_packed,
                                     bytes out_shape_packed,
                                     object fill_val_bytes, int fill_val_size,
                                     int ndim, int total):
        """Encode pad_constant kernel."""
        from candle._backends.mps.runtime import get_runtime  # pylint: disable=import-error,no-name-in-module
        rt = get_runtime()
        cdef object pipeline = self._get_pipeline(kernel_name)
        cdef int tpg = self._threads_per_group(pipeline)
        cdef int groups = (total + tpg - 1) // tpg

        cmd = rt.create_command_buffer()
        enc = rt.get_compute_encoder(cmd)

        if _HAS_PYOBJC:
            enc.setComputePipelineState_(pipeline)
            enc.setBuffer_offset_atIndex_(input_buf, 0, 0)
            enc.setBuffer_offset_atIndex_(output_buf, 0, 1)
            enc.setBytes_length_atIndex_(in_shape_packed, len(in_shape_packed), 2)
            enc.setBytes_length_atIndex_(pad_before_packed, len(pad_before_packed), 3)
            enc.setBytes_length_atIndex_(out_shape_packed, len(out_shape_packed), 4)
            enc.setBytes_length_atIndex_(fill_val_bytes, fill_val_size, 5)
            enc.setBytes_length_atIndex_(struct.pack("I", ndim), 4, 6)
            enc.setBytes_length_atIndex_(struct.pack("I", total), 4, 7)
            enc.dispatchThreadgroups_threadsPerThreadgroup_(
                _MTLSize(groups, 1, 1), _MTLSize(tpg, 1, 1))
            enc.endEncoding()
        else:
            _encode_pad_constant_ctypes(enc, pipeline, input_buf, output_buf,
                                        in_shape_packed, pad_before_packed,
                                        out_shape_packed, fill_val_bytes,
                                        fill_val_size, ndim, total, groups, tpg)

        rt.commit_and_wait(cmd)

    # ------------------------------------------------------------------
    # Dispatch: flip
    # ------------------------------------------------------------------

    cpdef void dispatch_flip(self, str kernel_name, object input_buf,
                             object output_buf, bytes shape_packed,
                             bytes flip_mask_packed, int ndim, int total):
        """Encode flip kernel."""
        from candle._backends.mps.runtime import get_runtime  # pylint: disable=import-error,no-name-in-module
        rt = get_runtime()
        cdef object pipeline = self._get_pipeline(kernel_name)
        cdef int tpg = self._threads_per_group(pipeline)
        cdef int groups = (total + tpg - 1) // tpg

        cmd = rt.create_command_buffer()
        enc = rt.get_compute_encoder(cmd)

        if _HAS_PYOBJC:
            enc.setComputePipelineState_(pipeline)
            enc.setBuffer_offset_atIndex_(input_buf, 0, 0)
            enc.setBuffer_offset_atIndex_(output_buf, 0, 1)
            enc.setBytes_length_atIndex_(shape_packed, len(shape_packed), 2)
            enc.setBytes_length_atIndex_(flip_mask_packed, len(flip_mask_packed), 3)
            enc.setBytes_length_atIndex_(struct.pack("I", ndim), 4, 4)
            enc.setBytes_length_atIndex_(struct.pack("I", total), 4, 5)
            enc.dispatchThreadgroups_threadsPerThreadgroup_(
                _MTLSize(groups, 1, 1), _MTLSize(tpg, 1, 1))
            enc.endEncoding()
        else:
            _encode_flip_ctypes(enc, pipeline, input_buf, output_buf,
                                shape_packed, flip_mask_packed, ndim, total,
                                groups, tpg)

        rt.commit_and_wait(cmd)

    # ------------------------------------------------------------------
    # Dispatch: roll
    # ------------------------------------------------------------------

    cpdef void dispatch_roll(self, str kernel_name, object input_buf,
                             object output_buf, bytes shape_packed,
                             bytes shifts_packed, int ndim, int total):
        """Encode roll kernel."""
        from candle._backends.mps.runtime import get_runtime  # pylint: disable=import-error,no-name-in-module
        rt = get_runtime()
        cdef object pipeline = self._get_pipeline(kernel_name)
        cdef int tpg = self._threads_per_group(pipeline)
        cdef int groups = (total + tpg - 1) // tpg

        cmd = rt.create_command_buffer()
        enc = rt.get_compute_encoder(cmd)

        if _HAS_PYOBJC:
            enc.setComputePipelineState_(pipeline)
            enc.setBuffer_offset_atIndex_(input_buf, 0, 0)
            enc.setBuffer_offset_atIndex_(output_buf, 0, 1)
            enc.setBytes_length_atIndex_(shape_packed, len(shape_packed), 2)
            enc.setBytes_length_atIndex_(shifts_packed, len(shifts_packed), 3)
            enc.setBytes_length_atIndex_(struct.pack("I", ndim), 4, 4)
            enc.setBytes_length_atIndex_(struct.pack("I", total), 4, 5)
            enc.dispatchThreadgroups_threadsPerThreadgroup_(
                _MTLSize(groups, 1, 1), _MTLSize(tpg, 1, 1))
            enc.endEncoding()
        else:
            _encode_roll_ctypes(enc, pipeline, input_buf, output_buf,
                                shape_packed, shifts_packed, ndim, total,
                                groups, tpg)

        rt.commit_and_wait(cmd)

    # ------------------------------------------------------------------
    # Dispatch: Philox fill  (uniform / normal)
    # ------------------------------------------------------------------

    cpdef void dispatch_philox_fill(self, str kernel_name, object out_buf,
                                    unsigned int seed_lo, unsigned int seed_hi,
                                    unsigned int offset,
                                    object param1, object param2, int numel,
                                    str param_fmt="f"):
        """Encode Philox fill (uniform/normal) kernel."""
        from candle._backends.mps.runtime import get_runtime  # pylint: disable=import-error,no-name-in-module
        rt = get_runtime()
        cdef object pipeline = self._get_pipeline(kernel_name)
        cdef int tpg = self._threads_per_group(pipeline)
        cdef int threads = (numel + 3) // 4
        cdef int groups = (threads + tpg - 1) // tpg
        cdef bytes p1_bytes = struct.pack(param_fmt, param1)
        cdef bytes p2_bytes = struct.pack(param_fmt, param2)
        cdef int p_size = len(p1_bytes)

        cmd = rt.create_command_buffer()
        enc = rt.get_compute_encoder(cmd)

        if _HAS_PYOBJC:
            enc.setComputePipelineState_(pipeline)
            enc.setBuffer_offset_atIndex_(out_buf, 0, 0)
            enc.setBytes_length_atIndex_(struct.pack("I", seed_lo), 4, 1)
            enc.setBytes_length_atIndex_(struct.pack("I", seed_hi), 4, 2)
            enc.setBytes_length_atIndex_(struct.pack("I", offset), 4, 3)
            enc.setBytes_length_atIndex_(p1_bytes, p_size, 4)
            enc.setBytes_length_atIndex_(p2_bytes, p_size, 5)
            enc.setBytes_length_atIndex_(struct.pack("I", numel), 4, 6)
            enc.dispatchThreadgroups_threadsPerThreadgroup_(
                _MTLSize(groups, 1, 1), _MTLSize(tpg, 1, 1))
            enc.endEncoding()
        else:
            _encode_philox_fill_ctypes(enc, pipeline, out_buf, seed_lo,
                                       seed_hi, offset, p1_bytes, p2_bytes,
                                       p_size, numel, groups, tpg)

        rt.commit_and_wait(cmd)

    # ------------------------------------------------------------------
    # Dispatch: Philox bernoulli
    # ------------------------------------------------------------------

    cpdef void dispatch_philox_bernoulli(self, str kernel_name, object out_buf,
                                         float prob, unsigned int seed_lo,
                                         unsigned int seed_hi,
                                         unsigned int offset, int numel):
        """Encode Philox bernoulli kernel."""
        from candle._backends.mps.runtime import get_runtime  # pylint: disable=import-error,no-name-in-module
        rt = get_runtime()
        cdef object pipeline = self._get_pipeline(kernel_name)
        cdef int tpg = self._threads_per_group(pipeline)
        cdef int threads = (numel + 3) // 4
        cdef int groups = (threads + tpg - 1) // tpg

        cmd = rt.create_command_buffer()
        enc = rt.get_compute_encoder(cmd)

        if _HAS_PYOBJC:
            enc.setComputePipelineState_(pipeline)
            enc.setBuffer_offset_atIndex_(out_buf, 0, 0)
            enc.setBytes_length_atIndex_(struct.pack("f", prob), 4, 1)
            enc.setBytes_length_atIndex_(struct.pack("I", seed_lo), 4, 2)
            enc.setBytes_length_atIndex_(struct.pack("I", seed_hi), 4, 3)
            enc.setBytes_length_atIndex_(struct.pack("I", offset), 4, 4)
            enc.setBytes_length_atIndex_(struct.pack("I", numel), 4, 5)
            enc.dispatchThreadgroups_threadsPerThreadgroup_(
                _MTLSize(groups, 1, 1), _MTLSize(tpg, 1, 1))
            enc.endEncoding()
        else:
            _encode_philox_bernoulli_ctypes(enc, pipeline, out_buf, prob,
                                            seed_lo, seed_hi, offset,
                                            numel, groups, tpg)

        rt.commit_and_wait(cmd)

    # ------------------------------------------------------------------
    # Dispatch: Philox randint
    # ------------------------------------------------------------------

    cpdef void dispatch_philox_randint(self, str kernel_name, object out_buf,
                                       object low, object high,
                                       unsigned int seed_lo, unsigned int seed_hi,
                                       unsigned int offset,
                                       int numel, str int_fmt="i"):
        """Encode Philox randint kernel."""
        from candle._backends.mps.runtime import get_runtime  # pylint: disable=import-error,no-name-in-module
        rt = get_runtime()
        cdef object pipeline = self._get_pipeline(kernel_name)
        cdef int tpg = self._threads_per_group(pipeline)
        cdef int threads = (numel + 3) // 4
        cdef int groups = (threads + tpg - 1) // tpg

        cdef bytes lo_bytes = struct.pack(int_fmt, low)
        cdef bytes hi_bytes = struct.pack(int_fmt, high)
        cdef int i_size = len(lo_bytes)

        cmd = rt.create_command_buffer()
        enc = rt.get_compute_encoder(cmd)

        if _HAS_PYOBJC:
            enc.setComputePipelineState_(pipeline)
            enc.setBuffer_offset_atIndex_(out_buf, 0, 0)
            enc.setBytes_length_atIndex_(lo_bytes, i_size, 1)
            enc.setBytes_length_atIndex_(hi_bytes, i_size, 2)
            enc.setBytes_length_atIndex_(struct.pack("I", seed_lo), 4, 3)
            enc.setBytes_length_atIndex_(struct.pack("I", seed_hi), 4, 4)
            enc.setBytes_length_atIndex_(struct.pack("I", offset), 4, 5)
            enc.setBytes_length_atIndex_(struct.pack("I", numel), 4, 6)
            enc.dispatchThreadgroups_threadsPerThreadgroup_(
                _MTLSize(groups, 1, 1), _MTLSize(tpg, 1, 1))
            enc.endEncoding()
        else:
            _encode_philox_randint_ctypes(enc, pipeline, out_buf,
                                          lo_bytes, hi_bytes, i_size,
                                          seed_lo, seed_hi, offset,
                                          numel, groups, tpg)

        rt.commit_and_wait(cmd)

    # ------------------------------------------------------------------
    # Dispatch: Philox dropout
    # ------------------------------------------------------------------

    cpdef void dispatch_philox_dropout(self, str kernel_name, object a_buf,
                                       object out_buf, float prob, float scale,
                                       unsigned int seed_lo, unsigned int seed_hi,
                                       unsigned int offset,
                                       int numel):
        """Encode fused Philox dropout kernel."""
        from candle._backends.mps.runtime import get_runtime  # pylint: disable=import-error,no-name-in-module
        rt = get_runtime()
        cdef object pipeline = self._get_pipeline(kernel_name)
        cdef int tpg = self._threads_per_group(pipeline)
        cdef int threads = (numel + 3) // 4
        cdef int groups = (threads + tpg - 1) // tpg

        cmd = rt.create_command_buffer()
        enc = rt.get_compute_encoder(cmd)

        if _HAS_PYOBJC:
            enc.setComputePipelineState_(pipeline)
            enc.setBuffer_offset_atIndex_(a_buf, 0, 0)
            enc.setBuffer_offset_atIndex_(out_buf, 0, 1)
            enc.setBytes_length_atIndex_(struct.pack("f", prob), 4, 2)
            enc.setBytes_length_atIndex_(struct.pack("f", scale), 4, 3)
            enc.setBytes_length_atIndex_(struct.pack("I", seed_lo), 4, 4)
            enc.setBytes_length_atIndex_(struct.pack("I", seed_hi), 4, 5)
            enc.setBytes_length_atIndex_(struct.pack("I", offset), 4, 6)
            enc.setBytes_length_atIndex_(struct.pack("I", numel), 4, 7)
            enc.dispatchThreadgroups_threadsPerThreadgroup_(
                _MTLSize(groups, 1, 1), _MTLSize(tpg, 1, 1))
            enc.endEncoding()
        else:
            _encode_philox_dropout_ctypes(enc, pipeline, a_buf, out_buf,
                                          prob, scale, seed_lo, seed_hi,
                                          offset, numel, groups, tpg)

        rt.commit_and_wait(cmd)

    # ------------------------------------------------------------------
    # Dispatch: reduce_dim
    # ------------------------------------------------------------------

    cpdef void dispatch_reduce_dim(self, str kernel_name, object a_buf,
                                   object out_buf, int outer_size,
                                   int reduce_size, int inner_size,
                                   int out_numel):
        """Dispatch an axis-reduce kernel over one dimension."""
        from candle._backends.mps.runtime import get_runtime  # pylint: disable=import-error,no-name-in-module
        rt = get_runtime()
        cdef object pipeline = self._get_pipeline(kernel_name)
        cdef int tpg = self._threads_per_group(pipeline)
        cdef int groups = (out_numel + tpg - 1) // tpg

        cmd = rt.create_command_buffer()
        enc = rt.get_compute_encoder(cmd)

        if _HAS_PYOBJC:
            enc.setComputePipelineState_(pipeline)
            enc.setBuffer_offset_atIndex_(a_buf, 0, 0)
            enc.setBuffer_offset_atIndex_(out_buf, 0, 1)
            enc.setBytes_length_atIndex_(struct.pack("I", outer_size), 4, 2)
            enc.setBytes_length_atIndex_(struct.pack("I", reduce_size), 4, 3)
            enc.setBytes_length_atIndex_(struct.pack("I", inner_size), 4, 4)
            enc.dispatchThreadgroups_threadsPerThreadgroup_(
                _MTLSize(groups, 1, 1), _MTLSize(tpg, 1, 1))
            enc.endEncoding()
        else:
            _encode_reduce_dim_ctypes(enc, pipeline, a_buf, out_buf,
                                      outer_size, reduce_size, inner_size,
                                      groups, tpg)

        rt.commit_and_wait(cmd)

    # ------------------------------------------------------------------
    # Dispatch: softmax_2d
    # ------------------------------------------------------------------

    cpdef void dispatch_softmax_2d(self, str kernel_name, object a_buf,
                                   object out_buf, int rows, int cols):
        """Dispatch softmax over last dim of a 2D tensor."""
        from candle._backends.mps.runtime import get_runtime  # pylint: disable=import-error,no-name-in-module
        rt = get_runtime()
        cdef object pipeline = self._get_pipeline(kernel_name)

        cmd = rt.create_command_buffer()
        enc = rt.get_compute_encoder(cmd)

        if _HAS_PYOBJC:
            enc.setComputePipelineState_(pipeline)
            enc.setBuffer_offset_atIndex_(a_buf, 0, 0)
            enc.setBuffer_offset_atIndex_(out_buf, 0, 1)
            enc.setBytes_length_atIndex_(struct.pack("I", rows), 4, 2)
            enc.setBytes_length_atIndex_(struct.pack("I", cols), 4, 3)
            tpg_x = min(32, cols)
            tpg_y = min(8, rows)
            groups_x = (cols + tpg_x - 1) // tpg_x
            groups_y = (rows + tpg_y - 1) // tpg_y
            enc.dispatchThreadgroups_threadsPerThreadgroup_(
                _MTLSize(groups_x, groups_y, 1), _MTLSize(tpg_x, tpg_y, 1))
            enc.endEncoding()
        else:
            _encode_softmax_ctypes(enc, pipeline, a_buf, out_buf, rows, cols)

        rt.commit_and_wait(cmd)


# ---------------------------------------------------------------------------
# Singleton factory
# ---------------------------------------------------------------------------

cpdef object get_dispatcher():
    """Return the singleton MetalKernelDispatcher (lazy init)."""
    global _singleton
    if _singleton is not None:
        return _singleton
    lock = _get_lock()
    with lock:
        if _singleton is None:
            _singleton = MetalKernelDispatcher()
    return _singleton
