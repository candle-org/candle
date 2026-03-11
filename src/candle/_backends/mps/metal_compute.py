"""Metal GPU compute dispatch engine.

Lazily compiles MSL kernels, caches compute pipeline states, and dispatches
element-wise, reduction, and in-place kernels on Metal GPU.
"""
import ctypes
import struct
import threading

from .runtime import get_runtime, buffer_contents, _HAS_PYOBJC

_MTLSize = None
if _HAS_PYOBJC:
    from Metal import MTLSizeMake as _MTLSizeMake  # pylint: disable=import-error,no-name-in-module

    def _MTLSize(w, h, d):
        return _MTLSizeMake(w, h, d)
else:
    def _MTLSize(w, h, d):  # noqa: E302
        return None

# ---------------------------------------------------------------------------
# Singleton dispatcher
# ---------------------------------------------------------------------------
_dispatcher = None
_dispatcher_lock = threading.Lock()


def get_dispatcher():
    """Return the singleton MetalKernelDispatcher (lazy init)."""
    global _dispatcher
    if _dispatcher is not None:
        return _dispatcher
    with _dispatcher_lock:
        if _dispatcher is None:
            _dispatcher = MetalKernelDispatcher()
        return _dispatcher


class MetalKernelDispatcher:
    """Compiles MSL kernels lazily, caches pipelines, dispatches compute work."""

    def __init__(self):
        self._library = None
        self._pipeline_cache = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Compilation
    # ------------------------------------------------------------------

    def ensure_compiled(self):
        """Compile all MSL source on first use."""
        if self._library is not None:
            return
        with self._lock:
            if self._library is not None:
                return
            from .metal_shaders import MSL_SOURCE
            rt = get_runtime()
            self._library = rt.compile_library(MSL_SOURCE)

    def _get_pipeline(self, kernel_name):
        """Get or create a cached compute pipeline for *kernel_name*."""
        if kernel_name in self._pipeline_cache:
            return self._pipeline_cache[kernel_name]
        self.ensure_compiled()
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

    # ------------------------------------------------------------------
    # Thread dispatch helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _threads_per_group(pipeline):
        if _HAS_PYOBJC:
            return min(256, int(pipeline.maxTotalThreadsPerThreadgroup()))
        return 256  # safe default for Apple Silicon

    # ------------------------------------------------------------------
    # Dispatch: unary  (a → out)
    # ------------------------------------------------------------------

    def dispatch_unary(self, kernel_name, a_metal_buf, out_metal_buf, numel):
        """Encode and execute a unary element-wise kernel."""
        rt = get_runtime()
        pipeline = self._get_pipeline(kernel_name)
        tpg = self._threads_per_group(pipeline)
        groups = (numel + tpg - 1) // tpg

        cmd = rt.create_command_buffer()
        enc = rt.get_compute_encoder(cmd)

        if _HAS_PYOBJC:
            enc.setComputePipelineState_(pipeline)
            enc.setBuffer_offset_atIndex_(a_metal_buf, 0, 0)
            enc.setBuffer_offset_atIndex_(out_metal_buf, 0, 1)
            n_bytes = struct.pack("I", numel)
            enc.setBytes_length_atIndex_(n_bytes, 4, 2)
            enc.dispatchThreadgroups_threadsPerThreadgroup_(
                _MTLSize(groups, 1, 1), _MTLSize(tpg, 1, 1))
            enc.endEncoding()
        else:
            _encode_unary_ctypes(enc, pipeline, a_metal_buf, out_metal_buf,
                                 numel, groups, tpg)

        rt.commit_and_wait(cmd)

    # ------------------------------------------------------------------
    # Dispatch: binary  (a, b → out)
    # ------------------------------------------------------------------

    def dispatch_binary(self, kernel_name, a_buf, b_buf, out_buf, numel):
        """Encode and execute a binary element-wise kernel."""
        rt = get_runtime()
        pipeline = self._get_pipeline(kernel_name)
        tpg = self._threads_per_group(pipeline)
        groups = (numel + tpg - 1) // tpg

        cmd = rt.create_command_buffer()
        enc = rt.get_compute_encoder(cmd)

        if _HAS_PYOBJC:
            enc.setComputePipelineState_(pipeline)
            enc.setBuffer_offset_atIndex_(a_buf, 0, 0)
            enc.setBuffer_offset_atIndex_(b_buf, 0, 1)
            enc.setBuffer_offset_atIndex_(out_buf, 0, 2)
            n_bytes = struct.pack("I", numel)
            enc.setBytes_length_atIndex_(n_bytes, 4, 3)
            enc.dispatchThreadgroups_threadsPerThreadgroup_(
                _MTLSize(groups, 1, 1), _MTLSize(tpg, 1, 1))
            enc.endEncoding()
        else:
            _encode_binary_ctypes(enc, pipeline, a_buf, b_buf, out_buf,
                                  numel, groups, tpg)

        rt.commit_and_wait(cmd)

    # ------------------------------------------------------------------
    # Dispatch: binary with scalar  (a, scalar → out)
    # ------------------------------------------------------------------

    def dispatch_binary_scalar(self, kernel_name, a_buf, scalar, out_buf,
                               numel, scalar_fmt="f"):
        """Encode and execute a binary-scalar kernel (scalar embedded via setBytes)."""
        rt = get_runtime()
        pipeline = self._get_pipeline(kernel_name)
        tpg = self._threads_per_group(pipeline)
        groups = (numel + tpg - 1) // tpg

        cmd = rt.create_command_buffer()
        enc = rt.get_compute_encoder(cmd)

        scalar_bytes = struct.pack(scalar_fmt, scalar)
        scalar_size = len(scalar_bytes)

        if _HAS_PYOBJC:
            enc.setComputePipelineState_(pipeline)
            enc.setBuffer_offset_atIndex_(a_buf, 0, 0)
            enc.setBytes_length_atIndex_(scalar_bytes, scalar_size, 1)
            enc.setBuffer_offset_atIndex_(out_buf, 0, 2)
            n_bytes = struct.pack("I", numel)
            enc.setBytes_length_atIndex_(n_bytes, 4, 3)
            enc.dispatchThreadgroups_threadsPerThreadgroup_(
                _MTLSize(groups, 1, 1), _MTLSize(tpg, 1, 1))
            enc.endEncoding()
        else:
            _encode_binary_scalar_ctypes(enc, pipeline, a_buf,
                                         scalar_bytes, scalar_size,
                                         out_buf, numel, groups, tpg)

        rt.commit_and_wait(cmd)

    # ------------------------------------------------------------------
    # Dispatch: reduction  (input → scalar output, two-pass)
    # ------------------------------------------------------------------

    def dispatch_reduction(self, partial_kernel, final_kernel, a_buf, out_buf,
                           numel):
        """Two-pass parallel reduction: per-threadgroup partials → final."""
        rt = get_runtime()
        pipeline_p = self._get_pipeline(partial_kernel)
        pipeline_f = self._get_pipeline(final_kernel)
        tpg = self._threads_per_group(pipeline_p)
        num_groups = (numel + tpg - 1) // tpg

        # Allocate partial-results buffer (one float per group)
        partials_buf = rt.create_buffer(num_groups * 4)

        # --- Pass 1: per-threadgroup partial ---
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

        # --- Pass 2: reduce partials → single output ---
        final_tpg = self._threads_per_group(pipeline_f)
        # Final pass uses a single threadgroup
        final_tpg = max(final_tpg, num_groups)
        # Round up to next power-of-2 for proper tree reduction
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
    # Dispatch: argmax/argmin reduction  (two-pass with value+index)
    # ------------------------------------------------------------------

    def dispatch_arg_reduction(self, partial_kernel, final_kernel,
                               a_buf, out_buf, numel):
        """Two-pass argmax/argmin: partials carry (value, index) pairs."""
        rt = get_runtime()
        pipeline_p = self._get_pipeline(partial_kernel)
        pipeline_f = self._get_pipeline(final_kernel)
        tpg = self._threads_per_group(pipeline_p)
        num_groups = (numel + tpg - 1) // tpg

        partial_vals_buf = rt.create_buffer(num_groups * 4)   # float per group
        partial_idxs_buf = rt.create_buffer(num_groups * 4)   # uint per group

        # --- Pass 1 ---
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

        # --- Pass 2 ---
        final_tpg = 1
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
    # Dispatch: in-place unary  (a → a)
    # ------------------------------------------------------------------

    def dispatch_inplace_unary(self, kernel_name, a_buf, numel):
        """In-place unary: writes output back to input buffer."""
        rt = get_runtime()
        pipeline = self._get_pipeline(kernel_name)
        tpg = self._threads_per_group(pipeline)
        groups = (numel + tpg - 1) // tpg

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
    # Dispatch: in-place binary  (a, b → a)
    # ------------------------------------------------------------------

    def dispatch_inplace_binary(self, kernel_name, a_buf, b_buf, numel):
        """In-place binary: a[i] op= b[i]."""
        rt = get_runtime()
        pipeline = self._get_pipeline(kernel_name)
        tpg = self._threads_per_group(pipeline)
        groups = (numel + tpg - 1) // tpg

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
    # Dispatch: in-place binary scalar  (a, scalar → a)
    # ------------------------------------------------------------------

    def dispatch_inplace_binary_scalar(self, kernel_name, a_buf, scalar,
                                       numel, scalar_fmt="f"):
        """In-place binary-scalar: a[i] op= scalar."""
        rt = get_runtime()
        pipeline = self._get_pipeline(kernel_name)
        tpg = self._threads_per_group(pipeline)
        groups = (numel + tpg - 1) // tpg

        cmd = rt.create_command_buffer()
        enc = rt.get_compute_encoder(cmd)

        scalar_bytes = struct.pack(scalar_fmt, scalar)
        scalar_size = len(scalar_bytes)

        if _HAS_PYOBJC:
            enc.setComputePipelineState_(pipeline)
            enc.setBuffer_offset_atIndex_(a_buf, 0, 0)
            enc.setBytes_length_atIndex_(scalar_bytes, scalar_size, 1)
            enc.setBytes_length_atIndex_(struct.pack("I", numel), 4, 2)
            enc.dispatchThreadgroups_threadsPerThreadgroup_(
                _MTLSize(groups, 1, 1), _MTLSize(tpg, 1, 1))
            enc.endEncoding()
        else:
            _encode_inplace_scalar_ctypes(enc, pipeline, a_buf,
                                          scalar_bytes, scalar_size,
                                          numel, groups, tpg)

        rt.commit_and_wait(cmd)

    # ------------------------------------------------------------------
    # Dispatch: fill  (scalar → buffer)
    # ------------------------------------------------------------------

    def dispatch_fill(self, kernel_name, out_buf, scalar, numel,
                      scalar_fmt="f"):
        """Fill buffer with a scalar value."""
        rt = get_runtime()
        pipeline = self._get_pipeline(kernel_name)
        tpg = self._threads_per_group(pipeline)
        groups = (numel + tpg - 1) // tpg

        cmd = rt.create_command_buffer()
        enc = rt.get_compute_encoder(cmd)

        scalar_bytes = struct.pack(scalar_fmt, scalar)
        scalar_size = len(scalar_bytes)

        if _HAS_PYOBJC:
            enc.setComputePipelineState_(pipeline)
            enc.setBuffer_offset_atIndex_(out_buf, 0, 0)
            enc.setBytes_length_atIndex_(scalar_bytes, scalar_size, 1)
            enc.setBytes_length_atIndex_(struct.pack("I", numel), 4, 2)
            enc.dispatchThreadgroups_threadsPerThreadgroup_(
                _MTLSize(groups, 1, 1), _MTLSize(tpg, 1, 1))
            enc.endEncoding()
        else:
            _encode_inplace_scalar_ctypes(enc, pipeline, out_buf,
                                          scalar_bytes, scalar_size,
                                          numel, groups, tpg)

        rt.commit_and_wait(cmd)

    # ------------------------------------------------------------------
    # Dispatch: copy  (src → dst)
    # ------------------------------------------------------------------

    def dispatch_copy(self, kernel_name, src_buf, dst_buf, numel):
        """Copy src buffer to dst buffer."""
        self.dispatch_binary(kernel_name, src_buf, dst_buf, numel)

    # ------------------------------------------------------------------
    # Dispatch: unary strided  (a → out, with shape/strides)
    # ------------------------------------------------------------------

    def dispatch_unary_strided(self, kernel_name, a_metal_buf, out_metal_buf,
                               numel, shape_array, strides_a_array, ndim):
        """Encode and execute a strided unary element-wise kernel."""
        rt = get_runtime()
        pipeline = self._get_pipeline(kernel_name)
        tpg = self._threads_per_group(pipeline)
        groups = (numel + tpg - 1) // tpg

        cmd = rt.create_command_buffer()
        enc = rt.get_compute_encoder(cmd)

        shape_bytes = struct.pack(f"{ndim}I", *shape_array)
        strides_bytes = struct.pack(f"{ndim}i", *strides_a_array)

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
                                         out_metal_buf, numel,
                                         shape_bytes, strides_bytes, ndim,
                                         groups, tpg)

        rt.commit_and_wait(cmd)

    # ------------------------------------------------------------------
    # Dispatch: binary strided  (a, b → out, with shape/strides)
    # ------------------------------------------------------------------

    def dispatch_binary_strided(self, kernel_name, a_buf, b_buf, out_buf,
                                numel, shape_array, strides_a_array,
                                strides_b_array, ndim):
        """Encode and execute a strided binary element-wise kernel."""
        rt = get_runtime()
        pipeline = self._get_pipeline(kernel_name)
        tpg = self._threads_per_group(pipeline)
        groups = (numel + tpg - 1) // tpg

        cmd = rt.create_command_buffer()
        enc = rt.get_compute_encoder(cmd)

        shape_bytes = struct.pack(f"{ndim}I", *shape_array)
        strides_a_bytes = struct.pack(f"{ndim}i", *strides_a_array)
        strides_b_bytes = struct.pack(f"{ndim}i", *strides_b_array)

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
    # Dispatch: binary scalar strided  (a, scalar → out, with strides)
    # ------------------------------------------------------------------

    def dispatch_binary_scalar_strided(self, kernel_name, a_buf, scalar,
                                       out_buf, numel, shape_array,
                                       strides_a_array, ndim,
                                       scalar_fmt="f"):
        """Encode and execute a strided binary-scalar kernel."""
        rt = get_runtime()
        pipeline = self._get_pipeline(kernel_name)
        tpg = self._threads_per_group(pipeline)
        groups = (numel + tpg - 1) // tpg

        cmd = rt.create_command_buffer()
        enc = rt.get_compute_encoder(cmd)

        scalar_bytes = struct.pack(scalar_fmt, scalar)
        scalar_size = len(scalar_bytes)
        shape_bytes = struct.pack(f"{ndim}I", *shape_array)
        strides_bytes = struct.pack(f"{ndim}i", *strides_a_array)

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
    # Dispatch: comparison  (a, b → uchar out)
    # ------------------------------------------------------------------

    def dispatch_comparison(self, kernel_name, a_buf, b_buf, out_buf, numel):
        """Encode and execute a comparison kernel (typed input, uchar output)."""
        self.dispatch_binary(kernel_name, a_buf, b_buf, out_buf, numel)

    def dispatch_comparison_scalar(self, kernel_name, a_buf, scalar, out_buf,
                                   numel, scalar_fmt="f"):
        """Encode and execute a scalar comparison kernel."""
        self.dispatch_binary_scalar(kernel_name, a_buf, scalar, out_buf,
                                    numel, scalar_fmt=scalar_fmt)

    # ------------------------------------------------------------------
    # Dispatch: clamp with 2 scalars  (a, min, max → out)
    # ------------------------------------------------------------------

    def dispatch_clamp(self, kernel_name, a_buf, scalar1, scalar2, out_buf,
                       numel, scalar_fmt="f"):
        """Encode clamp kernel: out = clamp(a, min_val, max_val)."""
        rt = get_runtime()
        pipeline = self._get_pipeline(kernel_name)
        tpg = self._threads_per_group(pipeline)
        groups = (numel + tpg - 1) // tpg

        cmd = rt.create_command_buffer()
        enc = rt.get_compute_encoder(cmd)

        s1_bytes = struct.pack(scalar_fmt, scalar1)
        s2_bytes = struct.pack(scalar_fmt, scalar2)
        scalar_size = len(s1_bytes)

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
            _encode_clamp_ctypes(enc, pipeline, a_buf,
                                 s1_bytes, s2_bytes, scalar_size,
                                 out_buf, numel, groups, tpg)

        rt.commit_and_wait(cmd)

    def dispatch_clamp_strided(self, kernel_name, a_buf, scalar1, scalar2,
                               out_buf, numel, shape_array, strides_a_array,
                               ndim, scalar_fmt="f"):
        """Encode strided clamp kernel: out = clamp(a[strided], min, max)."""
        rt = get_runtime()
        pipeline = self._get_pipeline(kernel_name)
        tpg = self._threads_per_group(pipeline)
        groups = (numel + tpg - 1) // tpg

        cmd = rt.create_command_buffer()
        enc = rt.get_compute_encoder(cmd)

        s1_bytes = struct.pack(scalar_fmt, scalar1)
        s2_bytes = struct.pack(scalar_fmt, scalar2)
        scalar_size = len(s1_bytes)
        shape_bytes = struct.pack(f"{ndim}I", *shape_array)
        strides_bytes = struct.pack(f"{ndim}i", *strides_a_array)

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
            _encode_clamp_strided_ctypes(
                enc, pipeline, a_buf, s1_bytes, s2_bytes, scalar_size,
                out_buf, numel, shape_bytes, strides_bytes, ndim,
                groups, tpg)

        rt.commit_and_wait(cmd)

    # ------------------------------------------------------------------
    # Dispatch: where  (cond, x, y → out)
    # ------------------------------------------------------------------

    def dispatch_where(self, kernel_name, cond_buf, x_buf, y_buf, out_buf,
                       numel):
        """Encode and execute a where kernel (4 buffers + 1 uint)."""
        rt = get_runtime()
        pipeline = self._get_pipeline(kernel_name)
        tpg = self._threads_per_group(pipeline)
        groups = (numel + tpg - 1) // tpg

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

    # ------------------------------------------------------------------
    # Dispatch: where_scalar  (cond, tensor, scalar → out)
    # ------------------------------------------------------------------

    def dispatch_where_scalar(self, kernel_name, cond_buf, tensor_buf,
                              scalar, out_buf, numel, scalar_fmt="f"):
        """Encode where with one scalar operand (3 bufs + 1 scalar + 1 uint)."""
        rt = get_runtime()
        pipeline = self._get_pipeline(kernel_name)
        tpg = self._threads_per_group(pipeline)
        groups = (numel + tpg - 1) // tpg

        cmd = rt.create_command_buffer()
        enc = rt.get_compute_encoder(cmd)

        scalar_bytes = struct.pack(scalar_fmt, scalar)
        scalar_size = len(scalar_bytes)

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
    # Dispatch: masked_fill  (a, mask, scalar → out)
    # ------------------------------------------------------------------

    def dispatch_masked_fill(self, kernel_name, a_buf, mask_buf, scalar,
                             out_buf, numel, scalar_fmt="f"):
        """Encode masked_fill kernel (3 bufs + 1 scalar + 1 uint)."""
        rt = get_runtime()
        pipeline = self._get_pipeline(kernel_name)
        tpg = self._threads_per_group(pipeline)
        groups = (numel + tpg - 1) // tpg

        cmd = rt.create_command_buffer()
        enc = rt.get_compute_encoder(cmd)

        scalar_bytes = struct.pack(scalar_fmt, scalar)
        scalar_size = len(scalar_bytes)

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
    # Dispatch: tril/triu  (a → out, with rows/cols/diagonal/N)
    # ------------------------------------------------------------------

    def dispatch_tril_triu(self, kernel_name, a_buf, out_buf, rows, cols,
                           diagonal, numel):
        """Encode tril/triu kernel (2 bufs + 2 uint + 1 int + 1 uint)."""
        rt = get_runtime()
        pipeline = self._get_pipeline(kernel_name)
        tpg = self._threads_per_group(pipeline)
        groups = (numel + tpg - 1) // tpg

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
    # Dispatch: index_select / gather  (input, index → output)
    # ------------------------------------------------------------------

    def dispatch_index_gather(self, kernel_name, input_buf, index_buf,
                              out_buf, outer_size, idx_size, inner_size,
                              input_dim_size, numel):
        """Encode index_select or gather kernel (3 bufs + 4 uint)."""
        rt = get_runtime()
        pipeline = self._get_pipeline(kernel_name)
        tpg = self._threads_per_group(pipeline)
        groups = (numel + tpg - 1) // tpg

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
    # Dispatch: cat_copy  (src → dst region)
    # ------------------------------------------------------------------

    def dispatch_cat_copy(self, kernel_name, src_buf, dst_buf, outer_size,
                          src_dim, inner_size, dst_dim, offset, numel):
        """Encode cat_copy kernel (2 bufs + 5 uint)."""
        rt = get_runtime()
        pipeline = self._get_pipeline(kernel_name)
        tpg = self._threads_per_group(pipeline)
        groups = (numel + tpg - 1) // tpg

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
    # Dispatch: conv2d  (input, weight, bias, output + packed params)
    # ------------------------------------------------------------------

    def dispatch_conv2d(self, kernel_name, input_buf, weight_buf, bias_buf,
                        output_buf, N, C_in, H_in, W_in, C_out, kH, kW,
                        H_out, W_out, sH, sW, pH, pW, dH, dW,
                        has_bias, numel):
        """Encode conv2d kernel (4 bufs + 17 packed uint params)."""
        rt = get_runtime()
        pipeline = self._get_pipeline(kernel_name)
        tpg = self._threads_per_group(pipeline)
        groups = (numel + tpg - 1) // tpg

        cmd = rt.create_command_buffer()
        enc = rt.get_compute_encoder(cmd)

        params = struct.pack("17I", N, C_in, H_in, W_in, C_out, kH, kW,
                             H_out, W_out, sH, sW, pH, pW, dH, dW,
                             has_bias, numel)

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
                                  bias_buf, output_buf, params,
                                  groups, tpg)

        rt.commit_and_wait(cmd)

    # ------------------------------------------------------------------
    # Dispatch: layer_norm  (input, weight, bias → output)
    # ------------------------------------------------------------------

    def dispatch_layer_norm(self, kernel_name, input_buf, weight_buf, bias_buf,
                            output_buf, outer_size, inner_size, eps,
                            has_weight, has_bias):
        """Encode layer_norm kernel (4 bufs + 5 setBytes)."""
        rt = get_runtime()
        pipeline = self._get_pipeline(kernel_name)
        tpg = self._threads_per_group(pipeline)
        numel = outer_size * inner_size
        groups = (numel + tpg - 1) // tpg

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
    # Dispatch: rms_norm  (input, weight → output)
    # ------------------------------------------------------------------

    def dispatch_rms_norm(self, kernel_name, input_buf, weight_buf,
                          output_buf, outer_size, inner_size, eps,
                          has_weight):
        """Encode rms_norm kernel (3 bufs + 4 setBytes)."""
        rt = get_runtime()
        pipeline = self._get_pipeline(kernel_name)
        tpg = self._threads_per_group(pipeline)
        numel = outer_size * inner_size
        groups = (numel + tpg - 1) // tpg

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
    # Dispatch: batch_norm_stats  (input → mean, var)
    # ------------------------------------------------------------------

    def dispatch_batch_norm_stats(self, kernel_name, input_buf, mean_buf,
                                  var_buf, N, C, spatial_size):
        """Encode batch_norm_stats kernel (3 bufs + 3 setBytes)."""
        rt = get_runtime()
        pipeline = self._get_pipeline(kernel_name)
        tpg = self._threads_per_group(pipeline)
        groups = (C + tpg - 1) // tpg

        cmd = rt.create_command_buffer()
        enc = rt.get_compute_encoder(cmd)

        if _HAS_PYOBJC:
            enc.setComputePipelineState_(pipeline)
            enc.setBuffer_offset_atIndex_(input_buf, 0, 0)
            enc.setBuffer_offset_atIndex_(mean_buf, 0, 1)
            enc.setBuffer_offset_atIndex_(var_buf, 0, 2)
            enc.setBytes_length_atIndex_(struct.pack("I", N), 4, 3)
            enc.setBytes_length_atIndex_(struct.pack("I", C), 4, 4)
            enc.setBytes_length_atIndex_(struct.pack("I", spatial_size), 4, 5)
            enc.dispatchThreadgroups_threadsPerThreadgroup_(
                _MTLSize(groups, 1, 1), _MTLSize(tpg, 1, 1))
            enc.endEncoding()
        else:
            _encode_batch_norm_stats_ctypes(enc, pipeline, input_buf,
                                            mean_buf, var_buf,
                                            N, C, spatial_size,
                                            groups, tpg)

        rt.commit_and_wait(cmd)

    # ------------------------------------------------------------------
    # Dispatch: batch_norm_apply  (input, mean, var, weight, bias → output)
    # ------------------------------------------------------------------

    def dispatch_batch_norm_apply(self, kernel_name, input_buf, mean_buf,
                                  var_buf, weight_buf, bias_buf, output_buf,
                                  C, spatial_size, eps, has_weight, has_bias,
                                  total):
        """Encode batch_norm_apply kernel (6 bufs + 6 setBytes)."""
        rt = get_runtime()
        pipeline = self._get_pipeline(kernel_name)
        tpg = self._threads_per_group(pipeline)
        groups = (total + tpg - 1) // tpg

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
    # Dispatch: Philox RNG fill  (seed, offset → output)
    # ------------------------------------------------------------------

    def dispatch_philox_fill(self, kernel_name, out_buf, seed_lo, seed_hi,
                             offset, param1, param2, numel,
                             param_fmt="f"):
        """Encode Philox RNG fill kernel (1 buf + 2 seed uint + 1 offset uint
        + 2 params + 1 N uint). Works for uniform (low/high),
        normal (mean/std)."""
        rt = get_runtime()
        pipeline = self._get_pipeline(kernel_name)
        tpg = self._threads_per_group(pipeline)
        threads = (numel + 3) // 4  # each thread produces 4 values
        groups = (threads + tpg - 1) // tpg

        cmd = rt.create_command_buffer()
        enc = rt.get_compute_encoder(cmd)

        p1_bytes = struct.pack(param_fmt, param1)
        p2_bytes = struct.pack(param_fmt, param2)
        p_size = len(p1_bytes)

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
            _encode_philox_fill_ctypes(enc, pipeline, out_buf,
                                       seed_lo, seed_hi, offset,
                                       p1_bytes, p2_bytes, p_size,
                                       numel, groups, tpg)

        rt.commit_and_wait(cmd)

    # ------------------------------------------------------------------
    # Dispatch: Philox bernoulli  (prob, seed, offset → output)
    # ------------------------------------------------------------------

    def dispatch_philox_bernoulli(self, kernel_name, out_buf, prob,
                                  seed_lo, seed_hi, offset, numel):
        """Encode Philox bernoulli kernel (1 buf + prob + 2 seed + offset + N)."""
        rt = get_runtime()
        pipeline = self._get_pipeline(kernel_name)
        tpg = self._threads_per_group(pipeline)
        threads = (numel + 3) // 4
        groups = (threads + tpg - 1) // tpg

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
    # Dispatch: Philox randint  (low, high, seed, offset → output)
    # ------------------------------------------------------------------

    def dispatch_philox_randint(self, kernel_name, out_buf, low, high,
                                seed_lo, seed_hi, offset, numel,
                                int_fmt="i"):
        """Encode Philox randint kernel."""
        rt = get_runtime()
        pipeline = self._get_pipeline(kernel_name)
        tpg = self._threads_per_group(pipeline)
        threads = (numel + 3) // 4
        groups = (threads + tpg - 1) // tpg

        cmd = rt.create_command_buffer()
        enc = rt.get_compute_encoder(cmd)

        lo_bytes = struct.pack(int_fmt, low)
        hi_bytes = struct.pack(int_fmt, high)
        i_size = len(lo_bytes)

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
    # Dispatch: Philox dropout  (input → output, fused mask+scale)
    # ------------------------------------------------------------------

    def dispatch_philox_dropout(self, kernel_name, a_buf, out_buf,
                                prob, scale, seed_lo, seed_hi, offset,
                                numel):
        """Encode fused Philox dropout kernel."""
        rt = get_runtime()
        pipeline = self._get_pipeline(kernel_name)
        tpg = self._threads_per_group(pipeline)
        threads = (numel + 3) // 4
        groups = (threads + tpg - 1) // tpg

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
    # Dispatch: axis reduction  (input → reduced output along dim)
    # ------------------------------------------------------------------

    def dispatch_reduce_dim(self, kernel_name, a_buf, out_buf,
                            outer_size, reduce_size, inner_size,
                            out_numel):
        """Dispatch an axis-reduce kernel over one dimension."""
        rt = get_runtime()
        pipeline = self._get_pipeline(kernel_name)
        tpg = self._threads_per_group(pipeline)
        groups = (out_numel + tpg - 1) // tpg

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
    # Dispatch: softmax 2D  (input → output, with rows/cols)
    # ------------------------------------------------------------------

    def dispatch_softmax_2d(self, kernel_name, a_buf, out_buf, rows, cols):
        """Dispatch softmax over last dim of a 2D tensor."""
        rt = get_runtime()
        pipeline = self._get_pipeline(kernel_name)

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
            _encode_softmax_ctypes(enc, pipeline, a_buf, out_buf,
                                   rows, cols)

        rt.commit_and_wait(cmd)


# ---------------------------------------------------------------------------
# ctypes encoding helpers (fallback when no PyObjC)
# ---------------------------------------------------------------------------

def _library_get_function_ctypes(library, name):
    """Get a MTLFunction from a MTLLibrary by name (ctypes path)."""
    from .runtime import _libobjc, _load_objc_libs
    _load_objc_libs()
    # Create NSString for function name
    ns_string_class = _libobjc.objc_getClass(b"NSString")
    sel_alloc = _libobjc.sel_registerName(b"alloc")
    sel_init = _libobjc.sel_registerName(b"initWithUTF8String:")
    ns_str = _libobjc.objc_msgSend(ns_string_class, sel_alloc)
    name_bytes = name.encode("utf-8")
    _libobjc.objc_msgSend.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_char_p]
    _libobjc.objc_msgSend.restype = ctypes.c_void_p
    ns_str = _libobjc.objc_msgSend(ns_str, sel_init, name_bytes)
    _libobjc.objc_msgSend.argtypes = [ctypes.c_void_p, ctypes.c_void_p]

    sel = _libobjc.sel_registerName(b"newFunctionWithName:")
    _libobjc.objc_msgSend.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    _libobjc.objc_msgSend.restype = ctypes.c_void_p
    fn = _libobjc.objc_msgSend(library, sel, ns_str)
    _libobjc.objc_msgSend.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    return fn


def _ctypes_set_buffer(enc, buf, offset, index):
    """setBuffer:offset:atIndex: via ctypes."""
    from .runtime import _libobjc, _load_objc_libs
    _load_objc_libs()
    sel = _libobjc.sel_registerName(b"setBuffer:offset:atIndex:")
    _libobjc.objc_msgSend.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint64,
    ]
    _libobjc.objc_msgSend.restype = None
    _libobjc.objc_msgSend(enc, sel, buf, ctypes.c_uint64(offset),
                           ctypes.c_uint64(index))
    _libobjc.objc_msgSend.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    _libobjc.objc_msgSend.restype = ctypes.c_void_p


def _ctypes_set_bytes(enc, data_bytes, length, index):
    """setBytes:length:atIndex: via ctypes."""
    from .runtime import _libobjc, _load_objc_libs
    _load_objc_libs()
    sel = _libobjc.sel_registerName(b"setBytes:length:atIndex:")
    buf = ctypes.create_string_buffer(data_bytes)
    _libobjc.objc_msgSend.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint64,
    ]
    _libobjc.objc_msgSend.restype = None
    _libobjc.objc_msgSend(enc, sel, buf, ctypes.c_uint64(length),
                           ctypes.c_uint64(index))
    _libobjc.objc_msgSend.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    _libobjc.objc_msgSend.restype = ctypes.c_void_p


def _ctypes_set_pipeline(enc, pipeline):
    """setComputePipelineState: via ctypes."""
    from .runtime import _libobjc, _load_objc_libs
    _load_objc_libs()
    sel = _libobjc.sel_registerName(b"setComputePipelineState:")
    _libobjc.objc_msgSend.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    _libobjc.objc_msgSend.restype = None
    _libobjc.objc_msgSend(enc, sel, pipeline)
    _libobjc.objc_msgSend.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    _libobjc.objc_msgSend.restype = ctypes.c_void_p


def _ctypes_dispatch_threadgroups(enc, groups, tpg):
    """dispatchThreadgroups:threadsPerThreadgroup: via ctypes.

    MTLSize is a struct {uint64, uint64, uint64}.
    """
    from .runtime import _libobjc, _load_objc_libs
    _load_objc_libs()

    class MTLSize(ctypes.Structure):
        _fields_ = [("width", ctypes.c_uint64),
                     ("height", ctypes.c_uint64),
                     ("depth", ctypes.c_uint64)]

    sel = _libobjc.sel_registerName(b"dispatchThreadgroups:threadsPerThreadgroup:")
    grid = MTLSize(groups, 1, 1)
    tpg_s = MTLSize(tpg, 1, 1)
    _libobjc.objc_msgSend.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, MTLSize, MTLSize,
    ]
    _libobjc.objc_msgSend.restype = None
    _libobjc.objc_msgSend(enc, sel, grid, tpg_s)
    _libobjc.objc_msgSend.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    _libobjc.objc_msgSend.restype = ctypes.c_void_p


def _ctypes_end_encoding(enc):
    """endEncoding via ctypes."""
    from .runtime import _libobjc, _load_objc_libs
    _load_objc_libs()
    sel = _libobjc.sel_registerName(b"endEncoding")
    _libobjc.objc_msgSend.restype = None
    _libobjc.objc_msgSend(enc, sel)
    _libobjc.objc_msgSend.restype = ctypes.c_void_p


def _encode_unary_ctypes(enc, pipeline, a_buf, out_buf, numel, groups, tpg):
    """Encode a unary kernel dispatch via ctypes."""
    _ctypes_set_pipeline(enc, pipeline)
    _ctypes_set_buffer(enc, a_buf, 0, 0)
    _ctypes_set_buffer(enc, out_buf, 0, 1)
    _ctypes_set_bytes(enc, struct.pack("I", numel), 4, 2)
    _ctypes_dispatch_threadgroups(enc, groups, tpg)
    _ctypes_end_encoding(enc)


def _encode_binary_ctypes(enc, pipeline, a_buf, b_buf, out_buf,
                           numel, groups, tpg):
    """Encode a binary kernel dispatch via ctypes."""
    _ctypes_set_pipeline(enc, pipeline)
    _ctypes_set_buffer(enc, a_buf, 0, 0)
    _ctypes_set_buffer(enc, b_buf, 0, 1)
    _ctypes_set_buffer(enc, out_buf, 0, 2)
    _ctypes_set_bytes(enc, struct.pack("I", numel), 4, 3)
    _ctypes_dispatch_threadgroups(enc, groups, tpg)
    _ctypes_end_encoding(enc)


def _encode_binary_scalar_ctypes(enc, pipeline, a_buf, scalar_bytes,
                                  scalar_size, out_buf, numel, groups, tpg):
    """Encode a binary-scalar kernel dispatch via ctypes."""
    _ctypes_set_pipeline(enc, pipeline)
    _ctypes_set_buffer(enc, a_buf, 0, 0)
    _ctypes_set_bytes(enc, scalar_bytes, scalar_size, 1)
    _ctypes_set_buffer(enc, out_buf, 0, 2)
    _ctypes_set_bytes(enc, struct.pack("I", numel), 4, 3)
    _ctypes_dispatch_threadgroups(enc, groups, tpg)
    _ctypes_end_encoding(enc)


def _encode_arg_partial_ctypes(enc, pipeline, a_buf, vals_buf, idxs_buf,
                                numel, groups, tpg):
    """Encode argmax/argmin partial pass via ctypes."""
    _ctypes_set_pipeline(enc, pipeline)
    _ctypes_set_buffer(enc, a_buf, 0, 0)
    _ctypes_set_buffer(enc, vals_buf, 0, 1)
    _ctypes_set_buffer(enc, idxs_buf, 0, 2)
    _ctypes_set_bytes(enc, struct.pack("I", numel), 4, 3)
    _ctypes_dispatch_threadgroups(enc, groups, tpg)
    _ctypes_end_encoding(enc)


def _encode_arg_final_ctypes(enc, pipeline, vals_buf, idxs_buf, out_buf,
                              num_groups, tpg):
    """Encode argmax/argmin final pass via ctypes."""
    _ctypes_set_pipeline(enc, pipeline)
    _ctypes_set_buffer(enc, vals_buf, 0, 0)
    _ctypes_set_buffer(enc, idxs_buf, 0, 1)
    _ctypes_set_buffer(enc, out_buf, 0, 2)
    _ctypes_set_bytes(enc, struct.pack("I", num_groups), 4, 3)
    _ctypes_dispatch_threadgroups(enc, 1, tpg)
    _ctypes_end_encoding(enc)


def _encode_inplace_unary_ctypes(enc, pipeline, a_buf, numel, groups, tpg):
    """Encode an in-place unary kernel via ctypes."""
    _ctypes_set_pipeline(enc, pipeline)
    _ctypes_set_buffer(enc, a_buf, 0, 0)
    _ctypes_set_bytes(enc, struct.pack("I", numel), 4, 1)
    _ctypes_dispatch_threadgroups(enc, groups, tpg)
    _ctypes_end_encoding(enc)


def _encode_inplace_scalar_ctypes(enc, pipeline, a_buf, scalar_bytes,
                                   scalar_size, numel, groups, tpg):
    """Encode an in-place binary-scalar kernel via ctypes."""
    _ctypes_set_pipeline(enc, pipeline)
    _ctypes_set_buffer(enc, a_buf, 0, 0)
    _ctypes_set_bytes(enc, scalar_bytes, scalar_size, 1)
    _ctypes_set_bytes(enc, struct.pack("I", numel), 4, 2)
    _ctypes_dispatch_threadgroups(enc, groups, tpg)
    _ctypes_end_encoding(enc)


def _encode_softmax_ctypes(enc, pipeline, a_buf, out_buf, rows, cols):
    """Encode softmax 2D kernel via ctypes."""
    _ctypes_set_pipeline(enc, pipeline)
    _ctypes_set_buffer(enc, a_buf, 0, 0)
    _ctypes_set_buffer(enc, out_buf, 0, 1)
    _ctypes_set_bytes(enc, struct.pack("I", rows), 4, 2)
    _ctypes_set_bytes(enc, struct.pack("I", cols), 4, 3)
    tpg_x = min(32, cols)
    tpg_y = min(8, rows)
    groups_x = (cols + tpg_x - 1) // tpg_x
    groups_y = (rows + tpg_y - 1) // tpg_y

    from .runtime import _libobjc, _load_objc_libs
    _load_objc_libs()

    class MTLSize(ctypes.Structure):
        _fields_ = [("width", ctypes.c_uint64),
                     ("height", ctypes.c_uint64),
                     ("depth", ctypes.c_uint64)]

    sel = _libobjc.sel_registerName(b"dispatchThreadgroups:threadsPerThreadgroup:")
    grid = MTLSize(groups_x, groups_y, 1)
    tpg_s = MTLSize(tpg_x, tpg_y, 1)
    _libobjc.objc_msgSend.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, MTLSize, MTLSize,
    ]
    _libobjc.objc_msgSend.restype = None
    _libobjc.objc_msgSend(enc, sel, grid, tpg_s)
    _libobjc.objc_msgSend.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    _libobjc.objc_msgSend.restype = ctypes.c_void_p
    _ctypes_end_encoding(enc)


def _encode_unary_strided_ctypes(enc, pipeline, a_buf, out_buf, numel,
                                  shape_bytes, strides_bytes, ndim,
                                  groups, tpg):
    """Encode a strided unary kernel dispatch via ctypes."""
    _ctypes_set_pipeline(enc, pipeline)
    _ctypes_set_buffer(enc, a_buf, 0, 0)
    _ctypes_set_buffer(enc, out_buf, 0, 1)
    _ctypes_set_bytes(enc, struct.pack("I", numel), 4, 2)
    _ctypes_set_bytes(enc, shape_bytes, len(shape_bytes), 3)
    _ctypes_set_bytes(enc, strides_bytes, len(strides_bytes), 4)
    _ctypes_set_bytes(enc, struct.pack("I", ndim), 4, 5)
    _ctypes_dispatch_threadgroups(enc, groups, tpg)
    _ctypes_end_encoding(enc)


def _encode_binary_strided_ctypes(enc, pipeline, a_buf, b_buf, out_buf,
                                   numel, shape_bytes, strides_a_bytes,
                                   strides_b_bytes, ndim, groups, tpg):
    """Encode a strided binary kernel dispatch via ctypes."""
    _ctypes_set_pipeline(enc, pipeline)
    _ctypes_set_buffer(enc, a_buf, 0, 0)
    _ctypes_set_buffer(enc, b_buf, 0, 1)
    _ctypes_set_buffer(enc, out_buf, 0, 2)
    _ctypes_set_bytes(enc, struct.pack("I", numel), 4, 3)
    _ctypes_set_bytes(enc, shape_bytes, len(shape_bytes), 4)
    _ctypes_set_bytes(enc, strides_a_bytes, len(strides_a_bytes), 5)
    _ctypes_set_bytes(enc, strides_b_bytes, len(strides_b_bytes), 6)
    _ctypes_set_bytes(enc, struct.pack("I", ndim), 4, 7)
    _ctypes_dispatch_threadgroups(enc, groups, tpg)
    _ctypes_end_encoding(enc)


def _encode_binary_scalar_strided_ctypes(enc, pipeline, a_buf, scalar_bytes,
                                          scalar_size, out_buf, numel,
                                          shape_bytes, strides_bytes, ndim,
                                          groups, tpg):
    """Encode a strided binary-scalar kernel dispatch via ctypes."""
    _ctypes_set_pipeline(enc, pipeline)
    _ctypes_set_buffer(enc, a_buf, 0, 0)
    _ctypes_set_bytes(enc, scalar_bytes, scalar_size, 1)
    _ctypes_set_buffer(enc, out_buf, 0, 2)
    _ctypes_set_bytes(enc, struct.pack("I", numel), 4, 3)
    _ctypes_set_bytes(enc, shape_bytes, len(shape_bytes), 4)
    _ctypes_set_bytes(enc, strides_bytes, len(strides_bytes), 5)
    _ctypes_set_bytes(enc, struct.pack("I", ndim), 4, 6)
    _ctypes_dispatch_threadgroups(enc, groups, tpg)
    _ctypes_end_encoding(enc)


def _encode_reduce_dim_ctypes(enc, pipeline, a_buf, out_buf,
                               outer_size, reduce_size, inner_size,
                               groups, tpg):
    """Encode an axis-reduce kernel dispatch via ctypes."""
    _ctypes_set_pipeline(enc, pipeline)
    _ctypes_set_buffer(enc, a_buf, 0, 0)
    _ctypes_set_buffer(enc, out_buf, 0, 1)
    _ctypes_set_bytes(enc, struct.pack("I", outer_size), 4, 2)
    _ctypes_set_bytes(enc, struct.pack("I", reduce_size), 4, 3)
    _ctypes_set_bytes(enc, struct.pack("I", inner_size), 4, 4)
    _ctypes_dispatch_threadgroups(enc, groups, tpg)
    _ctypes_end_encoding(enc)


def _encode_clamp_ctypes(enc, pipeline, a_buf, s1_bytes, s2_bytes,
                          scalar_size, out_buf, numel, groups, tpg):
    """Encode a clamp (2-scalar) kernel dispatch via ctypes."""
    _ctypes_set_pipeline(enc, pipeline)
    _ctypes_set_buffer(enc, a_buf, 0, 0)
    _ctypes_set_bytes(enc, s1_bytes, scalar_size, 1)
    _ctypes_set_bytes(enc, s2_bytes, scalar_size, 2)
    _ctypes_set_buffer(enc, out_buf, 0, 3)
    _ctypes_set_bytes(enc, struct.pack("I", numel), 4, 4)
    _ctypes_dispatch_threadgroups(enc, groups, tpg)
    _ctypes_end_encoding(enc)


def _encode_clamp_strided_ctypes(enc, pipeline, a_buf, s1_bytes, s2_bytes,
                                  scalar_size, out_buf, numel,
                                  shape_bytes, strides_bytes, ndim,
                                  groups, tpg):
    """Encode a strided clamp (2-scalar) kernel dispatch via ctypes."""
    _ctypes_set_pipeline(enc, pipeline)
    _ctypes_set_buffer(enc, a_buf, 0, 0)
    _ctypes_set_bytes(enc, s1_bytes, scalar_size, 1)
    _ctypes_set_bytes(enc, s2_bytes, scalar_size, 2)
    _ctypes_set_buffer(enc, out_buf, 0, 3)
    _ctypes_set_bytes(enc, struct.pack("I", numel), 4, 4)
    _ctypes_set_bytes(enc, shape_bytes, len(shape_bytes), 5)
    _ctypes_set_bytes(enc, strides_bytes, len(strides_bytes), 6)
    _ctypes_set_bytes(enc, struct.pack("I", ndim), 4, 7)
    _ctypes_dispatch_threadgroups(enc, groups, tpg)
    _ctypes_end_encoding(enc)


# ---------------------------------------------------------------------------
# Philox RNG ctypes encoders
# ---------------------------------------------------------------------------

def _encode_philox_fill_ctypes(enc, pipeline, out_buf, seed_lo, seed_hi,
                                offset, p1_bytes, p2_bytes, p_size,
                                numel, groups, tpg):
    """Encode Philox fill (uniform/normal) kernel via ctypes."""
    _ctypes_set_pipeline(enc, pipeline)
    _ctypes_set_buffer(enc, out_buf, 0, 0)
    _ctypes_set_bytes(enc, struct.pack("I", seed_lo), 4, 1)
    _ctypes_set_bytes(enc, struct.pack("I", seed_hi), 4, 2)
    _ctypes_set_bytes(enc, struct.pack("I", offset), 4, 3)
    _ctypes_set_bytes(enc, p1_bytes, p_size, 4)
    _ctypes_set_bytes(enc, p2_bytes, p_size, 5)
    _ctypes_set_bytes(enc, struct.pack("I", numel), 4, 6)
    _ctypes_dispatch_threadgroups(enc, groups, tpg)
    _ctypes_end_encoding(enc)


def _encode_philox_bernoulli_ctypes(enc, pipeline, out_buf, prob,
                                     seed_lo, seed_hi, offset,
                                     numel, groups, tpg):
    """Encode Philox bernoulli kernel via ctypes."""
    _ctypes_set_pipeline(enc, pipeline)
    _ctypes_set_buffer(enc, out_buf, 0, 0)
    _ctypes_set_bytes(enc, struct.pack("f", prob), 4, 1)
    _ctypes_set_bytes(enc, struct.pack("I", seed_lo), 4, 2)
    _ctypes_set_bytes(enc, struct.pack("I", seed_hi), 4, 3)
    _ctypes_set_bytes(enc, struct.pack("I", offset), 4, 4)
    _ctypes_set_bytes(enc, struct.pack("I", numel), 4, 5)
    _ctypes_dispatch_threadgroups(enc, groups, tpg)
    _ctypes_end_encoding(enc)


def _encode_philox_randint_ctypes(enc, pipeline, out_buf,
                                   lo_bytes, hi_bytes, i_size,
                                   seed_lo, seed_hi, offset,
                                   numel, groups, tpg):
    """Encode Philox randint kernel via ctypes."""
    _ctypes_set_pipeline(enc, pipeline)
    _ctypes_set_buffer(enc, out_buf, 0, 0)
    _ctypes_set_bytes(enc, lo_bytes, i_size, 1)
    _ctypes_set_bytes(enc, hi_bytes, i_size, 2)
    _ctypes_set_bytes(enc, struct.pack("I", seed_lo), 4, 3)
    _ctypes_set_bytes(enc, struct.pack("I", seed_hi), 4, 4)
    _ctypes_set_bytes(enc, struct.pack("I", offset), 4, 5)
    _ctypes_set_bytes(enc, struct.pack("I", numel), 4, 6)
    _ctypes_dispatch_threadgroups(enc, groups, tpg)
    _ctypes_end_encoding(enc)


def _encode_philox_dropout_ctypes(enc, pipeline, a_buf, out_buf,
                                   prob, scale, seed_lo, seed_hi,
                                   offset, numel, groups, tpg):
    """Encode fused Philox dropout kernel via ctypes."""
    _ctypes_set_pipeline(enc, pipeline)
    _ctypes_set_buffer(enc, a_buf, 0, 0)
    _ctypes_set_buffer(enc, out_buf, 0, 1)
    _ctypes_set_bytes(enc, struct.pack("f", prob), 4, 2)
    _ctypes_set_bytes(enc, struct.pack("f", scale), 4, 3)
    _ctypes_set_bytes(enc, struct.pack("I", seed_lo), 4, 4)
    _ctypes_set_bytes(enc, struct.pack("I", seed_hi), 4, 5)
    _ctypes_set_bytes(enc, struct.pack("I", offset), 4, 6)
    _ctypes_set_bytes(enc, struct.pack("I", numel), 4, 7)
    _ctypes_dispatch_threadgroups(enc, groups, tpg)
    _ctypes_end_encoding(enc)

def _encode_where_ctypes(enc, pipeline, cond_buf, x_buf, y_buf, out_buf,
                          numel, groups, tpg):
    """Encode where kernel (4 buffers + 1 uint) via ctypes."""
    _ctypes_set_pipeline(enc, pipeline)
    _ctypes_set_buffer(enc, cond_buf, 0, 0)
    _ctypes_set_buffer(enc, x_buf, 0, 1)
    _ctypes_set_buffer(enc, y_buf, 0, 2)
    _ctypes_set_buffer(enc, out_buf, 0, 3)
    _ctypes_set_bytes(enc, struct.pack("I", numel), 4, 4)
    _ctypes_dispatch_threadgroups(enc, groups, tpg)
    _ctypes_end_encoding(enc)


def _encode_where_scalar_ctypes(enc, pipeline, cond_buf, tensor_buf,
                                 scalar_bytes, scalar_size, out_buf,
                                 numel, groups, tpg):
    """Encode where_scalar kernel (3 bufs + 1 scalar + 1 uint) via ctypes."""
    _ctypes_set_pipeline(enc, pipeline)
    _ctypes_set_buffer(enc, cond_buf, 0, 0)
    _ctypes_set_buffer(enc, tensor_buf, 0, 1)
    _ctypes_set_bytes(enc, scalar_bytes, scalar_size, 2)
    _ctypes_set_buffer(enc, out_buf, 0, 3)
    _ctypes_set_bytes(enc, struct.pack("I", numel), 4, 4)
    _ctypes_dispatch_threadgroups(enc, groups, tpg)
    _ctypes_end_encoding(enc)


def _encode_masked_fill_ctypes(enc, pipeline, a_buf, mask_buf,
                                scalar_bytes, scalar_size, out_buf,
                                numel, groups, tpg):
    """Encode masked_fill kernel (3 bufs + 1 scalar + 1 uint) via ctypes."""
    _ctypes_set_pipeline(enc, pipeline)
    _ctypes_set_buffer(enc, a_buf, 0, 0)
    _ctypes_set_buffer(enc, mask_buf, 0, 1)
    _ctypes_set_bytes(enc, scalar_bytes, scalar_size, 2)
    _ctypes_set_buffer(enc, out_buf, 0, 3)
    _ctypes_set_bytes(enc, struct.pack("I", numel), 4, 4)
    _ctypes_dispatch_threadgroups(enc, groups, tpg)
    _ctypes_end_encoding(enc)


def _encode_tril_triu_ctypes(enc, pipeline, a_buf, out_buf,
                              rows, cols, diagonal, numel, groups, tpg):
    """Encode tril/triu kernel (2 bufs + 2 uint + 1 int + 1 uint) via ctypes."""
    _ctypes_set_pipeline(enc, pipeline)
    _ctypes_set_buffer(enc, a_buf, 0, 0)
    _ctypes_set_buffer(enc, out_buf, 0, 1)
    _ctypes_set_bytes(enc, struct.pack("I", rows), 4, 2)
    _ctypes_set_bytes(enc, struct.pack("I", cols), 4, 3)
    _ctypes_set_bytes(enc, struct.pack("i", diagonal), 4, 4)
    _ctypes_set_bytes(enc, struct.pack("I", numel), 4, 5)
    _ctypes_dispatch_threadgroups(enc, groups, tpg)
    _ctypes_end_encoding(enc)


def _encode_index_gather_ctypes(enc, pipeline, input_buf, index_buf, out_buf,
                                 outer_size, idx_size, inner_size,
                                 input_dim_size, groups, tpg):
    """Encode index_select/gather kernel (3 bufs + 4 uint) via ctypes."""
    _ctypes_set_pipeline(enc, pipeline)
    _ctypes_set_buffer(enc, input_buf, 0, 0)
    _ctypes_set_buffer(enc, index_buf, 0, 1)
    _ctypes_set_buffer(enc, out_buf, 0, 2)
    _ctypes_set_bytes(enc, struct.pack("I", outer_size), 4, 3)
    _ctypes_set_bytes(enc, struct.pack("I", idx_size), 4, 4)
    _ctypes_set_bytes(enc, struct.pack("I", inner_size), 4, 5)
    _ctypes_set_bytes(enc, struct.pack("I", input_dim_size), 4, 6)
    _ctypes_dispatch_threadgroups(enc, groups, tpg)
    _ctypes_end_encoding(enc)


def _encode_cat_copy_ctypes(enc, pipeline, src_buf, dst_buf,
                             outer_size, src_dim, inner_size,
                             dst_dim, offset, groups, tpg):
    """Encode cat_copy kernel (2 bufs + 5 uint) via ctypes."""
    _ctypes_set_pipeline(enc, pipeline)
    _ctypes_set_buffer(enc, src_buf, 0, 0)
    _ctypes_set_buffer(enc, dst_buf, 0, 1)
    _ctypes_set_bytes(enc, struct.pack("I", outer_size), 4, 2)
    _ctypes_set_bytes(enc, struct.pack("I", src_dim), 4, 3)
    _ctypes_set_bytes(enc, struct.pack("I", inner_size), 4, 4)
    _ctypes_set_bytes(enc, struct.pack("I", dst_dim), 4, 5)
    _ctypes_set_bytes(enc, struct.pack("I", offset), 4, 6)
    _ctypes_dispatch_threadgroups(enc, groups, tpg)
    _ctypes_end_encoding(enc)


def _encode_conv2d_ctypes(enc, pipeline, input_buf, weight_buf, bias_buf,
                           output_buf, params, groups, tpg):
    """Encode conv2d kernel (4 bufs + packed params) via ctypes."""
    _ctypes_set_pipeline(enc, pipeline)
    _ctypes_set_buffer(enc, input_buf, 0, 0)
    _ctypes_set_buffer(enc, weight_buf, 0, 1)
    _ctypes_set_buffer(enc, bias_buf, 0, 2)
    _ctypes_set_buffer(enc, output_buf, 0, 3)
    _ctypes_set_bytes(enc, params, len(params), 4)
    _ctypes_dispatch_threadgroups(enc, groups, tpg)
    _ctypes_end_encoding(enc)


def _encode_layer_norm_ctypes(enc, pipeline, input_buf, weight_buf, bias_buf,
                               output_buf, outer_size, inner_size, eps,
                               has_weight, has_bias, groups, tpg):
    """Encode layer_norm kernel (4 bufs + 5 setBytes) via ctypes."""
    _ctypes_set_pipeline(enc, pipeline)
    _ctypes_set_buffer(enc, input_buf, 0, 0)
    _ctypes_set_buffer(enc, weight_buf, 0, 1)
    _ctypes_set_buffer(enc, bias_buf, 0, 2)
    _ctypes_set_buffer(enc, output_buf, 0, 3)
    _ctypes_set_bytes(enc, struct.pack("I", outer_size), 4, 4)
    _ctypes_set_bytes(enc, struct.pack("I", inner_size), 4, 5)
    _ctypes_set_bytes(enc, struct.pack("f", eps), 4, 6)
    _ctypes_set_bytes(enc, struct.pack("I", has_weight), 4, 7)
    _ctypes_set_bytes(enc, struct.pack("I", has_bias), 4, 8)
    _ctypes_dispatch_threadgroups(enc, groups, tpg)
    _ctypes_end_encoding(enc)


def _encode_rms_norm_ctypes(enc, pipeline, input_buf, weight_buf,
                             output_buf, outer_size, inner_size, eps,
                             has_weight, groups, tpg):
    """Encode rms_norm kernel (3 bufs + 4 setBytes) via ctypes."""
    _ctypes_set_pipeline(enc, pipeline)
    _ctypes_set_buffer(enc, input_buf, 0, 0)
    _ctypes_set_buffer(enc, weight_buf, 0, 1)
    _ctypes_set_buffer(enc, output_buf, 0, 2)
    _ctypes_set_bytes(enc, struct.pack("I", outer_size), 4, 3)
    _ctypes_set_bytes(enc, struct.pack("I", inner_size), 4, 4)
    _ctypes_set_bytes(enc, struct.pack("f", eps), 4, 5)
    _ctypes_set_bytes(enc, struct.pack("I", has_weight), 4, 6)
    _ctypes_dispatch_threadgroups(enc, groups, tpg)
    _ctypes_end_encoding(enc)


def _encode_batch_norm_stats_ctypes(enc, pipeline, input_buf, mean_buf,
                                     var_buf, N, C, spatial_size,
                                     groups, tpg):
    """Encode batch_norm_stats kernel (3 bufs + 3 setBytes) via ctypes."""
    _ctypes_set_pipeline(enc, pipeline)
    _ctypes_set_buffer(enc, input_buf, 0, 0)
    _ctypes_set_buffer(enc, mean_buf, 0, 1)
    _ctypes_set_buffer(enc, var_buf, 0, 2)
    _ctypes_set_bytes(enc, struct.pack("I", N), 4, 3)
    _ctypes_set_bytes(enc, struct.pack("I", C), 4, 4)
    _ctypes_set_bytes(enc, struct.pack("I", spatial_size), 4, 5)
    _ctypes_dispatch_threadgroups(enc, groups, tpg)
    _ctypes_end_encoding(enc)


def _encode_batch_norm_apply_ctypes(enc, pipeline, input_buf, mean_buf,
                                     var_buf, weight_buf, bias_buf,
                                     output_buf, C, spatial_size, eps,
                                     has_weight, has_bias, total,
                                     groups, tpg):
    """Encode batch_norm_apply kernel (6 bufs + 6 setBytes) via ctypes."""
    _ctypes_set_pipeline(enc, pipeline)
    _ctypes_set_buffer(enc, input_buf, 0, 0)
    _ctypes_set_buffer(enc, mean_buf, 0, 1)
    _ctypes_set_buffer(enc, var_buf, 0, 2)
    _ctypes_set_buffer(enc, weight_buf, 0, 3)
    _ctypes_set_buffer(enc, bias_buf, 0, 4)
    _ctypes_set_buffer(enc, output_buf, 0, 5)
    _ctypes_set_bytes(enc, struct.pack("I", C), 4, 6)
    _ctypes_set_bytes(enc, struct.pack("I", spatial_size), 4, 7)
    _ctypes_set_bytes(enc, struct.pack("f", eps), 4, 8)
    _ctypes_set_bytes(enc, struct.pack("I", has_weight), 4, 9)
    _ctypes_set_bytes(enc, struct.pack("I", has_bias), 4, 10)
    _ctypes_set_bytes(enc, struct.pack("I", total), 4, 11)
    _ctypes_dispatch_threadgroups(enc, groups, tpg)
    _ctypes_end_encoding(enc)
