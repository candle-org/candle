from libc.stdint cimport uint32_t, uintptr_t

cpdef capture_begin(uintptr_t stream, int mode)
cpdef uintptr_t capture_end(uintptr_t stream)
cpdef tuple capture_get_info(uintptr_t stream)
cpdef int capture_thread_exchange_mode(int mode)
cpdef ri_execute_async(uintptr_t model_ri, uintptr_t stream)
cpdef ri_execute(uintptr_t model_ri, int timeout=*)
cpdef ri_destroy(uintptr_t model_ri)
cpdef ri_set_name(uintptr_t model_ri, str name)
cpdef str ri_get_name(uintptr_t model_ri)
cpdef ri_debug_json_print(uintptr_t model_ri, str path, uint32_t flags=*)
cpdef ri_abort(uintptr_t model_ri)
cpdef synchronize_stream(uintptr_t handle)
