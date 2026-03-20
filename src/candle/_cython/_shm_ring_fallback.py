"""Pure-Python fallback for _shm_ring.pyx.

All symbols are None; _shm_buffer.py falls back to pickle IPC when
the Cython .so is absent.
"""
ShmRingBuffer = None
