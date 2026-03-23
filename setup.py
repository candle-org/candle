"""Package discovery for candle.

The main project metadata lives in pyproject.toml.  This file handles package
discovery because setuptools' ``packages.find`` in pyproject.toml conflicts
with ``py_modules`` — we need both to install the candle package tree AND the
standalone ``_candle_torch_compat`` bootstrap module and its ``.pth`` trigger.
"""

import os
import platform
import shutil

from setuptools import setup, find_packages, Extension
from setuptools.command.build_py import build_py
from Cython.Build import cythonize

# ---------------------------------------------------------------------------
# Required Cython extensions (must build on every supported platform)
# ---------------------------------------------------------------------------
required_extensions = [
    Extension(
        "candle._cython._stream",
        ["src/candle/_cython/_stream.pyx"],
    ),
    Extension(
        "candle._cython._future",
        ["src/candle/_cython/_future.pyx"],
    ),
]

# ---------------------------------------------------------------------------
# Cross-platform extensions (Linux + macOS)
# ---------------------------------------------------------------------------
_system = platform.system()
cross_platform_extensions = []
if _system in ("Linux", "Darwin"):
    cross_platform_extensions = [
        Extension(
            "candle._cython._dispatch",
            ["src/candle/_cython/_dispatch.pyx"],
        ),
        Extension(
            "candle._cython._allocator",
            ["src/candle/_cython/_allocator.pyx"],
        ),
        Extension(
            "candle._cython._storage",
            ["src/candle/_cython/_storage.pyx"],
        ),
        Extension(
            "candle._cython._tensor_impl",
            ["src/candle/_cython/_tensor_impl.pyx"],
        ),
        Extension(
            "candle._cython._dispatcher_core",
            ["src/candle/_cython/_dispatcher_core.pyx"],
        ),
        Extension(
            "candle._cython._device",
            ["src/candle/_cython/_device.pyx"],
        ),
        Extension(
            "candle._cython._dtype",
            ["src/candle/_cython/_dtype.pyx"],
        ),
        Extension(
            "candle._cython._autograd_node",
            ["src/candle/_cython/_autograd_node.pyx"],
        ),
        Extension(
            "candle._cython._autograd_graph",
            ["src/candle/_cython/_autograd_graph.pyx"],
        ),
        Extension(
            "candle._cython._autograd_engine",
            ["src/candle/_cython/_autograd_engine.pyx"],
        ),
        Extension(
            "candle._cython._autograd_function",
            ["src/candle/_cython/_autograd_function.pyx"],
        ),
        Extension(
            "candle._cython._autograd_ops",
            ["src/candle/_cython/_autograd_ops.pyx"],
        ),
        Extension(
            "candle._cython._functional_ops",
            ["src/candle/_cython/_functional_ops.pyx"],
        ),
        Extension(
            "candle._cython._fast_ops",
            ["src/candle/_cython/_fast_ops.pyx"],
        ),
        Extension(
            "candle._cython._tensor_api",
            ["src/candle/_cython/_tensor_api.pyx"],
        ),
        Extension(
            "candle._cython._cpu_kernels",
            ["src/candle/_cython/_cpu_kernels.pyx"],
            extra_compile_args=["-O3", "-ffast-math"],
        ),
        Extension(
            "candle.distributed._c10d",
            ["src/candle/distributed/_c10d.pyx"],
        ),
        Extension(
            "candle.distributed._c10d_gloo",
            ["src/candle/distributed/_c10d_gloo.pyx"],
        ),
        Extension(
            "candle._cython._mps_helpers",
            ["src/candle/_cython/_mps_helpers.pyx"],
        ),
        Extension(
            "candle._cython._mps_compute",
            ["src/candle/_cython/_mps_compute.pyx"],
        ),
        Extension(
            "candle._cython._mps_ops",
            ["src/candle/_cython/_mps_ops.pyx"],
        ),
        Extension(
            "candle._cython._dataloader_ops",
            ["src/candle/_cython/_dataloader_ops.pyx"],
        ),
        Extension(
            "candle._generated._functions_cy",
            ["src/candle/_generated/_functions_cy.pyx"],
        ),
        Extension(
            "candle._generated._variable_type_cy",
            ["src/candle/_generated/_variable_type_cy.pyx"],
        ),
    ]

# ---------------------------------------------------------------------------
# Linux-only extensions (NPU/CANN/HCCL — not available on macOS)
# ---------------------------------------------------------------------------
linux_only_extensions = []
if _system == "Linux":
    linux_only_extensions = [
        Extension(
            "candle._cython._aclnn_ffi",
            ["src/candle/_cython/_aclnn_ffi.pyx"],
            libraries=["dl"],
        ),
        Extension(
            "candle._cython._aclrt_ffi",
            ["src/candle/_cython/_aclrt_ffi.pyx"],
            libraries=["dl"],
        ),
        Extension(
            "candle._cython._aclgraph",
            ["src/candle/_cython/_aclgraph.pyx"],
            libraries=["dl"],
        ),
        Extension(
            "candle._cython._npu_ops",
            ["src/candle/_cython/_npu_ops.pyx"],
        ),
        Extension(
            "candle._cython._npu_storage",
            ["src/candle/_cython/_npu_storage.pyx"],
        ),
        Extension(
            "candle.distributed._c10d_hccl",
            ["src/candle/distributed/_c10d_hccl.pyx"],
        ),
    ]

ext_modules = cythonize(
    required_extensions + cross_platform_extensions + linux_only_extensions,
    compiler_directives={
        "language_level": "3",
        "boundscheck": False,
        "wraparound": False,
    },
)


class _BuildPy(build_py):
    """Copy the .pth file into build_lib so it lands in site-packages."""

    def run(self):
        super().run()
        src = os.path.join("src", "candle-torch-compat.pth")
        dst = os.path.join(self.build_lib, "candle-torch-compat.pth")
        shutil.copy2(src, dst)


setup(
    packages=find_packages(where="src", include=["candle*"]),
    package_dir={"": "src"},
    package_data={"candle": ["*.py", "*/*.py", "*/*/*.py", "*/*/*/*.py"]},
    py_modules=["_candle_torch_compat"],
    ext_modules=ext_modules,
    cmdclass={"build_py": _BuildPy},
)
