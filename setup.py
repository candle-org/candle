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

# ---------------------------------------------------------------------------
# Cython extension (optional — only built on Linux where CANN is available)
# ---------------------------------------------------------------------------
ext_modules = []
if platform.system() == "Linux":
    try:
        from Cython.Build import cythonize
        ext_modules = cythonize(
            [
                Extension(
                    "candle._cython._aclnn_ffi",
                    ["src/candle/_cython/_aclnn_ffi.pyx"],
                    libraries=["dl"],
                ),
                Extension(
                    "candle._cython._npu_ops",
                    ["src/candle/_cython/_npu_ops.pyx"],
                ),
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
                    "candle._cython._fast_ops",
                    ["src/candle/_cython/_fast_ops.pyx"],
                ),
            ],
            compiler_directives={
                "language_level": "3",
                "boundscheck": False,
                "wraparound": False,
            },
        )
    except ImportError:
        pass  # Cython not installed — skip extension, use fallback at runtime


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
