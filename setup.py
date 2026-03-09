"""Package discovery for candle.

The main project metadata lives in pyproject.toml.  This file handles package
discovery because setuptools' ``packages.find`` in pyproject.toml conflicts
with ``py_modules`` — we need both to install the candle package tree AND the
standalone ``_candle_torch_compat`` bootstrap module and its ``.pth`` trigger.
"""

import os
import shutil

from setuptools import setup, find_packages
from setuptools.command.build_py import build_py


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
    cmdclass={"build_py": _BuildPy},
)
