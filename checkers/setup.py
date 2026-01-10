
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import sys
import setuptools

# Define the extension module
ext_modules = [
    Extension(
        "checkers_cpp",
        tuple([
            "checkers8x8/cpp/bindings.cpp",
            "checkers8x8/cpp/game.cpp",
            "checkers8x8/cpp/mcts.cpp",
        ]),
        include_dirs=["checkers8x8/cpp"],
        language="c++",
        extra_compile_args=["-std=c++17", "-O3"],
    ),
]

# Custom build class to handle pybind11 include path
class BuildExt(build_ext):
    """A custom build extension for adding compiler-specific options."""
    def build_extensions(self):
        ct = self.compiler.compiler_type
        if ct == "unix":
            self.extensions[0].extra_compile_args.append("-fvisibility=hidden")
        
        # Add pybind11 include path
        import pybind11
        self.extensions[0].include_dirs.append(pybind11.get_include())
        
        build_ext.build_extensions(self)

setup(
    name="checkers_cpp",
    version="0.1.0",
    author="Antigravity",
    description="Fast C++ implementation of Checkers Game and MCTS",
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExt},
    zip_safe=False,
    install_requires=["pybind11>=2.10.0"],
    setup_requires=["pybind11>=2.10.0"],
)
