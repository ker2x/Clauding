@echo off
REM Build script for Checkers C++ extension (Windows)

echo Building Checkers C++ Extension...
echo ==================================

REM Check if pybind11 is installed
python -c "import pybind11" 2>nul
if errorlevel 1 (
    echo Error: pybind11 not found. Installing...
    pip install pybind11>=2.10.0
)

REM Build extension
echo Compiling C++ extension...
python setup.py build_ext --inplace

REM Verify
python -c "import checkers_cpp" 2>nul
if errorlevel 1 (
    echo.
    echo Build completed but import failed.
    echo Please check for error messages above.
    exit /b 1
) else (
    echo.
    echo Success! C++ extension built and loaded.
    echo.
    echo The MCTS will now use the fast C++ implementation.
    echo Expected speedup: 10-100x over pure Python.
)
