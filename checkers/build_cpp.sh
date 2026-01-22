#!/bin/bash
# Build script for Checkers C++ extension

set -e  # Exit on error

echo "Building Checkers C++ Extension..."
echo "=================================="

# Check if pybind11 is installed
if ! python -c "import pybind11" 2>/dev/null; then
    echo "Error: pybind11 not found. Installing..."
    pip install pybind11>=2.10.0
fi

# Build extension
echo "Compiling C++ extension..."
python setup.py build_ext --inplace

# Verify
if python -c "import checkers_cpp" 2>/dev/null; then
    echo ""
    echo "✓ Success! C++ extension built and loaded."
    echo ""
    echo "The MCTS will now use the fast C++ implementation."
    echo "Expected speedup: 10-100x over pure Python."
else
    echo ""
    echo "✗ Build completed but import failed."
    echo "Please check for error messages above."
    exit 1
fi
