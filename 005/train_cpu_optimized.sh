#!/bin/bash
#
# Optimized CPU training script for CarRacing SAC agent
#
# This script sets optimal environment variables for CPU multithreading performance.
# Without these settings, PyTorch may only use 1-2 CPU cores, resulting in poor performance.
#

# Detect number of CPU cores
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    NUM_CORES=$(sysctl -n hw.physicalcpu)
else
    # Linux
    NUM_CORES=$(nproc --all)
fi

echo "============================================"
echo "CPU-Optimized Training Configuration"
echo "============================================"
echo "Detected CPU cores: $NUM_CORES"
echo ""

# Set OpenMP threading (used by MKL/OpenBLAS for linear algebra)
export OMP_NUM_THREADS=$NUM_CORES

# Set MKL threading (Intel Math Kernel Library)
export MKL_NUM_THREADS=$NUM_CORES

# Set OpenBLAS threading (alternative to MKL)
export OPENBLAS_NUM_THREADS=$NUM_CORES

# Disable MKL dynamic thread adjustment for consistent performance
export MKL_DYNAMIC=FALSE

# Set PyTorch to prefer simple threading for CPU
# (The code now sets torch.set_num_threads() internally)
export OMP_SCHEDULE=static
export OMP_PROC_BIND=close

echo "Environment variables set:"
echo "  OMP_NUM_THREADS=$OMP_NUM_THREADS"
echo "  MKL_NUM_THREADS=$MKL_NUM_THREADS"
echo "  OPENBLAS_NUM_THREADS=$OPENBLAS_NUM_THREADS"
echo "  MKL_DYNAMIC=$MKL_DYNAMIC"
echo ""
echo "Expected CPU usage: up to $((NUM_CORES * 100))%"
echo "============================================"
echo ""

# Run training with all arguments passed through
python train.py --device cpu "$@"
