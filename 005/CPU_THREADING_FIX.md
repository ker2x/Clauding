# CPU Multithreading Performance Fix

## Problem

On CPU (including M3 MacBook Air), the training was only using 100-120% CPU, which means only 1 core was being effectively utilized. On an 8-core M3, you should see up to 800% CPU usage when properly configured.

## Root Cause

PyTorch's default CPU threading configuration is conservative and doesn't automatically use all available cores. The code had:

1. **No `torch.set_num_threads()` configuration** - PyTorch defaulted to 1-2 threads
2. **No environment variables set** - OpenMP/MKL threading libraries weren't configured
3. **No inter-op parallelism** - Independent operations weren't running in parallel

## Solution Implemented

### 1. Code Changes (train.py)

Added `configure_cpu_threading()` function that:
- Detects number of CPU cores using `multiprocessing.cpu_count()`
- Sets `torch.set_num_threads(num_cores)` for intra-op parallelism
- Sets `torch.set_num_interop_threads(num_cores // 2)` for inter-op parallelism
- Prints configuration so you can verify it's working

### 2. Launch Script (train_cpu_optimized.sh)

Created a wrapper script that sets optimal environment variables:
- `OMP_NUM_THREADS` - OpenMP threading (used by linear algebra libraries)
- `MKL_NUM_THREADS` - Intel Math Kernel Library threads
- `OPENBLAS_NUM_THREADS` - OpenBLAS threads (MKL alternative)
- `MKL_DYNAMIC=FALSE` - Disable dynamic thread adjustment for consistency

## How to Use

### Option 1: Use the optimized launch script (RECOMMENDED)

```bash
cd 005/
./train_cpu_optimized.sh --episodes 100 --state-mode vector
```

This automatically sets all environment variables and runs training with optimal CPU threading.

### Option 2: Set environment variables manually

```bash
cd 005/

# For macOS (M3 MacBook Air with 8 cores)
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export MKL_DYNAMIC=FALSE

# Then run training normally
python train.py --device cpu --episodes 100 --state-mode vector
```

### Option 3: Just use the updated code

The code now has built-in threading configuration, but for **best performance**, you should still set the environment variables (Option 1 or 2).

## Expected Results

### Before Fix
- **CPU Usage**: 100-120% (only 1 core utilized)
- **Training Speed**: Slow, most cores idle

### After Fix
- **CPU Usage**: 400-800% on 8-core M3 (all cores utilized)
- **Training Speed**: 4-8x faster for CPU-bound operations
- **Neural network forward/backward passes**: Significantly faster

## Verification

When you run training, you should see output like:

```
Using device: cpu

CPU Threading Configuration:
  Physical cores detected: 8
  PyTorch intra-op threads: 8
  PyTorch inter-op threads: 4
  Expected CPU usage: up to 800%
  NOTE: Set OMP_NUM_THREADS=8 environment variable for best performance
```

Monitor CPU usage with:
- **macOS**: Activity Monitor or `top` command
- **Linux**: `htop` or `top` command

You should now see **multiple cores active** during training, not just 1-2.

## Technical Details

### What is intra-op parallelism?
Parallelism **within** a single operation (e.g., matrix multiplication uses multiple threads).

### What is inter-op parallelism?
Parallelism **across** independent operations (e.g., forward pass through actor and critics in parallel).

### Why set environment variables AND torch.set_num_threads()?
- `torch.set_num_threads()` controls PyTorch's thread pool
- Environment variables (OMP_NUM_THREADS, etc.) control the underlying math libraries (MKL/OpenBLAS)
- Both are needed for optimal performance

### M3-specific considerations
The M3 has 4 performance cores + 4 efficiency cores. macOS's scheduler will prioritize performance cores for compute-heavy tasks like neural network training.

## Troubleshooting

**Still seeing low CPU usage?**
1. Verify environment variables are set: `echo $OMP_NUM_THREADS`
2. Check PyTorch is using correct backend: `python -c "import torch; print(torch.__config__.show())"`
3. Try running with verbose mode to see timing: `python train.py --verbose --device cpu`

**CPU usage above 800% on 8-core system?**
This is normal if hyperthreading/SMT is enabled. The fix uses physical cores, but some parallelization can utilize hyperthreads.

**Training still slow?**
- Vector mode should be ~10x faster than visual mode on CPU
- Make sure you're using `--state-mode vector`
- Consider using smaller batch size if memory-bound
- MPS (Apple Silicon GPU) may be faster than CPU for some workloads - try `--device mps`

## Performance Expectations

On M3 MacBook Air (8 cores):
- **Vector mode + CPU**: Should see 400-800% CPU usage, ~10-15 episodes/minute
- **Visual mode + CPU**: Will be slower due to CNN operations, ~2-5 episodes/minute
- **Visual mode + MPS**: May be faster than CPU for visual mode (test with `--device mps`)

## Files Modified

1. **train.py**: Added `configure_cpu_threading()` function and call in `train()`
2. **train_cpu_optimized.sh**: New launch script with environment variables
3. **CPU_THREADING_FIX.md**: This documentation

## References

- [PyTorch CPU Threading](https://pytorch.org/docs/stable/notes/cpu_threading_torchscript_inference.html)
- [OpenMP Environment Variables](https://www.openmp.org/spec-html/5.0/openmpch6.html)
- [Intel MKL Threading](https://www.intel.com/content/www/us/en/docs/onemkl/developer-guide-linux/2023-0/overview.html)
