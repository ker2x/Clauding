# C++ Extension for Checkers MCTS

This directory contains a high-performance C++ implementation of the Checkers game engine and MCTS algorithm using PyBind11.

## Performance Benefits

The C++ extension provides **10-100x speedup** over pure Python MCTS:
- Batched neural network evaluation (configurable batch size)
- Efficient game state representation using bitboards
- Virtual loss mechanism for better exploration
- Zero-copy operations where possible

## Prerequisites

```bash
pip install pybind11>=2.10.0
```

You'll also need a C++17 compatible compiler:
- **Linux**: GCC 7+ or Clang 5+
- **macOS**: Xcode 10+
- **Windows**: Visual Studio 2017+

## Building the Extension

### Quick Build

From the `checkers/` directory:

```bash
python setup.py build_ext --inplace
```

This will create `checkers_cpp.*.so` (Linux/macOS) or `checkers_cpp.*.pyd` (Windows) in the checkers directory.

### Install

For a proper installation:

```bash
pip install -e .
```

### Verify Installation

```bash
python -c "import checkers_cpp; print('C++ extension loaded successfully!')"
```

## Usage

The MCTS class will automatically use the C++ extension if available:

```python
from checkers8x8.mcts.mcts import MCTS
from checkers8x8.network.resnet import CheckersNetwork
from checkers8x8.engine.game import CheckersGame

network = CheckersNetwork()
game = CheckersGame()

# Automatically uses C++ if available, falls back to Python
mcts = MCTS(
    network=network,
    num_simulations=800,  # Much faster with C++!
    batch_size=32  # Configurable batch size
)

policy = mcts.search(game)
```

## Batch Size Tuning

The `batch_size` parameter controls how many leaf nodes are evaluated in parallel:

- **Small batch (8-16)**: Lower latency, good for CPU inference
- **Medium batch (32-64)**: Balanced, good for most GPUs
- **Large batch (128+)**: Maximum throughput for powerful GPUs

Tune based on your hardware and network size.

## Troubleshooting

### Import Error

If you see `Warning: checkers_cpp not found, using slow Python MCTS`:
1. Make sure you built the extension (`python setup.py build_ext --inplace`)
2. Check that the `.so`/`.pyd` file exists in the checkers directory
3. Verify Python can find it: `python -c "import sys; print(sys.path)"`

### Build Errors

**Missing pybind11**:
```bash
pip install pybind11
```

**Compiler not found**:
- Linux: `sudo apt-get install build-essential`
- macOS: Install Xcode Command Line Tools
- Windows: Install Visual Studio Build Tools

### Performance Issues

If C++ MCTS is still slow:
1. Check if GPU is being used: `nvidia-smi` (for NVIDIA)
2. Increase `batch_size` for better GPU utilization
3. Profile with: `python -m cProfile your_script.py`

## Architecture

- `game.cpp/h`: Fast bitboard-based game engine
- `mcts.cpp/h`: MCTS with batched evaluation and virtual loss
- `bindings.cpp`: PyBind11 interface to Python

## Performance Tips

1. **Use C++ extension**: 10-100x faster than Python
2. **Tune batch_size**: Match to your GPU capacity
3. **Use GPU inference**: Move network to CUDA
4. **Increase num_simulations**: Better play quality
5. **Profile bottlenecks**: Use `cProfile` or `nvprof`
