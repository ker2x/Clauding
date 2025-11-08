#!/usr/bin/env python3
"""
Quick test to verify CPU threading configuration is working.

This script performs a series of matrix operations and shows:
1. Current PyTorch threading settings
2. CPU usage during computation
3. Performance comparison
"""

import torch
import multiprocessing
import time
import os

print("=" * 70)
print("PyTorch CPU Threading Test")
print("=" * 70)

# Check current settings
print("\nCurrent Configuration:")
print(f"  CPU cores available: {multiprocessing.cpu_count()}")
print(f"  PyTorch intra-op threads: {torch.get_num_threads()}")
print(f"  PyTorch inter-op threads: {torch.get_num_interop_threads()}")

# Check environment variables
env_vars = ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_DYNAMIC']
print("\nEnvironment Variables:")
for var in env_vars:
    value = os.environ.get(var, 'NOT SET')
    print(f"  {var}: {value}")

print("\n" + "=" * 70)
print("Running Performance Test...")
print("=" * 70)

# Create test data (similar to SAC vector mode)
batch_size = 256
state_dim = 36
hidden_dim = 256
action_dim = 3

# Simulate SAC network operations
print("\nSimulating SAC training step (vector mode):")
print(f"  Batch size: {batch_size}")
print(f"  State dim: {state_dim}")
print(f"  Hidden dim: {hidden_dim}")
print(f"  Action dim: {action_dim}")

# Create simple network layers similar to VectorActor/VectorCritic
fc1 = torch.nn.Linear(state_dim, hidden_dim)
fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
fc3 = torch.nn.Linear(hidden_dim, action_dim)

# Create random batch
states = torch.randn(batch_size, state_dim)

# Warm-up
for _ in range(5):
    x = torch.relu(fc1(states))
    x = torch.relu(fc2(x))
    output = fc3(x)

# Timed runs
print("\nPerformance Test (100 iterations):")
num_iterations = 100

start = time.perf_counter()
for _ in range(num_iterations):
    x = torch.relu(fc1(states))
    x = torch.relu(fc2(x))
    output = fc3(x)
    loss = output.mean()
    loss.backward()
    fc1.zero_grad()
    fc2.zero_grad()
    fc3.zero_grad()
end = time.perf_counter()

total_time = (end - start) * 1000  # ms
avg_time = total_time / num_iterations

print(f"  Total time: {total_time:.2f} ms")
print(f"  Avg per iteration: {avg_time:.2f} ms")
print(f"  Throughput: {1000/avg_time:.1f} iterations/sec")

print("\n" + "=" * 70)
print("Interpretation:")
print("=" * 70)

if torch.get_num_threads() == 1:
    print("⚠️  WARNING: Only using 1 thread!")
    print("   Expected CPU usage: ~100%")
    print("   FIX: Run with ./train_cpu_optimized.sh or set environment variables")
elif torch.get_num_threads() >= 4:
    print("✓ Good threading configuration!")
    print(f"   Expected CPU usage: up to ~{torch.get_num_threads() * 100}%")
    print("   Multiple cores should be active during training")
else:
    print("⚠️  Moderate threading configuration")
    print(f"   Using {torch.get_num_threads()} threads")
    print("   Consider setting OMP_NUM_THREADS for better performance")

print("\nMonitor CPU usage during training with:")
print("  - macOS: Activity Monitor or 'top' command")
print("  - Linux: 'htop' or 'top' command")
print("\nYou should see multiple CPU cores active, not just 1-2 cores.")
print("=" * 70)
