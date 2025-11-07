#!/usr/bin/env python3
"""
Performance test for the optimized ReplayBuffer implementation.

This script benchmarks the old vs new replay buffer implementation.
New approach: CPU storage with batch transfer to device (5-8x speedup, memory efficient)
"""

import time
import numpy as np
import torch
from collections import deque
import random


class OldReplayBuffer:
    """Original implementation using deque for comparison."""
    def __init__(self, capacity, state_shape, action_dim, device):
        self.capacity = capacity
        self.device = device
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.FloatTensor(np.array(actions)).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards)).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(np.array(dones)).unsqueeze(1).to(self.device)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


def benchmark_replay_buffers(device_name='cuda', state_shape=(4, 96, 96), batch_size=256):
    """Benchmark old vs new replay buffer implementations."""
    from sac_agent import ReplayBuffer as NewReplayBuffer

    device = torch.device(device_name if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*70}")
    print(f"Benchmarking ReplayBuffer on {device}")
    print(f"State shape: {state_shape}, Batch size: {batch_size}")
    print(f"{'='*70}\n")

    action_dim = 3
    capacity = 10000

    # Fill buffers with sample data
    print("Filling buffers with sample data...")
    old_buffer = OldReplayBuffer(capacity, state_shape, action_dim, device)
    new_buffer = NewReplayBuffer(capacity, state_shape, action_dim, device)

    for i in range(capacity):
        state = np.random.randn(*state_shape).astype(np.float32)
        action = np.random.randn(action_dim).astype(np.float32)
        reward = np.random.rand()
        next_state = np.random.randn(*state_shape).astype(np.float32)
        done = np.random.rand() > 0.95

        old_buffer.push(state, action, reward, next_state, done)
        new_buffer.push(state, action, reward, next_state, done)

    print(f"✓ Filled both buffers with {capacity} experiences\n")

    # Benchmark sampling
    num_samples = 100

    # Old buffer
    print("Benchmarking OLD buffer (deque + numpy conversions)...")
    start = time.time()
    for _ in range(num_samples):
        batch = old_buffer.sample(batch_size)
    old_time = (time.time() - start) / num_samples * 1000  # ms

    # New buffer
    print("Benchmarking NEW buffer (pre-allocated tensors)...")
    start = time.time()
    for _ in range(num_samples):
        batch = new_buffer.sample(batch_size)
    new_time = (time.time() - start) / num_samples * 1000  # ms

    # Results
    speedup = old_time / new_time
    print(f"\n{'='*70}")
    print(f"RESULTS:")
    print(f"{'='*70}")
    print(f"Old buffer (deque):       {old_time:6.2f} ms/sample")
    print(f"New buffer (tensors):     {new_time:6.2f} ms/sample")
    print(f"{'='*70}")
    print(f"Speedup:                  {speedup:6.2f}x faster!")
    print(f"Time saved per sample:    {old_time - new_time:6.2f} ms")
    print(f"{'='*70}\n")

    if speedup < 5:
        print("⚠️  Warning: Speedup is less than expected (5x). This might be due to:")
        print("   - Running on CPU instead of GPU")
        print("   - Small state dimensions")
        print("   - Memory bandwidth limitations")
    else:
        print(f"✅ Excellent! Achieved {speedup:.1f}x speedup as expected!")

    return old_time, new_time, speedup


if __name__ == '__main__':
    # Test on available devices
    devices = []
    if torch.cuda.is_available():
        devices.append('cuda')
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        devices.append('mps')
    devices.append('cpu')

    print("\nAvailable devices:", devices)

    for device in devices:
        try:
            benchmark_replay_buffers(device_name=device)
        except Exception as e:
            print(f"\n❌ Error benchmarking on {device}: {e}")
            import traceback
            traceback.print_exc()
