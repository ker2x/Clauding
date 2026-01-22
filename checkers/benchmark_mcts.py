#!/usr/bin/env python
"""
Benchmark script for MCTS performance comparison.

This script compares different MCTS configurations to help you:
1. Verify C++ extension is working
2. Find optimal batch_size for your hardware
3. Measure speedup from optimizations

Usage:
    python benchmark_mcts.py
"""

import time
import torch
import numpy as np
from checkers8x8.engine.game import CheckersGame
from checkers8x8.network.resnet import CheckersNetwork
from checkers8x8.mcts.mcts import MCTS, USE_CPP

def benchmark_config(name, mcts, game, num_moves=5):
    """Benchmark a single MCTS configuration."""
    print(f"\n{name}")
    print("-" * 60)

    times = []
    test_game = game.clone()

    for i in range(num_moves):
        start = time.time()
        policy = mcts.search(test_game, add_noise=False)
        elapsed = time.time() - start
        times.append(elapsed)

        # Make a move
        action = np.argmax(policy)
        if not test_game.make_action(action):
            break

        print(f"  Move {i+1}: {elapsed*1000:.1f}ms")

    avg_time = np.mean(times)
    print(f"  Average: {avg_time*1000:.1f}ms")
    print(f"  Simulations/sec: {mcts.num_simulations/avg_time:.0f}")

    return avg_time

def main():
    print("=" * 60)
    print("MCTS Performance Benchmark")
    print("=" * 60)

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    print(f"C++ Extension: {'✓ Available' if USE_CPP else '✗ Not found (using Python)'}")

    # Create network and game
    print("\nInitializing network...")
    network = CheckersNetwork(num_filters=128, num_res_blocks=6, policy_size=128)
    network.to(device)
    network.eval()

    game = CheckersGame()

    # Benchmark configurations
    configs = [
        {
            'name': '1. Baseline (100 sims, batch=8)',
            'num_simulations': 100,
            'batch_size': 8
        },
        {
            'name': '2. Medium (400 sims, batch=16)',
            'num_simulations': 400,
            'batch_size': 16
        },
        {
            'name': '3. Large batch (400 sims, batch=32)',
            'num_simulations': 400,
            'batch_size': 32
        },
        {
            'name': '4. High quality (800 sims, batch=32)',
            'num_simulations': 800,
            'batch_size': 32
        },
    ]

    results = []

    for config in configs:
        mcts = MCTS(
            network=network,
            c_puct=1.0,
            num_simulations=config['num_simulations'],
            batch_size=config['batch_size'],
            device=device
        )

        avg_time = benchmark_config(config['name'], mcts, game)
        results.append({
            'name': config['name'],
            'time': avg_time,
            'sims': config['num_simulations']
        })

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    baseline_time = results[0]['time']

    for r in results:
        speedup = baseline_time / r['time']
        efficiency = r['sims'] / r['time']  # sims per second
        print(f"\n{r['name']}")
        print(f"  Time: {r['time']*1000:.1f}ms")
        print(f"  Speedup: {speedup:.2f}x")
        print(f"  Efficiency: {efficiency:.0f} sims/sec")

    # Recommendations
    print("\n" + "=" * 60)
    print("RECOMMENDATIONS")
    print("=" * 60)

    if not USE_CPP:
        print("\n⚠️  C++ extension not found!")
        print("   Build it for 10-100x speedup:")
        print("   $ cd checkers && ./build_cpp.sh")
    else:
        print("\n✓ C++ extension is active")

    if str(device) == 'cpu':
        print("\n⚠️  Running on CPU")
        print("   Use GPU for 5-20x speedup:")
        print("   - Install CUDA + PyTorch with CUDA support")
    else:
        print("\n✓ Running on GPU")

    # Find best config
    best = max(results, key=lambda x: x['sims'] / x['time'])
    print(f"\n✓ Best efficiency: {best['name']}")
    print(f"  ({best['sims']/best['time']:.0f} simulations/sec)")

    # Batch size recommendation
    print("\n📊 Batch Size Tuning:")
    print("   Try increasing batch_size until:")
    print("   - GPU memory full (reduce if OOM)")
    print("   - No more speedup (hit other bottleneck)")

    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()
