#!/usr/bin/env python3
"""
Benchmark powertrain performance to measure optimization impact.
"""
import time
import sys
sys.path.insert(0, 'env')
from mx5_powertrain import MX5Powertrain

def benchmark_powertrain(iterations=100000):
    """Benchmark powertrain calls."""
    powertrain = MX5Powertrain()

    print(f"Benchmarking {iterations:,} powertrain calls...")
    print("-" * 60)

    # Warm up
    for _ in range(1000):
        powertrain.get_wheel_torque(0.8, 500)
        powertrain.update(0.02, 500)

    # Benchmark get_wheel_torque + update (typical physics loop)
    start = time.perf_counter()
    for i in range(iterations):
        throttle = 0.5 + 0.5 * (i % 100) / 100.0  # Vary throttle
        wheel_rpm = 300 + (i % 500)  # Vary RPM
        powertrain.get_wheel_torque(throttle, wheel_rpm)
        powertrain.update(0.02, wheel_rpm)

    elapsed = time.perf_counter() - start
    per_call = (elapsed / iterations) * 1e6  # microseconds
    calls_per_sec = iterations / elapsed

    print(f"Total time:        {elapsed:.3f} seconds")
    print(f"Time per call:     {per_call:.2f} µs")
    print(f"Calls per second:  {calls_per_sec:,.0f}")
    print(f"Frames per second: {calls_per_sec/50:,.0f} (at 50 physics steps/frame)")
    print("-" * 60)

if __name__ == '__main__':
    benchmark_powertrain()
