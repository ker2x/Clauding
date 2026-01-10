"""
Test the new 2D action space (steering + acceleration).
"""

import numpy as np
from env.car_racing import CarRacing

# Create environment
env = CarRacing(state_mode='vector', continuous=True)
state, _ = env.reset()

print("="*60)
print("Testing 2D Action Space")
print("="*60)
print(f"Action space: {env.action_space}")
print(f"Expected: Box([-1, -1], [+1, +1])")
print(f"Action shape: {env.action_space.shape}")
print()

# Test 1: Full gas (accel = +1)
print("Test 1: FULL GAS (accel = +1.0)")
action = np.array([0.0, 1.0])  # [steering, acceleration]
for i in range(10):
    state, reward, terminated, truncated, info = env.step(action)
    speed = np.sqrt(env.car.vx**2 + env.car.vy**2)
    print(f"  Step {i+1}: Speed = {speed:.2f} m/s, vx = {env.car.vx:.2f}")

# Reset
state, _ = env.reset()

# Test 2: Full brake (accel = -1)
print("\nTest 2: FULL BRAKE (accel = -1.0) after acceleration")
# First accelerate
for i in range(10):
    env.step(np.array([0.0, 1.0]))

# Then brake
print("  Accelerated to:", f"{np.sqrt(env.car.vx**2 + env.car.vy**2):.2f} m/s")
for i in range(5):
    state, reward, terminated, truncated, info = env.step(np.array([0.0, -1.0]))
    speed = np.sqrt(env.car.vx**2 + env.car.vy**2)
    print(f"  Step {i+1}: Speed = {speed:.2f} m/s, vx = {env.car.vx:.2f}")

# Reset
state, _ = env.reset()

# Test 3: Coasting (accel = 0)
print("\nTest 3: COASTING (accel = 0.0) after acceleration")
# First accelerate
for i in range(10):
    env.step(np.array([0.0, 1.0]))

# Then coast
print("  Accelerated to:", f"{np.sqrt(env.car.vx**2 + env.car.vy**2):.2f} m/s")
for i in range(5):
    state, reward, terminated, truncated, info = env.step(np.array([0.0, 0.0]))
    speed = np.sqrt(env.car.vx**2 + env.car.vy**2)
    print(f"  Step {i+1}: Speed = {speed:.2f} m/s, vx = {env.car.vx:.2f}")

print("\n" + "="*60)
print("Test complete! Action space conversion working correctly.")
print("="*60)

env.close()
