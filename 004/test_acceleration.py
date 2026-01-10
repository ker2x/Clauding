"""
Quick test to verify car accelerates with gas pedal.
Tests the action space and physics integration.
"""

import numpy as np
from env.car_racing import CarRacing

# Create environment
env = CarRacing(state_mode='vector', continuous=True)
state, _ = env.reset()

print("="*60)
print("Testing Car Acceleration")
print("="*60)

# Test 1: No input (should stay stationary or roll slightly)
print("\n1. Testing NO INPUT (gas=0, brake=0):")
action = np.array([0.0, 0.0, 0.0])  # [steering, gas, brake]
for i in range(5):
    state, reward, terminated, truncated, info = env.step(action)
    speed = np.sqrt(env.car.vx**2 + env.car.vy**2)
    print(f"   Step {i+1}: Speed = {speed:.2f} m/s, vx = {env.car.vx:.2f}, Reward = {reward:.2f}")

# Reset
state, _ = env.reset()

# Test 2: Full gas (should accelerate)
print("\n2. Testing FULL GAS (gas=1.0, brake=0):")
action = np.array([0.0, 1.0, 0.0])  # [steering, gas, brake]
for i in range(10):
    state, reward, terminated, truncated, info = env.step(action)
    speed = np.sqrt(env.car.vx**2 + env.car.vy**2)
    print(f"   Step {i+1}: Speed = {speed:.2f} m/s, vx = {env.car.vx:.2f}, Reward = {reward:.2f}")

# Reset
state, _ = env.reset()

# Test 3: Gas just above deadzone (should accelerate slowly)
print("\n3. Testing LOW GAS (gas=0.2, brake=0):")
action = np.array([0.0, 0.2, 0.0])
for i in range(10):
    state, reward, terminated, truncated, info = env.step(action)
    speed = np.sqrt(env.car.vx**2 + env.car.vy**2)
    print(f"   Step {i+1}: Speed = {speed:.2f} m/s, vx = {env.car.vx:.2f}, Reward = {reward:.2f}")

# Reset
state, _ = env.reset()

# Test 4: Gas with small brake (brake < deadzone, should still accelerate)
print("\n4. Testing GAS + SMALL BRAKE (gas=1.0, brake=0.05):")
action = np.array([0.0, 1.0, 0.05])
for i in range(10):
    state, reward, terminated, truncated, info = env.step(action)
    speed = np.sqrt(env.car.vx**2 + env.car.vy**2)
    print(f"   Step {i+1}: Speed = {speed:.2f} m/s, vx = {env.car.vx:.2f}, Reward = {reward:.2f}")

print("\n" + "="*60)
print("Test complete!")
print("If speed increases in tests 2-4, acceleration is working!")
print("="*60)

env.close()
