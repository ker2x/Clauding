"""
Debug script to see what actions the agent is producing.
"""

import numpy as np
import torch
from preprocessing import make_carracing_env
from sac_agent import SACAgent

# Create environment
env = make_carracing_env(state_mode='vector')

# Get state/action dimensions
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

print("="*60)
print("Agent Action Debugging")
print("="*60)
print(f"State dim: {state_dim}")
print(f"Action dim: {action_dim}")

# Create fresh agent (untrained)
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
agent = SACAgent(
    state_shape=state_dim,
    action_dim=action_dim,
    state_mode='vector',
    device=device
)

# Reset environment
state, _ = env.reset()

print("\n" + "="*60)
print("Testing UNTRAINED agent actions (should be random)")
print("="*60)

# Collect 50 actions from untrained agent
actions_list = []
for i in range(50):
    action = agent.select_action(state, evaluate=False)
    actions_list.append(action)

actions = np.array(actions_list)

# Analyze actions
print(f"\nSteering (dim 0): should be ~N(0, 1) bounded to [-1, 1]")
print(f"  Mean: {actions[:, 0].mean():.3f}")
print(f"  Std:  {actions[:, 0].std():.3f}")
print(f"  Min:  {actions[:, 0].min():.3f}")
print(f"  Max:  {actions[:, 0].max():.3f}")

print(f"\nGas (dim 1): should be ~U(0, 1) with scaled tanh")
print(f"  Mean: {actions[:, 1].mean():.3f} (should be ~0.5)")
print(f"  Std:  {actions[:, 1].std():.3f}")
print(f"  Min:  {actions[:, 1].min():.3f}")
print(f"  Max:  {actions[:, 1].max():.3f}")
print(f"  Count > 0.1: {(actions[:, 1] > 0.1).sum()}/50")

print(f"\nBrake (dim 2): should be ~U(0, 1) with scaled tanh")
print(f"  Mean: {actions[:, 2].mean():.3f} (should be ~0.5)")
print(f"  Std:  {actions[:, 2].std():.3f}")
print(f"  Min:  {actions[:, 2].min():.3f}")
print(f"  Max:  {actions[:, 2].max():.3f}")
print(f"  Count > 0.1: {(actions[:, 2] > 0.1).sum()}/50")

# Test actual execution
print("\n" + "="*60)
print("Testing actual environment execution with agent actions")
print("="*60)

state, _ = env.reset()
print("\nRunning 10 steps with agent actions:")

for i in range(10):
    action = agent.select_action(state, evaluate=False)
    state, reward, terminated, truncated, info = env.step(action)

    # Access car through wrapped env
    car = env.env.car  # env.env because it's wrapped with RewardShaper
    speed = np.sqrt(car.vx**2 + car.vy**2)

    print(f"Step {i+1}:")
    print(f"  Action: steer={action[0]:+.3f}, gas={action[1]:.3f}, brake={action[2]:.3f}")
    print(f"  Speed: {speed:.3f} m/s, vx={car.vx:.3f}, vy={car.vy:.3f}")
    print(f"  Reward: {reward:.2f}")

    if terminated or truncated:
        print(f"  Episode ended!")
        break

print("\n" + "="*60)
print("If gas values are consistently < 0.1 or speed stays ~0,")
print("there's an issue with action generation/application.")
print("="*60)

env.close()
