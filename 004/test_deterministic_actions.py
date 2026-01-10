"""
Test deterministic vs stochastic actions.
"""

import numpy as np
import torch
from preprocessing import make_carracing_env
from sac_agent import SACAgent

# Create environment
env = make_carracing_env(state_mode='vector')
state, _ = env.reset()

# Create fresh agent
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
agent = SACAgent(
    state_shape=env.observation_space.shape[0],
    action_dim=env.action_space.shape[0],
    state_mode='vector',
    device=device
)

print("="*60)
print("Comparing Stochastic vs Deterministic Actions")
print("="*60)

# Test stochastic actions (sampling)
print("\nSTOCHASTIC (evaluate=False, sampling from distribution):")
stoch_actions = []
for i in range(20):
    action = agent.select_action(state, evaluate=False)
    stoch_actions.append(action)

stoch_actions = np.array(stoch_actions)
print(f"Gas:   mean={stoch_actions[:, 1].mean():.3f}, std={stoch_actions[:, 1].std():.3f}")
print(f"Brake: mean={stoch_actions[:, 2].mean():.3f}, std={stoch_actions[:, 2].std():.3f}")

# Test deterministic actions (mean)
print("\nDETERMINISTIC (evaluate=True, using mean):")
det_actions = []
for i in range(20):
    action = agent.select_action(state, evaluate=True)
    det_actions.append(action)

det_actions = np.array(det_actions)
print(f"Gas:   mean={det_actions[:, 1].mean():.3f}, std={det_actions[:, 1].std():.3f}")
print(f"Brake: mean={det_actions[:, 2].mean():.3f}, std={det_actions[:, 2].std():.3f}")

print("\n" + "="*60)
print("Testing deterministic actions in environment:")
print("="*60)

state, _ = env.reset()
for i in range(10):
    action = agent.select_action(state, evaluate=True)
    state, reward, terminated, truncated, info = env.step(action)

    car = env.env.car
    speed = np.sqrt(car.vx**2 + car.vy**2)

    print(f"Step {i+1}: gas={action[1]:.3f}, brake={action[2]:.3f}, speed={speed:.3f} m/s")

    if terminated or truncated:
        break

env.close()
