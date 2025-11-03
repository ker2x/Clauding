"""
Alternative training script with faster epsilon decay

This makes the agent exploit learned strategies sooner,
which might help it discover better policies faster.
"""

import sys
from train import Trainer
import gymnasium as gym

# Register ALE environments
try:
    import ale_py
    gym.register_envs(ale_py)
except Exception:
    pass

# Create trainer
trainer = Trainer(
    env_name='ALE/Breakout-v5',
    num_episodes=1000,
    learning_starts=10000
)

# Modify agent's epsilon decay (faster)
print("Using FASTER epsilon decay (200k instead of 1M)")
print("Agent will exploit learned policy sooner")
trainer.agent.epsilon_decay = 200000  # Much faster decay

# Train
trainer.train()
