#!/usr/bin/env python3
"""
Analyze if agent is braking while steering (which locks tires).
"""

import torch
import numpy as np
from sac_agent import SACAgent
from preprocessing import make_carracing_env

def analyze_action_coordination():
    """Check if agent brakes while steering."""
    print("\nAnalyzing steering + braking coordination...")
    print("="*60)

    # Load agent
    checkpoint = torch.load('checkpoints/final_model.pt', map_location='cpu')
    state_mode = checkpoint['state_mode']
    action_dim = checkpoint.get('action_dim', 3)

    env = make_carracing_env(
        state_mode=state_mode,
        render_mode=None,
        terminate_stationary=True,
    )

    agent = SACAgent(
        state_shape=env.observation_space.shape,
        action_dim=action_dim,
        state_mode=state_mode
    )
    agent.load('checkpoints/final_model.pt')
    agent.actor.eval()

    # Test over multiple episodes
    problematic_steps = 0
    safe_steering_steps = 0
    strong_braking_steps = 0
    strong_steering_steps = 0

    for ep in range(3):
        state, _ = env.reset()

        for step in range(200):
            action = agent.select_action(state, evaluate=True)
            steering, gas, brake = action

            # Check for problematic combinations
            if abs(steering) > 0.3 and brake > 0.3:
                # Strong steering + strong braking = tire lockup!
                problematic_steps += 1
                if step % 40 == 0:
                    print(f"  ⚠️  Episode {ep+1}, Step {step}: STEERING={steering:.3f}, BRAKE={brake:.3f}")

            if abs(steering) > 0.5:
                strong_steering_steps += 1

            if brake > 0.5:
                strong_braking_steps += 1

            if abs(steering) > 0.3 and brake < 0.1:
                safe_steering_steps += 1

            next_state, _, terminated, truncated, _ = env.step(action)
            state = next_state

            if terminated or truncated:
                break

    total_steps = 600
    print(f"\nStatistics (across 600 steps):")
    print(f"  Strong steering + braking: {problematic_steps} steps ({100*problematic_steps/total_steps:.1f}%)")
    print(f"  Strong steering (alone): {safe_steering_steps} steps")
    print(f"  Strong braking: {strong_braking_steps} steps")
    print(f"  Strong steering signals: {strong_steering_steps} steps")

    print("\n" + "="*60)
    if problematic_steps > 20:
        print("❌ PROBLEM CONFIRMED: Agent brakes while steering!")
        print("\nThis causes tire lock-up and loss of control.")
        print("The agent needs to learn to NOT brake during sharp steering.")
        print("\nPossible fixes:")
        print("  1. Train longer (agent needs more experience)")
        print("  2. Add penalty for braking+steering combinations in reward")
        print("  3. Reduce brake effect in physics to prevent lock-up")
    else:
        print("✓ Agent mostly avoids braking while steering")

    env.close()

if __name__ == "__main__":
    analyze_action_coordination()
