#!/usr/bin/env python3
"""
Diagnose steering issues by inspecting checkpoint and action distributions.
"""

import torch
import numpy as np
import argparse
from sac_agent import SACAgent
from preprocessing import make_carracing_env

def diagnose_checkpoint(checkpoint_path):
    """Check what's in the checkpoint."""
    print(f"\n{'='*60}")
    print(f"CHECKPOINT DIAGNOSTICS: {checkpoint_path}")
    print(f"{'='*60}\n")

    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Check state mode
    if 'state_mode' in checkpoint:
        state_mode = checkpoint['state_mode']
        print(f"✓ State mode stored in checkpoint: {state_mode}")
    else:
        print("✗ State mode NOT stored in checkpoint, detecting from architecture...")
        actor_state = checkpoint['actor']
        if 'conv1.weight' in actor_state:
            state_mode = 'visual'
            print(f"  Auto-detected: visual (has conv layers)")
        else:
            state_mode = 'vector'
            print(f"  Auto-detected: vector (no conv layers)")

    # Check action dimension
    action_dim = checkpoint.get('action_dim', 3)
    print(f"Action dimension: {action_dim}")

    # Check keys
    print(f"\nCheckpoint contains:")
    for key in checkpoint.keys():
        if isinstance(checkpoint[key], dict):
            print(f"  - {key}: dict with {len(checkpoint[key])} entries")
        elif isinstance(checkpoint[key], torch.Tensor):
            print(f"  - {key}: tensor shape {checkpoint[key].shape}")
        else:
            print(f"  - {key}: {type(checkpoint[key]).__name__}")

    return state_mode, action_dim

def test_actions(checkpoint_path, episodes=3):
    """Test what actions the agent actually outputs."""
    print(f"\n{'='*60}")
    print(f"ACTION OUTPUT TEST")
    print(f"{'='*60}\n")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Detect state mode
    if 'state_mode' in checkpoint:
        state_mode = checkpoint['state_mode']
    else:
        actor_state = checkpoint['actor']
        state_mode = 'visual' if 'conv1.weight' in actor_state else 'vector'

    action_dim = checkpoint.get('action_dim', 3)

    # Create environment
    env = make_carracing_env(
        stack_size=4,
        terminate_stationary=True,
        state_mode=state_mode,
        render_mode=None
    )

    # Create and load agent
    agent = SACAgent(
        state_shape=env.observation_space.shape,
        action_dim=action_dim,
        state_mode=state_mode
    )
    agent.load(checkpoint_path)
    agent.actor.eval()

    print(f"State mode: {state_mode}")
    print(f"Action space: {env.action_space}")

    # Test actions over multiple episodes
    all_steering_actions = []
    all_gas_actions = []
    all_brake_actions = []

    for ep in range(episodes):
        state, _ = env.reset()
        steering_actions = []
        gas_actions = []
        brake_actions = []

        for step in range(200):  # Sample first 200 steps
            # Get deterministic action
            action = agent.select_action(state, evaluate=True)

            steering_actions.append(action[0])
            gas_actions.append(action[1])
            brake_actions.append(action[2])

            next_state, _, terminated, truncated, _ = env.step(action)
            state = next_state

            if terminated or truncated:
                break

        all_steering_actions.extend(steering_actions)
        all_gas_actions.extend(gas_actions)
        all_brake_actions.extend(brake_actions)

        print(f"\nEpisode {ep+1}:")
        print(f"  Steering: min={min(steering_actions):.4f}, max={max(steering_actions):.4f}, "
              f"mean={np.mean(steering_actions):.4f}, std={np.std(steering_actions):.4f}")
        print(f"  Gas:      min={min(gas_actions):.4f}, max={max(gas_actions):.4f}, "
              f"mean={np.mean(gas_actions):.4f}, std={np.std(gas_actions):.4f}")
        print(f"  Brake:    min={min(brake_actions):.4f}, max={max(brake_actions):.4f}, "
              f"mean={np.mean(brake_actions):.4f}, std={np.std(brake_actions):.4f}")

    # Overall statistics
    print(f"\n{'='*60}")
    print(f"Overall Statistics (across {len(all_steering_actions)} steps):")
    print(f"{'='*60}")
    print(f"Steering: min={min(all_steering_actions):.4f}, max={max(all_steering_actions):.4f}, "
          f"mean={np.mean(all_steering_actions):.4f}, std={np.std(all_steering_actions):.4f}")
    print(f"Gas:      min={min(all_gas_actions):.4f}, max={max(all_gas_actions):.4f}, "
          f"mean={np.mean(all_gas_actions):.4f}, std={np.std(all_gas_actions):.4f}")
    print(f"Brake:    min={min(all_brake_actions):.4f}, max={max(all_brake_actions):.4f}, "
          f"mean={np.mean(all_brake_actions):.4f}, std={np.std(all_brake_actions):.4f}")

    # Check if steering is actually being used
    print(f"\n{'='*60}")
    print(f"ANALYSIS:")
    print(f"{'='*60}")

    steering_zeros = sum(1 for s in all_steering_actions if abs(s) < 0.01)
    steering_pct_zero = 100 * steering_zeros / len(all_steering_actions)
    print(f"Steering ~0 ({steering_zeros}/{len(all_steering_actions)} steps, {steering_pct_zero:.1f}%)")

    if steering_pct_zero > 90:
        print(f"\n⚠️  WARNING: Agent is outputting almost NO steering!")
        print(f"   This could mean:")
        print(f"   1. Agent was never trained on proper steering (training issue)")
        print(f"   2. State mode mismatch (agent trained on wrong state representation)")
        print(f"   3. Network architecture issue")

    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pt',
                       help='Path to checkpoint to diagnose')
    parser.add_argument('--episodes', type=int, default=3,
                       help='Number of episodes to test')
    args = parser.parse_args()

    # Diagnose checkpoint
    state_mode, action_dim = diagnose_checkpoint(args.checkpoint)

    # Test actions
    test_actions(args.checkpoint, episodes=args.episodes)
