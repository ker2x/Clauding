"""
Inspect a trained model checkpoint

This script shows you what's in a checkpoint and helps you decide
whether to resume training or start fresh.
"""

import argparse
import torch
import numpy as np


def inspect_checkpoint(checkpoint_path):
    """Load and inspect a checkpoint file"""
    print("="*60)
    print("CHECKPOINT INSPECTION")
    print("="*60)

    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    print(f"\nCheckpoint: {checkpoint_path}")
    print("\nContents:")
    for key in checkpoint.keys():
        print(f"  - {key}")

    # Steps trained
    steps_done = checkpoint.get('steps_done', 0)
    print(f"\nTraining Progress:")
    print(f"  Total steps trained: {steps_done:,}")

    # Calculate current epsilon
    epsilon_start = 1.0
    epsilon_end = 0.01
    epsilon_decay = 1000000

    current_epsilon = epsilon_end + (epsilon_start - epsilon_end) * \
                     np.exp(-1. * steps_done / epsilon_decay)

    print(f"\nExploration (Epsilon):")
    print(f"  Current epsilon: {current_epsilon:.4f}")
    print(f"  Means: {current_epsilon*100:.1f}% random actions, {(1-current_epsilon)*100:.1f}% learned actions")

    if current_epsilon > 0.5:
        exploration_level = "HIGH - Still exploring a lot"
    elif current_epsilon > 0.2:
        exploration_level = "MEDIUM - Balanced exploration/exploitation"
    elif current_epsilon > 0.05:
        exploration_level = "LOW - Mostly using learned policy"
    else:
        exploration_level = "VERY LOW - Almost no exploration"

    print(f"  Exploration level: {exploration_level}")

    # Network weights info
    policy_params = checkpoint['policy_net']
    total_params = sum(p.numel() for p in policy_params.values())

    print(f"\nNeural Network:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Network has been trained")

    # Recommendations
    print("\n" + "="*60)
    print("RECOMMENDATIONS")
    print("="*60)

    if current_epsilon > 0.7:
        print("\n✅ Resume Training (Normal)")
        print("  Your model is still exploring heavily.")
        print("  Command: python3 train.py --resume", checkpoint_path, "--episodes 500")

        print("\n❌ Don't Reset Epsilon")
        print("  No need - epsilon is already high")

    elif current_epsilon > 0.3:
        print("\n✅ Resume Training (With More Exploration)")
        print("  Your model needs more exploration to find better strategies.")
        print("  Command: python3 train.py --resume", checkpoint_path, "--reset-epsilon --episodes 500")

        print("\n⚠️  Alternative: Continue Without Reset")
        print("  If you think it just needs more time")
        print("  Command: python3 train.py --resume", checkpoint_path, "--episodes 500")

    else:
        print("\n⚠️  Low Exploration - Might Be Stuck")
        print("  Your model is not exploring much anymore.")
        print("  This is why it learned a 'lazy' strategy!")

        print("\n✅ Option 1: Resume + Reset Epsilon (Recommended)")
        print("  Force it to explore more aggressive strategies")
        print("  Command: python3 train.py --resume", checkpoint_path, "--reset-epsilon --episodes 500")

        print("\n✅ Option 2: Start Fresh with Slower Decay")
        print("  New model that explores longer")
        print("  Modify dqn_agent.py: epsilon_decay=2000000 (instead of 1000000)")
        print("  Command: python3 train.py --episodes 1000")

        print("\n⚠️  Option 3: Continue As-Is (Not Recommended)")
        print("  Unlikely to improve much")
        print("  Command: python3 train.py --resume", checkpoint_path, "--episodes 500")

    print("\n" + "="*60)


def main():
    parser = argparse.ArgumentParser(description='Inspect DQN checkpoint')
    parser.add_argument('checkpoint', type=str, help='Path to checkpoint file')
    args = parser.parse_args()

    inspect_checkpoint(args.checkpoint)


if __name__ == '__main__':
    main()
