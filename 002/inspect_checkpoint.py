"""
Inspect saved DDQN agent checkpoints.

This script provides detailed information about a checkpoint including:
- Training progress (steps, epsilon)
- Network architecture
- Training recommendations

Useful for understanding agent state before resuming training or evaluation.

Usage:
    python inspect_checkpoint.py checkpoints/final_model.pt
    python inspect_checkpoint.py checkpoints/checkpoint_ep1000.pt
"""

import argparse
import torch
import sys


def inspect_checkpoint(checkpoint_path):
    """
    Inspect a checkpoint file and print detailed information.

    Args:
        checkpoint_path: Path to checkpoint file
    """
    print("=" * 60)
    print("DDQN Checkpoint Inspector")
    print("=" * 60)
    print(f"Checkpoint: {checkpoint_path}\n")

    try:
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # Extract information
        steps_done = checkpoint.get('steps_done', 'Unknown')
        epsilon = checkpoint.get('epsilon', 'Unknown')

        print("Training State:")
        print("-" * 60)
        print(f"Steps trained: {steps_done:,}" if isinstance(steps_done, int) else f"Steps trained: {steps_done}")
        print(f"Current epsilon: {epsilon:.6f}" if isinstance(epsilon, (int, float)) else f"Current epsilon: {epsilon}")

        # Calculate how much epsilon has decayed (assuming default decay)
        if isinstance(epsilon, (int, float)) and isinstance(steps_done, int):
            epsilon_decay_steps = 1_000_000  # default value
            decay_progress = (steps_done / epsilon_decay_steps) * 100
            print(f"Epsilon decay progress: {decay_progress:.1f}% of {epsilon_decay_steps:,} steps")

        print()

        # Network information
        if 'policy_net_state_dict' in checkpoint:
            print("Network Architecture:")
            print("-" * 60)
            policy_state = checkpoint['policy_net_state_dict']

            # Count parameters
            total_params = sum(p.numel() for p in policy_state.values())
            print(f"Total parameters: {total_params:,}")

            # Show layer shapes
            print("\nLayer shapes:")
            for name, param in policy_state.items():
                print(f"  {name:30s}: {str(tuple(param.shape)):20s} ({param.numel():,} params)")

        print()

        # Optimizer information
        if 'optimizer_state_dict' in checkpoint:
            print("Optimizer State:")
            print("-" * 60)
            optimizer_state = checkpoint['optimizer_state_dict']

            if 'param_groups' in optimizer_state:
                for i, pg in enumerate(optimizer_state['param_groups']):
                    print(f"Parameter group {i}:")
                    print(f"  Learning rate: {pg.get('lr', 'Unknown')}")
                    print(f"  Betas: {pg.get('betas', 'Unknown')}")
                    print(f"  Eps: {pg.get('eps', 'Unknown')}")
                    print(f"  Weight decay: {pg.get('weight_decay', 'Unknown')}")

        print()

        # Training recommendations
        print("Recommendations:")
        print("-" * 60)

        if isinstance(epsilon, (int, float)) and isinstance(steps_done, int):
            if epsilon > 0.5:
                print("⚠️  Epsilon is still high (>{:.2f})".format(epsilon))
                print("   The agent is still mostly random during training.")
                print("   Recommendation: Continue training for much longer.")
                estimated_steps_needed = int((1.0 - 0.1) / (1.0 - epsilon) * 1_000_000)
                print(f"   Estimated steps to reach ε=0.1: {estimated_steps_needed:,}")

            elif epsilon > 0.1:
                print("ℹ️  Epsilon is moderate ({:.2f})".format(epsilon))
                print("   The agent is still exploring significantly.")
                print("   Recommendation: Continue training for better performance.")
                estimated_steps_needed = int((epsilon - 0.01) / (1.0 - 0.01) * 1_000_000)
                print(f"   Estimated steps to reach ε=0.01: {estimated_steps_needed:,}")

            elif epsilon > 0.01:
                print("✓ Epsilon is low ({:.2f})".format(epsilon))
                print("  The agent is mostly exploiting learned policy.")
                print("  Recommendation: Agent may be ready for evaluation.")
                print("  If performance is poor, consider:")
                print("    - Resuming with --reset-epsilon for more exploration")
                print("    - Training from scratch with different hyperparameters")

            else:
                print("✓ Epsilon has reached minimum ({:.2f})".format(epsilon))
                print("  The agent is using learned policy with minimal exploration.")
                print("  Recommendation:")
                print("    - Evaluate performance with watch_agent.py")
                print("    - If performance is good: training complete!")
                print("    - If performance is poor: agent may be stuck in local optimum")
                print("      → Consider starting fresh or resuming with --reset-epsilon")

            # Steps recommendation
            print()
            if steps_done < 100_000:
                print(f"⚠️  Only {steps_done:,} steps trained - very early in training")
                print("   Recommendation: Train for at least 500k-1M steps")
            elif steps_done < 500_000:
                print(f"ℹ️  {steps_done:,} steps trained - early training phase")
                print("   Recommendation: Train for at least 1M steps for good performance")
            elif steps_done < 1_000_000:
                print(f"✓ {steps_done:,} steps trained - substantial training")
                print("  Agent should show some learned behavior")
            else:
                print(f"✓ {steps_done:,} steps trained - extensive training")
                print("  Agent should have strong performance if hyperparameters are good")

        print()

        # Usage examples
        print("Usage Examples:")
        print("-" * 60)
        print("# Watch agent play:")
        print(f"  python watch_agent.py --checkpoint {checkpoint_path}")
        print()
        print("# Resume training:")
        print(f"  python train.py --resume {checkpoint_path} --episodes 1000")
        print()
        print("# Resume with reset epsilon (more exploration):")
        print(f"  python train.py --resume {checkpoint_path} --reset-epsilon --episodes 1000")

        print()
        print("=" * 60)

    except FileNotFoundError:
        print(f"❌ Error: Checkpoint file not found: {checkpoint_path}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description='Inspect DDQN checkpoint file')
    parser.add_argument('checkpoint', type=str, help='Path to checkpoint file')
    args = parser.parse_args()

    inspect_checkpoint(args.checkpoint)


if __name__ == "__main__":
    main()
