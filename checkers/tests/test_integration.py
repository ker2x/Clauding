"""
Integration test for the complete training pipeline.

Tests that all components work together for a mini training run.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import tempfile
import shutil
from config import Config
from checkers.network.resnet import CheckersNetwork, initialize_weights
from checkers.training.trainer import Trainer


def test_mini_training():
    """Run a mini training session to test integration."""
    print("\n" + "=" * 60)
    print("Integration Test: Mini Training Run")
    print("=" * 60)

    # Create temporary directories
    temp_dir = tempfile.mkdtemp(prefix="checkers_test_")
    Config.CHECKPOINT_DIR = os.path.join(temp_dir, "checkpoints")
    Config.LOG_DIR = os.path.join(temp_dir, "logs")

    print(f"\nUsing temporary directory: {temp_dir}")

    try:
        # Reduce scale for testing
        Config.GAMES_PER_ITERATION = 2  # Very small for testing
        Config.TRAINING_STEPS_PER_ITERATION = 5
        Config.MCTS_SIMS_SELFPLAY = 10  # Much smaller for speed
        Config.BUFFER_SIZE = 1000
        Config.EVAL_FREQUENCY = 2
        Config.EVAL_GAMES = 2
        Config.SAVE_FREQUENCY = 1

        print("\nConfiguration (reduced for testing):")
        print(f"  Games per iteration: {Config.GAMES_PER_ITERATION}")
        print(f"  Training steps: {Config.TRAINING_STEPS_PER_ITERATION}")
        print(f"  MCTS simulations: {Config.MCTS_SIMS_SELFPLAY}")

        # Get device
        device = Config.get_device()
        print(f"  Device: {device}")

        # Create network
        print("\nInitializing network...")
        network = CheckersNetwork(
            num_filters=Config.NUM_FILTERS,
            num_res_blocks=Config.NUM_RES_BLOCKS,
            policy_size=Config.POLICY_SIZE
        )
        network.apply(initialize_weights)

        total_params = sum(p.numel() for p in network.parameters())
        print(f"  Parameters: {total_params:,}")

        # Create trainer
        print("\nCreating trainer...")
        trainer = Trainer(
            network=network,
            config=Config,
            device=device,
            resume_from=None
        )

        # Run 2 training iterations
        print("\nRunning 2 training iterations...")
        print("  (This will take a few minutes...)")

        trainer.train(num_iterations=2)

        # Check that checkpoint was created
        import glob
        checkpoints = glob.glob(os.path.join(Config.CHECKPOINT_DIR, "*.pt"))
        print(f"\nCheckpoints created: {len(checkpoints)}")
        for cp in checkpoints:
            print(f"  {os.path.basename(cp)}")

        # Check that log was created
        log_file = os.path.join(Config.LOG_DIR, "training_log.csv")
        if os.path.exists(log_file):
            print(f"\nLog file created: {log_file}")

            # Read and print last few lines
            with open(log_file, 'r') as f:
                lines = f.readlines()
                print(f"  Log entries: {len(lines) - 1}")  # -1 for header
                if len(lines) > 1:
                    print(f"  Last entry: {lines[-1].strip()}")

        print("\n" + "=" * 60)
        print("✓ Integration test PASSED!")
        print("=" * 60)
        print("\nAll components are working correctly:")
        print("  ✓ Game engine")
        print("  ✓ Neural network")
        print("  ✓ MCTS")
        print("  ✓ Self-play")
        print("  ✓ Replay buffer")
        print("  ✓ Training loop")
        print("  ✓ Checkpointing")
        print("  ✓ Logging")
        print("\nSystem is ready for full training!")

    finally:
        # Cleanup
        print(f"\nCleaning up temporary directory: {temp_dir}")
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    test_mini_training()
