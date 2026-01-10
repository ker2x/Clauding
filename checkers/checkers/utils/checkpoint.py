"""
Checkpoint management for saving and loading models.
"""

import os
import torch
from typing import Optional, Dict, Any
from pathlib import Path


class CheckpointManager:
    """
    Manages saving and loading of model checkpoints.
    """

    def __init__(self, checkpoint_dir: str = "checkpoints"):
        """
        Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory to save checkpoints
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.best_checkpoint_path = self.checkpoint_dir / "best_model.pt"
        self.latest_checkpoint_path = self.checkpoint_dir / "latest_model.pt"

    def save_checkpoint(
        self,
        network: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        iteration: int,
        metrics: Dict[str, float],
        is_best: bool = False
    ) -> str:
        """
        Save a checkpoint.

        Args:
            network: Neural network
            optimizer: Optimizer
            iteration: Current training iteration
            metrics: Training metrics to save
            is_best: Whether this is the best model so far

        Returns:
            Path to saved checkpoint
        """
        checkpoint = {
            "iteration": iteration,
            "network_state_dict": network.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
        }

        # Save iteration checkpoint
        iter_path = self.checkpoint_dir / f"checkpoint_iter_{iteration}.pt"
        torch.save(checkpoint, iter_path)

        # Save latest checkpoint
        torch.save(checkpoint, self.latest_checkpoint_path)

        # Save best checkpoint if this is the best model
        if is_best:
            torch.save(checkpoint, self.best_checkpoint_path)
            print(f"  💾 Saved best model at iteration {iteration}")

        return str(iter_path)

    def load_checkpoint(
        self,
        network: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        checkpoint_path: Optional[str] = None,
        load_best: bool = False
    ) -> Dict[str, Any]:
        """
        Load a checkpoint.

        Args:
            network: Neural network to load weights into
            optimizer: Optimizer to load state into (optional)
            checkpoint_path: Specific checkpoint to load (optional)
            load_best: Load best model instead of latest

        Returns:
            Checkpoint dictionary with iteration and metrics
        """
        # Determine which checkpoint to load
        if checkpoint_path:
            path = Path(checkpoint_path)
        elif load_best:
            path = self.best_checkpoint_path
        else:
            path = self.latest_checkpoint_path

        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        # Load checkpoint
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)

        # Load network weights
        network.load_state_dict(checkpoint["network_state_dict"])

        # Load optimizer state if provided
        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        print(f"  📂 Loaded checkpoint from iteration {checkpoint['iteration']}")

        return checkpoint

    def get_latest_iteration(self) -> int:
        """
        Get the iteration number of the latest checkpoint.

        Returns:
            Iteration number, or 0 if no checkpoint exists
        """
        if not self.latest_checkpoint_path.exists():
            return 0

        checkpoint = torch.load(self.latest_checkpoint_path, map_location="cpu", weights_only=False)
        return checkpoint.get("iteration", 0)

    def list_checkpoints(self) -> list[str]:
        """
        List all checkpoint files in the directory.

        Returns:
            List of checkpoint file paths
        """
        checkpoints = sorted(self.checkpoint_dir.glob("checkpoint_iter_*.pt"))
        return [str(cp) for cp in checkpoints]

    def cleanup_old_checkpoints(self, keep_last_n: int = 10):
        """
        Remove old checkpoints, keeping only the most recent N.

        Args:
            keep_last_n: Number of recent checkpoints to keep
        """
        checkpoints = sorted(
            self.checkpoint_dir.glob("checkpoint_iter_*.pt"),
            key=lambda p: p.stat().st_mtime
        )

        # Remove old checkpoints (keep best and latest separate)
        if len(checkpoints) > keep_last_n:
            for checkpoint in checkpoints[:-keep_last_n]:
                checkpoint.unlink()
                print(f"  🗑️  Removed old checkpoint: {checkpoint.name}")


def save_model_for_inference(
    network: torch.nn.Module,
    save_path: str,
    metadata: Optional[Dict[str, Any]] = None
):
    """
    Save model for inference (no optimizer state).

    Args:
        network: Neural network
        save_path: Path to save model
        metadata: Optional metadata to save with model
    """
    save_dict = {
        "network_state_dict": network.state_dict(),
        "metadata": metadata or {},
    }

    torch.save(save_dict, save_path)
    print(f"  💾 Saved inference model to {save_path}")


def load_model_for_inference(
    network: torch.nn.Module,
    load_path: str
) -> Dict[str, Any]:
    """
    Load model for inference.

    Args:
        network: Neural network to load weights into
        load_path: Path to load model from

    Returns:
        Metadata dictionary
    """
    checkpoint = torch.load(load_path, map_location="cpu", weights_only=False)
    network.load_state_dict(checkpoint["network_state_dict"])

    print(f"  📂 Loaded inference model from {load_path}")

    return checkpoint.get("metadata", {})


# Testing
if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

    from checkers.network.resnet import CheckersNetwork
    from config import Config

    print("Testing checkpoint manager...")

    # Create temporary checkpoint directory
    test_dir = "test_checkpoints"
    manager = CheckpointManager(test_dir)

    # Create network and optimizer
    network = CheckersNetwork(
        num_filters=Config.NUM_FILTERS,
        num_res_blocks=Config.NUM_RES_BLOCKS,
        policy_size=Config.POLICY_SIZE
    )
    optimizer = torch.optim.Adam(network.parameters(), lr=Config.LEARNING_RATE)

    # Save checkpoint
    print("\nSaving checkpoint...")
    metrics = {"loss": 2.5, "policy_loss": 1.5, "value_loss": 1.0}
    path = manager.save_checkpoint(network, optimizer, iteration=10, metrics=metrics)
    print(f"Saved to: {path}")

    # Save best checkpoint
    print("\nSaving best checkpoint...")
    metrics_best = {"loss": 2.0, "policy_loss": 1.2, "value_loss": 0.8}
    manager.save_checkpoint(network, optimizer, iteration=20, metrics=metrics_best, is_best=True)

    # List checkpoints
    print("\nListing checkpoints...")
    checkpoints = manager.list_checkpoints()
    for cp in checkpoints:
        print(f"  {cp}")

    # Load checkpoint
    print("\nLoading latest checkpoint...")
    new_network = CheckersNetwork(
        num_filters=Config.NUM_FILTERS,
        num_res_blocks=Config.NUM_RES_BLOCKS,
        policy_size=Config.POLICY_SIZE
    )
    new_optimizer = torch.optim.Adam(new_network.parameters(), lr=Config.LEARNING_RATE)

    checkpoint = manager.load_checkpoint(new_network, new_optimizer)
    print(f"Loaded iteration: {checkpoint['iteration']}")
    print(f"Loaded metrics: {checkpoint['metrics']}")

    # Load best checkpoint
    print("\nLoading best checkpoint...")
    best_checkpoint = manager.load_checkpoint(new_network, load_best=True)
    print(f"Best iteration: {best_checkpoint['iteration']}")
    print(f"Best metrics: {best_checkpoint['metrics']}")

    # Test inference save/load
    print("\nTesting inference model save/load...")
    inference_path = f"{test_dir}/inference_model.pt"
    save_model_for_inference(network, inference_path, metadata={"version": "1.0"})

    inference_network = CheckersNetwork(
        num_filters=Config.NUM_FILTERS,
        num_res_blocks=Config.NUM_RES_BLOCKS,
        policy_size=Config.POLICY_SIZE
    )
    metadata = load_model_for_inference(inference_network, inference_path)
    print(f"Metadata: {metadata}")

    # Cleanup
    print("\nCleaning up test checkpoints...")
    import shutil
    shutil.rmtree(test_dir)

    print("\n✓ Checkpoint tests passed!")
