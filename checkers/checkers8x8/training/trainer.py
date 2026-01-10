"""
Main training loop for 8x8 Checkers.
"""

import torch
import torch.nn.functional as F
import numpy as np
import time
import csv
from pathlib import Path
from typing import Dict

try:
    from ..network.resnet import CheckersNetwork
    from .replay_buffer import ReplayBuffer
    from .self_play import play_games_sequential
    from .evaluation import evaluate_models
except ImportError:
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
    from checkers8x8.network.resnet import CheckersNetwork
    from checkers8x8.training.replay_buffer import ReplayBuffer
    from checkers8x8.training.self_play import play_games_sequential
    from checkers8x8.training.evaluation import evaluate_models


class Trainer:
    """
    Main training loop for 8x8 checkers.

    Orchestrates:
    - Self-play game generation
    - Replay buffer management
    - Network training
    - Checkpointing and logging
    """

    def __init__(
        self,
        network: CheckersNetwork,
        config,
        device: torch.device,
        selfplay_device: torch.device,
        game_visualizer=None
    ):
        """
        Initialize trainer.

        Args:
            network: Neural network to train
            config: Configuration object
            device: PyTorch device for training
            selfplay_device: PyTorch device for self-play
            game_visualizer: Optional visualizer for game rendering and metrics
        """
        self.network = network
        self.config = config
        self.device = device
        self.selfplay_device = selfplay_device
        self.game_visualizer = game_visualizer

        self.network.to(self.device)

        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.config.LEARNING_RATE,
            weight_decay=self.config.WEIGHT_DECAY
        )

        # Replay buffer
        self.replay_buffer = ReplayBuffer(
            capacity=self.config.BUFFER_SIZE,
            recency_tau=self.config.RECENCY_TAU
        )

        # Training state
        self.start_iteration = 0
        self.best_model_path = Path(self.config.CHECKPOINT_DIR) / "best_model.pt"

        # Logging
        self.setup_logging()

    def setup_logging(self):
        """Setup CSV logging for metrics."""
        log_dir = Path(self.config.LOG_DIR)
        log_dir.mkdir(parents=True, exist_ok=True)

        self.log_file = log_dir / "training_log.csv"

        # Create CSV file with headers
        if not self.log_file.exists():
            with open(self.log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'iteration', 'total_loss', 'policy_loss', 'value_loss',
                    'buffer_size', 'games_played',
                    'time_selfplay', 'time_training'
                ])

    def log_metrics(self, metrics: Dict):
        """Log metrics to CSV file."""
        with open(self.log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                metrics.get('iteration', 0),
                metrics.get('total_loss', 0),
                metrics.get('policy_loss', 0),
                metrics.get('value_loss', 0),
                metrics.get('buffer_size', 0),
                metrics.get('games_played', 0),
                metrics.get('time_selfplay', 0),
                metrics.get('time_training', 0),
            ])

    def train(self, num_iterations: int):
        """
        Main training loop.

        Args:
            num_iterations: Number of training iterations to run
        """
        print("=" * 70)
        print("🚀 Starting 8x8 Checkers Training (Fixed Action Space)")
        print("=" * 70)
        print(f"Target: {num_iterations} iterations")
        if self.start_iteration > 0:
            print(f"Resuming from iteration {self.start_iteration}")
            if num_iterations <= self.start_iteration:
                print("\n" + "=" * 70)
                print(f"⚠️  ERROR: Cannot resume!")
                print(f"Current iteration: {self.start_iteration}")
                print(f"Target iterations: {num_iterations}")
                print(f"\nYou need to set --iterations to a value GREATER than {self.start_iteration}")
                print(f"Example: --iterations {self.start_iteration + 50}")
                print("=" * 70)
                return
        print(f"Training device: {self.device}")
        print(f"Self-play device: {self.selfplay_device}")
        print(f"MCTS simulations: {self.config.MCTS_SIMS_SELFPLAY}")
        print("=" * 70)

        for iteration in range(self.start_iteration, num_iterations):
            metrics = {'iteration': iteration + 1}

            # 1. Self-play
            print(f"\n[Iteration {iteration + 1}/{num_iterations}]")
            print(f"[1/2] Self-play ({self.config.GAMES_PER_ITERATION} games)...")
            t_start = time.time()

            # Move network to self-play device
            self.network.to(self.selfplay_device)
            self.network.eval()

            # Create visualization callback if game visualizer exists
            on_move_callback = None
            if self.game_visualizer:
                def on_move(game, policy, move_count, current_player):
                    """Convert game state to visualizer format and render."""
                    # Get absolute board array (no perspective switching)
                    board = game.to_absolute_board_array()
                    
                    # Render
                    self.game_visualizer.render(board, policy, move_count, current_player)


                on_move_callback = on_move

            states, policies, values = play_games_sequential(
                self.network,
                self.config,
                self.selfplay_device,
                num_games=self.config.GAMES_PER_ITERATION,
                on_move=on_move_callback
            )

            # Move network back to training device
            self.network.to(self.device)

            metrics['time_selfplay'] = time.time() - t_start
            metrics['games_played'] = self.config.GAMES_PER_ITERATION

            # Add to replay buffer
            self.replay_buffer.add_batch(states, policies, values)

            print(f"  Generated {len(states)} training examples")
            print(f"  Time: {metrics['time_selfplay']:.1f}s")

            # 2. Training
            print(f"\n[2/2] Training ({self.config.TRAINING_STEPS_PER_ITERATION} steps)...")
            t_start = time.time()

            if len(self.replay_buffer) >= self.config.BATCH_SIZE:
                train_metrics = self.train_network(self.config.TRAINING_STEPS_PER_ITERATION)
                metrics.update(train_metrics)
            else:
                print(f"  Skipping (buffer size {len(self.replay_buffer)} < batch size {self.config.BATCH_SIZE})")
                metrics['total_loss'] = 0.0
                metrics['policy_loss'] = 0.0
                metrics['value_loss'] = 0.0

            metrics['time_training'] = time.time() - t_start
            metrics['buffer_size'] = len(self.replay_buffer)

            print(f"  Loss: {metrics.get('total_loss', 0):.4f} "
                  f"(policy: {metrics.get('policy_loss', 0):.4f}, "
                  f"value: {metrics.get('value_loss', 0):.4f})")
            print(f"  Time: {metrics['time_training']:.1f}s")
            
            # Buffer capacity visualization
            buffer_size = len(self.replay_buffer)
            buffer_capacity = self.config.BUFFER_SIZE
            buffer_pct = buffer_size / buffer_capacity
            bar_width = 30
            filled = int(bar_width * buffer_pct)
            bar = '█' * filled + '░' * (bar_width - filled)
            print(f"  Buffer: [{bar}] {buffer_pct*100:.1f}% ({buffer_size:,}/{buffer_capacity:,})")

            # Log metrics
            self.log_metrics(metrics)

            # Update visualizer
            if self.game_visualizer:
                self.game_visualizer.update_metrics(metrics)

            # Increment buffer generation
            self.replay_buffer.increment_generation()

            # Save checkpoint
            if (iteration + 1) % self.config.SAVE_FREQUENCY == 0:
                self.save_checkpoint(iteration + 1)

            # Evaluate and update best model
            if (iteration + 1) % self.config.EVAL_FREQUENCY == 0:
                self.evaluate_and_update_best(iteration + 1)

        # Final save and evaluation regardless of frequency
        self.save_checkpoint(num_iterations)
        if num_iterations > self.start_iteration and num_iterations % self.config.EVAL_FREQUENCY != 0:
            self.evaluate_and_update_best(num_iterations)

        print("\n" + "=" * 70)
        print("🎉 TRAINING COMPLETE!")
        print("=" * 70)

    def train_network(self, num_steps: int) -> Dict[str, float]:
        """
        Train network for specified number of gradient steps.

        Args:
            num_steps: Number of training steps

        Returns:
            Dictionary of training metrics
        """
        self.network.train()

        total_losses = []
        policy_losses = []
        value_losses = []

        # Print progress bar
        print("  ", end="", flush=True)
        bar_width = 40

        for step in range(num_steps):
            # Sample batch from replay buffer
            states, policies, values = self.replay_buffer.sample(self.config.BATCH_SIZE)

            # Convert to tensors
            states_t = torch.from_numpy(states).to(self.device)
            policies_t = torch.from_numpy(policies).to(self.device)
            values_t = torch.from_numpy(values).unsqueeze(1).to(self.device)

            # Forward pass
            pred_policies, pred_values = self.network(states_t)

            # Compute losses
            policy_loss = -torch.sum(policies_t * F.log_softmax(pred_policies, dim=1)) / len(states_t)
            value_loss = F.mse_loss(pred_values, values_t)
            total_loss = policy_loss + value_loss

            # Backward pass
            self.optimizer.zero_grad()
            total_loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.network.parameters(),
                self.config.GRAD_CLIP
            )

            self.optimizer.step()

            # Record losses
            total_losses.append(total_loss.item())
            policy_losses.append(policy_loss.item())
            value_losses.append(value_loss.item())

            # Update progress bar every 10 steps or at end
            if (step + 1) % 10 == 0 or (step + 1) == num_steps:
                progress = (step + 1) / num_steps
                filled = int(bar_width * progress)
                bar = '█' * filled + '░' * (bar_width - filled)
                avg_loss = np.mean(total_losses[-10:]) if total_losses else 0
                print(f"\r  [{bar}] {step + 1}/{num_steps} steps | "
                      f"loss: {avg_loss:.4f}", end="", flush=True)

        print()  # New line after completion
        self.network.eval()

        return {
            'total_loss': np.mean(total_losses),
            'policy_loss': np.mean(policy_losses),
            'value_loss': np.mean(value_losses),
        }

    def save_checkpoint(self, iteration: int):
        """Save training checkpoint."""
        checkpoint_dir = Path(self.config.CHECKPOINT_DIR)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            'iteration': iteration,
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'replay_buffer_state_dict': self.replay_buffer.state_dict(),
        }

        # Save numbered checkpoint
        checkpoint_path = checkpoint_dir / f"checkpoint_iter_{iteration}.pt"
        torch.save(checkpoint, checkpoint_path)
        print(f"  💾 Saved checkpoint: {checkpoint_path}")

        # Save latest checkpoint (always overwrite)
        latest_path = checkpoint_dir / "latest.pt"
        torch.save(checkpoint, latest_path)
        print(f"  💾 Updated latest checkpoint: {latest_path}")

    def evaluate_and_update_best(self, iteration: int):
        """
        Evaluate current model against best and update if better.

        Args:
            iteration: Current training iteration
        """
        print(f"\n[Evaluation at iteration {iteration}]")
        print("-" * 70)

        checkpoint_dir = Path(self.config.CHECKPOINT_DIR)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Load best model (or save current as best if none exists)
        if self.best_model_path.exists():
            print(f"  Loading best model from {self.best_model_path}...")
            best_model = CheckersNetwork(
                num_filters=self.config.NUM_FILTERS,
                num_res_blocks=self.config.NUM_RES_BLOCKS,
                policy_size=self.config.POLICY_SIZE
            )
            checkpoint = torch.load(self.best_model_path, map_location=self.device, weights_only=False)
            best_model.load_state_dict(checkpoint['network_state_dict'])
            best_model.to(self.selfplay_device)
            best_model.eval()
        else:
            print("  No best model found, creating INITIAL baseline from current model")
            torch.save({
                'network_state_dict': self.network.state_dict(),
                'iteration': iteration,
                'win_rate': 0.5,
            }, self.best_model_path)
            print(f"  💾 Saved initial baseline to {self.best_model_path}")
            
            # In a short run, there is no point evaluating against exactly ourselves
            print("  Skipping evaluation for this step as we just established the baseline.")
            print("-" * 70)
            return

        # Move current model to eval device
        self.network.to(self.selfplay_device)
        self.network.eval()

        # Create visualization callback if game visualizer exists
        on_move_callback = None
        if self.game_visualizer:
            def on_move(game, policy, move_count, current_player):
                """Convert game state to visualizer format and render."""
                board = game.to_absolute_board_array()
                self.game_visualizer.render(board, policy, move_count, current_player)
            on_move_callback = on_move

        # Evaluate
        results = evaluate_models(
            current_model=self.network,
            best_model=best_model,
            config=self.config,
            device=self.selfplay_device,
            on_move=on_move_callback
        )

        # Move network back to training device
        self.network.to(self.device)

        win_rate = results['win_rate']
        print(f"\n  Results: W{results['wins']} L{results['losses']} D{results['draws']}")
        print(f"  Win rate: {win_rate:.1%}")

        # Promote if win rate exceeds threshold
        if win_rate >= self.config.PROMOTION_THRESHOLD:
            print(f"  🏆 NEW BEST MODEL! (Win rate: {win_rate:.1%} >= {self.config.PROMOTION_THRESHOLD:.1%})")
            torch.save({
                'network_state_dict': self.network.state_dict(),
                'iteration': iteration,
                'win_rate': win_rate,
            }, self.best_model_path)
            print(f"  💾 Saved to {self.best_model_path}")
        else:
            print(f"  ⚠️  Model not promoted (Win rate: {win_rate:.1%} < {self.config.PROMOTION_THRESHOLD:.1%})")

        print("-" * 70)

    def load_checkpoint(self, checkpoint_path: str):
        """
        Load training checkpoint to resume training.

        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        print(f"📂 Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        # Restore network
        self.network.load_state_dict(checkpoint['network_state_dict'])
        print("  ✓ Network state restored")

        # Restore optimizer
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("  ✓ Optimizer state restored")

        # Restore replay buffer
        if 'replay_buffer_state_dict' in checkpoint:
            self.replay_buffer.load_state_dict(checkpoint['replay_buffer_state_dict'])
            print(f"  ✓ Replay buffer restored (size: {len(self.replay_buffer)})")
        else:
            print("  ⚠ No replay buffer state in checkpoint")

        # Restore training state
        self.start_iteration = checkpoint['iteration']

        print(f"  ✓ Resuming from iteration {self.start_iteration}")

        # Restore visualizer history if available
        if self.game_visualizer:
            print("  📊 Restoring metrics history for visualization...")
            self.game_visualizer.load_history(str(self.log_file), max_iteration=self.start_iteration)


# Testing
if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
    from config8x8 import Config

    print("Testing trainer...")

    # Setup
    device = Config.get_device()
    selfplay_device = Config.get_selfplay_device()

    network = CheckersNetwork(
        num_filters=64,  # Smaller for testing
        num_res_blocks=2,
        policy_size=Config.POLICY_SIZE
    )

    trainer = Trainer(network, Config, device, selfplay_device)

    print(f"\nTrainer created")
    print(f"Training device: {device}")
    print(f"Self-play device: {selfplay_device}")

    print("\n✓ Trainer tests passed!")
