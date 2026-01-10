"""
Main training loop for AlphaZero-style learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import csv
from pathlib import Path
from typing import Dict, Optional

from ..network.resnet import CheckersNetwork
from ..network.utils import augment_batch
from .replay_buffer import ReplayBuffer
from .self_play import play_games_parallel
from .evaluator import Evaluator
from ..engine.bitboard import square_to_row_col
from ..utils.checkpoint import CheckpointManager
from ..utils.progress import ProgressMonitor
from ..utils.visualizer import RealTimeVisualizer
from ..utils.visualizer_types import MetricsUpdate, GameStateUpdate, EvaluationUpdate, StatusUpdate


class Trainer:
    """
    Main training loop for checkers.

    Orchestrates:
    - Self-play game generation
    - Replay buffer management
    - Network training
    - Evaluation and model promotion
    - Checkpointing and logging
    """

    def __init__(
        self,
        network: CheckersNetwork,
        config,
        device: torch.device,
        resume_from: Optional[str] = None
    ):
        """
        Initialize trainer.

        Args:
            network: Neural network to train
            config: Configuration object
            device: PyTorch device
            resume_from: Optional checkpoint path to resume from
        """
        self.network = network
        self.config = config
        self.device = device  # Training device (can be GPU)
        self.selfplay_device = self.config.get_selfplay_device()  # Self-play device (usually CPU)

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

        # Evaluator
        self.evaluator = Evaluator(self.config, self.device)

        # Checkpoint manager
        self.checkpoint_manager = CheckpointManager(self.config.CHECKPOINT_DIR)

        # Best network for evaluation
        self.best_network = CheckersNetwork(
            num_filters=self.config.NUM_FILTERS,
            num_res_blocks=self.config.NUM_RES_BLOCKS,
            policy_size=self.config.POLICY_SIZE
        )
        self.best_network.to(self.device)
        self.best_network.load_state_dict(self.network.state_dict())

        # Training state
        self.start_iteration = 0
        self.best_win_rate = 0.0

        # Logging
        self.setup_logging(resume_from)

        # Visualizer (optional)
        self.visualizer: Optional[RealTimeVisualizer] = None
        if self.config.VISUALIZE_TRAINING:
            self.visualizer = RealTimeVisualizer(
                window_width=self.config.VISUALIZE_WINDOW_WIDTH,
                window_height=self.config.VISUALIZE_WINDOW_HEIGHT,
                log_file=str(self.log_file),
                max_history=self.config.VISUALIZE_HISTORY_LENGTH
            )

        # Resume from checkpoint if specified
        if resume_from:
            self.resume_training(resume_from)

    def setup_logging(self, resume_from: Optional[str]):
        """
        Setup CSV logging for metrics.
        
        Args:
            resume_from: Checkpoint path if resuming, None if fresh start
        """
        log_dir = Path(self.config.LOG_DIR)
        log_dir.mkdir(parents=True, exist_ok=True)

        self.log_file = log_dir / "training_log.csv"

        # If starting fresh and log file exists, backup old log and start fresh
        if resume_from is None and self.log_file.exists():
            timestamp = int(time.time())
            backup_file = log_dir / f"training_log_backup_{timestamp}.csv"
            self.log_file.rename(backup_file)
            print(f"  📝 Fresh start: Backed up old log to {backup_file.name}")

        # Create CSV file with headers
        if not self.log_file.exists():
            with open(self.log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'iteration', 'total_loss', 'policy_loss', 'value_loss',
                    'buffer_size', 'games_played', 'eval_win_rate',
                    'time_selfplay', 'time_training', 'time_eval'
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
                metrics.get('eval_win_rate', -1),
                metrics.get('time_selfplay', 0),
                metrics.get('time_training', 0),
                metrics.get('time_eval', 0),
            ])

    def train(self, num_iterations: int):
        """
        Main training loop.

        Args:
            num_iterations: Number of training iterations to run
        """
        print("=" * 70)
        print("🚀 Starting AlphaZero Training")
        print("=" * 70)
        print(f"Target: {num_iterations} iterations")
        print(f"Training device: {self.device}")
        print(f"Self-play device: {self.selfplay_device}")
        print(f"Workers: {self.config.NUM_WORKERS}")
        print(f"MCTS simulations: {self.config.MCTS_SIMS_SELFPLAY}")
        if self.visualizer:
            print(f"Visualization: ENABLED")
        print("=" * 70)

        # Initialize progress monitor
        progress_monitor = ProgressMonitor(total_iterations=num_iterations)

        # Start visualizer if enabled
        if self.visualizer:
            self.visualizer.start()
            print("\n✓ Visualizer started! Check the pygame window for real-time updates.\n")

        try:
            for iteration in range(self.start_iteration, self.start_iteration + num_iterations):
                metrics = {'iteration': iteration + 1}

                # Update visualizer status
                if self.visualizer:
                    self.visualizer.update(StatusUpdate(
                        message=f"Self-play iteration {iteration + 1}/{self.start_iteration + num_iterations}",
                        iteration=iteration + 1,
                        phase="selfplay"
                    ))

                # Define callback for real-time visualization
                on_step_callback = None
                if self.visualizer:
                    def on_step(state, policy, move_count, current_player=1, legal_moves=None):
                        # Convert state to board array
                        board = np.zeros((10, 10), dtype=np.int8)
                        board += state[0].astype(np.int8)       # Player men
                        board += state[1].astype(np.int8) * 2   # Player kings
                        board -= state[2].astype(np.int8)       # Opponent men
                        board -= state[3].astype(np.int8) * 2   # Opponent kings
                        
                        # Map policy (move indices) to board squares (for visualization)
                        visual_policy = np.zeros(100, dtype=np.float32)
                        
                        if policy is not None and legal_moves:
                            # Policy is a distribution over legal moves
                            # We map each probability to the destination square of the move
                            # (Simplified visualization: showing "where the AI wants to go")
                            norm_policy = policy[:len(legal_moves)]
                            if norm_policy.sum() > 0:
                                norm_policy = norm_policy / norm_policy.sum()
                                
                            for i, prob in enumerate(norm_policy):
                                if prob > 0.01:  # Filter noise
                                    move = legal_moves[i]
                                    to_sq_idx = move.to_square  # 0-49 index
                                    
                                    # Convert 0-49 index to 0-99 board index
                                    row, col = square_to_row_col(to_sq_idx)
                                    tod = row * 10 + col
                                    
                                    if 0 <= tod < 100:
                                        # Accumulate probability (multiple moves might land on same square)
                                        visual_policy[tod] += prob

                        # Unflip board if it's player 2's turn
                        if current_player == 2:
                            # 1. Vertical flip (restores P2 to top)
                            board = np.flipud(board)
                            
                            # 2. Fix parity column shifts (Checkers board specific)
                            board[::2] = np.roll(board[::2], 1, axis=1)
                            board[1::2] = np.roll(board[1::2], -1, axis=1)
                            
                            board = -board  # Swap P1/P2 identity so colors stay consistent

                            # Flip policy for visualization to match board
                            # Canonical view (P2 playing up) -> Global view (P2 playing down)
                            # We must apply the EXACT same transformation as the board (flipud + per-row shifts)
                            if visual_policy is not None:
                                # Reshape to 10x10 to apply board transforms
                                p_grid = visual_policy[:100].reshape(10, 10).copy()
                                
                                # 1. Vertical flip
                                p_grid = np.flipud(p_grid)
                                
                                # 2. Fix parity column shifts (same as board)
                                p_grid[::2] = np.roll(p_grid[::2], 1, axis=1)
                                p_grid[1::2] = np.roll(p_grid[1::2], -1, axis=1)
                                
                                # Flatten back to 1D
                                visual_policy[:100] = p_grid.flatten()
                        
                        self.visualizer.update(GameStateUpdate(
                            game_array=board,
                            policy=visual_policy,
                            move_count=move_count
                        ))
                        # Poll events to keep window responsive
                        self.visualizer.poll_events()
                    
                    on_step_callback = on_step

                # 1. Self-play
                print(f"\n[1/4] Self-play ({self.config.GAMES_PER_ITERATION} games)...")
                t_start = time.time()

                # Move network to self-play device (usually CPU for better MCTS performance)
                self.network.to(self.selfplay_device)
                self.network.eval()

                states, policies, values = play_games_parallel(
                    self.network,
                    self.config,
                    self.selfplay_device,
                    num_games=self.config.GAMES_PER_ITERATION,
                    on_step=on_step_callback
                )

                # Move network back to training device
                self.network.to(self.device)

                metrics['time_selfplay'] = time.time() - t_start
                metrics['games_played'] = self.config.GAMES_PER_ITERATION

                # Add to replay buffer
                self.replay_buffer.add_batch(states, policies, values)

                print(f"  Generated {len(states)} training examples")
                print(f"  Time: {metrics['time_selfplay']:.1f}s")

                # Update visualizer status
                if self.visualizer:
                    self.visualizer.update(StatusUpdate(
                        message=f"Training iteration {iteration + 1}",
                        iteration=iteration + 1,
                        phase="training"
                    ))

                # 2. Training
                print(f"\n[2/4] Training ({self.config.TRAINING_STEPS_PER_ITERATION} steps)...")
                t_start = time.time()

                train_metrics = self.train_network(self.config.TRAINING_STEPS_PER_ITERATION)

                metrics['time_training'] = time.time() - t_start
                metrics.update(train_metrics)
                metrics['buffer_size'] = len(self.replay_buffer)

                print(f"  Loss: {metrics['total_loss']:.4f} "
                      f"(policy: {metrics['policy_loss']:.4f}, "
                      f"value: {metrics['value_loss']:.4f})")
                print(f"  Time: {metrics['time_training']:.1f}s")

                # Update visualizer with metrics
                if self.visualizer:
                    self.visualizer.update(MetricsUpdate(
                        iteration=iteration + 1,
                        total_loss=metrics['total_loss'],
                        policy_loss=metrics['policy_loss'],
                        value_loss=metrics['value_loss'],
                        buffer_size=metrics['buffer_size'],
                        time_selfplay=metrics['time_selfplay'],
                        time_training=metrics['time_training']
                    ))

                # Update visualizer status
                if self.visualizer and (iteration + 1) % self.config.EVAL_FREQUENCY == 0:
                    self.visualizer.update(StatusUpdate(
                        message=f"Evaluating iteration {iteration + 1}",
                        iteration=iteration + 1,
                        phase="evaluation"
                    ))

                # 3. Evaluation
                if (iteration + 1) % self.config.EVAL_FREQUENCY == 0:
                    print(f"\n[3/4] Evaluation ({self.config.EVAL_GAMES} games)...")
                    t_start = time.time()

                    # Move networks to self-play device for evaluation (MCTS is faster on CPU)
                    self.network.to(self.selfplay_device)
                    self.best_network.to(self.selfplay_device)

                    # Temporarily override evaluator device
                    original_eval_device = self.evaluator.device
                    self.evaluator.device = self.selfplay_device

                    win_rate, wins, draws, losses = self.evaluator.evaluate(
                        self.network,
                        self.best_network,
                        num_games=self.config.EVAL_GAMES
                    )

                    # Restore evaluator device and move networks back
                    self.evaluator.device = original_eval_device
                    self.network.to(self.device)
                    self.best_network.to(self.device)

                    metrics['time_eval'] = time.time() - t_start
                    metrics['eval_win_rate'] = win_rate

                    print(f"  Win rate: {win_rate:.2%} "
                          f"(W/D/L: {wins}/{draws}/{losses})")
                    print(f"  Time: {metrics['time_eval']:.1f}s")

                    # Promote if better
                    if win_rate >= self.config.PROMOTION_THRESHOLD:
                        print(f"  ⭐ New best model! (win rate: {win_rate:.2%})")
                        self.best_network.load_state_dict(self.network.state_dict())
                        self.best_win_rate = win_rate
                        is_best = True
                    else:
                        is_best = False

                    # Update visualizer with evaluation results
                    if self.visualizer:
                        self.visualizer.update(EvaluationUpdate(
                            iteration=iteration + 1,
                            win_rate=win_rate,
                            wins=wins,
                            draws=draws,
                            losses=losses,
                            is_best=is_best
                        ))
                else:
                    metrics['eval_win_rate'] = -1
                    metrics['time_eval'] = 0
                    is_best = False

                # 4. Checkpointing
                if (iteration + 1) % self.config.SAVE_FREQUENCY == 0:
                    print(f"\n[4/4] Saving checkpoint...")

                    if self.visualizer:
                        self.visualizer.update(StatusUpdate(
                            message="Saving checkpoint",
                            iteration=iteration + 1,
                            phase="checkpoint"
                        ))

                    self.checkpoint_manager.save_checkpoint(
                        self.network,
                        self.optimizer,
                        iteration + 1,
                        metrics,
                        is_best=is_best
                    )

                # Log metrics
                self.log_metrics(metrics)

                # Increment buffer generation
                self.replay_buffer.increment_generation()

                # Update progress monitor
                progress_monitor.update(iteration + 1 - self.start_iteration, metrics)

                progress_monitor.update(iteration + 1 - self.start_iteration, metrics)

        finally:
            # Stop visualizer if running
            if self.visualizer:
                print("\nStopping visualizer...")
                self.visualizer.stop()

        # Print final summary
        progress_monitor.print_summary()

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

            # Data augmentation (8x)
            if self.config.AUGMENTATION:
                states, policies, values = augment_batch(
                    states, policies, values, aug_factor=self.config.AUG_FACTOR
                )

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

            # Poll events if visualizer is running
            if self.visualizer:
                self.visualizer.poll_events()

        print()  # New line after completion
        self.network.eval()

        return {
            'total_loss': np.mean(total_losses),
            'policy_loss': np.mean(policy_losses),
            'value_loss': np.mean(value_losses),
        }

    def resume_training(self, checkpoint_path: str):
        """
        Resume training from a checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
        """
        print(f"Resuming from checkpoint: {checkpoint_path}")

        checkpoint = self.checkpoint_manager.load_checkpoint(
            self.network,
            self.optimizer,
            checkpoint_path=checkpoint_path
        )

        self.start_iteration = checkpoint['iteration']

        print(f"Resumed from iteration {self.start_iteration}")

        # Initialize best_network
        if self.checkpoint_manager.best_checkpoint_path.exists():
            print("Loading best model from checkpoint...")
            self.checkpoint_manager.load_checkpoint(
                self.best_network,
                load_best=True
            )
        else:
            print("⚠️  No best_model.pt found. Initializing best_network with CURRENT resumed model weights.")
            print("    (This ensures the agent plays against a clone of itself, not a random network)")
            self.best_network.load_state_dict(self.network.state_dict())
