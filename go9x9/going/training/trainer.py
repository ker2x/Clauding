"""
Main training loop for 9x9 Go.
"""

import torch
import torch.nn.functional as F
import numpy as np
import time
import csv
from pathlib import Path
from typing import Dict

try:
    from ..network.resnet import GoNetwork
    from .replay_buffer import ReplayBuffer
    from .self_play import play_games_sequential, play_games_parallel
    from .evaluation import evaluate_models
except ImportError:
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
    from going.network.resnet import GoNetwork
    from going.training.replay_buffer import ReplayBuffer
    from going.training.self_play import play_games_sequential, play_games_parallel
    from going.training.evaluation import evaluate_models


class Trainer:
    """
    Main training loop for 9x9 Go.

    Orchestrates self-play, replay buffer, network training,
    checkpointing, and evaluation.
    """

    def __init__(self, network: GoNetwork, config,
                 device: torch.device, selfplay_device: torch.device):
        self.network = network
        self.config = config
        self.device = device
        self.selfplay_device = selfplay_device

        self.network.to(self.device)

        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.config.LEARNING_RATE,
            weight_decay=self.config.WEIGHT_DECAY
        )

        self.replay_buffer = ReplayBuffer(
            capacity=self.config.BUFFER_SIZE,
            recency_tau=self.config.RECENCY_TAU
        )

        self.start_iteration = 0
        self.best_model_path = Path(self.config.CHECKPOINT_DIR) / "best_model.pt"

        self.setup_logging()

    def setup_logging(self):
        log_dir = Path(self.config.LOG_DIR)
        log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = log_dir / "training_log.csv"

        if not self.log_file.exists():
            with open(self.log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'iteration', 'total_loss', 'policy_loss', 'value_loss',
                    'buffer_size', 'games_played',
                    'time_selfplay', 'time_training'
                ])

    def log_metrics(self, metrics: Dict):
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
        print("=" * 70)
        print("Starting 9x9 Go Training (AlphaZero)")
        print("=" * 70)
        print(f"Target: {num_iterations} iterations")
        if self.start_iteration > 0:
            print(f"Resuming from iteration {self.start_iteration}")
            if num_iterations <= self.start_iteration:
                print(f"\nERROR: Target ({num_iterations}) <= current ({self.start_iteration})")
                print(f"Use --iterations {self.start_iteration + 50}")
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

            self.network.to(self.selfplay_device)
            self.network.share_memory()
            self.network.eval()

            states, policies, values = play_games_parallel(
                self.network, self.config, self.selfplay_device,
                num_games=self.config.GAMES_PER_ITERATION
            )

            self.network.to(self.device)

            metrics['time_selfplay'] = time.time() - t_start
            metrics['games_played'] = self.config.GAMES_PER_ITERATION

            self.replay_buffer.add_batch(states, policies, values)

            print(f"  Generated {len(states)} training examples")
            print(f"  Time: {metrics['time_selfplay']:.1f}s")

            # 2. Training
            num_new_samples = len(states)
            num_steps = self._get_dynamic_steps(num_new_samples)
            print(f"\n[2/2] Training ({num_steps} steps)...")
            t_start = time.time()

            if len(self.replay_buffer) >= self.config.BATCH_SIZE:
                train_metrics = self.train_network(num_steps)
                metrics.update(train_metrics)
            else:
                print(f"  Skipping (buffer {len(self.replay_buffer)} < batch {self.config.BATCH_SIZE})")
                metrics['total_loss'] = 0.0
                metrics['policy_loss'] = 0.0
                metrics['value_loss'] = 0.0

            metrics['time_training'] = time.time() - t_start
            metrics['buffer_size'] = len(self.replay_buffer)

            print(f"  Loss: {metrics.get('total_loss', 0):.4f} "
                  f"(policy: {metrics.get('policy_loss', 0):.4f}, "
                  f"value: {metrics.get('value_loss', 0):.4f})")
            print(f"  Time: {metrics['time_training']:.1f}s")

            buffer_pct = len(self.replay_buffer) / self.config.BUFFER_SIZE
            bar_width = 30
            filled = int(bar_width * buffer_pct)
            bar = '=' * filled + '-' * (bar_width - filled)
            print(f"  Buffer: [{bar}] {buffer_pct*100:.1f}% "
                  f"({len(self.replay_buffer):,}/{self.config.BUFFER_SIZE:,})")

            self.log_metrics(metrics)
            self.replay_buffer.increment_generation()

            if (iteration + 1) % self.config.SAVE_FREQUENCY == 0:
                self.save_checkpoint(iteration + 1)

            if (iteration + 1) % self.config.EVAL_FREQUENCY == 0:
                self.evaluate_and_update_best(iteration + 1)

        self.save_checkpoint(num_iterations)
        if num_iterations > self.start_iteration and num_iterations % self.config.EVAL_FREQUENCY != 0:
            self.evaluate_and_update_best(num_iterations)

        print("\n" + "=" * 70)
        print("TRAINING COMPLETE!")
        print("=" * 70)

    def _get_dynamic_steps(self, num_new_samples: int) -> int:
        current_size = len(self.replay_buffer)
        capacity = self.config.BUFFER_SIZE

        if current_size < self.config.BATCH_SIZE:
            return int((num_new_samples * self.config.MIN_SAMPLE_REUSE) / self.config.BATCH_SIZE)

        saturation_point = capacity * 0.5
        scale = min(1.0, current_size / saturation_point)
        current_reuse = self.config.MIN_SAMPLE_REUSE + scale * (
            self.config.MAX_SAMPLE_REUSE - self.config.MIN_SAMPLE_REUSE
        )
        steps = (num_new_samples * current_reuse) / self.config.BATCH_SIZE
        return max(10, int(steps))

    def train_network(self, num_steps: int) -> Dict[str, float]:
        self.network.train()

        total_losses = []
        policy_losses = []
        value_losses = []

        bar_width = 40
        print("  ", end="", flush=True)

        for step in range(num_steps):
            states, policies, values = self.replay_buffer.sample(self.config.BATCH_SIZE)

            states_t = torch.from_numpy(states).to(self.device)
            policies_t = torch.from_numpy(policies).to(self.device)
            values_t = torch.from_numpy(values).unsqueeze(1).to(self.device)

            pred_policies, pred_values = self.network(states_t)

            policy_loss = -torch.sum(policies_t * F.log_softmax(pred_policies, dim=1)) / len(states_t)
            value_loss = F.mse_loss(pred_values, values_t)
            total_loss = policy_loss + value_loss

            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.config.GRAD_CLIP)
            self.optimizer.step()

            total_losses.append(total_loss.item())
            policy_losses.append(policy_loss.item())
            value_losses.append(value_loss.item())

            if (step + 1) % 10 == 0 or (step + 1) == num_steps:
                progress = (step + 1) / num_steps
                filled = int(bar_width * progress)
                bar = '=' * filled + '-' * (bar_width - filled)
                avg_loss = np.mean(total_losses[-10:])
                print(f"\r  [{bar}] {step + 1}/{num_steps} | "
                      f"loss: {avg_loss:.4f}", end="", flush=True)

        print()
        self.network.eval()

        return {
            'total_loss': np.mean(total_losses),
            'policy_loss': np.mean(policy_losses),
            'value_loss': np.mean(value_losses),
        }

    def save_checkpoint(self, iteration: int):
        checkpoint_dir = Path(self.config.CHECKPOINT_DIR)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            'iteration': iteration,
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'replay_buffer_state_dict': self.replay_buffer.state_dict(),
        }

        checkpoint_path = checkpoint_dir / f"checkpoint_iter_{iteration}.pt"
        torch.save(checkpoint, checkpoint_path)
        print(f"  Saved checkpoint: {checkpoint_path}")

        latest_path = checkpoint_dir / "latest.pt"
        torch.save(checkpoint, latest_path)

    def evaluate_and_update_best(self, iteration: int):
        print(f"\n[Evaluation at iteration {iteration}]")
        print("-" * 70)

        checkpoint_dir = Path(self.config.CHECKPOINT_DIR)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        if self.best_model_path.exists():
            print(f"  Loading best model from {self.best_model_path}...")
            best_model = GoNetwork(
                num_filters=self.config.NUM_FILTERS,
                num_res_blocks=self.config.NUM_RES_BLOCKS,
                policy_size=self.config.POLICY_SIZE,
                input_planes=self.config.INPUT_PLANES
            )
            checkpoint = torch.load(
                self.best_model_path, map_location=self.device, weights_only=False
            )
            best_model.load_state_dict(checkpoint['network_state_dict'])
            best_model.to(self.selfplay_device)
            best_model.eval()
        else:
            print("  No best model found, saving current as baseline")
            torch.save({
                'network_state_dict': self.network.state_dict(),
                'iteration': iteration,
                'win_rate': 0.5,
            }, self.best_model_path)
            print(f"  Saved initial baseline to {self.best_model_path}")
            print("-" * 70)
            return

        self.network.to(self.selfplay_device)
        self.network.eval()

        results = evaluate_models(
            current_model=self.network,
            best_model=best_model,
            config=self.config,
            device=self.selfplay_device,
        )

        self.network.to(self.device)

        win_rate = results['win_rate']
        print(f"\n  Results: W{results['wins']} L{results['losses']} D{results['draws']}")
        print(f"  Win rate: {win_rate:.1%}")

        win_loss_diff = results['win_loss_diff']
        if win_loss_diff > 0:
            print(f"  NEW BEST MODEL! (W{results['wins']} > L{results['losses']})")
            torch.save({
                'network_state_dict': self.network.state_dict(),
                'iteration': iteration,
                'win_rate': win_rate,
            }, self.best_model_path)
        else:
            print(f"  Model not promoted (W{results['wins']} <= L{results['losses']})")

        print("-" * 70)

    def load_checkpoint(self, checkpoint_path: str):
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        self.network.load_state_dict(checkpoint['network_state_dict'])
        print("  Network state restored")

        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("  Optimizer state restored")

        if 'replay_buffer_state_dict' in checkpoint:
            self.replay_buffer.load_state_dict(checkpoint['replay_buffer_state_dict'])
            print(f"  Replay buffer restored (size: {len(self.replay_buffer)})")

        self.start_iteration = checkpoint['iteration']
        print(f"  Resuming from iteration {self.start_iteration}")
