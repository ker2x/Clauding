"""
Main training loop for 9x9 Go.
"""

import torch
import torch.nn.functional as F
import numpy as np
import socket
import time
import csv
from pathlib import Path
from typing import Dict

try:
    from ..network.resnet import GoNetwork
    from .replay_buffer import ReplayBuffer
    from .self_play import play_games_sequential, play_games_parallel
    from .evaluation import evaluate_models
    from .distributed import send_msg, recv_msg, config_to_dict, DEFAULT_PORT
except ImportError:
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
    from going.network.resnet import GoNetwork
    from going.training.replay_buffer import ReplayBuffer
    from going.training.self_play import play_games_sequential, play_games_parallel
    from going.training.evaluation import evaluate_models
    from going.training.distributed import send_msg, recv_msg, config_to_dict, DEFAULT_PORT


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

        lr_schedule = getattr(self.config, 'LR_SCHEDULE', 'cosine')
        if lr_schedule == "flat":
            self.scheduler = None
        else:
            lr_min = getattr(self.config, 'LR_MIN', 1e-4)
            cosine_t0 = getattr(self.config, 'COSINE_T0', 50)
            cosine_t_mult = getattr(self.config, 'COSINE_T_MULT', 1)
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer, T_0=cosine_t0, T_mult=cosine_t_mult, eta_min=lr_min
            )

        self.replay_buffer = ReplayBuffer(
            capacity=self.config.BUFFER_SIZE,
            recency_tau=self.config.RECENCY_TAU,
            input_planes=self.config.INPUT_PLANES,
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
                    'ownership_loss', 'buffer_size', 'games_played',
                    'time_selfplay', 'time_training', 'learning_rate',
                    'avg_game_length'
                ])

    def log_metrics(self, metrics: Dict):
        with open(self.log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                metrics.get('iteration', 0),
                metrics.get('total_loss', 0),
                metrics.get('policy_loss', 0),
                metrics.get('value_loss', 0),
                metrics.get('ownership_loss', 0),
                metrics.get('buffer_size', 0),
                metrics.get('games_played', 0),
                metrics.get('time_selfplay', 0),
                metrics.get('time_training', 0),
                metrics.get('learning_rate', 0),
                metrics.get('avg_game_length', 0),
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

        if self.scheduler is None:
            print(f"LR schedule: flat ({self.config.LEARNING_RATE})")
        else:
            cosine_t0 = getattr(self.config, 'COSINE_T0', 50)
            print(f"LR schedule: CosineAnnealingWarmRestarts (T0={cosine_t0}, min={getattr(self.config, 'LR_MIN', 1e-4)})")
        print("=" * 70)

        for iteration in range(self.start_iteration, num_iterations):
            metrics = {'iteration': iteration + 1}

            # 1. Self-play
            print(f"\n[Iteration {iteration + 1}/{num_iterations}]")
            remote_host = getattr(self.config, 'REMOTE_SELFPLAY_HOST', None)
            if remote_host:
                print(f"[1/2] Remote self-play ({self.config.GAMES_PER_ITERATION} games → {remote_host})...")
            else:
                print(f"[1/2] Self-play ({self.config.GAMES_PER_ITERATION} games)...")
            t_start = time.time()

            if remote_host:
                states, policies, values, ownerships, surprises, game_lengths = \
                    self._remote_selfplay()
            else:
                self.network.to(self.selfplay_device)
                if self.selfplay_device.type == "cpu":
                    self.network.share_memory()
                self.network.eval()

                states, policies, values, ownerships, surprises, game_lengths = play_games_parallel(
                    self.network, self.config, self.selfplay_device,
                    num_games=self.config.GAMES_PER_ITERATION
                )

                self.network.to(self.device)

            metrics['time_selfplay'] = time.time() - t_start
            metrics['games_played'] = self.config.GAMES_PER_ITERATION
            metrics['avg_game_length'] = np.mean(game_lengths) if game_lengths else 0

            self.replay_buffer.add_batch(states, policies, values, ownerships, surprises)

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
                metrics['ownership_loss'] = 0.0

            metrics['time_training'] = time.time() - t_start
            metrics['buffer_size'] = len(self.replay_buffer)
            metrics['learning_rate'] = self.optimizer.param_groups[0]['lr']

            print(f"  Loss: {metrics.get('total_loss', 0):.4f} "
                  f"(policy: {metrics.get('policy_loss', 0):.4f}, "
                  f"value: {metrics.get('value_loss', 0):.4f}, "
                  f"ownership: {metrics.get('ownership_loss', 0):.4f})")
            print(f"  Time: {metrics['time_training']:.1f}s")

            buffer_pct = len(self.replay_buffer) / self.config.BUFFER_SIZE
            bar_width = 30
            filled = int(bar_width * buffer_pct)
            bar = '=' * filled + '-' * (bar_width - filled)
            print(f"  Buffer: [{bar}] {buffer_pct*100:.1f}% "
                  f"({len(self.replay_buffer):,}/{self.config.BUFFER_SIZE:,})")

            # Step LR scheduler
            if self.scheduler is not None:
                self.scheduler.step()
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f"  LR: {current_lr:.6f}")

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

    def _remote_selfplay(self):
        """Send weights to remote server, receive training data."""
        host = self.config.REMOTE_SELFPLAY_HOST
        port = getattr(self.config, 'REMOTE_SELFPLAY_PORT', DEFAULT_PORT)

        # Move weights to CPU for serialization
        cpu_state_dict = {k: v.cpu() for k, v in self.network.state_dict().items()}

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.connect((host, port))
            send_msg(sock, {
                'type': 'selfplay',
                'state_dict': cpu_state_dict,
                'config': config_to_dict(self.config),
            })
            print(f"  Weights sent, waiting for games...")

            result = recv_msg(sock)
            if result is None:
                raise ConnectionError("Remote server closed connection")
        finally:
            sock.close()

        return (
            result['states'],
            result['policies'],
            result['values'],
            result['ownerships'],
            result['surprises'],
            result['game_lengths'],
        )

    def train_network(self, num_steps: int) -> Dict[str, float]:
        self.network.train()

        total_losses = []
        policy_losses = []
        value_losses = []
        ownership_losses = []

        ownership_weight = getattr(self.config, 'OWNERSHIP_LOSS_WEIGHT', 1.5)
        bar_width = 40
        print("  ", end="", flush=True)

        for step in range(num_steps):
            states, policies, values, ownerships = self.replay_buffer.sample(self.config.BATCH_SIZE)

            states_t = torch.from_numpy(states).to(self.device)
            policies_t = torch.from_numpy(policies).to(self.device)
            values_t = torch.from_numpy(values).unsqueeze(1).to(self.device)
            ownerships_t = torch.from_numpy(ownerships).to(self.device)

            pred_policies, pred_values, pred_ownership = self.network(states_t)

            policy_loss = -torch.sum(policies_t * F.log_softmax(pred_policies, dim=1)) / len(states_t)
            value_loss = F.mse_loss(pred_values, values_t)
            ownership_loss = F.binary_cross_entropy(pred_ownership, ownerships_t)
            total_loss = policy_loss + value_loss + ownership_weight * ownership_loss

            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.config.GRAD_CLIP)
            self.optimizer.step()

            total_losses.append(total_loss.item())
            policy_losses.append(policy_loss.item())
            value_losses.append(value_loss.item())
            ownership_losses.append(ownership_loss.item())

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
            'ownership_loss': np.mean(ownership_losses),
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
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

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
                input_planes=self.config.INPUT_PLANES,
                global_pool_freq=getattr(self.config, 'GLOBAL_POOL_FREQ', 3),
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

        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("  Optimizer state restored")
        else:
            print("  No optimizer state in checkpoint, starting fresh")

        if 'replay_buffer_state_dict' in checkpoint:
            self.replay_buffer.load_state_dict(checkpoint['replay_buffer_state_dict'])
            print(f"  Replay buffer restored (size: {len(self.replay_buffer)})")

        if self.scheduler is None:
            # Flat LR: override whatever the optimizer loaded
            for pg in self.optimizer.param_groups:
                pg['lr'] = self.config.LEARNING_RATE
            print(f"  Flat LR set to {self.config.LEARNING_RATE}")
        elif 'scheduler_state_dict' in checkpoint:
            saved = checkpoint['scheduler_state_dict']
            is_saved_plateau = 'best' in saved
            is_current_plateau = isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)
            # Only restore if scheduler types match
            if is_saved_plateau == is_current_plateau:
                self.scheduler.load_state_dict(saved)
                print("  Scheduler state restored")
            else:
                print("  Scheduler type changed, resetting LR to configured value")
                for pg in self.optimizer.param_groups:
                    pg['lr'] = self.config.LEARNING_RATE

        self.start_iteration = checkpoint.get('iteration', 0)
        print(f"  Resuming from iteration {self.start_iteration}")
