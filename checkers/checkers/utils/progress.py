"""
Real-time progress monitoring for training.
"""

import time
from typing import Dict, Optional
from collections import deque


class ProgressMonitor:
    """
    Monitor and display training progress in real-time.
    """

    def __init__(self, total_iterations: int):
        """
        Initialize progress monitor.

        Args:
            total_iterations: Total number of training iterations
        """
        self.total_iterations = total_iterations
        self.start_time = time.time()
        self.iteration_times = deque(maxlen=10)  # Keep last 10 for ETA

        self.metrics_history = {
            'loss': deque(maxlen=50),
            'policy_loss': deque(maxlen=50),
            'value_loss': deque(maxlen=50),
            'win_rate': deque(maxlen=50),
        }

    def update(self, iteration: int, metrics: Dict):
        """
        Update progress with new iteration metrics.

        Args:
            iteration: Current iteration number
            metrics: Dictionary of metrics
        """
        # Record iteration time
        current_time = time.time()
        if iteration > 0:
            self.iteration_times.append(
                metrics.get('time_selfplay', 0) +
                metrics.get('time_training', 0) +
                metrics.get('time_eval', 0)
            )

        # Record metrics
        if 'total_loss' in metrics:
            self.metrics_history['loss'].append(metrics['total_loss'])
        if 'policy_loss' in metrics:
            self.metrics_history['policy_loss'].append(metrics['policy_loss'])
        if 'value_loss' in metrics:
            self.metrics_history['value_loss'].append(metrics['value_loss'])
        if 'eval_win_rate' in metrics and metrics['eval_win_rate'] >= 0:
            self.metrics_history['win_rate'].append(metrics['eval_win_rate'])

        # Print progress
        self._print_progress(iteration, metrics)

    def _print_progress(self, iteration: int, metrics: Dict):
        """Print formatted progress update."""
        # Calculate progress percentage
        progress = (iteration / self.total_iterations) * 100

        # Calculate ETA
        if len(self.iteration_times) > 0:
            avg_time = sum(self.iteration_times) / len(self.iteration_times)
            remaining = self.total_iterations - iteration
            eta_seconds = avg_time * remaining
            eta_str = self._format_time(eta_seconds)
        else:
            eta_str = "calculating..."

        # Calculate elapsed time
        elapsed = time.time() - self.start_time
        elapsed_str = self._format_time(elapsed)

        # Print header
        print(f"\n{'='*70}")
        print(f"Iteration {iteration}/{self.total_iterations} ({progress:.1f}%)")
        print(f"{'='*70}")

        # Print timing
        print(f"⏱️  Elapsed: {elapsed_str} | ETA: {eta_str}")

        # Print iteration breakdown
        iter_time = (metrics.get('time_selfplay', 0) +
                     metrics.get('time_training', 0) +
                     metrics.get('time_eval', 0))
        print(f"⏳ Iteration time: {iter_time:.1f}s "
              f"(self-play: {metrics.get('time_selfplay', 0):.1f}s, "
              f"training: {metrics.get('time_training', 0):.1f}s, "
              f"eval: {metrics.get('time_eval', 0):.1f}s)")

        # Print losses with trend
        loss = metrics.get('total_loss', 0)
        loss_trend = self._get_trend('loss')
        print(f"📉 Loss: {loss:.4f} {loss_trend} "
              f"(policy: {metrics.get('policy_loss', 0):.4f}, "
              f"value: {metrics.get('value_loss', 0):.4f})")

        # Print buffer info
        buffer_size = metrics.get('buffer_size', 0)
        buffer_pct = (buffer_size / 500_000) * 100 if buffer_size else 0
        print(f"💾 Buffer: {buffer_size:,} samples ({buffer_pct:.1f}% full)")

        # Print evaluation results
        if 'eval_win_rate' in metrics and metrics['eval_win_rate'] >= 0:
            win_rate = metrics['eval_win_rate']
            win_trend = self._get_trend('win_rate')
            promotion = "⭐ PROMOTED!" if win_rate >= 0.55 else ""
            print(f"🎯 Win rate: {win_rate:.1%} {win_trend} {promotion}")

        # Print progress bar
        self._print_progress_bar(iteration, self.total_iterations)

    def _print_progress_bar(self, current: int, total: int, width: int = 50):
        """Print a text-based progress bar."""
        progress = current / total
        filled = int(width * progress)
        bar = '█' * filled + '░' * (width - filled)
        print(f"[{bar}] {progress*100:.1f}%")

    def _get_trend(self, metric: str) -> str:
        """Get trend indicator for a metric."""
        history = self.metrics_history[metric]
        if len(history) < 2:
            return ""

        # Compare recent average to older average
        recent = list(history)[-5:]
        older = list(history)[-10:-5] if len(history) >= 10 else list(history)[:-5]

        if not older:
            return ""

        recent_avg = sum(recent) / len(recent)
        older_avg = sum(older) / len(older)

        if metric in ['loss', 'policy_loss', 'value_loss']:
            # For losses, lower is better
            if recent_avg < older_avg * 0.95:
                return "↓"
            elif recent_avg > older_avg * 1.05:
                return "↑"
        else:
            # For win rate, higher is better
            if recent_avg > older_avg * 1.05:
                return "↑"
            elif recent_avg < older_avg * 0.95:
                return "↓"

        return "→"

    def _format_time(self, seconds: float) -> str:
        """Format seconds into human-readable time."""
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.1f}m"
        elif seconds < 86400:
            hours = seconds / 3600
            return f"{hours:.1f}h"
        else:
            days = seconds / 86400
            return f"{days:.1f}d"

    def print_summary(self):
        """Print final training summary."""
        total_time = time.time() - self.start_time

        print("\n" + "="*70)
        print("🎉 TRAINING COMPLETE!")
        print("="*70)
        print(f"Total time: {self._format_time(total_time)}")

        if len(self.metrics_history['loss']) > 0:
            final_loss = self.metrics_history['loss'][-1]
            initial_loss = self.metrics_history['loss'][0]
            improvement = ((initial_loss - final_loss) / initial_loss) * 100
            print(f"Final loss: {final_loss:.4f} (improved {improvement:.1f}%)")

        if len(self.metrics_history['win_rate']) > 0:
            best_win_rate = max(self.metrics_history['win_rate'])
            print(f"Best win rate: {best_win_rate:.1%}")

        print("="*70)


class SimpleProgressBar:
    """
    Simple progress bar for loops (when tqdm not available).
    """

    def __init__(self, total: int, desc: str = ""):
        self.total = total
        self.desc = desc
        self.current = 0
        self.start_time = time.time()

    def update(self, n: int = 1):
        """Update progress."""
        self.current += n
        self._print()

    def _print(self):
        """Print progress bar."""
        progress = self.current / self.total
        filled = int(30 * progress)
        bar = '█' * filled + '░' * (30 - filled)

        elapsed = time.time() - self.start_time
        if self.current > 0:
            rate = self.current / elapsed
            eta = (self.total - self.current) / rate if rate > 0 else 0
            eta_str = f"{eta:.0f}s"
        else:
            eta_str = "?"

        print(f"\r{self.desc}: [{bar}] {self.current}/{self.total} | ETA: {eta_str}",
              end='', flush=True)

        if self.current >= self.total:
            print()  # New line when complete

    def close(self):
        """Close progress bar."""
        if self.current < self.total:
            self.current = self.total
            self._print()


# Example usage
if __name__ == "__main__":
    import random

    print("Testing ProgressMonitor...")

    monitor = ProgressMonitor(total_iterations=10)

    for i in range(1, 11):
        # Simulate metrics
        metrics = {
            'time_selfplay': random.uniform(40, 60),
            'time_training': random.uniform(8, 12),
            'time_eval': 30 if i % 3 == 0 else 0,
            'total_loss': 10 / (i + 1),
            'policy_loss': 7 / (i + 1),
            'value_loss': 3 / (i + 1),
            'buffer_size': i * 5000,
        }

        if i % 3 == 0:
            metrics['eval_win_rate'] = min(0.4 + i * 0.05, 0.9)

        monitor.update(i, metrics)
        time.sleep(0.5)  # Simulate work

    monitor.print_summary()

    print("\n\nTesting SimpleProgressBar...")
    pbar = SimpleProgressBar(100, desc="Processing")
    for i in range(100):
        pbar.update(1)
        time.sleep(0.01)
    pbar.close()

    print("\n✓ Progress monitoring tests passed!")
