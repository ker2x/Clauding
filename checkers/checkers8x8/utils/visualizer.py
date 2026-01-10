"""
Real-time training visualization using matplotlib.
"""

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque
import threading


class TrainingVisualizer:
    """
    Real-time visualization of training metrics.

    Shows:
    - Total loss over time
    - Policy loss over time
    - Value loss over time
    - Buffer size
    """

    def __init__(self, max_points: int = 100):
        """
        Initialize visualizer.

        Args:
            max_points: Maximum number of points to show in plots
        """
        self.max_points = max_points

        # Data storage
        self.iterations = deque(maxlen=max_points)
        self.total_losses = deque(maxlen=max_points)
        self.policy_losses = deque(maxlen=max_points)
        self.value_losses = deque(maxlen=max_points)
        self.buffer_sizes = deque(maxlen=max_points)

        # Setup plot
        self.fig, self.axes = plt.subplots(2, 2, figsize=(12, 8))
        self.fig.suptitle('8x8 Checkers Training (Fixed Action Space)', fontsize=14, fontweight='bold')

        # Configure axes
        self.axes[0, 0].set_title('Total Loss')
        self.axes[0, 0].set_xlabel('Iteration')
        self.axes[0, 0].set_ylabel('Loss')
        self.axes[0, 0].grid(True, alpha=0.3)

        self.axes[0, 1].set_title('Policy Loss')
        self.axes[0, 1].set_xlabel('Iteration')
        self.axes[0, 1].set_ylabel('Loss')
        self.axes[0, 1].grid(True, alpha=0.3)

        self.axes[1, 0].set_title('Value Loss')
        self.axes[1, 0].set_xlabel('Iteration')
        self.axes[1, 0].set_ylabel('Loss')
        self.axes[1, 0].grid(True, alpha=0.3)

        self.axes[1, 1].set_title('Buffer Size')
        self.axes[1, 1].set_xlabel('Iteration')
        self.axes[1, 1].set_ylabel('Samples')
        self.axes[1, 1].grid(True, alpha=0.3)

        # Initialize lines
        self.lines = {
            'total': self.axes[0, 0].plot([], [], 'b-', linewidth=2)[0],
            'policy': self.axes[0, 1].plot([], [], 'r-', linewidth=2)[0],
            'value': self.axes[1, 0].plot([], [], 'g-', linewidth=2)[0],
            'buffer': self.axes[1, 1].plot([], [], 'm-', linewidth=2)[0],
        }

        plt.tight_layout()
        plt.ion()  # Interactive mode
        plt.show(block=False)

    def update(self, metrics: dict):
        """
        Update visualization with new metrics.

        Args:
            metrics: Dictionary with keys: iteration, total_loss, policy_loss, value_loss, buffer_size
        """
        # Add new data
        self.iterations.append(metrics.get('iteration', 0))
        self.total_losses.append(metrics.get('total_loss', 0))
        self.policy_losses.append(metrics.get('policy_loss', 0))
        self.value_losses.append(metrics.get('value_loss', 0))
        self.buffer_sizes.append(metrics.get('buffer_size', 0))

        # Update plots
        self._update_line('total', self.iterations, self.total_losses, self.axes[0, 0])
        self._update_line('policy', self.iterations, self.policy_losses, self.axes[0, 1])
        self._update_line('value', self.iterations, self.value_losses, self.axes[1, 0])
        self._update_line('buffer', self.iterations, self.buffer_sizes, self.axes[1, 1])

        # Refresh display
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.01)

    def _update_line(self, key: str, x_data, y_data, ax):
        """Update a single plot line."""
        if len(x_data) > 0 and len(y_data) > 0:
            self.lines[key].set_data(list(x_data), list(y_data))

            # Auto-scale
            ax.relim()
            ax.autoscale_view()

    def close(self):
        """Close the visualization window."""
        plt.close(self.fig)

    def show(self):
        """Show the plot window (blocking)."""
        plt.show(block=True)


# Testing
if __name__ == "__main__":
    import time
    import numpy as np

    print("Testing Training Visualizer...")

    viz = TrainingVisualizer(max_points=50)

    # Simulate training
    for i in range(1, 31):
        metrics = {
            'iteration': i,
            'total_loss': 4.0 * np.exp(-i / 10) + 0.5 + np.random.rand() * 0.2,
            'policy_loss': 3.0 * np.exp(-i / 10) + 0.3 + np.random.rand() * 0.1,
            'value_loss': 1.0 * np.exp(-i / 10) + 0.2 + np.random.rand() * 0.1,
            'buffer_size': min(i * 500, 50000),
        }

        viz.update(metrics)
        time.sleep(0.2)

    print("Visualization test complete. Close window to exit.")
    viz.show()
