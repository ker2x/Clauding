"""
Demo script for the real-time training visualizer.

Demonstrates the pygame visualization without running actual training.
Loads historical data from CSV logs and simulates real-time updates.

Usage:
    python scripts/demo_visualizer.py
"""

import sys
import os
import time
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from checkers.utils.visualizer import RealTimeVisualizer
from checkers.utils.visualizer_types import (
    MetricsUpdate, GameStateUpdate, EvaluationUpdate, StatusUpdate
)


def create_sample_board(iteration: int) -> np.ndarray:
    """Create a sample board state for demonstration."""
    board = np.zeros((10, 10), dtype=np.int8)
    
    # Initial setup (simplified)
    # Player 1 pieces (bottom half)
    for row in range(4):
        for col in range(10):
            if (row + col) % 2 == 1:
                board[row, col] = 1
    
    # Player 2 pieces (top half)
    for row in range(6, 10):
        for col in range(10):
            if (row + col) % 2 == 1:
                board[row, col] = -1
    
    # Add some kings and variation based on iteration
    if iteration % 5 == 0:
        board[2, 3] = 2  # Player 1 king
    if iteration % 7 == 0:
        board[7, 6] = -2  # Player 2 king
    
    # Simulate some moves by removing random pieces
    num_moves = min(iteration // 2, 15)
    for _ in range(num_moves):
        row = np.random.randint(0, 10)
        col = np.random.randint(0, 10)
        if (row + col) % 2 == 1:  # Only playable squares
            board[row, col] = 0
    
    return board


def main():
    """Run visualizer demo."""
    print("=" * 70)
    print("Real-Time Training Visualizer Demo")
    print("=" * 70)
    print("\nThis demo shows the pygame visualization without running training.")
    print("It will:")
    print("  1. Load historical data from logs/training_log.csv (if available)")
    print("  2. Simulate real-time training updates")
    print("  3. Display live game boards and metrics")
    print("\nControls:")
    print("  Q - Quit")
    print("  S - Save screenshot")
    print("\nStarting visualizer...")
    print("=" * 70)
    
    # Create visualizer
    log_file = "logs/training_log.csv"
    if not os.path.exists(log_file):
        print(f"\nNote: {log_file} not found. Starting with empty charts.")
        log_file = None
    
    viz = RealTimeVisualizer(
        window_width=1400,
        window_height=900,
        log_file=log_file,
        max_history=100
    )
    
    # Start visualizer thread
    viz.start()
    
    print("\n✓ Visualizer window opened!")
    print("\nSimulating training iterations...")
    print("(Close the pygame window or press Ctrl+C to stop)\n")
    
    # Simulate training updates
    try:
        start_iteration = viz.current_iteration + 1 if viz.current_iteration > 0 else 1
        iteration = start_iteration
        last_update_time = time.time()
        update_interval = 1.0  # Update every second
        
        running = True
        while running and iteration < start_iteration + 50:
            # Poll pygame events
            if not viz.poll_events():
                running = False
                break
            
            # Simulate updates at regular intervals
            current_time = time.time()
            if current_time - last_update_time >= update_interval:
                last_update_time = current_time
                
                # Simulate iteration phases
                
                # 1. Self-play phase
                status = StatusUpdate(
                    message=f"Self-play iteration {iteration}/50",
                    iteration=iteration,
                    phase="selfplay"
                )
                viz.update(status)
                
                # Send game state
                board = create_sample_board(iteration)
                policy = np.random.dirichlet(np.ones(150) * 0.5)  # Realistic-ish policy
                game_state = GameStateUpdate(
                    game_array=board,
                    policy=policy,
                    move_count=iteration * 15
                )
                viz.update(game_state)
                
                # 2. Training phase
                status = StatusUpdate(
                    message=f"Training iteration {iteration}",
                    iteration=iteration,
                    phase="training"
                )
                viz.update(status)
                
                # Send metrics (simulated loss decrease)
                base_loss = 10.0
                decay_rate = 0.05
                noise = np.random.normal(0, 0.3)
                
                total_loss = max(0.5, base_loss * np.exp(-decay_rate * iteration) + noise)
                policy_loss = total_loss * 0.65
                value_loss = total_loss * 0.35
                
                metrics = MetricsUpdate(
                    iteration=iteration,
                    total_loss=total_loss,
                    policy_loss=policy_loss,
                    value_loss=value_loss,
                    buffer_size=iteration * 5000,
                    time_selfplay=45.0 + np.random.normal(0, 5),
                    time_training=10.0 + np.random.normal(0, 2),
                    time_eval=2.5 if iteration % 10 == 0 else 0
                )
                viz.update(metrics)
                
                # 3. Evaluation phase (every 10 iterations)
                if iteration % 10 == 0:
                    status = StatusUpdate(
                        message=f"Evaluating iteration {iteration}",
                        iteration=iteration,
                        phase="evaluation"
                    )
                    viz.update(status)
                    
                    # Simulated win rate improvement
                    win_rate = min(0.95, 0.50 + (iteration / 100) * 0.45 + np.random.normal(0, 0.03))
                    is_best = (iteration % 20 == 0) and (iteration > 0)
                    
                    eval_update = EvaluationUpdate(
                        iteration=iteration,
                        win_rate=win_rate,
                        wins=int(50 * win_rate),
                        draws=int(50 * 0.2),
                        losses=int(50 * (1 - win_rate - 0.2)),
                        is_best=is_best
                    )
                    viz.update(eval_update)
                    
                    if is_best:
                        print(f"  ⭐ New best model at iteration {iteration}! Win rate: {win_rate:.1%}")
                
                # 4. Checkpoint phase (every 10 iterations)
                if iteration % 10 == 0:
                    status = StatusUpdate(
                        message="Saving checkpoint",
                        iteration=iteration,
                        phase="checkpoint"
                    )
                    viz.update(status)
                
                print(f"  Simulated iteration {iteration}/{start_iteration + 49} - Loss: {total_loss:.3f}")
                iteration += 1
        
        if running:
            print("\n" + "=" * 70)
            print("Demo simulation complete!")
            print("The visualizer will continue running. Close the window to exit.")
            print("=" * 70)
            
            # Keep visualizer running until user closes it
            while viz.poll_events():
                time.sleep(0.01)  # Small delay to avoid busy loop
            
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user")
    finally:
        print("\nStopping visualizer...")
        viz.stop()
        time.sleep(1)
    
    print("\n✓ Demo complete!")


if __name__ == "__main__":
    main()
