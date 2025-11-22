"""
Watch a trained SAC agent play CarRacing-v3 with "Brain" visualization.

This script visualizes the internal state of the agent:
1. Q-Value Landscape: A heatmap showing the estimated value of different actions.
2. Policy Distribution: The probability distribution of the agent's policy.
3. Game View: The standard environment render.

Usage:
    python watch_brain.py --checkpoint checkpoints/best_model.pt
    python watch_brain.py --checkpoint checkpoints/best_model.pt --resolution 20
"""

from __future__ import annotations

import argparse
import time
from typing import Any, Tuple

import cv2
import gymnasium as gym
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import torch
from matplotlib.backends.backend_agg import FigureCanvasAgg

from preprocessing import make_carracing_env
from sac import SACAgent
from utils.display import format_action, get_car_speed
from frame_buffer import FrameBuffer
from config.physics_config import ObservationParams, get_base_observation_dim

# Use non-interactive backend
matplotlib.use('Agg')


class BrainVisualizer:
    def __init__(self, resolution: int = 16):
        """
        Initialize the brain visualizer.

        Args:
            resolution: Resolution for the Q-value heatmap (resolution x resolution grid).
        """
        self.resolution = resolution
        
        # Create figure
        self.fig = plt.figure(figsize=(14, 10), facecolor='#1a1a1a')
        
        # Grid layout
        # Top Left: Game View (placeholder, handled by OpenCV usually, but we can embed it)
        # Top Right: Q-Value Heatmap
        # Bottom: Policy Distributions
        
        gs = self.fig.add_gridspec(2, 2, height_ratios=[1.5, 1])
        
        self.ax_q = self.fig.add_subplot(gs[0, 1])
        self.ax_policy_steer = self.fig.add_subplot(gs[1, 0])
        self.ax_policy_accel = self.fig.add_subplot(gs[1, 1])
        
        # Pre-compute action grid for Q-value heatmap
        # Steering: [-1, 1]
        # Acceleration: [-1, 1] (covers gas and brake combined)
        steer_vals = np.linspace(-1, 1, resolution)
        accel_vals = np.linspace(-1, 1, resolution)
        self.steer_grid, self.accel_grid = np.meshgrid(steer_vals, accel_vals)
        
        # Flatten for batch processing
        self.action_batch = np.column_stack([
            self.steer_grid.ravel(),
            self.accel_grid.ravel()
        ])
        self.action_batch_tensor = torch.FloatTensor(self.action_batch)

    def compute_q_values(self, agent: SACAgent, state: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """Compute Q-values for the entire action grid given the current state."""
        # Prepare state batch (repeat state for each action in grid)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
        state_batch = state_tensor.repeat(len(self.action_batch), 1)
        
        action_batch = self.action_batch_tensor.to(agent.device)
        
        with torch.no_grad():
            # Use critic 1 for visualization
            # The critic forward expects (state, action)
            q_values = agent.critic_1(state_batch, action_batch)
            
        return q_values.cpu().numpy().reshape(self.resolution, self.resolution)

    def render(
        self,
        agent: SACAgent,
        state: npt.NDArray[np.float32],
        action: npt.NDArray[np.float32],
        game_frame: npt.NDArray[np.uint8]
    ) -> npt.NDArray[np.uint8]:
        """
        Render the brain visualization.

        Args:
            agent: The SAC agent
            state: Current state vector
            action: Current action taken
            game_frame: The RGB frame from the game

        Returns:
            BGR image of the visualization
        """
        self.fig.clear()
        
        # Re-create subplots (clearing is safer than updating for complex plots)
        gs = self.fig.add_gridspec(2, 2, height_ratios=[1.5, 1])
        ax_game = self.fig.add_subplot(gs[0, 0])
        ax_q = self.fig.add_subplot(gs[0, 1])
        ax_policy_steer = self.fig.add_subplot(gs[1, 0])
        ax_policy_accel = self.fig.add_subplot(gs[1, 1])
        
        # Style
        for ax in [ax_game, ax_q, ax_policy_steer, ax_policy_accel]:
            ax.set_facecolor('#1a1a1a')
            ax.tick_params(colors='white')
            for spine in ax.spines.values():
                spine.set_color('white')
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
            ax.title.set_color('white')

        # 1. Game View
        ax_game.imshow(game_frame)
        ax_game.set_title("Game View", fontsize=12, fontweight='bold')
        ax_game.axis('off')
        
        # 2. Q-Value Heatmap
        q_grid = self.compute_q_values(agent, state)

        # Use percentile-based clipping for better contrast
        # This removes extreme outliers and focuses on the "interesting" range
        q_min = np.percentile(q_grid, 5)
        q_max = np.percentile(q_grid, 95)

        # Use a diverging colormap centered at median for better visual contrast
        q_median = np.median(q_grid)

        # Symmetric range around median for diverging colormap
        q_range = max(abs(q_max - q_median), abs(q_median - q_min))
        vmin = q_median - q_range
        vmax = q_median + q_range

        # Plot heatmap with enhanced contrast
        im = ax_q.imshow(
            q_grid,
            extent=[-1, 1, -1, 1],
            origin='lower',
            cmap='RdYlGn',  # Red (bad) -> Yellow (neutral) -> Green (good)
            aspect='auto',
            vmin=vmin,
            vmax=vmax
        )
        cbar = plt.colorbar(im, ax=ax_q, label='Q-Value')

        # Add statistics text
        max_q = np.max(q_grid)
        min_q = np.min(q_grid)
        stats_text = f'Range: [{min_q:.1f}, {max_q:.1f}]\nMedian: {q_median:.1f}'
        ax_q.text(0.02, 0.98, stats_text, transform=ax_q.transAxes,
                 fontsize=8, verticalalignment='top', color='white',
                 bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))
        
        # Mark best Q-value action
        best_idx = np.argmax(q_grid)
        best_steer = self.steer_grid.ravel()[best_idx]
        best_accel = self.accel_grid.ravel()[best_idx]
        ax_q.plot(best_steer, best_accel, 'g^', markersize=12, markeredgecolor='white',
                 markeredgewidth=2, label='Best Q-Value')

        # Mark current action
        ax_q.plot(action[0], action[1], 'ro', markersize=10, markeredgecolor='white', label='Current Action')

        # Mark policy mean (what the agent "wants" to do deterministically)
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
            mean, _ = agent.actor(state_tensor)
            mean_np = mean.cpu().numpy()[0]
            # Apply tanh to mean because the actor outputs are pre-tanh in some implementations,
            # but looking at actor.py:
            # mean = self.mean(x) -> Linear layer
            # The actor outputs mean and log_std.
            # In select_action, it does: action = torch.tanh(mean) if evaluate=True.
            # So the raw mean is NOT bounded. We must apply tanh.
            mean_action = np.tanh(mean_np)

        ax_q.plot(mean_action[0], mean_action[1], 'c*', markersize=12, markeredgecolor='white', label='Policy Mean')
        
        ax_q.set_title("Q-Value Landscape", fontsize=12, fontweight='bold')
        ax_q.set_xlabel("Steering")
        ax_q.set_ylabel("Acceleration")
        ax_q.grid(True, alpha=0.3)
        ax_q.legend(loc='upper right', fontsize='small')
        
        # 3. Policy Distributions
        # We need the distribution parameters
        with torch.no_grad():
            mean, log_std = agent.actor(state_tensor)
            std = log_std.exp()
            
            # Create Normal distributions
            dist_steer = torch.distributions.Normal(mean[0, 0], std[0, 0])
            dist_accel = torch.distributions.Normal(mean[0, 1], std[0, 1])
            
            # Generate points for plotting
            x_steer = torch.linspace(-3, 3, 100).to(agent.device)
            prob_steer = torch.exp(dist_steer.log_prob(x_steer))
            
            x_accel = torch.linspace(-3, 3, 100).to(agent.device)
            prob_accel = torch.exp(dist_accel.log_prob(x_accel))
            
            # Convert to numpy
            x_steer_np = x_steer.cpu().numpy()
            prob_steer_np = prob_steer.cpu().numpy()
            x_accel_np = x_accel.cpu().numpy()
            prob_accel_np = prob_accel.cpu().numpy()
            
            # Note: These are distributions over the PRE-TANH values.
            # The actual action is tanh(z). 
            # Visualizing the pre-tanh distribution is often clearer for Gaussian structure.
            # We will mark the pre-tanh value of the current action.
            # Inverse tanh is atanh.
            
        # Steer Policy
        ax_policy_steer.plot(x_steer_np, prob_steer_np, 'c-', linewidth=2)
        ax_policy_steer.fill_between(x_steer_np, prob_steer_np, alpha=0.3, color='cyan')
        ax_policy_steer.set_title("Steering Policy (Pre-tanh)", fontsize=10)
        ax_policy_steer.set_xlim(-3, 3)
        ax_policy_steer.grid(True, alpha=0.3)
        
        # Accel Policy
        ax_policy_accel.plot(x_accel_np, prob_accel_np, 'm-', linewidth=2)
        ax_policy_accel.fill_between(x_accel_np, prob_accel_np, alpha=0.3, color='magenta')
        ax_policy_accel.set_title("Acceleration Policy (Pre-tanh)", fontsize=10)
        ax_policy_accel.set_xlim(-3, 3)
        ax_policy_accel.grid(True, alpha=0.3)

        # Draw
        self.fig.tight_layout()
        self.fig.canvas.draw()
        
        # Convert to numpy
        buf = self.fig.canvas.buffer_rgba()
        img = np.asarray(buf)
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        
        return img_bgr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Watch SAC agent with Brain Visualization')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--resolution', type=int, default=20, help='Resolution of Q-value heatmap')
    parser.add_argument('--episodes', type=int, default=5, help='Number of episodes')
    parser.add_argument('--fps', type=int, default=30, help='Target FPS')
    parser.add_argument('--save-video', action='store_true', help='Save visualization to brain_watch.mp4')
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location='cpu')

    # Setup Agent
    # Read state_dim from checkpoint to ensure compatibility
    # (observation space is configurable via config/physics_config.py)
    state_dim = checkpoint.get('state_dim', 73)  # Default to 73 if not found (legacy checkpoints)
    action_dim = checkpoint.get('action_dim', 2)

    print(f"Creating agent with state_dim={state_dim}, action_dim={action_dim}")
    agent = SACAgent(state_dim=state_dim, action_dim=action_dim)
    agent.load(args.checkpoint)
    print("Agent loaded.")

    # Setup Environment
    # Import training constants for consistency
    from config.constants import (
        DEFAULT_TERMINATE_STATIONARY,
        DEFAULT_STATIONARY_PATIENCE,
        DEFAULT_REWARD_SHAPING,
        DEFAULT_MIN_EPISODE_STEPS,
        DEFAULT_SHORT_EPISODE_PENALTY,
        DEFAULT_MAX_EPISODE_STEPS,
    )

    env = make_carracing_env(
        render_mode='rgb_array',
        terminate_stationary=DEFAULT_TERMINATE_STATIONARY,
        stationary_patience=DEFAULT_STATIONARY_PATIENCE,
        reward_shaping=DEFAULT_REWARD_SHAPING,
        min_episode_steps=DEFAULT_MIN_EPISODE_STEPS,
        short_episode_penalty=DEFAULT_SHORT_EPISODE_PENALTY,
        max_episode_steps=DEFAULT_MAX_EPISODE_STEPS,
    )

    # Detect frame stacking from checkpoint dimension
    obs_params = ObservationParams()
    base_state_dim = get_base_observation_dim(obs_params.NUM_LOOKAHEAD)
    env_state_dim = env.observation_space.shape[0]

    # Check if checkpoint uses frame stacking
    if state_dim > base_state_dim and state_dim % base_state_dim == 0:
        frame_stack = state_dim // base_state_dim
        print(f"✓ Detected frame stacking: {frame_stack} frames ({base_state_dim} × {frame_stack} = {state_dim}D)")
    else:
        frame_stack = 1
        print(f"✓ No frame stacking (single frame: {state_dim}D)")

    # Verify base dimensions match
    if env_state_dim != base_state_dim:
        print(f"\n{'='*60}")
        print(f"⚠️  WARNING: Base Dimension Mismatch!")
        print(f"{'='*60}")
        print(f"Checkpoint base dimension: {base_state_dim}D")
        print(f"Current environment: {env_state_dim}D")
        print(f"\nCurrent config (config/physics_config.py):")
        print(f"  NUM_LOOKAHEAD = {obs_params.NUM_LOOKAHEAD}")
        print(f"  WAYPOINT_STRIDE = {obs_params.WAYPOINT_STRIDE}")
        print(f"\nTo use this checkpoint, adjust config to match:")
        print(f"Common configurations:")
        print(f"  - 73D: NUM_LOOKAHEAD=20, WAYPOINT_STRIDE=1")
        print(f"  - 53D: NUM_LOOKAHEAD=10, WAYPOINT_STRIDE=2")
        print(f"  - 63D: NUM_LOOKAHEAD=15, WAYPOINT_STRIDE=2")
        print(f"{'='*60}")
        env.close()
        raise RuntimeError(f"Cannot use checkpoint with base {base_state_dim}D and environment {env_state_dim}D")

    # Create frame buffer if frame stacking is used
    frame_buffer = None
    if frame_stack > 1:
        frame_buffer = FrameBuffer(frame_stack=frame_stack, state_shape=base_state_dim)
        print(f"✓ Frame buffer created (stack depth: {frame_stack})")
    
    # Visualizer
    visualizer = BrainVisualizer(resolution=args.resolution)
    
    # Video Writer
    video_writer = None
    if args.save_video:
        # We need to know the frame size first. Let's do a dummy render or just wait for the first frame.
        # But we can also just hardcode or infer from figure size.
        # Matplotlib figure is 14x10 inches. At 100 DPI (default), that's 1400x1000.
        # Let's wait for the first frame to be safe.
        pass

    try:
        for episode in range(args.episodes):
            state, _ = env.reset()

            # Initialize frame buffer for this episode
            if frame_buffer is not None:
                frame_buffer.reset(state)

            done = False
            total_reward = 0
            step = 0

            print(f"Episode {episode+1} started...")

            while not done:
                start_time = time.time()

                # Get observation for agent (stacked if frame stacking enabled)
                if frame_buffer is not None:
                    obs = frame_buffer.get()
                else:
                    obs = state

                # Select action
                action = agent.select_action(obs, evaluate=True)

                # Step
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                total_reward += reward
                step += 1

                # Update frame buffer
                if frame_buffer is not None:
                    frame_buffer.append(next_state)

                # Render (visualize with current single-frame state for interpretability)
                game_frame = env.render()
                brain_view = visualizer.render(agent, obs, action, game_frame)
                
                # Initialize video writer if needed
                if args.save_video and video_writer is None:
                    height, width, _ = brain_view.shape
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    video_writer = cv2.VideoWriter('brain_watch.mp4', fourcc, args.fps, (width, height))
                    print(f"Recording video to brain_watch.mp4 ({width}x{height})...")
                
                # Write frame
                if video_writer is not None:
                    video_writer.write(brain_view)
                
                cv2.imshow('Brain Watch', brain_view)
                
                # Input handling
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:
                    print("Quitting...")
                    return
                
                state = next_state
                
                # FPS Control
                dt = time.time() - start_time
                if dt < 1.0 / args.fps:
                    time.sleep(1.0 / args.fps - dt)
            
            print(f"Episode {episode+1} finished. Total Reward: {total_reward:.2f}")
            
    except KeyboardInterrupt:
        print("Interrupted.")
    finally:
        env.close()
        if video_writer is not None:
            video_writer.release()
            print("Video saved to brain_watch.mp4")
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
