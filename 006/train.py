"""
Training script for SAC agent on CarRacing-v3.

This script:
1. Creates the CarRacing environment with preprocessing
2. Initializes the SAC agent
3. Trains the agent with experience replay
4. Saves checkpoints periodically
5. Logs training metrics (rewards, losses, alpha)
6. Generates training progress plots

Usage:
    # Basic training (no domain randomization)
    python train.py

    # Training with domain randomization (conservative)
    python train.py --domain-randomization conservative

    # Training with moderate domain randomization
    python train.py --domain-randomization moderate

    # Training with aggressive domain randomization
    python train.py --domain-randomization aggressive

    # Training for wet/slippery conditions
    python train.py --domain-randomization wet

    # Custom configuration
    python train.py --episodes 2000 --learning-starts 5000

    # Force CPU-only mode
    python train.py --device cpu

    # Resume from checkpoint
    python train.py --resume checkpoints/best_model.pt --episodes 1000

    # Combine options (training with domain randomization on CPU)
    python train.py --domain-randomization moderate --device cpu --episodes 1000
"""

from __future__ import annotations

import argparse
import os
import time
import csv
from datetime import datetime
from typing import Any
import numpy as np
import torch
import multiprocessing

# Try to import matplotlib with non-GUI backend
MATPLOTLIB_AVAILABLE = False
try:
    import matplotlib
    # Set non-GUI backend to avoid tkinter dependency
    # This must be done BEFORE importing pyplot
    try:
        matplotlib.use('Agg', force=False)  # Don't force if already set
    except:
        # If backend is already set, check if it's suitable
        current_backend = matplotlib.get_backend()
        print(f"Matplotlib backend already set to: {current_backend}")
        if current_backend.lower() in ['agg', 'pdf', 'ps', 'svg']:
            print("  (non-GUI backend detected - OK for headless training)")
        else:
            print("  (GUI backend detected - this may cause issues in headless environments)")

    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
    print(f"Matplotlib loaded successfully (backend: {matplotlib.get_backend()})")
except (ImportError, RuntimeError) as e:
    print(f"Warning: matplotlib not available ({e})")
    print("Training will continue without plotting functionality.")

from preprocessing import make_carracing_env
from sac import SACAgent, ReplayBuffer
from training_utils import get_device, configure_cpu_threading, evaluate_agent, setup_logging
from config.constants import *
from config.domain_randomization import (
    DomainRandomizationConfig,
    conservative_randomization,
    moderate_randomization,
    aggressive_randomization,
    wet_surface_conditions,
)
from config.physics_config import ObservationParams, get_base_observation_dim


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Train SAC agent on CarRacing-v3')

    # Training parameters
    parser.add_argument('--episodes', type=int, default=DEFAULT_EPISODES,
                        help=f'Number of episodes to train (default: {DEFAULT_EPISODES})')
    parser.add_argument('--learning-starts', type=int, default=DEFAULT_LEARNING_STARTS,
                        help=f'Steps before training starts (default: {DEFAULT_LEARNING_STARTS})')
    parser.add_argument('--eval-frequency', type=int, default=DEFAULT_EVAL_FREQUENCY,
                        help=f'Evaluate every N episodes (default: {DEFAULT_EVAL_FREQUENCY})')
    parser.add_argument('--checkpoint-frequency', type=int, default=DEFAULT_CHECKPOINT_FREQUENCY,
                        help=f'Save checkpoint every N episodes (default: {DEFAULT_CHECKPOINT_FREQUENCY})')

    # Agent hyperparameters
    parser.add_argument('--lr-actor', type=float, default=DEFAULT_LR_ACTOR,
                        help=f'Actor learning rate (default: {DEFAULT_LR_ACTOR})')
    parser.add_argument('--lr-critic', type=float, default=DEFAULT_LR_CRITIC,
                        help=f'Critic learning rate (default: {DEFAULT_LR_CRITIC})')
    parser.add_argument('--lr-alpha', type=float, default=DEFAULT_LR_ALPHA,
                        help=f'Alpha learning rate (default: {DEFAULT_LR_ALPHA})')
    parser.add_argument('--gamma', type=float, default=DEFAULT_GAMMA,
                        help=f'Discount factor (default: {DEFAULT_GAMMA})')
    parser.add_argument('--tau', type=float, default=DEFAULT_TAU,
                        help=f'Soft update coefficient (default: {DEFAULT_TAU})')
    parser.add_argument('--buffer-size', type=int, default=DEFAULT_BUFFER_SIZE,
                        help=f'Replay buffer size (default: {DEFAULT_BUFFER_SIZE})')
    parser.add_argument('--batch-size', type=int, default=DEFAULT_BATCH_SIZE,
                        help=f'Training batch size (default: {DEFAULT_BATCH_SIZE})')
    parser.add_argument('--auto-entropy-tuning', action='store_true', default=True,
                        help='Use automatic entropy tuning (default: True)')

    # Environment parameters (vector mode only)

    # Domain randomization
    parser.add_argument('--domain-randomization', type=str, default='none',
                        choices=['none', 'conservative', 'moderate', 'aggressive', 'wet'],
                        help='Domain randomization preset: none (default), conservative, moderate, aggressive, or wet')

    # Resume training
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')

    # Paths
    parser.add_argument('--checkpoint-dir', type=str, default=DEFAULT_CHECKPOINT_DIR,
                        help=f'Directory to save checkpoints (default: {DEFAULT_CHECKPOINT_DIR})')
    parser.add_argument('--log-dir', type=str, default=DEFAULT_LOG_DIR,
                        help=f'Directory to save logs (default: {DEFAULT_LOG_DIR})')

    # Device selection
    parser.add_argument('--device', type=str, default=DEFAULT_DEVICE, choices=['auto', 'cpu', 'cuda', 'mps'],
                        help=f'Device to use for training: auto (default), cpu, cuda, or mps')

    # Debugging
    parser.add_argument('--verbose', action='store_true', default=False,
                        help='Enable verbose mode from environment for debugging (default: False)')

    return parser.parse_args()


def plot_training_progress(
    episode_rewards: list[float], metrics: dict[str, list[float]], save_path: str
) -> None:
    """
    Plot training metrics and save to file.

    Args:
        episode_rewards: List of episode rewards
        metrics: Dictionary of lists containing training metrics
        save_path: Path to save plot
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Skipping plot generation (matplotlib not available)")
        return

    try:
        fig, axes = plt.subplots(4, 1, figsize=(10, 14))

        # Plot rewards
        axes[0].plot(episode_rewards, alpha=0.6, label='Episode Reward')
        # Moving average (last 100 episodes)
        if len(episode_rewards) >= 100:
            moving_avg = np.convolve(episode_rewards, np.ones(100)/100, mode='valid')
            axes[0].plot(range(99, len(episode_rewards)), moving_avg, 'r-', linewidth=2, label='Moving Avg (100 ep)')
        axes[0].set_xlabel('Episode')
        axes[0].set_ylabel('Reward')
        axes[0].set_title('Training Rewards')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Plot critic losses
        if len(metrics['critic_1_loss']) > 0:
            axes[1].plot(metrics['critic_1_loss'], alpha=0.6, label='Critic 1 Loss')
            axes[1].plot(metrics['critic_2_loss'], alpha=0.6, label='Critic 2 Loss')
            axes[1].set_xlabel('Episode')
            axes[1].set_ylabel('Loss')
            axes[1].set_title('Critic Losses')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)

        # Plot actor loss
        if len(metrics['actor_loss']) > 0:
            axes[2].plot(metrics['actor_loss'], alpha=0.6, label='Actor Loss', color='green')
            axes[2].set_xlabel('Episode')
            axes[2].set_ylabel('Loss')
            axes[2].set_title('Actor Loss')
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)

        # Plot alpha (entropy coefficient)
        if len(metrics['alpha']) > 0:
            axes[3].plot(metrics['alpha'], alpha=0.8, label='Alpha (Entropy Coef)', color='purple')
            axes[3].set_xlabel('Episode')
            axes[3].set_ylabel('Alpha')
            axes[3].set_title('Entropy Coefficient (Alpha)')
            axes[3].legend()
            axes[3].grid(True, alpha=0.3)

        plt.tight_layout()

        # Save with explicit flushing
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)  # Close specific figure
        plt.close('all')  # Close all figures to prevent memory leaks

        # Force matplotlib to clear its internal cache
        if hasattr(plt, '_cachedRenderer'):
            plt._cachedRenderer = None

        # Verify the file was actually written
        if os.path.exists(save_path):
            file_size = os.path.getsize(save_path)
            file_mtime = os.path.getmtime(save_path)
            from datetime import datetime
            mod_time = datetime.fromtimestamp(file_mtime).strftime('%Y-%m-%d %H:%M:%S')
            print(f"Training plot saved to {save_path} ({file_size} bytes, modified: {mod_time})")
        else:
            print(f"WARNING: Plot file was not created at {save_path}")
    except Exception as e:
        print(f"ERROR: Failed to generate training plot: {e}")
        print(f"  Type: {type(e).__name__}")
        import traceback
        traceback.print_exc()


def train(args: argparse.Namespace) -> None:
    """Main training loop."""
    # Environment configuration (using constants from constants.py)
    TERMINATE_STATIONARY = DEFAULT_TERMINATE_STATIONARY
    STATIONARY_PATIENCE = DEFAULT_STATIONARY_PATIENCE
    REWARD_SHAPING = DEFAULT_REWARD_SHAPING
    MIN_EPISODE_STEPS = DEFAULT_MIN_EPISODE_STEPS
    SHORT_EPISODE_PENALTY = DEFAULT_SHORT_EPISODE_PENALTY
    MAX_EPISODE_STEPS = DEFAULT_MAX_EPISODE_STEPS

    # Create directories
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    # Determine device
    device = get_device(args.device)
    print(f"Using device: {device}")

    # Configure CPU threading for optimal performance
    configure_cpu_threading(device)

    # Configure domain randomization
    domain_rand_config = None
    if args.domain_randomization != 'none':
        if args.domain_randomization == 'conservative':
            domain_rand_config = conservative_randomization()
        elif args.domain_randomization == 'moderate':
            domain_rand_config = moderate_randomization()
        elif args.domain_randomization == 'aggressive':
            domain_rand_config = aggressive_randomization()
        elif args.domain_randomization == 'wet':
            domain_rand_config = wet_surface_conditions()
        print(f"Domain randomization: {args.domain_randomization}")
    else:
        print("Domain randomization: disabled")

    # Create training environment
    print("Creating CarRacing-v3 environment...")
    env = make_carracing_env(
        terminate_stationary=TERMINATE_STATIONARY,
        stationary_patience=STATIONARY_PATIENCE,
        render_mode=None,
        reward_shaping=REWARD_SHAPING,
        min_episode_steps=MIN_EPISODE_STEPS,
        short_episode_penalty=SHORT_EPISODE_PENALTY,
        max_episode_steps=MAX_EPISODE_STEPS,
        verbose=args.verbose,
        domain_randomization_config=domain_rand_config,
    )

    action_dim = env.action_space.shape[0]
    state_dim = env.observation_space.shape[0]  # Environment handles frame stacking internally

    # Frame stacking configuration (done by environment)
    obs_params = ObservationParams()
    frame_stack = obs_params.FRAME_STACK
    base_obs_dim = get_base_observation_dim(obs_params.NUM_LOOKAHEAD)

    print(f"Environment created:")
    print(f"  State mode: vector")
    print(f"  Base observation dimension: {base_obs_dim}")
    if frame_stack > 1:
        print(f"  Frame stacking: {frame_stack} frames (handled by environment)")
        print(f"  Stacked state dimension: {state_dim} ({base_obs_dim} × {frame_stack})")
    else:
        print(f"  Frame stacking: disabled")
        print(f"  State dimension: {state_dim}")
    print(f"  Action space: Continuous (2D)")
    print(f"  Max episode steps: {MAX_EPISODE_STEPS} (prevents infinite episodes)")
    print(f"  Early termination enabled (patience={STATIONARY_PATIENCE} frames)")
    print(f"  Reward shaping enabled (penalty {SHORT_EPISODE_PENALTY} for episodes < {MIN_EPISODE_STEPS} steps)")
    if domain_rand_config and domain_rand_config.enabled:
        print(f"  Domain randomization: {args.domain_randomization} preset")

    # Create agent (environment returns stacked states)
    print("\nCreating SAC agent...")
    agent = SACAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        lr_actor=args.lr_actor,
        lr_critic=args.lr_critic,
        lr_alpha=args.lr_alpha,
        gamma=args.gamma,
        tau=args.tau,
        auto_entropy_tuning=args.auto_entropy_tuning,
        device=device
    )

    # Create replay buffer (environment already provides stacked observations)
    print("Creating replay buffer...")
    replay_buffer = ReplayBuffer(
        capacity=args.buffer_size,
        state_shape=state_dim,  # Store stacked observations directly
        action_dim=action_dim,
        device=device,
        frame_stack=1  # No additional stacking needed
    )

    # Resume from checkpoint if specified
    start_episode = 0
    total_steps = 0
    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")
        agent.load(args.resume)
        print("Agent loaded successfully!")

    # Initialize logging infrastructure
    print("\nInitializing logging...")
    config = {
        'stationary_patience': STATIONARY_PATIENCE,
        'min_episode_steps': MIN_EPISODE_STEPS,
        'short_episode_penalty': SHORT_EPISODE_PENALTY,
        'max_episode_steps': MAX_EPISODE_STEPS
    }
    training_csv, eval_csv, log_handle = setup_logging(args.log_dir, args, mode='standard', env=env, agent=agent, config=config)

    # Training metrics
    episode_rewards = []
    training_metrics = {
        'actor_loss': [],
        'critic_1_loss': [],
        'critic_2_loss': [],
        'alpha': []
    }
    best_avg_reward = -float('inf')
    learning_started = False

    # Start training
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)
    print(f"Episodes: {args.episodes}")
    print(f"Learning starts: {args.learning_starts} steps")
    print(f"Device: {agent.device}")
    print(f"Plotting: {'enabled' if MATPLOTLIB_AVAILABLE else 'disabled (matplotlib/tkinter not available)'}")
    print("=" * 60 + "\n")

    start_time = time.time()

    for episode in range(start_episode, start_episode + args.episodes):
        state, _ = env.reset()  # Environment returns stacked observation
        episode_reward = 0
        episode_metrics = {
            'actor_loss': [],
            'critic_1_loss': [],
            'critic_2_loss': [],
            'alpha': []
        }
        done = False
        steps = 0

        # Episode loop
        while not done:
            # Select action (state is already stacked by environment)
            action = agent.select_action(state, evaluate=False)

            # Take step in environment
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Store stacked experience in replay buffer
            replay_buffer.push(state, action, reward, next_state, float(done))

            # Increment step counter
            total_steps += 1

            # Train agent (only after learning_starts steps)
            if total_steps >= args.learning_starts and len(replay_buffer) >= args.batch_size:
                metrics = agent.update(replay_buffer, args.batch_size)
                episode_metrics['actor_loss'].append(metrics['actor_loss'])
                episode_metrics['critic_1_loss'].append(metrics['critic_1_loss'])
                episode_metrics['critic_2_loss'].append(metrics['critic_2_loss'])
                episode_metrics['alpha'].append(metrics['alpha'])

            state = next_state
            episode_reward += reward
            steps += 1

        # Record metrics
        episode_rewards.append(episode_reward)
        avg_actor_loss = np.mean(episode_metrics['actor_loss']) if episode_metrics['actor_loss'] else 0.0
        avg_critic_1_loss = np.mean(episode_metrics['critic_1_loss']) if episode_metrics['critic_1_loss'] else 0.0
        avg_critic_2_loss = np.mean(episode_metrics['critic_2_loss']) if episode_metrics['critic_2_loss'] else 0.0
        avg_alpha = np.mean(episode_metrics['alpha']) if episode_metrics['alpha'] else 0.0

        training_metrics['actor_loss'].append(avg_actor_loss)
        training_metrics['critic_1_loss'].append(avg_critic_1_loss)
        training_metrics['critic_2_loss'].append(avg_critic_2_loss)
        training_metrics['alpha'].append(avg_alpha)

        # Calculate stats
        elapsed_time = time.time() - start_time
        avg_reward_100 = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards)
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # Get latest Q-values and log_prob from last training step
        mean_q1 = 0.0
        mean_q2 = 0.0
        mean_log_prob = 0.0
        if total_steps >= args.learning_starts and len(replay_buffer) >= args.batch_size:
            mean_q1 = metrics.get('mean_q1', 0.0)
            mean_q2 = metrics.get('mean_q2', 0.0)
            mean_log_prob = metrics.get('mean_log_prob', 0.0)

        # Write to CSV file (every episode)
        with open(training_csv, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                episode + 1,
                total_steps,
                steps,
                episode_reward,
                avg_actor_loss,
                avg_critic_1_loss,
                avg_critic_2_loss,
                avg_alpha,
                avg_alpha,  # Current alpha value
                mean_q1,
                mean_q2,
                mean_log_prob,
                elapsed_time,
                avg_reward_100,
                timestamp
            ])

        # Check if learning just started
        if not learning_started and total_steps >= args.learning_starts:
            learning_started = True
            msg = f">>> Learning started (reached {args.learning_starts} steps)"
            print(msg)
            log_handle.write(f"[{timestamp}] {msg}\n")

        # Print progress (every 10 episodes)
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1}/{start_episode + args.episodes} | "
                  f"Reward: {episode_reward:7.2f} | "
                  f"Avg(100): {avg_reward_100:7.2f} | "
                  f"Actor Loss: {avg_actor_loss:7.4f} | "
                  f"Critic Loss: {avg_critic_1_loss:7.4f} | "
                  f"Alpha: {avg_alpha:.4f} | "
                  f"Steps: {total_steps:7d} | "
                  f"Buffer: {len(replay_buffer):6d} | "
                  f"Time: {elapsed_time/60:.1f}m")

        # Evaluate periodically
        if (episode + 1) % args.eval_frequency == 0:
            eval_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print("\n" + "-" * 60)
            print(f"Evaluating at episode {episode + 1}...")
            log_handle.write(f"[{eval_timestamp}] {'=' * 50}\n")
            log_handle.write(f"[{eval_timestamp}] Evaluation at episode {episode + 1}\n")

            eval_mean, eval_std, eval_rewards_list = evaluate_agent(agent, env, n_episodes=DEFAULT_INTERMEDIATE_EVAL_EPISODES, log_handle=log_handle, return_details=True)
            print(f"Evaluation reward ({DEFAULT_INTERMEDIATE_EVAL_EPISODES} episodes): {eval_mean:.2f} (±{eval_std:.2f})")
            print("-" * 60 + "\n")

            log_handle.write(f"[{eval_timestamp}] Evaluation complete | Mean: {eval_mean:.2f} | Std: {eval_std:.2f}\n")

            # Write evaluation to CSV
            is_best = eval_mean > best_avg_reward
            with open(eval_csv, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    episode + 1,
                    total_steps,
                    eval_mean,
                    eval_std,
                    str(eval_rewards_list),
                    int(is_best),
                    elapsed_time,
                    eval_timestamp
                ])

            # Save best model
            if is_best:
                prev_best = f"{best_avg_reward:.2f}" if best_avg_reward != -float('inf') else 'N/A'
                best_avg_reward = eval_mean
                best_path = os.path.join(args.checkpoint_dir, 'best_model.pt')
                agent.save(best_path)
                msg = f"*** NEW BEST MODEL *** Reward: {eval_mean:.2f} (prev: {prev_best})"
                print(f"New best model saved! Avg reward: {eval_mean:.2f}\n")
                log_handle.write(f"[{eval_timestamp}] {msg}\n")
                log_handle.write(f"[{eval_timestamp}] Saved: {best_path}\n")

            log_handle.write(f"[{eval_timestamp}] {'=' * 50}\n\n")

        # Save checkpoint periodically
        if (episode + 1) % args.checkpoint_frequency == 0:
            checkpoint_path = os.path.join(args.checkpoint_dir, f'checkpoint_ep{episode + 1}.pt')
            agent.save(checkpoint_path)
            checkpoint_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            log_handle.write(f"[{checkpoint_timestamp}] >>> Checkpoint saved: {checkpoint_path}\n")

            # Save training plot
            plot_path = os.path.join(args.log_dir, 'training_progress.png')
            plot_training_progress(episode_rewards, training_metrics, plot_path)

    # Save final model
    final_path = os.path.join(args.checkpoint_dir, 'final_model.pt')
    agent.save(final_path)
    final_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_handle.write(f"[{final_timestamp}] Final model saved: {final_path}\n\n")

    # Final evaluation
    print("\n" + "=" * 60)
    print("Training complete! Running final evaluation...")
    log_handle.write(f"[{final_timestamp}] {'=' * 50}\n")
    log_handle.write(f"[{final_timestamp}] Final evaluation ({DEFAULT_FINAL_EVAL_EPISODES} episodes)\n")

    final_eval_mean, final_eval_std, final_eval_rewards = evaluate_agent(agent, env, n_episodes=DEFAULT_FINAL_EVAL_EPISODES, log_handle=log_handle, return_details=True)
    print(f"Final evaluation reward ({DEFAULT_FINAL_EVAL_EPISODES} episodes): {final_eval_mean:.2f} (±{final_eval_std:.2f})")
    print("=" * 60)

    log_handle.write(f"[{final_timestamp}] Final evaluation complete | Mean: {final_eval_mean:.2f} | Std: {final_eval_std:.2f}\n")
    log_handle.write(f"[{final_timestamp}] {'=' * 50}\n\n")

    # Save final plot
    plot_path = os.path.join(args.log_dir, 'training_progress.png')
    plot_training_progress(episode_rewards, training_metrics, plot_path)

    # Print summary
    total_time = time.time() - start_time
    print("\n" + "=" * 60)
    print("Training Summary")
    print("=" * 60)
    print(f"Total episodes: {args.episodes}")
    print(f"Total steps: {total_steps}")
    print(f"Total time: {total_time/3600:.2f} hours")
    print(f"Final alpha: {avg_alpha:.4f}")
    print(f"Best evaluation reward: {best_avg_reward:.2f}")
    print(f"Final evaluation reward: {final_eval_mean:.2f}")

    # Performance analysis
    print("\nPerformance Analysis:")
    hours, remainder = divmod(int(total_time), 3600)
    minutes, seconds = divmod(remainder, 60)
    time_per_step = (total_time * 1000) / total_steps if total_steps > 0 else 0
    print(f"  - Total time: {hours}:{minutes:02d}:{seconds:02d} ({total_time:.2f}s)")
    print(f"  - Total steps: {total_steps:,}")
    print(f"  - Time/step: {time_per_step:.2f} ms/step")

    print(f"\nCheckpoints saved in: {args.checkpoint_dir}")
    print(f"Logs saved in: {args.log_dir}")
    print("=" * 60)

    # Write training summary to log file
    log_handle.write("=" * 70 + "\n")
    log_handle.write("Training Session Complete\n")
    log_handle.write("=" * 70 + "\n")
    log_handle.write(f"Timestamp: {final_timestamp}\n")
    log_handle.write(f"Total episodes: {args.episodes}\n")
    log_handle.write(f"Total steps: {total_steps}\n")
    log_handle.write(f"Total time: {total_time/3600:.2f} hours ({total_time/60:.1f} minutes)\n")
    log_handle.write(f"Final alpha: {avg_alpha:.4f}\n")
    log_handle.write(f"Best evaluation reward: {best_avg_reward:.2f}\n")
    log_handle.write(f"Final evaluation reward: {final_eval_mean:.2f} (±{final_eval_std:.2f})\n")
    log_handle.write("\nPerformance Analysis:\n")
    log_handle.write(f"  - Total time: {hours}:{minutes:02d}:{seconds:02d} ({total_time:.2f}s)\n")
    log_handle.write(f"  - Total steps: {total_steps:,}\n")
    log_handle.write(f"  - Time/step: {time_per_step:.2f} ms/step\n")
    log_handle.write(f"\nCheckpoints saved in: {args.checkpoint_dir}\n")
    log_handle.write(f"Logs saved in: {args.log_dir}\n")
    log_handle.write("=" * 70 + "\n")

    # Close log file
    log_handle.close()

    env.close()


if __name__ == "__main__":
    args = parse_args()
    train(args)
