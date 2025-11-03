"""
Training script for DDQN agent on CarRacing-v3.

This script:
1. Creates the CarRacing environment with preprocessing
2. Initializes the DDQN agent
3. Trains the agent with experience replay
4. Saves checkpoints periodically
5. Logs training metrics (rewards, losses, epsilon)
6. Generates training progress plots

Usage:
    # Basic training
    python train.py

    # Custom configuration
    python train.py --episodes 2000 --learning-starts 10000

    # Force CPU-only mode (useful if CPU is faster than MPS)
    python train.py --device cpu

    # Resume from checkpoint
    python train.py --resume checkpoints/final_model.pt --episodes 1000

    # Resume with reset epsilon (more exploration)
    python train.py --resume checkpoints/final_model.pt --reset-epsilon
"""

import argparse
import os
import time
import csv
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

from preprocessing import make_carracing_env
from ddqn_agent import DDQNAgent


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Train DDQN agent on CarRacing-v3')

    # Training parameters
    parser.add_argument('--episodes', type=int, default=2000,
                        help='Number of episodes to train (default: 2000)')
    parser.add_argument('--learning-starts', type=int, default=10000,
                        help='Steps before training starts (default: 10000)')
    parser.add_argument('--eval-frequency', type=int, default=100,
                        help='Evaluate every N episodes (default: 100)')
    parser.add_argument('--checkpoint-frequency', type=int, default=500,
                        help='Save checkpoint every N episodes (default: 500)')

    # Agent hyperparameters
    parser.add_argument('--lr', type=float, default=0.00025,
                        help='Learning rate (default: 0.00025)')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='Discount factor (default: 0.99)')
    parser.add_argument('--buffer-size', type=int, default=100000,
                        help='Replay buffer size (default: 100000)')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Training batch size (default: 32)')
    parser.add_argument('--epsilon-decay', type=int, default=1000000,
                        help='Steps to decay epsilon (default: 1000000)')
    parser.add_argument('--target-update-freq', type=int, default=10000,
                        help='Target network update frequency (default: 10000)')

    # Environment parameters
    parser.add_argument('--steering-bins', type=int, default=3,
                        help='Number of discrete steering values (default: 3)')
    parser.add_argument('--gas-brake-bins', type=int, default=3,
                        help='Number of discrete gas/brake values (default: 3)')
    parser.add_argument('--state-mode', type=str, default='vector', choices=['visual', 'vector', 'snapshot'],
                        help='State representation: visual (images), vector (state), or snapshot (track geometry vector with lookahead) - snapshot is fastest and most informative (default: vector)')

    # Resume training
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--reset-epsilon', action='store_true',
                        help='Reset epsilon when resuming (for more exploration)')

    # Paths
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints (default: checkpoints)')
    parser.add_argument('--log-dir', type=str, default='logs',
                        help='Directory to save logs (default: logs)')

    # Device selection
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'cuda', 'mps'],
                        help='Device to use for training: auto (default), cpu, cuda, or mps')

    return parser.parse_args()


def setup_logging(log_dir, args, env, agent):
    """
    Setup logging infrastructure: CSV files, log file, system info.

    Args:
        log_dir: Directory to save log files
        args: Training arguments
        env: CarRacing environment
        agent: DDQN agent

    Returns:
        Tuple of (training_csv_path, eval_csv_path, log_file_handle)
    """
    # Create CSV for training metrics
    training_csv = os.path.join(log_dir, 'training_metrics.csv')
    with open(training_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'episode', 'total_steps', 'episode_steps', 'reward', 'avg_loss',
            'epsilon', 'learning_rate', 'buffer_size', 'elapsed_time_sec', 'avg_reward_100', 'timestamp'
        ])

    # Create CSV for evaluation metrics
    eval_csv = os.path.join(log_dir, 'evaluation_metrics.csv')
    with open(eval_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'episode', 'total_steps', 'eval_mean_reward', 'eval_std_reward',
            'eval_rewards', 'is_best', 'elapsed_time_sec', 'timestamp'
        ])

    # Create human-readable log file
    log_file = os.path.join(log_dir, 'training.log')
    log_handle = open(log_file, 'w', buffering=1)  # Line buffering for real-time updates

    # Write header to log file
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_handle.write("=" * 70 + "\n")
    log_handle.write("Training Session Started\n")
    log_handle.write("=" * 70 + "\n")
    log_handle.write(f"Timestamp: {timestamp}\n")
    log_handle.write(f"Device: {agent.device}\n")
    log_handle.write(f"State mode: {args.state_mode}\n")
    log_handle.write(f"State shape: {env.observation_space.shape}\n")
    log_handle.write(f"Number of actions: {env.action_space.n}\n")
    log_handle.write(f"Episodes: {args.episodes}\n")
    log_handle.write(f"Learning starts: {args.learning_starts} steps\n")
    log_handle.write(f"Epsilon decay: {args.epsilon_decay} steps (1.0 → 0.01)\n")
    log_handle.write(f"Target update frequency: {args.target_update_freq} steps\n")
    log_handle.write(f"Learning rate: {args.lr} (adaptive with plateau detection)\n")
    log_handle.write(f"Early termination: enabled (patience=200)\n")
    log_handle.write(f"Reward shaping: enabled (penalty -50 for episodes < 150 steps)\n")
    if args.resume:
        log_handle.write(f"Resumed from: {args.resume}\n")
        log_handle.write(f"Reset epsilon: {args.reset_epsilon}\n")
    log_handle.write("=" * 70 + "\n\n")

    # Create system info file
    system_info_path = os.path.join(log_dir, 'system_info.txt')
    with open(system_info_path, 'w') as f:
        f.write("Training Configuration\n")
        f.write("=" * 70 + "\n")
        f.write(f"Date: {timestamp}\n")
        f.write(f"Device: {agent.device}\n")
        f.write(f"State mode: {args.state_mode}\n")
        f.write(f"State shape: {env.observation_space.shape}\n\n")

        f.write("Environment:\n")
        f.write(f"  Name: CarRacing-v3\n")
        f.write(f"  Actions: {env.action_space.n} discrete\n")
        f.write(f"  Steering bins: {args.steering_bins}\n")
        f.write(f"  Gas/brake bins: {args.gas_brake_bins}\n")
        f.write(f"  Early termination: True (patience=100)\n\n")

        f.write("Agent Hyperparameters:\n")
        f.write(f"  Learning rate: {args.lr}\n")
        f.write(f"  Gamma: {args.gamma}\n")
        f.write(f"  Epsilon: 1.0 → 0.01 over {args.epsilon_decay} steps\n")
        f.write(f"  Buffer size: {args.buffer_size}\n")
        f.write(f"  Batch size: {args.batch_size}\n")
        f.write(f"  Target update frequency: {args.target_update_freq} steps\n")
        f.write(f"  Learning starts: {args.learning_starts} steps\n\n")

        f.write("Training Parameters:\n")
        f.write(f"  Episodes: {args.episodes}\n")
        f.write(f"  Eval frequency: {args.eval_frequency} episodes\n")
        f.write(f"  Checkpoint frequency: {args.checkpoint_frequency} episodes\n\n")

        f.write("Resume Settings:\n")
        f.write(f"  Resumed from: {args.resume if args.resume else 'None'}\n")
        f.write(f"  Reset epsilon: {args.reset_epsilon}\n")

    print(f"Logging initialized:")
    print(f"  Training metrics: {training_csv}")
    print(f"  Evaluation metrics: {eval_csv}")
    print(f"  Training log: {log_file}")
    print(f"  System info: {system_info_path}")

    return training_csv, eval_csv, log_handle


def evaluate_agent(agent, env, n_episodes=5, log_handle=None):
    """
    Evaluate agent performance over multiple episodes.

    Args:
        agent: DDQN agent
        env: CarRacing environment
        n_episodes: Number of episodes to evaluate
        log_handle: Optional file handle for logging

    Returns:
        Tuple of (mean_reward, std_reward, all_rewards)
    """
    total_rewards = []
    total_steps = []

    for ep in range(n_episodes):
        state, _ = env.reset()
        episode_reward = 0
        episode_steps = 0
        done = False

        while not done:
            # Use greedy policy (no exploration)
            action = agent.select_action(state, training=False)
            state, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            episode_steps += 1
            done = terminated or truncated

        total_rewards.append(episode_reward)
        total_steps.append(episode_steps)
        msg = f"  Eval episode {ep + 1}/{n_episodes}: reward = {episode_reward:.2f}, steps = {episode_steps}"
        print(msg)
        if log_handle:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            log_handle.write(f"[{timestamp}] {msg}\n")

    mean_reward = np.mean(total_rewards)
    std_reward = np.std(total_rewards)
    mean_steps = np.mean(total_steps)

    # Add summary with average steps
    summary_msg = f"  Average steps per eval episode: {mean_steps:.1f}"
    print(summary_msg)
    if log_handle:
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_handle.write(f"[{timestamp}] {summary_msg}\n")

    return mean_reward, std_reward, total_rewards


def plot_training_progress(episode_rewards, episode_losses, episode_epsilons, save_path):
    """
    Plot training metrics and save to file.

    Args:
        episode_rewards: List of episode rewards
        episode_losses: List of average losses per episode
        episode_epsilons: List of epsilon values per episode
        save_path: Path to save plot
    """
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))

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

    # Plot losses
    axes[1].plot(episode_losses, alpha=0.6, label='Average Loss')
    # Moving average
    if len(episode_losses) >= 100:
        moving_avg = np.convolve(episode_losses, np.ones(100)/100, mode='valid')
        axes[1].plot(range(99, len(episode_losses)), moving_avg, 'r-', linewidth=2, label='Moving Avg (100 ep)')
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Training Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Plot epsilon
    axes[2].plot(episode_epsilons, 'g-', alpha=0.8, label='Epsilon')
    axes[2].set_xlabel('Episode')
    axes[2].set_ylabel('Epsilon')
    axes[2].set_title('Exploration Rate (Epsilon)')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Training plot saved to {save_path}")


def train(args):
    """Main training loop."""
    # Create directories
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    # Create training environment (with early termination for efficiency)
    print("Creating CarRacing-v3 environment...")
    env = make_carracing_env(
        stack_size=4,
        discretize_actions=True,
        steering_bins=args.steering_bins,
        gas_brake_bins=args.gas_brake_bins,
        terminate_stationary=True,  # Speed up training
        stationary_patience=200,  # Increased from 100 to prevent exploiting early termination
        render_mode=None,
        state_mode=args.state_mode,
        reward_shaping=True,  # Penalize very short episodes
        min_episode_steps=150,
        short_episode_penalty=-50.0
    )

    n_actions = env.action_space.n
    state_shape = env.observation_space.shape

    print(f"Environment created:")
    print(f"  State mode: {args.state_mode}")
    print(f"  State shape: {state_shape}")
    print(f"  Number of actions: {n_actions}")
    print(f"  Early termination enabled (patience=200 frames)")
    print(f"  Reward shaping enabled (penalty -50 for episodes < 150 steps)")

    # Create agent
    print("\nCreating DDQN agent...")
    agent = DDQNAgent(
        state_shape=state_shape,
        n_actions=n_actions,
        learning_rate=args.lr,
        gamma=args.gamma,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay_steps=args.epsilon_decay,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        target_update_freq=args.target_update_freq,
        state_mode=args.state_mode,
        device=args.device
    )

    # Resume from checkpoint if specified
    start_episode = 0
    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")
        agent.load(args.resume, load_optimizer=True)
        if args.reset_epsilon:
            agent.epsilon = 1.0
            print("Epsilon reset to 1.0 for exploration")

    # Initialize logging infrastructure
    print("\nInitializing logging...")
    training_csv, eval_csv, log_handle = setup_logging(args.log_dir, args, env, agent)

    # Training metrics
    episode_rewards = []
    episode_losses = []
    episode_epsilons = []
    best_avg_reward = -float('inf')
    learning_started = False  # Track when training actually starts

    # Start training
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)
    print(f"Episodes: {args.episodes}")
    print(f"Learning starts: {args.learning_starts} steps")
    print(f"Epsilon decay: {args.epsilon_decay} steps")
    print(f"Device: {agent.device}")
    print("=" * 60 + "\n")

    start_time = time.time()

    for episode in range(start_episode, start_episode + args.episodes):
        state, _ = env.reset()
        episode_reward = 0
        episode_loss = []
        done = False
        steps = 0

        # Episode loop
        while not done:
            # Select action
            action = agent.select_action(state, training=True)

            # Take step in environment
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Store experience
            agent.store_experience(state, action, reward, next_state, done)

            # Increment step counter (needed for epsilon decay and learning_starts)
            agent.steps_done += 1

            # Train agent (only after learning_starts steps)
            if agent.steps_done >= args.learning_starts:
                loss = agent.train_step()
                if loss is not None:
                    episode_loss.append(loss)

            # Update epsilon based on steps
            agent.update_epsilon()

            state = next_state
            episode_reward += reward
            steps += 1

        # Record metrics
        episode_rewards.append(episode_reward)
        avg_loss = np.mean(episode_loss) if episode_loss else 0.0
        episode_losses.append(avg_loss)
        episode_epsilons.append(agent.epsilon)

        # Calculate stats
        elapsed_time = time.time() - start_time
        avg_reward_100 = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards)
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # Write to CSV file (every episode)
        with open(training_csv, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                episode + 1,
                agent.steps_done,
                steps,
                episode_reward,
                avg_loss,
                agent.epsilon,
                agent.get_learning_rate(),
                len(agent.replay_buffer),
                elapsed_time,
                avg_reward_100,
                timestamp
            ])

        # Write to log file (every episode - critical for monitoring episode length)
        if (episode + 1) % 10 == 0:
            log_msg = (f"Episode {episode + 1} | Steps: {steps} (total: {agent.steps_done}) | "
                      f"Reward: {episode_reward:.2f} | Loss: {avg_loss:.4f} | "
                      f"Epsilon: {agent.epsilon:.4f} | Avg(100): {avg_reward_100:.2f}")
            log_handle.write(f"[{timestamp}] {log_msg}\n")

        # Also log warning for very short episodes (potential exploit)
        if steps < 150 and (episode + 1) % 5 == 0:
            warning_msg = (f"Episode {episode + 1} | SHORT EPISODE: {steps} steps | "
                          f"Reward: {episode_reward:.2f} (shaped with -50 penalty)")
            log_handle.write(f"[{timestamp}] ⚠️  {warning_msg}\n")

        # Check if learning just started
        if not learning_started and agent.steps_done >= args.learning_starts:
            learning_started = True
            msg = f">>> Learning started (reached {args.learning_starts} steps)"
            print(msg)
            log_handle.write(f"[{timestamp}] {msg}\n")

        # Print progress (every 10 episodes)
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1}/{start_episode + args.episodes} | "
                  f"Reward: {episode_reward:7.2f} | "
                  f"Avg(100): {avg_reward_100:7.2f} | "
                  f"Loss: {avg_loss:7.4f} | "
                  f"Epsilon: {agent.epsilon:.4f} | "
                  f"Steps: {agent.steps_done:7d} | "
                  f"Buffer: {len(agent.replay_buffer):6d} | "
                  f"Time: {elapsed_time/60:.1f}m")

        # Evaluate periodically
        if (episode + 1) % args.eval_frequency == 0:
            eval_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print("\n" + "-" * 60)
            print(f"Evaluating at episode {episode + 1}...")
            log_handle.write(f"[{eval_timestamp}] {'=' * 50}\n")
            log_handle.write(f"[{eval_timestamp}] Evaluation at episode {episode + 1}\n")

            eval_mean, eval_std, eval_rewards_list = evaluate_agent(agent, env, n_episodes=5, log_handle=log_handle)
            print(f"Evaluation reward (5 episodes): {eval_mean:.2f} (±{eval_std:.2f})")
            print("-" * 60 + "\n")

            log_handle.write(f"[{eval_timestamp}] Evaluation complete | Mean: {eval_mean:.2f} | Std: {eval_std:.2f}\n")

            # Write evaluation to CSV
            is_best = eval_mean > best_avg_reward
            with open(eval_csv, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    episode + 1,
                    agent.steps_done,
                    eval_mean,
                    eval_std,
                    str(eval_rewards_list),  # Store as string representation of list
                    int(is_best),
                    elapsed_time,
                    eval_timestamp
                ])

            # Update learning rate based on evaluation performance
            old_lr = agent.get_learning_rate()
            agent.update_learning_rate(eval_mean)
            new_lr = agent.get_learning_rate()
            if new_lr != old_lr:
                log_handle.write(f"[{eval_timestamp}] Learning rate reduced: {old_lr:.6f} → {new_lr:.6f}\n")

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
            plot_training_progress(episode_rewards, episode_losses, episode_epsilons, plot_path)

    # Save final model
    final_path = os.path.join(args.checkpoint_dir, 'final_model.pt')
    agent.save(final_path)
    final_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_handle.write(f"[{final_timestamp}] Final model saved: {final_path}\n\n")

    # Final evaluation
    print("\n" + "=" * 60)
    print("Training complete! Running final evaluation...")
    log_handle.write(f"[{final_timestamp}] {'=' * 50}\n")
    log_handle.write(f"[{final_timestamp}] Final evaluation (10 episodes)\n")

    final_eval_mean, final_eval_std, final_eval_rewards = evaluate_agent(agent, env, n_episodes=10, log_handle=log_handle)
    print(f"Final evaluation reward (10 episodes): {final_eval_mean:.2f} (±{final_eval_std:.2f})")
    print("=" * 60)

    log_handle.write(f"[{final_timestamp}] Final evaluation complete | Mean: {final_eval_mean:.2f} | Std: {final_eval_std:.2f}\n")
    log_handle.write(f"[{final_timestamp}] {'=' * 50}\n\n")

    # Save final plot
    plot_path = os.path.join(args.log_dir, 'training_progress.png')
    plot_training_progress(episode_rewards, episode_losses, episode_epsilons, plot_path)

    # Print summary
    total_time = time.time() - start_time
    print("\n" + "=" * 60)
    print("Training Summary")
    print("=" * 60)
    print(f"Total episodes: {args.episodes}")
    print(f"Total steps: {agent.steps_done}")
    print(f"Total time: {total_time/3600:.2f} hours")
    print(f"Final epsilon: {agent.epsilon:.4f}")
    print(f"Best evaluation reward: {best_avg_reward:.2f}")
    print(f"Final evaluation reward: {final_eval_mean:.2f}")
    print(f"\nCheckpoints saved in: {args.checkpoint_dir}")
    print(f"Logs saved in: {args.log_dir}")
    print("=" * 60)

    # Write training summary to log file
    log_handle.write("=" * 70 + "\n")
    log_handle.write("Training Session Complete\n")
    log_handle.write("=" * 70 + "\n")
    log_handle.write(f"Timestamp: {final_timestamp}\n")
    log_handle.write(f"Total episodes: {args.episodes}\n")
    log_handle.write(f"Total steps: {agent.steps_done}\n")
    log_handle.write(f"Total time: {total_time/3600:.2f} hours ({total_time/60:.1f} minutes)\n")
    log_handle.write(f"Final epsilon: {agent.epsilon:.4f}\n")
    log_handle.write(f"Best evaluation reward: {best_avg_reward:.2f}\n")
    log_handle.write(f"Final evaluation reward: {final_eval_mean:.2f} (±{final_eval_std:.2f})\n")
    log_handle.write(f"Checkpoints saved in: {args.checkpoint_dir}\n")
    log_handle.write(f"Logs saved in: {args.log_dir}\n")
    log_handle.write("=" * 70 + "\n")

    # Close log file
    log_handle.close()

    env.close()


if __name__ == "__main__":
    args = parse_args()
    train(args)
