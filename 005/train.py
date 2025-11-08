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
    # Basic training
    python train.py

    # Custom configuration
    python train.py --episodes 2000 --learning-starts 5000

    # Force CPU-only mode
    python train.py --device cpu

    # Resume from checkpoint
    python train.py --resume checkpoints/best_model.pt --episodes 1000
"""

import argparse
import os
import time
import csv
from datetime import datetime
import numpy as np
import torch

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
from sac_agent import SACAgent, ReplayBuffer
from env.car_racing import (
    NUM_CHECKPOINTS, CHECKPOINT_REWARD, FORWARD_VEL_REWARD,
    STEP_PENALTY, OFFTRACK_PENALTY, OFFTRACK_THRESHOLD
)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Train SAC agent on CarRacing-v3')

    # Training parameters
    parser.add_argument('--episodes', type=int, default=2000,
                        help='Number of episodes to train (default: 2000)')
    parser.add_argument('--learning-starts', type=int, default=5000,
                        help='Steps before training starts (default: 5000)')
    parser.add_argument('--eval-frequency', type=int, default=100,
                        help='Evaluate every N episodes (default: 100)')
    parser.add_argument('--checkpoint-frequency', type=int, default=100,
                        help='Save checkpoint every N episodes (default: 100)')

    # Agent hyperparameters
    parser.add_argument('--lr-actor', type=float, default=3e-4,
                        help='Actor learning rate (default: 3e-4)')
    parser.add_argument('--lr-critic', type=float, default=3e-4,
                        help='Critic learning rate (default: 3e-4)')
    parser.add_argument('--lr-alpha', type=float, default=3e-4,
                        help='Alpha learning rate (default: 3e-4)')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='Discount factor (default: 0.99)')
    parser.add_argument('--tau', type=float, default=0.005,
                        help='Soft update coefficient (default: 0.005)')
    parser.add_argument('--buffer-size', type=int, default=100000,
                        help='Replay buffer size (default: 100000)')
    parser.add_argument('--batch-size', type=int, default=256,
                        help='Training batch size (default: 256)')
    parser.add_argument('--auto-entropy-tuning', action='store_true', default=True,
                        help='Use automatic entropy tuning (default: True)')

    # Environment parameters
    parser.add_argument('--state-mode', type=str, default='vector', choices=['visual', 'vector'],
                        help='State representation: visual (images) or vector (track geometry with lookahead) - vector is fastest (default: vector)')

    # Resume training
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')

    # Paths
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints (default: checkpoints)')
    parser.add_argument('--log-dir', type=str, default='logs',
                        help='Directory to save logs (default: logs)')

    # Device selection
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'cuda', 'mps'],
                        help='Device to use for training: auto (default), cpu, cuda, or mps')

    # Debugging
    parser.add_argument('--verbose', action='store_true', default=False,
                        help='Enable verbose mode from environment for debugging (default: False)')

    return parser.parse_args()


def get_device(device_arg):
    """Determine which device to use for training."""
    if device_arg == 'auto':
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif torch.backends.mps.is_available():
            return torch.device('mps')
        else:
            return torch.device('cpu')
    else:
        return torch.device(device_arg)


def setup_logging(log_dir, args, env, agent, config):
    """
    Setup logging infrastructure: CSV files, log file, system info.

    Args:
        log_dir: Directory to save log files
        args: Training arguments
        env: CarRacing environment
        agent: SAC agent
        config: Dict with environment configuration values

    Returns:
        Tuple of (training_csv_path, eval_csv_path, log_file_handle)
    """
    # Create CSV for training metrics
    training_csv = os.path.join(log_dir, 'training_metrics.csv')
    with open(training_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'episode', 'total_steps', 'episode_steps', 'reward',
            'actor_loss', 'critic_1_loss', 'critic_2_loss', 'alpha_loss',
            'alpha', 'mean_q1', 'mean_q2', 'mean_log_prob',
            'elapsed_time_sec', 'avg_reward_100', 'timestamp'
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
    log_handle.write("SAC Training Session Started\n")
    log_handle.write("=" * 70 + "\n")
    log_handle.write(f"Timestamp: {timestamp}\n")
    log_handle.write(f"Device: {agent.device}\n")
    log_handle.write(f"State mode: {args.state_mode}\n")
    log_handle.write(f"State shape: {env.observation_space.shape}\n")
    log_handle.write(f"Action space: Continuous (3D)\n")
    log_handle.write(f"Episodes: {args.episodes}\n")
    log_handle.write(f"Learning starts: {args.learning_starts} steps\n")
    log_handle.write(f"Auto entropy tuning: {args.auto_entropy_tuning}\n")
    log_handle.write(f"Early termination: enabled (patience={config['stationary_patience']})\n")
    log_handle.write(f"Reward shaping: enabled (penalty {config['short_episode_penalty']} for episodes < {config['min_episode_steps']} steps)\n")
    if args.resume:
        log_handle.write(f"Resumed from: {args.resume}\n")
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
        f.write(f"  Actions: Continuous [steering, gas, brake]\n")
        f.write(f"  Early termination: True (patience={config['stationary_patience']})\n")
        f.write(f"  Reward shaping: True (penalty {config['short_episode_penalty']} for < {config['min_episode_steps']} steps)\n\n")

        f.write("Reward Structure (from env/car_racing.py):\n")
        f.write(f"  Checkpoints: {NUM_CHECKPOINTS} x {CHECKPOINT_REWARD} points\n")
        f.write(f"  Forward velocity: {FORWARD_VEL_REWARD} per m/s per frame\n")
        f.write(f"  Step penalty: {STEP_PENALTY} per frame\n")
        f.write(f"  Off-track penalty: {OFFTRACK_PENALTY} per wheel (>{OFFTRACK_THRESHOLD} wheels)\n\n")

        f.write("Agent Hyperparameters:\n")
        f.write(f"  Actor learning rate: {args.lr_actor}\n")
        f.write(f"  Critic learning rate: {args.lr_critic}\n")
        f.write(f"  Alpha learning rate: {args.lr_alpha}\n")
        f.write(f"  Gamma: {args.gamma}\n")
        f.write(f"  Tau: {args.tau}\n")
        f.write(f"  Buffer size: {args.buffer_size}\n")
        f.write(f"  Batch size: {args.batch_size}\n")
        f.write(f"  Auto entropy tuning: {args.auto_entropy_tuning}\n")
        f.write(f"  Learning starts: {args.learning_starts} steps\n\n")

        f.write("Training Parameters:\n")
        f.write(f"  Episodes: {args.episodes}\n")
        f.write(f"  Eval frequency: {args.eval_frequency} episodes\n")
        f.write(f"  Checkpoint frequency: {args.checkpoint_frequency} episodes\n\n")

        f.write("Resume Settings:\n")
        f.write(f"  Resumed from: {args.resume if args.resume else 'None'}\n")

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
        agent: SAC agent
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
            # Use deterministic policy (mean action)
            action = agent.select_action(state, evaluate=True)
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


def plot_training_progress(episode_rewards, metrics, save_path):
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


def train(args):
    """Main training loop."""
    # Environment configuration (single source of truth)
    STACK_SIZE = 4
    TERMINATE_STATIONARY = True
    STATIONARY_PATIENCE = 50
    REWARD_SHAPING = True
    MIN_EPISODE_STEPS = 100
    SHORT_EPISODE_PENALTY = -50.0

    # Create directories
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    # Determine device
    device = get_device(args.device)
    print(f"Using device: {device}")

    # Create training environment
    print("Creating CarRacing-v3 environment...")
    env = make_carracing_env(
        stack_size=STACK_SIZE,
        terminate_stationary=TERMINATE_STATIONARY,
        stationary_patience=STATIONARY_PATIENCE,
        render_mode=None,
        state_mode=args.state_mode,
        reward_shaping=REWARD_SHAPING,
        min_episode_steps=MIN_EPISODE_STEPS,
        short_episode_penalty=SHORT_EPISODE_PENALTY,
        verbose=args.verbose
    )

    action_dim = env.action_space.shape[0]
    state_shape = env.observation_space.shape

    print(f"Environment created:")
    print(f"  State mode: {args.state_mode}")
    print(f"  State shape: {state_shape}")
    print(f"  Action space: Continuous (3D)")
    print(f"  Early termination enabled (patience={STATIONARY_PATIENCE} frames)")
    print(f"  Reward shaping enabled (penalty {SHORT_EPISODE_PENALTY} for episodes < {MIN_EPISODE_STEPS} steps)")

    # Create agent
    print("\nCreating SAC agent...")
    agent = SACAgent(
        state_shape=state_shape,
        action_dim=action_dim,
        state_mode=args.state_mode,
        lr_actor=args.lr_actor,
        lr_critic=args.lr_critic,
        lr_alpha=args.lr_alpha,
        gamma=args.gamma,
        tau=args.tau,
        auto_entropy_tuning=args.auto_entropy_tuning,
        device=device
    )

    # Enable verbose mode if requested
    if args.verbose:
        agent.verbose = True
        print("  Verbose mode enabled for agent (timing will be printed every 100 updates)")

    # Create replay buffer
    print("Creating replay buffer...")
    replay_buffer = ReplayBuffer(
        capacity=args.buffer_size,
        state_shape=state_shape,
        action_dim=action_dim,
        device=device
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
        'short_episode_penalty': SHORT_EPISODE_PENALTY
    }
    training_csv, eval_csv, log_handle = setup_logging(args.log_dir, args, env, agent, config)

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
        state, _ = env.reset()
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
            # Select action (stochastic during training)
            action = agent.select_action(state, evaluate=False)

            # Take step in environment
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Store experience
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
    log_handle.write(f"[{final_timestamp}] Final evaluation (10 episodes)\n")

    final_eval_mean, final_eval_std, final_eval_rewards = evaluate_agent(agent, env, n_episodes=10, log_handle=log_handle)
    print(f"Final evaluation reward (10 episodes): {final_eval_mean:.2f} (±{final_eval_std:.2f})")
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
    log_handle.write(f"Checkpoints saved in: {args.checkpoint_dir}\n")
    log_handle.write(f"Logs saved in: {args.log_dir}\n")
    log_handle.write("=" * 70 + "\n")

    # Close log file
    log_handle.close()

    env.close()


if __name__ == "__main__":
    args = parse_args()
    train(args)
