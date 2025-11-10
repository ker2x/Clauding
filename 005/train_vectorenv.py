"""
VectorEnv training script with synchronized seeds for SAC agent on CarRacing-v3.

This script uses Gymnasium VectorEnv to run N environments in parallel with synchronized seeds:
1. Creates N separate CarRacing environments (each single-car)
2. Generates a random seed per episode
3. All N environments use the same seed → same track → fair competition
4. True parallel execution (N× CPU usage)
5. Collects experiences from all N environments simultaneously

Benefits compared to train_multicar.py:
- TRUE PARALLEL EXECUTION (4× CPU usage, potentially faster wall-clock time)
- Same fair competition (synchronized seeds = same track)
- Natural selection (best car identified each episode)

Tradeoffs:
- Higher memory usage (N× environments = N× memory, ~200 MB total for 4 envs)
- More complex setup (VectorEnv, separate processes)

Usage:
    # Train with 4 parallel environments (4× parallelism)
    python train_vectorenv.py --num-envs 4

    # Train with 8 environments for maximum parallelism
    python train_vectorenv.py --num-envs 8 --episodes 1000

    # Resume training
    python train_vectorenv.py --num-envs 4 --resume checkpoints/best_model.pt
"""

import argparse
import os
import time
import csv
from datetime import datetime
import numpy as np
import torch
import multiprocessing
from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv

# Try to import matplotlib with non-GUI backend
MATPLOTLIB_AVAILABLE = False
try:
    import matplotlib
    try:
        matplotlib.use('Agg', force=False)
    except:
        pass
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except (ImportError, RuntimeError) as e:
    print(f"Warning: matplotlib not available ({e})")

from preprocessing import make_carracing_env
from sac_agent import SACAgent, ReplayBuffer
from env.car_racing import (
    NUM_CHECKPOINTS, CHECKPOINT_REWARD, LAP_COMPLETION_REWARD, FORWARD_VEL_REWARD,
    STEP_PENALTY, OFFTRACK_PENALTY, OFFTRACK_THRESHOLD
)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Train SAC agent on CarRacing-v3 using VectorEnv with synchronized seeds')

    # VectorEnv parameters
    parser.add_argument('--num-envs', type=int, default=8,
                        help='Number of parallel environments (default: 8)')
    parser.add_argument('--vec-env-type', type=str, default='async', choices=['async', 'sync'],
                        help='VectorEnv type: async (default, true parallel) or sync (sequential, less overhead)')
    parser.add_argument('--steps-per-update', type=int, default=20,
                        help='Collect N steps before each training update (default: 20, higher reduces overhead)')

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

    # Resume training
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')

    # Paths
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints_vectorenv',
                        help='Directory to save checkpoints (default: checkpoints_vectorenv)')
    parser.add_argument('--log-dir', type=str, default='logs_vectorenv',
                        help='Directory to save logs (default: logs_vectorenv)')

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


def configure_cpu_threading(device):
    """Configure PyTorch CPU threading for optimal performance."""
    if device.type == 'cpu':
        num_cores = multiprocessing.cpu_count()
        num_threads = max(1, num_cores // 2)

        torch.set_num_threads(num_threads)
        torch.set_num_interop_threads(1)

        print(f"CPU Threading Configuration:")
        print(f"  Physical CPU cores: {num_cores}")
        print(f"  PyTorch threads: {num_threads}")
        print(f"  Interop threads: 1")


def make_env_fn(verbose=False):
    """
    Factory function to create environment instances for VectorEnv.
    Returns a function that creates a single-car CarRacing environment.
    """
    def _make_env():
        return make_carracing_env(
            state_mode='vector',
            num_cars=1,  # Single car per environment
            render_mode=None,
            verbose=verbose,
            terminate_stationary=True,
            stationary_patience=50,
            reward_shaping=True,
            min_episode_steps=100,
            short_episode_penalty=-50.0,
            max_episode_steps=1500
        )
    return _make_env


def evaluate_agent(agent, n_episodes=5, log_handle=None):
    """
    Evaluate agent performance over multiple episodes (single-car evaluation).

    Args:
        agent: SAC agent
        n_episodes: Number of episodes to evaluate
        log_handle: Optional file handle for logging

    Returns:
        Tuple of (mean_reward, std_reward, all_rewards)
    """
    # Create single-car environment for evaluation
    eval_env = make_carracing_env(
        state_mode='vector',
        num_cars=1,
        render_mode=None,
        verbose=False
    )

    total_rewards = []
    total_steps = []

    for ep in range(n_episodes):
        state, _ = eval_env.reset()
        episode_reward = 0
        episode_steps = 0
        done = False

        while not done:
            # Use deterministic policy (mean action)
            action = agent.select_action(state, evaluate=True)
            state, reward, terminated, truncated, _ = eval_env.step(action)
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

    eval_env.close()

    mean_reward = np.mean(total_rewards)
    std_reward = np.std(total_rewards)
    mean_steps = np.mean(total_steps)

    # Add summary with average steps
    summary_msg = f"  Average: {mean_reward:.2f} ± {std_reward:.2f} over {n_episodes} episodes ({mean_steps:.1f} steps avg)"
    print(summary_msg)
    if log_handle:
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_handle.write(f"[{timestamp}] {summary_msg}\n")

    return mean_reward, std_reward, total_rewards


def setup_logging(log_dir, args, agent, config, num_envs):
    """Setup logging infrastructure."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = f'training_vectorenv_{num_envs}envs_{timestamp}.csv'
    eval_filename = f'eval_vectorenv_{num_envs}envs_{timestamp}.csv'
    config_filename = f'config_vectorenv_{num_envs}envs_{timestamp}.txt'

    training_csv = os.path.join(log_dir, log_filename)
    eval_csv = os.path.join(log_dir, eval_filename)
    config_txt = os.path.join(log_dir, config_filename)

    # Training CSV header
    with open(training_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'episode', 'total_steps', 'episode_steps', 'best_env_idx', 'best_reward',
            'avg_reward_all_envs', 'min_reward', 'max_reward',
            'actor_loss', 'critic_1_loss', 'critic_2_loss', 'alpha',
            'mean_q1', 'mean_q2', 'mean_log_prob',
            'avg_reward_100', 'elapsed_time', 'timestamp'
        ])

    # Eval CSV header
    with open(eval_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'episode', 'total_steps', 'eval_mean_reward', 'eval_std_reward', 'timestamp'
        ])

    # Save configuration
    with open(config_txt, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write(f"VectorEnv Training Configuration (Synchronized Seeds)\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Timestamp: {timestamp}\n\n")

        f.write("VectorEnv Settings:\n")
        f.write(f"  Number of environments: {num_envs}\n")
        f.write(f"  Parallelism: TRUE (AsyncVectorEnv)\n")
        f.write(f"  Seed synchronization: ENABLED (same track for all envs)\n")
        f.write(f"  Data collection rate: {num_envs}× parallel experiences\n\n")

        f.write("Environment:\n")
        f.write(f"  State mode: vector (67D)\n")
        f.write(f"  Observation space per env: (67,)\n")
        f.write(f"  Max episode steps: {config['max_episode_steps']}\n")
        f.write(f"  Stationary patience: {config['stationary_patience']}\n")
        f.write(f"  Min episode steps: {config['min_episode_steps']}\n")
        f.write(f"  Short episode penalty: {config['short_episode_penalty']}\n\n")

        f.write("Agent:\n")
        for key, value in vars(args).items():
            if key not in ['resume', 'checkpoint_dir', 'log_dir', 'verbose']:
                f.write(f"  {key}: {value}\n")
        f.write(f"\nDevice: {agent.device}\n")

    print(f"Logging initialized:")
    print(f"  Training log: {training_csv}")
    print(f"  Eval log: {eval_csv}")
    print(f"  Config: {config_txt}")

    return training_csv, eval_csv, open(config_txt, 'a')


def train(args):
    """Main VectorEnv training loop with synchronized seeds."""
    # Environment configuration
    STATIONARY_PATIENCE = 50
    MIN_EPISODE_STEPS = 100
    SHORT_EPISODE_PENALTY = -50.0
    MAX_EPISODE_STEPS = 1500

    # Validate num_envs
    if args.num_envs < 1:
        raise ValueError(f"num_envs must be >= 1, got {args.num_envs}")

    # Create directories
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    # Determine device
    device = get_device(args.device)
    print(f"Using device: {device}")

    # Configure CPU threading
    configure_cpu_threading(device)

    # Create VectorEnv with specified type
    print(f"\nCreating VectorEnv with {args.num_envs} parallel environments...")
    print(f"  Using {'AsyncVectorEnv (true parallel)' if args.vec_env_type == 'async' else 'SyncVectorEnv (sequential, less overhead)'}")
    print("  Each environment will receive the same seed per episode")

    VecEnvClass = AsyncVectorEnv if args.vec_env_type == 'async' else SyncVectorEnv
    vec_env = VecEnvClass([
        make_env_fn(verbose=args.verbose)
        for _ in range(args.num_envs)
    ])

    action_dim = vec_env.single_action_space.shape[0]
    state_shape = vec_env.single_observation_space.shape

    print(f"VectorEnv created:")
    print(f"  Number of environments: {args.num_envs}")
    print(f"  Type: {args.vec_env_type}")
    print(f"  Parallelism: {'TRUE (separate processes)' if args.vec_env_type == 'async' else 'FALSE (sequential)'}")
    print(f"  Steps per update: {args.steps_per_update}")
    print(f"  Data collection rate: {args.num_envs}× parallel experiences")
    print(f"  Observation space per env: {state_shape}")
    print(f"  Action space: Continuous (2D)")
    print(f"  Max episode steps: {MAX_EPISODE_STEPS}")
    print(f"  Memory usage: ~{args.num_envs * 50} MB (estimated)")

    # Create agent (same agent for all envs - shared policy)
    print("\nCreating SAC agent (shared policy for all environments)...")
    agent = SACAgent(
        state_shape=state_shape,
        action_dim=action_dim,
        state_mode='vector',
        lr_actor=args.lr_actor,
        lr_critic=args.lr_critic,
        lr_alpha=args.lr_alpha,
        gamma=args.gamma,
        tau=args.tau,
        auto_entropy_tuning=args.auto_entropy_tuning,
        device=device
    )

    if args.verbose:
        agent.verbose = True

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

    # Initialize logging
    print("\nInitializing logging...")
    config = {
        'stationary_patience': STATIONARY_PATIENCE,
        'min_episode_steps': MIN_EPISODE_STEPS,
        'short_episode_penalty': SHORT_EPISODE_PENALTY,
        'max_episode_steps': MAX_EPISODE_STEPS
    }
    training_csv, eval_csv, log_handle = setup_logging(
        args.log_dir, args, agent, config, args.num_envs
    )

    # Training metrics
    episode_best_rewards = []  # Best environment reward per episode
    episode_avg_rewards = []   # Average reward across all environments per episode
    training_metrics = {
        'actor_loss': [],
        'critic_1_loss': [],
        'critic_2_loss': [],
        'alpha': []
    }
    best_avg_reward = -float('inf')

    # Start training
    print("\n" + "=" * 60)
    print("Starting VectorEnv Training (Synchronized Seeds)")
    print("=" * 60)
    print(f"Number of environments: {args.num_envs}")
    print(f"Episodes: {args.episodes}")
    print(f"Learning starts: {args.learning_starts} steps")
    print(f"Data collection: {args.num_envs}× parallel")
    print(f"Device: {agent.device}")
    print(f"Plotting: {'enabled' if MATPLOTLIB_AVAILABLE else 'disabled'}")
    print("=" * 60 + "\n")

    start_time = time.time()

    for episode in range(start_episode, start_episode + args.episodes):
        # Generate random seed for this episode
        # Use base seed (1000) + episode number for reproducibility
        episode_seed = 1000 + episode

        # Reset all environments with the SAME seed (synchronized track generation)
        obs, _ = vec_env.reset(seed=[episode_seed] * args.num_envs)

        # Per-environment episode tracking
        env_episode_rewards = np.zeros(args.num_envs)
        episode_metrics = {
            'actor_loss': [],
            'critic_1_loss': [],
            'critic_2_loss': [],
            'alpha': []
        }
        done_flags = np.zeros(args.num_envs, dtype=bool)
        steps = 0

        # Episode loop (continue until all environments are done)
        while not all(done_flags):
            # Select actions for all environments (shared policy)
            actions = np.array([
                agent.select_action(obs[i], evaluate=False)
                for i in range(args.num_envs)
            ])

            # Take step - returns vectorized outputs
            next_obs, rewards, terminated, truncated, infos = vec_env.step(actions)

            # Store experiences from all environments
            for i in range(args.num_envs):
                # Only store if environment hasn't terminated yet
                if not done_flags[i]:
                    replay_buffer.push(
                        obs[i],
                        actions[i],
                        float(rewards[i]),  # Convert to Python float
                        next_obs[i],
                        float(terminated[i] or truncated[i])
                    )
                    env_episode_rewards[i] += rewards[i]

                    # Update done flag
                    if terminated[i] or truncated[i]:
                        done_flags[i] = True

            # Increment step counter (count per environment step)
            total_steps += 1

            # Train agent (only after learning_starts steps)
            # Train every N steps to reduce overhead
            if (total_steps >= args.learning_starts and
                len(replay_buffer) >= args.batch_size and
                total_steps % args.steps_per_update == 0):
                metrics = agent.update(replay_buffer, args.batch_size)
                episode_metrics['actor_loss'].append(metrics['actor_loss'])
                episode_metrics['critic_1_loss'].append(metrics['critic_1_loss'])
                episode_metrics['critic_2_loss'].append(metrics['critic_2_loss'])
                episode_metrics['alpha'].append(metrics['alpha'])

            obs = next_obs
            steps += 1

        # Select best environment for this episode
        best_env_idx = np.argmax(env_episode_rewards)
        best_reward = env_episode_rewards[best_env_idx]
        avg_reward_all_envs = np.mean(env_episode_rewards)
        min_reward = np.min(env_episode_rewards)
        max_reward = np.max(env_episode_rewards)

        # Record metrics
        episode_best_rewards.append(best_reward)
        episode_avg_rewards.append(avg_reward_all_envs)

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
        avg_best_reward_100 = np.mean(episode_best_rewards[-100:]) if len(episode_best_rewards) >= 100 else np.mean(episode_best_rewards)
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # Get latest Q-values and log_prob
        mean_q1 = metrics.get('mean_q1', 0.0) if total_steps >= args.learning_starts else 0.0
        mean_q2 = metrics.get('mean_q2', 0.0) if total_steps >= args.learning_starts else 0.0
        mean_log_prob = metrics.get('mean_log_prob', 0.0) if total_steps >= args.learning_starts else 0.0

        # Write to CSV
        with open(training_csv, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                episode + 1,
                total_steps,
                steps,
                best_env_idx,
                f"{best_reward:.2f}",
                f"{avg_reward_all_envs:.2f}",
                f"{min_reward:.2f}",
                f"{max_reward:.2f}",
                f"{avg_actor_loss:.6f}",
                f"{avg_critic_1_loss:.6f}",
                f"{avg_critic_2_loss:.6f}",
                f"{avg_alpha:.6f}",
                f"{mean_q1:.2f}",
                f"{mean_q2:.2f}",
                f"{mean_log_prob:.4f}",
                f"{avg_best_reward_100:.2f}",
                f"{elapsed_time:.0f}",
                timestamp
            ])

        # Print detailed progress
        if (episode + 1) % 10 == 0:
            hours = elapsed_time / 3600
            print(f"\nEpisode {episode + 1}/{start_episode + args.episodes} | Time: {hours:.2f}h | Seed: {episode_seed}")
            print(f"  Envs: [{', '.join([f'{r:6.1f}' for r in env_episode_rewards])}]")
            print(f"  Best: Env {best_env_idx} ({best_reward:7.2f}) | Avg: {avg_reward_all_envs:7.2f} | Range: [{min_reward:.1f}, {max_reward:.1f}]")
            print(f"  Reward avg(100): {avg_best_reward_100:7.2f}")
            print(f"  Steps: {steps:4d} | Total steps: {total_steps:7d} | Buffer: {len(replay_buffer):6d}")
            if total_steps >= args.learning_starts:
                print(f"  Loss: Actor={avg_actor_loss:.4f} Critic1={avg_critic_1_loss:.4f} Critic2={avg_critic_2_loss:.4f}")
                print(f"  Alpha: {avg_alpha:.4f} | Q1: {mean_q1:.2f} | Q2: {mean_q2:.2f} | LogProb: {mean_log_prob:.4f}")
            else:
                warmup_remaining = args.learning_starts - total_steps
                print(f"  Warmup: {warmup_remaining} steps remaining before training starts")

        # Evaluation
        if (episode + 1) % args.eval_frequency == 0 and total_steps >= args.learning_starts:
            print(f"\n{'='*60}")
            print(f"Evaluation at episode {episode + 1}")
            print(f"{'='*60}")
            eval_mean, eval_std, eval_rewards = evaluate_agent(agent, n_episodes=5, log_handle=log_handle)

            # Write to eval CSV
            with open(eval_csv, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    episode + 1,
                    total_steps,
                    f"{eval_mean:.2f}",
                    f"{eval_std:.2f}",
                    datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                ])

            print(f"{'='*60}\n")

        # Save checkpoint
        if (episode + 1) % args.checkpoint_frequency == 0:
            checkpoint_path = os.path.join(
                args.checkpoint_dir,
                f'checkpoint_{args.num_envs}envs_ep{episode + 1}.pt'
            )
            agent.save(checkpoint_path)
            print(f"✓ Checkpoint saved: {checkpoint_path}")

        # Save best model
        if avg_best_reward_100 > best_avg_reward:
            best_avg_reward = avg_best_reward_100
            best_path = os.path.join(args.checkpoint_dir, f'best_model_{args.num_envs}envs.pt')
            agent.save(best_path)
            print(f"✓ New best model saved: {best_path} (avg reward: {best_avg_reward:.2f})")

    # Final evaluation
    print("\n" + "=" * 60)
    print("FINAL EVALUATION")
    print("=" * 60)
    final_eval_mean, final_eval_std, final_eval_rewards = evaluate_agent(
        agent, n_episodes=10, log_handle=log_handle
    )

    # Training complete
    vec_env.close()
    log_handle.close()

    elapsed_time = time.time() - start_time
    hours = elapsed_time / 3600

    # Print comprehensive summary
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print(f"\nTraining Configuration:")
    print(f"  Number of environments: {args.num_envs}")
    print(f"  Parallelism: TRUE (AsyncVectorEnv)")
    print(f"  Episodes: {args.episodes}")
    print(f"  Total steps: {total_steps:,}")
    print(f"  Data collected: {total_steps * args.num_envs:,} transitions ({args.num_envs}× multiplier)")
    print(f"  Learning started at: {args.learning_starts:,} steps")
    print(f"  Buffer size: {len(replay_buffer):,} / {args.buffer_size:,}")

    print(f"\nTraining Results:")
    print(f"  Best avg reward (100 ep): {best_avg_reward:.2f}")
    print(f"  Final episode best env reward: {best_reward:.2f}")
    print(f"  Final episode avg all envs: {avg_reward_all_envs:.2f}")

    print(f"\nFinal Evaluation (10 episodes, single env):")
    print(f"  Mean reward: {final_eval_mean:.2f} ± {final_eval_std:.2f}")
    print(f"  Min reward: {min(final_eval_rewards):.2f}")
    print(f"  Max reward: {max(final_eval_rewards):.2f}")

    print(f"\nTraining Time:")
    print(f"  Total time: {hours:.2f} hours ({elapsed_time/60:.1f} minutes)")
    print(f"  Time per episode: {elapsed_time/args.episodes:.1f} seconds")
    print(f"  Steps per second: {total_steps/elapsed_time:.1f}")
    print(f"  Transitions per second: {(total_steps * args.num_envs)/elapsed_time:.1f} ({args.num_envs}× multiplier)")

    print(f"\nSaved Models:")
    print(f"  Best model: checkpoints_vectorenv/best_model_{args.num_envs}envs.pt")
    print(f"  Latest checkpoint: checkpoints_vectorenv/checkpoint_{args.num_envs}envs_ep{args.episodes}.pt")

    print(f"\nLogs:")
    print(f"  Training log: {training_csv}")
    print(f"  Evaluation log: {eval_csv}")

    print("\n" + "=" * 60)
    print("VectorEnv training with synchronized seeds complete! 🏁")
    print("=" * 60)


if __name__ == '__main__':
    args = parse_args()
    train(args)
