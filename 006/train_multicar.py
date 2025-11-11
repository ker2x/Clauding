"""
Multi-car competitive training script for SAC agent on CarRacing-v3.

This script extends train.py to support N cars racing competitively on the same track:
1. Creates CarRacing environment with num_cars parameter
2. All cars race as ghost cars (no collision with each other)
3. Collects experiences from all N cars simultaneously (N× data collection)
4. Tracks per-car episode rewards
5. Selects and logs the best performer each episode
6. Maintains shared replay buffer for all cars

Benefits:
- N× faster data collection (same wall-clock time)
- Natural selection mechanism (best car identified)
- Fair competition (same track for all cars)
- Minimal memory overhead (~300 bytes per car)

Usage:
    # Train with 4 cars (4× data collection rate)
    python train_multicar.py --num-cars 4

    # Train with 8 cars for faster learning
    python train_multicar.py --num-cars 8 --episodes 1000

    # Resume training
    python train_multicar.py --num-cars 8 --resume checkpoints/best_model.pt
"""

import argparse
import os
import time
import csv
from datetime import datetime
import numpy as np
import torch
import multiprocessing

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
    PROGRESS_REWARD_SCALE, LAP_COMPLETION_REWARD,
    STEP_PENALTY, OFFTRACK_PENALTY, OFFTRACK_THRESHOLD,
    OFFTRACK_TERMINATION_PENALTY, ONTRACK_REWARD, FORWARD_SPEED_REWARD_SCALE
)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Train SAC agent on CarRacing-v3 with multi-car competitive racing')

    # Multi-car parameters
    parser.add_argument('--num-cars', type=int, default=4,
                        help='Number of cars racing competitively (default: 4)')

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
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints_multicar',
                        help='Directory to save checkpoints (default: checkpoints_multicar)')
    parser.add_argument('--log-dir', type=str, default='logs_multicar',
                        help='Directory to save logs (default: logs_multicar)')

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


def evaluate_agent(agent, num_cars, n_episodes=5, log_handle=None):
    """
    Evaluate agent performance over multiple episodes (single-car evaluation).

    Args:
        agent: SAC agent
        num_cars: Number of cars for training (but eval uses 1 car)
        n_episodes: Number of episodes to evaluate
        log_handle: Optional file handle for logging

    Returns:
        Tuple of (mean_reward, std_reward, all_rewards)
    """
    # Create single-car environment for evaluation
    eval_env = make_carracing_env(
        state_mode='vector',
        num_cars=1,  # Always evaluate with single car
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


def setup_logging(log_dir, args, env, agent, config, num_cars):
    """Setup logging infrastructure."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = f'training_multicar_{num_cars}cars_{timestamp}.csv'
    eval_filename = f'eval_multicar_{num_cars}cars_{timestamp}.csv'
    config_filename = f'config_multicar_{num_cars}cars_{timestamp}.txt'

    training_csv = os.path.join(log_dir, log_filename)
    eval_csv = os.path.join(log_dir, eval_filename)
    config_txt = os.path.join(log_dir, config_filename)

    # Training CSV header
    with open(training_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'episode', 'total_steps', 'episode_steps', 'best_car_idx', 'best_reward',
            'avg_reward_all_cars', 'min_reward', 'max_reward',
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
        f.write(f"Multi-Car Competitive Training Configuration\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Timestamp: {timestamp}\n\n")

        f.write("Multi-Car Settings:\n")
        f.write(f"  Number of cars: {num_cars}\n")
        f.write(f"  Data collection rate: {num_cars}× simultaneous experiences\n\n")

        f.write("Environment:\n")
        f.write(f"  State mode: vector (67D)\n")
        f.write(f"  Observation space: ({num_cars}, 67)\n")
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
    """Main multi-car training loop."""
    # Environment configuration
    STACK_SIZE = 4
    TERMINATE_STATIONARY = True
    STATIONARY_PATIENCE = 50
    REWARD_SHAPING = True
    MIN_EPISODE_STEPS = 100
    SHORT_EPISODE_PENALTY = -50.0
    MAX_EPISODE_STEPS = 1500

    # Validate num_cars
    if args.num_cars < 1:
        raise ValueError(f"num_cars must be >= 1, got {args.num_cars}")

    # Create directories
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    # Determine device
    device = get_device(args.device)
    print(f"Using device: {device}")

    # Configure CPU threading
    configure_cpu_threading(device)

    # Create multi-car environment
    print(f"\nCreating CarRacing-v3 environment with {args.num_cars} cars...")
    env = make_carracing_env(
        stack_size=STACK_SIZE,
        terminate_stationary=TERMINATE_STATIONARY,
        stationary_patience=STATIONARY_PATIENCE,
        render_mode=None,
        state_mode='vector',  # Multi-car only supports vector mode
        reward_shaping=REWARD_SHAPING,
        min_episode_steps=MIN_EPISODE_STEPS,
        short_episode_penalty=SHORT_EPISODE_PENALTY,
        max_episode_steps=MAX_EPISODE_STEPS,
        verbose=args.verbose,
        num_cars=args.num_cars
    )

    action_dim = env.action_space.shape[0]
    obs_space_shape = env.observation_space.shape

    # State shape for agent is single car observation (67,)
    if args.num_cars > 1:
        state_shape = (obs_space_shape[1],)  # (67,) from (num_cars, 67)
    else:
        state_shape = obs_space_shape

    print(f"Environment created:")
    print(f"  Number of cars: {args.num_cars}")
    print(f"  Data collection rate: {args.num_cars}× simultaneous experiences")
    print(f"  Observation space shape: {obs_space_shape}")
    print(f"  Single car state shape: {state_shape}")
    print(f"  Action space: Continuous (2D)")
    print(f"  Max episode steps: {MAX_EPISODE_STEPS}")
    print(f"  Reward shaping enabled (penalty {SHORT_EPISODE_PENALTY} for episodes < {MIN_EPISODE_STEPS} steps)")

    # Create agent (same agent for all cars - shared policy)
    print("\nCreating SAC agent (shared policy for all cars)...")
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
        args.log_dir, args, env, agent, config, args.num_cars
    )

    # Training metrics
    episode_best_rewards = []  # Best car reward per episode
    episode_avg_rewards = []   # Average reward across all cars per episode
    training_metrics = {
        'actor_loss': [],
        'critic_1_loss': [],
        'critic_2_loss': [],
        'alpha': []
    }
    best_avg_reward = -float('inf')

    # Start training
    print("\n" + "=" * 60)
    print("Starting Multi-Car Competitive Training")
    print("=" * 60)
    print(f"Number of cars: {args.num_cars}")
    print(f"Episodes: {args.episodes}")
    print(f"Learning starts: {args.learning_starts} steps")
    print(f"Data collection: {args.num_cars}× per step")
    print(f"Device: {agent.device}")
    print(f"Plotting: {'enabled' if MATPLOTLIB_AVAILABLE else 'disabled'}")
    print("=" * 60 + "\n")

    start_time = time.time()

    for episode in range(start_episode, start_episode + args.episodes):
        # Reset environment - returns (num_cars, 67) observations
        obs, _ = env.reset()

        # Per-car episode tracking
        car_episode_rewards = np.zeros(args.num_cars)
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
            # Select actions for all cars (shared policy)
            actions = np.array([
                agent.select_action(obs[i], evaluate=False)
                for i in range(args.num_cars)
            ])

            # Take step - returns vectorized outputs
            next_obs, rewards, terminated, truncated, infos = env.step(actions)

            # Store experiences from all cars
            for i in range(args.num_cars):
                # Only store if car hasn't terminated yet
                if not (terminated[i] or truncated[i]):
                    replay_buffer.push(
                        obs[i],
                        actions[i],
                        float(rewards[i]),  # Convert numpy.float32 to Python float
                        next_obs[i],
                        float(terminated[i] or truncated[i])
                    )
                car_episode_rewards[i] += rewards[i]

            # Increment step counter (count per environment step, not per car)
            total_steps += 1

            # Train agent (only after learning_starts steps)
            if total_steps >= args.learning_starts and len(replay_buffer) >= args.batch_size:
                metrics = agent.update(replay_buffer, args.batch_size)
                episode_metrics['actor_loss'].append(metrics['actor_loss'])
                episode_metrics['critic_1_loss'].append(metrics['critic_1_loss'])
                episode_metrics['critic_2_loss'].append(metrics['critic_2_loss'])
                episode_metrics['alpha'].append(metrics['alpha'])

            obs = next_obs
            steps += 1

            # Episode done when ALL cars are done
            done = all(terminated) or all(truncated)

        # Select best car for this episode
        best_car_idx = np.argmax(car_episode_rewards)
        best_reward = car_episode_rewards[best_car_idx]
        avg_reward_all_cars = np.mean(car_episode_rewards)
        min_reward = np.min(car_episode_rewards)
        max_reward = np.max(car_episode_rewards)

        # Record metrics
        episode_best_rewards.append(best_reward)
        episode_avg_rewards.append(avg_reward_all_cars)

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
                best_car_idx,
                f"{best_reward:.2f}",
                f"{avg_reward_all_cars:.2f}",
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
            print(f"\nEpisode {episode + 1}/{start_episode + args.episodes} | Time: {hours:.2f}h")
            print(f"  Cars: [{', '.join([f'{r:6.1f}' for r in car_episode_rewards])}]")
            print(f"  Best: Car {best_car_idx} ({best_reward:7.2f}) | Avg: {avg_reward_all_cars:7.2f} | Range: [{min_reward:.1f}, {max_reward:.1f}]")
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
            eval_mean, eval_std, eval_rewards = evaluate_agent(agent, args.num_cars, n_episodes=5, log_handle=log_handle)

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
                f'checkpoint_{args.num_cars}cars_ep{episode + 1}.pt'
            )
            agent.save(checkpoint_path)
            print(f"✓ Checkpoint saved: {checkpoint_path}")

        # Save best model
        if avg_best_reward_100 > best_avg_reward:
            best_avg_reward = avg_best_reward_100
            best_path = os.path.join(args.checkpoint_dir, f'best_model_{args.num_cars}cars.pt')
            agent.save(best_path)
            print(f"✓ New best model saved: {best_path} (avg reward: {best_avg_reward:.2f})")

    # Final evaluation
    print("\n" + "=" * 60)
    print("FINAL EVALUATION")
    print("=" * 60)
    final_eval_mean, final_eval_std, final_eval_rewards = evaluate_agent(
        agent, args.num_cars, n_episodes=10, log_handle=log_handle
    )

    # Training complete
    env.close()
    log_handle.close()

    elapsed_time = time.time() - start_time
    hours = elapsed_time / 3600

    # Print comprehensive summary
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print(f"\nTraining Configuration:")
    print(f"  Number of cars: {args.num_cars}")
    print(f"  Episodes: {args.episodes}")
    print(f"  Total steps: {total_steps:,}")
    print(f"  Data collected: {total_steps * args.num_cars:,} transitions ({args.num_cars}× multiplier)")
    print(f"  Learning started at: {args.learning_starts:,} steps")
    print(f"  Buffer size: {len(replay_buffer):,} / {args.buffer_size:,}")

    print(f"\nTraining Results:")
    print(f"  Best avg reward (100 ep): {best_avg_reward:.2f}")
    print(f"  Final episode best car reward: {best_reward:.2f}")
    print(f"  Final episode avg all cars: {avg_reward_all_cars:.2f}")

    print(f"\nFinal Evaluation (10 episodes, single car):")
    print(f"  Mean reward: {final_eval_mean:.2f} ± {final_eval_std:.2f}")
    print(f"  Min reward: {min(final_eval_rewards):.2f}")
    print(f"  Max reward: {max(final_eval_rewards):.2f}")

    print(f"\nTraining Time:")
    print(f"  Total time: {hours:.2f} hours ({elapsed_time/60:.1f} minutes)")
    print(f"  Time per episode: {elapsed_time/args.episodes:.1f} seconds")
    print(f"  Steps per second: {total_steps/elapsed_time:.1f}")
    print(f"  Transitions per second: {(total_steps * args.num_cars)/elapsed_time:.1f} ({args.num_cars}× multiplier)")

    print(f"\nSaved Models:")
    print(f"  Best model: checkpoints_multicar/best_model_{args.num_cars}cars.pt")
    print(f"  Latest checkpoint: checkpoints_multicar/checkpoint_{args.num_cars}cars_ep{args.episodes}.pt")

    print(f"\nLogs:")
    print(f"  Training log: {training_csv}")
    print(f"  Evaluation log: {eval_csv}")

    print("\n" + "=" * 60)
    print("Multi-car competitive training complete! 🏁")
    print("=" * 60)


if __name__ == '__main__':
    args = parse_args()
    train(args)
