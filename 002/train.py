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
    parser.add_argument('--state-mode', type=str, default='vector', choices=['visual', 'vector'],
                        help='State representation: visual (images) or vector (state) - vector is 3-5x faster (default: vector)')

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


def evaluate_agent(agent, env, n_episodes=5):
    """
    Evaluate agent performance over multiple episodes.

    Args:
        agent: DDQN agent
        env: CarRacing environment
        n_episodes: Number of episodes to evaluate

    Returns:
        Average reward over evaluation episodes
    """
    total_rewards = []

    for ep in range(n_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False

        while not done:
            # Use greedy policy (no exploration)
            action = agent.select_action(state, training=False)
            state, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            done = terminated or truncated

        total_rewards.append(episode_reward)
        print(f"  Eval episode {ep + 1}/{n_episodes}: reward = {episode_reward:.2f}")

    return np.mean(total_rewards)


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
        stationary_patience=100,
        render_mode=None,
        state_mode=args.state_mode
    )

    n_actions = env.action_space.n
    state_shape = env.observation_space.shape

    print(f"Environment created:")
    print(f"  State mode: {args.state_mode}")
    print(f"  State shape: {state_shape}")
    print(f"  Number of actions: {n_actions}")
    print(f"  Early termination enabled (patience=100 frames)")

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

    # Training metrics
    episode_rewards = []
    episode_losses = []
    episode_epsilons = []
    best_avg_reward = -float('inf')

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

        # Print progress
        elapsed_time = time.time() - start_time
        if (episode + 1) % 10 == 0:
            avg_reward_recent = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards)
            print(f"Episode {episode + 1}/{start_episode + args.episodes} | "
                  f"Reward: {episode_reward:7.2f} | "
                  f"Avg(100): {avg_reward_recent:7.2f} | "
                  f"Loss: {avg_loss:7.4f} | "
                  f"Epsilon: {agent.epsilon:.4f} | "
                  f"Steps: {agent.steps_done:7d} | "
                  f"Buffer: {len(agent.replay_buffer):6d} | "
                  f"Time: {elapsed_time/60:.1f}m")

        # Evaluate periodically
        if (episode + 1) % args.eval_frequency == 0:
            print("\n" + "-" * 60)
            print(f"Evaluating at episode {episode + 1}...")
            eval_reward = evaluate_agent(agent, env, n_episodes=5)
            print(f"Evaluation reward (5 episodes): {eval_reward:.2f}")
            print("-" * 60 + "\n")

            # Save best model
            if eval_reward > best_avg_reward:
                best_avg_reward = eval_reward
                best_path = os.path.join(args.checkpoint_dir, 'best_model.pt')
                agent.save(best_path)
                print(f"New best model saved! Avg reward: {eval_reward:.2f}\n")

        # Save checkpoint periodically
        if (episode + 1) % args.checkpoint_frequency == 0:
            checkpoint_path = os.path.join(args.checkpoint_dir, f'checkpoint_ep{episode + 1}.pt')
            agent.save(checkpoint_path)

            # Save training plot
            plot_path = os.path.join(args.log_dir, 'training_progress.png')
            plot_training_progress(episode_rewards, episode_losses, episode_epsilons, plot_path)

    # Save final model
    final_path = os.path.join(args.checkpoint_dir, 'final_model.pt')
    agent.save(final_path)

    # Final evaluation
    print("\n" + "=" * 60)
    print("Training complete! Running final evaluation...")
    final_eval_reward = evaluate_agent(agent, env, n_episodes=10)
    print(f"Final evaluation reward (10 episodes): {final_eval_reward:.2f}")
    print("=" * 60)

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
    print(f"Final evaluation reward: {final_eval_reward:.2f}")
    print(f"\nCheckpoints saved in: {args.checkpoint_dir}")
    print(f"Logs saved in: {args.log_dir}")
    print("=" * 60)

    env.close()


if __name__ == "__main__":
    args = parse_args()
    train(args)
