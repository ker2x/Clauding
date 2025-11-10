#!/usr/bin/env python3
"""
Selection-Based Training for CarRacing-v3

Trains N independent models simultaneously with periodic selection:
- N models train independently for M episodes
- Every M episodes: evaluate all models, select best
- Clone best model to all N positions
- Repeat: models diverge through exploration, selection pressure maintained

This creates evolutionary pressure without mutation - pure selection.
"""

import argparse
import csv
import os
import time
from datetime import datetime
import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except (ImportError, RuntimeError) as e:
    print(f"Warning: matplotlib not available ({e})")
    MATPLOTLIB_AVAILABLE = False

from preprocessing import make_carracing_env
from sac_agent import SACAgent, ReplayBuffer
from env.car_racing import (
    PROGRESS_REWARD_SCALE, LAP_COMPLETION_REWARD,
    STEP_PENALTY, OFFTRACK_PENALTY, OFFTRACK_THRESHOLD
)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Train SAC agents on CarRacing-v3 with selection-based evolution'
    )

    # Selection parameters
    parser.add_argument('--num-agents', type=int, default=8,
                        help='Number of independent agents (default: 8)')
    parser.add_argument('--selection-frequency', type=int, default=50,
                        help='Select best agent every N episodes (default: 50)')
    parser.add_argument('--eval-episodes', type=int, default=5,
                        help='Episodes to evaluate each agent during selection (default: 5)')

    # Training parameters
    parser.add_argument('--episodes', type=int, default=2000,
                        help='Total training episodes (default: 2000)')
    parser.add_argument('--learning-starts', type=int, default=5000,
                        help='Steps before training starts (default: 5000)')
    parser.add_argument('--checkpoint-frequency', type=int, default=100,
                        help='Save checkpoint every N episodes (default: 100)')

    # Agent hyperparameters
    parser.add_argument('--lr-actor', type=float, default=3e-4,
                        help='Actor learning rate (default: 3e-4)')
    parser.add_argument('--lr-critic', type=float, default=3e-4,
                        help='Critic learning rate (default: 3e-4)')
    parser.add_argument('--lr-alpha', type=float, default=3e-4,
                        help='Alpha (temperature) learning rate (default: 3e-4)')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='Discount factor (default: 0.99)')
    parser.add_argument('--tau', type=float, default=0.005,
                        help='Target network update rate (default: 0.005)')
    parser.add_argument('--auto-entropy-tuning', action='store_true', default=True,
                        help='Automatically tune entropy coefficient')
    parser.add_argument('--buffer-size', type=int, default=1000000,
                        help='Replay buffer size (default: 1000000)')
    parser.add_argument('--batch-size', type=int, default=256,
                        help='Batch size (default: 256)')

    # Environment parameters
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cpu', 'cuda', 'mps'],
                        help='Device to use (default: auto)')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose output')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from (for all agents)')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints_selection',
                        help='Directory to save checkpoints (default: checkpoints_selection)')
    parser.add_argument('--log-dir', type=str, default='logs_selection',
                        help='Directory to save logs (default: logs_selection)')

    return parser.parse_args()


def get_device(device_arg):
    """Determine the device to use."""
    import torch
    if device_arg == 'auto':
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device('mps')
        else:
            return torch.device('cpu')
    else:
        return torch.device(device_arg)


def configure_cpu_threading(device):
    """Configure CPU threading for optimal performance."""
    import torch
    if device.type == 'cpu':
        torch.set_num_threads(4)
        print(f"CPU threading configured: {torch.get_num_threads()} threads")


def evaluate_agent(agent, env, num_episodes, seed_offset=10000):
    """
    Evaluate an agent over multiple episodes.

    Returns:
        float: Average reward over evaluation episodes
    """
    total_reward = 0.0

    for ep in range(num_episodes):
        obs, _ = env.reset(seed=seed_offset + ep)
        episode_reward = 0.0
        terminated = False
        truncated = False

        while not (terminated or truncated):
            action = agent.select_action(obs, evaluate=True)  # Deterministic
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward

        total_reward += episode_reward

    return total_reward / num_episodes


def setup_logging(log_dir, args, config):
    """Set up logging files."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Training log
    training_csv = os.path.join(log_dir, f'training_{timestamp}.csv')
    with open(training_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'episode', 'generation', 'agent_id', 'reward', 'best_agent_id',
            'best_reward', 'avg_reward', 'actor_loss', 'critic_1_loss',
            'critic_2_loss', 'alpha', 'selection_occurred'
        ])

    # Selection log
    selection_csv = os.path.join(log_dir, f'selection_{timestamp}.csv')
    with open(selection_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'generation', 'episode', 'winner_id', 'winner_eval_reward',
            'avg_eval_reward', 'min_eval_reward', 'max_eval_reward'
        ])

    # Hyperparameters log
    hyperparam_file = os.path.join(log_dir, f'hyperparameters_{timestamp}.txt')
    with open(hyperparam_file, 'w') as f:
        f.write("Selection-Based Training Hyperparameters\n")
        f.write("=" * 50 + "\n\n")
        f.write("Selection Parameters:\n")
        f.write(f"  Number of agents: {args.num_agents}\n")
        f.write(f"  Selection frequency: {args.selection_frequency} episodes\n")
        f.write(f"  Evaluation episodes: {args.eval_episodes}\n\n")
        f.write("Training Parameters:\n")
        for key, value in vars(args).items():
            f.write(f"  {key}: {value}\n")
        f.write("\n")
        f.write("Environment Configuration:\n")
        for key, value in config.items():
            f.write(f"  {key}: {value}\n")

    print(f"Logging initialized:")
    print(f"  Training log: {training_csv}")
    print(f"  Selection log: {selection_csv}")
    print(f"  Hyperparameters: {hyperparam_file}")

    return training_csv, selection_csv


def main():
    args = parse_args()

    # Environment constants
    MAX_EPISODE_STEPS = 2500
    STATIONARY_PATIENCE = 50
    STATIONARY_MIN_STEPS = 50
    SHORT_EPISODE_PENALTY = -50.0
    MIN_EPISODE_STEPS = 150

    # Create directories
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    # Determine device
    device = get_device(args.device)
    print(f"Using device: {device}")
    configure_cpu_threading(device)

    # Create environment factory
    def make_env():
        return make_carracing_env(
            state_mode='vector',
            max_episode_steps=MAX_EPISODE_STEPS,
            terminate_stationary=True,
            stationary_patience=STATIONARY_PATIENCE,
            stationary_min_steps=STATIONARY_MIN_STEPS,
            reward_shaping=True,
            min_episode_steps=MIN_EPISODE_STEPS,
            short_episode_penalty=SHORT_EPISODE_PENALTY,
            verbose=args.verbose
        )

    # Create evaluation environment
    eval_env = make_env()
    state_shape = eval_env.observation_space.shape
    action_dim = eval_env.action_space.shape[0]

    print(f"\n{'='*60}")
    print("Selection-Based Training Configuration")
    print(f"{'='*60}")
    print(f"Number of agents: {args.num_agents}")
    print(f"Selection frequency: every {args.selection_frequency} episodes")
    print(f"Evaluation episodes per selection: {args.eval_episodes}")
    print(f"Total episodes: {args.episodes}")
    print(f"Observation space: {state_shape}")
    print(f"Action space: Continuous (2D)")
    print(f"{'='*60}\n")

    # Create N independent agents
    print(f"Creating {args.num_agents} independent agents...")
    agents = []
    replay_buffers = []

    for i in range(args.num_agents):
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

        # Each agent gets its own replay buffer
        buffer = ReplayBuffer(
            capacity=args.buffer_size,
            state_shape=state_shape,
            action_dim=action_dim,
            device=device
        )

        agents.append(agent)
        replay_buffers.append(buffer)

        # Resume from checkpoint if specified
        if args.resume:
            agent.load(args.resume)

    if args.resume:
        print(f"All agents initialized from checkpoint: {args.resume}")
    else:
        print(f"All agents initialized with random weights")

    # Create individual environments for each agent
    envs = [make_env() for _ in range(args.num_agents)]

    # Initialize logging
    config = {
        'max_episode_steps': MAX_EPISODE_STEPS,
        'stationary_patience': STATIONARY_PATIENCE,
        'stationary_min_steps': STATIONARY_MIN_STEPS,
        'min_episode_steps': MIN_EPISODE_STEPS,
        'short_episode_penalty': SHORT_EPISODE_PENALTY,
    }
    training_csv, selection_csv = setup_logging(args.log_dir, args, config)

    # Training state
    generation = 0
    total_steps = 0
    best_overall_reward = float('-inf')

    # Track agent performance
    agent_episode_rewards = [[] for _ in range(args.num_agents)]

    print(f"\n{'='*60}")
    print("Starting Selection-Based Training")
    print(f"{'='*60}\n")

    start_time = time.time()

    for episode in range(args.episodes):
        # Check if selection should occur
        selection_occurred = False
        if episode > 0 and episode % args.selection_frequency == 0:
            generation += 1
            print(f"\n{'='*60}")
            print(f"GENERATION {generation}: Selection Tournament")
            print(f"{'='*60}")

            # Evaluate all agents
            eval_rewards = []
            for agent_id, agent in enumerate(agents):
                eval_reward = evaluate_agent(
                    agent, eval_env, args.eval_episodes, seed_offset=10000 + episode
                )
                eval_rewards.append(eval_reward)
                print(f"  Agent {agent_id}: {eval_reward:.2f} avg reward")

            # Select winner
            winner_id = np.argmax(eval_rewards)
            winner_reward = eval_rewards[winner_id]

            print(f"\n  🏆 WINNER: Agent {winner_id} ({winner_reward:.2f})")
            print(f"  Cloning Agent {winner_id} to all positions...")

            # Clone winner to all positions
            winner_state = agents[winner_id].get_state_dict()
            for agent_id, agent in enumerate(agents):
                if agent_id != winner_id:
                    agent.load_state_dict(winner_state)

            print(f"  All agents now copies of Agent {winner_id}")
            print(f"{'='*60}\n")

            selection_occurred = True

            # Log selection
            with open(selection_csv, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    generation,
                    episode,
                    winner_id,
                    f"{winner_reward:.2f}",
                    f"{np.mean(eval_rewards):.2f}",
                    f"{np.min(eval_rewards):.2f}",
                    f"{np.max(eval_rewards):.2f}"
                ])

        # Train all agents in parallel for this episode
        for agent_id in range(args.num_agents):
            agent = agents[agent_id]
            env = envs[agent_id]
            buffer = replay_buffers[agent_id]

            # Reset environment
            obs, _ = env.reset(seed=1000 + episode + agent_id * 10000)
            episode_reward = 0.0
            terminated = False
            truncated = False
            steps = 0

            # Episode metrics
            actor_losses = []
            critic_1_losses = []
            critic_2_losses = []
            alphas = []

            # Episode loop
            while not (terminated or truncated):
                # Select action
                action = agent.select_action(obs, evaluate=False)

                # Take step
                next_obs, reward, terminated, truncated, _ = env.step(action)

                # Store experience
                buffer.push(obs, action, float(reward), next_obs, float(terminated or truncated))
                episode_reward += reward

                # Train agent
                if (total_steps >= args.learning_starts and
                    len(buffer) >= args.batch_size):
                    metrics = agent.update(buffer, args.batch_size)
                    actor_losses.append(metrics['actor_loss'])
                    critic_1_losses.append(metrics['critic_1_loss'])
                    critic_2_losses.append(metrics['critic_2_loss'])
                    alphas.append(metrics['alpha'])

                obs = next_obs
                steps += 1
                total_steps += 1

            # Record episode reward
            agent_episode_rewards[agent_id].append(episode_reward)

            # Calculate averages
            avg_actor_loss = np.mean(actor_losses) if actor_losses else 0.0
            avg_critic_1_loss = np.mean(critic_1_losses) if critic_1_losses else 0.0
            avg_critic_2_loss = np.mean(critic_2_losses) if critic_2_losses else 0.0
            avg_alpha = np.mean(alphas) if alphas else 0.0

            # Find current best agent
            recent_rewards = [np.mean(rewards[-10:]) if len(rewards) >= 10 else np.mean(rewards)
                            for rewards in agent_episode_rewards]
            best_agent_id = np.argmax(recent_rewards)
            best_reward = recent_rewards[best_agent_id]
            avg_reward = np.mean(recent_rewards)

            # Log to CSV
            with open(training_csv, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    episode + 1,
                    generation,
                    agent_id,
                    f"{episode_reward:.2f}",
                    best_agent_id,
                    f"{best_reward:.2f}",
                    f"{avg_reward:.2f}",
                    f"{avg_actor_loss:.6f}",
                    f"{avg_critic_1_loss:.6f}",
                    f"{avg_critic_2_loss:.6f}",
                    f"{avg_alpha:.6f}",
                    selection_occurred
                ])

        # Print progress
        if (episode + 1) % 10 == 0:
            elapsed = time.time() - start_time
            recent_rewards = [np.mean(rewards[-10:]) if len(rewards) >= 10 else np.mean(rewards)
                            for rewards in agent_episode_rewards]
            print(f"Episode {episode + 1}/{args.episodes} | "
                  f"Gen {generation} | "
                  f"Best: {max(recent_rewards):.1f} | "
                  f"Avg: {np.mean(recent_rewards):.1f} | "
                  f"Time: {elapsed/60:.1f}m")

        # Save checkpoints
        if (episode + 1) % args.checkpoint_frequency == 0:
            # Save all agents
            for agent_id, agent in enumerate(agents):
                checkpoint_path = os.path.join(
                    args.checkpoint_dir,
                    f'agent{agent_id}_ep{episode + 1}.pt'
                )
                agent.save(checkpoint_path)

            # Save best agent
            recent_rewards = [np.mean(rewards[-100:]) if len(rewards) >= 100 else np.mean(rewards)
                            for rewards in agent_episode_rewards]
            best_agent_id = np.argmax(recent_rewards)
            if recent_rewards[best_agent_id] > best_overall_reward:
                best_overall_reward = recent_rewards[best_agent_id]
                best_path = os.path.join(args.checkpoint_dir, 'best_model.pt')
                agents[best_agent_id].save(best_path)
                print(f"✓ New best model saved: Agent {best_agent_id} ({best_overall_reward:.2f})")

    # Cleanup
    for env in envs:
        env.close()
    eval_env.close()

    print(f"\n{'='*60}")
    print("Training Complete!")
    print(f"{'='*60}")
    print(f"Total generations: {generation}")
    print(f"Total time: {(time.time() - start_time)/60:.1f} minutes")
    print(f"Best overall reward: {best_overall_reward:.2f}")
    print(f"Checkpoints saved in: {args.checkpoint_dir}")
    print(f"Logs saved in: {args.log_dir}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
