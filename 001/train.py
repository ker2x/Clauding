"""
Training Script for DQN on Atari Breakout

This script trains a DQN agent to play Atari Breakout.
It demonstrates the full RL training loop:
1. Interact with environment (collect experiences)
2. Store experiences in replay buffer
3. Sample batches and train the network
4. Evaluate and monitor progress
"""

import argparse
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path
import gymnasium as gym

from dqn_agent import DQNAgent
from preprocessing import make_atari_env

# Register ALE environments (required for gymnasium 1.0+)
try:
    import ale_py
    gym.register_envs(ale_py)
except Exception:
    pass


class Trainer:
    """
    DQN Trainer for Atari games

    This class manages the training loop and handles:
    - Experience collection
    - Training updates
    - Evaluation
    - Logging and checkpointing
    """

    def __init__(
        self,
        env_name='ALE/Breakout-v5',
        num_episodes=10000,
        max_steps_per_episode=10000,
        learning_starts=50000,
        train_freq=4,
        target_update_freq=10000,
        eval_freq=100,
        save_freq=500,
        checkpoint_dir='checkpoints',
        log_dir='logs'
    ):
        """
        Initialize trainer

        Args:
            env_name: Atari environment name
            num_episodes: Total number of episodes to train
            max_steps_per_episode: Maximum steps per episode
            learning_starts: Steps before training starts (for filling replay buffer)
            train_freq: Train every N steps
            target_update_freq: Update target network every N steps
            eval_freq: Evaluate every N episodes
            save_freq: Save checkpoint every N episodes
            checkpoint_dir: Directory to save checkpoints
            log_dir: Directory to save logs
        """
        self.env_name = env_name
        self.num_episodes = num_episodes
        self.max_steps_per_episode = max_steps_per_episode
        self.learning_starts = learning_starts
        self.train_freq = train_freq
        self.target_update_freq = target_update_freq
        self.eval_freq = eval_freq
        self.save_freq = save_freq

        # Create directories
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)

        # Create environments (separate for training and evaluation)
        print(f"Creating environment: {env_name}")
        self.train_env = make_atari_env(env_name)
        self.eval_env = make_atari_env(env_name)

        # Get environment info
        state_shape = self.train_env.observation_space.shape
        num_actions = self.train_env.action_space.n
        print(f"State shape: {state_shape}")
        print(f"Number of actions: {num_actions}")

        # Create agent
        self.agent = DQNAgent(
            state_shape=state_shape,
            num_actions=num_actions
        )

        # Training metrics
        self.episode_rewards = []
        self.episode_lengths = []
        self.losses = []
        self.total_steps = 0

    def train(self):
        """
        Main training loop

        The training loop alternates between:
        1. Collecting experiences by interacting with the environment
        2. Training the agent on batches from the replay buffer
        3. Periodically evaluating and saving the agent
        """
        print("\nStarting training...")
        print(f"Training will start after {self.learning_starts} steps (filling replay buffer)")

        for episode in range(self.num_episodes):
            # Reset environment
            state, _ = self.train_env.reset()
            episode_reward = 0
            episode_length = 0

            # Episode loop
            for step in range(self.max_steps_per_episode):
                # Select action
                action = self.agent.select_action(state, training=True)

                # Take step in environment
                next_state, reward, terminated, truncated, info = self.train_env.step(action)
                done = terminated or truncated

                # Store experience in replay buffer
                self.agent.replay_buffer.push(state, action, reward, next_state, done)

                # Update state and metrics
                state = next_state
                episode_reward += reward
                episode_length += 1
                self.total_steps += 1

                # Train agent (if enough experiences collected)
                if self.total_steps >= self.learning_starts and self.total_steps % self.train_freq == 0:
                    loss = self.agent.train_step()
                    if loss is not None:
                        self.losses.append(loss)

                # Update target network
                if self.total_steps % self.target_update_freq == 0:
                    self.agent.update_target_network()
                    print(f"Target network updated at step {self.total_steps}")

                if done:
                    break

            # Store episode metrics
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)

            # Calculate statistics
            avg_reward = np.mean(self.episode_rewards[-100:])
            avg_loss = np.mean(self.losses[-1000:]) if self.losses else 0
            epsilon = self.agent.get_epsilon()

            # Print progress
            print(f"Episode {episode + 1}/{self.num_episodes} | "
                  f"Steps: {self.total_steps} | "
                  f"Reward: {episode_reward:.1f} | "
                  f"Avg Reward (100): {avg_reward:.1f} | "
                  f"Epsilon: {epsilon:.3f} | "
                  f"Loss: {avg_loss:.4f}")

            # Evaluate agent
            if (episode + 1) % self.eval_freq == 0:
                eval_reward = self.evaluate()
                print(f">>> Evaluation | Avg Reward: {eval_reward:.1f}")

            # Save checkpoint
            if (episode + 1) % self.save_freq == 0:
                checkpoint_path = self.checkpoint_dir / f"checkpoint_ep{episode + 1}.pt"
                self.agent.save(checkpoint_path)
                self.save_training_plot()

        print("\nTraining completed!")
        # Save final model
        self.agent.save(self.checkpoint_dir / "final_model.pt")
        self.save_training_plot()

    def evaluate(self, num_episodes=5):
        """
        Evaluate agent performance

        Args:
            num_episodes: Number of episodes to evaluate

        Returns:
            Average reward over evaluation episodes
        """
        eval_rewards = []

        for _ in range(num_episodes):
            state, _ = self.eval_env.reset()
            episode_reward = 0
            done = False

            while not done:
                # Select action (no exploration)
                action = self.agent.select_action(state, training=False)
                state, reward, terminated, truncated, _ = self.eval_env.step(action)
                done = terminated or truncated
                episode_reward += reward

            eval_rewards.append(episode_reward)

        return np.mean(eval_rewards)

    def save_training_plot(self):
        """Save plots of training progress"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Episode rewards
        axes[0, 0].plot(self.episode_rewards, alpha=0.6, label='Episode Reward')
        if len(self.episode_rewards) >= 100:
            avg_rewards = [np.mean(self.episode_rewards[max(0, i-100):i+1])
                          for i in range(len(self.episode_rewards))]
            axes[0, 0].plot(avg_rewards, label='Avg Reward (100 eps)', linewidth=2)
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Episode lengths
        axes[0, 1].plot(self.episode_lengths, alpha=0.6)
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Length')
        axes[0, 1].set_title('Episode Lengths')
        axes[0, 1].grid(True, alpha=0.3)

        # Training loss
        if self.losses:
            axes[1, 0].plot(self.losses, alpha=0.6)
            axes[1, 0].set_xlabel('Training Step')
            axes[1, 0].set_ylabel('Loss')
            axes[1, 0].set_title('Training Loss')
            axes[1, 0].grid(True, alpha=0.3)

        # Epsilon decay
        epsilons = [self.agent.epsilon_end + (self.agent.epsilon_start - self.agent.epsilon_end) *
                   np.exp(-1. * step / self.agent.epsilon_decay)
                   for step in range(self.total_steps)]
        axes[1, 1].plot(epsilons)
        axes[1, 1].set_xlabel('Step')
        axes[1, 1].set_ylabel('Epsilon')
        axes[1, 1].set_title('Exploration Rate (Epsilon)')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.log_dir / 'training_progress.png', dpi=100)
        plt.close()


def main():
    """Main entry point for training"""
    parser = argparse.ArgumentParser(description='Train DQN on Atari Breakout')
    parser.add_argument('--env', type=str, default='ALE/Breakout-v5',
                       help='Atari environment name')
    parser.add_argument('--episodes', type=int, default=10000,
                       help='Number of episodes to train')
    parser.add_argument('--learning-starts', type=int, default=50000,
                       help='Steps before training starts')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                       help='Directory to save checkpoints')
    parser.add_argument('--log-dir', type=str, default='logs',
                       help='Directory to save logs')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--reset-epsilon', action='store_true',
                       help='Reset epsilon to encourage exploration when resuming')
    args = parser.parse_args()

    # Create trainer
    trainer = Trainer(
        env_name=args.env,
        num_episodes=args.episodes,
        learning_starts=args.learning_starts,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir
    )

    # Resume from checkpoint if specified
    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")
        trainer.agent.load(args.resume)
        if args.reset_epsilon:
            print("Resetting epsilon to encourage exploration")
            trainer.agent.steps_done = 0  # Reset to epsilon_start
        else:
            print(f"Continuing with epsilon at step {trainer.agent.steps_done}")
        print()

    # Start training
    trainer.train()


if __name__ == '__main__':
    main()
