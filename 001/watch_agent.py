"""
Watch a Trained DQN Agent Play

This script loads a trained agent and visualizes it playing Atari Breakout.
Useful for:
1. Seeing how well your agent learned
2. Debugging agent behavior
3. Creating demo videos
"""

import argparse
import numpy as np
import torch
import gymnasium as gym
from preprocessing import make_atari_env
from dqn_agent import DQNAgent
import time

# Register ALE environments (required for gymnasium 1.0+)
try:
    import ale_py
    gym.register_envs(ale_py)
except Exception:
    pass


def watch_agent(checkpoint_path, env_name='ALE/Breakout-v5', num_episodes=5, render=True):
    """
    Load and watch a trained agent play

    Args:
        checkpoint_path: Path to model checkpoint
        env_name: Atari environment name
        num_episodes: Number of episodes to watch
        render: Whether to render the environment
    """
    # Create environment with rendering
    if render:
        env = gym.make(env_name, render_mode='human')
        # Still need to apply our preprocessing
        from preprocessing import AtariPreprocessing
        env = AtariPreprocessing(env, frame_stack=4)
    else:
        env = make_atari_env(env_name, clip_rewards=False)

    # Create agent
    state_shape = env.observation_space.shape
    num_actions = env.action_space.n
    agent = DQNAgent(state_shape=state_shape, num_actions=num_actions)

    # Load checkpoint
    print(f"Loading model from {checkpoint_path}")
    agent.load(checkpoint_path)

    # Watch agent play
    episode_rewards = []

    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        step = 0

        print(f"\nEpisode {episode + 1}/{num_episodes}")

        while not done:
            # Select action (no exploration)
            action = agent.select_action(state, training=False)

            # Take step
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            step += 1

            # Small delay for human viewing
            if render:
                time.sleep(0.02)  # ~50 FPS

            if done:
                print(f"Episode finished after {step} steps | Reward: {episode_reward}")
                episode_rewards.append(episode_reward)

    env.close()

    # Print statistics
    print(f"\n{'='*50}")
    print(f"Average Reward: {np.mean(episode_rewards):.2f}")
    print(f"Std Reward: {np.std(episode_rewards):.2f}")
    print(f"Min Reward: {np.min(episode_rewards):.2f}")
    print(f"Max Reward: {np.max(episode_rewards):.2f}")
    print(f"{'='*50}")


def main():
    parser = argparse.ArgumentParser(description='Watch trained DQN agent play')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--env', type=str, default='ALE/Breakout-v5',
                       help='Atari environment name')
    parser.add_argument('--episodes', type=int, default=5,
                       help='Number of episodes to watch')
    parser.add_argument('--no-render', action='store_true',
                       help='Disable rendering (just compute rewards)')
    args = parser.parse_args()

    watch_agent(
        checkpoint_path=args.checkpoint,
        env_name=args.env,
        num_episodes=args.episodes,
        render=not args.no_render
    )


if __name__ == '__main__':
    main()
