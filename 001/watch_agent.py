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
    # Use rgb_array mode and manual display (more reliable on macOS than pygame)
    print("Creating environment...")
    if render:
        import cv2
        env = gym.make(env_name, render_mode='rgb_array')
        # Apply FIRE wrapper (launches the ball at game start and after losing lives)
        from preprocessing import FireResetWrapper, NoopFireLeftRightActions, AtariPreprocessing
        env = FireResetWrapper(env)
        # Simplify action space (remove FIRE)
        env = NoopFireLeftRightActions(env)
        # Apply our preprocessing wrapper
        env = AtariPreprocessing(env, frame_stack=4)
    else:
        env = make_atari_env(env_name, clip_rewards=False)

    # Create agent
    print("Creating agent...")
    state_shape = env.observation_space.shape
    num_actions = env.action_space.n
    agent = DQNAgent(state_shape=state_shape, num_actions=num_actions)

    # Load checkpoint
    print(f"Loading checkpoint from {checkpoint_path}...")
    agent.load(checkpoint_path)
    print("Agent loaded successfully!\n")

    # Watch agent play
    episode_rewards = []

    # Import cv2 for rendering (only if rendering)
    if render:
        import cv2
        cv2.namedWindow('Breakout Agent', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Breakout Agent', 640, 840)

    for episode in range(num_episodes):
        print(f"\nEpisode {episode + 1}/{num_episodes}")
        state, _ = env.reset()
        episode_reward = 0
        done = False
        step = 0

        # Track action distribution
        action_counts = {0: 0, 1: 0, 2: 0}

        while not done:
            # Select action (no exploration)
            action = agent.select_action(state, training=False)
            action_counts[action] += 1

            # Debug: print first 10 actions
            if step < 10:
                action_names = ['NOOP', 'RIGHT', 'LEFT']
                print(f"    [DEBUG] Step {step}: action={action} ({action_names[action]})")

            # Take step
            state, reward, terminated, truncated, info = env.step(action)

            if reward != 0:
                print(f"    [REWARD] Step {step}: reward={reward}")

            # Manual rendering with OpenCV (more reliable on macOS)
            if render:
                frame = env.render()
                if frame is not None:
                    # Debug: log every 500 steps
                    if step % 500 == 0:
                        print(f"    [DEBUG] Rendering frame at step {step}, shape: {frame.shape}")
                    # Convert RGB to BGR for OpenCV
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    # Resize for better visibility
                    frame_resized = cv2.resize(frame_bgr, (640, 840), interpolation=cv2.INTER_NEAREST)
                    cv2.imshow('Breakout Agent', frame_resized)
                    # Wait 1ms and process events (critical for macOS)
                    key = cv2.waitKey(1)
                    if key & 0xFF == ord('q'):
                        env.close()
                        cv2.destroyAllWindows()
                        return

            done = terminated or truncated
            episode_reward += reward
            step += 1

            # Print progress every 1000 steps
            if step % 1000 == 0:
                print(f"  Step {step}, Reward so far: {episode_reward}")

            if done:
                print(f"Episode {episode + 1}/{num_episodes} finished | Steps: {step} | Reward: {episode_reward}")
                print(f"Action distribution: NOOP={action_counts[0]}, RIGHT={action_counts[1]}, LEFT={action_counts[2]}")
                episode_rewards.append(episode_reward)

    env.close()
    if render:
        cv2.destroyAllWindows()

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
