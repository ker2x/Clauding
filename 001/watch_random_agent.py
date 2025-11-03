"""
Watch a Random Agent Play Breakout

This script plays Breakout with random actions to verify that:
1. The game mechanics work properly
2. The ball launches and moves
3. The paddle moves left and right
4. Rewards are earned when hitting bricks
5. The rendering displays correctly
"""

import argparse
import numpy as np
import gymnasium as gym
import cv2

# Register ALE environments
try:
    import ale_py
    gym.register_envs(ale_py)
except Exception:
    pass


def watch_random_agent(env_name='ALE/Breakout-v5', num_episodes=3):
    """
    Play Breakout with random actions

    Args:
        env_name: Atari environment name
        num_episodes: Number of episodes to play
    """
    print("Creating environment...")
    env = gym.make(env_name, render_mode='rgb_array')

    print(f"Actions available: {env.unwrapped.get_action_meanings()}")
    print(f"Number of actions: {env.action_space.n}\n")

    # Setup OpenCV window
    cv2.namedWindow('Random Agent - Breakout', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Random Agent - Breakout', 640, 840)

    episode_rewards = []

    for episode in range(num_episodes):
        print(f"\n{'='*50}")
        print(f"Episode {episode + 1}/{num_episodes}")
        print(f"{'='*50}")

        obs, _ = env.reset()

        # Fire to start the game
        print("Firing to start game...")
        obs, _, _, _, _ = env.step(1)  # FIRE

        episode_reward = 0
        done = False
        step = 0
        action_counts = {0: 0, 1: 0, 2: 0, 3: 0}

        while not done:
            # Random action
            action = env.action_space.sample()
            action_counts[action] += 1

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            step += 1

            # Render frame
            frame = env.render()
            if frame is not None:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                frame_resized = cv2.resize(frame_bgr, (640, 840), interpolation=cv2.INTER_NEAREST)
                cv2.imshow('Random Agent - Breakout', frame_resized)

                # Wait 1ms and check for 'q' to quit
                key = cv2.waitKey(1)
                if key & 0xFF == ord('q'):
                    env.close()
                    cv2.destroyAllWindows()
                    return

            # Print rewards
            if reward > 0:
                print(f"  Step {step}: REWARD = {reward}! Total: {episode_reward}")

            # Print progress every 500 steps
            if step % 500 == 0:
                print(f"  Step {step}, Reward: {episode_reward}")

        print(f"\nEpisode finished!")
        print(f"  Total steps: {step}")
        print(f"  Total reward: {episode_reward}")
        print(f"  Action distribution: NOOP={action_counts[0]}, FIRE={action_counts[1]}, RIGHT={action_counts[2]}, LEFT={action_counts[3]}")
        episode_rewards.append(episode_reward)

    env.close()
    cv2.destroyAllWindows()

    # Print statistics
    print(f"\n{'='*50}")
    print("RANDOM AGENT STATISTICS")
    print(f"{'='*50}")
    print(f"Episodes played: {num_episodes}")
    print(f"Average reward: {np.mean(episode_rewards):.2f}")
    print(f"Std reward: {np.std(episode_rewards):.2f}")
    print(f"Min reward: {np.min(episode_rewards):.2f}")
    print(f"Max reward: {np.max(episode_rewards):.2f}")
    print(f"{'='*50}")


def main():
    parser = argparse.ArgumentParser(description='Watch random agent play Breakout')
    parser.add_argument('--env', type=str, default='ALE/Breakout-v5',
                       help='Atari environment name')
    parser.add_argument('--episodes', type=int, default=3,
                       help='Number of episodes to watch')
    args = parser.parse_args()

    watch_random_agent(env_name=args.env, num_episodes=args.episodes)


if __name__ == '__main__':
    main()
