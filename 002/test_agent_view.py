"""
Test script to visualize what the agent actually sees during training.

This shows:
1. The optimized 96x96 rendering that's used during headless training
2. How it compares to the standard view
3. The preprocessed stacked frames that go into the neural network

Usage:
    python test_agent_view.py

Controls:
    - Arrow keys: Control the car
    - ESC: Quit
    - R: Reset environment
"""

import pygame
import numpy as np
from env.car_racing import CarRacing
from preprocessing import make_carracing_env


def test_agent_view_comparison():
    """
    Show side-by-side comparison of standard view vs agent view.
    """
    print("=" * 60)
    print("Agent View Visualization Test")
    print("=" * 60)
    print("\nThis test shows what the agent actually sees during training.")
    print("\nControls:")
    print("  Arrow keys: Control the car")
    print("  ESC: Quit")
    print("  R: Reset environment")
    print("\nPress any key to start...")
    print("=" * 60)

    # Create environment with agent_view render mode
    env = CarRacing(render_mode="agent_view", continuous=True)

    print("\nEnvironment created with 'agent_view' mode")
    print("Window shows 96x96 optimized view (scaled 6x for visibility)")
    print("\nStarting simulation...")

    # Initialize pygame for input
    pygame.init()
    clock = pygame.time.Clock()

    # Action array for manual control
    action = np.array([0.0, 0.0, 0.0])  # [steering, gas, brake]

    quit_sim = False
    obs, info = env.reset()

    episode_count = 0
    step_count = 0

    while not quit_sim:
        # Handle input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                quit_sim = True
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    quit_sim = True
                elif event.key == pygame.K_r:
                    obs, info = env.reset()
                    episode_count += 1
                    step_count = 0
                    print(f"\nReset environment (Episode {episode_count})")
                elif event.key == pygame.K_LEFT:
                    action[0] = -1.0
                elif event.key == pygame.K_RIGHT:
                    action[0] = 1.0
                elif event.key == pygame.K_UP:
                    action[1] = 1.0
                elif event.key == pygame.K_DOWN:
                    action[2] = 0.8
            elif event.type == pygame.KEYUP:
                if event.key == pygame.K_LEFT or event.key == pygame.K_RIGHT:
                    action[0] = 0.0
                elif event.key == pygame.K_UP:
                    action[1] = 0.0
                elif event.key == pygame.K_DOWN:
                    action[2] = 0.0

        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)
        step_count += 1

        # Render (this will show the agent view window)
        env.render()

        # Print info periodically
        if step_count % 100 == 0:
            print(f"Step {step_count}, Reward: {reward:.2f}")

        # Auto-reset on done
        if terminated or truncated:
            print(f"\nEpisode finished after {step_count} steps")
            obs, info = env.reset()
            episode_count += 1
            step_count = 0

        clock.tick(50)  # 50 FPS

    env.close()
    pygame.quit()
    print("\nTest complete!")


def test_preprocessed_view():
    """
    Show the fully preprocessed view (grayscale, resized, stacked) that goes into the neural network.
    """
    print("\n" + "=" * 60)
    print("Testing Preprocessed Agent View")
    print("=" * 60)
    print("\nThis shows the final preprocessed frames that go into the neural network:")
    print("  - Grayscale conversion")
    print("  - Resized to 84x84")
    print("  - Normalized to [0, 1]")
    print("  - Stacked (4 frames)")
    print("\nRunning 100 steps to show preprocessing...")

    # Create fully preprocessed environment
    env = make_carracing_env(
        stack_size=4,
        frame_size=(96, 96),
        discretize_actions=True,
        render_mode=None  # Headless mode - super fast!
    )

    obs, info = env.reset()

    print(f"\nObservation shape: {obs.shape}")
    print(f"Expected: (4, 96, 96) for 4 stacked 96x96 grayscale frames")
    print(f"Data type: {obs.dtype}")
    print(f"Value range: [{obs.min():.3f}, {obs.max():.3f}]")

    # Run a few steps
    print("\nRunning 100 steps with random actions...")
    total_reward = 0

    for i in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated or truncated:
            print(f"  Episode finished at step {i+1}, reward: {total_reward:.2f}")
            obs, info = env.reset()
            total_reward = 0

    print(f"\nFinal observation stats:")
    print(f"  Shape: {obs.shape}")
    print(f"  Mean: {obs.mean():.3f}")
    print(f"  Std: {obs.std():.3f}")
    print(f"  Min/Max: [{obs.min():.3f}, {obs.max():.3f}]")

    # Show what each frame in the stack looks like
    print(f"\nFrame stack contents:")
    for i in range(4):
        frame = obs[i]
        print(f"  Frame {i}: mean={frame.mean():.3f}, std={frame.std():.3f}")

    env.close()
    print("\nPreprocessing test complete!")


if __name__ == "__main__":
    # Test 1: Visual agent view
    test_agent_view_comparison()

    # Test 2: Preprocessed numerical view
    test_preprocessed_view()

    print("\n" + "=" * 60)
    print("All tests complete!")
    print("=" * 60)
