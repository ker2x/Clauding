"""
Test multi-car racing implementation.

Tests:
1. Single-car mode (backward compatibility)
2. Multi-car mode with 4 cars
3. Observation shapes
4. Action handling
5. Reward tracking per car
"""

import numpy as np
from preprocessing import make_carracing_env


def test_single_car():
    """Test single-car mode (backward compatibility)."""
    print("\n" + "="*60)
    print("TEST 1: Single-car mode (backward compatibility)")
    print("="*60)

    env = make_carracing_env(state_mode='vector', num_cars=1)

    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    # Reset
    obs, info = env.reset()
    print(f"Reset observation shape: {obs.shape}")
    print(f"Expected shape: (67,)")
    assert obs.shape == (67,), f"Expected (67,), got {obs.shape}"
    print("✓ Single-car reset works!")

    # Step
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"Step observation shape: {obs.shape}")
    print(f"Step reward type: {type(reward)}, value: {reward}")
    print(f"Step terminated type: {type(terminated)}, value: {terminated}")
    assert obs.shape == (67,), f"Expected (67,), got {obs.shape}"
    assert isinstance(reward, (float, np.floating)), f"Expected scalar reward, got {type(reward)}"
    assert isinstance(terminated, (bool, np.bool_)), f"Expected bool terminated, got {type(terminated)}"
    print("✓ Single-car step works!")

    env.close()
    print("✓ Single-car mode: PASSED")


def test_multi_car():
    """Test multi-car mode with 4 cars."""
    print("\n" + "="*60)
    print("TEST 2: Multi-car mode (4 cars)")
    print("="*60)

    num_cars = 4
    env = make_carracing_env(state_mode='vector', num_cars=num_cars)

    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    # Reset
    obs, info = env.reset()
    print(f"Reset observation shape: {obs.shape}")
    print(f"Expected shape: ({num_cars}, 67)")
    assert obs.shape == (num_cars, 67), f"Expected ({num_cars}, 67), got {obs.shape}"
    print("✓ Multi-car reset works!")

    # Step with random actions for all cars
    actions = np.array([env.action_space.sample() for _ in range(num_cars)])
    print(f"Actions shape: {actions.shape}")
    assert actions.shape == (num_cars, 2), f"Expected ({num_cars}, 2), got {actions.shape}"

    obs, rewards, terminated, truncated, infos = env.step(actions)
    print(f"Step observation shape: {obs.shape}")
    print(f"Step rewards shape: {rewards.shape}")
    print(f"Step terminated shape: {terminated.shape}")
    print(f"Step rewards: {rewards}")
    print(f"Number of info dicts: {len(infos)}")

    assert obs.shape == (num_cars, 67), f"Expected ({num_cars}, 67), got {obs.shape}"
    assert rewards.shape == (num_cars,), f"Expected ({num_cars},), got {rewards.shape}"
    assert terminated.shape == (num_cars,), f"Expected ({num_cars},), got {terminated.shape}"
    assert truncated.shape == (num_cars,), f"Expected ({num_cars},), got {truncated.shape}"
    assert len(infos) == num_cars, f"Expected {num_cars} info dicts, got {len(infos)}"
    print("✓ Multi-car step works!")

    env.close()
    print("✓ Multi-car mode: PASSED")


def test_multiple_steps():
    """Test running multiple steps with multi-car."""
    print("\n" + "="*60)
    print("TEST 3: Multiple steps (50 steps, 4 cars)")
    print("="*60)

    num_cars = 4
    env = make_carracing_env(state_mode='vector', num_cars=num_cars)
    obs, info = env.reset()

    total_rewards = np.zeros(num_cars)

    for step in range(50):
        actions = np.array([env.action_space.sample() for _ in range(num_cars)])
        obs, rewards, terminated, truncated, infos = env.step(actions)
        total_rewards += rewards

        # Print progress every 10 steps
        if (step + 1) % 10 == 0:
            print(f"Step {step+1}: Total rewards = {total_rewards}")

    print(f"\nFinal total rewards after 50 steps:")
    for i, r in enumerate(total_rewards):
        print(f"  Car {i}: {r:.2f}")

    env.close()
    print("✓ Multiple steps: PASSED")


def test_car_colors():
    """Test that cars get different colors."""
    print("\n" + "="*60)
    print("TEST 4: Car colors (4 cars)")
    print("="*60)

    num_cars = 4
    env = make_carracing_env(state_mode='vector', num_cars=num_cars)
    obs, info = env.reset()

    # Check that cars have different colors
    colors = [car.hull.color for car in env.unwrapped.cars]
    print(f"Car colors:")
    for i, color in enumerate(colors):
        print(f"  Car {i}: {color}")

    # Verify all colors are different
    assert len(set(colors)) == num_cars, "Cars should have different colors!"
    print("✓ Car colors: PASSED")

    env.close()


def test_selection():
    """Test selecting the best car based on reward."""
    print("\n" + "="*60)
    print("TEST 5: Car selection (best performer)")
    print("="*60)

    num_cars = 8
    env = make_carracing_env(state_mode='vector', num_cars=num_cars, verbose=False)
    obs, info = env.reset()

    episode_rewards = np.zeros(num_cars)

    # Run for 100 steps
    for step in range(100):
        actions = np.array([env.action_space.sample() for _ in range(num_cars)])
        obs, rewards, terminated, truncated, infos = env.step(actions)
        episode_rewards += rewards

    # Select best car
    best_car_idx = np.argmax(episode_rewards)
    best_reward = episode_rewards[best_car_idx]

    print(f"\nEpisode rewards:")
    for i, r in enumerate(episode_rewards):
        marker = " <-- BEST" if i == best_car_idx else ""
        print(f"  Car {i}: {r:7.2f}{marker}")

    print(f"\n✓ Best car: Car {best_car_idx} with reward {best_reward:.2f}")
    print("✓ Selection: PASSED")

    env.close()


if __name__ == "__main__":
    print("\n" + "#"*60)
    print("# MULTI-CAR RACING TEST SUITE")
    print("#"*60)

    try:
        test_single_car()
        test_multi_car()
        test_multiple_steps()
        test_car_colors()
        test_selection()

        print("\n" + "="*60)
        print("ALL TESTS PASSED! ✓✓✓")
        print("="*60)
        print("\nMulti-car racing implementation is working correctly!")
        print("Ready to train with competitive multi-car racing.")

    except Exception as e:
        print("\n" + "="*60)
        print(f"TEST FAILED: {e}")
        print("="*60)
        import traceback
        traceback.print_exc()
        exit(1)
