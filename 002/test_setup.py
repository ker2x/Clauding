"""
Test script to verify CarRacing-v3 environment setup and dependencies.

This script checks:
1. All required packages are installed
2. CarRacing-v3 environment can be created
3. Preprocessing pipeline works correctly
4. DDQN agent can be initialized
5. Device (CUDA/MPS/CPU) is correctly detected

Run this before training to ensure everything is set up correctly.
"""

import sys


def test_imports():
    """Test that all required packages can be imported."""
    print("=" * 60)
    print("Testing imports...")
    print("=" * 60)

    required_packages = {
        'gymnasium': 'gymnasium',
        'torch': 'torch',
        'numpy': 'numpy',
        'cv2': 'opencv-python',
        'matplotlib': 'matplotlib',
    }

    all_imports_ok = True

    for module_name, package_name in required_packages.items():
        try:
            __import__(module_name)
            print(f"✓ {package_name:20s} - OK")
        except ImportError as e:
            print(f"✗ {package_name:20s} - MISSING")
            print(f"  Install with: pip install {package_name}")
            all_imports_ok = False

    # Check for Box2D (required for CarRacing)
    try:
        import Box2D
        print(f"✓ {'Box2D':20s} - OK")
    except ImportError:
        print(f"✗ {'Box2D':20s} - MISSING")
        print(f"  Install with: pip install gymnasium[box2d]")
        all_imports_ok = False

    if not all_imports_ok:
        print("\n❌ Some packages are missing. Please install them before proceeding.")
        return False

    print("\n✅ All required packages are installed!")
    return True


def test_versions():
    """Print version information for key packages."""
    print("\n" + "=" * 60)
    print("Package versions:")
    print("=" * 60)

    import gymnasium
    import torch
    import numpy

    print(f"Gymnasium: {gymnasium.__version__}")
    print(f"PyTorch:   {torch.__version__}")
    print(f"NumPy:     {numpy.__version__}")


def test_device():
    """Test PyTorch device availability."""
    print("\n" + "=" * 60)
    print("Testing PyTorch device:")
    print("=" * 60)

    import torch

    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  Device: {torch.cuda.get_device_name(0)}")

    print(f"MPS available:  {torch.backends.mps.is_available()}")
    if torch.backends.mps.is_available():
        print(f"  Device: Apple Silicon GPU")

    # Determine which device will be used
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'

    print(f"\n✅ Will use device: {device.upper()}")


def test_environment():
    """Test CarRacing-v3 environment creation and basic functionality."""
    print("\n" + "=" * 60)
    print("Testing CarRacing-v3 environment:")
    print("=" * 60)

    try:
        import gymnasium as gym
        from env.car_racing import CarRacing

        # Create base environment using local CarRacing class
        print("Creating base CarRacing-v3 environment...")
        env = CarRacing(render_mode=None, continuous=True)

        print(f"✓ Environment created successfully")
        print(f"  Observation space: {env.observation_space.shape}")
        print(f"  Action space: {env.action_space.shape}")
        print(f"  Action space bounds: {env.action_space.low} to {env.action_space.high}")

        # Test reset
        print("\nTesting environment reset...")
        obs, info = env.reset()
        print(f"✓ Reset successful")
        print(f"  Observation shape: {obs.shape}")
        print(f"  Observation dtype: {obs.dtype}")

        # Test step with random action
        print("\nTesting environment step...")
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"✓ Step successful")
        print(f"  Reward: {reward}")
        print(f"  Terminated: {terminated}, Truncated: {truncated}")

        env.close()
        print("\n✅ CarRacing-v3 environment works correctly!")
        return True

    except Exception as e:
        print(f"\n❌ Error testing environment: {e}")
        print("  Make sure Box2D is installed: pip install gymnasium[box2d]")
        return False


def test_preprocessing():
    """Test preprocessing pipeline."""
    print("\n" + "=" * 60)
    print("Testing preprocessing pipeline:")
    print("=" * 60)

    try:
        from preprocessing import make_carracing_env, ActionDiscretizer

        # Create preprocessed environment
        print("Creating preprocessed environment...")
        env = make_carracing_env(
            stack_size=4,
            frame_size=(96, 96),
            discretize_actions=True,
            steering_bins=3,
            gas_brake_bins=3,
            render_mode=None
        )

        print(f"✓ Preprocessed environment created")
        print(f"  Observation space: {env.observation_space.shape}")
        print(f"  Expected: (4, 96, 96) for 4 stacked 96×96 frames")
        print(f"  Action space: Discrete({env.action_space.n})")

        # Print action meanings
        print("\n  Discrete actions:")
        # Find the ActionDiscretizer wrapper
        temp_env = env
        while hasattr(temp_env, 'env'):
            if isinstance(temp_env, ActionDiscretizer):
                meanings = temp_env.get_action_meanings()
                for i, meaning in enumerate(meanings):
                    print(f"    {i}: {meaning}")
                break
            temp_env = temp_env.env

        # Test reset and step
        print("\nTesting preprocessed environment...")
        obs, info = env.reset()
        print(f"✓ Reset successful")
        print(f"  Observation shape: {obs.shape}")
        print(f"  Observation dtype: {obs.dtype}")
        print(f"  Observation range: [{obs.min():.3f}, {obs.max():.3f}]")

        # Test a few steps
        for i in range(3):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)

        print(f"✓ Stepping works correctly")

        env.close()
        print("\n✅ Preprocessing pipeline works correctly!")
        return True

    except Exception as e:
        print(f"\n❌ Error testing preprocessing: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_agent():
    """Test DDQN agent initialization."""
    print("\n" + "=" * 60)
    print("Testing DDQN agent:")
    print("=" * 60)

    try:
        from ddqn_agent import DDQNAgent
        import numpy as np

        # Create agent
        print("Creating DDQN agent...")
        agent = DDQNAgent(
            state_shape=(4, 84, 84),
            n_actions=9,
            learning_rate=0.00025,
            gamma=0.99,
            buffer_size=1000,  # Small buffer for testing
            batch_size=32
        )

        print(f"✓ Agent created successfully")
        print(f"  Device: {agent.device}")
        print(f"  Number of actions: {agent.n_actions}")
        print(f"  Epsilon: {agent.epsilon:.3f}")

        # Test action selection
        print("\nTesting action selection...")
        dummy_state = np.random.rand(4, 84, 84).astype(np.float32)
        action = agent.select_action(dummy_state)
        print(f"✓ Action selection works")
        print(f"  Selected action: {action}")

        # Test experience storage
        print("\nTesting experience storage...")
        agent.store_experience(dummy_state, action, 1.0, dummy_state, False)
        print(f"✓ Experience storage works")
        print(f"  Replay buffer size: {len(agent.replay_buffer)}")

        print("\n✅ DDQN agent works correctly!")
        return True

    except Exception as e:
        print(f"\n❌ Error testing agent: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("CarRacing-v3 DDQN Setup Verification")
    print("=" * 60)

    results = {
        'imports': test_imports(),
    }

    if not results['imports']:
        print("\n" + "=" * 60)
        print("SETUP INCOMPLETE")
        print("=" * 60)
        print("Please install missing packages before proceeding.")
        sys.exit(1)

    test_versions()
    test_device()

    results['environment'] = test_environment()
    results['preprocessing'] = test_preprocessing()
    results['agent'] = test_agent()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    all_passed = all(results.values())

    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {test_name.capitalize()}")

    if all_passed:
        print("\n" + "=" * 60)
        print("🎉 ALL TESTS PASSED!")
        print("=" * 60)
        print("Your environment is ready for training.")
        print("\nNext steps:")
        print("  1. Train agent: python train.py")
        print("  2. Watch random agent: python watch_random_agent.py")
        print("  3. Watch trained agent: python watch_agent.py --checkpoint <path>")
    else:
        print("\n" + "=" * 60)
        print("❌ SOME TESTS FAILED")
        print("=" * 60)
        print("Please fix the issues above before training.")
        sys.exit(1)


if __name__ == "__main__":
    main()
