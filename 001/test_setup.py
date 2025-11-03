"""
Test Setup Script

Run this to verify your installation is correct before training.
This script checks:
1. All required packages are installed
2. Gymnasium Atari environment works
3. PyTorch and CUDA availability
4. Basic preprocessing works
"""

import sys


def check_imports():
    """Check if all required packages are installed"""
    print("Checking package installations...")
    packages = {
        'gymnasium': 'Gymnasium (RL environments)',
        'torch': 'PyTorch (deep learning)',
        'numpy': 'NumPy (numerical computing)',
        'matplotlib': 'Matplotlib (plotting)',
        'cv2': 'OpenCV (image processing)',
        'tqdm': 'tqdm (progress bars)',
    }

    failed = []
    for package, description in packages.items():
        try:
            __import__(package)
            print(f"  ✓ {description}")
        except ImportError:
            print(f"  ✗ {description} - NOT FOUND")
            failed.append(package)

    if failed:
        print(f"\n❌ Missing packages: {', '.join(failed)}")
        print("Run: pip install -r requirements.txt")
        return False

    print("\n✅ All packages installed!\n")
    return True


def check_pytorch():
    """Check PyTorch and CUDA/MPS availability"""
    print("Checking PyTorch configuration...")
    import torch

    print(f"  PyTorch version: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")

    # Check for MPS (Apple Silicon)
    has_mps = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
    if has_mps:
        print(f"  MPS (Apple Silicon) available: True")

    if torch.cuda.is_available():
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  GPU device: {torch.cuda.get_device_name(0)}")
        print("\n✅ GPU training available (CUDA)! Training will be much faster.\n")
    elif has_mps:
        print("\n✅ GPU training available (Apple Silicon MPS)! Training will be accelerated.\n")
    else:
        print("\n⚠️  GPU not available. Training will be slow on CPU.")
        print("   Consider using Google Colab for free GPU access.\n")

    return True


def check_atari_env():
    """Check if Atari environment can be created"""
    print("Checking Atari environment...")
    try:
        import gymnasium as gym
        import ale_py
        gym.register_envs(ale_py)
        env = gym.make('ALE/Breakout-v5')
        env.reset()
        env.close()
        print("  ✓ Environment created successfully")
        print(f"  ✓ Observation space: {env.observation_space.shape}")
        print(f"  ✓ Action space: {env.action_space.n} actions")
        print("\n✅ Atari environment works!\n")
        return True
    except Exception as e:
        print(f"\n❌ Error creating environment: {e}")
        print("\nTry: pip install gymnasium[accept-rom-license]")
        return False


def check_preprocessing():
    """Check if preprocessing works"""
    print("Checking preprocessing pipeline...")
    try:
        from preprocessing import make_atari_env
        env = make_atari_env('ALE/Breakout-v5')
        state, _ = env.reset()
        print(f"  ✓ Preprocessed state shape: {state.shape}")
        print(f"  ✓ Expected: (4, 84, 84)")

        if state.shape == (4, 84, 84):
            print("\n✅ Preprocessing works!\n")
            env.close()
            return True
        else:
            print(f"\n❌ Unexpected shape: {state.shape}")
            return False
    except Exception as e:
        print(f"\n❌ Error in preprocessing: {e}")
        return False


def check_dqn():
    """Check if DQN agent can be created"""
    print("Checking DQN agent...")
    try:
        from dqn_agent import DQNAgent
        import torch

        agent = DQNAgent(
            state_shape=(4, 84, 84),
            num_actions=4
        )

        # Test forward pass
        dummy_state = torch.randn(1, 4, 84, 84).to(agent.device)
        with torch.no_grad():
            output = agent.policy_net(dummy_state)

        print(f"  ✓ Agent created successfully")
        print(f"  ✓ Network output shape: {output.shape}")
        print(f"  ✓ Device: {agent.device}")
        print("\n✅ DQN agent works!\n")
        return True
    except Exception as e:
        print(f"\n❌ Error creating agent: {e}")
        return False


def main():
    """Run all checks"""
    print("="*60)
    print("DQN SETUP VERIFICATION")
    print("="*60)
    print()

    checks = [
        ("Package Installation", check_imports),
        ("PyTorch Configuration", check_pytorch),
        ("Atari Environment", check_atari_env),
        ("Preprocessing Pipeline", check_preprocessing),
        ("DQN Agent", check_dqn),
    ]

    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append(result)
        except Exception as e:
            print(f"\n❌ Unexpected error in {name}: {e}\n")
            results.append(False)

    print("="*60)
    print("SUMMARY")
    print("="*60)

    for (name, _), result in zip(checks, results):
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {name}")

    print("="*60)

    if all(results):
        print("\n🎉 All checks passed! You're ready to start training!")
        print("\nNext steps:")
        print("  1. python visualize_preprocessing.py  # Understand preprocessing")
        print("  2. python train.py                    # Start training")
        print("  3. python watch_agent.py --checkpoint checkpoints/final_model.pt")
    else:
        print("\n⚠️  Some checks failed. Please fix the issues above.")
        sys.exit(1)


if __name__ == '__main__':
    main()
