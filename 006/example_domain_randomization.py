"""
Domain Randomization Example for CarRacing Environment.

This script demonstrates how to use domain randomization with the
CarRacing environment to improve policy robustness and generalization.

Run this script to see domain randomization in action:
    python example_domain_randomization.py
"""

import numpy as np
from env.car_racing import CarRacing
from config.domain_randomization import (
    DomainRandomizationConfig,
    conservative_randomization,
    moderate_randomization,
    aggressive_randomization,
    wet_surface_conditions,
)


def example_1_basic_usage():
    """Example 1: Basic domain randomization usage."""
    print("\n" + "="*70)
    print("Example 1: Basic Domain Randomization")
    print("="*70)

    # Create environment with conservative randomization
    config = conservative_randomization()
    env = CarRacing(
        render_mode=None,
        verbose=True,  # Enable verbose to see randomization info
        domain_randomization_config=config
    )

    # Run a few episodes to see randomization
    for episode in range(3):
        print(f"\n--- Episode {episode + 1} ---")
        obs, info = env.reset()
        print(f"Initial observation shape: {obs.shape}")

        # Take a few random steps
        for step in range(10):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break

    env.close()
    print("\nExample 1 complete!")


def example_2_custom_config():
    """Example 2: Custom domain randomization configuration."""
    print("\n" + "="*70)
    print("Example 2: Custom Domain Randomization Config")
    print("="*70)

    from config.domain_randomization import (
        VehicleRandomization,
        TireRandomization,
        TrackRandomization,
    )

    # Create custom configuration
    config = DomainRandomizationConfig(
        enabled=True,
        vehicle=VehicleRandomization(
            mass_range=(0.9, 1.1),  # ±10% mass variation
            cg_height_range=(0.95, 1.05),  # ±5% CG height
        ),
        tire=TireRandomization(
            pacejka_d_lat_range=(0.85, 1.15),  # ±15% lateral grip
            pacejka_d_lon_range=(0.85, 1.15),  # ±15% longitudinal grip
        ),
        track=TrackRandomization(
            surface_friction_range=(0.9, 1.1),  # ±10% surface friction
        ),
    )

    env = CarRacing(
        render_mode=None,
        verbose=True,
        domain_randomization_config=config
    )

    print("\nRunning with custom randomization config...")
    obs, info = env.reset()
    print(f"Observation shape: {obs.shape}")

    env.close()
    print("\nExample 2 complete!")


def example_3_preset_configs():
    """Example 3: Using preset configurations."""
    print("\n" + "="*70)
    print("Example 3: Preset Configurations")
    print("="*70)

    presets = {
        "Conservative": conservative_randomization(),
        "Moderate": moderate_randomization(),
        "Aggressive": aggressive_randomization(),
        "Wet Surface": wet_surface_conditions(),
    }

    for name, config in presets.items():
        print(f"\n--- Testing {name} Preset ---")
        env = CarRacing(
            render_mode=None,
            verbose=True,
            domain_randomization_config=config
        )

        obs, info = env.reset()
        print(f"Reset complete with {name} randomization")

        env.close()

    print("\nExample 3 complete!")


def example_4_training_integration():
    """Example 4: How to use with training scripts."""
    print("\n" + "="*70)
    print("Example 4: Training Integration")
    print("="*70)

    print("""
Domain randomization can be integrated into training scripts as follows:

1. In train.py or train_selection_parallel.py:

    from config.domain_randomization import moderate_randomization

    # Create environment with domain randomization
    domain_rand_config = moderate_randomization()
    env = CarRacing(
        state_mode="vector",
        max_episode_steps=2500,
        domain_randomization_config=domain_rand_config,
    )

2. For curriculum learning (start easy, increase difficulty):

    from config.domain_randomization import (
        conservative_randomization,
        moderate_randomization,
        aggressive_randomization,
    )

    # Start with conservative
    if episode < 100:
        config = conservative_randomization()
    elif episode < 500:
        config = moderate_randomization()
    else:
        config = aggressive_randomization()

    env = CarRacing(domain_randomization_config=config)

3. For specific conditions (e.g., wet weather training):

    from config.domain_randomization import wet_surface_conditions

    # Train specifically on wet surfaces
    config = wet_surface_conditions()
    env = CarRacing(domain_randomization_config=config)

4. Disable randomization for evaluation:

    # Training environment (with randomization)
    train_env = CarRacing(
        domain_randomization_config=moderate_randomization()
    )

    # Evaluation environment (no randomization)
    eval_env = CarRacing(
        domain_randomization_config=DomainRandomizationConfig(enabled=False)
    )
    """)

    print("Example 4 complete!")


def example_5_statistics():
    """Example 5: Collect statistics on randomization."""
    print("\n" + "="*70)
    print("Example 5: Randomization Statistics")
    print("="*70)

    config = moderate_randomization()
    env = CarRacing(
        render_mode=None,
        verbose=False,  # Disable verbose for cleaner output
        domain_randomization_config=config
    )

    # Collect randomization statistics
    num_episodes = 20
    mass_values = []
    grip_lateral = []
    grip_longitudinal = []

    print(f"\nCollecting statistics from {num_episodes} episodes...")

    for episode in range(num_episodes):
        env.reset()
        info = env.domain_randomizer.get_info_dict()
        mass_values.append(info['mass'])
        grip_lateral.append(info['pacejka_d_lat'])
        grip_longitudinal.append(info['pacejka_d_lon'])

    # Print statistics
    print(f"\nRandomization Statistics (n={num_episodes}):")
    print(f"  Mass:")
    print(f"    Mean: {np.mean(mass_values):.1f} kg")
    print(f"    Std:  {np.std(mass_values):.1f} kg")
    print(f"    Min:  {np.min(mass_values):.1f} kg")
    print(f"    Max:  {np.max(mass_values):.1f} kg")

    print(f"\n  Lateral Grip (D_lat):")
    print(f"    Mean: {np.mean(grip_lateral):.3f}")
    print(f"    Std:  {np.std(grip_lateral):.3f}")
    print(f"    Min:  {np.min(grip_lateral):.3f}")
    print(f"    Max:  {np.max(grip_lateral):.3f}")

    print(f"\n  Longitudinal Grip (D_lon):")
    print(f"    Mean: {np.mean(grip_longitudinal):.3f}")
    print(f"    Std:  {np.std(grip_longitudinal):.3f}")
    print(f"    Min:  {np.min(grip_longitudinal):.3f}")
    print(f"    Max:  {np.max(grip_longitudinal):.3f}")

    env.close()
    print("\nExample 5 complete!")


def main():
    """Run all examples."""
    print("\n" + "="*70)
    print("Domain Randomization Examples for CarRacing")
    print("="*70)

    examples = [
        ("Basic Usage", example_1_basic_usage),
        ("Custom Config", example_2_custom_config),
        ("Preset Configs", example_3_preset_configs),
        ("Training Integration", example_4_training_integration),
        ("Statistics", example_5_statistics),
    ]

    print("\nAvailable examples:")
    for i, (name, _) in enumerate(examples, 1):
        print(f"  {i}. {name}")
    print(f"  {len(examples) + 1}. Run all examples")

    try:
        choice = input("\nSelect example to run (1-6, or press Enter for all): ").strip()

        if choice == "" or choice == str(len(examples) + 1):
            # Run all examples
            for name, func in examples:
                func()
        elif choice.isdigit() and 1 <= int(choice) <= len(examples):
            # Run selected example
            idx = int(choice) - 1
            name, func = examples[idx]
            func()
        else:
            print("Invalid choice. Running all examples...")
            for name, func in examples:
                func()

    except KeyboardInterrupt:
        print("\n\nExamples interrupted by user.")
    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "="*70)
    print("Examples complete!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
