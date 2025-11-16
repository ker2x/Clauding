"""
Simple test script for domain randomization.

Tests that:
1. Domain randomization can be enabled/disabled
2. Parameters are randomized correctly
3. Environment runs without errors
4. Preset configurations work
"""

import numpy as np
from env.car_racing import CarRacing
from config.domain_randomization import (
    DomainRandomizationConfig,
    conservative_randomization,
    moderate_randomization,
    aggressive_randomization,
    wet_surface_conditions,
    VehicleRandomization,
    TireRandomization,
)


def test_disabled_randomization():
    """Test that randomization can be disabled."""
    print("Testing disabled randomization...", end=" ")

    env = CarRacing(
        render_mode=None,
        domain_randomization_config=DomainRandomizationConfig(enabled=False)
    )

    obs, info = env.reset()
    assert obs.shape == (71,), f"Wrong observation shape: {obs.shape}"

    # Take a step
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    env.close()
    print("✓ PASSED")


def test_enabled_randomization():
    """Test that randomization works when enabled."""
    print("Testing enabled randomization...", end=" ")

    config = DomainRandomizationConfig(
        enabled=True,
        vehicle=VehicleRandomization(
            mass_range=(0.9, 1.1),
        ),
        tire=TireRandomization(
            pacejka_d_lat_range=(0.9, 1.1),
        ),
    )

    env = CarRacing(
        render_mode=None,
        domain_randomization_config=config
    )

    # Reset multiple times and check randomization varies
    masses = []
    for _ in range(10):
        obs, info = env.reset()
        rand_info = env.domain_randomizer.get_info_dict()
        masses.append(rand_info['mass'])

    # Check that we got different values
    unique_masses = len(set(masses))
    assert unique_masses > 1, f"Mass not randomizing (got {unique_masses} unique values)"

    # Check that values are in expected range
    base_mass = 1062.0
    assert all(0.9 * base_mass <= m <= 1.1 * base_mass for m in masses), \
        f"Mass outside expected range: {masses}"

    env.close()
    print("✓ PASSED")


def test_preset_conservative():
    """Test conservative preset configuration."""
    print("Testing conservative preset...", end=" ")

    config = conservative_randomization()
    env = CarRacing(
        render_mode=None,
        domain_randomization_config=config
    )

    obs, info = env.reset()
    rand_info = env.domain_randomizer.get_info_dict()

    # Check key parameters are present
    assert 'mass' in rand_info
    assert 'pacejka_d_lat' in rand_info
    assert 'pacejka_d_lon' in rand_info

    env.close()
    print("✓ PASSED")


def test_preset_moderate():
    """Test moderate preset configuration."""
    print("Testing moderate preset...", end=" ")

    config = moderate_randomization()
    env = CarRacing(
        render_mode=None,
        domain_randomization_config=config
    )

    obs, info = env.reset()
    rand_info = env.domain_randomizer.get_info_dict()

    assert 'mass' in rand_info
    assert 'engine_power' in rand_info

    env.close()
    print("✓ PASSED")


def test_preset_aggressive():
    """Test aggressive preset configuration."""
    print("Testing aggressive preset...", end=" ")

    config = aggressive_randomization()
    env = CarRacing(
        render_mode=None,
        domain_randomization_config=config
    )

    obs, info = env.reset()

    # Take some steps to ensure physics work with aggressive randomization
    for _ in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break

    env.close()
    print("✓ PASSED")


def test_wet_conditions():
    """Test wet surface conditions preset."""
    print("Testing wet surface conditions...", end=" ")

    config = wet_surface_conditions()
    env = CarRacing(
        render_mode=None,
        domain_randomization_config=config
    )

    obs, info = env.reset()
    rand_info = env.domain_randomizer.get_info_dict()

    # Wet conditions should have reduced grip
    base_lat_grip = 0.95  # Base lateral grip
    assert rand_info['pacejka_d_lat'] < base_lat_grip, \
        f"Wet surface should have reduced grip, got {rand_info['pacejka_d_lat']}"

    env.close()
    print("✓ PASSED")


def test_parameter_ranges():
    """Test that randomization respects specified ranges."""
    print("Testing parameter ranges...", end=" ")

    config = DomainRandomizationConfig(
        enabled=True,
        vehicle=VehicleRandomization(
            mass_range=(0.8, 1.2),  # ±20%
        ),
        tire=TireRandomization(
            pacejka_d_lat_range=(0.7, 1.3),  # ±30%
        ),
    )

    env = CarRacing(
        render_mode=None,
        domain_randomization_config=config
    )

    base_mass = 1062.0
    base_d_lat = 0.95

    # Sample many times
    masses = []
    d_lats = []

    for _ in range(50):
        obs, info = env.reset()
        rand_info = env.domain_randomizer.get_info_dict()
        masses.append(rand_info['mass'])
        d_lats.append(rand_info['pacejka_d_lat'])

    # Check ranges
    assert all(0.8 * base_mass <= m <= 1.2 * base_mass for m in masses), \
        f"Mass outside range: min={min(masses)}, max={max(masses)}"

    assert all(0.7 * base_d_lat <= d <= 1.3 * base_d_lat for d in d_lats), \
        f"D_lat outside range: min={min(d_lats)}, max={max(d_lats)}"

    env.close()
    print("✓ PASSED")


def test_multi_episode():
    """Test that randomization works across multiple episodes."""
    print("Testing multi-episode randomization...", end=" ")

    config = moderate_randomization()
    env = CarRacing(
        render_mode=None,
        domain_randomization_config=config
    )

    # Run multiple episodes
    for episode in range(5):
        obs, info = env.reset()

        # Take a few steps
        for step in range(20):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break

    env.close()
    print("✓ PASSED")


def test_seed_reproducibility():
    """Test that setting seed makes randomization reproducible."""
    print("Testing seed reproducibility...", end=" ")

    # Create two environments with same seed
    config1 = DomainRandomizationConfig(
        enabled=True,
        seed=42,
        vehicle=VehicleRandomization(mass_range=(0.9, 1.1)),
    )
    config2 = DomainRandomizationConfig(
        enabled=True,
        seed=42,
        vehicle=VehicleRandomization(mass_range=(0.9, 1.1)),
    )

    env1 = CarRacing(render_mode=None, domain_randomization_config=config1)
    env2 = CarRacing(render_mode=None, domain_randomization_config=config2)

    # Reset both and check they get same randomization
    env1.reset()
    env2.reset()

    info1 = env1.domain_randomizer.get_info_dict()
    info2 = env2.domain_randomizer.get_info_dict()

    assert abs(info1['mass'] - info2['mass']) < 1e-6, \
        f"Same seed should give same randomization: {info1['mass']} vs {info2['mass']}"

    env1.close()
    env2.close()
    print("✓ PASSED")


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*70)
    print("Domain Randomization Tests")
    print("="*70 + "\n")

    tests = [
        test_disabled_randomization,
        test_enabled_randomization,
        test_preset_conservative,
        test_preset_moderate,
        test_preset_aggressive,
        test_wet_conditions,
        test_parameter_ranges,
        test_multi_episode,
        test_seed_reproducibility,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"✗ FAILED: {e}")
            failed += 1
            import traceback
            traceback.print_exc()

    print("\n" + "="*70)
    print(f"Results: {passed} passed, {failed} failed")
    print("="*70 + "\n")

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
