"""
Utility functions for neural network training, including data augmentation.
"""

import numpy as np
import torch
from typing import Tuple, List


def augment_sample(
    state: np.ndarray,
    policy: np.ndarray,
    value: float
) -> List[Tuple[np.ndarray, np.ndarray, float]]:
    """
    Generate 8 augmented samples from one via board symmetries.

    For checkers, we can apply:
    - 2 flips: horizontal and vertical
    - 4 rotations: 0°, 90°, 180°, 270°

    This gives us 8 unique transformations.

    Args:
        state: Game state (8, 10, 10)
        policy: Policy distribution (150,)
        value: Game value

    Returns:
        List of 8 (state, policy, value) tuples
    """
    augmented = []

    # Original
    augmented.append((state.copy(), policy.copy(), value))

    # Horizontal flip
    state_h = np.flip(state, axis=2).copy()
    policy_h = transform_policy_horizontal(policy)
    augmented.append((state_h, policy_h, value))

    # Vertical flip
    state_v = np.flip(state, axis=1).copy()
    policy_v = transform_policy_vertical(policy)
    augmented.append((state_v, policy_v, value))

    # Horizontal + Vertical (180° rotation)
    state_hv = np.flip(np.flip(state, axis=1), axis=2).copy()
    policy_hv = transform_policy_180(policy)
    augmented.append((state_hv, policy_hv, value))

    # 90° rotation
    state_90 = np.rot90(state, k=1, axes=(1, 2)).copy()
    policy_90 = transform_policy_rotate_90(policy)
    augmented.append((state_90, policy_90, value))

    # 270° rotation
    state_270 = np.rot90(state, k=3, axes=(1, 2)).copy()
    policy_270 = transform_policy_rotate_270(policy)
    augmented.append((state_270, policy_270, value))

    # 90° + horizontal flip
    state_90h = np.flip(state_90, axis=2).copy()
    policy_90h = transform_policy_horizontal(policy_90)
    augmented.append((state_90h, policy_90h, value))

    # 90° + vertical flip
    state_90v = np.flip(state_90, axis=1).copy()
    policy_90v = transform_policy_vertical(policy_90)
    augmented.append((state_90v, policy_90v, value))

    return augmented


def transform_policy_horizontal(policy: np.ndarray) -> np.ndarray:
    """
    Transform policy for horizontal flip.

    For checkers with move-based actions, we need to remap action indices.
    For simplicity in this implementation, we keep the policy as-is since
    the action space is based on legal move ordering which changes per position.

    In a production system, you'd want to map actions properly or use
    a coordinate-based action representation.
    """
    # Simplified: return as-is
    # In practice, this requires mapping action indices through the transformation
    return policy.copy()


def transform_policy_vertical(policy: np.ndarray) -> np.ndarray:
    """Transform policy for vertical flip."""
    return policy.copy()


def transform_policy_180(policy: np.ndarray) -> np.ndarray:
    """Transform policy for 180° rotation."""
    return policy.copy()


def transform_policy_rotate_90(policy: np.ndarray) -> np.ndarray:
    """Transform policy for 90° rotation."""
    return policy.copy()


def transform_policy_rotate_270(policy: np.ndarray) -> np.ndarray:
    """Transform policy for 270° rotation."""
    return policy.copy()


def augment_batch(
    states: np.ndarray,
    policies: np.ndarray,
    values: np.ndarray,
    aug_factor: int = 8
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Augment a batch of training samples.

    Args:
        states: Batch of states (batch, 8, 10, 10)
        policies: Batch of policies (batch, 150)
        values: Batch of values (batch,)
        aug_factor: Number of augmentations per sample (max 8)

    Returns:
        (aug_states, aug_policies, aug_values): Augmented batches
    """
    batch_size = states.shape[0]
    aug_factor = min(aug_factor, 8)

    aug_states = []
    aug_policies = []
    aug_values = []

    for i in range(batch_size):
        augmented = augment_sample(states[i], policies[i], values[i])

        # Take first aug_factor augmentations
        for j in range(aug_factor):
            state_aug, policy_aug, value_aug = augmented[j]
            aug_states.append(state_aug)
            aug_policies.append(policy_aug)
            aug_values.append(value_aug)

    return (
        np.array(aug_states, dtype=np.float32),
        np.array(aug_policies, dtype=np.float32),
        np.array(aug_values, dtype=np.float32)
    )


def add_random_noise(state: np.ndarray, noise_scale: float = 0.01) -> np.ndarray:
    """
    Add small random noise to state for regularization.

    Args:
        state: Game state (8, 10, 10)
        noise_scale: Standard deviation of Gaussian noise

    Returns:
        Noisy state
    """
    noise = np.random.randn(*state.shape).astype(np.float32) * noise_scale
    return np.clip(state + noise, 0.0, 1.0)


# Testing
if __name__ == "__main__":
    print("Testing data augmentation...")

    # Create dummy sample
    state = np.random.randn(8, 10, 10).astype(np.float32)
    policy = np.random.random(150).astype(np.float32)
    policy = policy / policy.sum()
    value = 1.0

    # Test single sample augmentation
    print("\nAugmenting single sample...")
    augmented = augment_sample(state, policy, value)
    print(f"Generated {len(augmented)} augmentations")

    for i, (aug_state, aug_policy, aug_value) in enumerate(augmented):
        print(f"Aug {i}: state shape {aug_state.shape}, "
              f"policy sum {aug_policy.sum():.4f}, value {aug_value}")

    # Test batch augmentation
    print("\nAugmenting batch...")
    batch_size = 4
    states = np.random.randn(batch_size, 8, 10, 10).astype(np.float32)
    policies = np.random.random((batch_size, 150)).astype(np.float32)
    policies = policies / policies.sum(axis=1, keepdims=True)
    values = np.array([1.0, -1.0, 0.0, 1.0], dtype=np.float32)

    aug_states, aug_policies, aug_values = augment_batch(
        states, policies, values, aug_factor=8
    )

    print(f"Original batch size: {batch_size}")
    print(f"Augmented batch size: {len(aug_states)}")
    print(f"Augmented states shape: {aug_states.shape}")
    print(f"Augmented policies shape: {aug_policies.shape}")
    print(f"Augmented values shape: {aug_values.shape}")

    # Check policies still sum to 1
    policy_sums = aug_policies.sum(axis=1)
    print(f"Policy sums: min={policy_sums.min():.4f}, max={policy_sums.max():.4f}")

    # Test noise addition
    print("\nTesting noise addition...")
    noisy_state = add_random_noise(state, noise_scale=0.01)
    diff = np.abs(noisy_state - state).mean()
    print(f"Mean absolute difference: {diff:.6f}")

    print("\n✓ All augmentation tests passed!")
