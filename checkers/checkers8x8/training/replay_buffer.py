"""
Replay buffer for storing training examples with recency weighting.
"""

import numpy as np
from typing import Tuple


class ReplayBuffer:
    """
    Circular replay buffer with recency-weighted sampling.

    Stores (state, policy, value) tuples from self-play.
    """

    def __init__(self, capacity: int, recency_tau: float = 50.0):
        """
        Initialize replay buffer.

        Args:
            capacity: Maximum number of positions to store
            recency_tau: Temperature for recency weighting
        """
        self.capacity = capacity
        self.recency_tau = recency_tau

        # Storage arrays
        self.states = np.zeros((capacity, 8, 8, 8), dtype=np.float32)
        self.policies = np.zeros((capacity, 128), dtype=np.float32)
        self.values = np.zeros(capacity, dtype=np.float32)
        self.generations = np.zeros(capacity, dtype=np.int32)

        self.size = 0
        self.position = 0
        self.current_generation = 0

    def add(self, state: np.ndarray, policy: np.ndarray, value: float):
        """Add a single training example."""
        self.states[self.position] = state
        self.policies[self.position] = policy
        self.values[self.position] = value
        self.generations[self.position] = self.current_generation

        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def add_batch(self, states: list, policies: list, values: list):
        """Add a batch of training examples."""
        for state, policy, value in zip(states, policies, values):
            self.add(state, policy, value)

    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Sample a batch with recency weighting.

        More recent generations are sampled with higher probability.

        Args:
            batch_size: Number of samples

        Returns:
            (states, policies, values) arrays
        """
        if self.size == 0:
            raise ValueError("Cannot sample from empty buffer")

        # Calculate sampling weights (exponential recency)
        if self.recency_tau > 0:
            age_diffs = self.current_generation - self.generations[:self.size]
            weights = np.exp(-age_diffs / self.recency_tau)
            weights = weights / weights.sum()
        else:
            # Uniform sampling
            weights = np.ones(self.size) / self.size

        # Sample indices
        indices = np.random.choice(self.size, size=batch_size, p=weights, replace=True)

        return (
            self.states[indices],
            self.policies[indices],
            self.values[indices]
        )

    def increment_generation(self):
        """Increment the generation counter."""
        self.current_generation += 1

    def state_dict(self) -> dict:
        """
        Get buffer state for checkpointing.

        Returns:
            Dictionary containing buffer state
        """
        return {
            'states': self.states[:self.size].copy(),
            'policies': self.policies[:self.size].copy(),
            'values': self.values[:self.size].copy(),
            'generations': self.generations[:self.size].copy(),
            'size': self.size,
            'position': self.position,
            'current_generation': self.current_generation,
            'capacity': self.capacity,
            'recency_tau': self.recency_tau,
        }

    def load_state_dict(self, state_dict: dict):
        """
        Restore buffer state from checkpoint.

        Args:
            state_dict: Dictionary from state_dict()
        """
        # Verify capacity matches
        if state_dict['capacity'] != self.capacity:
            print(f"Warning: Checkpoint capacity ({state_dict['capacity']}) "
                  f"differs from current ({self.capacity}). May lose data.")

        # Restore metadata
        self.size = state_dict['size']
        self.position = state_dict['position']
        self.current_generation = state_dict['current_generation']
        self.recency_tau = state_dict.get('recency_tau', self.recency_tau)

        # Restore data
        restored_size = min(self.size, self.capacity)
        self.states[:restored_size] = state_dict['states'][:restored_size]
        self.policies[:restored_size] = state_dict['policies'][:restored_size]
        self.values[:restored_size] = state_dict['values'][:restored_size]
        self.generations[:restored_size] = state_dict['generations'][:restored_size]

    def __len__(self) -> int:
        """Get current buffer size."""
        return self.size


# Testing
if __name__ == "__main__":
    print("Testing Replay Buffer")
    print("=" * 60)

    # Create buffer
    buffer = ReplayBuffer(capacity=1000, recency_tau=50.0)

    print(f"Buffer created with capacity {buffer.capacity}")
    print(f"Initial size: {len(buffer)}")

    # Add some examples
    for i in range(100):
        state = np.random.randn(8, 8, 8).astype(np.float32)
        policy = np.random.rand(128).astype(np.float32)
        policy = policy / policy.sum()  # Normalize
        value = np.random.rand() * 2 - 1  # Random value in [-1, 1]

        buffer.add(state, policy, value)

    print(f"After adding 100 examples: size = {len(buffer)}")

    # Sample a batch
    states, policies, values = buffer.sample(batch_size=32)

    print(f"\nSampled batch:")
    print(f"  States shape: {states.shape}")
    print(f"  Policies shape: {policies.shape}")
    print(f"  Values shape: {values.shape}")

    # Test generation increment
    buffer.increment_generation()
    print(f"\nGeneration incremented to {buffer.current_generation}")

    # Add more examples in new generation
    for i in range(50):
        state = np.random.randn(8, 8, 8).astype(np.float32)
        policy = np.random.rand(128).astype(np.float32)
        policy = policy / policy.sum()
        value = np.random.rand() * 2 - 1

        buffer.add(state, policy, value)

    print(f"After adding 50 more examples: size = {len(buffer)}")

    # Sample should favor newer generation
    states2, policies2, values2 = buffer.sample(batch_size=32)
    print(f"Sampled another batch (should favor recent examples)")

    print("\n✓ Replay buffer tests passed!")
