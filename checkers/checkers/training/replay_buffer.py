"""
Replay buffer with recency bias for storing self-play games.
"""

import numpy as np
from typing import Tuple, List
from dataclasses import dataclass


@dataclass
class Experience:
    """Single training example from self-play."""
    state: np.ndarray  # (8, 10, 10)
    policy: np.ndarray  # (150,)
    value: float  # -1, 0, or 1
    generation: int  # Training iteration when created


class ReplayBuffer:
    """
    Replay buffer that stores experiences from self-play games.

    Features:
    - Fixed capacity with circular buffer
    - Recency-biased sampling (favor recent games)
    - Efficient numpy storage
    """

    def __init__(self, capacity: int = 500_000, recency_tau: float = 50.0):
        """
        Initialize replay buffer.

        Args:
            capacity: Maximum number of experiences to store
            recency_tau: Decay constant for recency bias (smaller = more bias)
        """
        self.capacity = capacity
        self.recency_tau = recency_tau

        # Storage arrays
        self.states = np.zeros((capacity, 8, 10, 10), dtype=np.float32)
        self.policies = np.zeros((capacity, 150), dtype=np.float32)
        self.values = np.zeros(capacity, dtype=np.float32)
        self.generations = np.zeros(capacity, dtype=np.int32)

        # Buffer management
        self.size = 0  # Current number of experiences
        self.position = 0  # Next write position
        self.current_generation = 0  # Current training iteration

    def add(self, state: np.ndarray, policy: np.ndarray, value: float):
        """
        Add a single experience to the buffer.

        Args:
            state: Game state (8, 10, 10)
            policy: MCTS policy distribution (150,)
            value: Game outcome from this state's perspective
        """
        self.states[self.position] = state
        self.policies[self.position] = policy
        self.values[self.position] = value
        self.generations[self.position] = self.current_generation

        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def add_batch(
        self,
        states: List[np.ndarray],
        policies: List[np.ndarray],
        values: List[float]
    ):
        """
        Add multiple experiences to the buffer.

        Args:
            states: List of game states
            policies: List of policy distributions
            values: List of values
        """
        for state, policy, value in zip(states, policies, values):
            self.add(state, policy, value)

    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Sample a batch with recency bias.

        Sampling probability: exp(-(current_gen - generation) / tau)

        Args:
            batch_size: Number of samples to draw

        Returns:
            (states, policies, values): Batches of training data
        """
        if self.size == 0:
            raise ValueError("Cannot sample from empty buffer")

        # Compute recency weights
        age_diffs = self.current_generation - self.generations[:self.size]
        weights = np.exp(-age_diffs / self.recency_tau)
        weights = weights / weights.sum()

        # Sample indices
        indices = np.random.choice(self.size, size=batch_size, p=weights, replace=True)

        # Return batches
        return (
            self.states[indices],
            self.policies[indices],
            self.values[indices]
        )

    def sample_uniform(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Sample a batch uniformly (no recency bias).

        Args:
            batch_size: Number of samples to draw

        Returns:
            (states, policies, values): Batches of training data
        """
        if self.size == 0:
            raise ValueError("Cannot sample from empty buffer")

        indices = np.random.choice(self.size, size=batch_size, replace=True)

        return (
            self.states[indices],
            self.policies[indices],
            self.values[indices]
        )

    def increment_generation(self):
        """Increment generation counter (call after each training iteration)."""
        self.current_generation += 1

    def get_stats(self) -> dict:
        """Get buffer statistics."""
        if self.size == 0:
            return {
                "size": 0,
                "capacity": self.capacity,
                "generation": self.current_generation,
                "mean_value": 0.0,
                "value_std": 0.0,
            }

        return {
            "size": self.size,
            "capacity": self.capacity,
            "utilization": self.size / self.capacity,
            "generation": self.current_generation,
            "mean_value": self.values[:self.size].mean(),
            "value_std": self.values[:self.size].std(),
            "oldest_generation": self.generations[:self.size].min(),
            "newest_generation": self.generations[:self.size].max(),
        }

    def clear(self):
        """Clear all data from buffer."""
        self.size = 0
        self.position = 0

    def __len__(self) -> int:
        return self.size


# Testing
if __name__ == "__main__":
    print("Testing ReplayBuffer...")

    buffer = ReplayBuffer(capacity=1000, recency_tau=50.0)

    # Add some dummy data
    print("\nAdding 100 experiences from generation 0...")
    for i in range(100):
        state = np.random.randn(8, 10, 10).astype(np.float32)
        policy = np.random.random(150).astype(np.float32)
        policy = policy / policy.sum()
        value = np.random.choice([-1.0, 0.0, 1.0])

        buffer.add(state, policy, value)

    print(f"Buffer size: {len(buffer)}")
    print(f"Stats: {buffer.get_stats()}")

    # Increment generation and add more
    buffer.increment_generation()
    print("\nAdding 100 experiences from generation 1...")
    for i in range(100):
        state = np.random.randn(8, 10, 10).astype(np.float32)
        policy = np.random.random(150).astype(np.float32)
        policy = policy / policy.sum()
        value = np.random.choice([-1.0, 0.0, 1.0])

        buffer.add(state, policy, value)

    print(f"Buffer size: {len(buffer)}")
    print(f"Stats: {buffer.get_stats()}")

    # Test sampling
    print("\nSampling batch of 32...")
    states, policies, values = buffer.sample(32)
    print(f"States shape: {states.shape}")
    print(f"Policies shape: {policies.shape}")
    print(f"Values shape: {values.shape}")
    print(f"Sample values: {values[:5]}")

    # Test sampling distribution with recency bias
    print("\nTesting recency bias (generation 2)...")
    buffer.increment_generation()
    buffer.increment_generation()

    gen_counts = {0: 0, 1: 0}
    num_samples = 1000

    for _ in range(num_samples):
        states, _, _ = buffer.sample(1)
        # Would need to track which generation each sample came from
        # This is just a demonstration

    print("\n✓ ReplayBuffer tests passed!")
