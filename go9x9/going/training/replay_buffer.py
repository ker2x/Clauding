"""
Replay buffer for storing training examples with recency weighting.
"""

import numpy as np
from typing import Tuple


class ReplayBuffer:
    """Circular replay buffer with recency-weighted sampling."""

    def __init__(self, capacity: int, recency_tau: float = 50.0):
        self.capacity = capacity
        self.recency_tau = recency_tau

        # Storage arrays (17 planes × 9 × 9 states, 82-dim policies)
        self.states = np.zeros((capacity, 17, 9, 9), dtype=np.float32)
        self.policies = np.zeros((capacity, 82), dtype=np.float32)
        self.values = np.zeros(capacity, dtype=np.float32)
        self.generations = np.zeros(capacity, dtype=np.int32)

        self.size = 0
        self.position = 0
        self.current_generation = 0

    def add(self, state: np.ndarray, policy: np.ndarray, value: float):
        self.states[self.position] = state
        self.policies[self.position] = policy
        self.values[self.position] = value
        self.generations[self.position] = self.current_generation

        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def add_batch(self, states: list, policies: list, values: list):
        for state, policy, value in zip(states, policies, values):
            self.add(state, policy, value)

    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.size == 0:
            raise ValueError("Cannot sample from empty buffer")

        if self.recency_tau > 0:
            age_diffs = self.current_generation - self.generations[:self.size]
            weights = np.exp(-age_diffs / self.recency_tau)
            weights = weights / weights.sum()
        else:
            weights = np.ones(self.size) / self.size

        indices = np.random.choice(self.size, size=batch_size, p=weights, replace=True)

        return (
            self.states[indices],
            self.policies[indices],
            self.values[indices]
        )

    def increment_generation(self):
        self.current_generation += 1

    def state_dict(self) -> dict:
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
        if state_dict['capacity'] != self.capacity:
            print(f"Warning: Checkpoint capacity ({state_dict['capacity']}) "
                  f"differs from current ({self.capacity}).")

        self.size = state_dict['size']
        self.position = state_dict['position']
        self.current_generation = state_dict['current_generation']
        self.recency_tau = state_dict.get('recency_tau', self.recency_tau)

        restored_size = min(self.size, self.capacity)
        self.states[:restored_size] = state_dict['states'][:restored_size]
        self.policies[:restored_size] = state_dict['policies'][:restored_size]
        self.values[:restored_size] = state_dict['values'][:restored_size]
        self.generations[:restored_size] = state_dict['generations'][:restored_size]

    def __len__(self) -> int:
        return self.size
