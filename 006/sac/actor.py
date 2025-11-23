"""
Actor (policy) network for SAC algorithm.

The actor network outputs a stochastic policy using a Gaussian distribution.
It uses the reparameterization trick for training with gradient descent.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorActor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256) -> None:
        super(VectorActor, self).__init__()
        # OPTIMIZATION 1: Reduced width 384 -> 256 (matches your Stable Rank of ~19 better)

        # Layer 1
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)

        # Layer 2
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)

        # OPTIMIZATION 2: Removed fc3.
        # Deep RL Actors usually struggle with depth > 2 due to gradient delay.
        # This also removes the "High Spectral Norm" layer that wasn't normalized.

        # Output Heads
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Block 1
        x = self.fc1(state)
        x = self.ln1(x)
        x = F.leaky_relu(x, negative_slope=0.01)

        # Block 2
        x = self.fc2(x)
        x = self.ln2(x)
        x = F.leaky_relu(x, negative_slope=0.01)

        # Heads
        mean = self.mean(x)
        log_std = self.log_std(x)

        # Standard SAC log_std clamping
        log_std = torch.clamp(log_std, min=-20, max=2)

        return mean, log_std
