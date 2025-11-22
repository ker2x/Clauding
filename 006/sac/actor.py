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
    """
    Actor (policy) network for vector state mode with frame stacking.
    Outputs mean and log_std for a Gaussian policy.

    Architecture for frame-stacked observations (165D+):
    - 3 hidden layers for temporal feature extraction
    - 384 default hidden units (increased for frame stacking)
    - LayerNorm for training stability
    - Deeper network to extract temporal patterns from stacked frames

    Uses LeakyReLU activation (negative_slope=0.01) to prevent dead neurons
    and improve gradient flow compared to standard ReLU.
    """
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 384) -> None:
        super(VectorActor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)  # Normalize after first layer for stability
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)  # Additional normalization for deeper network
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)

        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = F.leaky_relu(self.ln1(self.fc1(state)), negative_slope=0.01)
        x = F.leaky_relu(self.ln2(self.fc2(x)), negative_slope=0.01)
        x = F.leaky_relu(self.fc3(x), negative_slope=0.01)

        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, min=-20, max=2)  # Prevent numerical instability

        return mean, log_std
