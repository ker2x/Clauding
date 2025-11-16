"""
Actor (policy) network for SAC algorithm.

The actor network outputs a stochastic policy using a Gaussian distribution.
It uses the reparameterization trick for training with gradient descent.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorActor(nn.Module):
    """
    Actor (policy) network for vector state mode (71D input as of 006).
    Outputs mean and log_std for a Gaussian policy.

    Optimized architecture (~2.1x faster forward pass):
    - 2 hidden layers (reduced from 3)
    - 128 hidden units (reduced from 256)
    - Keeps LayerNorm for training stability
    - 26,500 parameters (83% reduction from 151,556)

    Uses LeakyReLU activation (negative_slope=0.01) to prevent dead neurons
    and improve gradient flow compared to standard ReLU.
    """
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(VectorActor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)  # Normalize after first layer for stability
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        x = F.leaky_relu(self.ln1(self.fc1(state)), negative_slope=0.01)
        x = F.leaky_relu(self.fc2(x), negative_slope=0.01)

        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, min=-20, max=2)  # Prevent numerical instability

        return mean, log_std


class VectorActorNoLN(nn.Module):
    """
    Alternative actor network WITHOUT LayerNorm.

    Same architecture as VectorActor but without LayerNorm:
    - 2 hidden layers
    - 128 hidden units
    - NO LayerNorm (for performance comparison)
    - 26,244 parameters (slightly fewer than LayerNorm version)

    Use --no-layernorm flag to enable this architecture.
    """
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(VectorActorNoLN, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        x = F.leaky_relu(self.fc1(state), negative_slope=0.01)
        x = F.leaky_relu(self.fc2(x), negative_slope=0.01)
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, min=-20, max=2)
        return mean, log_std
