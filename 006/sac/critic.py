"""
Critic (Q-function) network for SAC algorithm.

The critic estimates the Q-value (expected return) for a given state-action pair.
SAC uses twin critics to reduce overestimation bias.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorCritic(nn.Module):
    """
    Critic (Q-function) network for vector state mode (71D input as of 006).
    Takes state and action as input, outputs Q-value.

    Optimized architecture (~2.5x faster forward pass):
    - 3 hidden layers (reduced from 4) - critic needs more capacity than actor
    - 128 hidden units (reduced from 256)
    - LayerNorm for training stability
    - 42,881 parameters (72% reduction from 151,297)

    Uses LeakyReLU activation (negative_slope=0.01) to prevent dead neurons
    and improve gradient flow compared to standard ReLU.

    Hidden dimension of 128 is sufficient for 73D input (state+action) without overfitting.
    Deeper than actor (3 vs 2 layers) as Q-value estimation is more complex than policy.
    """
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128) -> None:
        super(VectorCritic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim * 2)
        self.ln1 = nn.LayerNorm(hidden_dim * 2)  # Normalize after first layer for stability
        self.fc2 = nn.Linear(hidden_dim * 2, hidden_dim)
#        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, 1)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, action], dim=1)
        x = F.leaky_relu(self.ln1(self.fc1(x)), negative_slope=0.01)
        x = F.leaky_relu(self.fc2(x), negative_slope=0.01)
#        x = F.leaky_relu(self.fc3(x), negative_slope=0.01)
        q_value = self.fc4(x)
        return q_value
