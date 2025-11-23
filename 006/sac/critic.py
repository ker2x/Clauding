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
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256) -> None:
        super(VectorCritic, self).__init__()
        # REDUCTION: Width 384 -> 256 is plenty for a Rank of 10.

        # Layer 1
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)

        # Layer 2
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)

        # Output Layer (Standard SAC usually only has 2 hidden layers)
        self.fc_out = nn.Linear(hidden_dim, 1)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, action], dim=1)

        # Block 1
        x = self.fc1(x)
        x = self.ln1(x)
        x = F.leaky_relu(x, negative_slope=0.01)

        # Block 2
        x = self.fc2(x)
        x = self.ln2(x)
        x = F.leaky_relu(x, negative_slope=0.01)

        # Output
        q_value = self.fc_out(x)
        return q_value
