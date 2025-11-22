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
    Critic (Q-function) network for vector state mode with frame stacking.
    Takes state and action as input, outputs Q-value.

    Architecture for large observation spaces (165D+ with frame stacking):
    - 4 hidden layers for increased capacity
    - 384 default hidden units (increased for frame-stacked observations)
    - LayerNorm for training stability
    - Deeper network to handle temporal information from frame stacking

    Uses LeakyReLU activation (negative_slope=0.01) to prevent dead neurons
    and improve gradient flow compared to standard ReLU.

    With frame stacking and prev_action, the critic needs more capacity to
    properly estimate Q-values from the richer temporal information.
    """
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 384) -> None:
        super(VectorCritic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)  # Normalize after first layer for stability
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)  # Additional normalization for deeper network
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim // 2)  # Gradual dimensionality reduction
        self.fc5 = nn.Linear(hidden_dim // 2, 1)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, action], dim=1)
        x = F.leaky_relu(self.ln1(self.fc1(x)), negative_slope=0.01)
        x = F.leaky_relu(self.ln2(self.fc2(x)), negative_slope=0.01)
        x = F.leaky_relu(self.fc3(x), negative_slope=0.01)
        x = F.leaky_relu(self.fc4(x), negative_slope=0.01)
        q_value = self.fc5(x)
        return q_value
