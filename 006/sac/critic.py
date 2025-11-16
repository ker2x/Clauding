"""
Critic (Q-function) network for SAC algorithm.

The critic estimates the Q-value (expected return) for a given state-action pair.
SAC uses twin critics to reduce overestimation bias.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorCritic(nn.Module):
    """
    Critic (Q-function) network for vector state mode (71D input as of 006).
    Takes state and action as input, outputs Q-value.

    Uses LeakyReLU activation (negative_slope=0.01) to prevent dead neurons
    and improve gradient flow compared to standard ReLU.

    Hidden dimension set to 256 for efficiency - sufficient capacity for 71D
    state space without overfitting.
    """
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(VectorCritic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)  # Normalize after first layer for stability
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = F.leaky_relu(self.ln1(self.fc1(x)), negative_slope=0.01)
        x = F.leaky_relu(self.fc2(x), negative_slope=0.01)
        x = F.leaky_relu(self.fc3(x), negative_slope=0.01)
        q_value = self.fc4(x)
        return q_value
