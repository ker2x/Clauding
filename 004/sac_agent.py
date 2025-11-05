"""
Soft Actor-Critic (SAC) agent for CarRacing-v3 with continuous actions.

SAC is a state-of-the-art off-policy algorithm that:
1. Maximizes both reward AND entropy (encourages exploration)
2. Uses two Q-networks to reduce overestimation bias
3. Automatically tunes the entropy coefficient (temperature parameter)

Reference: "Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor"
           Haarnoja et al., 2018 (https://arxiv.org/abs/1801.01290)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
from collections import deque
import random


# ===========================
# Network Architectures
# ===========================

class VectorActor(nn.Module):
    """
    Actor (policy) network for vector state mode (36D input).
    Outputs mean and log_std for a Gaussian policy.
    """
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(VectorActor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)

        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, min=-20, max=2)  # Prevent numerical instability

        return mean, log_std


class VectorCritic(nn.Module):
    """
    Critic (Q-function) network for vector state mode (36D input).
    Takes state and action as input, outputs Q-value.
    """
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(VectorCritic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        q_value = self.fc4(x)
        return q_value


class VisualActor(nn.Module):
    """
    Actor network for visual state mode (4x96x96 stacked frames).
    Uses CNN to extract features, then outputs action distribution.
    """
    def __init__(self, state_shape, action_dim):
        super(VisualActor, self).__init__()
        c, h, w = state_shape

        # CNN feature extractor
        self.conv1 = nn.Conv2d(c, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # Calculate conv output size
        conv_out_size = self._get_conv_out_size(state_shape)

        self.fc1 = nn.Linear(conv_out_size, 512)
        self.mean = nn.Linear(512, action_dim)
        self.log_std = nn.Linear(512, action_dim)

    def _get_conv_out_size(self, shape):
        """Calculate the output size of conv layers."""
        o = self.conv1(torch.zeros(1, *shape))
        o = self.conv2(o)
        o = self.conv3(o)
        return int(np.prod(o.size()))

    def forward(self, state):
        x = F.relu(self.conv1(state))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))

        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, min=-20, max=2)

        return mean, log_std


class VisualCritic(nn.Module):
    """
    Critic network for visual state mode (4x96x96 stacked frames).
    Uses CNN to extract features, then combines with action.
    """
    def __init__(self, state_shape, action_dim):
        super(VisualCritic, self).__init__()
        c, h, w = state_shape

        # CNN feature extractor
        self.conv1 = nn.Conv2d(c, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # Calculate conv output size
        conv_out_size = self._get_conv_out_size(state_shape)

        self.fc1 = nn.Linear(conv_out_size + action_dim, 512)
        self.fc2 = nn.Linear(512, 1)

    def _get_conv_out_size(self, shape):
        """Calculate the output size of conv layers."""
        o = self.conv1(torch.zeros(1, *shape))
        o = self.conv2(o)
        o = self.conv3(o)
        return int(np.prod(o.size()))

    def forward(self, state, action):
        x = F.relu(self.conv1(state))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = torch.cat([x, action], dim=1)
        x = F.relu(self.fc1(x))
        q_value = self.fc2(x)
        return q_value


# ===========================
# Replay Buffer
# ===========================

class ReplayBuffer:
    """Experience replay buffer for SAC."""
    def __init__(self, capacity, state_shape, action_dim, device):
        self.capacity = capacity
        self.device = device
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """Add experience to buffer."""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """Sample a batch of experiences."""
        batch = random.sample(self.buffer, batch_size)

        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.FloatTensor(np.array(actions)).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards)).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(np.array(dones)).unsqueeze(1).to(self.device)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


# ===========================
# SAC Agent
# ===========================

class SACAgent:
    """
    Soft Actor-Critic agent with automatic entropy tuning.

    Key features:
    - Twin Q-networks to reduce overestimation
    - Entropy regularization for better exploration
    - Automatic temperature (alpha) tuning
    - Supports both vector and visual state modes
    """

    def __init__(
        self,
        state_shape,
        action_dim,
        state_mode='vector',
        lr_actor=3e-4,
        lr_critic=3e-4,
        lr_alpha=3e-4,
        gamma=0.99,
        tau=0.005,
        alpha=0.2,
        auto_entropy_tuning=True,
        device=None
    ):
        """
        Args:
            state_shape: Shape of state (int for vector, tuple for visual)
            action_dim: Dimension of action space
            state_mode: 'vector' or 'visual'
            lr_actor: Learning rate for actor
            lr_critic: Learning rate for critics
            lr_alpha: Learning rate for alpha (entropy coefficient)
            gamma: Discount factor
            tau: Soft update coefficient for target networks
            alpha: Initial entropy coefficient (if not auto-tuning)
            auto_entropy_tuning: Whether to automatically tune alpha
            device: torch device (cuda/mps/cpu)
        """
        self.action_dim = action_dim
        self.state_mode = state_mode
        self.gamma = gamma
        self.tau = tau
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Create networks based on state mode
        if state_mode == 'vector':
            state_dim = state_shape if isinstance(state_shape, int) else state_shape[0]
            self.actor = VectorActor(state_dim, action_dim).to(self.device)
            self.critic_1 = VectorCritic(state_dim, action_dim).to(self.device)
            self.critic_2 = VectorCritic(state_dim, action_dim).to(self.device)
            self.critic_target_1 = VectorCritic(state_dim, action_dim).to(self.device)
            self.critic_target_2 = VectorCritic(state_dim, action_dim).to(self.device)
        else:  # visual mode
            self.actor = VisualActor(state_shape, action_dim).to(self.device)
            self.critic_1 = VisualCritic(state_shape, action_dim).to(self.device)
            self.critic_2 = VisualCritic(state_shape, action_dim).to(self.device)
            self.critic_target_1 = VisualCritic(state_shape, action_dim).to(self.device)
            self.critic_target_2 = VisualCritic(state_shape, action_dim).to(self.device)

        # Initialize target networks
        self.critic_target_1.load_state_dict(self.critic_1.state_dict())
        self.critic_target_2.load_state_dict(self.critic_2.state_dict())

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_1_optimizer = optim.Adam(self.critic_1.parameters(), lr=lr_critic)
        self.critic_2_optimizer = optim.Adam(self.critic_2.parameters(), lr=lr_critic)

        # Entropy tuning
        self.auto_entropy_tuning = auto_entropy_tuning
        if auto_entropy_tuning:
            # Target entropy = -dim(A) (heuristic)
            self.target_entropy = -action_dim
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr_alpha)
            self.alpha = self.log_alpha.exp()
        else:
            self.alpha = torch.tensor(alpha, device=self.device)

    def _apply_action_bounds(self, z):
        """
        Apply proper bounds to sampled actions (native action space):
        - Steering (dim 0): tanh → [-1, 1]
        - Acceleration (dim 1): tanh → [-1 (brake), +1 (gas)]

        Args:
            z: Unbounded sampled actions from Gaussian

        Returns:
            action: Actions in native bounds
        """
        # Both actions use tanh squashing to [-1, 1]
        action = torch.tanh(z)
        return action

    def select_action(self, state, evaluate=False):
        """
        Select action from policy.

        Args:
            state: Current state
            evaluate: If True, use mean action (deterministic). If False, sample from distribution.

        Returns:
            action: Action to take in native bounds
        """
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            mean, log_std = self.actor(state)

            if evaluate:
                # Deterministic action (mean) - apply bounds
                action = self._apply_action_bounds(mean)
            else:
                # Stochastic action (sample from distribution)
                std = log_std.exp()
                normal = Normal(mean, std)
                z = normal.sample()
                action = self._apply_action_bounds(z)

        return action.cpu().numpy()[0]

    def _sample_action(self, state):
        """
        Sample action and compute log probability.
        Used during training for policy updates.

        Applies native action bounds:
        - Steering (dim 0): tanh → [-1, 1]
        - Acceleration (dim 1): tanh → [-1 (brake), +1 (gas)]
        """
        mean, log_std = self.actor(state)
        std = log_std.exp()

        normal = Normal(mean, std)
        z = normal.rsample()  # Reparameterization trick
        action = self._apply_action_bounds(z)

        # Compute log probability with tanh correction
        # For y = tanh(z): log|dy/dz| = log(1 - tanh^2(z))
        log_prob = normal.log_prob(z)

        # Apply tanh Jacobian correction for all dimensions
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)

        # Sum over action dimensions
        log_prob = log_prob.sum(1, keepdim=True)

        return action, log_prob

    def update(self, replay_buffer, batch_size):
        """
        Update actor and critics using a batch from replay buffer.

        Returns:
            Dictionary with training metrics
        """
        # Sample batch
        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

        # ===========================
        # Update Critics
        # ===========================

        with torch.no_grad():
            # Sample next actions from current policy
            next_actions, next_log_probs = self._sample_action(next_states)

            # Compute target Q-values (minimum of two critics)
            target_q1 = self.critic_target_1(next_states, next_actions)
            target_q2 = self.critic_target_2(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2)

            # Add entropy term
            target_q = target_q - self.alpha * next_log_probs

            # Compute target
            target_q = rewards + (1 - dones) * self.gamma * target_q

        # Update critic 1
        current_q1 = self.critic_1(states, actions)
        critic_1_loss = F.mse_loss(current_q1, target_q)
        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        self.critic_1_optimizer.step()

        # Update critic 2
        current_q2 = self.critic_2(states, actions)
        critic_2_loss = F.mse_loss(current_q2, target_q)
        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        self.critic_2_optimizer.step()

        # ===========================
        # Update Actor
        # ===========================

        # Sample actions from current policy
        new_actions, log_probs = self._sample_action(states)

        # Compute Q-values for new actions
        q1 = self.critic_1(states, new_actions)
        q2 = self.critic_2(states, new_actions)
        q = torch.min(q1, q2)

        # Actor loss: maximize Q-value and entropy
        # J = E[α log π(a|s) - Q(s,a)]
        actor_loss = (self.alpha * log_probs - q).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ===========================
        # Update Alpha (Entropy Coefficient)
        # ===========================

        if self.auto_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()

            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

            self.alpha = self.log_alpha.exp()
        else:
            alpha_loss = torch.tensor(0.0)

        # ===========================
        # Soft Update Target Networks
        # ===========================

        self._soft_update(self.critic_1, self.critic_target_1)
        self._soft_update(self.critic_2, self.critic_target_2)

        # Return metrics
        return {
            'critic_1_loss': critic_1_loss.item(),
            'critic_2_loss': critic_2_loss.item(),
            'actor_loss': actor_loss.item(),
            'alpha_loss': alpha_loss.item() if self.auto_entropy_tuning else 0.0,
            'alpha': self.alpha.item() if self.auto_entropy_tuning else self.alpha,
            'mean_q1': current_q1.mean().item(),
            'mean_q2': current_q2.mean().item(),
            'mean_log_prob': log_probs.mean().item()
        }

    def _soft_update(self, source, target):
        """Soft update target network parameters."""
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(self.tau * source_param.data + (1 - self.tau) * target_param.data)

    def save(self, filepath):
        """Save agent checkpoint."""
        torch.save({
            'actor': self.actor.state_dict(),
            'critic_1': self.critic_1.state_dict(),
            'critic_2': self.critic_2.state_dict(),
            'critic_target_1': self.critic_target_1.state_dict(),
            'critic_target_2': self.critic_target_2.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_1_optimizer': self.critic_1_optimizer.state_dict(),
            'critic_2_optimizer': self.critic_2_optimizer.state_dict(),
            'log_alpha': self.log_alpha if self.auto_entropy_tuning else None,
            'alpha_optimizer': self.alpha_optimizer.state_dict() if self.auto_entropy_tuning else None,
            'state_mode': self.state_mode,
            'action_dim': self.action_dim
        }, filepath)

    def load(self, filepath):
        """Load agent checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic_1.load_state_dict(checkpoint['critic_1'])
        self.critic_2.load_state_dict(checkpoint['critic_2'])
        self.critic_target_1.load_state_dict(checkpoint['critic_target_1'])
        self.critic_target_2.load_state_dict(checkpoint['critic_target_2'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_1_optimizer.load_state_dict(checkpoint['critic_1_optimizer'])
        self.critic_2_optimizer.load_state_dict(checkpoint['critic_2_optimizer'])

        if self.auto_entropy_tuning and checkpoint['log_alpha'] is not None:
            self.log_alpha.data = checkpoint['log_alpha'].data
            self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer'])
            self.alpha = self.log_alpha.exp()
