"""
Soft Actor-Critic (SAC) Agent for CarRacing-v3

Implements SAC, a state-of-the-art off-policy RL algorithm for continuous control:

Key Features:
1. Maximum entropy objective (balances reward and exploration)
2. Twin Q-networks (reduces overestimation bias)
3. Automatic entropy tuning (learns optimal exploration coefficient)
4. Continuous action space support (steering, acceleration)
5. Vector mode only (67D state vector)

Architecture:
- Actor: Gaussian policy with reparameterization trick
- Critics: Two Q-networks with soft target updates
- Alpha: Learned entropy coefficient

References:
- Haarnoja et al., 2018: "Soft Actor-Critic: Off-Policy Maximum Entropy Deep RL"
- Haarnoja et al., 2019: "Soft Actor-Critic Algorithms and Applications"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import numpy as np


# ===========================
# Network Architectures
# ===========================

class VectorActor(nn.Module):
    """
    Actor (policy) network for vector state mode (67D input as of 006).
    Outputs mean and log_std for a Gaussian policy.

    Uses LeakyReLU activation (negative_slope=0.01) to prevent dead neurons
    and improve gradient flow compared to standard ReLU.
    """
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(VectorActor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)  # Normalize after first layer for stability
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)

        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        x = F.leaky_relu(self.ln1(self.fc1(state)), negative_slope=0.01)
        x = F.leaky_relu(self.fc2(x), negative_slope=0.01)
        x = F.leaky_relu(self.fc3(x), negative_slope=0.01)

        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, min=-20, max=2)  # Prevent numerical instability

        return mean, log_std


class VectorCritic(nn.Module):
    """
    Critic (Q-function) network for vector state mode (67D input as of 006).
    Takes state and action as input, outputs Q-value.

    Uses LeakyReLU activation (negative_slope=0.01) to prevent dead neurons
    and improve gradient flow compared to standard ReLU.

    Hidden dimension increased to 512 (from 256) to handle the expanded
    state space (36D → 67D) with better capacity.
    """
    def __init__(self, state_dim, action_dim, hidden_dim=512):
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


# ===========================
# Replay Buffer
# ===========================

class ReplayBuffer:
    """
    Optimized experience replay buffer for SAC using pre-allocated torch tensors.

    Memory-efficient approach: stores data on CPU, transfers only sampled batches to device.
    This avoids memory issues while maintaining speed.

    Optimizations:
    - Pinned memory: Faster CPU->GPU transfers (when CUDA available)
    - Non-blocking transfers: Asynchronous GPU copies for better throughput
    - Pre-allocated tensors: Avoid repeated allocations

    Performance improvement: ~5-8x faster sampling (20ms → 2-4ms)
    Memory usage: Only batch size * state_size on device (vs full buffer on device)
    """
    def __init__(self, capacity, state_shape, action_dim, device):
        self.capacity = capacity
        self.device = device
        self.action_dim = action_dim
        self.state_shape = state_shape if isinstance(state_shape, tuple) else (state_shape,)

        # Use pinned memory for faster GPU transfers (CUDA only)
        self.use_pinned_memory = device.type == 'cuda'

        # Pre-allocate tensors on CPU for memory efficiency
        # Pinned memory enables faster async transfers to GPU
        self.states = torch.zeros((capacity, *self.state_shape), dtype=torch.float32, device='cpu')
        self.actions = torch.zeros((capacity, action_dim), dtype=torch.float32, device='cpu')
        self.rewards = torch.zeros((capacity, 1), dtype=torch.float32, device='cpu')
        self.next_states = torch.zeros((capacity, *self.state_shape), dtype=torch.float32, device='cpu')
        self.dones = torch.zeros((capacity, 1), dtype=torch.float32, device='cpu')

        if self.use_pinned_memory:
            self.states = self.states.pin_memory()
            self.actions = self.actions.pin_memory()
            self.rewards = self.rewards.pin_memory()
            self.next_states = self.next_states.pin_memory()
            self.dones = self.dones.pin_memory()

        # Circular buffer management
        self.ptr = 0  # Current write position
        self.size = 0  # Current buffer size (until we fill capacity)

    def push(self, state, action, reward, next_state, done):
        """
        Add experience to buffer.

        Accepts both numpy arrays and torch tensors as input for compatibility.
        Data is converted to torch tensors and stored on CPU.
        """
        # Convert inputs to torch tensors if they aren't already
        if not isinstance(state, torch.Tensor):
            state = torch.from_numpy(np.array(state)).float()
        if not isinstance(action, torch.Tensor):
            action = torch.from_numpy(np.array(action)).float()
        if not isinstance(next_state, torch.Tensor):
            next_state = torch.from_numpy(np.array(next_state)).float()

        # Store in pre-allocated CPU tensors (no device transfer on push)
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = float(done)

        # Update circular buffer pointer
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        """
        Sample a batch of experiences.

        Samples on CPU (fast indexing), then transfers batch to target device.
        Uses non-blocking transfers for better throughput when pinned memory is enabled.
        This is ~5-8x faster than the old numpy conversion approach.
        """
        # Generate random indices on CPU (avoid cross-device operations)
        indices = torch.randint(0, self.size, (batch_size,), device='cpu')

        # Index on CPU, then transfer batch to device
        # non_blocking=True enables async transfer when using pinned memory
        return (
            self.states[indices].to(self.device, non_blocking=True),
            self.actions[indices].to(self.device, non_blocking=True),
            self.rewards[indices].to(self.device, non_blocking=True),
            self.next_states[indices].to(self.device, non_blocking=True),
            self.dones[indices].to(self.device, non_blocking=True)
        )

    def __len__(self):
        return self.size


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
    - Vector state mode only (67D state vector)
    """

    def __init__(
        self,
        state_dim,
        action_dim,
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
            state_dim: Dimension of state vector (67 for current version)
            action_dim: Dimension of action space
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
        self.state_dim = state_dim
        self.gamma = gamma
        self.tau = tau
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Create networks
        self.actor = VectorActor(state_dim, action_dim).to(self.device)
        self.critic_1 = VectorCritic(state_dim, action_dim).to(self.device)
        self.critic_2 = VectorCritic(state_dim, action_dim).to(self.device)
        self.critic_target_1 = VectorCritic(state_dim, action_dim).to(self.device)
        self.critic_target_2 = VectorCritic(state_dim, action_dim).to(self.device)

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
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device, dtype=torch.float32)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr_alpha)
            self.alpha = self.log_alpha.exp()
        else:
            self.alpha = torch.tensor(alpha, device=self.device, dtype=torch.float32)

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
            alpha_loss = torch.tensor(0.0, device=self.device, dtype=torch.float32)

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
            'action_dim': self.action_dim,
            'state_dim': self.state_dim
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

    def get_state_dict(self):
        """Get agent state dictionary (for in-memory cloning)."""
        import copy
        return {
            'actor': copy.deepcopy(self.actor.state_dict()),
            'critic_1': copy.deepcopy(self.critic_1.state_dict()),
            'critic_2': copy.deepcopy(self.critic_2.state_dict()),
            'critic_target_1': copy.deepcopy(self.critic_target_1.state_dict()),
            'critic_target_2': copy.deepcopy(self.critic_target_2.state_dict()),
            'actor_optimizer': copy.deepcopy(self.actor_optimizer.state_dict()),
            'critic_1_optimizer': copy.deepcopy(self.critic_1_optimizer.state_dict()),
            'critic_2_optimizer': copy.deepcopy(self.critic_2_optimizer.state_dict()),
            'log_alpha': copy.deepcopy(self.log_alpha) if self.auto_entropy_tuning else None,
            'alpha_optimizer': copy.deepcopy(self.alpha_optimizer.state_dict()) if self.auto_entropy_tuning else None,
        }

    def load_state_dict(self, state_dict):
        """Load agent state from dictionary (for in-memory cloning)."""
        self.actor.load_state_dict(state_dict['actor'])
        self.critic_1.load_state_dict(state_dict['critic_1'])
        self.critic_2.load_state_dict(state_dict['critic_2'])
        self.critic_target_1.load_state_dict(state_dict['critic_target_1'])
        self.critic_target_2.load_state_dict(state_dict['critic_target_2'])
        self.actor_optimizer.load_state_dict(state_dict['actor_optimizer'])
        self.critic_1_optimizer.load_state_dict(state_dict['critic_1_optimizer'])
        self.critic_2_optimizer.load_state_dict(state_dict['critic_2_optimizer'])

        if self.auto_entropy_tuning and state_dict['log_alpha'] is not None:
            self.log_alpha.data = state_dict['log_alpha'].data
            self.alpha_optimizer.load_state_dict(state_dict['alpha_optimizer'])
            self.alpha = self.log_alpha.exp()
