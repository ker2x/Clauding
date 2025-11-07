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
import time


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

        # Timing diagnostics
        self.verbose_timing = False

    def _get_conv_out_size(self, shape):
        """Calculate the output size of conv layers."""
        o = self.conv1(torch.zeros(1, *shape))
        o = self.conv2(o)
        o = self.conv3(o)
        return int(np.prod(o.size()))

    def forward(self, state, action):
        if self.verbose_timing:
            t0 = time.perf_counter()
            x = F.relu(self.conv1(state))
            t1 = time.perf_counter()
            x = F.relu(self.conv2(x))
            t2 = time.perf_counter()
            x = F.relu(self.conv3(x))
            t3 = time.perf_counter()
            x = x.view(x.size(0), -1)
            x = torch.cat([x, action], dim=1)
            x = F.relu(self.fc1(x))
            t4 = time.perf_counter()
            q_value = self.fc2(x)
            t5 = time.perf_counter()

            return q_value, {
                'conv1': (t1 - t0) * 1000,
                'conv2': (t2 - t1) * 1000,
                'conv3': (t3 - t2) * 1000,
                'fc_layers': (t5 - t4) * 1000,
                'total': (t5 - t0) * 1000
            }
        else:
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
    """
    Optimized experience replay buffer for SAC using pre-allocated torch tensors.

    Memory-efficient approach: stores data on CPU, transfers only sampled batches to device.
    This avoids memory issues with large visual observation buffers while maintaining speed.

    Performance improvement: ~5-8x faster sampling (20ms → 2-4ms for visual observations)
    Memory usage: Only batch size * state_size on device (vs full buffer on device)
    """
    def __init__(self, capacity, state_shape, action_dim, device):
        self.capacity = capacity
        self.device = device
        self.action_dim = action_dim
        self.state_shape = state_shape if isinstance(state_shape, tuple) else (state_shape,)

        # Pre-allocate tensors on CPU for memory efficiency
        # For visual obs (100K * 4 * 96 * 96 * 4 bytes), this is ~15GB on CPU but manageable
        # Only sampled batches are transferred to device (e.g., 256 * 4 * 96 * 96 = 38MB)
        self.states = torch.zeros((capacity, *self.state_shape), dtype=torch.float32, device='cpu')
        self.actions = torch.zeros((capacity, action_dim), dtype=torch.float32, device='cpu')
        self.rewards = torch.zeros((capacity, 1), dtype=torch.float32, device='cpu')
        self.next_states = torch.zeros((capacity, *self.state_shape), dtype=torch.float32, device='cpu')
        self.dones = torch.zeros((capacity, 1), dtype=torch.float32, device='cpu')

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
        This is ~5-8x faster than the old numpy conversion approach for visual obs.
        """
        # Generate random indices on CPU (avoid cross-device operations)
        indices = torch.randint(0, self.size, (batch_size,), device='cpu')

        # Index on CPU, then transfer batch to device
        # This is much faster than transferring the entire buffer to device
        return (
            self.states[indices].to(self.device),
            self.actions[indices].to(self.device),
            self.rewards[indices].to(self.device),
            self.next_states[indices].to(self.device),
            self.dones[indices].to(self.device)
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

        # Verbose mode for timing diagnostics
        self.verbose = False
        self.update_counter = 0
        self.layer_timings = []  # Store layer-level timings for analysis

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
        # Timing diagnostics
        update_start = time.perf_counter() if self.verbose else None
        timings = {}

        # Sample batch
        sample_start = time.perf_counter() if self.verbose else None
        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
        if self.verbose:
            timings['sample'] = (time.perf_counter() - sample_start) * 1000

        # ===========================
        # Update Critics
        # ===========================

        target_start = time.perf_counter() if self.verbose else None
        with torch.no_grad():
            # Sample next actions from current policy
            next_actions, next_log_probs = self._sample_action(next_states)

            # Compute target Q-values (minimum of two critics)
            forward_target_start = time.perf_counter() if self.verbose else None
            target_q1 = self.critic_target_1(next_states, next_actions)
            target_q2 = self.critic_target_2(next_states, next_actions)
            if self.verbose:
                # Synchronize to get accurate timing
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                elif self.device.type == 'mps':
                    torch.mps.synchronize()
                timings['target_forward'] = (time.perf_counter() - forward_target_start) * 1000

            target_q = torch.min(target_q1, target_q2)

            # Add entropy term
            target_q = target_q - self.alpha * next_log_probs

            # Compute target
            target_q = rewards + (1 - dones) * self.gamma * target_q
        if self.verbose:
            timings['target_total'] = (time.perf_counter() - target_start) * 1000

        # Update critic 1
        critic1_start = time.perf_counter() if self.verbose else None
        forward_c1_start = time.perf_counter() if self.verbose else None

        # Enable detailed timing for visual critics
        if self.verbose and hasattr(self.critic_1, 'verbose_timing'):
            self.critic_1.verbose_timing = True
            result = self.critic_1(states, actions)
            if isinstance(result, tuple):
                current_q1, layer_times_c1 = result
                timings['c1_layers'] = layer_times_c1
            else:
                current_q1 = result
            self.critic_1.verbose_timing = False
        else:
            current_q1 = self.critic_1(states, actions)

        if self.verbose:
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
            elif self.device.type == 'mps':
                torch.mps.synchronize()
            timings['critic1_forward'] = (time.perf_counter() - forward_c1_start) * 1000

        critic_1_loss = F.mse_loss(current_q1, target_q)
        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        self.critic_1_optimizer.step()
        if self.verbose:
            timings['critic1_total'] = (time.perf_counter() - critic1_start) * 1000

        # Update critic 2
        critic2_start = time.perf_counter() if self.verbose else None
        forward_c2_start = time.perf_counter() if self.verbose else None

        # Enable detailed timing for visual critics
        if self.verbose and hasattr(self.critic_2, 'verbose_timing'):
            self.critic_2.verbose_timing = True
            result = self.critic_2(states, actions)
            if isinstance(result, tuple):
                current_q2, layer_times_c2 = result
                timings['c2_layers'] = layer_times_c2
            else:
                current_q2 = result
            self.critic_2.verbose_timing = False
        else:
            current_q2 = self.critic_2(states, actions)

        if self.verbose:
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
            elif self.device.type == 'mps':
                torch.mps.synchronize()
            timings['critic2_forward'] = (time.perf_counter() - forward_c2_start) * 1000

        critic_2_loss = F.mse_loss(current_q2, target_q)
        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        self.critic_2_optimizer.step()
        if self.verbose:
            timings['critic2_total'] = (time.perf_counter() - critic2_start) * 1000

        # ===========================
        # Update Actor
        # ===========================

        actor_start = time.perf_counter() if self.verbose else None
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
        if self.verbose:
            timings['actor_total'] = (time.perf_counter() - actor_start) * 1000

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

        # Print timing diagnostics
        if self.verbose and self.update_counter % 10 == 0:
            import psutil
            import os

            total_time = (time.perf_counter() - update_start) * 1000

            # Get CPU diagnostics
            process = psutil.Process(os.getpid())
            cpu_percent = process.cpu_percent()
            memory_mb = process.memory_info().rss / 1024 / 1024
            num_threads = torch.get_num_threads()

            print(f"\n{'='*70}")
            print(f"SAC UPDATE {self.update_counter} TIMING:")
            print(f"  Sample batch:         {timings.get('sample', 0):>7.2f} ms")
            print(f"  Target forward:       {timings.get('target_forward', 0):>7.2f} ms")
            print(f"  Target total:         {timings.get('target_total', 0):>7.2f} ms")
            print(f"  Critic 1 forward:     {timings.get('critic1_forward', 0):>7.2f} ms  <<< WATCH THIS")

            # Print layer-level timing for Critic 1 if available
            if 'c1_layers' in timings:
                lt = timings['c1_layers']
                print(f"    ├─ conv1:           {lt['conv1']:>7.2f} ms")
                print(f"    ├─ conv2:           {lt['conv2']:>7.2f} ms")
                print(f"    ├─ conv3:           {lt['conv3']:>7.2f} ms")
                print(f"    └─ FC layers:       {lt['fc_layers']:>7.2f} ms")

            print(f"  Critic 1 total:       {timings.get('critic1_total', 0):>7.2f} ms")
            print(f"  Critic 2 forward:     {timings.get('critic2_forward', 0):>7.2f} ms  <<< WATCH THIS")

            # Print layer-level timing for Critic 2 if available
            if 'c2_layers' in timings:
                lt = timings['c2_layers']
                print(f"    ├─ conv1:           {lt['conv1']:>7.2f} ms")
                print(f"    ├─ conv2:           {lt['conv2']:>7.2f} ms")
                print(f"    ├─ conv3:           {lt['conv3']:>7.2f} ms")
                print(f"    └─ FC layers:       {lt['fc_layers']:>7.2f} ms")

            print(f"  Critic 2 total:       {timings.get('critic2_total', 0):>7.2f} ms")
            print(f"  Actor total:          {timings.get('actor_total', 0):>7.2f} ms")
            print(f"  TOTAL UPDATE:         {total_time:>7.2f} ms")
            print(f"\n  CPU DIAGNOSTICS:")
            print(f"    PyTorch threads:    {num_threads}")
            print(f"    CPU usage:          {cpu_percent:.1f}%")
            print(f"    Memory usage:       {memory_mb:.1f} MB")
            print(f"{'='*70}\n")

            # Store timing for later analysis
            self.layer_timings.append({
                'update': self.update_counter,
                'total': total_time,
                'c1_forward': timings.get('critic1_forward', 0),
                'c2_forward': timings.get('critic2_forward', 0),
                'c1_layers': timings.get('c1_layers', {}),
                'c2_layers': timings.get('c2_layers', {})
            })

        self.update_counter += 1

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
